"""Fail-closed GP-only paired MAIN analysis; does not read canonical CSV or produce MEAN6."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from environment.online_crp import load_episode_manifest
from reproducibility.protocol import label_invariant_fingerprint, main_conditions

DEFAULT_DATA = ROOT / "data" / "caserta_table2_840.json"
DEFAULT_RULE = ROOT / "configs" / "gp_rule.json"
ARMS = ("NATIVE20", "UWSAA")
FAMILIES = ("subset_shuffle", "time_window_shuffle", "delayed_positions")
METRICS = ("relocations", "relocations_per_container")
DRAWS, SEED = 10_000, 202610195903


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def condition_registry() -> dict[str, dict[str, Any]]:
    result = {str(x["condition_id"]): dict(x) for x in main_conditions()}
    if len(result) != 41:
        raise AssertionError("registered MAIN condition count changed")
    return result


def active_gp_record(rule: Mapping[str, Any]) -> dict[str, Any]:
    if rule.get("schema") != "uwsaa_gp_rule_v1" or rule.get("status") != "trained":
        raise ValueError("active GP rule is not a trained uwsaa_gp_rule_v1")
    formula, formula_sha, identity = rule.get("formula"), rule.get("formula_sha256"), rule.get("model_identity")
    if not isinstance(formula, str) or not is_sha(formula_sha):
        raise ValueError("trained GP rule lacks formula text/SHA")
    if sha_bytes(formula.encode("utf-8")) != formula_sha:
        raise ValueError("active formula differs from formula_sha256")
    if not isinstance(identity, dict) or int(rule.get("best_fixed_h", -1)) != 20:
        raise ValueError("trained model identity or best_fixed_h=20 is missing")
    for field in ("source_manifest_sha256", "protocol_sha256", "train_bank_sha256",
                  "implementation_sha256", "seed_receipt_sha256"):
        value = rule.get(field, identity.get(field))
        if not is_sha(value) or identity.get(field) != value:
            raise ValueError(f"trained identity has inconsistent {field}")
    if identity.get("formula_sha256") != formula_sha or int(identity.get("best_fixed_h", -1)) != 20:
        raise ValueError("model identity differs from active formula/fixed horizon")
    return {
        "source_manifest_sha256": rule["source_manifest_sha256"],
        "protocol_sha256": rule["protocol_sha256"],
        "train_bank_sha256": rule["train_bank_sha256"],
        "implementation_sha256": rule["implementation_sha256"],
        "seed_receipt_sha256": identity["seed_receipt_sha256"],
        "status": "trained", "formula_sha256": formula_sha, "best_fixed_h": 20,
        "model_identity": identity,
    }


def episode_index(records: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("run manifest has no episode list")
    by_id, fingerprints = {}, set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("malformed episode record")
        episode_id, fingerprint = str(item.get("episode_id", "")), item.get("layout_fingerprint")
        geometry = str(item.get("parameter_group", ""))
        try:
            count = int(item["num_containers"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode record lacks num_containers") from exc
        if not episode_id or not is_sha(fingerprint) or not geometry or count <= 0:
            raise ValueError("invalid episode ID/fingerprint/geometry/size")
        if episode_id in by_id or fingerprint in fingerprints:
            raise ValueError("episode IDs and layout fingerprints must be unique")
        by_id[episode_id] = {"episode_id": episode_id, "layout_fingerprint": fingerprint.lower(),
                             "geometry": geometry, "num_containers": count}
        fingerprints.add(fingerprint.lower())
    geo = defaultdict(set)
    for item in by_id.values():
        geo[item["geometry"]].add(item["layout_fingerprint"])
    if len(by_id) != 840 or len(geo) != 21 or any(len(v) != 40 for v in geo.values()):
        raise ValueError("MAIN requires 840 unique layouts across 21 geometries × 40")
    return by_id


def actual_episode_index(data_path: Path) -> dict[str, dict[str, Any]]:
    episodes = load_episode_manifest(data_path)
    records = [{
        "episode_id": episode.episode_id,
        "layout_fingerprint": label_invariant_fingerprint(episode),
        "parameter_group": episode.parameter_group,
        "num_containers": episode.num_containers,
    } for episode in episodes]
    return episode_index(records)


def validate_endpoint_row(
    row: Mapping[str, Any],
    episode: Mapping[str, Any],
    condition: Mapping[str, Any],
    expected_gp: Mapping[str, Any],
) -> None:
    if row.get("complete") is not True:
        raise ValueError("endpoint is not complete")
    if any(str(key).startswith("failure_") for key in row):
        raise ValueError("complete endpoint contains failure metadata")
    if row.get("gp_model") != expected_gp:
        raise ValueError("endpoint GP identity differs from active model")
    try:
        containers = int(row["num_containers"])
        relocations = int(row["relocations"])
        ratio = float(row["relocations_per_container"])
        decisions = int(row["decisions"])
        retrieved = int(row["num_retrieved"])
        reveals = int(row["provider_reveal_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("endpoint lacks valid count/outcome fields") from exc
    if (row.get("layout_fingerprint") != episode["layout_fingerprint"]
            or str(row.get("parameter_group", "")) != episode["geometry"]
            or containers != int(episode["num_containers"])
            or row.get("family") != condition["family"]
            or int(row.get("repeat", -1)) != int(condition.get("repeat", 0))
            or not is_sha(row.get("forecast_table_sha256"))):
        raise ValueError("endpoint metadata or full forecast SHA mismatch")
    if relocations < 0 or containers <= 0 or not math.isfinite(ratio):
        raise ValueError("endpoint has invalid relocation outcome")
    if not math.isclose(ratio, relocations / containers, rel_tol=0, abs_tol=1e-12):
        raise ValueError("relocations_per_container mismatch")
    if decisions != relocations or retrieved != containers or reveals != containers - 1:
        raise ValueError("complete endpoint violates decision/retrieval/reveal conservation")


def validate_pairs(rows: Sequence[Mapping[str, Any]], condition_ids: set[str],
                   episodes: Mapping[str, Mapping[str, Any]], expected_gp: Mapping[str, Any]) -> None:
    expected = {(c, e, a) for c in condition_ids for e in episodes for a in ARMS}
    keyed = {}
    for row in rows:
        key = (str(row.get("condition_id", "")), str(row.get("episode_id", "")), str(row.get("arm", "")))
        if key not in expected:
            raise ValueError(f"unexpected endpoint pairing key: {key}")
        if key in keyed:
            raise ValueError(f"duplicate endpoint pairing key: {key}")
        if row.get("gp_model") != expected_gp:
            raise ValueError("comparison mixes GP model identities")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError(f"incomplete endpoint matrix: {len(expected-set(keyed))} missing")
    for condition in condition_ids:
        for episode_id, episode in episodes.items():
            a, b = keyed[(condition, episode_id, ARMS[0])], keyed[(condition, episode_id, ARMS[1])]
            for field in ("layout_fingerprint", "forecast_table_sha256", "num_containers",
                          "parameter_group", "family", "repeat"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"paired arms disagree on {field}: {condition}/{episode_id}")
            if (a.get("layout_fingerprint") != episode["layout_fingerprint"]
                    or str(a.get("parameter_group")) != episode["geometry"]):
                raise ValueError(f"endpoint differs from episode manifest: {condition}/{episode_id}")


def load_endpoints(paths: Sequence[Path], data_path: Path, rule_path: Path):
    if not paths:
        raise ValueError("provide fresh runner endpoint JSONL files")
    data_path, rule_path = data_path.resolve(), rule_path.resolve()
    data_sha, rule_sha = sha_file(data_path), sha_file(rule_path)
    actual_episodes = actual_episode_index(data_path)
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    gp = active_gp_record(rule)
    registry, rows, seen = condition_registry(), [], set()
    common_episodes, inputs = None, []
    for endpoint_arg in paths:
        endpoint = endpoint_arg.resolve()
        manifest_path = Path(str(endpoint) + ".manifest.json")
        if not endpoint.is_file() or not manifest_path.is_file():
            raise ValueError(f"endpoint or companion manifest missing: {endpoint.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "uwsaa_trajectory_run_manifest_v1"
                or manifest.get("suite") != "main" or manifest.get("smoke_only") is not False
                or manifest.get("status") != "complete" or int(manifest.get("failed_rows", -1)) != 0):
            raise ValueError(f"not a complete non-smoke MAIN run: {endpoint.name}")
        if manifest.get("dataset_sha256") != data_sha:
            raise ValueError(f"run dataset SHA differs from current Caserta: {endpoint.name}")
        if manifest.get("gp_model") != gp:
            raise ValueError(f"stale/mixed GP identity: {endpoint.name}")
        if manifest.get("solvers") != ["GP"] or set(manifest.get("arms", ())) != set(ARMS):
            raise ValueError("run must contain exactly GP × NATIVE20/UWSAA")
        conds = manifest.get("conditions")
        if not isinstance(conds, list) or not conds:
            raise ValueError("run manifest has no conditions")
        ids = []
        for item in conds:
            if not isinstance(item, dict):
                raise ValueError("malformed condition record")
            cid = str(item.get("condition_id", ""))
            if registry.get(cid) != item:
                raise ValueError(f"condition outside frozen registry: {cid}")
            ids.append(cid)
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("condition shards overlap or duplicate conditions")
        seen.update(ids)
        episodes = episode_index(manifest.get("episodes"))
        if episodes != actual_episodes:
            raise ValueError("run episode manifest differs from the current Caserta data")
        if common_episodes is None:
            common_episodes = episodes
        elif episodes != common_episodes:
            raise ValueError("condition shards use different episode manifests")
        expected_rows = len(ids) * len(episodes) * 2
        if (manifest.get("endpoint_file") != endpoint.name
                or int(manifest.get("expected_rows", -1)) != expected_rows
                or int(manifest.get("actual_rows", -1)) != expected_rows):
            raise ValueError(f"manifest row count/filename mismatch: {endpoint.name}")
        count = 0
        with endpoint.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {endpoint.name}:{line_no}") from exc
                if not isinstance(row, dict) or row.get("schema") != "uwsaa_trajectory_endpoint_v1":
                    raise ValueError(f"non-runner data at {endpoint.name}:{line_no}")
                if row.get("solver") != "GP" or row.get("arm") not in ARMS or row.get("condition_id") not in ids:
                    raise ValueError(f"unexpected solver/arm/condition at {endpoint.name}:{line_no}")
                episode = episodes.get(str(row.get("episode_id", "")))
                condition = registry[str(row["condition_id"])]
                if episode is None:
                    raise ValueError(f"episode absent from manifest at {endpoint.name}:{line_no}")
                try:
                    validate_endpoint_row(row, episode, condition, gp)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{exc} at {endpoint.name}:{line_no}") from exc
                row["gp_model"] = gp
                row["_geometry"], row["_fingerprint"] = episode["geometry"], episode["layout_fingerprint"]
                rows.append(row)
                count += 1
        if count != expected_rows:
            raise ValueError(f"{endpoint.name}: got {count} rows, expected {expected_rows}")
        inputs.append({"endpoint_file": endpoint.name, "endpoint_sha256": sha_file(endpoint),
                       "manifest_file": manifest_path.name, "manifest_sha256": sha_file(manifest_path),
                       "conditions": ids, "rows": count})
    if seen != set(registry):
        raise ValueError(f"missing MAIN conditions: {sorted(set(registry)-seen)}")
    if common_episodes is None:
        raise ValueError("no episode manifest loaded")
    validate_pairs(rows, set(registry), common_episodes, gp)
    binding = {
        "gp": gp, "gp_sha": canonical_sha(gp), "data_sha": data_sha,
        "data_file": data_path.relative_to(ROOT).as_posix() if data_path.is_relative_to(ROOT) else data_path.name,
        "rule_sha": rule_sha,
        "rule_file": rule_path.relative_to(ROOT).as_posix() if rule_path.is_relative_to(ROOT) else rule_path.name,
        "conditions": registry, "inputs": inputs,
    }
    return rows, binding


def make_cells(rows: Sequence[Mapping[str, Any]], metric: str):
    cell, found = {}, defaultdict(set)
    for row in rows:
        geo, fp = str(row["_geometry"]), str(row["_fingerprint"])
        key = (geo, fp, str(row["condition_id"]), str(row["arm"]))
        if key in cell:
            raise ValueError(f"duplicate metric cell: {key}")
        cell[key] = float(row[metric])
        found[geo].add(fp)
    layouts = {g: sorted(fps) for g, fps in sorted(found.items())}
    return cell, layouts, tuple(sorted(layouts))


def macro_effect(cell, layouts, geometries, condition_ids, metric):
    base_by_geo, candidate_by_geo = [], []
    for geo in geometries:
        base_layout, candidate_layout = [], []
        for fp in layouts[geo]:
            base_layout.append(float(np.mean([cell[(geo, fp, c, "NATIVE20")] for c in condition_ids])))
            candidate_layout.append(float(np.mean([cell[(geo, fp, c, "UWSAA")] for c in condition_ids])))
        base_by_geo.append(float(np.mean(base_layout)))
        candidate_by_geo.append(float(np.mean(candidate_layout)))
    base, candidate = float(np.mean(base_by_geo)), float(np.mean(candidate_by_geo))
    if base <= 0:
        raise ValueError(f"nonpositive macro baseline for {metric}")
    return base, candidate, (base-candidate)/base


def paired_indices(layouts, geometries, draws=DRAWS, seed=SEED):
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return {g: rng.integers(0, len(layouts[g]), size=(draws, len(layouts[g])), dtype=np.int32)
            for g in sorted(geometries)}


def bootstrap_draws(cell, layouts, geometries, conditions, indices):
    base_geo, candidate_geo = [], []
    for geo in geometries:
        fps = layouts[geo]
        base_layout = np.asarray([np.mean([cell[(geo, fp, c, "NATIVE20")] for c in conditions])
                                  for fp in fps], dtype=np.float64)
        candidate_layout = np.asarray([np.mean([cell[(geo, fp, c, "UWSAA")] for c in conditions])
                                       for fp in fps], dtype=np.float64)
        # Same draw indices are applied to both arms and every condition.
        base_geo.append(base_layout[indices[geo]].mean(axis=1))
        candidate_geo.append(candidate_layout[indices[geo]].mean(axis=1))
    return np.stack(base_geo, axis=1).mean(axis=1), np.stack(candidate_geo, axis=1).mean(axis=1)


def summarize(cell, layouts, geometries, conditions, indices, metric):
    base, candidate, point = macro_effect(cell, layouts, geometries, conditions, metric)
    base_draw, candidate_draw = bootstrap_draws(cell, layouts, geometries, conditions, indices)
    rel = np.full(base_draw.shape, np.nan, dtype=np.float64)
    valid = base_draw > 0
    rel[valid] = (base_draw[valid]-candidate_draw[valid])/base_draw[valid]
    if not np.isfinite(rel).any():
        raise ValueError(f"no defined bootstrap draws: {metric}")
    low, high = np.nanquantile(rel, [0.025, 0.975])
    return ({"metric": metric, "baseline": "NATIVE20", "candidate": "UWSAA",
             "baseline_macro_mean": base, "candidate_macro_mean": candidate,
             "relative_reduction": point, "ci95_lower": float(low), "ci95_upper": float(high),
             "bootstrap_defined_draw_fraction": float(np.isfinite(rel).mean()),
             "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
             "evidence_status": "pending_review",
             "direction": "positive favors UWSAA; lower relocations are better"}, rel)


def analyze(rows, conditions):
    family_conditions = {f: sorted(c for c, x in conditions.items() if x["family"] == f)
                         for f in FAMILIES}
    if [len(family_conditions[f]) for f in FAMILIES] != [6, 8, 6]:
        raise AssertionError("registered SS/TW/DP counts changed")
    metric_cells = {m: make_cells(rows, m) for m in METRICS}
    layouts, geometries = metric_cells["relocations"][1:]
    indices = paired_indices(layouts, geometries)
    family_rows, condition_rows = [], []
    for metric in METRICS:
        cell, metric_layouts, metric_geometries = metric_cells[metric]
        if metric_layouts != layouts or metric_geometries != geometries:
            raise AssertionError("metrics use different endpoint layouts")
        family_rel = []
        family_draws = []
        for family in FAMILIES:
            summary, draws = summarize(cell, layouts, geometries, family_conditions[family], indices, metric)
            summary.update(family=family, condition_count=len(family_conditions[family]))
            family_rows.append(summary)
            family_rel.append(summary["relative_reduction"])
            family_draws.append(draws)
        combined = np.mean(np.stack(family_draws, axis=1), axis=1)
        low, high = np.nanquantile(combined, [0.025, 0.975])
        family_rows.append({
            "metric": metric, "family": "SS_TW_DP_equal_family_mean",
            "baseline": "NATIVE20", "candidate": "UWSAA",
            "baseline_macro_mean": "", "candidate_macro_mean": "",
            "relative_reduction": float(np.mean(family_rel)), "ci95_lower": float(low),
            "ci95_upper": float(high), "bootstrap_defined_draw_fraction": float(np.isfinite(combined).mean()),
            "bootstrap_draws": DRAWS, "bootstrap_seed": SEED, "condition_count": 20,
            "weighting": "equal mean of family-specific macro relative reductions",
            "evidence_status": "pending_review",
            "direction": "positive favors UWSAA; lower relocations are better",
        })
        for cid, definition in sorted(conditions.items()):
            summary, draws = summarize(cell, layouts, geometries, [cid], indices, metric)
            summary.update(condition_id=cid, family=definition["family"],
                           repeat=int(definition.get("repeat", 0)))
            if definition["family"] == "CLEAN" and metric == "relocations":
                summary["gp_only_clean_lower_q_0_05_over_6"] = float(np.nanquantile(draws, 0.05/6))
                summary["clean_gate_scope"] = "GP component only; not all-six NI or MEAN6"
            condition_rows.append(summary)
    return family_rows, condition_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def execute(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"output dir exists; refusing overwrite: {output.name}")
    rows, binding = load_endpoints(args.endpoint, args.data, args.rule)
    family_rows, condition_rows = analyze(rows, binding["conditions"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "family_effects.csv", family_rows)
    write_csv(output / "condition_effects.csv", condition_rows)
    manifest = {
        "schema": "uwsaa_gp_arm_paired_analysis_v1", "status": "pending_review",
        "analysis_script_sha256": sha_file(Path(__file__)), "suite": "main",
        "scope": "GP-only paired arms; no MEAN6 or all-six statistics",
        "arms": {"baseline": "NATIVE20", "candidate": "UWSAA"},
        "expected_design": {"conditions": 41, "layouts": 840, "geometries": 21,
                            "layouts_per_geometry": 40, "endpoint_rows": 68_880},
        "observed_endpoint_rows": len(rows),
        "dataset": {"file": binding["data_file"], "sha256": binding["data_sha"]},
        "gp_model": binding["gp"], "gp_model_identity_sha256": binding["gp_sha"],
        "active_rule_config": {"file": binding["rule_file"], "sha256": binding["rule_sha"]},
        "statistics": {
            "primary": "relocations", "secondary": "relocations_per_container",
            "pure_family_conditions": {"subset_shuffle": 6, "time_window_shuffle": 8,
                                       "delayed_positions": 6},
            "macro_order": ("conditions within family per layout; layouts within geometry; "
                            "equal 21 geometry means; family relative reduction; equal SS/TW/DP mean"),
            "bootstrap": {"draws": DRAWS, "generator": "NumPy PCG64", "seed": SEED,
                          "interval": "percentile 95%",
                          "pairing": "same layout indices across both arms and all conditions"},
            "condition_level": "CLEAN, overlay, and each regime condition remain separate",
            "clean_gate": "GP-only .05/6 component is descriptive; no all-six or MEAN6 conclusion",
        },
        "inputs": binding["inputs"],
        "outputs": {"family_effects.csv": sha_file(output / "family_effects.csv"),
                    "condition_effects.csv": sha_file(output / "condition_effects.csv")},
        "evidence_status": "pending_review",
        "limitations": ["Caserta remains sealed for UW-SAA and is never used for selection or tuning.",
                        "This comparison cannot produce six-solver statistics or MEAN6."],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path, action="append", required=True,
                        help="fresh runner JSONL; repeat for disjoint condition shards")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rule", type=Path, default=DEFAULT_RULE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"GP paired analysis refused: {exc}") from exc
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "endpoint_rows": result["observed_endpoint_rows"],
                      "evidence_status": result["evidence_status"]}, indent=2))


if __name__ == "__main__":
    main()