"""Independently audit GP MAIN using raw JSONL and multiplicity-weighted bootstrap.

Only the standard library and NumPy are imported. The registered analyzer,
its endpoint loader, its condition registry, and its statistics are not called.
Published CSVs are read only after the independent estimates are complete.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gp-caserta-main-20261024"
RAW_ARCHIVE_SHA256 = "5c2d76ab977dcd3e50853042a5d9452136ad7337266ecc63f1226710c4eb09a4"
DRAWS, SEED, TOLERANCE = 10000, 202610195903, 1e-12
ARMS = ("NATIVE20", "UWSAA")
METRICS = ("relocations", "relocations_per_container")
FAMILIES = ("subset_shuffle", "time_window_shuffle", "delayed_positions")
BINDINGS = ("source_manifest_sha256", "protocol_sha256", "train_bank_sha256",
            "implementation_sha256", "seed_receipt_sha256")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def condition_design() -> list[dict]:
    """Reconstruct the written 41-condition design without importing the registry."""
    rows = []
    families = [
        ("subset_shuffle", "p", ("0.1", "0.3", "0.5")),
        ("time_window_shuffle", "B", ("2", "5", "10", "20")),
        ("delayed_positions", "p", ("0.1", "0.3", "0.5")),
        ("overlay_mix", "p=0.3|B", ("5",)),
        ("regime_mix", "p=0.3|B=5|profile", (
            "C_EQ", "C_DOM_CLEAN", "C_DOM_TW", "C_DOM_SS", "C_DOM_DP",
            "N_EQ", "N_DOM_TW", "N_DOM_SS", "N_DOM_DP")),
    ]
    for family, parameter, levels in families:
        for level in levels:
            for repeat in (0, 1):
                cid = f"{family}|{parameter}={level}|repeat={repeat}"
                params = dict(bit.split("=", 1) for bit in cid.split("|")[1:])
                rows.append({"condition_id": cid, "family": family, "repeat": repeat,
                             "p": float(params["p"]) if "p" in params else None,
                             "window_size": int(params["B"]) if "B" in params else None,
                             "profile": params.get("profile")})
    rows.append({"condition_id": "CLEAN", "family": "CLEAN", "repeat": 0,
                 "p": None, "window_size": None, "profile": None})
    require(len(rows) == 41, "independent condition design is not 41 rows")
    return rows


def fingerprint(episode: dict) -> str:
    sequence = episode["retrieval_sequence"]
    labels = [value for stack in episode["initial_stacks"] for value in stack]
    require(len(labels) == len(set(labels)) == len(sequence) == len(set(sequence))
            and set(labels) == set(sequence), "dataset labels are not a retrieval permutation")
    ranks = {label: position + 1 for position, label in enumerate(sequence)}
    normalized = sorted([ranks[label] for label in stack] for stack in episode["initial_stacks"])
    data = {"max_tiers": episode["max_tiers"], "num_stacks": episode["num_stacks"],
            "rank_stacks": normalized}
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("ascii"))


def inspect_training(rule: dict, evaluation: list[dict]) -> dict:
    training = read_json(ROOT / "experiments/gp_h20/data/train_bank.json")
    train_fps = {fingerprint(row) for row in training}
    test_fps = {fingerprint(row) for row in evaluation}
    require(len(training) == len(train_fps) == 840, "training bank is incomplete or duplicated")
    require(len(evaluation) == len(test_fps) == 840, "evaluation bank is incomplete or duplicated")
    require(not train_fps & test_fps, "training/evaluation layout overlap")
    require(digest((ROOT / "experiments/gp_h20/data/train_bank.json").read_bytes())
            == rule["train_bank_sha256"], "training bank hash mismatch")
    require(digest((ROOT / "experiments/gp_h20/protocol.json").read_bytes())
            == rule["protocol_sha256"], "training protocol hash mismatch")
    source = read_json(ROOT / "third_party/crp_online/SOURCE_MANIFEST.json")
    require(source["manifest_sha256"] == rule["source_manifest_sha256"], "source identity mismatch")
    for item in source["files"]:
        payload = (ROOT / "third_party/crp_online" / item["path"]).read_bytes()
        require(digest(payload) == item["sha256"] and len(payload) == item["size"],
                f"vendored source file differs: {item['path']}")
    directory = ROOT / "artifacts/gp_h20/20261024_formal_30seed"
    scores = []
    for seed in range(1, 31):
        receipt = read_json(directory / f"seed_{seed:02}.json")
        require(receipt["status"] == "complete" and receipt["seed"] == seed
                and receipt["evaluations"] == 51000 and receipt["training_rows"] == 840,
                f"incomplete training receipt: seed {seed}")
        for field in BINDINGS[:-1]:
            require(receipt[field] == rule[field], f"training identity differs: seed {seed}/{field}")
        scores.append((receipt["training_total_relocations"], seed))
    best_score, best_seed = min(scores)
    identity = rule["model_identity"]
    require((best_score, best_seed) == (identity["training_total_relocations"], identity["selected_seed"]),
            "selected seed is not the training-only minimum with lower-seed tie break")
    selected = directory / f"seed_{best_seed:02}.json"
    require(digest(selected.read_bytes()) == identity["seed_receipt_sha256"], "selected receipt hash mismatch")
    require(read_json(selected)["formula"] == rule["formula"], "selected formula differs")
    totals = {int(key): value for key, value in identity["horizon_training_totals"].items()}
    require(set(totals) == {0, 2, 5, 10, 20}
            and min(totals, key=lambda horizon: (totals[horizon], horizon)) == rule["best_fixed_h"] == 20,
            "fixed horizon is not the training-only minimum")
    return {"training_layouts": 840, "evaluation_layouts": 840, "overlap": 0,
            "complete_seed_receipts": 30, "selected_seed": best_seed,
            "training_relocations": best_score, "best_fixed_h": 20,
            "source_files_verified": len(source["files"])}


def read_raw() -> tuple[np.ndarray, list[dict], dict]:
    """Stream archive rows into a strictly keyed geometry/layout/condition tensor."""
    archive = RESULTS / "gp_caserta_full_results.zip"
    require(digest(archive.read_bytes()) == RAW_ARCHIVE_SHA256, "raw archive differs from completed run")
    evaluation = read_json(ROOT / "data/caserta_table2_840.json")
    dataset_sha = digest((ROOT / "data/caserta_table2_840.json").read_bytes())
    rule = read_json(ROOT / "configs/gp_rule.json")
    require(rule["status"] == "trained" and digest(rule["formula"].encode()) == rule["formula_sha256"],
            "active GP rule formula/status mismatch")
    training = inspect_training(rule, evaluation)
    gp = {field: rule.get(field, rule["model_identity"].get(field)) for field in BINDINGS}
    gp.update(status="trained", formula_sha256=rule["formula_sha256"], best_fixed_h=20,
              model_identity=rule["model_identity"])
    for field in BINDINGS:
        require(gp[field] == rule["model_identity"][field], f"active model binding differs: {field}")
    episodes, geometry_layouts = {}, {}
    for row in evaluation:
        fp, geo = fingerprint(row), row["parameter_group"]
        require(row["episode_id"] not in episodes, "duplicate dataset episode ID")
        episodes[row["episode_id"]] = {"episode_id": row["episode_id"], "layout_fingerprint": fp,
                                       "parameter_group": geo,
                                       "num_containers": len(row["retrieval_sequence"])}
        geometry_layouts.setdefault(geo, []).append(fp)
    geometries = sorted(geometry_layouts)
    require(len(geometries) == 21 and all(len(set(fps)) == 40 for fps in geometry_layouts.values()),
            "dataset is not 21 geometries of 40 unique layouts")
    layout_index = {(geo, fp): (g, layout) for g, geo in enumerate(geometries)
                    for layout, fp in enumerate(sorted(geometry_layouts[geo]))}
    design = condition_design()
    shape = (21, 40, 41, 2, 2)  # geometry, layout, condition, metric, arm
    values, seen = np.zeros(shape, dtype=np.float64), np.zeros((21, 40, 41, 2), dtype=bool)
    keyed, endpoint_count = {}, 0
    input_hashes = {}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "duplicate archive member")
        sums = {}
        for line in z.read("SHA256SUMS.txt").decode().splitlines():
            sha, name = line.split("  ", 1)
            require(name not in sums, "duplicate archive checksum record")
            sums[name] = sha
        require(set(z.namelist()) == set(sums) | {"SHA256SUMS.txt"}, "archive checksum coverage mismatch")
        for name, sha in sums.items():
            require(digest(z.read(name)) == sha, f"archive payload hash differs: {name}")
        require({name for name in z.namelist() if name.startswith(("local/", "cloud/"))
                 and name.endswith(".jsonl")} == {
                     f"{'local' if index < 14 else 'cloud'}/condition-{index + 1:02}.jsonl"
                     for index in range(41)}, "extra or missing formal endpoint shard")
        for ci, definition in enumerate(design):
            partition = "local" if ci < 14 else "cloud"
            name = f"{partition}/condition-{ci+1:02}.jsonl"
            companion = name + ".manifest.json"
            manifest = json.loads(z.read(companion))
            require(manifest["status"] == "complete" and manifest["smoke_only"] is False
                    and manifest["suite"] == "main" and manifest["failed_rows"] == 0
                    and manifest["actual_rows"] == manifest["expected_rows"] == 1680
                    and manifest["conditions"] == [definition] and manifest["solvers"] == ["GP"]
                    and set(manifest["arms"]) == set(ARMS) and manifest["gp_model"] == gp
                    and manifest["dataset_sha256"] == dataset_sha
                    and manifest["endpoint_file"] == Path(name).name,
                    f"formal shard metadata differs: {name}")
            require(len(manifest["episodes"]) == 840
                    and {item["episode_id"]: item for item in manifest["episodes"]} == episodes,
                    f"manifest episodes differ from independently fingerprinted dataset: {name}")
            count = 0
            with z.open(name) as stream:
                for line in stream:
                    row = json.loads(line)
                    require(row["schema"] == "uwsaa_trajectory_endpoint_v1" and row["complete"] is True
                            and not any(key.startswith("failure_") for key in row), "failed/malformed endpoint")
                    episode = episodes[row["episode_id"]]
                    require(all(row[field] == expected for field, expected in episode.items()),
                            "endpoint dataset fingerprint/geometry/container metadata differs")
                    require(row["solver"] == "GP" and row["arm"] in ARMS and row["gp_model"] == gp
                            and row["condition_id"] == definition["condition_id"]
                            and row["family"] == definition["family"] and row["repeat"] == definition["repeat"],
                            "endpoint condition/solver/model mismatch")
                    for field in ("relocations", "num_containers", "decisions", "num_retrieved", "provider_reveal_count"):
                        require(type(row[field]) is int and row[field] >= 0, f"non-integer count: {field}")
                    n, moves = row["num_containers"], row["relocations"]
                    require(n > 0 and row["decisions"] == moves and row["num_retrieved"] == n
                            and row["provider_reveal_count"] == n-1, "terminal conservation mismatch")
                    require(math.isfinite(row["relocations_per_container"])
                            and abs(row["relocations_per_container"] - moves/n) <= 1e-12,
                            "per-container metric differs from raw count/N")
                    geo = row["parameter_group"]
                    g, layout = layout_index[(geo, row["layout_fingerprint"])]
                    arm = ARMS.index(row["arm"])
                    require(not seen[g, layout, ci, arm], "duplicate formal endpoint")
                    seen[g, layout, ci, arm] = True
                    values[g, layout, ci, :, arm] = moves, moves/n
                    sha = row["forecast_table_sha256"]
                    require(len(sha) == 64 and all(char in "0123456789abcdef" for char in sha),
                            "invalid forecast hash")
                    key = (row["condition_id"], row["episode_id"], row["arm"])
                    keyed[key] = row
                    count += 1
            require(count == 1680, f"formal row count differs: {name}")
            endpoint_count += count
            input_hashes[name] = sums[name]
            input_hashes[companion] = sums[companion]
        require(bool(seen.all()) and endpoint_count == 68880, "incomplete formal matrix")
        for definition in design:
            for eid in episodes:
                first = keyed[(definition["condition_id"], eid, "NATIVE20")]
                second = keyed[(definition["condition_id"], eid, "UWSAA")]
                require(first["forecast_table_sha256"] == second["forecast_table_sha256"],
                        "paired policies use different forecasts")
        # Independently compare all stored probe rows with both the local
        # cross-host reference and the formal matrix, rather than trusting a flag.
        reference = json.loads(z.read("integrity/probe_diagnostic_only/reference.json"))
        probe = [json.loads(line) for line in z.read("integrity/probe_diagnostic_only/probe.jsonl").splitlines()]
        key_fields = ("condition_id", "episode_id", "arm")
        refs = {tuple(row[k] for k in key_fields): row for row in reference["records"]}
        require(len(probe) == len(refs) == len({tuple(row[k] for k in key_fields) for row in probe}) == 252,
                "probe/reference coverage differs")
        for row in probe:
            key = tuple(row[k] for k in key_fields)
            require(key in refs and key in keyed, "probe pairing key missing")
            for field in reference["compared_fields"]:
                require(row[field] == refs[key][field] == keyed[key][field], f"probe/formal mismatch: {field}")
        runtime_files = set()
        for partition, indexes in (("local", list(range(1,15))), ("cloud", list(range(15,42)))):
            manifest = json.loads(z.read(f"{partition}/run_manifest.json"))
            require(manifest["status"] == "complete" and not manifest["failures"]
                    and not manifest["active_children"] and manifest["condition_indexes"] == indexes
                    and manifest["completed_endpoint_rows"] == manifest["expected_endpoint_rows"] == len(indexes)*1680
                    and manifest["gp_model"] == gp and manifest["dataset_sha256"] == dataset_sha,
                    f"partition manifest incomplete: {partition}")
            for path, sha in manifest["runtime_inventory"]["files"].items():
                require(digest((ROOT / path).read_bytes()) == sha, f"recorded runtime file differs: {path}")
                runtime_files.add(path)
    facts = {"formal_rows": endpoint_count, "paired_cells": endpoint_count//2,
             "formal_shards": 41, "geometries": geometries, "layouts_per_geometry": 40,
             "archive_members_checked": len(sums), "probe_reference_formal_matches": 252,
             "runtime_files_verified": len(runtime_files), "training_selection": training,
             "raw_archive_sha256": RAW_ARCHIVE_SHA256, "dataset_sha256": dataset_sha,
             "rule_sha256": digest((ROOT / "configs/gp_rule.json").read_bytes()),
             "gp_model_identity_sha256": digest(json.dumps(gp, sort_keys=True, separators=(",", ":"),
                                                          ensure_ascii=True).encode("ascii")),
             "formal_input_hashes": input_hashes}
    return values, design, facts


def multiplicities(samples: np.ndarray, layouts: int) -> np.ndarray:
    """Convert draws of layout indices to independent frequency-weight rows."""
    offsets = np.arange(samples.shape[0], dtype=np.int64)[:, None] * layouts
    return np.bincount((samples + offsets).reshape(-1), minlength=samples.shape[0]*layouts).reshape(-1, layouts)


def linear_percentile(values: np.ndarray, probability: float):
    """Compute a linear percentile explicitly, without the analyzer's quantile call."""
    ordered = np.sort(values, axis=0)
    position = (ordered.shape[0]-1)*probability
    below, above = math.floor(position), math.ceil(position)
    return ordered[below] + (ordered[above]-ordered[below])*(position-below)


def independent_statistics(values: np.ndarray, design: list[dict]) -> dict[tuple, dict]:
    groups = [[index for index, row in enumerate(design) if row["family"] == family]
              for family in FAMILIES]
    require([len(group) for group in groups] == [6,8,6], "pure family condition count mismatch")
    # All condition estimates and the three condition-averaged family estimates
    # are formed at layout level before either geometry or bootstrap averaging.
    family_values = np.stack([values[:, :, group].sum(axis=2)/len(group) for group in groups], axis=2)
    layout_values = np.concatenate((values, family_values), axis=2)
    geometry_means = layout_values.sum(axis=1)/40
    means = geometry_means.sum(axis=0)/21
    boot = np.zeros((DRAWS, *means.shape), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    for geometry in range(21):
        samples = rng.integers(0,40,size=DRAWS*40,dtype=np.int32).reshape(DRAWS,40)
        weights = multiplicities(samples,40)
        require(bool((weights.sum(axis=1) == 40).all()), "bootstrap multiplicity is not 40 per draw")
        weighted = weights @ layout_values[geometry].reshape(40,-1)
        boot += weighted.reshape(DRAWS,*means.shape)/40/21
    require(bool((means[...,0] > 0).all()) and bool((boot[...,0] > 0).all()),
            "undefined nonpositive bootstrap baseline")
    points = (means[...,0]-means[...,1])/means[...,0]
    draws = (boot[...,0]-boot[...,1])/boot[...,0]
    low, high = linear_percentile(draws,.025),linear_percentile(draws,.975)
    records = {}
    for metric_index, metric in enumerate(METRICS):
        for component in range(44):
            is_family = component >= 41
            row = design[component] if not is_family else {"family":FAMILIES[component-41]}
            label = row["family"] if is_family else row["condition_id"]
            record = {"baseline_macro_mean":float(means[component,metric_index,0]),
                      "candidate_macro_mean":float(means[component,metric_index,1]),
                      "relative_reduction":float(points[component,metric_index]),
                      "ci95_lower":float(low[component,metric_index]),
                      "ci95_upper":float(high[component,metric_index]),
                      "bootstrap_defined_draw_fraction":1.0,"bootstrap_draws":DRAWS,"bootstrap_seed":SEED}
            if is_family:
                record["condition_count"] = len(groups[component-41])
            else:
                record["repeat"] = row["repeat"]
            if label == "CLEAN" and metric == "relocations":
                record["gp_only_clean_lower_q_0_05_over_6"] = float(linear_percentile(draws[:,component,metric_index],.05/6))
            records[("family" if is_family else "condition",metric,label)] = record
        combined = draws[:,41:,metric_index].sum(axis=1)/3
        records[("family",metric,"SS_TW_DP_equal_family_mean")] = {
            "relative_reduction":float(points[41:,metric_index].sum()/3),
            "ci95_lower":float(linear_percentile(combined,.025)),
            "ci95_upper":float(linear_percentile(combined,.975)),
            "bootstrap_defined_draw_fraction":1.0,"bootstrap_draws":DRAWS,"bootstrap_seed":SEED,
            "condition_count":20}
    return records


def compare_published(records: dict) -> tuple[list[dict], dict]:
    differences, row_counts = [], {}
    expected_labels = {
        "family_effects.csv":{key for key in records if key[0] == "family"},
        "condition_effects.csv":{key for key in records if key[0] == "condition"},
        "summary.csv":{key for key in records if key[0] == "family" or key[2] == "CLEAN"},
    }
    for name in expected_labels:
        with (RESULTS/name).open(encoding="utf-8",newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = []
        for row in rows:
            kind = "condition" if row.get("condition_id") else "family"
            key = (kind,row["metric"],row["condition_id"] if kind == "condition" else row["family"])
            require(key in records, f"unknown published comparison: {key}")
            require(row["baseline"] == "NATIVE20" and row["candidate"] == "UWSAA"
                    and row["direction"] == "positive favors UWSAA; lower relocations are better",
                    f"published arm/direction label differs: {key}")
            keys.append(key)
            for field, value in records[key].items():
                require(row.get(field,"") != "", f"published statistic missing: {name}/{key}/{field}")
                published = float(row[field])
                error = abs(value-published)
                differences.append({"table":name,"metric":key[1],"comparison":key[2],"field":field,
                                    "published":published,"independent":value,"absolute_difference":error,
                                    "tolerance":TOLERANCE,"pass":math.isfinite(error) and error <= TOLERANCE})
        require(len(keys) == len(set(keys)) and set(keys) == expected_labels[name],
                f"published comparison coverage differs: {name}")
        row_counts[name] = len(rows)
    require(all(row["pass"] for row in differences), "independently recomputed statistics differ beyond tolerance")
    return differences,row_counts


def run(output: Path, comparison: Path) -> dict:
    values,design,facts = read_raw()
    records = independent_statistics(values,design)
    differences,counts = compare_published(records)
    condition_rows = [records[("condition","relocations",row["condition_id"])] for row in design]
    primary = records[("family","relocations","SS_TW_DP_equal_family_mean")]
    clean = records[("condition","relocations","CLEAN")]
    output.parent.mkdir(parents=True,exist_ok=True)
    comparison.parent.mkdir(parents=True,exist_ok=True)
    with comparison.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(differences[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(differences)
    report = {
        "schema":"uwsaa_gp_main_independent_implementation_review_v1",
        "status":"PASS_INDEPENDENT_NUMERICAL_REVIEW",
        "review_date":datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "reviewer":"Codex; same assistant session, separately written statistical implementation",
        "independence":"Implementation independence only; no external human or peer review is claimed",
        "implementation_sha256":digest(Path(__file__).read_bytes()),
        "repository_base_commit":subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree_note":"Base commit alone does not identify this uncommitted worktree; implementation, formal inputs, active rule and published table hashes are recorded separately.",
        "imports":"Python standard library and NumPy only; no repository/analyzer imports",
        "environment":{"python":platform.python_version(),"numpy":np.__version__},
        "checks":facts,
        "statistics":{"draws":DRAWS,"seed":SEED,"tolerance_absolute":TOLERANCE,
                      "bootstrap_implementation":"layout multiplicity counts multiplied by outcome matrices; shared across arms, metrics and conditions; equal geometry weights",
                      "quantile_implementation":"explicit sorted linear interpolation",
                      "family_effect":"equal average of SS/TW/DP family-relative reductions",
                      "tables_checked":counts,"numeric_fields_checked":len(differences),
                      "max_absolute_difference":max(row["absolute_difference"] for row in differences),
                      "max_interval_difference":max(row["absolute_difference"] for row in differences if row["field"].startswith("ci95_")),
                      "mismatches":0},
        "primary_relocations":primary,"clean_relocations":clean,
        "condition_direction_checks":{"positive_estimates":sum(row["relative_reduction"]>0 for row in condition_rows),
                                      "pointwise_intervals_above_zero":sum(row["ci95_lower"]>0 for row in condition_rows),
                                      "pointwise_intervals_below_zero":sum(row["ci95_upper"]<0 for row in condition_rows)},
        "interpretation_review":{"primary_improvement_and_CLEAN_loss_both_reported":True,
                                 "intervals_are_pointwise_not_simultaneous":True,
                                 "no_new_GP_vs_other_solver_or_MEAN6_claim":True,
                                 "GP_only_clean_lower_bound_does_not_establish_all_six_noninferiority":True},
        "limits":["Raw endpoint evidence is checked; original trajectories are not rerun and full action histories are not available in this archive.",
                  "Training receipts and train/test layout separation are checked; training is not repeated and this does not establish the chronology of all prior human decisions.",
                  "Scientific scope remains the frozen GP two-arm Caserta MAIN comparison.",
                  "The original CSV evidence_status and frozen analyzer are preserved; this review is a separate evidence record."],
        "comparison_csv_sha256":digest(comparison.read_bytes()),
        "published_table_sha256":{name:digest((RESULTS/name).read_bytes()) for name in
                                  ("family_effects.csv","condition_effects.csv","summary.csv")},
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"work/gp_main_independent_review.json")
    parser.add_argument("--comparison",type=Path,default=ROOT/"work/gp_main_independent_comparison.csv")
    args = parser.parse_args()
    report = run(args.output.resolve(),args.comparison.resolve())
    print(json.dumps({"status":report["status"],"checks":{key:report["checks"][key] for key in
          ("formal_rows","paired_cells","probe_reference_formal_matches")},
          "statistics":report["statistics"],"primary_relocations":report["primary_relocations"],
          "clean_relocations":report["clean_relocations"]},indent=2))


if __name__ == "__main__":
    main()
