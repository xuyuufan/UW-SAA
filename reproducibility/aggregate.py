"""Strictly validate trajectory endpoints and write descriptive aggregates.

Failed or partial runs, duplicate pair keys, missing rows, and inconsistent
endpoint accounting are rejected before any aggregate file is created.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from reproducibility.runner import _semantic_key


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank endpoint row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"endpoint row {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _validate(rows: list[dict], manifest: dict) -> None:
    if manifest.get("schema") != "uwsaa_trajectory_run_manifest_v1":
        raise ValueError("unrecognized or missing run manifest")
    if manifest.get("status") != "complete" or int(manifest.get("failed_rows", -1)) != 0:
        raise ValueError("run manifest is not complete or contains failures")
    expected_count = int(manifest.get("expected_rows", -1))
    if expected_count <= 0 or len(rows) != expected_count or int(manifest.get("actual_rows", -1)) != expected_count:
        raise ValueError("endpoint row count differs from the registered run matrix")
    episodes = manifest.get("episodes", [])
    conditions = manifest.get("conditions", [])
    solvers = manifest.get("solvers", [])
    arms = manifest.get("arms", [])
    if not episodes or not conditions or not solvers or not arms:
        raise ValueError("run manifest has an empty registered dimension")
    if len({item["episode_id"] for item in episodes}) != len(episodes):
        raise ValueError("run manifest contains duplicate episode IDs")
    episode_by_id = {item["episode_id"]: item for item in episodes}
    condition_by_id = {item["condition_id"]: item for item in conditions}
    if len(condition_by_id) != len(conditions):
        raise ValueError("run manifest contains duplicate condition IDs")
    expected = set()
    for episode in episodes:
        for condition in conditions:
            for solver in solvers:
                for arm in arms:
                    expected.add(_semantic_key(episode["episode_id"], episode["layout_fingerprint"], condition, solver, arm))
    if len(expected) != expected_count:
        raise ValueError("run manifest matrix has duplicate semantic pairing keys")
    observed = set()
    paired: dict[str, set[str]] = defaultdict(set)
    forecast_hashes: dict[tuple[str, str, str], str] = {}
    for index, row in enumerate(rows, start=1):
        if row.get("schema") != "uwsaa_trajectory_endpoint_v1":
            raise ValueError(f"unrecognized endpoint schema at row {index}")
        if row.get("complete") is not True:
            raise ValueError(f"incomplete or failed trajectory at row {index}")
        episode_id = str(row.get("episode_id", ""))
        if episode_id not in episode_by_id:
            raise ValueError(f"unexpected episode at row {index}")
        episode = episode_by_id[episode_id]
        condition_id = str(row.get("condition_id", ""))
        if condition_id not in condition_by_id:
            raise ValueError(f"unexpected condition at row {index}")
        condition = condition_by_id[condition_id]
        solver, arm = str(row.get("solver", "")), str(row.get("arm", ""))
        if row.get("layout_fingerprint") != episode.get("layout_fingerprint"):
            raise ValueError(f"layout fingerprint differs from episode manifest at row {index}")
        if row.get("parameter_group") != episode.get("parameter_group"):
            raise ValueError(f"parameter group differs from episode manifest at row {index}")
        if int(row.get("num_containers", -1)) != int(episode.get("num_containers", -2)):
            raise ValueError(f"container count differs from episode manifest at row {index}")
        if row.get("family") != condition.get("family") or int(row.get("repeat", -1)) != int(condition.get("repeat", 0)):
            raise ValueError(f"condition metadata differs from run manifest at row {index}")
        expected_key = _semantic_key(episode_id, str(episode["layout_fingerprint"]), condition, solver, arm)
        if row.get("semantic_key") != expected_key:
            raise ValueError(f"semantic pairing key does not match row fields at row {index}")
        if int(row.get("num_retrieved", -1)) != int(row.get("num_containers", -2)):
            raise ValueError(f"retrieval count is incomplete at row {index}")
        if int(row.get("provider_reveal_count", -1)) != int(row.get("num_containers", -2)) - 1:
            raise ValueError(f"provider reveal count is incomplete at row {index}")
        relocations = int(row.get("relocations", -1))
        count = int(row.get("num_containers", 0))
        ratio = float(row.get("relocations_per_container", math.nan))
        if relocations < 0 or count <= 0 or not math.isfinite(ratio):
            raise ValueError(f"invalid endpoint metric at row {index}")
        if not math.isclose(ratio, relocations / count, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"relocations_per_container does not match endpoint count at row {index}")
        key = expected_key
        if not key or key in observed:
            raise ValueError(f"missing or duplicate semantic pairing key at row {index}")
        observed.add(key)
        pair_key = f"{episode_id}|{condition_id}|{solver}"
        if arm in paired[pair_key]:
            raise ValueError(f"duplicate paired arm at row {index}")
        paired[pair_key].add(arm)
        forecast_hash = str(row.get("forecast_table_sha256", ""))
        if len(forecast_hash) != 64 or any(char not in "0123456789abcdef" for char in forecast_hash):
            raise ValueError(f"missing or invalid forecast table hash at row {index}")
        old_hash = forecast_hashes.setdefault(pair_key, forecast_hash)
        if old_hash != forecast_hash:
            raise ValueError(f"paired arms have different forecast tables at row {index}")
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise ValueError(f"endpoint matrix mismatch: missing={missing}, unexpected={extra}")
    if any(values != set(arms) for values in paired.values()):
        raise ValueError("one or more paired episode-condition-solver groups omit an arm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="permit explicitly labelled smoke summaries")
    args = parser.parse_args()
    manifest_path = args.manifest or args.endpoints.with_suffix(args.endpoints.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(args.endpoints)
    if manifest.get("smoke_only") and not args.allow_smoke:
        parser.error("smoke endpoints are not benchmark results; pass --allow-smoke for a diagnostic summary")
    _validate(rows, manifest)

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition_id"], row["family"], row["repeat"], row["solver"], row["arm"], row["parameter_group"])].append(row)
    fields = ["condition_id", "family", "repeat", "solver", "arm", "geometry",
              "episode_count", "mean_relocations", "mean_relocations_per_container"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or sidecar.exists():
        parser.error("aggregate output already exists; choose a new path")
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            values = grouped[key]
            condition_id, family, repeat, solver, arm, geometry = key
            writer.writerow({
                "condition_id": condition_id,
                "family": family,
                "repeat": repeat,
                "solver": solver,
                "arm": arm,
                "geometry": geometry,
                "episode_count": len(values),
                "mean_relocations": sum(int(row["relocations"]) for row in values) / len(values),
                "mean_relocations_per_container": sum(float(row["relocations_per_container"]) for row in values) / len(values),
            })
    endpoint_hash = hashlib.sha256(args.endpoints.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps({
        "schema": "uwsaa_endpoint_aggregate_manifest_v1",
        "status": "PASS_STRICT_ENDPOINT_VALIDATION",
        "endpoint_file": args.endpoints.name,
        "endpoint_sha256": endpoint_hash,
        "input_rows": len(rows),
        "aggregate_rows": len(grouped),
        "smoke_only": bool(manifest.get("smoke_only")),
        "statistics": "descriptive geometry-by-condition means; no confidence interval",
        "metrics": {"relocations": "integer relocation count", "relocations_per_container": "relocations divided by initial container count"},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_STRICT_ENDPOINT_VALIDATION", "input_rows": len(rows),
                      "aggregate_rows": len(grouped), "output": str(args.output),
                      "scope": "descriptive means only; bootstrap intervals are produced by the registered report script"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
