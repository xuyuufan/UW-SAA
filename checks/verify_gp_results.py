"""Verify the published GP archive and reproduce its registered paired statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility import gp_arm_paired_analysis as analysis
from reproducibility.gp_caserta_partition_runner import runtime_inventory

RESULTS = ROOT / "results" / "gp-caserta-main-20261024"
ARCHIVE_SHA256 = "5c2d76ab977dcd3e50853042a5d9452136ad7337266ecc63f1226710c4eb09a4"


def checksums(text: str) -> dict[str, str]:
    records = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"invalid SHA-256 record: {line!r}")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name in records:
            raise ValueError(f"unsafe or duplicate checksum path: {name!r}")
        records[name] = digest
    return records


def verify_files(directory: Path) -> int:
    expected = checksums((directory / "SHA256SUMS.txt").read_text(encoding="utf-8"))
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(expected) | {"SHA256SUMS.txt"}:
        raise ValueError("result directory and checksum inventory differ")
    for name, digest in expected.items():
        if analysis.sha_file(directory / name) != digest:
            raise ValueError(f"published file hash mismatch: {name}")
    return len(expected)


def extract_verified_archive(directory: Path) -> tuple[list[Path], int]:
    archive = RESULTS / "gp_caserta_full_results.zip"
    if analysis.sha_file(archive) != ARCHIVE_SHA256:
        raise ValueError("raw GP result archive differs from the pinned completed run")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate members")
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16) or item.is_dir()):
                raise ValueError(f"unsafe archive member: {item.filename}")
        expected = checksums(z.read("SHA256SUMS.txt").decode("utf-8"))
        if set(names) != set(expected) | {"SHA256SUMS.txt"}:
            raise ValueError("archive members and checksum inventory differ")
        for name, digest in expected.items():
            payload = z.read(name)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"archive member hash mismatch: {name}")
            # Scripts and historical report copies inside the archive are evidence,
            # not executable inputs. Only formal endpoint shards are extracted.
            if name.startswith(("local/", "cloud/")):
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        probe = json.loads(z.read("integrity/probe_diagnostic_only/probe_receipt.json"))
        if probe.get("status") != "pass" or probe.get("failures"):
            raise ValueError("cross-host probe did not complete successfully")
        inventory = runtime_inventory()
        for partition, indexes in (("local", list(range(1, 15))), ("cloud", list(range(15, 42)))):
            manifest = json.loads((directory / partition / "run_manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("status") != "complete" or manifest.get("failures")
                    or manifest.get("active_children") or manifest.get("condition_indexes") != indexes
                    or manifest.get("completed_endpoint_rows") != len(indexes) * 1680
                    or manifest.get("runtime_inventory") != inventory
                    or manifest.get("partition_runner_sha256") != analysis.sha_file(
                        ROOT / "reproducibility" / "gp_caserta_partition_runner.py")):
                raise ValueError(f"{partition} completion or frozen runtime binding mismatch")
    endpoints = sorted(directory.glob("local/condition-*.jsonl"))
    endpoints += sorted(directory.glob("cloud/condition-*.jsonl"))
    if len(endpoints) != 41:
        raise ValueError("archive must contain exactly 41 formal condition shards")
    return endpoints, len(expected)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(output: Path) -> dict:
    published_count = verify_files(RESULTS)
    frozen = json.loads((RESULTS / "analysis_manifest.json").read_text(encoding="utf-8"))
    if frozen["analysis_script_sha256"] != analysis.sha_file(Path(analysis.__file__)):
        raise ValueError("registered GP analyzer has changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory creates a unique child and removes the extracted ~300 MB
    # after verification. The receipt is the only persistent verification output.
    with tempfile.TemporaryDirectory(prefix="gp-results-", dir=output.parent) as temp:
        scratch = Path(temp)
        endpoints, archived_count = extract_verified_archive(scratch)
        rebuilt = analysis.execute(argparse.Namespace(
            endpoint=endpoints, data=analysis.DEFAULT_DATA, rule=analysis.DEFAULT_RULE,
            output_dir=scratch / "analysis",
        ))
        if rebuilt != frozen:
            raise ValueError("recomputed analysis manifest differs from the published run")
        for name in ("family_effects.csv", "condition_effects.csv"):
            if analysis.sha_file(scratch / "analysis" / name) != analysis.sha_file(RESULTS / name):
                raise ValueError(f"recomputed table differs from published {name}")
        family = csv_rows(scratch / "analysis" / "family_effects.csv")
        conditions = csv_rows(scratch / "analysis" / "condition_effects.csv")
        expected_summary = []
        for metric in analysis.METRICS:
            expected_summary.extend(row for row in family if row["metric"] == metric)
            expected_summary.extend(row for row in conditions
                                    if row["metric"] == metric and row["family"] == "CLEAN")
        summary = csv_rows(RESULTS / "summary.csv")
        if len(summary) != len(expected_summary):
            raise ValueError("summary row count differs from registered tables")
        for actual, expected in zip(summary, expected_summary, strict=True):
            if any(actual.get(key, "") != expected.get(key, "") for key in actual.keys() | expected.keys()):
                raise ValueError("summary differs from registered family/CLEAN rows")
    receipt = {
        "schema": "uwsaa_gp_published_results_verification_v1",
        "status": "PASS", "scope": "artifact integrity and exact numerical reproduction; not independent scientific review",
        "published_files_checked": published_count, "archive_members_checked": archived_count,
        "formal_endpoint_rows": 68880, "formal_condition_shards": 41,
        "paired_cells": 34440, "geometries": 21, "layouts": 840,
        "bootstrap_draws": analysis.DRAWS, "bootstrap_seed": analysis.SEED,
        "recomputed_table_rows": len(family) + len(conditions), "summary_rows_checked": len(summary),
        "tables_and_analysis_manifest_match_exactly": True,
        "archive_sha256": ARCHIVE_SHA256,
        "analysis_script_sha256": frozen["analysis_script_sha256"],
        "gp_model_identity_sha256": frozen["gp_model_identity_sha256"],
        "evidence_status": frozen["evidence_status"],
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "gp_results_verification.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
