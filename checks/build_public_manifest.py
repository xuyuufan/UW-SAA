"""Build and verify a SHA-256 inventory from the staged public Git files."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = "manifest.json"
PUBLIC_MANIFEST = "reproducibility/public_bundle_manifest.json"
EXCLUDED_EXACT = {PUBLIC_MANIFEST}
EXCLUDED_PREFIXES = (
    ".git/", "local-archive/", "reproducibility/results/",
)
EXCLUDED_PARTS = {"__pycache__", ".venv", "venv", "node_modules"}
PROVENANCE_INDEX = {
    "frozen_core_source_and_adaptation_map": "manifest.json",
    "endpoint_source_rows_statuses_and_field_map": "reproducibility/data_inventory.json",
    "published_result_interpretation_and_replacements": "results/uwsaa-benchmark-evaluation/README.md",
    "runner_policy_source_differential": "checks/source_differential_smoke_result.json",
    "statistics_estimator_source_parity": "checks/statistics_source_parity_result.json",
    "accepted_statistics_recomputation": "checks/accepted_statistics_verification.json",
    "accepted_statistics_recomputation_script": "checks/verify_accepted_statistics.py",
    "registered_trajectory_and_statistics_protocol": "reproducibility/PROTOCOL.md",
}


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _staged_paths() -> list[str]:
    values = _git_bytes("ls-files", "-z").split(b"\0")
    paths = [value.decode("utf-8") for value in values if value]
    return sorted(path for path in paths if _include(path))


def _include(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in EXCLUDED_EXACT or normalized.startswith(EXCLUDED_PREFIXES):
        return False
    return not any(part in EXCLUDED_PARTS for part in normalized.split("/"))


def _blob(path: str) -> bytes:
    return _git_bytes("show", f":{path}")


def _file_record(path: str) -> dict[str, Any]:
    data = _blob(path)
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def refresh_core_manifest() -> None:
    """Update old destination naming and core file hashes from staged blobs."""
    path = ROOT / CORE_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest.get("entries", []):
        if entry.get("destination_path") == "baseline/eri_eg_plan059.py":
            entry["destination_path"] = "baseline/eri_eg.py"
        destination = str(entry.get("destination_path", ""))
        if not destination or destination not in _staged_paths():
            raise ValueError(f"core manifest destination is not staged: {destination}")
        entry["final_sha256"] = hashlib.sha256(_blob(destination)).hexdigest()
    _write_json(path, manifest)


def build_manifest() -> dict[str, Any]:
    paths = _staged_paths()
    missing_sources = sorted(set(PROVENANCE_INDEX.values()) - set(paths))
    if missing_sources:
        raise ValueError(f"provenance index targets are not staged: {missing_sources}")
    records = [_file_record(path) for path in paths]
    return {
        "schema": "uwsaa_public_bundle_manifest_v1",
        "hash_algorithm": "SHA-256",
        "hash_basis": "staged Git blob bytes after the repository's declared text normalization",
        "scope": "all staged public distribution files; generated endpoint-run outputs, local archives, and caches are excluded",
        "provenance_index": PROVENANCE_INDEX,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "max_file": max(records, key=lambda record: record["bytes"]) if records else None,
        "files": records,
    }


def verify_manifest() -> dict[str, Any]:
    stored = json.loads(_blob(PUBLIC_MANIFEST).decode("utf-8"))
    expected = build_manifest()
    if stored != expected:
        expected_by_path = {row["path"]: row for row in expected["files"]}
        stored_by_path = {row["path"]: row for row in stored.get("files", [])}
        missing = sorted(set(expected_by_path) - set(stored_by_path))
        extra = sorted(set(stored_by_path) - set(expected_by_path))
        changed = sorted(path for path in set(expected_by_path) & set(stored_by_path)
                         if expected_by_path[path] != stored_by_path[path])
        raise ValueError(f"public manifest differs from Git index: missing={missing}, extra={extra}, changed={changed}")
    return {"status": "PASS", "file_count": stored["file_count"],
            "total_bytes": stored["total_bytes"], "max_file": stored["max_file"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--refresh-core", action="store_true", help="rewrite core manifest hashes from staged blobs")
    actions.add_argument("--write", action="store_true", help="write the complete public bundle manifest")
    actions.add_argument("--verify", action="store_true", help="verify the staged manifest against Git blobs")
    args = parser.parse_args()
    if args.refresh_core:
        refresh_core_manifest()
        print(json.dumps({"status": "UPDATED", "manifest": CORE_MANIFEST}))
    elif args.write:
        manifest = build_manifest()
        _write_json(ROOT / PUBLIC_MANIFEST, manifest)
        print(json.dumps({"status": "WRITTEN", "file_count": manifest["file_count"],
                          "total_bytes": manifest["total_bytes"], "max_file": manifest["max_file"]}))
    else:
        print(json.dumps(verify_manifest(), ensure_ascii=False))


if __name__ == "__main__":
    main()
