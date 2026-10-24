"""Build a SHA-256 inventory from an explicit repository worktree file list."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = "manifest.json"
PUBLIC_MANIFEST = "reproducibility/public_bundle_manifest.json"
DEFAULT_PUBLISHED_FILES = ROOT / "reproducibility" / "published_files.json"
EXCLUDED_EXACT = {PUBLIC_MANIFEST}
EXCLUDED_PREFIXES = (".git/", "local-archive/", "reproducibility/results/")
EXCLUDED_PARTS = {"__pycache__", ".venv", "venv", "node_modules"}
RAW_PREFIXES = ("third_party/crp_online/",)
RAW_EXACT = {
    "experiments/gp_h20/data/train_bank.json",
    "artifacts/gp_h20/20261024_formal_30seed/acceleration_differential_verification.json",
    "artifacts/gp_h20/20261024_formal_30seed/pretraining_migration_verification.json",
}
BINARY_SUFFIXES = (".npz", ".gz", ".png", ".pdf", ".zip")


def _is_binary_path(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(suffix) for suffix in BINARY_SUFFIXES) or ".gz." in lowered
PROVENANCE_INDEX = {
    "frozen_core_source_and_adaptation_map": "manifest.json",
    "endpoint_source_rows_statuses_and_field_map": "reproducibility/data_inventory.json",
    "result_migration_status": "results/uwsaa-benchmark-evaluation/README.md",
    "gp_pretraining_migration_verification": "checks/gp_migration_verification.json",
    "gp_trained_source_yard_verification": "checks/gp_trained_source_yard_verification.json",
    "gp_trained_runner_smoke_verification": "checks/gp_trained_runner_smoke_verification.json",
    "gp_migration_smoke": "checks/gp_migration_smoke.py",
    "gp_acceleration_differential": "checks/gp_acceleration_differential_result.json",
    "gp_acceleration_differential_check": "checks/gp_acceleration_differential.py",
    "gp_only_paired_analysis": "reproducibility/gp_arm_paired_analysis.py",
    "gp_training_environment": "experiments/gp_h20/requirements-training.txt",
    "registered_trajectory_and_statistics_protocol": "reproducibility/PROTOCOL.md",
    "cloud_gp_caserta_evaluation_prompt": "reproducibility/CLOUD_GP_MIGRATION_PROMPT.md",
    "gp_main_partition_runner": "reproducibility/gp_caserta_partition_runner.py",
    "gp_crosshost_probe_reference": "reproducibility/references/gp_caserta_crosshost_probe_reference.json",
}


def _include(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in EXCLUDED_EXACT or normalized.startswith(EXCLUDED_PREFIXES):
        return False
    return not any(part in EXCLUDED_PARTS for part in normalized.split("/"))


def _safe_path(path: str) -> Path:
    normalized = path.replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts or not normalized:
        raise ValueError(f"unsafe published path: {path!r}")
    resolved = (ROOT / relative).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"published path escapes repository: {path!r}")
    return resolved


def _published_paths(list_path: Path) -> list[str]:
    resolved_list = list_path.resolve()
    if not resolved_list.is_relative_to(ROOT):
        raise ValueError("published-files list must be inside the repository")
    payload = json.loads(resolved_list.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "uwsaa_published_files_v1" or not isinstance(payload.get("files"), list):
        raise ValueError("published-files list has an unknown schema")
    paths = [str(value).replace("\\", "/") for value in payload["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("published-files list contains duplicate paths")
    excluded = [path for path in paths if not _include(path)]
    if excluded:
        raise ValueError(f"published-files list includes excluded files: {excluded}")
    for path in paths:
        if not _safe_path(path).is_file():
            raise FileNotFoundError(f"published file is missing: {path}")
    return sorted(paths)


def _worktree_bytes(path: str) -> bytes:
    data = _safe_path(path).read_bytes()
    if path in RAW_EXACT or path.startswith(RAW_PREFIXES) or _is_binary_path(path):
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _file_record(path: str) -> dict[str, Any]:
    data = _worktree_bytes(path)
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def refresh_core_manifest(published_files: Path) -> None:
    """Refresh provenance destination hashes from normalized worktree bytes."""
    paths = set(_published_paths(published_files))
    manifest_path = _safe_path(CORE_MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for entry in manifest.get("entries", []):
        destination = str(entry.get("destination_path", ""))
        if not destination or destination not in paths:
            raise ValueError(f"core manifest destination is not in published_files: {destination}")
        entry["final_sha256"] = _file_record(destination)["sha256"]
    manifest["included_file_count"] = len(manifest.get("entries", []))
    _write_json(manifest_path, manifest)


def build_manifest(published_files: Path) -> dict[str, Any]:
    paths = _published_paths(published_files)
    missing_sources = sorted(set(PROVENANCE_INDEX.values()) - set(paths))
    if missing_sources:
        raise ValueError(f"provenance index targets are absent from published_files: {missing_sources}")
    records = [_file_record(path) for path in paths if path != PUBLIC_MANIFEST]
    return {
        "schema": "uwsaa_public_bundle_manifest_v1",
        "hash_algorithm": "SHA-256",
        "hash_basis": "explicit worktree file list; text normalized to LF per .gitattributes; vendored source, training bank, and declared binary files byte-preserved",
        "scope": "explicit published_files list only; generated training runs, endpoint outputs, local archives, and unrelated untracked files are excluded",
        "published_files_list": "reproducibility/published_files.json",
        "provenance_index": PROVENANCE_INDEX,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "max_file": max(records, key=lambda record: record["bytes"]) if records else None,
        "files": records,
    }


def verify_manifest(published_files: Path) -> dict[str, Any]:
    stored = json.loads(_safe_path(PUBLIC_MANIFEST).read_text(encoding="utf-8-sig"))
    expected = build_manifest(published_files)
    if stored != expected:
        expected_by_path = {row["path"]: row for row in expected["files"]}
        stored_by_path = {row["path"]: row for row in stored.get("files", [])}
        missing = sorted(set(expected_by_path) - set(stored_by_path))
        extra = sorted(set(stored_by_path) - set(expected_by_path))
        changed = sorted(path for path in set(expected_by_path) & set(stored_by_path)
                         if expected_by_path[path] != stored_by_path[path])
        raise ValueError(f"public manifest differs from worktree file list: missing={missing}, extra={extra}, changed={changed}")
    return {"status": "PASS", "file_count": stored["file_count"],
            "total_bytes": stored["total_bytes"], "max_file": stored["max_file"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-files", type=Path, default=DEFAULT_PUBLISHED_FILES,
                        help="explicit JSON inventory; unrelated untracked files are never scanned")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--refresh-core", action="store_true", help="refresh core manifest hashes from normalized worktree bytes")
    actions.add_argument("--write", action="store_true", help="write the complete manifest for the explicit worktree list")
    actions.add_argument("--verify", action="store_true", help="verify manifest against explicit worktree files")
    args = parser.parse_args()
    if args.refresh_core:
        refresh_core_manifest(args.published_files)
        print(json.dumps({"status": "UPDATED", "manifest": CORE_MANIFEST}))
    elif args.write:
        manifest = build_manifest(args.published_files)
        _write_json(_safe_path(PUBLIC_MANIFEST), manifest)
        print(json.dumps({"status": "WRITTEN", "file_count": manifest["file_count"],
                          "total_bytes": manifest["total_bytes"], "max_file": manifest["max_file"]}))
    else:
        print(json.dumps(verify_manifest(args.published_files), ensure_ascii=False))


if __name__ == "__main__":
    main()
