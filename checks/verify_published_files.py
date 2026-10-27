"""Verify the explicit publication inventory and core destination hashes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PREFIXES = ("third_party/crp_online/", "results/gp-caserta-main-20261024/")
RAW_EXACT = {
    "experiments/gp_h20/data/train_bank.json",
    "artifacts/gp_h20/20261024_formal_30seed/acceleration_differential_verification.json",
    "artifacts/gp_h20/20261024_formal_30seed/pretraining_migration_verification.json",
}


def publication_bytes(name: str) -> bytes:
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"invalid or missing published path: {name}")
    data = path.read_bytes()
    binary = path.suffix in {".npz", ".gz", ".png", ".pdf", ".zip"} or ".gz." in name
    if name in RAW_EXACT or name.startswith(RAW_PREFIXES) or binary:
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def main() -> None:
    inventory = json.loads((ROOT / "reproducibility/published_files.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "reproducibility/public_bundle_manifest.json").read_text(encoding="utf-8"))
    paths = inventory["files"]
    records = {item["path"]: item for item in manifest["files"]}
    if len(paths) != len(set(paths)) or len(records) != len(manifest["files"]) or set(paths) != set(records):
        raise ValueError("publication inventory has duplicate or mismatched paths")
    for name in paths:
        data = publication_bytes(name)
        if len(data) != records[name]["bytes"] or hashlib.sha256(data).hexdigest() != records[name]["sha256"]:
            raise ValueError(f"published bytes differ from manifest: {name}")
        if name.endswith(".json"):
            json.loads(data.decode("utf-8"))
        elif name.endswith(".py"):
            ast.parse(data.decode("utf-8"), filename=name)
    for name in manifest["provenance_index"].values():
        if name not in records:
            raise ValueError(f"unpublished provenance target: {name}")
    core = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for item in core["entries"]:
        name = item["destination_path"]
        if name not in records or item["final_sha256"] != records[name]["sha256"]:
            raise ValueError(f"stale core destination hash: {name}")
    if manifest["file_count"] != len(records) or manifest["total_bytes"] != sum(item["bytes"] for item in records.values()):
        raise ValueError("publication count or total size mismatch")
    print(json.dumps({"status": "PASS_PUBLISHED_FILES", "files_checked": len(records),
                      "total_bytes": manifest["total_bytes"],
                      "scope": "explicit file inventory, SHA-256, core provenance, syntax and JSON"}))


if __name__ == "__main__":
    main()
