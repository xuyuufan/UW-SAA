"""Verify the explicit public bundle inventory and current migration status."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from checks import build_public_manifest as public

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_ROOTS = {"R", "Mijn_RL", "EM", "GP", "Chain", "Minmax", "Leveling"}
OBSOLETE_PATHS = {"baseline/gp_expression.py", "configs/gp_accepted_rule.json"}


def main() -> None:
    list_path = ROOT / "reproducibility" / "published_files.json"
    paths = public._published_paths(list_path)
    python_paths = [path for path in paths if path.endswith(".py")]
    json_paths = [path for path in paths if path.endswith(".json")]
    forbidden_imports: list[str] = []
    for relative in python_paths:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    forbidden_imports.append(f"{relative}:{node.lineno}:{name}")
    for relative in json_paths:
        json.loads((ROOT / relative).read_text(encoding="utf-8"))

    listed = set(paths)
    if listed & OBSOLETE_PATHS:
        raise AssertionError(f"retired GP source/config remains published: {sorted(listed & OBSOLETE_PATHS)}")
    if forbidden_imports:
        raise AssertionError(f"legacy imports found in the explicit public inventory: {forbidden_imports}")

    core = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    migration = core.get("gp_migration", {})
    rule = json.loads((ROOT / "configs" / "gp_rule.json").read_text(encoding="utf-8"))
    status = rule.get("status")
    if status not in {"pending", "trained"} or migration.get("rule_status") != status:
        raise AssertionError("core manifest and GP rule status do not agree")
    if core.get("source_hash_verification", {}).get("status", "").startswith("PASS"):
        raise AssertionError("the migration-pending package cannot claim a whole-bundle PASS")
    if status == "pending" and (rule.get("formula") is not None or rule.get("model_identity") is not None):
        raise AssertionError("pending GP config unexpectedly contains a selected model")
    for entry in core.get("entries", []):
        destination = entry["destination_path"]
        if destination not in listed:
            raise AssertionError(f"core provenance destination is outside published_files: {destination}")
        expected = public._file_record(destination)["sha256"]
        if entry.get("final_sha256") != expected:
            raise AssertionError(f"core manifest final hash is stale: {destination}")

    public_status = public.verify_manifest(list_path)
    print(json.dumps({
        "status": "PASS_BUNDLE_PENDING_GP" if status == "pending" else "PASS_BUNDLE_TRAINED_GP",
        "python_files_parsed": len(python_paths), "json_files_parsed": len(json_paths),
        "published_files": len(paths), "retired_gp_paths_published": [],
        "gp_rule_status": status, "public_manifest": public_status,
        "scope": "explicit inventory, syntax/JSON, hash binding and migration status; no performance claim",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
