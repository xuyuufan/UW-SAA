from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_ROOTS = {"R", "Mijn_RL", "EM", "GP", "Chain", "Minmax", "Leveling"}


def main() -> None:
    python_files = sorted(ROOT.rglob("*.py"))
    parsed = 0
    forbidden_imports: list[str] = []
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        parsed += 1
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                    forbidden_imports.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
    json_files = sorted(ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    parity = json.loads((ROOT / "checks/semantic_parity.json").read_text(encoding="utf-8"))
    if parity.get("status") != "PASS" or parity.get("comparison_fields") != {
        "horizon_proposals": 30,
        "solver_method_actions_and_traces": 12,
        "eg_same_group_second_move_and_reset": True,
    }:
        raise AssertionError("semantic parity evidence is missing required comparison fields")
    if forbidden_imports:
        raise AssertionError(f"source or legacy imports found: {forbidden_imports}")
    bytecode = list(ROOT.rglob("*.pyc"))
    if bytecode:
        raise AssertionError(f"compiled bytecode found: {bytecode}")
    result = {
        "status": "PASS",
        "python_ast_files": parsed,
        "json_files_parsed": len(json_files),
        "forbidden_imports": forbidden_imports,
        "compiled_bytecode_files": len(bytecode),
        "semantic_parity_artifact": "checks/semantic_parity.json",
        "semantic_parity_fields": parity["comparison_fields"],
        "scope": "static parse/import audit and retained CPU semantic-check evidence; no experiment or training",
    }
    target = ROOT / "checks/verification.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
