from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retained source-vs-clean parity evidence.")
    parser.add_argument("--source-root", type=Path, help="optional source checkout root for provenance hash checks")
    args = parser.parse_args()
    record = json.loads((ROOT / "checks/semantic_parity.json").read_text(encoding="utf-8"))
    expected = {
        "horizon_proposals": 30,
        "solver_method_actions_and_traces": 12,
        "eg_same_group_second_move_and_reset": True,
    }
    if record.get("status") != "PASS" or record.get("comparison_fields") != expected:
        raise SystemExit("semantic parity summary is incomplete or failed")
    checks = record.get("checks", {})
    if checks.get("proposals_equal") is not True or checks.get("methods_equal") is not True or checks.get("eg_group_equal") is not True:
        raise SystemExit("a recorded source-vs-clean equality check is false or missing")
    verified = 0
    if args.source_root:
        source_root = args.source_root.resolve()
        for anchor in record["source_anchors"]:
            source = (source_root / anchor["path"]).resolve()
            if not source.is_relative_to(source_root) or not source.is_file() or sha256(source) != anchor["sha256"]:
                raise SystemExit(f"source anchor mismatch: {anchor['path']}")
            verified += 1
    print(json.dumps({
        "status": "PASS",
        "record": "checks/semantic_parity.json",
        "comparison_fields": record["comparison_fields"],
        "source_anchors_checked": verified,
        "note": "validates retained comparison evidence; does not rerun the source differential",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
