"""Exercise strict endpoint rejection cases without running trajectories."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.aggregate import _validate
from reproducibility.runner import _semantic_key


def _fixture() -> tuple[list[dict], dict]:
    episode = {
        "episode_id": "fixture-1", "layout_fingerprint": "f" * 64,
        "parameter_group": "3-3", "num_containers": 4,
    }
    condition = {
        "condition_id": "CLEAN", "family": "CLEAN", "p": None,
        "window_size": None, "repeat": 0, "profile": None,
    }
    arms = ("UWSAA", "NATIVE20")
    rows = []
    for arm, relocations in zip(arms, (2, 3), strict=True):
        rows.append({
            "schema": "uwsaa_trajectory_endpoint_v1", "complete": True,
            "episode_id": episode["episode_id"], "layout_fingerprint": episode["layout_fingerprint"],
            "parameter_group": episode["parameter_group"], "condition_id": condition["condition_id"],
            "family": "CLEAN", "repeat": 0, "solver": "EM", "arm": arm,
            "semantic_key": _semantic_key(episode["episode_id"], episode["layout_fingerprint"], condition, "EM", arm),
            "num_containers": 4, "num_retrieved": 4, "provider_reveal_count": 3,
            "relocations": relocations, "relocations_per_container": relocations / 4,
            "forecast_table_sha256": "a" * 64,
        })
    manifest = {
        "schema": "uwsaa_trajectory_run_manifest_v1", "status": "complete", "failed_rows": 0,
        "expected_rows": 2, "actual_rows": 2, "episodes": [episode],
        "conditions": [condition], "solvers": ["EM"], "arms": list(arms),
    }
    return rows, manifest


def _must_reject(rows: list[dict], manifest: dict, label: str) -> None:
    try:
        _validate(rows, manifest)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"strict aggregate validation accepted {label}")


def main() -> None:
    rows, manifest = _fixture()
    _validate(rows, manifest)

    _must_reject(rows[:1], manifest, "an incomplete matrix")
    duplicate = deepcopy(rows + [rows[0]])
    duplicate_manifest = deepcopy(manifest)
    duplicate_manifest.update(expected_rows=3, actual_rows=3)
    _must_reject(duplicate, duplicate_manifest, "a duplicate semantic key")

    mislabeled = deepcopy(rows)
    mislabeled[0]["arm"] = "NATIVE20"
    _must_reject(mislabeled, manifest, "a mislabeled arm with a stale semantic key")

    mismatched_forecast = deepcopy(rows)
    mismatched_forecast[1]["forecast_table_sha256"] = "b" * 64
    _must_reject(mismatched_forecast, manifest, "paired arms with different forecast hashes")

    mismatched_metadata = deepcopy(rows)
    mismatched_metadata[0]["parameter_group"] = "4-4"
    _must_reject(mismatched_metadata, manifest, "episode metadata inconsistent with the manifest")
    print("PASS: complete, incomplete, duplicate, mislabeled, forecast, and metadata cases")


if __name__ == "__main__":
    main()
