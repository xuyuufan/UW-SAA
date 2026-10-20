"""CPU-only one-frame validation of the copied public policies and baselines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from EM.scrp.noise_aware_forecast import PublicObservation
from R.experiments.uwsaa_entrypoint import SOLVERS, load_gp_rule, load_runtime, make_policy
from R.scrp.r_order_public import PublicHistoryTracker, attach_history_summary


@dataclass(frozen=True)
class PublicFrame:
    observation: object
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int = 0
    relocation_count: int = 0
    crane_position: int | None = 0
    crane_time: float = 0.0


def make_fixture() -> PublicFrame:
    """Return one valid public frame: target + ranks 2..21 + four anonymous tails."""
    rows = (
        (1, 21, 22, 22, 0),
        (2, 3, 4, 5, 0),
        (6, 7, 8, 9, 0),
        (10, 11, 12, 13, 0),
        (14, 15, 16, 0, 0),
        (17, 18, 19, 0, 0),
        (20, 22, 22, 0, 0),
    )
    ranks = np.asarray(rows, dtype=np.int32)
    occupied = ranks != 0
    legal = np.asarray((False, True, True, True, True, True, True), dtype=np.bool_)
    observation = PublicObservation(
        rank_tensor=ranks,
        occupied_mask=occupied,
        legal_destination_mask=legal,
        history_hits=(),
        q_prior=5183 / 9920,
        q_t=5183 / 9920,
    )
    frame = PublicFrame(
        observation=observation,
        legal_destinations=(1, 2, 3, 4, 5, 6),
        max_tiers=5,
    )
    summary = PublicHistoryTracker().summary(frame)
    return attach_history_summary(frame, summary)


def main() -> int:
    frame = make_fixture()
    runtime = load_runtime()
    gp_rule = load_gp_rule()
    results: dict[str, dict[str, object]] = {}
    methods = ("UWSAA", "NATIVE20")
    for solver in SOLVERS:
        results[solver] = {}
        for method in methods:
            action, trace = make_policy(solver, method, runtime, gp_rule).select(frame)
            if action not in frame.legal_destinations:
                raise AssertionError(f"{solver}/{method} returned illegal destination {action}")
            results[solver][method] = {"action": action, "trace": trace}
    payload = {
        "check": "cpu_public_one_frame_smoke",
        "status": "PASS",
        "fixture": "public-25-item-one-frame",
        "solvers": list(SOLVERS),
        "methods": list(methods),
        "scope": "one decision per solver/method; no training, trajectory, or performance evaluation",
        "results": results,
    }
    result_path = ROOT / "checks" / "public_smoke_result.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
