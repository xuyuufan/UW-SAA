"""Run one synthetic public-frame decision for each available non-GP policy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.noise_aware_forecast import PublicObservation
from environment.public_frame import PublicDecisionFrame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from uwsaa.entrypoint import SOLVERS, load_runtime, make_policy


def main() -> None:
    rows = (
        (1, 21, 22, 22, 0), (2, 3, 4, 5, 0), (6, 7, 8, 9, 0),
        (10, 11, 12, 13, 0), (14, 15, 16, 0, 0),
        (17, 18, 19, 0, 0), (20, 22, 22, 0, 0),
    )
    ranks = np.asarray(rows, dtype=np.int32)
    observation = PublicObservation(
        rank_tensor=ranks, occupied_mask=ranks != 0,
        legal_destination_mask=np.asarray((False, True, True, True, True, True, True), dtype=np.bool_),
        history_hits=(), q_prior=5183 / 9920, q_t=5183 / 9920,
    )
    frame = PublicDecisionFrame(
        observation=observation, legal_destinations=(1, 2, 3, 4, 5, 6),
        max_tiers=5, crane_position=0, crane_time=0.0, retrieval_count=0, relocation_count=0,
    )
    frame = attach_history_summary(frame, PublicHistoryTracker().summary(frame))
    runtime = load_runtime()
    solvers = tuple(solver for solver in SOLVERS if solver != "GP")
    results = {}
    for solver in solvers:
        results[solver] = {}
        for method in ("UWSAA", "NATIVE20"):
            action, trace = make_policy(solver, method, runtime).select(frame)
            if action not in frame.legal_destinations:
                raise AssertionError((solver, method, action))
            results[solver][method] = {"action": action, "trace": trace}
    payload = {
        "check": "cpu_public_one_frame_non_gp_smoke", "status": "PASS",
        "fixture": "synthetic-public-25-item-one-frame", "solvers": list(solvers),
        "methods": ["UWSAA", "NATIVE20"], "gp_status": "pending; GP deliberately skipped",
        "scope": "one decision per non-GP solver/method; no training, trajectory, or performance evaluation",
        "results": results,
    }
    print(json.dumps(payload, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
