"""Exercise every registered UWSAA arm on one complete small trajectory."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.online_crp import load_episode_manifest
from reproducibility.protocol import CONDITION_SETS, SUITE_ARMS, forecast_table
from reproducibility.runner import DEFAULT_DATA, load_gp_rule, load_runtime, run_episode


def main() -> None:
    episode = load_episode_manifest(DEFAULT_DATA)[0]
    forecast_count = 0
    for condition in CONDITION_SETS["main"]:
        table, digest = forecast_table(episode, condition)
        if len(table) != episode.num_containers or table[-1] != () or len(digest) != 64:
            raise AssertionError(f"invalid registered forecast table: {condition['condition_id']}")
        forecast_count += 1

    runtime, gp_rule = load_runtime(), load_gp_rule()
    policy_count = 0
    for suite, arms in SUITE_ARMS.items():
        condition = CONDITION_SETS[suite][-1]
        for solver in ("EM", "EG"):
            for arm in arms:
                result = run_episode(episode, condition, solver, arm, runtime, gp_rule)
                if not result["complete"]:
                    raise AssertionError(
                        f"{suite}/{solver}/{arm}: {result.get('failure_type')}: "
                        f"{result.get('failure_message')}"
                    )
                if result["num_retrieved"] != episode.num_containers:
                    raise AssertionError("completed trajectory did not retrieve every container")
                if result["provider_reveal_count"] != episode.num_containers - 1:
                    raise AssertionError("forecast provider reveal count is incomplete")
                policy_count += 1
    print(json.dumps({
        "status": "PASS",
        "forecast_tables_checked": forecast_count,
        "policy_trajectories_checked": policy_count,
        "scope": "one small complete episode per registered suite arm for EM and EG; no performance measurement",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
