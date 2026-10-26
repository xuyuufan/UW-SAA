from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment
from uwsaa.entrypoint import load_runtime


def main() -> None:
    episode = OnlineCRPEpisode(
        episode_id="cpu-smoke",
        num_stacks=3,
        max_tiers=3,
        initial_stacks=((1, 2), (3,), (4,)),
        retrieval_sequence=(1, 3, 2, 4),
        parameter_group="smoke",
        metadata={},
    )
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=1)
    legal = environment.legal_destinations()
    state, reward, terminated = environment.step(legal[0])

    # Import the actual policy runtime without adding source or legacy roots.
    load_runtime()
    targets = [
        module
        for name, module in sys.modules.items()
        if name.startswith(("uwsaa", "baseline", "environment"))
        and getattr(module, "__file__", None)
    ]
    outside_root = [
        str(Path(module.__file__).resolve())
        for module in targets
        if not Path(module.__file__).resolve().is_relative_to(ROOT)
    ]
    legacy_modules = [
        name
        for name in ("R", "scrp", "Mijn_RL", "EM", "GP", "Chain", "Minmax", "Leveling")
        if name in sys.modules
    ]
    if outside_root or legacy_modules:
        raise AssertionError({"outside_root": outside_root, "legacy_modules": legacy_modules})
    if state.relocation_count != 1 or reward != -1 or terminated:
        raise AssertionError("one-step scalar environment transition did not match its contract")

    payload = {
        "check": "isolated_import_and_online_crp_step",
        "status": "PASS",
        "initial_legal_destinations": list(legal),
        "relocation_count_after_step": state.relocation_count,
        "reward": reward,
        "terminated": terminated,
        "target_module_count": len(targets),
        "outside_bundle_modules": outside_root,
        "legacy_or_source_modules_loaded": legacy_modules,
        "gp_status": "pending; not loaded by this non-GP environment smoke",
        "scope": "one scalar environment step and non-GP import-path audit; no experiment or trajectory",
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
