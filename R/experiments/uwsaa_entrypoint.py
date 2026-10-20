"""Local UWSAA and Native20 policy entry points for public decision frames."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from R.scrp.uwsaa_fast import UWSAAFastSelector

WORKSPACE = Path(__file__).resolve().parents[2]
SOLVERS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")


def _install_package_roots() -> None:
    roots = (WORKSPACE / "R", WORKSPACE / "EM", WORKSPACE / "Mijn_RL", WORKSPACE)
    for root in reversed(roots):
        value = str(root.resolve())
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    scrp = importlib.import_module("scrp")
    locations = [str((root / "scrp").resolve()) for root in roots[:3]]
    scrp.__path__ = locations
    if scrp.__spec__ is not None and scrp.__spec__.submodule_search_locations is not None:
        scrp.__spec__.submodule_search_locations[:] = locations


def load_runtime() -> dict[str, Any]:
    """Build the same public actor capability slots used by the frozen runner."""
    _install_package_roots()
    em = importlib.import_module("scrp.noise_aware_em")
    eri = importlib.import_module("R.scrp.eri_eg_public")
    cross = importlib.import_module("R.scrp.cross_solver_reliability_r")
    corrected_eg = importlib.import_module("R.scrp.eri_eg_plan059")
    runtime = {
        "truncate_public_frame": cross.truncate_public_frame,
        "em_select": em.select_em_destination,
        "eri_select": eri.select_eri20_destination,
        "eri_eg": eri,
    }
    return corrected_eg.runtime_with_corrected_eg(runtime)


def load_gp_rule() -> dict[str, Any]:
    """Compile the copied, already accepted public GP selector expression."""
    payload = json.loads((WORKSPACE / "models/gp_accepted_rule.json").read_text(encoding="utf-8"))
    from GP.expression import Node
    from GP.simulator import select_destination
    tree = Node.from_prefix(tuple(int(value) for value in payload["selected_best_prefix"]))
    if tree.expression() != payload["selected_best_expression"]:
        raise ValueError("copied GP prefix does not match its accepted expression")

    def selector(state: Any, legal_destinations: Any, *, max_tiers: int) -> int:
        return int(select_destination(tree, state, legal_destinations, max_tiers=max_tiers))

    return {"selector": selector, "tree": tree, "identity": payload["model_identity"]}


class _SingleStepPolicy:
    def __init__(self, solver: str, proposer: Any) -> None:
        self.solver, self.proposer = solver, proposer

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        action, trace = self.proposer(frame, 20)
        return int(action), {"policy": "NATIVE20", "solver": self.solver, **dict(trace)}


def make_policy(solver: str, method: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]) -> Any:
    """Create one registered policy: final UWSAA or its Native20 baseline."""
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if method == "UWSAA":
        return UWSAAFastSelector(solver, runtime, gp_rule)
    if method != "NATIVE20":
        raise ValueError("method must be UWSAA or NATIVE20")
    if solver in ("EM", "GP", "ERI", "EG"):
        from R.scrp.r_order_actor import FixedHorizonActor
        return FixedHorizonActor(solver, 20, runtime, gp_rule)
    proposer = importlib.import_module(
        "Chain.chain_public" if solver == "Chain" else "Minmax.minmax_public"
    ).propose
    return _SingleStepPolicy(solver, proposer)


def select_once(solver: str, method: str, frame: object, *,
                runtime: Mapping[str, Any] | None = None,
                gp_rule: Mapping[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Run one decision from a caller-provided legal public frame."""
    runtime_value = load_runtime() if runtime is None else runtime
    gp_value = load_gp_rule() if gp_rule is None else gp_rule
    return make_policy(solver, method, runtime_value, gp_value).select(frame)
