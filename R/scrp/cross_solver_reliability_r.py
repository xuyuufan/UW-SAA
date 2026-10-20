"""PLAN-042 public-prefix adapter for frozen GP/H20 rules and frozen R scoring.

Only ``PublicDecisionFrame`` enters this module. Episode truth, forecasts,
evaluator labels, and method-selection metadata belong to the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np


_R_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = _R_ROOT.parent
_EM_ROOT = _WORKSPACE / "EM"
_BASE_ROOT = _WORKSPACE / "Mijn_RL"


def _install_workspace_import_paths() -> None:
    """Expose the intentionally split EM/R/Mijn_RL packages to this runner."""
    for item in (_WORKSPACE, _R_ROOT, _EM_ROOT, _BASE_ROOT):
        text = str(item.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)

    # These three roots share legacy top-level package names. Keep the modules
    # in their owning project directories while allowing the shared adapter to
    # import their distinct submodules in one process.
    for package, roots in (
        ("scrp", (_R_ROOT, _EM_ROOT, _BASE_ROOT)),
        ("experiments", (_EM_ROOT, _R_ROOT, _BASE_ROOT)),
    ):
        module = sys.modules.get(package)
        if module is None:
            __import__(package)
            module = sys.modules[package]
        locations = [str((root / package).resolve()) for root in roots]
        module.__path__ = locations
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.submodule_search_locations is not None:
            spec.submodule_search_locations[:] = locations


_install_workspace_import_paths()

from scrp.noise_aware_forecast import HORIZONS, PublicObservation  # noqa: E402
from scrp.noise_aware_em import ActionCandidate, SelectorDecision  # noqa: E402
from scrp.history_reliability_r import (  # noqa: E402
    FROZEN_R_COEFFICIENTS,
    FROZEN_R_Q_PRIOR,
    _m2_select_scored_candidates,
)
from experiments.noise_aware_reference_p0 import (  # noqa: E402
    PUBLIC_UNKNOWN_RANK,
    PublicDecisionFrame,
    gp_action,
    h20_action_from_tensor,
    make_public_frame,
    public_frame_to_legacy_state,
)
from scrp.scale_diverse_policy import Online2DPriorityAdapter  # noqa: E402


FIXED_METHOD_TO_H = {
    "FIXED0": 0,
    "FIXED2": 2,
    "FIXED5": 5,
    "FIXED10": 10,
    "FIXED20": 20,
}
TRANSFER_METHODS = ("R_TRANSFER", "R_CONST_TRANSFER")
SCIENTIFIC_METHODS = tuple(FIXED_METHOD_TO_H) + TRANSFER_METHODS
SOLVERS = ("GP", "H20")


@dataclass(frozen=True)
class RawSolverCandidate:
    """One horizon-specific action and its actor-only timing breakdown."""

    requested_h: int
    destination: int
    prefix_view_ns: int
    bridge_ns: int
    inference_ns: int
    total_call_ns: int


@dataclass(frozen=True)
class ActorCall:
    """Full five-call candidate pool and one registered method's decision."""

    solver: str
    method: str
    action: int
    selected_h: int
    delta: float
    raw_candidates: tuple[RawSolverCandidate, ...]
    deduplicated_candidates: tuple[ActionCandidate, ...]
    solver_invocation_count: int
    timing_ns: Mapping[str, int]


def truncate_public_frame(
    frame: PublicDecisionFrame, requested_h: int
) -> PublicDecisionFrame:
    """Censor every occupied rank beyond current+``requested_h`` as 22.

    Empty padding remains 0. The legacy bridge then maps occupied 22 to its
    unknown value 0, which GP and the H20 priority adapter already understand.
    """
    h = int(requested_h)
    if h not in HORIZONS:
        raise ValueError(f"requested horizon must be one of {HORIZONS}, got {h}")
    observation = frame.observation
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    legal = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public rank and occupancy shapes differ")
    if ranks.shape[0] != legal.shape[0] or ranks.shape[1] != frame.max_tiers:
        raise ValueError("public frame shape differs from its mask/tier contract")
    if tuple(int(x) for x in np.flatnonzero(legal)) != frame.legal_destinations:
        raise ValueError("public frame legal destinations changed")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("public frame must contain exactly one current target")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty padding must remain rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > PUBLIC_UNKNOWN_RANK))):
        raise ValueError("occupied public ranks must be in 1..22")

    censored = ranks.copy()
    censored[occupied & (censored > h + 1)] = PUBLIC_UNKNOWN_RANK
    prefix = PublicObservation(
        rank_tensor=censored,
        occupied_mask=occupied,
        legal_destination_mask=legal,
        history_hits=observation.history_hits,
        q_prior=observation.q_prior,
        q_t=observation.q_t,
    )
    return replace(frame, observation=prefix)


def _run_one_horizon(
    frame: PublicDecisionFrame,
    *,
    requested_h: int,
    solver: str,
    frozen_solver: Any,
) -> RawSolverCandidate:
    call_start = perf_counter_ns()
    prefix_start = perf_counter_ns()
    prefix = truncate_public_frame(frame, requested_h)
    prefix_ns = perf_counter_ns() - prefix_start

    bridge_start = perf_counter_ns()
    legacy = public_frame_to_legacy_state(prefix)
    priorities = None
    if solver == "H20":
        priorities = Online2DPriorityAdapter(prefix.max_tiers).build(legacy)
    bridge_ns = perf_counter_ns() - bridge_start

    inference_start = perf_counter_ns()
    if solver == "GP":
        destination = int(
            frozen_solver["selector"](
                legacy,
                prefix.legal_destinations,
                max_tiers=prefix.max_tiers,
            )
        )
    elif solver == "H20":
        if priorities is None:
            raise AssertionError("H20 bridge did not build priorities")
        destination = h20_action_from_tensor(
            frozen_solver, priorities, prefix.legal_destinations
        )
    else:
        raise ValueError(f"unknown frozen solver {solver!r}")
    inference_ns = perf_counter_ns() - inference_start
    if destination not in prefix.legal_destinations:
        raise ValueError(f"{solver} returned illegal action {destination}")
    return RawSolverCandidate(
        requested_h=int(requested_h),
        destination=destination,
        prefix_view_ns=prefix_ns,
        bridge_ns=bridge_ns,
        inference_ns=inference_ns,
        total_call_ns=perf_counter_ns() - call_start,
    )


def _deduplicate_r_candidates(
    raw_candidates: Sequence[RawSolverCandidate], *, q: float
) -> tuple[ActionCandidate, ...]:
    by_destination: dict[int, ActionCandidate] = {}
    for raw in raw_candidates:
        if raw.destination in by_destination:
            continue
        scaled_h = raw.requested_h / 20.0
        error = 1.0 - float(q)
        by_destination[raw.destination] = ActionCandidate(
            destination=raw.destination,
            requested_h=raw.requested_h,
            features=(1.0, scaled_h, error, scaled_h * error),
        )
    return tuple(sorted(by_destination.values(), key=lambda item: item.requested_h))


def frozen_actor_call(
    frame: PublicDecisionFrame,
    *,
    solver: str,
    frozen_solver: Any,
    method: str,
) -> ActorCall:
    """Run the complete five-call pool and apply one of seven frozen methods.

    ``frozen_solver`` is the selected public GP rule mapping or the frozen H20
    module. No evaluator-only values are accepted by this interface.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}")
    if method not in SCIENTIFIC_METHODS:
        raise ValueError(f"method must be one of {SCIENTIFIC_METHODS}")
    total_start = perf_counter_ns()
    raw = tuple(
        _run_one_horizon(
            frame,
            requested_h=h,
            solver=solver,
            frozen_solver=frozen_solver,
        )
        for h in HORIZONS
    )
    if len(raw) != 5 or tuple(row.requested_h for row in raw) != tuple(HORIZONS):
        raise AssertionError("every actor call must execute the complete five-horizon pool")

    selector_start = perf_counter_ns()
    if method in FIXED_METHOD_TO_H:
        selected_h = FIXED_METHOD_TO_H[method]
        selected = next(item for item in raw if item.requested_h == selected_h)
        action = selected.destination
        delta = 0.0
        scored_candidates: tuple[ActionCandidate, ...] = ()
    else:
        q = (
            frame.observation.q_prior
            if method == "R_CONST_TRANSFER"
            else frame.observation.q_t
        )
        scored_candidates = _deduplicate_r_candidates(raw, q=q)
        if not scored_candidates or scored_candidates[0].requested_h != 0:
            raise AssertionError("R candidate deduplication lost the same-solver h=0 anchor")
        decision: SelectorDecision = _m2_select_scored_candidates(
            scored_candidates,
            coefficients=FROZEN_R_COEFFICIENTS,
            variant=method,
        )
        action = int(decision.destination)
        selected_h = int(decision.requested_h)
        delta = float(decision.predicted_delta)
    selector_ns = perf_counter_ns() - selector_start
    if action not in frame.legal_destinations:
        raise ValueError(f"{method} returned illegal action {action}")

    timing = {
        "prefix_view_ns": sum(row.prefix_view_ns for row in raw),
        "bridge_ns": sum(row.bridge_ns for row in raw),
        "inference_ns": sum(row.inference_ns for row in raw),
        "selector_ns": selector_ns,
        "total_actor_ns": perf_counter_ns() - total_start,
    }
    return ActorCall(
        solver=solver,
        method=method,
        action=action,
        selected_h=selected_h,
        delta=delta,
        raw_candidates=raw,
        deduplicated_candidates=scored_candidates,
        solver_invocation_count=len(raw),
        timing_ns=timing,
    )


__all__ = [
    "ActorCall",
    "FIXED_METHOD_TO_H",
    "RawSolverCandidate",
    "SCIENTIFIC_METHODS",
    "SOLVERS",
    "TRANSFER_METHODS",
    "frozen_actor_call",
    "make_public_frame",
    "truncate_public_frame",
]
