"""Six registered public Native20 solvers and horizon-specific actor adapters."""
from __future__ import annotations

from typing import Any, Mapping

from environment.noise_aware_forecast import HORIZONS
from environment.public_frame import truncate_public_frame, public_frame_to_legacy_state
from baseline.em import select_em_destination
from baseline.eri_eg_public import select_eri20_destination
from baseline import eri_eg_plan059
from baseline.gp import select_destination as gp_select_destination
from baseline import chain, minmax

SOLVERS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")


def _strict_horizon(value: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("horizon must be an integer")
    result = int(value)
    if result not in HORIZONS:
        raise ValueError("actor horizon is outside the registered set")
    return result


class EMNativeActor:
    """One source-equivalent EM call on a censored public horizon."""

    def __init__(self, runtime: Mapping[str, Any], horizon: int) -> None:
        self.runtime = runtime
        self.horizon = _strict_horizon(horizon)
        self.invocation_count = 0

    def __call__(self, frame: object) -> int:
        prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
        action = int(self.runtime["em_select"](
            prefix.observation, max_tiers=frame.max_tiers, h=self.horizon
        ))
        self.invocation_count += 1
        return action


class GPNativeActor:
    """One GP public-prefix bridge/inference call with its registered legality check."""

    def __init__(self, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any], horizon: int) -> None:
        self.runtime = runtime
        self.gp_rule = gp_rule
        self.horizon = _strict_horizon(horizon)
        self.invocation_count = 0

    def __call__(self, frame: object) -> int:
        prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
        legacy = self.runtime["public_frame_to_legacy_state"](prefix)
        action = int(self.gp_rule["selector"](
            legacy, prefix.legal_destinations, max_tiers=prefix.max_tiers
        ))
        if action not in prefix.legal_destinations:
            raise ValueError(f"GP returned illegal action {action}")
        self.invocation_count += 1
        return action


class ERINativeActor:
    """One source-equivalent ERI call on a censored public horizon."""

    def __init__(self, runtime: Mapping[str, Any], horizon: int) -> None:
        self.runtime = runtime
        self.horizon = _strict_horizon(horizon)
        self.invocation_count = 0

    def __call__(self, frame: object) -> int:
        prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
        action = int(self.runtime["eri_select"](prefix))
        self.invocation_count += 1
        return action


class EGNativeActor:
    """The committed group actor exposed by the public EG implementation."""

    def __init__(self, runtime: Mapping[str, Any]) -> None:
        self.actor = runtime["eri_eg"].EG20Actor()

    @property
    def is_group_start(self) -> bool:
        return getattr(self.actor, "_plan", None) is None

    @property
    def cached_moves(self) -> int:
        return int(getattr(self.actor, "cached_moves", 0))

    def reset(self) -> None:
        self.actor.reset()

    def __call__(self, frame: object) -> int:
        return int(self.actor.select_destination(frame))


def make_native_actor(
    algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
    horizon: int = 20,
) -> Any:
    if algorithm == "EM":
        return EMNativeActor(runtime, horizon)
    if algorithm == "GP":
        return GPNativeActor(runtime, gp_rule, horizon)
    if algorithm == "ERI":
        return ERINativeActor(runtime, horizon)
    if algorithm == "EG":
        if _strict_horizon(horizon) != 20:
            raise ValueError("the registered EG cached actor is defined only at h=20")
        return EGNativeActor(runtime)
    raise ValueError(f"unknown native actor: {algorithm}")


def build_native_proposals(
    frame: object,
    algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
) -> dict[int, tuple[int, ...]]:
    """Build the exact five registered native proposals used by UWSAA."""
    if algorithm not in SOLVERS[:4]:
        raise ValueError("unknown native proposer")
    proposals: dict[int, tuple[int, ...]] = {}
    for horizon in HORIZONS:
        if algorithm == "EG":
            prefix = runtime["truncate_public_frame"](frame, horizon)
            plan = runtime["eri_eg"].build_eg20_plan(prefix)
            proposal = tuple(int(value) for value in plan.destinations_top_down)
        else:
            actor = make_native_actor(algorithm, runtime, gp_rule, horizon)
            proposal = (int(actor(frame)),)
        proposals[horizon] = proposal
    return proposals


class FixedHorizonActor:
    """One exact fixed-horizon actor; EG keeps its committed group cache."""

    def __init__(self, algorithm: str, horizon: int, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]):
        if algorithm not in SOLVERS[:4] or _strict_horizon(horizon) != horizon:
            raise ValueError("fixed actor algorithm/horizon is outside registration")
        self.algorithm = algorithm
        self.horizon = int(horizon)
        self.runtime = runtime
        self.gp_rule = gp_rule
        self._eg_actor = make_native_actor("EG", runtime, gp_rule, 20) if algorithm == "EG" else None
        self._native_actor = None if algorithm == "EG" else make_native_actor(algorithm, runtime, gp_rule, self.horizon)

    @property
    def is_group_start(self) -> bool:
        return self._eg_actor is None or self._eg_actor.is_group_start

    @property
    def cached_moves(self) -> int:
        return 0 if self._eg_actor is None else int(self._eg_actor.cached_moves)

    def reset(self) -> None:
        if self._eg_actor is not None:
            self._eg_actor.reset()

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        if self.algorithm == "EG":
            start = self.is_group_start
            prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
            action = int(self._eg_actor(prefix))
            return action, {
                "policy": f"FIXED{self.horizon}",
                "requested_h": self.horizon,
                "group_start": bool(start),
                "cached_moves": self.cached_moves,
            }
        action = int(self._native_actor(frame))
        return action, {"policy": f"FIXED{self.horizon}", "requested_h": self.horizon}


class SingleStepPolicy:
    def __init__(self, solver: str, proposer: Any) -> None:
        self.solver, self.proposer = solver, proposer

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        action, trace = self.proposer(frame, 20)
        return int(action), {"policy": "NATIVE20", "solver": self.solver, **dict(trace)}
