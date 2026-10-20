"""Frozen-R selectors adapted to the public ERI/EG rules.

This module only accepts the same detached public frame used by the PLAN-043
references. It deliberately reuses PLAN-042's fixed-22 prefix projection and
the already-frozen R candidate scorer. No condition labels, evaluator p,
forecast permutation, private IDs, or physical truth enter either actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

from scrp.cross_solver_reliability_r import truncate_public_frame
from scrp.eri_eg_public import EG20Actor, EGPlanMismatchError, select_eri20_destination
from scrp.history_reliability_r import (
    FROZEN_R_COEFFICIENTS,
    _m2_select_scored_candidates,
)
from scrp.noise_aware_em import ActionCandidate, SelectorDecision
from scrp.noise_aware_forecast import HORIZONS


R_ERI_METHOD = "R_ERI_FROZEN"
R_EG_METHOD = "R_EG_GROUP_FROZEN"


@dataclass(frozen=True)
class _HorizonAction:
    requested_h: int
    destination: int
    prefix_view_ns: int
    selector_ns: int
    total_ns: int


def _candidate_features(requested_h: int, q_t: float) -> tuple[float, ...]:
    scaled_h = int(requested_h) / 20.0
    reliability_error = 1.0 - float(q_t)
    return (
        1.0,
        scaled_h,
        reliability_error,
        scaled_h * reliability_error,
    )


def _score_first_source(
    raw: Sequence[_HorizonAction], *, q_t: float, variant: str
) -> SelectorDecision:
    by_destination: dict[int, ActionCandidate] = {}
    for item in raw:  # HORIZONS is ascending; first destination source is minimum h.
        if item.destination in by_destination:
            continue
        by_destination[item.destination] = ActionCandidate(
            destination=int(item.destination),
            requested_h=int(item.requested_h),
            features=_candidate_features(item.requested_h, q_t),
        )
    candidates = tuple(
        sorted(by_destination.values(), key=lambda item: item.requested_h)
    )
    if not candidates or candidates[0].requested_h != 0:
        raise AssertionError("first-source deduplication lost the h=0 anchor")
    return _m2_select_scored_candidates(
        candidates,
        coefficients=FROZEN_R_COEFFICIENTS,
        variant=variant,
    )


def _candidate_record(item: _HorizonAction) -> dict[str, int]:
    return {
        "requested_h": int(item.requested_h),
        "destination": int(item.destination),
        "prefix_view_ns": int(item.prefix_view_ns),
        "selector_ns": int(item.selector_ns),
        "total_ns": int(item.total_ns),
    }


def _scored_candidates(decision: SelectorDecision) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in decision.candidates:
        rows.append(
            {
                "requested_h": int(item.requested_h),
                "destination": int(item.destination),
                "features": [float(value) for value in item.features],
            }
        )
    return rows


class RERIActor:
    """R-scored ERI destination selection, independently repeated per move."""

    def __init__(self) -> None:
        self.actor_calls = 0
        self.horizon_selector_calls = 0
        self.score_evaluations = 0

    def select_destination(self, frame: Any) -> dict[str, Any]:
        total_start = perf_counter_ns()
        raw: list[_HorizonAction] = []
        for requested_h in HORIZONS:
            one_start = perf_counter_ns()
            prefix_start = perf_counter_ns()
            prefix = truncate_public_frame(frame, requested_h)
            prefix_ns = perf_counter_ns() - prefix_start
            selector_start = perf_counter_ns()
            destination = int(select_eri20_destination(prefix))
            selector_ns = perf_counter_ns() - selector_start
            if destination not in frame.legal_destinations:
                raise ValueError("ERI horizon candidate is outside the shared legal mask")
            raw.append(
                _HorizonAction(
                    requested_h=int(requested_h),
                    destination=destination,
                    prefix_view_ns=int(prefix_ns),
                    selector_ns=int(selector_ns),
                    total_ns=int(perf_counter_ns() - one_start),
                )
            )

        q_t = float(frame.observation.q_t)
        decision = _score_first_source(raw, q_t=q_t, variant=R_ERI_METHOD)
        if decision.destination not in frame.legal_destinations:
            raise ValueError("R-ERI selected a destination outside the shared legal mask")
        if not any(
            item.requested_h == decision.requested_h
            and item.destination == decision.destination
            for item in decision.candidates
        ):
            raise AssertionError("R-ERI action/source does not identify a scored candidate")
        self.actor_calls += 1
        self.horizon_selector_calls += len(raw)
        self.score_evaluations += max(0, len(decision.candidates) - 1)
        return {
            "action": int(decision.destination),
            "selected_h": int(decision.requested_h),
            "predicted_delta": float(decision.predicted_delta),
            "q_used": q_t,
            "candidate_pool": [_candidate_record(item) for item in raw],
            "deduplicated_candidates": _scored_candidates(decision),
            "horizon_selector_calls": len(raw),
            "score_evaluations": max(0, len(decision.candidates) - 1),
            "actor_total_ns": int(perf_counter_ns() - total_start),
        }


@dataclass(frozen=True)
class _PublicSnapshot:
    ranks: np.ndarray
    occupied: np.ndarray
    legal_mask: np.ndarray
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int
    relocation_count: int
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    crane_position: int | None
    crane_time: float


def _snapshot(frame: Any) -> _PublicSnapshot:
    obs = frame.observation
    return _PublicSnapshot(
        ranks=np.asarray(obs.rank_tensor, dtype=np.int32).copy(),
        occupied=np.asarray(obs.occupied_mask, dtype=np.bool_).copy(),
        legal_mask=np.asarray(obs.legal_destination_mask, dtype=np.bool_).copy(),
        legal_destinations=tuple(int(value) for value in frame.legal_destinations),
        max_tiers=int(frame.max_tiers),
        retrieval_count=int(frame.retrieval_count),
        relocation_count=int(frame.relocation_count),
        history_hits=tuple(bool(value) for value in obs.history_hits),
        q_prior=float(obs.q_prior),
        q_t=float(obs.q_t),
        crane_position=(None if frame.crane_position is None else int(frame.crane_position)),
        crane_time=float(frame.crane_time),
    )


def _after_move(
    frame: Any,
    destination: int,
    *,
    crane_travel_time: float,
    pick_drop_time: float,
) -> _PublicSnapshot:
    before = _snapshot(frame)
    target_locations = np.argwhere(before.occupied & (before.ranks == 1))
    if target_locations.shape != (1, 2):
        raise ValueError("R-EG cache requires exactly one public current target")
    source = int(target_locations[0, 0])
    heights = before.occupied.sum(axis=1).astype(int).tolist()
    if destination not in before.legal_destinations:
        raise ValueError("R-EG plan emitted a destination outside the current legal mask")
    src_height = heights[source]
    dst_height = heights[destination]
    ranks = before.ranks.copy()
    occupied = before.occupied.copy()
    ranks[source, src_height - 1] = 0
    occupied[source, src_height - 1] = False
    ranks[destination, dst_height] = int(frame.observation.rank_tensor[source, src_height - 1])
    occupied[destination, dst_height] = True
    heights[source] -= 1
    heights[destination] += 1
    next_legal_mask = np.asarray(
        [stack != source and heights[stack] < before.max_tiers for stack in range(len(heights))],
        dtype=np.bool_,
    )

    source_position = source + 1
    destination_position = int(destination) + 1
    travel = float(crane_travel_time)
    crane_time = before.crane_time
    if before.crane_position is not None:
        crane_time += abs(before.crane_position - source_position) * travel
    crane_time += abs(source_position - destination_position) * travel + float(pick_drop_time)
    return _PublicSnapshot(
        ranks=ranks,
        occupied=occupied,
        legal_mask=next_legal_mask,
        legal_destinations=tuple(int(value) for value in np.flatnonzero(next_legal_mask)),
        max_tiers=before.max_tiers,
        retrieval_count=before.retrieval_count,
        relocation_count=before.relocation_count + 1,
        history_hits=before.history_hits,
        q_prior=before.q_prior,
        q_t=before.q_t,
        crane_position=destination_position,
        crane_time=crane_time,
    )


def _assert_snapshot(frame: Any, expected: _PublicSnapshot) -> None:
    actual = _snapshot(frame)
    if not (
        np.array_equal(actual.ranks, expected.ranks)
        and np.array_equal(actual.occupied, expected.occupied)
        and np.array_equal(actual.legal_mask, expected.legal_mask)
        and actual.legal_destinations == expected.legal_destinations
        and actual.max_tiers == expected.max_tiers
        and actual.retrieval_count == expected.retrieval_count
        and actual.relocation_count == expected.relocation_count
        and actual.history_hits == expected.history_hits
        and math.isclose(actual.q_prior, expected.q_prior, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(actual.q_t, expected.q_t, rel_tol=0.0, abs_tol=1e-15)
        and actual.crane_position == expected.crane_position
        and math.isclose(actual.crane_time, expected.crane_time, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise EGPlanMismatchError("full H20 public frame changed during a committed R-EG group")


def _plan_record(actor: EG20Actor, frame: Any, first_action: int, requested_h: int) -> dict[str, Any]:
    plan = actor._plan  # frozen EG20Actor stores only public ranks/destinations
    if plan is None:
        # A one-blocker group is consumed by the first call itself.
        targets = np.argwhere(
            np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
            & (np.asarray(frame.observation.rank_tensor, dtype=np.int32) == 1)
        )
        if targets.shape != (1, 2):
            raise ValueError("EG first-action plan has no unique public target")
        source, target_tier = (int(value) for value in targets[0])
        source_height = int(np.count_nonzero(frame.observation.occupied_mask[source]))
        blocker_ranks = (
            int(frame.observation.rank_tensor[source, source_height - 1]),
        )
        destinations = (int(first_action),)
        if target_tier != source_height - 2:
            raise AssertionError("cleared EG actor plan was not a one-blocker group")
    else:
        if int(plan.destinations_top_down[0]) != int(first_action):
            raise AssertionError("EG actor's first action differs from its cached group plan")
        blocker_ranks = tuple(int(value) for value in plan.blocker_ranks_top_down)
        destinations = tuple(int(value) for value in plan.destinations_top_down)
    return {
        "requested_h": int(requested_h),
        "first_destination": int(first_action),
        "blocker_ranks_top_down": list(blocker_ranks),
        "destinations_top_down": list(destinations),
    }


class REGGroupActor:
    """Select one horizon plan per target group, then commit its full suffix."""

    def __init__(
        self,
        *,
        crane_travel_time: float = 1.2,
        pick_drop_time: float = 30.0,
    ) -> None:
        if not math.isfinite(crane_travel_time) or crane_travel_time < 0:
            raise ValueError("crane_travel_time must be finite and nonnegative")
        if not math.isfinite(pick_drop_time) or pick_drop_time < 0:
            raise ValueError("pick_drop_time must be finite and nonnegative")
        self.crane_travel_time = float(crane_travel_time)
        self.pick_drop_time = float(pick_drop_time)
        self.actor_calls = 0
        self.group_plan_build_calls = 0
        self.groups_selected = 0
        self.cached_moves = 0
        self.score_evaluations = 0
        self._selected_actor: EG20Actor | None = None
        self._selected_h: int | None = None
        self._selected_delta = 0.0
        self._selected_q_t = 0.0
        self._expected_full: _PublicSnapshot | None = None
        self._expected_selected_prefix: _PublicSnapshot | None = None
        self._group_id = 0
        self._group_move_index = 0
        self._group_plan_length = 0
        self._selected_plan: dict[str, Any] | None = None

    def reset(self) -> None:
        self.actor_calls = 0
        self.group_plan_build_calls = 0
        self.groups_selected = 0
        self.cached_moves = 0
        self.score_evaluations = 0
        self._selected_actor = None
        self._selected_h = None
        self._selected_delta = 0.0
        self._selected_q_t = 0.0
        self._expected_full = None
        self._expected_selected_prefix = None
        self._group_id = 0
        self._group_move_index = 0
        self._group_plan_length = 0
        self._selected_plan = None

    def _new_group(self, frame: Any) -> tuple[int, dict[str, Any]]:
        total_start = perf_counter_ns()
        raw: list[_HorizonAction] = []
        plan_records: list[dict[str, Any]] = []
        actors_by_h: dict[int, EG20Actor] = {}
        for requested_h in HORIZONS:
            candidate_start = perf_counter_ns()
            prefix_start = perf_counter_ns()
            prefix = truncate_public_frame(frame, requested_h)
            prefix_ns = perf_counter_ns() - prefix_start
            actor = EG20Actor()
            selector_start = perf_counter_ns()
            first_destination = int(actor.select_destination(prefix))
            selector_ns = perf_counter_ns() - selector_start
            if first_destination not in frame.legal_destinations:
                raise ValueError("EG horizon plan starts outside the shared legal mask")
            record = _plan_record(actor, prefix, first_destination, requested_h)
            if not record["destinations_top_down"]:
                raise AssertionError("EG plan has no first destination")
            raw.append(
                _HorizonAction(
                    requested_h=int(requested_h),
                    destination=first_destination,
                    prefix_view_ns=int(prefix_ns),
                    selector_ns=int(selector_ns),
                    total_ns=int(perf_counter_ns() - candidate_start),
                )
            )
            plan_records.append(record)
            actors_by_h[int(requested_h)] = actor

        q_t = float(frame.observation.q_t)
        decision = _score_first_source(raw, q_t=q_t, variant=R_EG_METHOD)
        chosen_h = int(decision.requested_h)
        selected_actor = actors_by_h[chosen_h]
        selected_record = next(row for row in plan_records if row["requested_h"] == chosen_h)
        if int(decision.destination) != int(selected_record["first_destination"]):
            raise AssertionError("R-EG selected action does not match selected plan's first move")
        if decision.destination not in frame.legal_destinations:
            raise ValueError("R-EG selected destination is outside the shared legal mask")

        self._selected_actor = selected_actor
        self._selected_h = chosen_h
        self._selected_delta = float(decision.predicted_delta)
        self._selected_q_t = q_t
        self._group_move_index = 1
        self._group_plan_length = len(selected_record["destinations_top_down"])
        self._selected_plan = selected_record
        self._group_id += 1
        self.groups_selected += 1
        self.group_plan_build_calls += len(raw)
        self.score_evaluations += max(0, len(decision.candidates) - 1)
        self._expected_full = None
        self._expected_selected_prefix = None
        output = {
            "action": int(decision.destination),
            "selected_h": chosen_h,
            "predicted_delta": float(decision.predicted_delta),
            "q_used": q_t,
            "group_started": True,
            "group_id": self._group_id,
            "group_move_index": 1,
            "group_plan_length": self._group_plan_length,
            "candidate_plans": plan_records,
            "deduplicated_candidates": _scored_candidates(decision),
            "committed_plan": selected_record,
            "horizon_plan_build_calls": len(raw),
            "score_evaluations": max(0, len(decision.candidates) - 1),
            "candidate_pool": [_candidate_record(item) for item in raw],
            "group_pool_total_ns": int(perf_counter_ns() - total_start),
            "candidate_plan_times_ns": [int(item.total_ns) for item in raw],
        }
        return int(decision.destination), output

    def select_destination(self, frame: Any) -> dict[str, Any]:
        total_start = perf_counter_ns()
        if self._selected_actor is None:
            action, output = self._new_group(frame)
            group_started = True
        else:
            if self._expected_full is None or self._selected_h is None:
                raise EGPlanMismatchError("R-EG cache is missing its expected H20 frame")
            _assert_snapshot(frame, self._expected_full)
            prefix = truncate_public_frame(frame, self._selected_h)
            if self._expected_selected_prefix is None:
                raise EGPlanMismatchError("R-EG cache is missing its expected selected-prefix frame")
            _assert_snapshot(prefix, self._expected_selected_prefix)
            before_cached = int(self._selected_actor.cached_moves)
            action = int(self._selected_actor.select_destination(prefix))
            if action not in frame.legal_destinations:
                raise ValueError("cached R-EG action is outside the current full-frame legal mask")
            self.cached_moves += int(self._selected_actor.cached_moves) - before_cached
            self._group_move_index += 1
            group_started = False
            output = {
                "action": action,
                "selected_h": int(self._selected_h),
                "predicted_delta": float(self._selected_delta),
                "q_used": float(self._selected_q_t),
                "group_started": False,
                "group_id": self._group_id,
                "group_move_index": self._group_move_index,
                "group_plan_length": self._group_plan_length,
                "candidate_plans": [],
                "deduplicated_candidates": [],
                "committed_plan": self._selected_plan,
                "horizon_plan_build_calls": 0,
                "score_evaluations": 0,
                "candidate_pool": [],
                "cached_plan_move": True,
            }

        selected_h = int(output["selected_h"])
        expected_full = _after_move(
            frame,
            action,
            crane_travel_time=self.crane_travel_time,
            pick_drop_time=self.pick_drop_time,
        )
        selected_prefix = truncate_public_frame(frame, selected_h)
        expected_prefix = _after_move(
            selected_prefix,
            action,
            crane_travel_time=self.crane_travel_time,
            pick_drop_time=self.pick_drop_time,
        )
        self.actor_calls += 1
        if self._group_move_index < self._group_plan_length:
            self._expected_full = expected_full
            # The EG20Actor owns and verifies the selected-prefix transition.
            self._expected_selected_prefix = expected_prefix
        else:
            # The just-emitted move ends the blocker group. The environment may
            # now reveal one or more targets; the next call starts a fresh pool.
            self._selected_actor = None
            self._selected_h = None
            self._expected_full = None
            self._selected_plan = None
            self._group_move_index = 0
            self._group_plan_length = 0
            self._expected_selected_prefix = None
        output["cached_plan_move"] = bool(not group_started)
        output["remaining_plan_moves"] = max(0, int(output["group_plan_length"]) - int(output["group_move_index"]))
        output["q_t_current"] = float(frame.observation.q_t)
        output["actor_total_ns"] = int(perf_counter_ns() - total_start)
        return output


__all__ = [
    "R_EG_METHOD",
    "R_ERI_METHOD",
    "REGGroupActor",
    "RERIActor",
]
