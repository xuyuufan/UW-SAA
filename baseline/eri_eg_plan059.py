"""PLAN-059 isolated correction for EG's finite source sentinel.

The public EG adaptation and its tie/group rules remain the source of truth.
This module carries one narrowly corrected copy of the group builder: when
``diffMin`` is nonpositive, the maximum is taken over legal non-source,
non-full candidates so the finite source sentinel cannot beat every action.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Mapping

import numpy as np

from baseline import eri_eg_public as _base


EGPlan = _base.EGPlan
EGPlanMismatchError = _base.EGPlanMismatchError
ERI_EG_AUTHOR_CODE_URLS = _base.ERI_EG_AUTHOR_CODE_URLS
ERI_EG_VERSION = _base.ERI_EG_VERSION + "+plan059-candidate-diffmin"
PUBLIC_UNKNOWN_RANK = _base.PUBLIC_UNKNOWN_RANK
PublicFrameError = _base.PublicFrameError
PublicDecisionFrame = _base.PublicDecisionFrame


def select_eri20_destination(frame: PublicDecisionFrame) -> int:
    """Keep the established ERI public selector available on the runtime slot."""
    return _base.select_eri20_destination(frame)


def build_eg20_plan(frame: PublicDecisionFrame) -> EGPlan:
    """Build the established two-phase EG plan with the sentinel fix only."""
    state = _base._public_state(frame)
    source_height = state.heights[state.source]
    source_bottom_up = state.ranks[state.source, :source_height]
    blockers = tuple(int(value) for value in source_bottom_up[state.target_tier + 1 :][::-1])
    if not blockers:
        raise PublicFrameError("EG is only defined when the current target is blocked")

    stack_count = state.ranks.shape[0]
    max_tiers = state.ranks.shape[1]
    max_public_rank = int(np.max(state.ranks[state.occupied]))
    virtual_min = [0] * stack_count
    for stack in range(stack_count):
        if stack == state.source:
            continue
        height = state.heights[stack]
        if height == 0:
            virtual_min[stack] = max_public_rank + 1
        elif height == max_tiers:
            virtual_min[stack] = 0
        else:
            virtual_min[stack] = int(np.min(state.ranks[stack, :height]))

    positions: dict[int, list[int]] = {}
    for position, rank in enumerate(blockers, start=1):
        positions.setdefault(rank, []).append(position)
    assigned_positions: list[list[int]] = [[] for _ in range(stack_count)]
    first_phase_order = tuple(sorted(blockers, reverse=True))
    first_phase_relocated: list[bool] = []

    for blocker in first_phase_order:
        source_position = positions[blocker][0]
        target_stack = None
        if blocker < max(virtual_min):
            eligible = [
                stack
                for stack in range(stack_count)
                if virtual_min[stack] > blocker
                and max(assigned_positions[stack], default=0) < source_position
            ]
            if eligible:
                target_stack = min(eligible, key=lambda stack: (virtual_min[stack], stack))
        if target_stack is None:
            first_phase_relocated.append(False)
            continue

        first_phase_relocated.append(True)
        assigned_positions[target_stack].append(source_position)
        positions[blocker].pop(0)
        added = len(assigned_positions[target_stack])
        if state.heights[target_stack] + added == max_tiers:
            virtual_min[target_stack] = 0
        else:
            virtual_min[target_stack] = blocker

    for stack in range(stack_count):
        if stack == state.source:
            continue
        added = len(assigned_positions[stack])
        if added > 1 or state.heights[stack] + added == max_tiers:
            virtual_min[stack] = 0

    if sum(len(items) for items in assigned_positions) < len(blockers):
        second_phase = [
            blocker
            for blocker, relocated in zip(first_phase_order, first_phase_relocated, strict=True)
            if not relocated
        ]
        second_phase.sort()
        for blocker in second_phase:
            source_position = positions[blocker][0]
            diff_min: list[int] = []
            for stack in range(stack_count):
                if stack == state.source or state.heights[stack] + len(assigned_positions[stack]) == max_tiers:
                    diff_min.append(-max_tiers * stack_count)
                elif len(assigned_positions[stack]) == 1:
                    diff_min.append(blocker - virtual_min[stack])
                else:
                    diff_min.append(virtual_min[stack] - blocker)

            candidates = [
                stack
                for stack in range(stack_count)
                if stack != state.source
                and state.heights[stack] + len(assigned_positions[stack]) < max_tiers
            ]
            if not candidates:
                raise RuntimeError("EG source-code adaptation found no remaining capacity")

            if max(diff_min) > 0:
                target_min = min(value for value in diff_min if value > 0)
                tied = [stack for stack in candidates if diff_min[stack] == target_min]
                if not tied:
                    raise RuntimeError("EG diffMin has no feasible positive destination")
                target_stack = min(
                    tied,
                    key=lambda stack: (-(state.heights[stack] + len(assigned_positions[stack])), stack),
                )
            else:
                # Only this maximum changes: excluded source/full sentinels
                # are not actions and cannot determine a candidate tie set.
                target_min = max(diff_min[stack] for stack in candidates)
                tie_rows: list[tuple[int, int, int]] = []
                for stack in candidates:
                    if diff_min[stack] != target_min:
                        continue
                    existing = state.ranks[stack, : state.heights[stack]]
                    existing_min = int(np.min(existing)) if existing.size else None
                    n_min = (
                        0
                        if existing_min is None
                        else int(np.count_nonzero(existing == existing_min))
                    )
                    # Preserve retrieveEG.m's literal logical-index behavior:
                    # the true prefix selects the first assigned-count items
                    # from the top-down blocker vector.
                    assigned_prefix = blockers[: len(assigned_positions[stack])]
                    if assigned_prefix:
                        assigned_min = min(assigned_prefix)
                        if n_min == 0 or (existing_min is not None and assigned_min < existing_min):
                            n_min = sum(value == assigned_min for value in assigned_prefix)
                    resulting_height = state.heights[stack] + len(assigned_positions[stack])
                    tie_rows.append((n_min, -resulting_height, stack))
                if not tie_rows:
                    raise RuntimeError("EG diffMin has no feasible nonpositive destination")
                target_stack = min(tie_rows)[2]

            assigned_positions[target_stack].append(source_position)
            positions[blocker].pop(0)
            if state.heights[target_stack] + len(assigned_positions[target_stack]) == max_tiers:
                virtual_min[target_stack] = 0

    by_source_position: dict[int, int] = {}
    for stack in range(stack_count):
        for position in assigned_positions[stack]:
            if position in by_source_position:
                raise AssertionError("EG assigned one source blocker more than once")
            by_source_position[position] = stack
    if set(by_source_position) != set(range(1, len(blockers) + 1)):
        raise RuntimeError("EG group plan did not assign every blocker exactly once")

    destinations = tuple(by_source_position[position] for position in range(1, len(blockers) + 1))
    simulated_heights = list(state.heights)
    for destination in destinations:
        if destination == state.source or simulated_heights[destination] >= max_tiers:
            raise RuntimeError("EG plan exceeds stack capacity")
        simulated_heights[destination] += 1

    return EGPlan(
        source=state.source,
        blocker_ranks_top_down=blockers,
        destinations_top_down=destinations,
    )


class EG20Actor(_base.EG20Actor):
    """The original state/cache contract using this module's corrected builder."""

    def select_destination(self, frame: PublicDecisionFrame) -> int:
        state = _base._public_state(frame)
        if self._expected is not None:
            self._verify_expected(frame, self._expected)
            self._expected = None
            self._next_move += 1
            self.cached_moves += 1

        if self._plan is None:
            self._plan = build_eg20_plan(frame)
            self._next_move = 0
            self.groups_planned += 1
        if self._plan.source != state.source:
            raise EGPlanMismatchError("EG source stack changed before its group finished")
        destination = self._plan.destinations_top_down[self._next_move]
        final_move = self._next_move == len(self._plan.destinations_top_down) - 1
        if final_move:
            self._plan = None
            self._next_move = 0
            self._expected = None
        else:
            self._expected = _base._expected_after_move(frame, state, destination)
        return int(destination)


def runtime_with_corrected_eg(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a runtime mapping and replace only its ``eri_eg`` capability slot."""
    if "eri_eg" not in runtime:
        raise KeyError("PLAN-059 runtime lacks its eri_eg slot")
    corrected = dict(runtime)
    corrected["eri_eg"] = sys.modules[__name__]
    return corrected


__all__ = [
    "EG20Actor",
    "EGPlan",
    "EGPlanMismatchError",
    "ERI_EG_AUTHOR_CODE_URLS",
    "ERI_EG_VERSION",
    "PUBLIC_UNKNOWN_RANK",
    "PublicDecisionFrame",
    "PublicFrameError",
    "build_eg20_plan",
    "runtime_with_corrected_eg",
    "select_eri20_destination",
]
