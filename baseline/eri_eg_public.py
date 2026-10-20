"""Public-frame adaptations of Galle et al.'s ERI and EG relocation rules.

The actors accept only the forecast-derived ``PublicDecisionFrame`` contract.
Rank 22 is one exchangeable, censored tail class; rank 0 is reserved for empty
padding. This module is independent of model loading, private forecasts, and
evaluator condition metadata.

Algorithm provenance: Galle et al. (2018), Sections 2.1.1 and 2.1.3, plus the
author's public ``retrieveERI.m`` / ``retrieveEG.m`` implementations. The EG
implementation intentionally follows the MATLAB code's phase-specific
``virtualMinVector`` / ``diffMin`` / ``nMin`` updates and its leftmost tie
behavior. It does not silently merge the paper's different phase-1 height-tie
description. This is a clean-room public-input adaptation, not a source copy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


PUBLIC_UNKNOWN_RANK = 22
ERI_EG_VERSION = "forecast-input-adapted-v1"
ERI_EG_AUTHOR_CODE_URLS = (
    "https://github.com/vgalle/StochasticCRP/blob/master/retrieveERI.m",
    "https://github.com/vgalle/StochasticCRP/blob/master/retrieveEG.m",
)


class _Observation(Protocol):
    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float


class PublicDecisionFrame(Protocol):
    """Structural type only; runtime actors receive the existing public frame."""

    observation: _Observation
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int
    relocation_count: int


class PublicFrameError(ValueError):
    """The supplied public frame violates the actor contract."""


class EGPlanMismatchError(RuntimeError):
    """The public state no longer matches an unfinished EG group plan."""


@dataclass(frozen=True)
class _PublicState:
    ranks: np.ndarray
    occupied: np.ndarray
    legal_mask: np.ndarray
    legal_destinations: tuple[int, ...]
    heights: tuple[int, ...]
    source: int
    target_tier: int


def _public_state(frame: PublicDecisionFrame) -> _PublicState:
    observation = frame.observation
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    legal_mask = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise PublicFrameError("rank and occupancy arrays must share shape [S,T]")
    stacks, tiers = ranks.shape
    if stacks <= 1 or tiers != int(frame.max_tiers) or legal_mask.shape != (stacks,):
        raise PublicFrameError("frame geometry differs from its rank/mask contract")
    legal = tuple(int(item) for item in np.flatnonzero(legal_mask))
    if legal != tuple(int(item) for item in frame.legal_destinations):
        raise PublicFrameError("legal destination tuple differs from the public mask")
    if not legal:
        raise PublicFrameError("a nonterminal decision must have a legal destination")
    if np.any(ranks[~occupied] != 0):
        raise PublicFrameError("empty padding must have rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > PUBLIC_UNKNOWN_RANK))):
        raise PublicFrameError("occupied ranks must be in 1..22")

    heights: list[int] = []
    visible_singletons: list[int] = []
    for stack in range(stacks):
        height = int(np.count_nonzero(occupied[stack]))
        expected = np.arange(tiers) < height
        if not np.array_equal(occupied[stack], expected):
            raise PublicFrameError("occupied cells must form a bottom-filled prefix")
        heights.append(height)
        visible_singletons.extend(
            int(value) for value in ranks[stack, :height] if int(value) < PUBLIC_UNKNOWN_RANK
        )
    target_locations = np.argwhere(occupied & (ranks == 1))
    if target_locations.shape != (1, 2):
        raise PublicFrameError("public frame must contain exactly one rank-1 target")
    if len(visible_singletons) != len(set(visible_singletons)):
        raise PublicFrameError("known public forecast ranks must be unique")
    source, target_tier = (int(item) for item in target_locations[0])
    if any(destination == source for destination in legal):
        raise PublicFrameError("current target source cannot be a legal destination")
    if any(heights[destination] >= tiers for destination in legal):
        raise PublicFrameError("a full stack cannot be a legal destination")
    expected_legal = tuple(
        stack for stack in range(stacks) if stack != source and heights[stack] < tiers
    )
    if legal != expected_legal:
        raise PublicFrameError("legal mask must enumerate every non-source non-full stack")
    return _PublicState(
        ranks=ranks,
        occupied=occupied,
        legal_mask=legal_mask,
        legal_destinations=legal,
        heights=tuple(heights),
        source=source,
        target_tier=target_tier,
    )


def select_eri20_destination(frame: PublicDecisionFrame) -> int:
    """Choose one ERI destination for the current topmost blocker.

    The source rule is the author's per-blocker expected-inversion score:
    each destination box with public rank below the blocker contributes one;
    an equal label contributes one half. The rank-22 exchangeable tail thus
    contributes one half per unknown box when the moved blocker is also rank
    22. Ties prefer the taller destination and then the leftmost stack.
    """
    state = _public_state(frame)
    source_ranks = state.ranks[state.source, : state.heights[state.source]]
    blockers_above_target = state.heights[state.source] - state.target_tier - 1
    if blockers_above_target <= 0:
        raise PublicFrameError("ERI is only defined when the current target is blocked")
    blocker = int(source_ranks[-1])

    scored: list[tuple[float, int, int]] = []
    for destination in state.legal_destinations:
        height = state.heights[destination]
        values = state.ranks[destination, :height]
        expected_inversions = float(np.count_nonzero(values < blocker))
        expected_inversions += 0.5 * float(np.count_nonzero(values == blocker))
        scored.append((expected_inversions, -height, destination))
    return min(scored)[2]


@dataclass(frozen=True)
class EGPlan:
    """Public-rank group plan in top-to-bottom physical execution order."""

    source: int
    blocker_ranks_top_down: tuple[int, ...]
    destinations_top_down: tuple[int, ...]


def build_eg20_plan(frame: PublicDecisionFrame) -> EGPlan:
    """Construct the author's two-phase EG assignment using public ranks only."""
    state = _public_state(frame)
    source_height = state.heights[state.source]
    # Tiers are bottom-up in PublicObservation; EG executes source blockers top-down.
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

    # Positions are one-based indexes into the top-to-bottom blocker sequence,
    # mirroring the source code's Positions(rank).Pos queues without physical IDs.
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
                # The MATLAB loop updates only on strict improvement, so an
                # equal virtual minimum retains the leftmost stack.
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

    # retrieveEG.m applies these adjustments after its first phase, including
    # the author-code rule that more than one first-phase assignment sets vmin=0.
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
        second_phase.sort()  # stable for the repeated unknown-tail label 22
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
                # Strict height improvement preserves the leftmost final tie.
                target_stack = min(
                    tied,
                    key=lambda stack: (-(state.heights[stack] + len(assigned_positions[stack])), stack),
                )
            else:
                target_min = max(diff_min)
                tie_rows: list[tuple[int, int, int]] = []
                for stack in candidates:
                    if diff_min[stack] != target_min:
                        continue
                    existing = state.ranks[stack, : state.heights[stack]]
                    existing_min = int(np.min(existing)) if existing.size else None
                    if existing_min is None:
                        n_min = 0
                    else:
                        n_min = int(np.count_nonzero(existing == existing_min))
                    # Preserve retrieveEG.m's literal logical-index behavior:
                    # IndexRelocated(:,s)>0 is a TRUE prefix of length a[s],
                    # so applying it to blockingContainers selects that prefix
                    # rather than the source positions stored in the matrix.
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
    # Verify capacity and legal execution in the actual top-to-bottom order.
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


@dataclass(frozen=True)
class _ExpectedPublicState:
    ranks: np.ndarray
    occupied: np.ndarray
    legal_mask: np.ndarray
    retrieval_count: int
    relocation_count: int
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float


def _expected_after_move(
    frame: PublicDecisionFrame, state: _PublicState, destination: int
) -> _ExpectedPublicState:
    ranks = state.ranks.copy()
    occupied = state.occupied.copy()
    source_height = state.heights[state.source]
    destination_height = state.heights[destination]
    moved_rank = int(ranks[state.source, source_height - 1])
    ranks[state.source, source_height - 1] = 0
    occupied[state.source, source_height - 1] = False
    ranks[destination, destination_height] = moved_rank
    occupied[destination, destination_height] = True
    heights = [height + int(stack == destination) - int(stack == state.source) for stack, height in enumerate(state.heights)]
    legal = np.asarray(
        [stack != state.source and heights[stack] < frame.max_tiers for stack in range(len(heights))],
        dtype=np.bool_,
    )
    return _ExpectedPublicState(
        ranks=ranks,
        occupied=occupied,
        legal_mask=legal,
        retrieval_count=int(frame.retrieval_count),
        relocation_count=int(frame.relocation_count) + 1,
        history_hits=tuple(frame.observation.history_hits),
        q_prior=float(frame.observation.q_prior),
        q_t=float(frame.observation.q_t),
    )


class EG20Actor:
    """Stateful group actor with public-state verification between moves.

    One instance belongs to one episode. An unfinished group is executed from
    its cached destination queue; changed rank/occupancy/legal/history state
    raises ``EGPlanMismatchError`` instead of silently replanning.
    """

    def __init__(self) -> None:
        self._plan: EGPlan | None = None
        self._next_move = 0
        self._expected: _ExpectedPublicState | None = None
        self.groups_planned = 0
        self.cached_moves = 0

    def reset(self) -> None:
        self._plan = None
        self._next_move = 0
        self._expected = None
        self.groups_planned = 0
        self.cached_moves = 0

    @staticmethod
    def _verify_expected(frame: PublicDecisionFrame, expected: _ExpectedPublicState) -> None:
        observation = frame.observation
        actual_ranks = np.asarray(observation.rank_tensor, dtype=np.int32)
        actual_occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
        actual_legal = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
        equal = (
            np.array_equal(actual_ranks, expected.ranks)
            and np.array_equal(actual_occupied, expected.occupied)
            and np.array_equal(actual_legal, expected.legal_mask)
            and tuple(int(item) for item in frame.legal_destinations)
            == tuple(int(item) for item in np.flatnonzero(expected.legal_mask))
            and int(frame.retrieval_count) == expected.retrieval_count
            and int(frame.relocation_count) == expected.relocation_count
            and tuple(observation.history_hits) == expected.history_hits
            and math.isclose(float(observation.q_prior), expected.q_prior, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(float(observation.q_t), expected.q_t, rel_tol=0.0, abs_tol=1e-15)
        )
        if not equal:
            raise EGPlanMismatchError("public state changed during an unfinished EG group plan")

    def select_destination(self, frame: PublicDecisionFrame) -> int:
        state = _public_state(frame)
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
            self._expected = _expected_after_move(frame, state, destination)
        return int(destination)


__all__ = [
    "EG20Actor",
    "EGPlan",
    "EGPlanMismatchError",
    "ERI_EG_AUTHOR_CODE_URLS",
    "ERI_EG_VERSION",
    "PUBLIC_UNKNOWN_RANK",
    "PublicFrameError",
    "build_eg20_plan",
    "select_eri20_destination",
]
