"""Leakage-safe GP policy evaluation and a Numba-accelerated OCRP kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .online_crp_oracle import (
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    OnlineCRPPublicState,
)

from .expression import FUNCTION_COUNT, Node, evaluate_postfix

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in minimal environments
    def njit(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        def decorate(function):  # type: ignore[no-untyped-def]
            return function
        return decorate


MAX_STACKS = 10
MAX_TIERS = 12
MAX_CONTAINERS = 100
MAX_RELOCATIONS = 100_000
CRANE_TRAVEL_TIME = 1.2
PICK_DROP_TIME = 30.0


@dataclass(frozen=True)
class PackedEpisodes:
    layouts: np.ndarray
    initial_heights: np.ndarray
    num_stacks: np.ndarray
    max_tiers: np.ndarray
    num_containers: np.ndarray
    canonical_ids: tuple[str, ...]
    parameter_groups: tuple[str, ...]

    @property
    def episode_count(self) -> int:
        return int(self.num_stacks.shape[0])


@dataclass(frozen=True)
class KernelTraceDecision:
    state: OnlineCRPPublicState
    legal_destinations: tuple[int, ...]
    selected_destination: int


@dataclass(frozen=True)
class KernelTrace:
    decisions: tuple[KernelTraceDecision, ...]
    final_state: OnlineCRPPublicState


def pack_episodes(episodes: Sequence[OnlineCRPEpisode]) -> PackedEpisodes:
    count = len(episodes)
    layouts = np.zeros((count, MAX_STACKS, MAX_TIERS), dtype=np.int16)
    heights = np.zeros((count, MAX_STACKS), dtype=np.int16)
    num_stacks = np.zeros(count, dtype=np.int16)
    max_tiers = np.zeros(count, dtype=np.int16)
    num_containers = np.zeros(count, dtype=np.int16)
    ids: list[str] = []
    groups: list[str] = []
    for episode_index, episode in enumerate(episodes):
        if episode.num_stacks > MAX_STACKS or episode.max_tiers > MAX_TIERS:
            raise ValueError("episode exceeds the fixed Caserta kernel dimensions")
        if episode.num_containers > MAX_CONTAINERS:
            raise ValueError("episode exceeds the fixed Caserta container limit")
        if episode.retrieval_sequence != tuple(range(1, episode.num_containers + 1)):
            raise ValueError("packed GP episodes require priorities 1..N")
        num_stacks[episode_index] = episode.num_stacks
        max_tiers[episode_index] = episode.max_tiers
        num_containers[episode_index] = episode.num_containers
        for stack_index, stack in enumerate(episode.initial_stacks):
            heights[episode_index, stack_index] = len(stack)
            layouts[episode_index, stack_index, : len(stack)] = stack
        ids.append(episode.canonical_id)
        groups.append(episode.parameter_group)
    return PackedEpisodes(
        layouts=layouts,
        initial_heights=heights,
        num_stacks=num_stacks,
        max_tiers=max_tiers,
        num_containers=num_containers,
        canonical_ids=tuple(ids),
        parameter_groups=tuple(groups),
    )


def _known(stack: Sequence[int]) -> list[int]:
    return [int(value) for value in stack if int(value) > 0]


def terminal_features(
    state: OnlineCRPPublicState,
    destination: int,
    *,
    max_tiers: int,
    crane_travel_time: float = CRANE_TRAVEL_TIME,
    pick_drop_time: float = PICK_DROP_TIME,
) -> tuple[float, ...]:
    """Calculate the 17 paper terminals using public information only."""

    source_candidates = [
        index for index, stack in enumerate(state.stacks) if 1 in stack
    ]
    if len(source_candidates) != 1:
        raise ValueError("public state must contain exactly one current target")
    source = source_candidates[0]
    if not state.stacks[source] or state.stacks[source][-1] == 1:
        raise ValueError("terminal features require a blocked current target")
    if destination == source or destination < 0 or destination >= len(state.stacks):
        raise ValueError("destination must be a non-source stack")
    destination_stack = state.stacks[destination]
    if len(destination_stack) >= max_tiers:
        raise ValueError("destination stack is full")

    blocker = int(state.stacks[source][-1])
    known = _known(destination_stack)
    minimum = min(known) if known else 0
    average = float(sum(known) / len(known)) if known else 0.0
    reshuffle_index = (
        sum(value < blocker for value in known) if blocker > 0 else 0
    )
    difference = float(minimum - blocker) if minimum and blocker else 0.0

    source_position = source + 1
    destination_position = destination + 1
    duration = abs(source_position - destination_position) * crane_travel_time
    if state.crane_position is not None:
        duration += abs(state.crane_position - source_position) * crane_travel_time
    duration += pick_drop_time

    # NEXT denotes the target immediately after the current target (relative ID 2).
    next_positions = [
        index for index, value in enumerate(destination_stack) if value == 2
    ]
    next_present = int(bool(next_positions))
    remaining = (
        len(destination_stack) - 1 - next_positions[0] if next_positions else 0
    )

    well_located = 0
    non_located = 0
    for position, raw_value in enumerate(destination_stack):
        value = int(raw_value)
        if value <= 0:
            continue
        larger_above = any(
            int(above) > value
            for above in destination_stack[position + 1 :]
            if int(above) > 0
        )
        if larger_above:
            non_located += 1
        else:
            well_located += 1

    dsm = 0
    if blocker > 0:
        for position, raw_value in enumerate(destination_stack):
            value = int(raw_value)
            if 0 < value < blocker:
                dsm = position
                break
    unknown = sum(int(value) == 0 for value in destination_stack)
    known_count = len(known)
    unknown_percentage = unknown / len(destination_stack) if destination_stack else 0.0
    known_unknown_ratio = known_count / unknown if unknown else float(known_count)

    return (
        float(len(destination_stack)), average, float(max_tiers - len(destination_stack)),
        float(blocker), float(reshuffle_index), difference,
        float(duration), float(minimum), float(remaining), float(next_present),
        float(not destination_stack), float(well_located), float(non_located), float(dsm),
        float(unknown), float(unknown_percentage), float(known_unknown_ratio),
    )


def select_destination(
    tree: Node,
    state: OnlineCRPPublicState,
    legal_destinations: Sequence[int],
    *,
    max_tiers: int,
) -> int:
    if not legal_destinations:
        raise ValueError("GP policy requires at least one legal destination")
    postfix = tree.postfix()
    priorities = [
        evaluate_postfix(
            postfix,
            terminal_features(state, destination, max_tiers=max_tiers),
        )
        for destination in legal_destinations
    ]
    selected_index = min(
        range(len(legal_destinations)),
        key=lambda index: (priorities[index], legal_destinations[index]),
    )
    return int(legal_destinations[selected_index])


def run_tree(
    tree: Node,
    episode: OnlineCRPEpisode,
    *,
    horizon: int,
    return_actions: bool = False,
) -> int | tuple[int, tuple[int, ...]]:
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=horizon)
    actions: list[int] = []
    while not environment.state.terminated:
        destination = select_destination(
            tree,
            environment.state,
            environment.legal_destinations(),
            max_tiers=episode.max_tiers,
        )
        actions.append(destination)
        environment.step(destination)
    if return_actions:
        return environment.relocation_count, tuple(actions)
    return environment.relocation_count


@njit(cache=True)
def _visible_value(container: int, target: int, horizon: int) -> int:
    if target <= container <= target + horizon:
        return container - target + 1
    return 0


@njit(cache=True)
def _evaluate_postfix_numba(tokens: np.ndarray, features: np.ndarray) -> float:
    stack = np.empty(128, dtype=np.float64)
    pointer = 0
    for position in range(tokens.shape[0]):
        token = int(tokens[position])
        if token >= FUNCTION_COUNT:
            stack[pointer] = features[token - FUNCTION_COUNT]
            pointer += 1
            continue
        right = stack[pointer - 1]
        left = stack[pointer - 2]
        pointer -= 2
        if token == 0:
            value = left + right
        elif token == 1:
            value = left - right
        elif token == 2:
            value = left * right
        elif abs(right) < 1e-9:
            value = 1.0
        else:
            value = left / right
        if not math.isfinite(value):
            value = 1e12
        elif value > 1e12:
            value = 1e12
        elif value < -1e12:
            value = -1e12
        stack[pointer] = value
        pointer += 1
    return stack[0]


@njit(cache=True)
def _destination_features(
    yard: np.ndarray,
    heights: np.ndarray,
    source: int,
    destination: int,
    target: int,
    horizon: int,
    max_tier: int,
    crane_position: int,
) -> np.ndarray:
    features = np.zeros(17, dtype=np.float64)
    height = int(heights[destination])
    blocker = _visible_value(
        int(yard[source, int(heights[source]) - 1]), target, horizon
    )
    known_count = 0
    unknown_count = 0
    known_sum = 0.0
    minimum = 0
    reshuffle_index = 0
    next_position = -1
    for position in range(height):
        value = _visible_value(int(yard[destination, position]), target, horizon)
        if value == 0:
            unknown_count += 1
        else:
            known_count += 1
            known_sum += value
            if minimum == 0 or value < minimum:
                minimum = value
            if blocker > 0 and value < blocker:
                reshuffle_index += 1
            if value == 2:
                next_position = position

    well_located = 0
    non_located = 0
    for position in range(height):
        value = _visible_value(int(yard[destination, position]), target, horizon)
        if value == 0:
            continue
        larger_above = False
        for above_position in range(position + 1, height):
            above = _visible_value(
                int(yard[destination, above_position]), target, horizon
            )
            if above > value:
                larger_above = True
                break
        if larger_above:
            non_located += 1
        else:
            well_located += 1

    dsm = 0
    if blocker > 0:
        for position in range(height):
            value = _visible_value(int(yard[destination, position]), target, horizon)
            if 0 < value < blocker:
                dsm = position
                break

    source_position = source + 1
    destination_position = destination + 1
    duration = abs(source_position - destination_position) * CRANE_TRAVEL_TIME
    if crane_position >= 0:
        duration += abs(crane_position - source_position) * CRANE_TRAVEL_TIME
    duration += PICK_DROP_TIME

    features[0] = height
    features[1] = known_sum / known_count if known_count else 0.0
    features[2] = max_tier - height
    features[3] = blocker
    features[4] = reshuffle_index if blocker > 0 else 0.0
    features[5] = minimum - blocker if minimum and blocker else 0.0
    features[6] = duration
    features[7] = minimum
    features[8] = height - 1 - next_position if next_position >= 0 else 0.0
    features[9] = 1.0 if next_position >= 0 else 0.0
    features[10] = 1.0 if height == 0 else 0.0
    features[11] = well_located
    features[12] = non_located
    features[13] = dsm
    features[14] = unknown_count
    features[15] = unknown_count / height if height else 0.0
    features[16] = known_count / unknown_count if unknown_count else known_count
    return features


@njit(cache=True)
def _simulate_one(
    tokens: np.ndarray,
    initial_layout: np.ndarray,
    initial_heights: np.ndarray,
    stack_count: int,
    max_tier: int,
    container_count: int,
    horizon: int,
) -> int:
    yard = np.zeros((MAX_STACKS, MAX_TIERS), dtype=np.int16)
    heights = np.zeros(MAX_STACKS, dtype=np.int16)
    locations = np.full(MAX_CONTAINERS + 1, -1, dtype=np.int16)
    for stack in range(stack_count):
        heights[stack] = initial_heights[stack]
        for position in range(int(heights[stack])):
            container = int(initial_layout[stack, position])
            yard[stack, position] = container
            locations[container] = stack

    target = 1
    relocations = 0
    crane_position = -1
    while target <= container_count:
        while target <= container_count:
            source = int(locations[target])
            if int(yard[source, int(heights[source]) - 1]) != target:
                break
            heights[source] -= 1
            locations[target] = -1
            crane_position = 0
            target += 1
        if target > container_count:
            break

        if relocations >= MAX_RELOCATIONS:
            return 2_000_000_000

        source = int(locations[target])
        best_destination = -1
        best_priority = 1.7976931348623157e308
        for destination in range(stack_count):
            if destination == source or int(heights[destination]) >= max_tier:
                continue
            features = _destination_features(
                yard, heights, source, destination, target, horizon,
                max_tier, crane_position,
            )
            priority = _evaluate_postfix_numba(tokens, features)
            if priority < best_priority:
                best_priority = priority
                best_destination = destination
        if best_destination < 0:
            return 2_000_000_000
        source_height = int(heights[source]) - 1
        blocker = int(yard[source, source_height])
        heights[source] -= 1
        destination_height = int(heights[best_destination])
        yard[best_destination, destination_height] = blocker
        heights[best_destination] += 1
        locations[blocker] = best_destination
        crane_position = best_destination + 1
        relocations += 1
    return relocations


@njit(cache=True)
def _trace_one(
    tokens: np.ndarray,
    initial_layout: np.ndarray,
    initial_heights: np.ndarray,
    stack_count: int,
    max_tier: int,
    container_count: int,
    horizon: int,
    max_steps: int,
):
    """Instrumented version of the accelerated kernel used by differential tests."""

    yard = np.zeros((MAX_STACKS, MAX_TIERS), dtype=np.int16)
    heights = np.zeros(MAX_STACKS, dtype=np.int16)
    locations = np.full(MAX_CONTAINERS + 1, -1, dtype=np.int16)
    for stack in range(stack_count):
        heights[stack] = initial_heights[stack]
        for position in range(int(heights[stack])):
            container = int(initial_layout[stack, position])
            yard[stack, position] = container
            locations[container] = stack

    visible_states = np.zeros(
        (max_steps, MAX_STACKS, MAX_TIERS), dtype=np.int16
    )
    trace_heights = np.zeros((max_steps, MAX_STACKS), dtype=np.int16)
    legal_masks = np.zeros((max_steps, MAX_STACKS), dtype=np.uint8)
    retrieval_counts = np.zeros(max_steps, dtype=np.int16)
    relocation_counts = np.zeros(max_steps, dtype=np.int32)
    crane_positions = np.full(max_steps, -1, dtype=np.int16)
    crane_times = np.zeros(max_steps, dtype=np.float64)
    actions = np.full(max_steps, -1, dtype=np.int16)

    target = 1
    relocations = 0
    crane_position = -1
    crane_time = 0.0
    step = 0
    while target <= container_count:
        while target <= container_count:
            source = int(locations[target])
            if int(yard[source, int(heights[source]) - 1]) != target:
                break
            source_position = source + 1
            if crane_position >= 0:
                crane_time += abs(crane_position - source_position) * CRANE_TRAVEL_TIME
            crane_time += source_position * CRANE_TRAVEL_TIME + PICK_DROP_TIME
            heights[source] -= 1
            locations[target] = -1
            crane_position = 0
            target += 1
        if target > container_count:
            break
        if step >= max_steps:
            return (
                visible_states, trace_heights, legal_masks, retrieval_counts,
                relocation_counts, crane_positions, crane_times, actions,
                -1, relocations, crane_position, crane_time, heights,
            )

        source = int(locations[target])
        for stack in range(stack_count):
            height = int(heights[stack])
            trace_heights[step, stack] = height
            for position in range(height):
                visible_states[step, stack, position] = _visible_value(
                    int(yard[stack, position]), target, horizon
                )
            if stack != source and height < max_tier:
                legal_masks[step, stack] = 1
        retrieval_counts[step] = target - 1
        relocation_counts[step] = relocations
        crane_positions[step] = crane_position
        crane_times[step] = crane_time

        best_destination = -1
        best_priority = 1.7976931348623157e308
        for destination in range(stack_count):
            if legal_masks[step, destination] == 0:
                continue
            features = _destination_features(
                yard, heights, source, destination, target, horizon,
                max_tier, crane_position,
            )
            priority = _evaluate_postfix_numba(tokens, features)
            if priority < best_priority:
                best_priority = priority
                best_destination = destination
        if best_destination < 0:
            return (
                visible_states, trace_heights, legal_masks, retrieval_counts,
                relocation_counts, crane_positions, crane_times, actions,
                -2, relocations, crane_position, crane_time, heights,
            )
        actions[step] = best_destination

        source_position = source + 1
        destination_position = best_destination + 1
        if crane_position >= 0:
            crane_time += abs(crane_position - source_position) * CRANE_TRAVEL_TIME
        crane_time += (
            abs(source_position - destination_position) * CRANE_TRAVEL_TIME
            + PICK_DROP_TIME
        )
        source_height = int(heights[source]) - 1
        blocker = int(yard[source, source_height])
        heights[source] -= 1
        destination_height = int(heights[best_destination])
        yard[best_destination, destination_height] = blocker
        heights[best_destination] += 1
        locations[blocker] = best_destination
        crane_position = destination_position
        relocations += 1
        step += 1

    return (
        visible_states, trace_heights, legal_masks, retrieval_counts,
        relocation_counts, crane_positions, crane_times, actions,
        step, relocations, crane_position, crane_time, heights,
    )


@njit(cache=True)
def _simulate_all_numba(
    tokens: np.ndarray,
    layouts: np.ndarray,
    initial_heights: np.ndarray,
    num_stacks: np.ndarray,
    max_tiers: np.ndarray,
    num_containers: np.ndarray,
    horizon: int,
) -> np.ndarray:
    results = np.empty(num_stacks.shape[0], dtype=np.int32)
    for episode in range(num_stacks.shape[0]):
        results[episode] = _simulate_one(
            tokens,
            layouts[episode],
            initial_heights[episode],
            int(num_stacks[episode]),
            int(max_tiers[episode]),
            int(num_containers[episode]),
            horizon,
        )
    return results


def simulate_postfix(
    postfix: np.ndarray, packed: PackedEpisodes, *, horizon: int
) -> np.ndarray:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    tokens = np.ascontiguousarray(postfix, dtype=np.int16)
    if tokens.size > 127:
        raise ValueError("expression exceeds evaluator stack capacity")
    return _simulate_all_numba(
        tokens,
        packed.layouts,
        packed.initial_heights,
        packed.num_stacks,
        packed.max_tiers,
        packed.num_containers,
        horizon,
    )


def fitness(tree: Node, packed: PackedEpisodes, *, horizon: int) -> int:
    values = simulate_postfix(tree.postfix(), packed, horizon=horizon)
    return int(values.astype(np.int64).sum())


def trace_tree(
    tree: Node,
    episode: OnlineCRPEpisode,
    *,
    horizon: int,
    max_steps: int = MAX_RELOCATIONS,
) -> KernelTrace:
    """Return accelerated-kernel decision states for semantic differential tests."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    packed = pack_episodes((episode,))
    result = _trace_one(
        np.ascontiguousarray(tree.postfix(), dtype=np.int16),
        packed.layouts[0],
        packed.initial_heights[0],
        int(packed.num_stacks[0]),
        int(packed.max_tiers[0]),
        int(packed.num_containers[0]),
        horizon,
        max_steps,
    )
    (
        visible_states,
        trace_heights,
        legal_masks,
        retrieval_counts,
        relocation_counts,
        crane_positions,
        crane_times,
        actions,
        count,
        relocations,
        final_crane_position,
        final_crane_time,
        final_heights,
    ) = result
    if count < 0:
        raise RuntimeError("accelerated trace reached its safety limit or had no legal action")

    decisions: list[KernelTraceDecision] = []
    for step in range(int(count)):
        stacks = tuple(
            tuple(
                int(visible_states[step, stack, position])
                for position in range(int(trace_heights[step, stack]))
            )
            for stack in range(episode.num_stacks)
        )
        crane_position = int(crane_positions[step])
        state = OnlineCRPPublicState(
            stacks=stacks,
            current_target_id=1,
            retrieval_count=int(retrieval_counts[step]),
            relocation_count=int(relocation_counts[step]),
            crane_position=None if crane_position < 0 else crane_position,
            crane_time=float(crane_times[step]),
            terminated=False,
        )
        legal = tuple(
            stack
            for stack in range(episode.num_stacks)
            if int(legal_masks[step, stack]) == 1
        )
        decisions.append(
            KernelTraceDecision(state, legal, int(actions[step]))
        )

    if any(int(final_heights[stack]) != 0 for stack in range(episode.num_stacks)):
        raise RuntimeError("accelerated trace terminated with containers remaining")
    final_stacks = tuple(() for _ in range(episode.num_stacks))
    final_position = int(final_crane_position)
    final_state = OnlineCRPPublicState(
        stacks=final_stacks,
        current_target_id=None,
        retrieval_count=episode.num_containers,
        relocation_count=int(relocations),
        crane_position=None if final_position < 0 else final_position,
        crane_time=float(final_crane_time),
        terminated=True,
    )
    return KernelTrace(tuple(decisions), final_state)


__all__ = [
    "CRANE_TRAVEL_TIME", "KernelTrace", "KernelTraceDecision", "MAX_CONTAINERS",
    "MAX_RELOCATIONS", "MAX_STACKS", "MAX_TIERS", "PICK_DROP_TIME",
    "PackedEpisodes", "fitness", "pack_episodes", "run_tree",
    "select_destination", "simulate_postfix", "terminal_features", "trace_tree",
]
