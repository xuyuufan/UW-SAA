"""Accepted GP rule evaluation on the detached public state."""
from __future__ import annotations
import math
from typing import Sequence
from environment.online_crp import OnlineCRPPublicState
from baseline.gp_expression import Node, evaluate_postfix

CRANE_TRAVEL_TIME=1.2
PICK_DROP_TIME=30.0

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
