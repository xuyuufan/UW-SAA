"""Paper-faithful offline Min–Max destination rule and solver."""

from __future__ import annotations

from typing import Sequence

from R.offline_brp_common import Selection, run_offline


def min_el_i(stack: Sequence[int], n: int) -> int:
    """Port of the author's min_el_i helper; empty p(i)=N+1 per the paper."""
    return min(stack) if stack else n + 1


def max_in_choosestack(choosestack: Sequence[int], blocker: int) -> int:
    """Port of heuristic.cpp:139–165; first stack index wins equal scores."""
    max_value = -1
    max_position = -1
    min_greater = float("inf")
    min_position = -1
    for index, value in enumerate(choosestack):
        if value > blocker and value < min_greater:
            min_greater = value
            min_position = index
        if value > max_value:
            max_value = value
            max_position = index
    if min_position >= 0:
        return min_position
    if max_position >= 0:
        return max_position
    raise ValueError("Minmax scoring needs at least one legal destination")


def choose_minmax_destination(
    bay: Sequence[Sequence[int]],
    blocker: int,
    legal: Sequence[int],
    n: int,
    *,
    excluded: Sequence[int] = (),
) -> tuple[int, int] | None:
    """Return (stack, d=p(i)-r) using the ported author score core.

    This follows the paper's Eq. (11): empty stacks enter the same min-safe
    comparison as p=N+1. The source wrapper's separate empty-first shortcut is
    intentionally not used.
    """
    excluded_set = set(excluded)
    candidates = [index for index in legal if index not in excluded_set]
    if not candidates:
        return None

    # The C++ core scans all stacks, with zero marking an ineligible stack.
    # Every p(i) is positive, so ineligible zeros cannot win either branch.
    scores = [0] * len(bay)
    for index in candidates:
        scores[index] = min_el_i(bay[index], n)
    chosen = max_in_choosestack(scores, blocker)
    return chosen, scores[chosen] - blocker


def _selector(bay, target, source, blocker, legal, n) -> Selection:
    del target, source
    choice = choose_minmax_destination(bay, blocker, legal, n)
    if choice is None:
        raise ValueError("Minmax needs at least one legal destination")
    destination, _ = choice
    return Selection(
        destination=destination,
        rule="minmax_eq11_empty_n_plus_1_then_min_index",
    )


def solve_offline(stacks, max_tiers):
    """Solve a complete-rank restricted instance using paper Eq. (11)."""
    return run_offline(stacks, max_tiers, _selector)
