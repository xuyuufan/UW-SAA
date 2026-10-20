"""Offline Chain heuristic: one physical Min–Max/Chain move per decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from baseline.minmax_core import choose_minmax_destination
from baseline.common import (
    Selection,
    legal_destinations,
    run_offline,
    stack_containing,
)


@dataclass(frozen=True)
class NextBlocker:
    target_rank: int
    blocker_rank: int
    source: int
    stacks: tuple[tuple[int, ...], ...]
    auto_retrieved: tuple[int, ...]
    skipped_ranks: tuple[int, ...]


def eq11_reverse_condition(d1: int, d2: int, r1: int, r2: int) -> bool:
    """Eq. (11) plus its three explanatory bullets; all conditions are AND."""
    return r2 < d1 and r2 > 0 and d1 > 0 and r1 * d2 > 0


def _preview_after_change(
    stacks: Sequence[Sequence[int]],
    target_rank: int,
    blocker_rank: int,
    source: int,
    n: int,
    destination: int | None,
) -> NextBlocker | None:
    bay = [list(stack) for stack in stacks]
    if source < 0 or source >= len(bay) or not bay[source] or bay[source][-1] != blocker_rank:
        raise ValueError("the current blocker must be the top item of its source")
    bay[source].pop()
    if destination is not None:
        if destination < 0 or destination >= len(bay) or destination == source:
            raise ValueError("virtual move needs a distinct in-range destination")
        bay[destination].append(blocker_rank)

    retrieved: list[int] = []
    skipped: list[int] = []
    for target in range(target_rank, n + 1):
        if destination is None and target == blocker_rank:
            # In the reverse virtual path rn is removed from its source. Skip its
            # retrieval rank rather than trying to locate the intentionally absent box.
            skipped.append(target)
            continue
        target_stack = stack_containing(bay, target)
        if bay[target_stack][-1] != target:
            return NextBlocker(
                target_rank=target,
                blocker_rank=bay[target_stack][-1],
                source=target_stack,
                stacks=tuple(tuple(stack) for stack in bay),
                auto_retrieved=tuple(retrieved),
                skipped_ranks=tuple(skipped),
            )
        bay[target_stack].pop()
        retrieved.append(target)
    return None


def preview_next_blocker_after_move(
    stacks: Sequence[Sequence[int]],
    target_rank: int,
    blocker_rank: int,
    source: int,
    destination: int,
    n: int,
) -> NextBlocker | None:
    """Apply one virtual direct move, then retrieve accessible targets in rank order."""
    return _preview_after_change(
        stacks, target_rank, blocker_rank, source, n, destination
    )


def preview_next_blocker_after_removal(
    stacks: Sequence[Sequence[int]],
    target_rank: int,
    blocker_rank: int,
    source: int,
    n: int,
) -> NextBlocker | None:
    """Remove the current blocker virtually, auto-retrieve, and find the next blocker."""
    return _preview_after_change(stacks, target_rank, blocker_rank, source, n, None)


def _fallback(destination: int, reason: str) -> Selection:
    return Selection(destination, "minmax_fallback", reason)


def _selector(bay, target, source, blocker, legal, n, max_tiers) -> Selection:
    direct = choose_minmax_destination(bay, blocker, legal, n)
    if direct is None:
        raise ValueError("Chain needs one legal direct Min–Max destination")
    direct_destination, d1 = direct

    direct_next = preview_next_blocker_after_move(
        bay, target, blocker, source, direct_destination, n
    )
    if direct_next is None:
        return _fallback(direct_destination, "no_next_blocker")
    if direct_next.blocker_rank == blocker:
        return _fallback(direct_destination, "repeated_current_blocker")

    # The paper's prose after Eq. (11) says the reverse-order rule can apply
    # only when p(r_n) < p(r_{n+1}); its pseudocode has the same outer guard.
    # Here p(r) is the complete offline retrieval rank of the blocker.
    if blocker >= direct_next.blocker_rank:
        return _fallback(direct_destination, "rank_order_precondition_failed")

    reverse_next = preview_next_blocker_after_removal(
        bay, target, blocker, source, n
    )
    if reverse_next is None or reverse_next.blocker_rank != direct_next.blocker_rank:
        return _fallback(direct_destination, "next_blocker_identity_mismatch")

    next_rank = direct_next.blocker_rank
    next_source_direct = stack_containing(direct_next.stacks, next_rank)
    direct_legal = legal_destinations(direct_next.stacks, next_source_direct, max_tiers)
    d_choice = choose_minmax_destination(
        direct_next.stacks, next_rank, direct_legal, n
    )
    if d_choice is None:
        return _fallback(direct_destination, "no_direct_next_destination")
    _, d2 = d_choice

    next_source_now = stack_containing(bay, next_rank)
    next_legal_now = legal_destinations(bay, next_source_now, max_tiers)
    r_choice = choose_minmax_destination(bay, next_rank, next_legal_now, n)
    if r_choice is None:
        return _fallback(direct_destination, "no_reverse_next_destination")
    reverse_next_destination, r2 = r_choice

    reverse_first_for_blocker = choose_minmax_destination(
        bay, blocker, legal, n, excluded=(reverse_next_destination,)
    )
    if reverse_first_for_blocker is None:
        return _fallback(direct_destination, "no_reverse_first_destination")
    _, r1 = reverse_first_for_blocker

    if not eq11_reverse_condition(d1, d2, r1, r2):
        return Selection(direct_destination, "chain_eq11_direct")
    if reverse_next_destination not in legal:
        return _fallback(direct_destination, "eq11_destination_illegal_for_current_move")
    return Selection(reverse_next_destination, "chain_eq11_reverse_destination")


def solve_offline(stacks, max_tiers):
    """Solve a complete-rank restricted instance with the Chain heuristic."""
    # A local closure binds capacity while preserving the shared runner contract.
    def selector(bay, target, source, blocker, legal, n):
        return _selector(bay, target, source, blocker, legal, n, max_tiers)

    return run_offline(stacks, max_tiers, selector)
