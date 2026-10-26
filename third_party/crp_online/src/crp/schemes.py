"""Relocation schemes: the fixed loop that asks a priority function where to move.

Mirrors cpp_reference/schemes/UnrestrictedAutomaticHeuristic.cpp (Algorithm 2).
solve_unrestricted is the readable reference; solve_unrestricted_fast is the one GP uses.
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Callable, Literal, Union

from crp.terminals import compute_terminals
from crp.yard import INT_MAX, Yard

PriorityFunction = Callable[[dict[str, int]], float]  # terminals -> score, lowest wins


def solve_unrestricted(yard: Yard, priority: PriorityFunction) -> int:
    """Retrieve every container; return the number of relocations."""
    while not yard.is_done():
        src = yard.target_stack()
        while yard.stacks[src][-1] != yard.next_target:
            dst = _best_destination(yard, src, priority)
            # preventive moves: don't let the moved container block dst's top
            while yard.stacks[dst] and yard.top_value(dst) < yard.top_value(src):
                safe = _safe_stack(yard, src, dst)
                if safe is None:
                    break
                yard.relocate(dst, safe)
            yard.relocate(src, dst)
        yard.retrieve()
    return yard.relocations


def _best_destination(yard: Yard, src: int, priority: PriorityFunction) -> int:
    """Lowest score among non-full stacks other than src; leftmost wins ties."""
    best, best_score = -1, float("inf")
    for dst in range(len(yard.stacks)):
        if dst != src and not yard.is_full(dst):
            score = priority(compute_terminals(yard, src, dst))
            if score < best_score:  # strict <, so the leftmost tie stays
                best, best_score = dst, score
    return best


def _safe_stack(yard: Yard, src: int, dst: int) -> int | None:
    """Where to put dst's top so it isn't blocked: copy of the C++ search."""
    top = yard.top_value(dst)
    chosen, lowest = None, INT_MAX
    for s in range(len(yard.stacks)):
        if s in (src, dst) or yard.is_full(s):
            continue
        smallest = yard.min_value(s)
        if smallest > top:
            if lowest > smallest:
                chosen, lowest = s, smallest
            elif smallest == INT_MAX:  # empty stack: always taken, last one wins
                chosen, lowest = s, 0
    return chosen


# f(SH, EMP, AVG, CUR, RI, DIFF) -> score, lowest wins
FastPriority = Callable[[float, float, float, float, float, float], float]
# per stack: sorted visible values, SH, EMP, AVG, MIN
StackNumbers = tuple[list[int], float, float, float, int]


def solve_unrestricted_fast(yard: Yard, f: FastPriority) -> int:
    """Same decisions as solve_unrestricted, only faster (tests/test_fast.py checks this).

    Differences in how, not what:
    - each stack's terminal numbers are computed once and reused until that stack changes
    - the window (last visible ID) is computed once per target, not once per container
    - f gets the six numbers directly instead of a dict
    """
    stacks, n = yard.stacks, len(yard.stacks)
    others = [[d for d in range(n) if d != s] for s in range(n)]
    while not yard.is_done():
        src, target = yard.target_stack(), yard.next_target
        last = yard.last_visible()
        hidden = last + 1  # what every hidden container looks like (Maximum strategy)
        cache: list[Union[StackNumbers, Literal[False], None]] = [None] * n  # None = unknown, False = full
        while stacks[src][-1] != target:
            c = stacks[src][-1]
            cur = c if c <= last else hidden
            best, best_score = -1, float("inf")
            for d in others[src]:
                if cache[d] is None:
                    cache[d] = _stack_numbers(yard, d, last, hidden)
                if cache[d] is False:
                    continue
                values, sh, emp, avg, smallest = cache[d]
                ri = bisect_left(values, cur)  # how many values are smaller than cur
                score = f(sh, emp, avg, float(cur), float(ri), float(smallest - cur))
                if score < best_score:  # strict <, so the leftmost tie stays
                    best, best_score = d, score
            dst = best
            while stacks[dst] and yard.top_value(dst) < yard.top_value(src):
                safe = _safe_stack(yard, src, dst)
                if safe is None:
                    break
                yard.relocate(dst, safe)
                cache[dst] = cache[safe] = None
            yard.relocate(src, dst)
            cache[dst] = cache[src] = None
        yard.retrieve()
    return yard.relocations


def _stack_numbers(yard: Yard, d: int, last: int, hidden: int) -> Union[StackNumbers, Literal[False]]:
    """Sorted visible values, SH, EMP, AVG and MIN of stack d; False if it is full."""
    height = len(yard.stacks[d])
    if height >= yard.max_height:
        return False
    values = sorted([c if c <= last else hidden for c in yard.stacks[d]])
    avg = int(sum(values) / len(values)) if height else 0
    smallest = values[0] if values else INT_MAX
    return values, float(height), float(yard.max_height - height), float(avg), smallest
