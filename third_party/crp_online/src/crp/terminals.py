"""Terminals: the numbers a priority function sees about one candidate stack.

Mirrors cpp_reference/gp/Terminals.cpp (CalculateTerminals) and the ContainerYard
helpers it calls (reshuffleIndex, GetMinimumStack, calculateAverageContainerID).
Terminal set C only: SH, EMP, AVG, CUR, RI, DIFF.
"""
from __future__ import annotations

from crp.yard import INT_MAX, Yard


def compute_terminals(yard: Yard, src: int, dst: int) -> dict[str, int]:
    """Terminal values for moving the top container of src onto dst."""
    cur = yard.top_value(src)
    values = [yard.value(c) for c in yard.stacks[dst]]
    height = len(yard.stacks[dst])
    smallest = min(values, default=INT_MAX)
    return {
        "SH": height,
        "EMP": yard.max_height - height,
        "AVG": int(sum(values) / len(values)) if height > 0 else 0,
        "CUR": cur,
        "RI": sum(1 for v in values if v < cur),
        "DIFF": smallest - cur,
    }
