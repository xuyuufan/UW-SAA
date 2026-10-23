"""Bounded loader for public Zhang author-instance layout and RPM fields.

The retrieval sequence is intentionally never accessed or returned here. An
outer evaluator may keep it private and pass only the currently observed target
ID to :meth:`AMCTSPolicy.select_state`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TypeAlias


Stacks: TypeAlias = tuple[tuple[int, ...], ...]
RPM: TypeAlias = tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ZhangAuthorInputs:
    """Only fields supplied to the policy: current layout, capacity, and RPM."""

    stacks: Stacks
    max_tiers: int
    rpm: RPM


def load_zhang_author_inputs(path: str | Path) -> ZhangAuthorInputs:
    """Read and validate public instance inputs without touching future order."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("author instance JSON must contain an object")
    for key in ("initial_layout", "max_tier", "retrieval_probability"):
        if key not in raw:
            raise ValueError(f"author instance is missing {key}")

    layout = raw["initial_layout"]
    if not isinstance(layout, list) or not layout:
        raise ValueError("initial_layout must be a nonempty list of stacks")
    stacks_list: list[tuple[int, ...]] = []
    flat_items: list[int] = []
    for stack in layout:
        if not isinstance(stack, list):
            raise ValueError("each initial_layout stack must be a list")
        values: list[int] = []
        for item in stack:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("container IDs must be integers")
            values.append(item)
            flat_items.append(item)
        stacks_list.append(tuple(values))
    if not flat_items or sorted(flat_items) != list(range(len(flat_items))):
        raise ValueError("container IDs must be unique and contiguous from zero")

    max_tiers = raw["max_tier"]
    if isinstance(max_tiers, bool) or not isinstance(max_tiers, int) or max_tiers <= 0:
        raise ValueError("max_tier must be a positive integer")
    if any(len(stack) > max_tiers for stack in stacks_list):
        raise ValueError("initial layout exceeds max_tier")

    rows = raw["retrieval_probability"]
    if not isinstance(rows, list) or len(rows) != len(flat_items) or not rows:
        raise ValueError("retrieval_probability must have one row per container")
    rpm_list: list[tuple[float, ...]] = []
    period_count: int | None = None
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError("each retrieval_probability row must be nonempty")
        values = tuple(float(value) for value in row)
        if period_count is None:
            period_count = len(values)
        if len(values) != period_count:
            raise ValueError("retrieval_probability rows must have equal length")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("retrieval_probability must be finite and nonnegative")
        if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("each retrieval_probability row must sum to one")
        rpm_list.append(values)

    return ZhangAuthorInputs(tuple(stacks_list), int(max_tiers), tuple(rpm_list))


__all__ = ["ZhangAuthorInputs", "load_zhang_author_inputs"]
