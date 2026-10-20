"""Minimal public-frame helpers shared by the PLAN-052 online proposals.

This module accepts only the registered actor-visible frame. Horizon censoring
reuses the exact PLAN-051 prefix constructor and forecast namespace.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


HORIZONS = (0, 2, 5, 10, 20)
NEW_SOLVERS = ("RIL", "Chain", "Minmax")
NEW_POLICIES = (
    "OLD_FROZEN_R",
    "R_SOURCE_FULL",
    "R_UTILITY_RIDGE",
    "R_QUAD_H",
    "R_ORDER_MAIN_TAU0_FULL",
    "NATIVE20",
)
PUBLIC_UNKNOWN_RANK = 22


def public_prefix(frame: object, horizon: int) -> object:
    """Validate and censor a full public frame using PLAN-051's H rule."""
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    from R.scrp.r_five_versions_actor import require_public_frame
    from R.scrp.cross_solver_reliability_r import truncate_public_frame

    require_public_frame(frame)
    prefix = truncate_public_frame(frame, int(horizon))
    ranks = np.asarray(prefix.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(prefix.observation.occupied_mask, dtype=np.bool_)
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty cells must remain public rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > PUBLIC_UNKNOWN_RANK))):
        raise ValueError("occupied public ranks must be in 1..22")
    return prefix


def public_stacks(frame: object) -> tuple[tuple[int, ...], ...]:
    ranks = np.asarray(frame.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public rank and occupancy shapes differ")
    return tuple(
        tuple(int(value) for value in ranks[index, : int(occupied[index].sum())])
        for index in range(ranks.shape[0])
    )


def max_occupied_public_rank(stacks: Sequence[Sequence[int]]) -> int:
    values = [int(value) for stack in stacks for value in stack]
    if not values:
        raise ValueError("a nonterminal public frame cannot have no occupied boxes")
    return max(values)
