"""Public-only expected frame updates for committed blocker groups."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import numpy as np
from baseline.eri_eg_public import EGPlanMismatchError

@dataclass(frozen=True)
class PublicSnapshot:
    ranks: np.ndarray
    occupied: np.ndarray
    legal_mask: np.ndarray
    legal_destinations: tuple[int, ...]
    max_tiers: int
    retrieval_count: int
    relocation_count: int
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    crane_position: int | None
    crane_time: float

def _snapshot(frame: Any) -> PublicSnapshot:
    obs = frame.observation
    return PublicSnapshot(
        ranks=np.asarray(obs.rank_tensor, dtype=np.int32).copy(),
        occupied=np.asarray(obs.occupied_mask, dtype=np.bool_).copy(),
        legal_mask=np.asarray(obs.legal_destination_mask, dtype=np.bool_).copy(),
        legal_destinations=tuple(int(value) for value in frame.legal_destinations),
        max_tiers=int(frame.max_tiers),
        retrieval_count=int(frame.retrieval_count),
        relocation_count=int(frame.relocation_count),
        history_hits=tuple(bool(value) for value in obs.history_hits),
        q_prior=float(obs.q_prior),
        q_t=float(obs.q_t),
        crane_position=(None if frame.crane_position is None else int(frame.crane_position)),
        crane_time=float(frame.crane_time),
    )

def _after_move(
    frame: Any,
    destination: int,
    *,
    crane_travel_time: float,
    pick_drop_time: float,
) -> PublicSnapshot:
    before = _snapshot(frame)
    target_locations = np.argwhere(before.occupied & (before.ranks == 1))
    if target_locations.shape != (1, 2):
        raise ValueError("EG cache requires exactly one public current target")
    source = int(target_locations[0, 0])
    heights = before.occupied.sum(axis=1).astype(int).tolist()
    if destination not in before.legal_destinations:
        raise ValueError("R-EG plan emitted a destination outside the current legal mask")
    src_height = heights[source]
    dst_height = heights[destination]
    ranks = before.ranks.copy()
    occupied = before.occupied.copy()
    ranks[source, src_height - 1] = 0
    occupied[source, src_height - 1] = False
    ranks[destination, dst_height] = int(frame.observation.rank_tensor[source, src_height - 1])
    occupied[destination, dst_height] = True
    heights[source] -= 1
    heights[destination] += 1
    next_legal_mask = np.asarray(
        [stack != source and heights[stack] < before.max_tiers for stack in range(len(heights))],
        dtype=np.bool_,
    )

    source_position = source + 1
    destination_position = int(destination) + 1
    travel = float(crane_travel_time)
    crane_time = before.crane_time
    if before.crane_position is not None:
        crane_time += abs(before.crane_position - source_position) * travel
    crane_time += abs(source_position - destination_position) * travel + float(pick_drop_time)
    return PublicSnapshot(
        ranks=ranks,
        occupied=occupied,
        legal_mask=next_legal_mask,
        legal_destinations=tuple(int(value) for value in np.flatnonzero(next_legal_mask)),
        max_tiers=before.max_tiers,
        retrieval_count=before.retrieval_count,
        relocation_count=before.relocation_count + 1,
        history_hits=before.history_hits,
        q_prior=before.q_prior,
        q_t=before.q_t,
        crane_position=destination_position,
        crane_time=crane_time,
    )

def _assert_snapshot(frame: Any, expected: PublicSnapshot) -> None:
    actual = _snapshot(frame)
    if not (
        np.array_equal(actual.ranks, expected.ranks)
        and np.array_equal(actual.occupied, expected.occupied)
        and np.array_equal(actual.legal_mask, expected.legal_mask)
        and actual.legal_destinations == expected.legal_destinations
        and actual.max_tiers == expected.max_tiers
        and actual.retrieval_count == expected.retrieval_count
        and actual.relocation_count == expected.relocation_count
        and actual.history_hits == expected.history_hits
        and math.isclose(actual.q_prior, expected.q_prior, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(actual.q_t, expected.q_t, rel_tol=0.0, abs_tol=1e-15)
        and actual.crane_position == expected.crane_position
        and math.isclose(actual.crane_time, expected.crane_time, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise EGPlanMismatchError("full H20 public frame changed during a committed EG group")
