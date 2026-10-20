"""Public C1 feature construction for the H20RL-R adaptation search."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np


HORIZONS = (0, 2, 5, 10, 20)
UNKNOWN_RANK = 22
Q_PRIOR = 5183.0 / 9920.0


@dataclass(frozen=True)
class C1Proposal:
    horizon: int
    destination: int


@dataclass(frozen=True)
class C1Candidate:
    destination: int
    requested_h: int
    provenance_horizons: tuple[int, ...]
    is_anchor: bool
    delta_features: tuple[float, ...]
    history_features: tuple[float, ...]
    no_history_features: tuple[float, ...]


@dataclass(frozen=True)
class C1CandidateSet:
    raw_proposals: tuple[C1Proposal, ...]
    candidates: tuple[C1Candidate, ...]
    anchor_destination: int


def _public_state(frame: object) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int, float]:
    observation = getattr(frame, "observation", None)
    if observation is None:
        raise ValueError("C1 requires a detached public decision frame")

    raw_ranks = np.asarray(getattr(observation, "rank_tensor", None))
    if raw_ranks.ndim != 2 or not np.issubdtype(raw_ranks.dtype, np.integer):
        raise ValueError("public rank tensor must be a two-dimensional integer array")
    ranks = raw_ranks.astype(np.int64, copy=False)
    occupied = np.asarray(getattr(observation, "occupied_mask", None), dtype=np.bool_)
    legal_mask = np.asarray(
        getattr(observation, "legal_destination_mask", None), dtype=np.bool_
    )
    tiers = int(getattr(frame, "max_tiers", -1))
    if ranks.shape != occupied.shape or ranks.ndim != 2:
        raise ValueError("public rank and occupancy shapes differ")
    if legal_mask.ndim != 1 or legal_mask.shape[0] != ranks.shape[0]:
        raise ValueError("public legal-action mask must match the stack dimension")
    if ranks.shape != (legal_mask.shape[0], tiers) or tiers <= 0:
        raise ValueError("public frame dimensions differ from its masks")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty padding must have rank zero")
    if np.any(occupied & ((ranks < 1) | (ranks > UNKNOWN_RANK))):
        raise ValueError("occupied ranks must be known ranks or sentinel 22")

    heights = occupied.sum(axis=1).astype(np.int64)
    expected_occupied = np.arange(tiers)[None, :] < heights[:, None]
    if not np.array_equal(occupied, expected_occupied):
        raise ValueError("occupied cells must form bottom-filled stack prefixes")
    known = ranks[occupied & (ranks < UNKNOWN_RANK)]
    if np.count_nonzero(known == 1) != 1 or len(np.unique(known)) != len(known):
        raise ValueError("public frame must have one current target and unique known ranks")

    source = int(np.argwhere(occupied & (ranks == 1))[0, 0])
    source_height = int(heights[source])
    if source_height < 2:
        raise ValueError("C1 features require a relocation decision with a source blocker")
    blocker = int(ranks[source, source_height - 1])
    if blocker == 1:
        raise ValueError("current target cannot be the moving blocker at a relocation decision")

    legal = tuple(int(value) for value in getattr(frame, "legal_destinations", ()))
    if legal != tuple(int(value) for value in np.flatnonzero(legal_mask)):
        raise ValueError("public legal destinations differ from the action mask")
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("C1 requires distinct legal destinations")
    if source in legal or any(heights[index] >= tiers for index in legal):
        raise ValueError("legal destinations must exclude source and full stacks")

    q_prior = float(getattr(observation, "q_prior", float("nan")))
    q_t = float(getattr(observation, "q_t", float("nan")))
    if not isfinite(q_prior) or abs(q_prior - Q_PRIOR) > 1e-15:
        raise ValueError("public q_prior differs from the frozen 5183/9920 value")
    if not isfinite(q_t) or not 0.0 <= q_t <= 1.0:
        raise ValueError("public q_t must be a finite probability")
    return ranks, occupied, legal, tiers, blocker, q_t


def c1_destination_relations(frame: object) -> dict[int, tuple[float, ...]]:
    """Return the six registered public relation features for each legal stack."""
    ranks, occupied, legal, tiers, blocker, _q_t = _public_state(frame)
    blocker_known = 2 <= blocker <= 21
    relations: dict[int, tuple[float, ...]] = {}
    for destination in legal:
        mask = occupied[destination]
        values = ranks[destination, mask]
        height = int(mask.sum())
        known_values = values[(values >= 2) & (values <= 21)]
        unknown_count = int(np.count_nonzero(values == UNKNOWN_RANK))
        minimum_known = int(known_values.min()) if len(known_values) else UNKNOWN_RANK
        safe_known = blocker_known and bool(
            np.all((values >= 2) & (values <= 21) & (values > blocker))
        )
        earlier_count = (
            int(np.count_nonzero((values >= 2) & (values < blocker)))
            if blocker_known
            else 0
        )
        denominator = max(height, 1)
        relations[destination] = (
            float(tiers - height) / tiers,
            float(height == 0),
            float(unknown_count) / denominator,
            float(minimum_known) / UNKNOWN_RANK,
            float(safe_known),
            float(earlier_count) / denominator,
        )
    return relations


def build_c1_candidate_set(
    frame: object, proposals_by_horizon: Mapping[int, int]
) -> C1CandidateSet:
    """Build C1 rows from five public H20 proposals without executing H20."""
    if set(proposals_by_horizon) != set(HORIZONS):
        raise ValueError(f"C1 proposals must cover exactly horizons {HORIZONS}")
    _ranks, _occupied, legal, _tiers, _blocker, q_t = _public_state(frame)
    proposals = tuple(
        C1Proposal(horizon=h, destination=int(proposals_by_horizon[h]))
        for h in HORIZONS
    )
    if any(proposal.destination not in legal for proposal in proposals):
        raise ValueError("every H20 horizon proposal must be a legal destination")

    anchor = int(proposals_by_horizon[20])
    relations = c1_destination_relations(frame)
    anchor_relation = relations[anchor]
    horizons_by_destination: dict[int, list[int]] = {}
    for proposal in proposals:
        horizons_by_destination.setdefault(proposal.destination, []).append(
            proposal.horizon
        )

    candidates = []
    for destination, source_horizons in horizons_by_destination.items():
        is_anchor = destination == anchor
        requested_h = 20 if is_anchor else min(source_horizons)
        delta = (
            (0.0,) * 8
            if is_anchor
            else (
                1.0,
                (requested_h - 20.0) / 20.0,
                *(left - right for left, right in zip(
                    relations[destination], anchor_relation, strict=True
                )),
            )
        )
        history_features = delta + tuple((1.0 - q_t) * value for value in delta)
        no_history_features = delta + tuple(
            (1.0 - Q_PRIOR) * value for value in delta
        )
        candidates.append(
            C1Candidate(
                destination=destination,
                requested_h=requested_h,
                provenance_horizons=tuple(source_horizons),
                is_anchor=is_anchor,
                delta_features=delta,
                history_features=history_features,
                no_history_features=no_history_features,
            )
        )
    candidates.sort(key=lambda row: (row.requested_h, row.destination))
    return C1CandidateSet(
        raw_proposals=proposals,
        candidates=tuple(candidates),
        anchor_destination=anchor,
    )


__all__ = (
    "C1Candidate",
    "C1CandidateSet",
    "C1Proposal",
    "HORIZONS",
    "Q_PRIOR",
    "UNKNOWN_RANK",
    "build_c1_candidate_set",
    "c1_destination_relations",
)
