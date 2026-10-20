"""Detached public frame contract, validation, and horizon censoring."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
from typing import Any, Sequence
import numpy as np
from .noise_aware_forecast import HORIZONS, PublicObservation
from .online_crp import OnlineCRPPublicState
from .public_history import PublicHistorySummary, PublicObservationWithHistory

PUBLIC_UNKNOWN_RANK=22

@dataclass(frozen=True)
class PublicDecisionFrame:
    """Only public ranks, legality, and physical scalar context reach an actor."""

    observation: PublicObservation
    legal_destinations: tuple[int, ...]
    max_tiers: int
    crane_position: int | None
    crane_time: float
    retrieval_count: int
    relocation_count: int

def make_public_frame(
    observation: PublicObservation,
    public_state: OnlineCRPPublicState,
    legal_destinations: tuple[int, ...],
    *,
    max_tiers: int,
) -> PublicDecisionFrame:
    legal = tuple(int(value) for value in legal_destinations)
    mask_legal = tuple(
        int(value)
        for value in np.flatnonzero(np.asarray(observation.legal_destination_mask))
    )
    if legal != mask_legal or len(set(legal)) != len(legal):
        raise ValueError("public frame legality differs from the observation mask")
    if not legal or max_tiers <= 0:
        raise ValueError("public frame needs a legal action and positive tier count")
    return PublicDecisionFrame(
        observation=observation,
        legal_destinations=legal,
        max_tiers=int(max_tiers),
        crane_position=public_state.crane_position,
        crane_time=float(public_state.crane_time),
        retrieval_count=int(public_state.retrieval_count),
        relocation_count=int(public_state.relocation_count),
    )

def public_frame_to_legacy_state(frame: PublicDecisionFrame) -> OnlineCRPPublicState:
    """Map occupied unknown rank 0 to old tuple 0; omit empty padding."""

    ranks = np.asarray(frame.observation.rank_tensor)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    legal = np.asarray(frame.observation.legal_destination_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public ranks and occupancy mask have incompatible shapes")
    if ranks.shape[0] != len(legal) or ranks.shape[1] != frame.max_tiers:
        raise ValueError("public observation shape differs from frame dimensions")
    if tuple(int(x) for x in np.flatnonzero(legal)) != frame.legal_destinations:
        raise ValueError("public state cannot be built from a changed legality mask")

    stacks: list[tuple[int, ...]] = []
    visible_ranks: list[int] = []
    for stack_index in range(ranks.shape[0]):
        occupied_row = occupied[stack_index]
        height = int(occupied_row.sum())
        expected_row = np.arange(frame.max_tiers) < height
        if not np.array_equal(occupied_row, expected_row):
            raise ValueError("public occupied cells must be a bottom-filled stack prefix")
        values = []
        for tier in range(height):
            rank = int(ranks[stack_index, tier])
            if rank < 0 or rank > PUBLIC_UNKNOWN_RANK:
                raise ValueError("public rank must be 1..21 or the occupied-unknown sentinel 22")
            values.append(0 if rank == PUBLIC_UNKNOWN_RANK else rank)
            if 0 < rank < PUBLIC_UNKNOWN_RANK:
                visible_ranks.append(rank)
        if np.any(ranks[stack_index, height:] != 0):
            raise ValueError("empty padding must remain zero")
        stacks.append(tuple(values))
    if visible_ranks.count(1) != 1 or len(set(visible_ranks)) != len(visible_ranks):
        raise ValueError("public frame must contain one current target and unique visible ranks")
    if min(visible_ranks) != 1:
        raise ValueError("public ranks must preserve current target rank 1")
    return OnlineCRPPublicState(
        stacks=tuple(stacks),
        current_target_id=1,
        retrieval_count=frame.retrieval_count,
        relocation_count=frame.relocation_count,
        crane_position=frame.crane_position,
        crane_time=frame.crane_time,
        terminated=False,
    )

def require_public_frame(frame: object) -> None:
    """Fail closed on missing crane scalars or nonpublic actor-side fields."""
    required = (
        "observation", "legal_destinations", "max_tiers", "retrieval_count",
        "relocation_count", "crane_position", "crane_time",
    )
    missing = [name for name in required if not hasattr(frame, name)]
    if missing:
        raise ValueError(f"public frame is missing registered fields: {missing}")
    forbidden = (
        "environment", "provider", "forecast", "retrieval_sequence", "seed",
        "family", "p", "window_size", "layout_fingerprint", "private_trace",
    )
    present = [name for name in forbidden if hasattr(frame, name)]
    if present:
        raise ValueError(f"actor frame contains evaluator/private fields: {present}")
    obs = frame.observation
    observation_fields = set(getattr(obs, "__dataclass_fields__", {}))
    if observation_fields:
        allowed_observation_fields = {
            "rank_tensor", "occupied_mask", "legal_destination_mask",
            "history_hits", "q_prior", "q_t", "public_history_summary",
        }
        unexpected = sorted(observation_fields - allowed_observation_fields)
        if unexpected:
            raise ValueError(f"actor observation contains unregistered fields: {unexpected}")
    nested_private = [
        name for name in forbidden
        if hasattr(obs, name)
    ]
    if nested_private:
        raise ValueError(f"actor observation contains evaluator/private fields: {nested_private}")
    ranks = np.asarray(obs.rank_tensor, dtype=np.int32)
    occupied = np.asarray(obs.occupied_mask, dtype=np.bool_)
    legal_mask = np.asarray(obs.legal_destination_mask, dtype=np.bool_)
    legal = tuple(int(value) for value in frame.legal_destinations)
    if ranks.ndim != 2 or ranks.shape != occupied.shape or ranks.shape[0] != legal_mask.shape[0]:
        raise ValueError("public rank, occupancy, and legal-mask shapes differ")
    if ranks.shape[1] != int(frame.max_tiers):
        raise ValueError("public rank width differs from max_tiers")
    if tuple(int(value) for value in np.flatnonzero(legal_mask)) != legal:
        raise ValueError("public legal destinations differ from their mask")
    if np.any(ranks[~occupied] != 0) or np.any(occupied & ((ranks < 1) | (ranks > 22))):
        raise ValueError("public occupancy/rank sentinel contract failed")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("public frame must contain exactly one current target")
    position = frame.crane_position
    if position is not None and (isinstance(position, bool) or int(position) != position or int(position) < 0):
        raise ValueError("crane_position must be None or a nonnegative public stack position")
    crane_time = float(frame.crane_time)
    if not math.isfinite(crane_time) or crane_time < 0:
        raise ValueError("crane_time must be finite and nonnegative")

def truncate_public_frame(
    frame: PublicDecisionFrame, requested_h: int
) -> PublicDecisionFrame:
    """Censor every occupied rank beyond current+``requested_h`` as 22.

    Empty padding remains 0. The legacy bridge then maps occupied 22 to its
    unknown value 0, which GP and the H20 priority adapter already understand.
    """
    h = int(requested_h)
    if h not in HORIZONS:
        raise ValueError(f"requested horizon must be one of {HORIZONS}, got {h}")
    observation = frame.observation
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    legal = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public rank and occupancy shapes differ")
    if ranks.shape[0] != legal.shape[0] or ranks.shape[1] != frame.max_tiers:
        raise ValueError("public frame shape differs from its mask/tier contract")
    if tuple(int(x) for x in np.flatnonzero(legal)) != frame.legal_destinations:
        raise ValueError("public frame legal destinations changed")
    if np.count_nonzero(occupied & (ranks == 1)) != 1:
        raise ValueError("public frame must contain exactly one current target")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty padding must remain rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > PUBLIC_UNKNOWN_RANK))):
        raise ValueError("occupied public ranks must be in 1..22")

    censored = ranks.copy()
    censored[occupied & (censored > h + 1)] = PUBLIC_UNKNOWN_RANK
    prefix = PublicObservation(
        rank_tensor=censored,
        occupied_mask=occupied,
        legal_destination_mask=legal,
        history_hits=observation.history_hits,
        q_prior=observation.q_prior,
        q_t=observation.q_t,
    )
    return replace(frame, observation=prefix)

def public_prefix(frame: object, horizon: int) -> object:
    """Validate and censor a full public frame using PLAN-051's H rule."""
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    from environment.public_frame import require_public_frame
    from environment.public_frame import truncate_public_frame

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
