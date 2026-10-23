"""Public-only rolling forecast history and candidate-sequence validation."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Sequence
import numpy as np

HORIZONS=(0,2,5,10,20)
ALGORITHMS=("EM","GP","ERI","EG","AMCTS","B&SDRL","Leveling")
Q_PRIOR=5183/9920
UNKNOWN_RANK=22
HISTORY_WINDOW=8

@dataclass(frozen=True)
class PublicHistorySummary:
    q: float
    displacement: float
    coverage: float
    retrieval_fraction: float
    records: tuple[float, ...]

@dataclass(frozen=True)
class PublicObservationWithHistory:
    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    public_history_summary: PublicHistorySummary

def _int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result

def _parts(frame: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], int, int]:
    observation = getattr(frame, "observation", None)
    if observation is None:
        raise ValueError("public decision frame has no observation")
    ranks = np.asarray(getattr(observation, "rank_tensor"), dtype=np.int32)
    occupied = np.asarray(getattr(observation, "occupied_mask"), dtype=np.bool_)
    legal_mask = np.asarray(getattr(observation, "legal_destination_mask"), dtype=np.bool_)
    tiers = _int(getattr(frame, "max_tiers"), name="max_tiers")
    retrieval_count = _int(getattr(frame, "retrieval_count"), name="retrieval_count")
    legal = tuple(_int(value, name="legal_destination") for value in getattr(frame, "legal_destinations"))
    if ranks.ndim != 2 or ranks.shape != occupied.shape or ranks.shape[1] != tiers:
        raise ValueError("public rank and occupancy shapes do not match max_tiers")
    if legal_mask.shape != (ranks.shape[0],) or tuple(np.flatnonzero(legal_mask)) != legal:
        raise ValueError("public legal tuple and mask differ")
    if np.any(ranks[~occupied] != 0) or np.any(occupied & ((ranks < 1) | (ranks > UNKNOWN_RANK))):
        raise ValueError("public occupied ranks must be 1..22 and empty ranks 0")
    heights = occupied.sum(axis=1).astype(int)
    for stack, height in enumerate(heights):
        if not np.array_equal(occupied[stack], np.arange(tiers) < int(height)):
            raise ValueError("public occupancy must be bottom-filled")
    target = np.argwhere(occupied & (ranks == 1))
    if target.shape != (1, 2):
        raise ValueError("public frame must have exactly one rank-1 current target")
    known = ranks[occupied & (ranks < UNKNOWN_RANK)]
    if len(known) != len(set(int(value) for value in known)):
        raise ValueError("known public ranks must be unique")
    expected_legal = tuple(
        index for index, height in enumerate(heights)
        if index != int(target[0, 0]) and int(height) < tiers
    )
    if legal != expected_legal:
        raise ValueError("public legal mask is not the complete non-source/non-full set")
    if retrieval_count < 0:
        raise ValueError("retrieval_count must be nonnegative")
    return ranks, occupied, legal_mask, legal, tiers, retrieval_count

def _q_values(frame: object) -> tuple[float, float]:
    observation = getattr(frame, "observation")
    q_prior = float(getattr(observation, "q_prior"))
    q_t = float(getattr(observation, "q_t"))
    if not math.isfinite(q_prior) or abs(q_prior - Q_PRIOR) > 1e-15:
        raise ValueError("public q_prior differs from the registered 5183/9920 prior")
    if not math.isfinite(q_t) or not 0.0 <= q_t <= 1.0:
        raise ValueError("public q_t must be finite and in [0,1]")
    return q_prior, q_t

def summarize_public_history(
    previous_frame: object, own_action: int, current_frame: object
) -> float | None:
    """Recover one newly visible target's old forecast position from public state.

    The return value is a normalized displacement record in [0,1], or None if
    no new public target was observed.  Multiple automatic reveals are treated
    as one visible transition: removed objects are not ordered or reconstructed.
    """
    before_ranks, before_occ, _before_legal, before_destinations, tiers, before_retrieval = _parts(previous_frame)
    after_ranks, after_occ, _after_legal, _after_destinations, after_tiers, after_retrieval = _parts(current_frame)
    if after_tiers != tiers or after_ranks.shape != before_ranks.shape:
        raise ValueError("public frame geometry changed within a history transition")
    action = _int(own_action, name="own_action")
    if action not in before_destinations:
        raise ValueError("own action was not legal in the previous public frame")
    before_target = np.argwhere(before_occ & (before_ranks == 1))
    source, target_tier = (int(value) for value in before_target[0])
    heights = before_occ.sum(axis=1).astype(int)
    source_height = int(heights[source])
    destination_height = int(heights[action])
    if source_height <= target_tier + 1 or destination_height >= tiers:
        raise ValueError("public action does not move a current target blocker legally")

    shadow_ranks = before_ranks.copy()
    shadow_occ = before_occ.copy()
    moved_rank = int(shadow_ranks[source, source_height - 1])
    shadow_ranks[source, source_height - 1] = 0
    shadow_occ[source, source_height - 1] = False
    shadow_ranks[action, destination_height] = moved_rank
    shadow_occ[action, destination_height] = True

    retrieval_delta = after_retrieval - before_retrieval
    if retrieval_delta < 0:
        raise ValueError("public retrieval count moved backwards")
    if np.any(after_occ & ~shadow_occ):
        raise ValueError("next public occupancy adds a box absent from the action shadow")
    removed_count = int(np.count_nonzero(shadow_occ & ~after_occ))
    if removed_count != retrieval_delta:
        raise ValueError("public occupancy removals must equal the retrieval-count increase")
    after_target = np.argwhere(after_occ & (after_ranks == 1))
    if after_target.shape != (1, 2):
        raise ValueError("next public frame must expose exactly one current target")
    after_location = tuple(int(value) for value in after_target[0])
    before_location = (source, target_tier)
    if retrieval_delta == 0:
        if after_location != before_location or not np.array_equal(after_occ, shadow_occ):
            raise ValueError("public occupancy changed without a retrieval disclosure")
        return None
    if after_occ[source, target_tier]:
        raise ValueError("the previous current target must be removed after a retrieval disclosure")
    if after_location == before_location:
        raise ValueError("a newly retrieved target cannot remain at its prior location")
    old_rank = int(shadow_ranks[after_location])
    if old_rank == UNKNOWN_RANK:
        return 1.0
    if not 2 <= old_rank <= 21:
        raise ValueError("new public target cannot be mapped to a prior future rank")
    previous_forecast_rank = old_rank - 1
    return float((previous_forecast_rank - 1) / 20.0)

class PublicHistoryTracker:
    """Per-actor rolling window of action-reconstructed public target ranks."""

    def __init__(self, records: Sequence[float] = ()) -> None:
        values = tuple(float(value) for value in records)
        if len(values) > HISTORY_WINDOW or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("public history must contain at most eight normalized records")
        self._records: deque[float] = deque(values, maxlen=HISTORY_WINDOW)

    @property
    def records(self) -> tuple[float, ...]:
        return tuple(self._records)

    def clone(self) -> "PublicHistoryTracker":
        return PublicHistoryTracker(self.records)

    def update(self, previous_frame: object, own_action: int, current_frame: object) -> PublicHistorySummary:
        record = summarize_public_history(previous_frame, own_action, current_frame)
        if record is not None:
            self._records.append(record)
        _ranks, occupied, _legal_mask, _legal, _tiers, retrieval_count = _parts(current_frame)
        q_prior, q_t = _q_values(current_frame)
        del q_prior
        count = len(self._records)
        denominator = retrieval_count + int(np.count_nonzero(occupied))
        retrieval_fraction = 0.0 if denominator == 0 else retrieval_count / float(denominator)
        return PublicHistorySummary(
            q=q_t,
            displacement=0.0 if count == 0 else float(sum(self._records) / count),
            coverage=count / float(HISTORY_WINDOW),
            retrieval_fraction=float(retrieval_fraction),
            records=self.records,
        )

    def summary(self, frame: object) -> PublicHistorySummary:
        _ranks, occupied, _legal_mask, _legal, _tiers, retrieval_count = _parts(frame)
        _q_prior, q_t = _q_values(frame)
        count = len(self._records)
        denominator = retrieval_count + int(np.count_nonzero(occupied))
        return PublicHistorySummary(
            q=q_t,
            displacement=0.0 if count == 0 else float(sum(self._records) / count),
            coverage=count / float(HISTORY_WINDOW),
            retrieval_fraction=0.0 if denominator == 0 else retrieval_count / float(denominator),
            records=self.records,
        )

def attach_history_summary(frame: object, summary: PublicHistorySummary) -> object:
    """Return a same-information frame carrying a public, evaluator-free summary."""
    from dataclasses import replace

    observation = getattr(frame, "observation")
    wrapped = PublicObservationWithHistory(
        rank_tensor=np.asarray(observation.rank_tensor, dtype=np.int32),
        occupied_mask=np.asarray(observation.occupied_mask, dtype=np.bool_),
        legal_destination_mask=np.asarray(observation.legal_destination_mask, dtype=np.bool_),
        history_hits=tuple(bool(value) for value in getattr(observation, "history_hits", ())),
        q_prior=float(observation.q_prior),
        q_t=float(observation.q_t),
        public_history_summary=summary,
    )
    try:
        return replace(frame, observation=wrapped)
    except TypeError:
        if hasattr(frame, "_replace"):
            return frame._replace(observation=wrapped)
        raise TypeError("public frame must support immutable observation replacement")

def _after_move(frame: object, destination: int) -> object:
    from dataclasses import replace

    ranks, occupied, _legal_mask, legal, tiers, retrieval_count = _parts(frame)
    action = _int(destination, name="destination")
    if action not in legal:
        raise ValueError("candidate destination is outside the current public legal set")
    target = np.argwhere(occupied & (ranks == 1))
    source, target_tier = (int(value) for value in target[0])
    heights = occupied.sum(axis=1).astype(int)
    if int(heights[source]) <= target_tier + 1 or int(heights[action]) >= tiers:
        raise ValueError("candidate relocation requires a blocked target and free capacity")
    result_ranks = ranks.copy()
    result_occupied = occupied.copy()
    top = int(heights[source]) - 1
    moved_rank = int(result_ranks[source, top])
    result_ranks[source, top] = 0
    result_occupied[source, top] = False
    result_ranks[action, int(heights[action])] = moved_rank
    result_occupied[action, int(heights[action])] = True
    next_heights = result_occupied.sum(axis=1).astype(int)
    next_legal_mask = np.asarray(
        [index != source and int(next_heights[index]) < tiers for index in range(len(next_heights))],
        dtype=np.bool_,
    )
    observation = getattr(frame, "observation")
    summary = getattr(observation, "public_history_summary", None)
    if summary is None:
        raise ValueError("candidate scoring requires one decision-fixed public history summary")
    updated_observation = PublicObservationWithHistory(
        rank_tensor=result_ranks,
        occupied_mask=result_occupied,
        legal_destination_mask=next_legal_mask,
        history_hits=tuple(bool(value) for value in getattr(observation, "history_hits", ())),
        q_prior=float(observation.q_prior),
        q_t=float(observation.q_t),
        public_history_summary=summary,
    )
    return replace(
        frame,
        observation=updated_observation,
        legal_destinations=tuple(int(value) for value in np.flatnonzero(next_legal_mask)),
        relocation_count=_int(getattr(frame, "relocation_count"), name="relocation_count") + 1,
        retrieval_count=retrieval_count,
    )

def validate_sequence(frame: object, destinations: Sequence[int], algorithm: str) -> tuple[int, ...]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown proposer algorithm {algorithm!r}")
    sequence = tuple(_int(value, name="destination") for value in destinations)
    if not sequence:
        raise ValueError("candidate execution sequence must be nonempty")
    initial_ranks, initial_occupied, _mask, _legal, _tiers, _retrieval = _parts(frame)
    target = np.argwhere(initial_occupied & (initial_ranks == 1))
    source, target_tier = (int(value) for value in target[0])
    blocker_count = int(initial_occupied[source].sum()) - target_tier - 1
    if blocker_count <= 0:
        raise ValueError("sequence candidates require a blocked target")
    group_commit = algorithm in {"EG", "B&SDRL"}
    if not group_commit and len(sequence) != 1:
        raise ValueError("single-action candidates must contain one destination")
    if group_commit and len(sequence) != blocker_count:
        raise ValueError(f"{algorithm} candidates must contain the entire current blocker group")
    virtual = frame
    for index, destination in enumerate(sequence):
        virtual = _after_move(virtual, destination)
        ranks, occupied, _mask, _legal, _tiers, _retrieval = _parts(virtual)
        target = np.argwhere(occupied & (ranks == 1))
        s, t = (int(value) for value in target[0])
        remains_blocked = int(occupied[s].sum()) > t + 1
        if not remains_blocked and index < len(sequence) - 1:
            raise ValueError("candidate continues after exposing the current target")
        if remains_blocked and index == len(sequence) - 1 and group_commit:
            raise ValueError(f"{algorithm} candidate ended before completing its blocker group")
    return sequence
