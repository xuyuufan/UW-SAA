"""Public-state R-Order risk model and action-history wrapper for PLAN-049.

The policy boundary in this module accepts only rank tensors, occupancy and
legal masks, public counters, public q, and its own executed action.  Evaluator
truth is deliberately accepted only by the separate training code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


HORIZONS = (0, 2, 5, 10, 20)
ALGORITHMS = ("EM", "GP", "ERI", "EG")
Q_PRIOR = 5183 / 9920
UNKNOWN_RANK = 22
HISTORY_WINDOW = 8
RATE_FEATURES = ("intercept", "q", "displacement", "coverage", "retrieval_fraction")


@dataclass(frozen=True)
class PublicHistorySummary:
    q: float
    displacement: float
    coverage: float
    retrieval_fraction: float
    records: tuple[float, ...]


@dataclass(frozen=True)
class _PublicObservation:
    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float
    r_order_summary: PublicHistorySummary


@dataclass(frozen=True)
class SequenceOrigin:
    source: str
    horizon: int | None
    proposal_id: str


@dataclass(frozen=True)
class ROrderCandidate:
    destinations: tuple[int, ...]
    origins: tuple[SequenceOrigin, ...]
    is_h20_anchor: bool


@dataclass(frozen=True)
class ROrderPool:
    algorithm: str
    candidates: tuple[ROrderCandidate, ...]
    anchor_sequence: tuple[int, ...]


@dataclass(frozen=True)
class ROrderDecision:
    algorithm: str
    sequence: tuple[int, ...]
    anchor_sequence: tuple[int, ...]
    score: float
    anchor_score: float
    tau: float
    used_anchor: bool
    scored_candidates: tuple[tuple[tuple[int, ...], float], ...]


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
    wrapped = _PublicObservation(
        rank_tensor=np.asarray(observation.rank_tensor, dtype=np.int32),
        occupied_mask=np.asarray(observation.occupied_mask, dtype=np.bool_),
        legal_destination_mask=np.asarray(observation.legal_destination_mask, dtype=np.bool_),
        history_hits=tuple(bool(value) for value in getattr(observation, "history_hits", ())),
        q_prior=float(observation.q_prior),
        q_t=float(observation.q_t),
        r_order_summary=summary,
    )
    try:
        return replace(frame, observation=wrapped)
    except TypeError:
        if hasattr(frame, "_replace"):
            return frame._replace(observation=wrapped)
        raise TypeError("public frame must support immutable observation replacement")


def rate_features(summary: PublicHistorySummary) -> np.ndarray:
    values = np.asarray(
        (1.0, summary.q, summary.displacement, summary.coverage, summary.retrieval_fraction),
        dtype=np.float64,
    )
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("R-Order rate features must be five finite public values")
    if not 0 <= values[1] <= 1 or not 0 <= values[2] <= 1 or not 0 <= values[3] <= 1 or not 0 <= values[4] <= 1:
        raise ValueError("R-Order public rate features must be in [0,1]")
    return values


@dataclass(frozen=True)
class PlackettLuceRates:
    coefficients: np.ndarray
    active_feature_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    model_id: str = "r_order"

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64).copy()
        if coefficients.shape != (20, 5) or not np.isfinite(coefficients).all():
            raise ValueError("Plackett-Luce rates require a finite [20,5] coefficient matrix")
        active = tuple(_int(value, name="active_feature_index") for value in self.active_feature_indices)
        if not active or any(value not in range(5) for value in active):
            raise ValueError("active feature indices must be a nonempty subset of 0..4")
        inactive = tuple(index for index in range(5) if index not in active)
        if inactive and np.any(coefficients[:, inactive] != 0.0):
            raise ValueError("inactive feature coefficients must be exactly zero")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "active_feature_indices", active)

    def log_rate(self, future_rank: int, summary: PublicHistorySummary) -> float:
        rank = _int(future_rank, name="future_rank")
        if rank == UNKNOWN_RANK:
            return 0.0
        if not 1 <= rank <= 20:
            raise ValueError("future rank must be 1..20 or unknown sentinel 22")
        return float(self.coefficients[rank - 1] @ rate_features(summary))

    def rate(self, future_rank: int, summary: PublicHistorySummary) -> float:
        return float(math.exp(self.log_rate(future_rank, summary)))


def _future_rank(public_rank: int) -> int | None:
    if public_rank == 1:
        return None
    if 2 <= public_rank <= 21:
        return public_rank - 1
    if public_rank == UNKNOWN_RANK:
        return UNKNOWN_RANK
    raise ValueError("public rank outside current/future/sentinel range")


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


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
    summary = getattr(observation, "r_order_summary", None)
    if summary is None:
        raise ValueError("candidate scoring requires one decision-fixed public history summary")
    updated_observation = _PublicObservation(
        rank_tensor=result_ranks,
        occupied_mask=result_occupied,
        legal_destination_mask=next_legal_mask,
        history_hits=tuple(bool(value) for value in getattr(observation, "history_hits", ())),
        q_prior=float(observation.q_prior),
        q_t=float(observation.q_t),
        r_order_summary=summary,
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
        raise ValueError("R-Order candidates require a blocked target")
    if algorithm != "EG" and len(sequence) != 1:
        raise ValueError("EM, GP, and ERI candidates must be one destination")
    if algorithm == "EG" and len(sequence) != blocker_count:
        raise ValueError("EG candidates must contain the entire current blocker group")
    virtual = frame
    for index, destination in enumerate(sequence):
        virtual = _after_move(virtual, destination)
        ranks, occupied, _mask, _legal, _tiers, _retrieval = _parts(virtual)
        target = np.argwhere(occupied & (ranks == 1))
        s, t = (int(value) for value in target[0])
        remains_blocked = int(occupied[s].sum()) > t + 1
        if not remains_blocked and index < len(sequence) - 1:
            raise ValueError("candidate continues after exposing the current target")
        if remains_blocked and index == len(sequence) - 1 and algorithm == "EG":
            raise ValueError("EG candidate ended before completing its blocker group")
    return sequence


def expected_blocking_units(frame: object, rates: PlackettLuceRates) -> float:
    """Compute the registered PL expected-blocking proxy on a public layout.

    The current target is deterministic: every box above it contributes one,
    while the target itself contributes zero. Future known ranks use the fitted
    rates and the exchangeable unknown tail has rate one. This is a risk proxy,
    not a true relocation count or safety bound.
    """
    ranks, occupied, _mask, _legal, _tiers, _retrieval = _parts(frame)
    observation = getattr(frame, "observation")
    summary = getattr(observation, "r_order_summary", None)
    if summary is None:
        raise ValueError("B(s) requires a public history summary fixed at this decision")
    target = np.argwhere(occupied & (ranks == 1))
    target_source, target_tier = (int(value) for value in target[0])
    terms: list[float] = []
    for stack in range(ranks.shape[0]):
        height = int(np.count_nonzero(occupied[stack]))
        for tier in range(height):
            public_rank = int(ranks[stack, tier])
            if public_rank == 1:
                continue
            if stack == target_source and tier > target_tier:
                terms.append(1.0)
                continue
            future_rank = _future_rank(public_rank)
            if future_rank is None:
                continue
            lower_ranks = [
                int(ranks[stack, lower])
                for lower in range(tier)
                if int(ranks[stack, lower]) != 1
            ]
            if not lower_ranks:
                terms.append(0.0)
                continue
            log_i = rates.log_rate(future_rank, summary)
            log_denominator = _logsumexp(
                [log_i, *(rates.log_rate(_future_rank(lower), summary) for lower in lower_ranks if _future_rank(lower) is not None)]
            )
            # The current target may be physically below this item only when it
            # shares its stack; that deterministic case was handled above.
            probability_first = math.exp(log_i - log_denominator)
            terms.append(1.0 - probability_first)
    result = float(sum(terms))
    if not math.isfinite(result) or result < -1e-12:
        raise ValueError("blocking proxy is nonfinite or negative")
    return max(0.0, result)


def score_sequence(
    frame: object, sequence: Sequence[int], algorithm: str, rates: PlackettLuceRates
) -> float:
    normalized = validate_sequence(frame, sequence, algorithm)
    virtual = frame
    for destination in normalized:
        virtual = _after_move(virtual, destination)
    score = float(len(normalized) + expected_blocking_units(virtual, rates))
    if not math.isfinite(score):
        raise ValueError("candidate score is nonfinite")
    return score


def build_risk_proposal(
    frame: object, algorithm: str, rates: PlackettLuceRates
) -> tuple[int, ...]:
    """Build one deterministic risk proposal; EG commits a full greedy group."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown proposer algorithm {algorithm!r}")
    _ranks, occupied, _mask, legal, _tiers, _retrieval = _parts(frame)
    ranks = np.asarray(getattr(frame.observation, "rank_tensor"), dtype=np.int32)
    target = np.argwhere(occupied & (ranks == 1))
    source, target_tier = (int(value) for value in target[0])
    blocker_count = int(occupied[source].sum()) - target_tier - 1
    if blocker_count <= 0 or not legal:
        raise ValueError("risk proposal requires a blocked target and legal destinations")
    if algorithm != "EG":
        scored = [
            (score_sequence(frame, (destination,), algorithm, rates), destination)
            for destination in legal
        ]
        return (int(min(scored)[1]),)
    sequence: list[int] = []
    virtual = frame
    for _step in range(blocker_count):
        _ranks, _occupied, _legal_mask, next_legal, _tiers, _retrieval = _parts(virtual)
        if not next_legal:
            raise RuntimeError("EG risk proposal has no legal destination before group completion")
        candidates: list[tuple[float, int, object]] = []
        for destination in next_legal:
            after = _after_move(virtual, destination)
            value = expected_blocking_units(after, rates)
            candidates.append((value, destination, after))
        value, destination, virtual = min(candidates, key=lambda row: (row[0], row[1]))
        del value
        sequence.append(int(destination))
    result = tuple(sequence)
    return validate_sequence(frame, result, "EG")


def build_candidate_pool(
    frame: object,
    algorithm: str,
    native_proposals: Mapping[int, Sequence[int]],
    risk_proposal: Sequence[int] | None = None,
) -> ROrderPool:
    if algorithm not in ALGORITHMS or set(native_proposals) != set(HORIZONS):
        raise ValueError("native candidates must cover the four algorithms and five horizons")
    grouped: dict[tuple[int, ...], list[SequenceOrigin]] = {}
    for horizon in HORIZONS:
        sequence = validate_sequence(frame, native_proposals[horizon], algorithm)
        grouped.setdefault(sequence, []).append(SequenceOrigin("native", horizon, f"{algorithm}:h{horizon}"))
    anchor = validate_sequence(frame, native_proposals[20], algorithm)
    if risk_proposal is not None:
        risk = validate_sequence(frame, risk_proposal, algorithm)
        grouped.setdefault(risk, []).append(SequenceOrigin("risk", None, f"{algorithm}:risk_greedy"))
    candidates = tuple(
        ROrderCandidate(
            destinations=sequence,
            origins=tuple(sorted(origins, key=lambda origin: (origin.source, -1 if origin.horizon is None else origin.horizon, origin.proposal_id))),
            is_h20_anchor=(sequence == anchor),
        )
        for sequence, origins in sorted(grouped.items())
    )
    if not candidates or sum(candidate.is_h20_anchor for candidate in candidates) != 1:
        raise AssertionError("candidate pool must retain exactly one explicit H20 anchor")
    if len(candidates) > (6 if risk_proposal is not None else 5):
        raise AssertionError("full-sequence deduplication exceeded the six-slot proposal limit")
    return ROrderPool(algorithm, candidates, anchor)


def select_candidate(
    frame: object,
    pool: ROrderPool,
    rates: PlackettLuceRates,
    *,
    tau: float,
) -> ROrderDecision:
    threshold = float(tau)
    if not math.isfinite(threshold) or threshold not in (0.0, 0.25):
        raise ValueError("tau must be exactly 0 or 0.25 expected blocking units")
    scored = tuple(
        (candidate.destinations, score_sequence(frame, candidate.destinations, pool.algorithm, rates))
        for candidate in pool.candidates
    )
    score_map = dict(scored)
    anchor_score = score_map[pool.anchor_sequence]
    alternatives = [
        row for row in scored
        if row[0] != pool.anchor_sequence and row[1] < anchor_score - threshold
    ]
    if not alternatives:
        selected, value = pool.anchor_sequence, anchor_score
        used_anchor = True
    else:
        selected, value = min(alternatives, key=lambda row: (row[1], row[0]))
        used_anchor = False
    return ROrderDecision(
        algorithm=pool.algorithm,
        sequence=selected,
        anchor_sequence=pool.anchor_sequence,
        score=float(value),
        anchor_score=float(anchor_score),
        tau=threshold,
        used_anchor=used_anchor,
        scored_candidates=tuple(sorted(scored)),
    )


__all__ = (
    "ALGORITHMS", "HISTORY_WINDOW", "HORIZONS", "PlackettLuceRates",
    "PublicHistorySummary", "PublicHistoryTracker", "Q_PRIOR", "RATE_FEATURES",
    "ROrderCandidate", "ROrderDecision", "ROrderPool", "SequenceOrigin",
    "UNKNOWN_RANK", "attach_history_summary", "build_candidate_pool",
    "build_risk_proposal", "expected_blocking_units", "rate_features",
    "score_sequence", "select_candidate", "summarize_public_history", "validate_sequence",
)
