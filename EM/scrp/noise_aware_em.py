"""Galle Expected Minmax on public forecast ranks and small selector interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from scrp.noise_aware_forecast import HORIZONS, PublicObservation


Variant = Literal["SR", "R", "S"]


@dataclass(frozen=True)
class ActionCandidate:
    destination: int
    requested_h: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class SelectorDecision:
    destination: int
    requested_h: int
    predicted_delta: float
    variant: str
    candidates: tuple[ActionCandidate, ...]


def horizon_ranks(observation: PublicObservation, h: int) -> np.ndarray:
    """Return an EM-h view from the same detached Hmax tensor."""
    if h < 0:
        raise ValueError("h must be non-negative")
    ranks = np.array(observation.rank_tensor, copy=True)
    occupied = observation.occupied_mask
    ranks[occupied & (ranks > h + 1)] = h + 2
    ranks[~occupied] = 0
    return ranks


def _source_and_blocker(
    observation: PublicObservation, ranks: np.ndarray
) -> tuple[int, int]:
    targets = np.argwhere(observation.occupied_mask & (ranks == 1))
    if targets.shape != (1, 2):
        raise ValueError("public state must contain exactly one current target")
    source, target_tier = (int(x) for x in targets[0])
    source_height = int(observation.occupied_mask[source].sum())
    if target_tier >= source_height - 1:
        raise ValueError("an accessible target must have been auto-retrieved")
    blocker_rank = int(ranks[source, source_height - 1])
    return source, blocker_rank


def select_em_destination(
    observation: PublicObservation, *, max_tiers: int, h: int
) -> int:
    """Choose one legal destination using the published EM rules and ties."""
    if max_tiers <= 0:
        raise ValueError("max_tiers must be positive")
    ranks = horizon_ranks(observation, h)
    _source, blocker = _source_and_blocker(observation, ranks)
    legal = tuple(int(x) for x in np.flatnonzero(observation.legal_destination_mask))
    if not legal:
        raise ValueError("EM requires at least one legal destination")

    heights = observation.occupied_mask.sum(axis=1).astype(int)

    def min_label(stack_id: int) -> float:
        labels = ranks[stack_id, observation.occupied_mask[stack_id]]
        return float(labels.min()) if labels.size else math.inf

    minima = {stack_id: min_label(stack_id) for stack_id in legal}
    safe = [stack_id for stack_id in legal if minima[stack_id] > blocker]
    if safe:
        best_min = min(minima[stack_id] for stack_id in safe)
        tied = [stack_id for stack_id in safe if minima[stack_id] == best_min]
        return min(tied, key=lambda stack_id: (-heights[stack_id], stack_id))

    best_min = max(minima.values())
    tied = [stack_id for stack_id in legal if minima[stack_id] == best_min]
    if math.isinf(best_min):
        raise RuntimeError("empty destinations should have entered EM Rule 1")
    min_count = min(
        int(np.count_nonzero(ranks[stack_id, observation.occupied_mask[stack_id]] == best_min))
        for stack_id in tied
    )
    tied = [
        stack_id
        for stack_id in tied
        if int(
            np.count_nonzero(
                ranks[stack_id, observation.occupied_mask[stack_id]] == best_min
            )
        )
        == min_count
    ]
    return min(tied, key=lambda stack_id: (-heights[stack_id], stack_id))


def _features_for_action(
    observation: PublicObservation,
    *,
    destination: int,
    h: int,
    max_tiers: int,
) -> tuple[float, ...]:
    ranks = horizon_ranks(observation, h)
    _source, blocker = _source_and_blocker(observation, ranks)
    destination_occupied = observation.occupied_mask[destination]
    height = int(destination_occupied.sum())
    c_h = height / float(max_tiers)
    if height:
        earlier = int(
            np.count_nonzero(ranks[destination, destination_occupied] < blocker)
        )
    else:
        earlier = 0
    v_h = earlier / float(max(1, height))
    error = 1.0 - observation.q_t
    h_scaled = h / 20.0
    return (
        1.0,
        h_scaled,
        error,
        c_h,
        v_h,
        h_scaled * error,
        error * c_h,
        error * v_h,
    )


def candidate_actions(
    observation: PublicObservation,
    *,
    max_tiers: int,
    horizons: Sequence[int] = HORIZONS,
) -> tuple[ActionCandidate, ...]:
    """Build EM-h actions from the shared public forecast; keep minimum h per action."""
    if not horizons or 0 not in horizons:
        raise ValueError("candidate horizons must include h=0")
    by_destination: dict[int, ActionCandidate] = {}
    for h in horizons:
        h = int(h)
        destination = select_em_destination(observation, max_tiers=max_tiers, h=h)
        if destination not in by_destination:
            by_destination[destination] = ActionCandidate(
                destination=destination,
                requested_h=h,
                features=_features_for_action(
                    observation,
                    destination=destination,
                    h=h,
                    max_tiers=max_tiers,
                ),
            )
    return tuple(sorted(by_destination.values(), key=lambda candidate: candidate.requested_h))


def selector_features(candidate: ActionCandidate, variant: Variant) -> tuple[float, ...]:
    if variant == "SR":
        return candidate.features
    if variant == "R":
        return tuple(candidate.features[index] for index in (0, 1, 2, 5))
    if variant == "S":
        return tuple(candidate.features[index] for index in (0, 1, 3, 4))
    raise ValueError(f"unknown selector variant {variant!r}")


def select_sr_em(
    observation: PublicObservation,
    *,
    max_tiers: int,
    variant: Variant,
    coefficients: Sequence[float],
) -> SelectorDecision:
    """Select a candidate using supplied frozen weights; this function never fits."""
    candidates = candidate_actions(observation, max_tiers=max_tiers)
    baseline = next(
        (candidate for candidate in candidates if candidate.requested_h == 0), None
    )
    if baseline is None:
        raise RuntimeError("h=0 EM anchor was lost during action deduplication")
    weights = np.asarray(tuple(float(x) for x in coefficients), dtype=np.float64)
    expected_width = {"SR": 8, "R": 4, "S": 4}[variant]
    if weights.shape != (expected_width,) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{variant} requires {expected_width} finite coefficients")

    scored: list[tuple[float, int, ActionCandidate]] = []
    for candidate in candidates:
        if candidate.requested_h == 0:
            continue
        features = np.asarray(selector_features(candidate, variant), dtype=np.float64)
        delta = float(np.dot(weights, features))
        scored.append((delta, candidate.requested_h, candidate))
    if not scored:
        return SelectorDecision(
            baseline.destination, baseline.requested_h, 0.0, variant, candidates
        )
    delta, h, chosen = min(scored, key=lambda item: (item[0], item[1]))
    if delta >= 0.0:
        chosen = baseline
        delta = 0.0
    return SelectorDecision(
        destination=chosen.destination,
        requested_h=chosen.requested_h,
        predicted_delta=delta,
        variant=variant,
        candidates=candidates,
    )


__all__ = [
    "ActionCandidate",
    "SelectorDecision",
    "candidate_actions",
    "horizon_ranks",
    "select_em_destination",
    "select_sr_em",
    "selector_features",
]
