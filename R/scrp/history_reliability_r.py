"""Frozen history-reliability selectors for PLAN-038 M1.

This module contains actor-visible transforms and the one permitted offline
baseline fit. Episode truth, forecast generation, and environment transitions
remain in the experiment runner/evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from scrp.noise_aware_em import (
    ActionCandidate,
    SelectorDecision,
    candidate_actions,
    select_sr_em,
)
from scrp.noise_aware_forecast import HORIZONS, PublicObservation


FROZEN_R_Q_PRIOR = 5183 / 9920
FROZEN_R_COEFFICIENTS = (
    -0.10860203113738334,
    -0.7425766885981918,
    0.20742547985980178,
    1.1399906546560534,
)
R_NO_HISTORY_RIDGE_LAMBDA = 1.0
R_NO_HISTORY_EXPECTED_LABEL_ROWS = 388
R_NO_HISTORY_EXPECTED_STATE_COUNT = 200
M2_R_NO_HISTORY_COEFFICIENTS = (
    0.010488524461631774,
    -0.34881235860282905,
)


@dataclass(frozen=True)
class RidgeFit:
    coefficients: tuple[float, float]
    label_row_count: int
    sampled_state_count: int
    ridge_lambda: float
    penalty_includes_bias: bool
    residual_rmse: float
    residual_mae: float
    residual_mean: float


def _validated_r_no_history_matrices(
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(label_rows) != R_NO_HISTORY_EXPECTED_LABEL_ROWS:
        raise ValueError(
            "R_NO_HISTORY requires exactly the 388 frozen D_fit label rows"
        )

    x_rows: list[tuple[float, float]] = []
    targets: list[float] = []
    state_ids: set[str] = set()
    unique_conditions: set[tuple[str, int]] = set()
    for row in label_rows:
        try:
            state_id = str(row["state_id"])
            requested_h = int(row["requested_h"])
            features_8d = tuple(float(value) for value in row["public_features_8d"])
            target = float(row["true_delta_R_label_evaluator_only"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid frozen N3 fit-label row") from exc
        if not state_id or requested_h not in HORIZONS:
            raise ValueError("fit labels contain an unknown state or horizon")
        if len(features_8d) != 8 or not all(math.isfinite(v) for v in features_8d):
            raise ValueError("fit labels must contain eight finite N3 features")
        if not math.isfinite(target):
            raise ValueError("fit labels must contain finite evaluator targets")
        if not math.isclose(features_8d[0], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fit-label intercept differs from the frozen contract")
        if not math.isclose(
            features_8d[1], requested_h / 20.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("fit-label horizon feature differs from requested_h")
        condition = (state_id, requested_h)
        if condition in unique_conditions:
            raise ValueError("duplicate state/horizon fit-label row")
        unique_conditions.add(condition)
        state_ids.add(state_id)
        if requested_h == 0 and target != 0.0:
            raise ValueError("the frozen h=0 anchor label must be exactly zero")
        x_rows.append((1.0, requested_h / 20.0))
        targets.append(target)

    if len(state_ids) != R_NO_HISTORY_EXPECTED_STATE_COUNT:
        raise ValueError("frozen fit labels must cover exactly 200 sampled states")

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    return x, y, len(state_ids)


def validate_r_no_history_label_rows(
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Validate the fixed D_fit row contract without performing a fit."""
    _x, _y, state_count = _validated_r_no_history_matrices(label_rows)
    return len(label_rows), state_count


def fit_r_no_history(label_rows: Sequence[Mapping[str, Any]]) -> RidgeFit:
    """Fit the fixed two-feature baseline on exactly the frozen 388 fit labels.

    The caller must bind these rows to the reviewed legacy D_fit artifact and
    verify its hash before calling. This function accepts no select/test data
    and exposes no tunable ridge penalty.
    """
    x, y, state_count = _validated_r_no_history_matrices(label_rows)
    gram = x.T @ x + R_NO_HISTORY_RIDGE_LAMBDA * np.eye(2, dtype=np.float64)
    coefficients = np.linalg.solve(gram, x.T @ y)
    residual = y - x @ coefficients
    return RidgeFit(
        coefficients=(float(coefficients[0]), float(coefficients[1])),
        label_row_count=len(label_rows),
        sampled_state_count=state_count,
        ridge_lambda=R_NO_HISTORY_RIDGE_LAMBDA,
        penalty_includes_bias=True,
        residual_rmse=float(np.sqrt(np.mean(residual**2))),
        residual_mae=float(np.mean(np.abs(residual))),
        residual_mean=float(np.mean(residual)),
    )


def observation_with_constant_reliability(
    observation: PublicObservation,
) -> PublicObservation:
    """Return the same public input with only q_t replaced by q_prior."""
    return PublicObservation(
        rank_tensor=observation.rank_tensor,
        occupied_mask=observation.occupied_mask,
        legal_destination_mask=observation.legal_destination_mask,
        history_hits=observation.history_hits,
        q_prior=observation.q_prior,
        q_t=observation.q_prior,
    )


def select_r_no_history(
    observation: PublicObservation,
    *,
    max_tiers: int,
    coefficients: Sequence[float],
) -> SelectorDecision:
    """Apply the fitted [1, h/20] baseline with the frozen EM candidate rule."""
    weights = np.asarray(tuple(float(value) for value in coefficients), dtype=np.float64)
    if weights.shape != (2,) or not np.all(np.isfinite(weights)):
        raise ValueError("R_NO_HISTORY requires two finite ridge coefficients")
    candidates = candidate_actions(observation, max_tiers=max_tiers)
    baseline = next(
        (candidate for candidate in candidates if candidate.requested_h == 0), None
    )
    if baseline is None:
        raise RuntimeError("h=0 EM anchor was lost during action deduplication")

    scored: list[tuple[float, int, ActionCandidate]] = []
    for candidate in candidates:
        if candidate.requested_h == 0:
            continue
        features = np.asarray(
            (1.0, candidate.requested_h / 20.0), dtype=np.float64
        )
        scored.append(
            (float(np.dot(weights, features)), candidate.requested_h, candidate)
        )
    if not scored:
        return SelectorDecision(
            baseline.destination, 0, 0.0, "R_NO_HISTORY", candidates
        )
    delta, _requested_h, chosen = min(scored, key=lambda item: (item[0], item[1]))
    if delta >= 0.0:
        return SelectorDecision(
            baseline.destination, 0, 0.0, "R_NO_HISTORY", candidates
        )
    return SelectorDecision(
        chosen.destination,
        chosen.requested_h,
        delta,
        "R_NO_HISTORY",
        candidates,
    )


def select_r_action(
    observation: PublicObservation,
    *,
    method: str,
    max_tiers: int,
    no_history_coefficients: Sequence[float] | None = None,
) -> SelectorDecision:
    """Select a frozen R-family action; only R_NO_HISTORY consumes fitted weights."""
    if method == "R":
        return select_sr_em(
            observation,
            max_tiers=max_tiers,
            variant="R",
            coefficients=FROZEN_R_COEFFICIENTS,
        )
    if method == "R_CONST":
        constant_observation = observation_with_constant_reliability(observation)
        return select_sr_em(
            constant_observation,
            max_tiers=max_tiers,
            variant="R",
            coefficients=FROZEN_R_COEFFICIENTS,
        )
    if method == "R_NO_HISTORY":
        if no_history_coefficients is None:
            raise ValueError("R_NO_HISTORY requires its frozen D_fit ridge fit")
        return select_r_no_history(
            observation,
            max_tiers=max_tiers,
            coefficients=no_history_coefficients,
        )
    raise ValueError(f"unknown history-reliability method {method!r}")


def require_legal_action(action: int, legal_destinations: Sequence[int]) -> int:
    """Fail immediately if an actor returns a destination outside the public mask."""
    selected = int(action)
    if selected not in legal_destinations:
        raise RuntimeError(
            f"history-reliability actor selected illegal destination {selected}"
        )
    return selected


@dataclass(frozen=True)
class _M2PublicStateStatistics:
    """Per-call raw-rank summaries shared across the frozen five horizons."""

    heights: tuple[int, ...]
    raw_minima: tuple[float, ...]
    raw_minimum_counts: tuple[int, ...]
    legal_destinations: tuple[int, ...]
    raw_blocker: int

    def minimum_at_h(self, stack_id: int, h: int) -> float:
        if h < 0:
            raise ValueError("h must be non-negative")
        if self.heights[stack_id] == 0:
            return math.inf
        return min(self.raw_minima[stack_id], float(h + 2))

    def minimum_count_at_h(self, stack_id: int, h: int) -> int:
        if self.heights[stack_id] == 0:
            return 0
        if self.raw_minima[stack_id] >= h + 2:
            return self.heights[stack_id]
        return self.raw_minimum_counts[stack_id]

    def destination_at_h(self, h: int) -> int:
        if h < 0:
            raise ValueError("h must be non-negative")
        blocker = min(self.raw_blocker, h + 2)
        minima = {
            stack_id: self.minimum_at_h(stack_id, h)
            for stack_id in self.legal_destinations
        }
        safe = [stack_id for stack_id in self.legal_destinations if minima[stack_id] > blocker]
        if safe:
            best_minimum = min(minima[stack_id] for stack_id in safe)
            tied = [stack_id for stack_id in safe if minima[stack_id] == best_minimum]
            return min(tied, key=lambda stack_id: (-self.heights[stack_id], stack_id))

        best_minimum = max(minima.values())
        tied = [
            stack_id
            for stack_id in self.legal_destinations
            if minima[stack_id] == best_minimum
        ]
        if math.isinf(best_minimum):
            raise RuntimeError("empty destinations should have entered EM Rule 1")
        best_count = min(self.minimum_count_at_h(stack_id, h) for stack_id in tied)
        tied = [
            stack_id
            for stack_id in tied
            if self.minimum_count_at_h(stack_id, h) == best_count
        ]
        return min(tied, key=lambda stack_id: (-self.heights[stack_id], stack_id))


def _m2_public_state_statistics(
    observation: PublicObservation, *, max_tiers: int
) -> _M2PublicStateStatistics:
    if max_tiers <= 0:
        raise ValueError("max_tiers must be positive")
    ranks = observation.rank_tensor
    occupied = observation.occupied_mask
    targets = np.argwhere(occupied & (ranks == 1))
    if targets.shape != (1, 2):
        raise ValueError("public state must contain exactly one current target")
    source, target_tier = (int(value) for value in targets[0])
    heights_array = occupied.sum(axis=1).astype(int)
    source_height = int(heights_array[source])
    if target_tier >= source_height - 1:
        raise ValueError("an accessible target must have been auto-retrieved")
    raw_blocker = int(ranks[source, source_height - 1])
    legal = tuple(int(value) for value in np.flatnonzero(observation.legal_destination_mask))
    if not legal:
        raise ValueError("EM requires at least one legal destination")

    raw_minima: list[float] = []
    raw_minimum_counts: list[int] = []
    heights: list[int] = []
    for stack_id, height_value in enumerate(heights_array):
        height = int(height_value)
        heights.append(height)
        if height == 0:
            # Empty remains +inf at every horizon; it is not an unknown-rank cap.
            raw_minima.append(math.inf)
            raw_minimum_counts.append(0)
            continue
        labels = ranks[stack_id, occupied[stack_id]]
        raw_minimum = int(labels.min())
        raw_minima.append(float(raw_minimum))
        raw_minimum_counts.append(int(np.count_nonzero(labels == raw_minimum)))

    return _M2PublicStateStatistics(
        heights=tuple(heights),
        raw_minima=tuple(raw_minima),
        raw_minimum_counts=tuple(raw_minimum_counts),
        legal_destinations=legal,
        raw_blocker=raw_blocker,
    )


def select_em_destination_optimized(
    observation: PublicObservation, *, max_tiers: int, h: int
) -> int:
    """M2 optional EM kernel using the same per-state raw summaries as optimized R."""
    if max_tiers <= 0:
        raise ValueError("max_tiers must be positive")
    if h < 0:
        raise ValueError("h must be non-negative")
    return _m2_public_state_statistics(
        observation, max_tiers=max_tiers
    ).destination_at_h(h)


def _m2_candidates(
    statistics: _M2PublicStateStatistics,
    *,
    reliability_error: float | None,
) -> tuple[ActionCandidate, ...]:
    by_destination: dict[int, ActionCandidate] = {}
    for h in HORIZONS:
        destination = statistics.destination_at_h(h)
        if destination in by_destination:
            continue
        h_scaled = h / 20.0
        if reliability_error is None:
            features = (1.0, h_scaled)
        else:
            # Keep the frozen R feature and np.dot order: [1, h/20, 1-q, h/20*(1-q)].
            features = (
                1.0,
                h_scaled,
                reliability_error,
                h_scaled * reliability_error,
            )
        by_destination[destination] = ActionCandidate(
            destination=destination,
            requested_h=h,
            features=features,
        )
    return tuple(sorted(by_destination.values(), key=lambda candidate: candidate.requested_h))


def _m2_select_scored_candidates(
    candidates: Sequence[ActionCandidate],
    *,
    coefficients: Sequence[float],
    variant: str,
) -> SelectorDecision:
    baseline = next(
        (candidate for candidate in candidates if candidate.requested_h == 0), None
    )
    if baseline is None:
        raise RuntimeError("h=0 EM anchor was lost during action deduplication")
    weights = np.asarray(tuple(float(value) for value in coefficients), dtype=np.float64)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("optimized selector requires finite coefficient vector")

    scored: list[tuple[float, int, ActionCandidate]] = []
    for candidate in candidates:
        if candidate.requested_h == 0:
            continue
        features = np.asarray(candidate.features, dtype=np.float64)
        if features.shape != weights.shape:
            raise ValueError("optimized candidate feature width differs from coefficients")
        delta = float(np.dot(weights, features))
        scored.append((delta, candidate.requested_h, candidate))
    if not scored:
        return SelectorDecision(
            baseline.destination, 0, 0.0, variant, tuple(candidates)
        )
    delta, _h, chosen = min(scored, key=lambda item: (item[0], item[1]))
    if delta >= 0.0:
        return SelectorDecision(
            baseline.destination, 0, 0.0, variant, tuple(candidates)
        )
    return SelectorDecision(
        chosen.destination,
        chosen.requested_h,
        delta,
        variant,
        tuple(candidates),
    )


def select_r_action_optimized(
    observation: PublicObservation,
    *,
    method: str,
    max_tiers: int,
) -> SelectorDecision:
    """Optional M2 kernel; reference selectors above remain unchanged.

    One raw-rank summary is reused across H=0/2/5/10/20. R/R_CONST construct
    only the four frozen R features. R_NO_HISTORY uses only the saved r01 fit
    coefficients and never calls a fit routine.
    """
    statistics = _m2_public_state_statistics(observation, max_tiers=max_tiers)
    if method == "R":
        reliability_error = 1.0 - observation.q_t
        coefficients = FROZEN_R_COEFFICIENTS
    elif method == "R_CONST":
        reliability_error = 1.0 - observation.q_prior
        coefficients = FROZEN_R_COEFFICIENTS
    elif method == "R_NO_HISTORY":
        reliability_error = None
        coefficients = M2_R_NO_HISTORY_COEFFICIENTS
    else:
        raise ValueError(f"unknown optimized history-reliability method {method!r}")
    candidates = _m2_candidates(statistics, reliability_error=reliability_error)
    return _m2_select_scored_candidates(
        candidates,
        coefficients=coefficients,
        variant="R" if method == "R_CONST" else method,
    )


__all__ = [
    "FROZEN_R_COEFFICIENTS",
    "FROZEN_R_Q_PRIOR",
    "M2_R_NO_HISTORY_COEFFICIENTS",
    "R_NO_HISTORY_EXPECTED_LABEL_ROWS",
    "R_NO_HISTORY_EXPECTED_STATE_COUNT",
    "R_NO_HISTORY_RIDGE_LAMBDA",
    "RidgeFit",
    "fit_r_no_history",
    "observation_with_constant_reliability",
    "require_legal_action",
    "select_r_action",
    "select_em_destination_optimized",
    "select_r_action_optimized",
    "select_r_no_history",
    "validate_r_no_history_label_rows",
]
