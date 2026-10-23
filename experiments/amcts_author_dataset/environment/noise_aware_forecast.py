"""Evaluator forecast generation and detached public observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np


HORIZONS = (0, 2, 5, 10, 20)
MAX_HORIZON = 20


def semantic_forecast_seed(semantic_fingerprint: str, repeat: int) -> int:
    """Stable forecast seed keyed by layout identity and repeat, never by H/method."""
    if not semantic_fingerprint:
        raise ValueError("semantic_fingerprint must be non-empty")
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    payload = f"noise-aware-lookahead-v1|{semantic_fingerprint}|r={int(repeat)}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "little")


@dataclass(frozen=True)
class PublicObservation:
    """Only policy-visible values: relative ranks, masks, and causal history."""

    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float

    def __post_init__(self) -> None:
        ranks = np.asarray(self.rank_tensor, dtype=np.int32).copy()
        occupied = np.asarray(self.occupied_mask, dtype=np.bool_).copy()
        legal = np.asarray(self.legal_destination_mask, dtype=np.bool_).copy()
        if ranks.ndim != 2 or occupied.shape != ranks.shape:
            raise ValueError("rank and occupied tensors must have the same [S,T] shape")
        if legal.shape != (ranks.shape[0],):
            raise ValueError("legal destination mask must have one value per stack")
        if not math.isfinite(self.q_prior) or not 0.0 <= self.q_prior <= 1.0:
            raise ValueError("q_prior must be finite and in [0, 1]")
        if not math.isfinite(self.q_t) or not 0.0 <= self.q_t <= 1.0:
            raise ValueError("q_t must be finite and in [0, 1]")
        if np.any(ranks[~occupied] != 0):
            raise ValueError("empty cells must use rank 0")
        ranks.setflags(write=False)
        occupied.setflags(write=False)
        legal.setflags(write=False)
        object.__setattr__(self, "rank_tensor", ranks)
        object.__setattr__(self, "occupied_mask", occupied)
        object.__setattr__(self, "legal_destination_mask", legal)
        object.__setattr__(self, "history_hits", tuple(bool(x) for x in self.history_hits))


class NextTargetHistory:
    """Causal rolling next-target hit history; raw target IDs are never retained."""

    def __init__(self, q_prior: float = 0.5, window: int = 8) -> None:
        if not math.isfinite(q_prior) or not 0.0 <= q_prior <= 1.0:
            raise ValueError("q_prior must be finite and in [0, 1]")
        if window <= 0:
            raise ValueError("window must be positive")
        self.q_prior = float(q_prior)
        self.window = int(window)
        self._hits: deque[bool] = deque(maxlen=self.window)
        self._total_events = 0

    @property
    def hits(self) -> tuple[bool, ...]:
        return tuple(self._hits)

    @property
    def q_t(self) -> float:
        count = len(self._hits)
        return (2.0 * self.q_prior + sum(self._hits)) / (2.0 + count)

    @property
    def total_events(self) -> int:
        return self._total_events

    def record(self, hit: bool) -> None:
        self._hits.append(bool(hit))
        self._total_events += 1


def _subset_permutation(
    true_future: Sequence[int], p: float, rng: np.random.Generator
) -> tuple[int, ...]:
    """Select positions independently and uniformly permute only selected items."""
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be finite and in [0, 1]")
    values = tuple(int(item) for item in true_future)
    if len(values) != len(set(values)):
        raise ValueError("true future must contain unique tokens")
    selected = np.flatnonzero(rng.random(len(values)) < p)
    result = list(values)
    if len(selected) > 1:
        moved = rng.permutation([values[int(i)] for i in selected])
        for index, token in zip(selected.tolist(), moved.tolist(), strict=True):
            result[index] = int(token)
    return tuple(result)


class ForecastProvider:
    """Evaluator-private truth plus one fixed legal prediction per episode.

    Construct this before the scalar environment: its reset can auto-retrieve
    multiple targets, whose disclosure events are then replayed one at a time
    by the evaluator. Policy code receives only PublicObservation.
    """

    def __init__(
        self,
        true_sequence: Sequence[int],
        *,
        p: float,
        seed: int,
        q_prior: float = 0.5,
        max_horizon: int = MAX_HORIZON,
        forecast_future: Sequence[int] | None = None,
    ) -> None:
        sequence = tuple(int(item) for item in true_sequence)
        if not sequence or len(sequence) != len(set(sequence)):
            raise ValueError("true_sequence must be non-empty and unique")
        if max_horizon < 0:
            raise ValueError("max_horizon must be non-negative")
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p must be finite and in [0, 1]")
        self._true_sequence = sequence
        self.max_horizon = int(max_horizon)
        true_future = sequence[1:]
        if forecast_future is None:
            rng = np.random.Generator(np.random.PCG64(int(seed)))
            forecast = _subset_permutation(true_future, float(p), rng)
        else:
            # Evaluator/test injection only. The policy never receives this value.
            forecast = tuple(int(item) for item in forecast_future)
            if len(forecast) != len(set(forecast)) or set(forecast) != set(true_future):
                raise ValueError("forecast_future must permute the true future tokens")
        self._forecast_future = forecast
        self._current = sequence[0]
        self._retrieved: set[int] = set()
        self._seen: set[int] = {sequence[0]}
        self.history = NextTargetHistory(q_prior=q_prior, window=8)
        self.p_private = float(p)

    @property
    def event_count(self) -> int:
        return self.history.total_events

    @property
    def remaining_future_count(self) -> int:
        return len(self._true_sequence) - len(self._retrieved) - 1

    def observe_reveal(self, newly_revealed_target: int) -> bool:
        """Advance one real disclosure event after the previous target retrieval."""
        if self._current is None:
            raise RuntimeError("cannot observe a reveal after episode termination")
        target = int(newly_revealed_target)
        if target in self._seen or target not in self._true_sequence:
            raise ValueError("newly revealed target is repeated or outside the episode")
        excluded = self._retrieved | {self._current}
        predicted = next(
            (item for item in self._forecast_future if item not in excluded), None
        )
        if predicted is None:
            raise RuntimeError("fixed forecast ended before a target was revealed")
        hit = predicted == target
        self.history.record(hit)
        self._retrieved.add(self._current)
        self._current = target
        self._seen.add(target)
        return hit

    def public_observation(
        self,
        physical_stacks: Sequence[Sequence[int]],
        legal_destinations: Iterable[int],
        *,
        max_tiers: int,
    ) -> PublicObservation:
        """Render opaque physical tokens as relative ranks without exposing IDs."""
        if self._current is None:
            raise RuntimeError("terminated episode has no policy observation")
        if len(physical_stacks) <= 1 or max_tiers <= 0:
            raise ValueError("invalid physical geometry")
        live = {int(item) for stack in physical_stacks for item in stack}
        expected_live = set(self._true_sequence) - self._retrieved
        if live != expected_live:
            raise ValueError("private physical state disagrees with retrieval events")
        if self._current not in live:
            raise ValueError("current target is absent from the physical state")

        future = tuple(
            item
            for item in self._forecast_future
            if item not in self._retrieved and item != self._current
        )
        if len(future) != self.remaining_future_count:
            raise RuntimeError("forecast filtering and event timeline disagree")
        rank_by_token = {self._current: 1}
        for offset, item in enumerate(future[: self.max_horizon]):
            rank_by_token[item] = offset + 2

        stacks = tuple(tuple(int(item) for item in stack) for stack in physical_stacks)
        ranks = np.zeros((len(stacks), max_tiers), dtype=np.int32)
        occupied = np.zeros((len(stacks), max_tiers), dtype=np.bool_)
        unknown_rank = self.max_horizon + 2
        for stack_id, stack in enumerate(stacks):
            if len(stack) > max_tiers:
                raise ValueError("physical stack exceeds max_tiers")
            for tier, item in enumerate(stack):
                occupied[stack_id, tier] = True
                ranks[stack_id, tier] = rank_by_token.get(item, unknown_rank)

        legal = np.zeros(len(stacks), dtype=np.bool_)
        destinations = tuple(int(item) for item in legal_destinations)
        if len(destinations) != len(set(destinations)):
            raise ValueError("legal destination IDs must be unique")
        for stack_id in destinations:
            if not 0 <= stack_id < len(stacks):
                raise ValueError("legal destination ID is outside the bay")
            legal[stack_id] = True
        return PublicObservation(
            rank_tensor=ranks,
            occupied_mask=occupied,
            legal_destination_mask=legal,
            history_hits=self.history.hits,
            q_prior=self.history.q_prior,
            q_t=self.history.q_t,
        )


__all__ = [
    "ForecastProvider",
    "HORIZONS",
    "MAX_HORIZON",
    "NextTargetHistory",
    "PublicObservation",
    "semantic_forecast_seed",
]
