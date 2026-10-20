"""PLAN-047 public candidate identity, 74-feature map, and branch labels.

The model-facing functions accept only a public frame and an execution
sequence. Proposal algorithm/horizon/IDs are kept in a separate provenance
record and are never passed to feature construction or scoring.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from R.scrp.h20_r_c2 import c2_global_features
from R.scrp.h20_r_data import HORIZONS, Q_PRIOR, c1_destination_relations


ALGORITHMS = ("EM", "GP", "ERI", "EG")
WINDOW_SIZES = (1, 2, 5, 10, 20)
REPEATS = (0, 1)
FEATURE_WIDTH = 74
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CandidateOrigin:
    """Evaluator-only proposal provenance; never consumed by the scorer."""

    algorithm: str
    horizon: int
    proposal_id: str


@dataclass(frozen=True)
class CandidateProposal:
    origin: CandidateOrigin
    destinations: tuple[int, ...]


@dataclass(frozen=True)
class CandidateSequence:
    destinations: tuple[int, ...]
    origins: tuple[CandidateOrigin, ...]
    is_h20_anchor: bool

    @property
    def horizon_for_control(self) -> int:
        """Canonical source bit for the horizon/q-only control."""
        if self.is_h20_anchor:
            return 20
        return min(int(origin.horizon) for origin in self.origins)


@dataclass(frozen=True)
class NativeCandidatePool:
    algorithm: str
    candidates: tuple[CandidateSequence, ...]
    anchor_sequence: tuple[int, ...]


@dataclass(frozen=True)
class ScoredSelection:
    algorithm: str
    sequence: tuple[int, ...]
    anchor_sequence: tuple[int, ...]
    score: float
    anchor_score: float
    used_anchor: bool
    scored_sequences: tuple[tuple[tuple[int, ...], float], ...]


@dataclass(frozen=True)
class ContinuationCost:
    algorithm: str
    r_raw: int
    r_over_n: float


@dataclass(frozen=True)
class CandidateBranchLabels:
    sequence: tuple[int, ...]
    forecast_sha256: str
    candidate_relocations: int
    continuations: tuple[ContinuationCost, ...]
    em0_sensitivity: ContinuationCost

    @property
    def mix4_r_raw(self) -> float:
        """Primary shared R_raw target across the four fixed-h=20 actors."""
        return sum(row.r_raw for row in self.continuations) / float(len(ALGORITHMS))

    @property
    def mix4_r_over_n(self) -> float:
        """Primary shared normalized target across the four fixed-h=20 actors."""
        return sum(row.r_over_n for row in self.continuations) / float(len(ALGORITHMS))

    def native_diagonal(self, proposer: str) -> ContinuationCost:
        if proposer not in ALGORITHMS:
            raise ValueError(f"unknown proposer algorithm: {proposer}")
        return next(row for row in self.continuations if row.algorithm == proposer)


@dataclass(frozen=True)
class _VirtualObservation:
    rank_tensor: np.ndarray
    occupied_mask: np.ndarray
    legal_destination_mask: np.ndarray
    history_hits: tuple[bool, ...]
    q_prior: float
    q_t: float


@dataclass(frozen=True)
class _VirtualFrame:
    observation: _VirtualObservation
    legal_destinations: tuple[int, ...]
    max_tiers: int
    crane_position: int | None
    crane_time: float
    retrieval_count: int
    relocation_count: int


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value:
        raise ValueError(f"{name} must be an integer")
    return result


def semantic_universal_time_window_seed(
    layout_fingerprint: str, *, window_size: int, repeat: int
) -> int:
    """Return a new namespace seed without changing PLAN-046's seed helper."""
    if not isinstance(layout_fingerprint, str) or not _HEX64.fullmatch(layout_fingerprint):
        raise ValueError("layout_fingerprint must be lowercase SHA-256 hex")
    width = _strict_int(window_size, name="window_size")
    repetition = _strict_int(repeat, name="repeat")
    if width not in WINDOW_SIZES or repetition not in REPEATS:
        raise ValueError("window size or repeat is outside the registered grid")
    payload = {
        "protocol_id": "universal_r_utility_v1",
        "layout_fingerprint": layout_fingerprint,
        "window_size": width,
        "repeat": repetition,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _validate_sequence(frame: object, destinations: Sequence[int]) -> tuple[int, ...]:
    sequence = tuple(_strict_int(value, name="destination") for value in destinations)
    if not sequence:
        raise ValueError("candidate execution sequence must not be empty")
    # Feature construction also validates rank/occupancy/q/legal consistency.
    virtual: object = frame
    for move_index, destination in enumerate(sequence):
        legal = tuple(int(item) for item in getattr(virtual, "legal_destinations"))
        if destination not in legal:
            raise ValueError(f"candidate move {move_index} is not currently legal")
        virtual = _virtual_after_public_move(virtual, destination)
        target = np.argwhere(
            np.asarray(virtual.observation.occupied_mask, dtype=np.bool_)
            & (np.asarray(virtual.observation.rank_tensor) == 1)
        )
        if target.shape != (1, 2):
            raise ValueError("virtual public move lost the unique current target")
        source, tier = (int(item) for item in target[0])
        heights = np.asarray(virtual.observation.occupied_mask, dtype=np.bool_).sum(axis=1)
        exposed = int(heights[source]) == tier + 1
        if exposed and move_index < len(sequence) - 1:
            raise ValueError("multi-move candidate continues after exposing its target")
        if not exposed and move_index == len(sequence) - 1 and len(sequence) > 1:
            raise ValueError("multi-move EG candidate must finish its current blocker group")
    return sequence


def _current_target_blocker_count(public_frame: object) -> int:
    observation = getattr(public_frame, "observation", None)
    if observation is None:
        raise ValueError("EG proposal validation requires a public decision frame")
    ranks = np.asarray(observation.rank_tensor)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public rank/occupancy shapes differ")
    targets = np.argwhere(occupied & (ranks == 1))
    if targets.shape != (1, 2):
        raise ValueError("public frame must contain one current target")
    source, target_tier = (int(value) for value in targets[0])
    height = int(occupied[source].sum())
    blocker_count = height - target_tier - 1
    if blocker_count <= 0:
        raise ValueError("EG proposal requires a blocked current target")
    return blocker_count


def build_native_candidate_pools(
    public_frame: object,
    proposals: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    proposal_ids: Mapping[tuple[str, int], str] | None = None,
) -> tuple[NativeCandidatePool, ...]:
    """Deduplicate by full sequence, retaining origins and h=20 anchor identity."""
    if set(proposals) != set(ALGORITHMS):
        raise ValueError(f"proposal algorithms must be exactly {ALGORITHMS}")
    ids = {} if proposal_ids is None else dict(proposal_ids)
    pools: list[NativeCandidatePool] = []
    for algorithm in ALGORITHMS:
        by_horizon = proposals[algorithm]
        if set(int(h) for h in by_horizon) != set(HORIZONS) or len(by_horizon) != len(HORIZONS):
            raise ValueError(f"{algorithm} must supply exactly one proposal for each horizon")
        normalized: dict[int, tuple[int, ...]] = {}
        origins_by_sequence: dict[tuple[int, ...], list[CandidateOrigin]] = {}
        for horizon in HORIZONS:
            if horizon not in by_horizon:
                raise ValueError(f"{algorithm} proposal horizon key must be integer {horizon}")
            sequence = _validate_sequence(public_frame, by_horizon[horizon])
            if algorithm != "EG" and len(sequence) != 1:
                raise ValueError("EM, GP, and ERI proposals must be single-destination sequences")
            if algorithm == "EG" and len(sequence) != _current_target_blocker_count(public_frame):
                raise ValueError(
                    "EG proposal must contain the complete current blocker group"
                )
            normalized[horizon] = sequence
            origin = CandidateOrigin(
                algorithm=algorithm,
                horizon=horizon,
                proposal_id=str(ids.get((algorithm, horizon), f"{algorithm}:h{horizon}")),
            )
            origins_by_sequence.setdefault(sequence, []).append(origin)
        anchor = normalized[20]
        candidates = tuple(
            CandidateSequence(
                destinations=sequence,
                origins=tuple(sorted(origins, key=lambda item: (item.horizon, item.proposal_id))),
                is_h20_anchor=(sequence == anchor),
            )
            for sequence, origins in sorted(origins_by_sequence.items())
        )
        if sum(row.is_h20_anchor for row in candidates) != 1:
            raise AssertionError("the h=20 full sequence must identify exactly one pool anchor")
        pools.append(NativeCandidatePool(algorithm, candidates, anchor))
    return tuple(pools)


def _virtual_after_public_move(frame: object, destination: int) -> _VirtualFrame:
    observation = getattr(frame, "observation", None)
    if observation is None:
        raise ValueError("candidate features require a public decision frame")
    ranks = np.asarray(observation.rank_tensor, dtype=np.int32).copy()
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_).copy()
    legal_mask = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    tiers = int(frame.max_tiers)
    if ranks.ndim != 2 or ranks.shape != occupied.shape or ranks.shape[1] != tiers:
        raise ValueError("public rank/occupancy shape differs from max tiers")
    legal = tuple(int(item) for item in np.flatnonzero(legal_mask))
    if legal != tuple(int(item) for item in frame.legal_destinations):
        raise ValueError("public legal tuple differs from its mask")
    if destination not in legal:
        raise ValueError("candidate move is not legal in its public frame")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty cells must have rank 0")
    if np.any(occupied & ((ranks < 1) | (ranks > 22))):
        raise ValueError("occupied ranks must be known or the public 22 sentinel")
    heights = occupied.sum(axis=1).astype(np.int64)
    target = np.argwhere(occupied & (ranks == 1))
    if target.shape != (1, 2):
        raise ValueError("public frame must contain exactly one current target")
    source, target_tier = (int(item) for item in target[0])
    if heights[source] <= target_tier + 1:
        raise ValueError("candidate feature requires a blocked current target")
    if heights[destination] >= tiers:
        raise ValueError("candidate destination is full")

    source_top = int(heights[source] - 1)
    destination_top = int(heights[destination])
    moved_rank = int(ranks[source, source_top])
    if moved_rank == 1:
        raise ValueError("current target cannot be the relocated blocker")
    ranks[source, source_top] = 0
    occupied[source, source_top] = False
    ranks[destination, destination_top] = moved_rank
    occupied[destination, destination_top] = True
    next_heights = heights.copy()
    next_heights[source] -= 1
    next_heights[destination] += 1
    next_legal_mask = np.asarray(
        [index != source and next_heights[index] < tiers for index in range(len(next_heights))],
        dtype=np.bool_,
    )
    next_legal = tuple(int(item) for item in np.flatnonzero(next_legal_mask))
    return _VirtualFrame(
        observation=_VirtualObservation(
            rank_tensor=ranks,
            occupied_mask=occupied,
            legal_destination_mask=next_legal_mask,
            history_hits=tuple(bool(item) for item in observation.history_hits),
            q_prior=float(observation.q_prior),
            q_t=float(observation.q_t),
        ),
        legal_destinations=next_legal,
        max_tiers=tiers,
        crane_position=getattr(frame, "crane_position", None),
        crane_time=float(getattr(frame, "crane_time", 0.0)),
        retrieval_count=int(getattr(frame, "retrieval_count", 0)),
        relocation_count=int(getattr(frame, "relocation_count", 0)) + 1,
    )


def _base37(public_frame: object, candidate_sequence: Sequence[int]) -> np.ndarray:
    sequence = tuple(_strict_int(value, name="destination") for value in candidate_sequence)
    if not sequence:
        raise ValueError("candidate execution sequence must not be empty")
    _validate_sequence(public_frame, sequence)
    root_global = np.asarray(c2_global_features(public_frame), dtype=np.float64)
    virtual = public_frame
    per_move: list[np.ndarray] = []
    for destination in sequence:
        relations = c1_destination_relations(virtual)
        if destination not in relations:
            raise ValueError("candidate destination is absent from public relation features")
        per_move.append(np.asarray(relations[destination], dtype=np.float64))
        virtual = _virtual_after_public_move(virtual, int(destination))
    matrix = np.stack(per_move, axis=0)
    positions = np.arange(1, len(sequence) + 1, dtype=np.float64)
    weighted = np.average(matrix, axis=0, weights=positions)
    base = np.concatenate(
        (
            root_global,
            matrix[0],
            matrix[-1],
            matrix.mean(axis=0),
            matrix.max(axis=0),
            weighted,
            np.asarray([len(sequence) / float(public_frame.max_tiers)], dtype=np.float64),
        )
    )
    if base.shape != (37,) or not np.isfinite(base).all():
        raise AssertionError("universal-R base feature construction is invalid")
    return base


def universal_r_utility_features(
    public_frame: object, candidate_sequence: Sequence[int]
) -> np.ndarray:
    """Build the fixed 74 public features; no proposal/source metadata is read."""
    base = _base37(public_frame, candidate_sequence)
    q = float(public_frame.observation.q_t)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("public q_t must be finite and in [0,1]")
    result = np.concatenate((base, (1.0 - q) * base))
    if result.shape != (FEATURE_WIDTH,) or not np.isfinite(result).all():
        raise AssertionError("universal-R feature construction returned invalid values")
    return result


def qfixed_utility_features(
    public_frame: object, candidate_sequence: Sequence[int]
) -> np.ndarray:
    """Same features with only the public reliability value fixed at q_prior."""
    base = _base37(public_frame, candidate_sequence)
    q_prior = float(public_frame.observation.q_prior)
    if not math.isfinite(q_prior) or abs(q_prior - Q_PRIOR) > 1e-15:
        raise ValueError("public q_prior differs from the frozen 5183/9920 protocol")
    return np.concatenate((base, (1.0 - q_prior) * base))


def horizon_q_only_features(
    public_frame: object,
    candidate_sequence: Sequence[int],
    source_horizon: int,
) -> np.ndarray:
    """Registered control-only source-horizon bits in the fixed 74-wide input."""
    _validate_sequence(public_frame, candidate_sequence)
    if source_horizon not in HORIZONS:
        raise ValueError(f"source horizon must be one of {HORIZONS}")
    q = float(public_frame.observation.q_t)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("public q_t must be finite and in [0,1]")
    output = np.zeros(FEATURE_WIDTH, dtype=np.float64)
    index = HORIZONS.index(int(source_horizon))
    output[index] = 1.0
    output[37 + index] = 1.0 - q
    return output


def score_candidate(
    public_frame: object,
    candidate_sequence: Sequence[int],
    scorer: Callable[[np.ndarray], float],
) -> float:
    """The shared public score boundary: only the frame and full sequence build x."""
    features = universal_r_utility_features(public_frame, candidate_sequence)
    value = float(scorer(features))
    if not math.isfinite(value):
        raise ValueError("candidate score must be finite")
    return value


def select_strictly_below_h20(
    public_frame: object,
    pool: NativeCandidatePool,
    scorer: Callable[[np.ndarray], float],
) -> ScoredSelection:
    """Select only a strict predicted improvement; exact tie retains h=20."""
    scores = tuple(
        (candidate.destinations, score_candidate(public_frame, candidate.destinations, scorer))
        for candidate in pool.candidates
    )
    score_by_sequence = dict(scores)
    anchor_score = score_by_sequence[pool.anchor_sequence]
    better = [row for row in scores if row[0] != pool.anchor_sequence and row[1] < anchor_score]
    if not better:
        return ScoredSelection(
            algorithm=pool.algorithm,
            sequence=pool.anchor_sequence,
            anchor_sequence=pool.anchor_sequence,
            score=anchor_score,
            anchor_score=anchor_score,
            used_anchor=True,
            scored_sequences=tuple(sorted(scores)),
        )
    sequence, score = min(better, key=lambda row: (row[1], row[0]))
    return ScoredSelection(
        algorithm=pool.algorithm,
        sequence=sequence,
        anchor_sequence=pool.anchor_sequence,
        score=score,
        anchor_score=anchor_score,
        used_anchor=False,
        scored_sequences=tuple(sorted(scores)),
    )


def candidate_label_cache_key(
    state_fingerprint: str, forecast_sha256: str, candidate_sequence: Sequence[int]
) -> str:
    """Origin-free identity for one physical state/F/complete-sequence label."""
    if not _HEX64.fullmatch(state_fingerprint) or not _HEX64.fullmatch(forecast_sha256):
        raise ValueError("state and forecast identities must be lowercase SHA-256 hex")
    sequence = tuple(_strict_int(value, name="destination") for value in candidate_sequence)
    if not sequence:
        raise ValueError("candidate sequence must not be empty")
    payload = {
        "state_fingerprint": state_fingerprint,
        "forecast_sha256": forecast_sha256,
        "candidate_sequence": sequence,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _actor_action(actor: Any, public_frame: object) -> int:
    if callable(actor):
        action = actor(public_frame)
    else:
        selector = getattr(actor, "select_destination", None)
        if not callable(selector):
            raise TypeError("continuation actor must be callable or expose select_destination")
        action = selector(public_frame)
    return _strict_int(action, name="continuation action")


def run_candidate_branches(
    episode: Any,
    source_environment: Any,
    source_provider: Any,
    public_frame: object,
    candidate_sequence: Sequence[int],
    *,
    continuation_actor_factories: Mapping[str, Callable[[], Any]],
    em0_actor_factory: Callable[[], Any],
    check_budget: Callable[[], None] | None = None,
    private_trace_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> CandidateBranchLabels:
    """Execute a full sequence, then isolated h=20 and EM0 continuations.

    The caller supplies a private trace sink for streaming each action row.
    Actors receive only detached public frames. A new actor is constructed
    after the candidate sequence for every continuation, so an EG candidate's
    group cache cannot leak into EG20 continuation.
    """
    if set(continuation_actor_factories) != set(ALGORITHMS):
        raise ValueError(f"continuation factories must be exactly {ALGORITHMS}")
    sequence = _validate_sequence(public_frame, candidate_sequence)
    from R.scrp.h20_r_bank import (  # local import avoids widening model-facing inputs
        _forecast_hash,
        _private_physical_stacks,
        _private_signature,
        _public_frame,
        _sync_retrieval_events_with_audit,
    )

    source_signature = _private_signature(source_environment, source_provider)
    source_start_relocations = int(source_environment.relocation_count)
    forecast = tuple(int(value) for value in source_provider._forecast_future)
    forecast_sha = _forecast_hash(forecast)
    branch_environment = deepcopy(source_environment)
    branch_provider = deepcopy(source_provider)
    branch_start_relocations = int(branch_environment.relocation_count)
    max_steps = int(episode.num_containers) * max(1, int(episode.max_tiers) - 1)
    candidate_trace_count = 0

    def advance(environment: Any, provider: Any, action: int, *, phase: str, algorithm: str | None) -> None:
        nonlocal candidate_trace_count
        if check_budget is not None:
            check_budget()
        frame = _public_frame(episode, environment, provider)
        legal = tuple(int(value) for value in frame.legal_destinations)
        if action not in legal:
            raise ValueError("branch action is outside the current public legal mask")
        before_stacks = _private_physical_stacks(environment)
        before_retrieval = int(environment.state.retrieval_count)
        before_relocations = int(environment.relocation_count)
        before_q = float(provider.history.q_t)
        before_hits = tuple(bool(value) for value in provider.history.hits)
        environment.step(action)
        after_retrieval = int(environment.state.retrieval_count)
        after_relocations = int(environment.relocation_count)
        reveals = _sync_retrieval_events_with_audit(
            provider, episode.retrieval_sequence, before_retrieval, after_retrieval
        )
        if after_relocations != before_relocations + 1:
            raise AssertionError("one candidate/continuation action must cause one relocation")
        if not reveals and float(provider.history.q_t) != before_q:
            raise AssertionError("q_t changed without a new reveal")
        after_stacks = _private_physical_stacks(environment)
        live_count = sum(len(stack) for stack in after_stacks)
        if live_count + after_retrieval != int(episode.num_containers):
            raise AssertionError("branch transition did not conserve all containers")
        if private_trace_sink is not None:
            private_trace_sink(
                {
                    "phase": phase,
                    "continuation_algorithm_evaluator_only": algorithm,
                    "candidate_sequence": list(sequence),
                    "action_destination_private": int(action),
                    "legal_destinations_public": list(legal),
                    "physical_stacks_before_private": before_stacks,
                    "physical_stacks_after_private": after_stacks,
                    "retrieval_count_before": before_retrieval,
                    "retrieval_count_after": after_retrieval,
                    "relocation_count_before": before_relocations,
                    "relocation_count_after": after_relocations,
                    "q_t_before_action": before_q,
                    "q_t_after_reveals": float(provider.history.q_t),
                    "history_hits_before_private": list(before_hits),
                    "history_hits_after_private": list(provider.history.hits),
                    "reveal_events_private": reveals,
                    "forecast_sha256_evaluator_only": forecast_sha,
                }
            )
        if phase == "candidate_sequence":
            candidate_trace_count += 1

    for destination in sequence:
        if branch_environment.state.terminated:
            raise ValueError("candidate sequence continues after terminal state")
        frame = _public_frame(episode, branch_environment, branch_provider)
        if int(destination) not in tuple(frame.legal_destinations):
            raise ValueError("candidate sequence became illegal in its physical branch")
        advance(
            branch_environment,
            branch_provider,
            int(destination),
            phase="candidate_sequence",
            algorithm=None,
        )

    if tuple(int(value) for value in branch_provider._forecast_future) != forecast:
        raise AssertionError("candidate execution changed the shared fixed forecast")

    def continue_to_terminal(
        algorithm: str, actor_factory: Callable[[], Any]
    ) -> ContinuationCost:
        environment = deepcopy(branch_environment)
        provider = deepcopy(branch_provider)
        if tuple(int(value) for value in provider._forecast_future) != forecast:
            raise AssertionError("continuation clone does not share the registered fixed F")
        actor = actor_factory()
        reset = getattr(actor, "reset", None)
        if callable(reset):
            reset()
        steps = 0
        while not environment.state.terminated:
            if check_budget is not None:
                check_budget()
            frame = _public_frame(episode, environment, provider)
            if not frame.legal_destinations:
                raise RuntimeError("nonterminal continuation has no legal destination")
            action = _actor_action(actor, frame)
            phase = "fixed_h20_continuation" if algorithm in ALGORITHMS else "em0_sensitivity"
            advance(environment, provider, action, phase=phase, algorithm=algorithm)
            steps += 1
            if steps + len(sequence) > max_steps:
                raise RuntimeError("candidate plus continuation exceeded physical safety bound")
        if int(environment.state.retrieval_count) != int(episode.num_containers):
            raise AssertionError("continuation did not retrieve every container")
        total = int(environment.relocation_count) - source_start_relocations
        if total < len(sequence):
            raise AssertionError("label omitted candidate-sequence relocation cost")
        return ContinuationCost(
            algorithm=algorithm,
            r_raw=total,
            r_over_n=total / float(episode.num_containers),
        )

    continuation_rows = tuple(
        continue_to_terminal(
            algorithm, continuation_actor_factories[algorithm]
        )
        for algorithm in ALGORITHMS
    )
    # Every unique sequence receives this sensitivity regardless of its proposer.
    em0_sensitivity = continue_to_terminal("EM0", em0_actor_factory)
    if source_signature != _private_signature(source_environment, source_provider):
        raise AssertionError("isolated candidate branches mutated the source state/provider")
    if candidate_trace_count != len(sequence):
        raise AssertionError("candidate sequence trace count differs from its execution length")
    return CandidateBranchLabels(
        sequence=sequence,
        forecast_sha256=forecast_sha,
        candidate_relocations=candidate_trace_count,
        continuations=continuation_rows,
        em0_sensitivity=em0_sensitivity,
    )


__all__ = (
    "ALGORITHMS",
    "CandidateBranchLabels",
    "CandidateOrigin",
    "CandidateProposal",
    "CandidateSequence",
    "ContinuationCost",
    "FEATURE_WIDTH",
    "NativeCandidatePool",
    "ScoredSelection",
    "WINDOW_SIZES",
    "build_native_candidate_pools",
    "candidate_label_cache_key",
    "horizon_q_only_features",
    "qfixed_utility_features",
    "run_candidate_branches",
    "score_candidate",
    "select_strictly_below_h20",
    "semantic_universal_time_window_seed",
    "universal_r_utility_features",
)
