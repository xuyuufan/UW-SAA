"""PLAN-051 adapters for five frozen R policies and their native H20 anchor.

This module contains no evaluator metadata, seed, condition, provider, truth, or
environment interface.  It receives only a detached public decision frame.
The original OLD_FROZEN_R implementation is delegated unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ALGORITHMS = ("EM", "GP", "ERI", "EG")
POLICIES = (
    "OLD_FROZEN_R",
    "R_SOURCE_FULL",
    "R_UTILITY_RIDGE",
    "R_QUAD_H",
    "R_ORDER_MAIN_TAU0_FULL",
    "NATIVE20",
)
HORIZONS = (0, 2, 5, 10, 20)
_CRANE_TRAVEL_TIME = 1.2
_PICK_DROP_TIME = 30.0


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
            "history_hits", "q_prior", "q_t", "r_order_summary",
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


def public_frame_record(frame: object) -> dict[str, Any]:
    """Serialize the complete actor-visible frame, including both crane fields."""
    require_public_frame(frame)
    obs = frame.observation
    summary = getattr(obs, "r_order_summary", None)
    summary_record = None
    if summary is not None:
        summary_record = {
            "q": float(summary.q),
            "displacement": float(summary.displacement),
            "coverage": float(summary.coverage),
            "retrieval_fraction": float(summary.retrieval_fraction),
            "records": [float(value) for value in summary.records],
        }
    return {
        "rank_tensor": np.asarray(obs.rank_tensor, dtype=np.int32).tolist(),
        "occupied_mask": np.asarray(obs.occupied_mask, dtype=np.bool_).tolist(),
        "legal_destination_mask": np.asarray(obs.legal_destination_mask, dtype=np.bool_).tolist(),
        "legal_destinations": [int(value) for value in frame.legal_destinations],
        "max_tiers": int(frame.max_tiers),
        "history_hits": [bool(value) for value in obs.history_hits],
        "q_prior": float(obs.q_prior),
        "q_t": float(obs.q_t),
        "retrieval_count": int(frame.retrieval_count),
        "relocation_count": int(frame.relocation_count),
        "crane_position": None if frame.crane_position is None else int(frame.crane_position),
        "crane_time": float(frame.crane_time),
        "r_order_summary": summary_record,
    }


def strict_public_frame_from_record(row: Mapping[str, Any]) -> object:
    """Restore a recorded frame; never infer or default missing crane values."""
    required = {"crane_position", "crane_time"}
    if not required.issubset(row):
        raise ValueError("stored public frame lacks crane_position/crane_time")
    from R.scrp.r_order_public import PublicHistorySummary, _PublicObservation
    from experiments.noise_aware_reference_p0 import PublicDecisionFrame
    from scrp.noise_aware_forecast import PublicObservation

    position = row["crane_position"]
    if position is not None and (isinstance(position, bool) or int(position) != position or int(position) < 0):
        raise ValueError("stored crane_position is invalid")
    crane_time = float(row["crane_time"])
    if not math.isfinite(crane_time) or crane_time < 0:
        raise ValueError("stored crane_time is invalid")
    summary_row = row.get("r_order_summary")
    summary = None if summary_row is None else PublicHistorySummary(
        q=float(summary_row["q"]),
        displacement=float(summary_row["displacement"]),
        coverage=float(summary_row["coverage"]),
        retrieval_fraction=float(summary_row["retrieval_fraction"]),
        records=tuple(float(value) for value in summary_row["records"]),
    )
    observation_kwargs = {
        "rank_tensor": np.asarray(row["rank_tensor"], dtype=np.int32),
        "occupied_mask": np.asarray(row["occupied_mask"], dtype=np.bool_),
        "legal_destination_mask": np.asarray(row["legal_destination_mask"], dtype=np.bool_),
        "history_hits": tuple(bool(value) for value in row["history_hits"]),
        "q_prior": float(row["q_prior"]),
        "q_t": float(row["q_t"]),
    }
    # The registered R-Order wrapper adds only its own causal public summary.
    # Preserve its explicit observation type on round trips rather than trying
    # to add an unsupported field to the frozen base PublicObservation.
    obs = (
        PublicObservation(**observation_kwargs)
        if summary is None
        else _PublicObservation(**observation_kwargs, r_order_summary=summary)
    )
    frame = PublicDecisionFrame(
        observation=obs,
        legal_destinations=tuple(int(value) for value in row["legal_destinations"]),
        max_tiers=int(row["max_tiers"]),
        crane_position=None if position is None else int(position),
        crane_time=crane_time,
        retrieval_count=int(row["retrieval_count"]),
        relocation_count=int(row["relocation_count"]),
    )
    require_public_frame(frame)
    return frame


@dataclass(frozen=True)
class RidgeScorer:
    coefficients: tuple[float, ...]

    def __call__(self, features: Sequence[float]) -> float:
        vector = np.asarray(features, dtype=np.float64)
        weights = np.asarray(self.coefficients, dtype=np.float64)
        if vector.shape != (74,) or weights.shape != (74,) or not np.isfinite(vector).all() or not np.isfinite(weights).all():
            raise ValueError("frozen PLAN-047 Ridge requires 74 finite features and coefficients")
        return float(np.dot(weights, vector))


class _SequenceCommitter:
    """Commit an already-selected full EG blocker group and verify every step."""

    def __init__(self, algorithm: str, selector: Callable[[object], tuple[tuple[int, ...], dict[str, Any]]]):
        self.algorithm = algorithm
        self.selector = selector
        self.sequence: tuple[int, ...] | None = None
        self.next_index = 0
        self.expected_full: Any | None = None
        self._cached_moves = 0

    @property
    def is_group_start(self) -> bool:
        return self.algorithm != "EG" or self.sequence is None

    @property
    def cached_moves(self) -> int:
        return self._cached_moves

    def reset(self) -> None:
        self._clear_pending()
        self._cached_moves = 0

    def _clear_pending(self) -> None:
        self.sequence = None
        self.next_index = 0
        self.expected_full = None

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        require_public_frame(frame)
        from R.scrp.r_order_public import validate_sequence

        if self.algorithm == "EG" and self.sequence is not None:
            from R.scrp.r_eri_eg_public import _after_move, _assert_snapshot

            if self.expected_full is None:
                raise RuntimeError("EG sequence cache lost its expected public frame")
            _assert_snapshot(frame, self.expected_full)
            action = int(self.sequence[self.next_index])
            self._cached_moves += 1
            self.next_index += 1
            cached_trace = {
                "group_start": False,
                "cached_move": True,
                "committed_sequence": list(self.sequence),
                "sequence_index": self.next_index,
            }
            if self.next_index < len(self.sequence):
                self.expected_full = _after_move(
                    frame, action,
                    crane_travel_time=_CRANE_TRAVEL_TIME,
                    pick_drop_time=_PICK_DROP_TIME,
                )
            else:
                self._clear_pending()
            return action, cached_trace

        sequence, trace = self.selector(frame)
        sequence = validate_sequence(frame, sequence, self.algorithm)
        if self.algorithm != "EG":
            if len(sequence) != 1:
                raise ValueError("non-EG R policies must commit one destination")
            return sequence[0], {**trace, "group_start": True, "cached_move": False}
        if self.sequence is not None:
            raise RuntimeError("EG selector attempted to replace a committed blocker group")
        self.sequence = sequence
        self.next_index = 1
        if len(sequence) > 1:
            from R.scrp.r_eri_eg_public import _after_move

            self.expected_full = _after_move(
                frame, sequence[0],
                crane_travel_time=_CRANE_TRAVEL_TIME,
                pick_drop_time=_PICK_DROP_TIME,
            )
        else:
            self._clear_pending()
        return sequence[0], {**trace, "group_start": True, "cached_move": False, "committed_sequence": list(sequence)}


class FiveRPolicy:
    """One frozen PLAN-051 policy instance for a single native proposer."""

    def __init__(
        self,
        method: str,
        algorithm: str,
        runtime: Mapping[str, Any],
        gp_rule: Mapping[str, Any],
        *,
        ridge_scorer: RidgeScorer | None = None,
        rates: Any | None = None,
    ) -> None:
        if method not in POLICIES or algorithm not in ALGORITHMS:
            raise ValueError("policy or solver is outside PLAN-051 registration")
        self.method, self.algorithm = method, algorithm
        self.runtime, self.gp_rule = runtime, gp_rule
        self._delegate: Any | None = None
        self._committer: _SequenceCommitter | None = None
        if method in ("OLD_FROZEN_R", "NATIVE20"):
            from R.scrp.r_order_actor import FixedHorizonActor, FrozenRActor

            self._delegate = (
                FrozenRActor(algorithm, runtime, gp_rule)
                if method == "OLD_FROZEN_R"
                else FixedHorizonActor(algorithm, 20, runtime, gp_rule)
            )
        else:
            if method == "R_UTILITY_RIDGE" and ridge_scorer is None:
                raise ValueError("frozen Ridge artifact is required")
            if method == "R_ORDER_MAIN_TAU0_FULL" and rates is None:
                raise ValueError("frozen r_order_main model is required")
            self.ridge_scorer = ridge_scorer
            self.rates = rates
            self._committer = _SequenceCommitter(algorithm, self._select_sequence)

    @property
    def is_group_start(self) -> bool:
        if self._delegate is not None:
            return bool(getattr(self._delegate, "is_group_start", True))
        return bool(self._committer is None or self._committer.is_group_start)

    @property
    def cached_moves(self) -> int:
        if self._delegate is not None:
            return int(getattr(self._delegate, "cached_moves", 0))
        return 0

    def reset(self) -> None:
        if self._delegate is not None and callable(getattr(self._delegate, "reset", None)):
            self._delegate.reset()
        if self._committer is not None:
            self._committer.reset()

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        require_public_frame(frame)
        if self._delegate is not None:
            if self.method == "NATIVE20":
                action, trace = self._delegate.select(frame)
            else:
                action, trace = self._delegate.select(frame)
            return int(action), dict(trace)
        if self._committer is None:
            raise AssertionError("candidate policy has no sequence committer")
        return self._committer.select(frame)

    def _select_sequence(self, frame: object) -> tuple[tuple[int, ...], dict[str, Any]]:
        require_public_frame(frame)
        if self.method == "R_SOURCE_FULL":
            from R.scrp.cross_solver_reliability_r import _m2_select_scored_candidates
            from R.scrp.history_reliability_r import FROZEN_R_COEFFICIENTS
            from scrp.noise_aware_em import ActionCandidate
            from R.scrp.r_order_actor import build_native_proposals

            proposals = build_native_proposals(frame, self.algorithm, self.runtime, self.gp_rule)
            q = float(frame.observation.q_t)
            candidates = tuple(
                ActionCandidate(
                    destination=int(proposals[h][0]),
                    requested_h=int(h),
                    features=(1.0, h / 20.0, 1.0 - q, (h / 20.0) * (1.0 - q)),
                )
                for h in HORIZONS
            )
            decision = _m2_select_scored_candidates(
                candidates, coefficients=FROZEN_R_COEFFICIENTS, variant="R_SOURCE_FULL"
            )
            sequence = tuple(int(value) for value in proposals[int(decision.requested_h)])
            return sequence, {
                "policy": self.method,
                "q_used": q,
                "selected_h": int(decision.requested_h),
                "delta": float(decision.predicted_delta),
                "source_candidates": [
                    {"h": int(h), "sequence": list(proposals[h])}
                    for h in HORIZONS
                ],
            }
        if self.method == "R_QUAD_H":
            from R.scrp.r_horizon_robustness import all_proxy_scores, choose_r_horizon

            q = float(frame.observation.q_t)
            chosen_h, score = choose_r_horizon(q, rule="quad")
            sequence = _one_native_proposal(
                frame, self.algorithm, int(chosen_h), self.runtime, self.gp_rule
            )
            return sequence, {
                "policy": self.method,
                "q_used": q,
                "selected_h": int(chosen_h),
                "proxy_score": float(score),
                "all_proxy_scores": [[int(h), float(value)] for h, value in all_proxy_scores(q, "quad")],
            }
        if self.method == "R_UTILITY_RIDGE":
            from R.scrp.r_order_actor import build_native_proposals
            from R.scrp.universal_r_utility import (
                CandidateOrigin, CandidateSequence, NativeCandidatePool,
                _validate_sequence, select_strictly_below_h20,
            )

            proposals = build_native_proposals(frame, self.algorithm, self.runtime, self.gp_rule)
            by_sequence: dict[tuple[int, ...], list[Any]] = {}
            for h in HORIZONS:
                sequence = _validate_sequence(frame, proposals[h])
                origin = CandidateOrigin(self.algorithm, int(h), f"{self.algorithm}:h{h}")
                by_sequence.setdefault(sequence, []).append(origin)
            anchor = tuple(int(value) for value in proposals[20])
            candidates = tuple(
                CandidateSequence(
                    destinations=sequence,
                    origins=tuple(sorted(origins, key=lambda row: (row.horizon, row.proposal_id))),
                    is_h20_anchor=(sequence == anchor),
                )
                for sequence, origins in sorted(by_sequence.items())
            )
            if sum(bool(item.is_h20_anchor) for item in candidates) != 1:
                raise AssertionError("R-Utility pool lost its explicit H20 sequence anchor")
            pool = NativeCandidatePool(self.algorithm, candidates, anchor)
            selection = select_strictly_below_h20(frame, pool, self.ridge_scorer)
            return tuple(selection.sequence), {
                "policy": self.method,
                "selected_sequence": list(selection.sequence),
                "anchor_sequence": list(selection.anchor_sequence),
                "score": float(selection.score),
                "anchor_score": float(selection.anchor_score),
                "used_anchor": bool(selection.used_anchor),
                "candidates": [
                    {
                        "sequence": list(item.destinations),
                        "origins": [
                            {"algorithm": row.algorithm, "h": int(row.horizon), "proposal_id": row.proposal_id}
                            for row in item.origins
                        ],
                        "is_h20_anchor": bool(item.is_h20_anchor),
                    }
                    for item in pool.candidates
                ],
            }
        if self.method == "R_ORDER_MAIN_TAU0_FULL":
            from R.scrp.r_order_public import build_candidate_pool, build_risk_proposal, select_candidate
            from R.scrp.r_order_actor import build_native_proposals

            proposals = build_native_proposals(frame, self.algorithm, self.runtime, self.gp_rule)
            risk = build_risk_proposal(frame, self.algorithm, self.rates)
            pool = build_candidate_pool(frame, self.algorithm, proposals, risk)
            decision = select_candidate(frame, pool, self.rates, tau=0.0)
            return tuple(decision.sequence), {
                "policy": self.method,
                "tau": 0.0,
                "selected_sequence": list(decision.sequence),
                "anchor_sequence": list(decision.anchor_sequence),
                "score": float(decision.score),
                "anchor_score": float(decision.anchor_score),
                "used_anchor": bool(decision.used_anchor),
                "candidate_pool": [
                    {"sequence": list(item.destinations), "origins": [
                        {"source": origin.source, "h": origin.horizon, "proposal_id": origin.proposal_id}
                        for origin in item.origins
                    ], "is_h20_anchor": bool(item.is_h20_anchor)}
                    for item in pool.candidates
                ],
            }
        raise ValueError(f"no sequence selector is registered for {self.method}")


def _one_native_proposal(
    frame: object, algorithm: str, horizon: int,
    runtime: Mapping[str, Any], gp_rule: Mapping[str, Any],
) -> tuple[int, ...]:
    if horizon not in HORIZONS:
        raise ValueError("requested H is outside the frozen horizon grid")
    from R.scrp.universal_r_utility_m1 import EMNativeActor, ERINativeActor, GPNativeActor

    if algorithm == "EM":
        return (int(EMNativeActor(runtime, horizon)(frame)),)
    if algorithm == "GP":
        return (int(GPNativeActor(runtime, gp_rule, horizon)(frame)),)
    if algorithm == "ERI":
        return (int(ERINativeActor(runtime, horizon)(frame)),)
    if algorithm == "EG":
        prefix = runtime["truncate_public_frame"](frame, horizon)
        plan = runtime["eri_eg"].build_eg20_plan(prefix)
        return tuple(int(value) for value in plan.destinations_top_down)
    raise ValueError(f"unknown native proposer {algorithm!r}")


__all__ = (
    "ALGORITHMS", "HORIZONS", "POLICIES", "FiveRPolicy", "RidgeScorer",
    "public_frame_record", "require_public_frame", "strict_public_frame_from_record",
)
