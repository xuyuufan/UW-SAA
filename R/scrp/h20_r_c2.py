"""Public-state C2 horizon gate and fit-only ridge design."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

R_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = R_ROOT.parent


def _install_workspace_paths() -> None:
    for root in (WORKSPACE, R_ROOT, WORKSPACE / "EM", WORKSPACE / "Mijn_RL"):
        value = str(root.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    for package, roots in (
        ("scrp", (R_ROOT, WORKSPACE / "EM", WORKSPACE / "Mijn_RL")),
        ("experiments", (WORKSPACE / "EM", R_ROOT, WORKSPACE / "Mijn_RL")),
    ):
        module = sys.modules.get(package)
        if module is None:
            __import__(package)
            module = sys.modules[package]
        locations = [str((root / package).resolve()) for root in roots]
        module.__path__ = locations
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.submodule_search_locations is not None:
            spec.submodule_search_locations[:] = locations


_install_workspace_paths()

from R.scrp.h20_r_data import HORIZONS, Q_PRIOR, UNKNOWN_RANK  # noqa: E402
from scrp.cross_solver_reliability_r import truncate_public_frame  # noqa: E402
from scrp.noise_aware_forecast import PublicObservation  # noqa: E402
from scrp.scale_diverse_policy import Online2DPriorityAdapter  # noqa: E402
from experiments.noise_aware_reference_p0 import (  # noqa: E402
    PublicDecisionFrame,
    h20_action_from_tensor,
    public_frame_to_legacy_state,
)


GLOBAL_FEATURE_WIDTH = 6
HORIZON_FEATURE_WIDTH = 12
FEATURE_WIDTH = 24
RIDGE_LAMBDA = 1.0
SELECTOR_MODES = ("history", "no_history_fit", "constant_q_intervention")


@dataclass(frozen=True)
class C2FitDesign:
    history_x: np.ndarray
    no_history_x: np.ndarray
    y: np.ndarray
    samples: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class C2HorizonDecision:
    mode: str
    q_used: float
    selected_h: int
    used_h20_fallback: bool
    scores: tuple[tuple[int, float | None], ...]


@dataclass(frozen=True)
class C2ActorCall:
    decision: C2HorizonDecision
    action: int
    censored_frame: PublicDecisionFrame
    solver_invocation_count: int
    gate_ns: int
    censor_ns: int
    bridge_ns: int
    inference_ns: int
    total_actor_ns: int


def c2_global_features(frame: PublicDecisionFrame) -> np.ndarray:
    """Build the registered six public global features, without private fields."""
    observation = frame.observation
    raw_ranks = np.asarray(observation.rank_tensor)
    if raw_ranks.ndim != 2 or not np.issubdtype(raw_ranks.dtype, np.integer):
        raise ValueError("C2 requires a two-dimensional integer public rank tensor")
    ranks = raw_ranks.astype(np.int64, copy=False)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    legal_mask = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    if ranks.shape != occupied.shape:
        raise ValueError("C2 public ranks and occupancy masks differ")
    if legal_mask.ndim != 1 or ranks.shape != (legal_mask.shape[0], int(frame.max_tiers)):
        raise ValueError("C2 public frame dimensions differ from the legal mask")
    if int(frame.max_tiers) <= 0 or ranks.shape[0] <= 0:
        raise ValueError("C2 public frame must have positive stack and tier dimensions")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("C2 empty padding must have rank zero")
    if np.any(occupied & ((ranks < 1) | (ranks > UNKNOWN_RANK))):
        raise ValueError("C2 occupied ranks must be known or the public unknown sentinel")
    heights = occupied.sum(axis=1).astype(np.int64)
    expected_occupancy = np.arange(ranks.shape[1])[None, :] < heights[:, None]
    if not np.array_equal(occupied, expected_occupancy):
        raise ValueError("C2 occupied cells must form bottom-filled stack prefixes")
    legal = tuple(int(value) for value in np.flatnonzero(legal_mask))
    if legal != tuple(int(value) for value in frame.legal_destinations):
        raise ValueError("C2 legal destinations differ from the public mask")
    if not legal:
        raise ValueError("C2 requires at least one legal destination")

    target = np.argwhere(occupied & (ranks == 1))
    known_future = occupied & (ranks >= 2) & (ranks <= 21)
    if target.shape != (1, 2) or len(np.unique(ranks[known_future])) != int(known_future.sum()):
        raise ValueError("C2 frame must contain one current target and unique known ranks")
    source_stack, target_tier = (int(value) for value in target[0])
    source_height = int(heights[source_stack])
    if source_height <= target_tier + 1:
        raise ValueError("C2 horizon gate requires an active relocation decision")
    if source_stack in tuple(int(value) for value in np.flatnonzero(legal_mask)):
        raise ValueError("C2 legal destinations must exclude the source stack")
    stack_count, tier_count = ranks.shape
    occupied_count = int(occupied.sum())
    unknown_count = int(np.count_nonzero(occupied & (ranks == UNKNOWN_RANK)))
    known_future_count = int(known_future.sum())
    empty_count = int(np.count_nonzero(heights == 0))

    values = np.asarray(
        [
            1.0,
            occupied_count / float(stack_count * tier_count),
            (source_height - target_tier - 1) / float(tier_count),
            unknown_count / float(max(occupied_count, 1)),
            known_future_count / float(max(occupied_count, 1)),
            empty_count / float(stack_count),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("C2 global public features must be finite")
    return values


def c2_horizon_features(
    frame: PublicDecisionFrame,
    horizon: int,
    *,
    q_value: float | None = None,
) -> np.ndarray:
    """Build x=concat(u,(1-q)*u) for one registered public horizon."""
    h = int(horizon)
    if h not in HORIZONS:
        raise ValueError(f"C2 horizon must be one of {HORIZONS}")
    observation = frame.observation
    q = float(observation.q_t if q_value is None else q_value)
    q_prior = float(observation.q_prior)
    if not math.isfinite(q_prior) or abs(q_prior - Q_PRIOR) > 1e-15:
        raise ValueError("C2 public q_prior differs from the frozen protocol value")
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("C2 q must be a finite probability")
    if h == 20:
        return np.zeros(FEATURE_WIDTH, dtype=np.float64)
    global_features = c2_global_features(frame)
    ratio = (20.0 - h) / 20.0
    horizon_features = np.concatenate((ratio * global_features, ratio * ratio * global_features))
    result = np.concatenate((horizon_features, (1.0 - q) * horizon_features))
    if result.shape != (FEATURE_WIDTH,) or not np.isfinite(result).all():
        raise AssertionError("C2 horizon feature construction returned an invalid vector")
    return result


def score_c2_horizons(
    frame: PublicDecisionFrame,
    weights: Sequence[float] | np.ndarray,
    *,
    mode: str = "history",
) -> C2HorizonDecision:
    """Score five horizons before any H20 call; h20 is the exact zero anchor."""
    if mode not in SELECTOR_MODES:
        raise ValueError(f"C2 mode must be one of {SELECTOR_MODES}")
    coefficient = np.asarray(weights, dtype=np.float64)
    if coefficient.shape != (FEATURE_WIDTH,) or not np.isfinite(coefficient).all():
        raise ValueError("C2 weights must be a finite width-24 vector")
    q = (
        float(frame.observation.q_t)
        if mode == "history"
        else float(frame.observation.q_prior)
    )
    scores: list[tuple[int, float | None]] = []
    for horizon in HORIZONS:
        if horizon == 20:
            score = 0.0
        else:
            score = float(np.dot(c2_horizon_features(frame, horizon, q_value=q), coefficient))
            if not math.isfinite(score):
                raise ValueError("C2 predicted horizon score is nonfinite")
        scores.append((int(horizon), score))
    improving = [(score, horizon) for horizon, score in scores if score is not None and score < 0.0]
    if not improving:
        selected_h = 20
        fallback = True
    else:
        selected_h = min(improving, key=lambda pair: (pair[0], pair[1]))[1]
        fallback = False
    return C2HorizonDecision(
        mode=mode,
        q_used=q,
        selected_h=int(selected_h),
        used_h20_fallback=fallback,
        scores=tuple(scores),
    )


def c2_actor_call(
    frame: PublicDecisionFrame,
    frozen_h20: Any,
    *,
    history_weights: Sequence[float] | np.ndarray,
    no_history_weights: Sequence[float] | np.ndarray,
    mode: str = "history",
) -> C2ActorCall:
    """Select h from public state, censor, then invoke the frozen H20 exactly once."""
    total_start = perf_counter_ns()
    gate_start = perf_counter_ns()
    if mode == "history":
        weights = history_weights
    elif mode == "no_history_fit":
        weights = no_history_weights
    elif mode == "constant_q_intervention":
        weights = history_weights
    else:
        raise ValueError(f"C2 mode must be one of {SELECTOR_MODES}")
    decision = score_c2_horizons(frame, weights, mode=mode)
    gate_ns = perf_counter_ns() - gate_start

    censor_start = perf_counter_ns()
    censored = truncate_public_frame(frame, decision.selected_h)
    censor_ns = perf_counter_ns() - censor_start

    bridge_start = perf_counter_ns()
    legacy = public_frame_to_legacy_state(censored)
    priorities = Online2DPriorityAdapter(censored.max_tiers).build(legacy)
    bridge_ns = perf_counter_ns() - bridge_start

    inference_start = perf_counter_ns()
    action = h20_action_from_tensor(frozen_h20, priorities, censored.legal_destinations)
    inference_ns = perf_counter_ns() - inference_start
    if action not in censored.legal_destinations:
        raise ValueError("C2 H20 returned an illegal destination")
    return C2ActorCall(
        decision=decision,
        action=int(action),
        censored_frame=censored,
        solver_invocation_count=1,
        gate_ns=gate_ns,
        censor_ns=censor_ns,
        bridge_ns=bridge_ns,
        inference_ns=inference_ns,
        total_actor_ns=perf_counter_ns() - total_start,
    )


def build_c2_fit_design(rows: Sequence[Mapping[str, Any]]) -> C2FitDesign:
    """Create five public horizon rows per M1 fit state from legal cost labels."""
    seen_states: set[str] = set()
    history_x: list[np.ndarray] = []
    no_history_x: list[np.ndarray] = []
    targets: list[float] = []
    samples: list[Mapping[str, Any]] = []
    for row in rows:
        if any(key in row for key in ("private_audit", "true_future", "fixed_forecast_private")):
            raise ValueError("C2 fit rows cannot contain evaluator-private audit fields")
        state_id = str(row.get("state_id", ""))
        if not state_id or state_id in seen_states:
            raise ValueError("C2 fit rows require unique nonempty state ids")
        seen_states.add(state_id)
        record = row.get("public_frame")
        if not isinstance(record, Mapping) or any(
            key in record
            for key in ("physical_stacks_private", "true_future", "fixed_forecast_private", "p_evaluator_only")
        ):
            raise ValueError("C2 fit actor inputs must contain only the detached public frame")
        frame = _frame_from_record(record)
        raw = row.get("raw_h20_proposals")
        if not isinstance(raw, Mapping):
            raise ValueError("C2 fit requires the five registered H20 proposal destinations")
        try:
            proposals = {int(key): int(value) for key, value in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("C2 H20 proposals must map integer horizons to destinations") from exc
        if set(proposals) != set(HORIZONS):
            raise ValueError("C2 fit proposals must cover every registered horizon exactly")
        anchor = int(row.get("anchor_destination", -1))
        if anchor != proposals[20] or anchor not in frame.legal_destinations:
            raise ValueError("C2 H20@20 anchor differs from the public legal proposal")
        if any(destination not in frame.legal_destinations for destination in proposals.values()):
            raise ValueError("C2 fit proposal is not a legal public destination")

        original_n = int(row.get("original_n", 0))
        if original_n <= 0:
            raise ValueError("C2 fit row must retain positive original N for label validation")
        labels = row.get("legal_destination_labels")
        if not isinstance(labels, list):
            raise ValueError("C2 fit requires all legal-destination cost labels")
        costs: dict[int, float] = {}
        for label in labels:
            destination = int(label["destination"])
            cost = float(label["normalized_remaining_cost"])
            remaining = int(label["remaining_relocations"])
            if (
                destination in costs
                or remaining < 0
                or not math.isfinite(cost)
                or cost < 0.0
                or not math.isclose(cost, remaining / original_n, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError("C2 destination labels contain duplicate or invalid normalized costs")
            costs[destination] = cost
        if set(costs) != set(frame.legal_destinations):
            raise ValueError("C2 cost labels must cover exactly all legal destinations")
        anchor_cost = costs[anchor]
        q_t = float(frame.observation.q_t)
        q_prior = float(frame.observation.q_prior)
        for horizon in HORIZONS:
            destination = proposals[int(horizon)]
            target = 0.0 if horizon == 20 else costs[destination] - anchor_cost
            if horizon != 20:
                matching_label = next(label for label in labels if int(label["destination"]) == destination)
                if not math.isclose(
                    float(matching_label["relative_label"]), target, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("C2 proposal relative label differs from normalized cost difference")
            history_x.append(c2_horizon_features(frame, int(horizon), q_value=q_t))
            no_history_x.append(c2_horizon_features(frame, int(horizon), q_value=q_prior))
            targets.append(float(target))
            samples.append(
                {
                    "state_id": state_id,
                    "horizon": int(horizon),
                    "destination": destination,
                    "target_relative_cost": float(target),
                }
            )
    if not rows:
        raise ValueError("C2 fit design cannot be empty")
    return C2FitDesign(
        history_x=np.asarray(history_x, dtype=np.float64),
        no_history_x=np.asarray(no_history_x, dtype=np.float64),
        y=np.asarray(targets, dtype=np.float64),
        samples=tuple(samples),
    )


def _frame_from_record(record: Mapping[str, Any]) -> PublicDecisionFrame:
    expected = {
        "rank_tensor",
        "occupied_mask",
        "legal_destination_mask",
        "legal_destinations",
        "max_tiers",
        "q_prior",
        "q_t",
        "history_hits",
        "retrieval_count",
        "relocation_count",
        "crane_position",
        "crane_time",
    }
    if set(record) != expected:
        raise ValueError("C2 public frame record has missing or unregistered fields")
    observation = PublicObservation(
        rank_tensor=np.asarray(record["rank_tensor"], dtype=np.int32),
        occupied_mask=np.asarray(record["occupied_mask"], dtype=np.bool_),
        legal_destination_mask=np.asarray(record["legal_destination_mask"], dtype=np.bool_),
        history_hits=tuple(bool(value) for value in record["history_hits"]),
        q_prior=float(record["q_prior"]),
        q_t=float(record["q_t"]),
    )
    return PublicDecisionFrame(
        observation=observation,
        legal_destinations=tuple(int(value) for value in record["legal_destinations"]),
        max_tiers=int(record["max_tiers"]),
        crane_position=None if record["crane_position"] is None else int(record["crane_position"]),
        crane_time=float(record["crane_time"]),
        retrieval_count=int(record["retrieval_count"]),
        relocation_count=int(record["relocation_count"]),
    )


def fit_c2_ridge(design: C2FitDesign) -> tuple[np.ndarray, np.ndarray]:
    """Solve the two registered float64 ridge systems with all 24 terms penalized."""
    if (
        design.history_x.ndim != 2
        or design.history_x.shape[1] != FEATURE_WIDTH
        or design.no_history_x.shape != design.history_x.shape
        or design.y.shape != (design.history_x.shape[0],)
        or design.history_x.shape[0] == 0
        or not np.isfinite(design.history_x).all()
        or not np.isfinite(design.no_history_x).all()
        or not np.isfinite(design.y).all()
    ):
        raise ValueError("C2 ridge design has invalid shapes or nonfinite values")
    identity = np.eye(FEATURE_WIDTH, dtype=np.float64)
    history = np.linalg.solve(
        design.history_x.T @ design.history_x + RIDGE_LAMBDA * identity,
        design.history_x.T @ design.y,
    )
    no_history = np.linalg.solve(
        design.no_history_x.T @ design.no_history_x + RIDGE_LAMBDA * identity,
        design.no_history_x.T @ design.y,
    )
    if not np.isfinite(history).all() or not np.isfinite(no_history).all():
        raise ValueError("C2 ridge produced nonfinite coefficients")
    return history, no_history


def fit_design_sha256(design: C2FitDesign) -> str:
    digest = hashlib.sha256()
    for matrix in (design.history_x, design.no_history_x, design.y):
        value = np.ascontiguousarray(matrix, dtype=np.float64)
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes())
    digest.update(
        json.dumps(list(design.samples), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


__all__ = [
    "C2ActorCall",
    "C2FitDesign",
    "C2HorizonDecision",
    "FEATURE_WIDTH",
    "GLOBAL_FEATURE_WIDTH",
    "HORIZON_FEATURE_WIDTH",
    "RIDGE_LAMBDA",
    "SELECTOR_MODES",
    "build_c2_fit_design",
    "c2_actor_call",
    "c2_global_features",
    "c2_horizon_features",
    "fit_c2_ridge",
    "fit_design_sha256",
    "score_c2_horizons",
]
