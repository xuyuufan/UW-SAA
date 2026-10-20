"""Public actor adapters and committed EG plans for PLAN-049."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from R.scrp.r_order_public import (
    ALGORITHMS,
    HORIZONS,
    PlackettLuceRates,
    PublicHistorySummary,
    PublicHistoryTracker,
    ROrderDecision,
    _after_move,
    _parts,
    attach_history_summary,
    build_candidate_pool,
    build_risk_proposal,
    select_candidate,
)


def _snapshot(frame: object) -> tuple[Any, ...]:
    ranks, occupied, legal_mask, legal, tiers, retrieval_count = _parts(frame)
    observation = getattr(frame, "observation")
    return (
        ranks.copy(),
        occupied.copy(),
        legal_mask.copy(),
        legal,
        tiers,
        retrieval_count,
        int(getattr(frame, "relocation_count")),
        tuple(bool(value) for value in getattr(observation, "history_hits", ())),
        float(getattr(observation, "q_prior")),
        float(getattr(observation, "q_t")),
    )


def _snapshots_equal(left: object, right: object) -> bool:
    a, b = _snapshot(left), _snapshot(right)
    return (
        np.array_equal(a[0], b[0])
        and np.array_equal(a[1], b[1])
        and np.array_equal(a[2], b[2])
        and a[3:] == b[3:]
    )


def build_native_proposals(
    frame: object,
    algorithm: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any],
) -> dict[int, tuple[int, ...]]:
    """Create exactly the five horizon proposals for the current proposer."""
    if algorithm not in ALGORITHMS:
        raise ValueError("unknown native proposer")
    from R.scrp.universal_r_utility_m1 import (
        EMNativeActor,
        ERINativeActor,
        GPNativeActor,
    )

    proposals: dict[int, tuple[int, ...]] = {}
    for horizon in HORIZONS:
        if algorithm == "EM":
            proposals[horizon] = (int(EMNativeActor(runtime, horizon)(frame)),)
        elif algorithm == "GP":
            proposals[horizon] = (int(GPNativeActor(runtime, gp_rule, horizon)(frame)),)
        elif algorithm == "ERI":
            proposals[horizon] = (int(ERINativeActor(runtime, horizon)(frame)),)
        else:
            prefix = runtime["truncate_public_frame"](frame, horizon)
            plan = runtime["eri_eg"].build_eg20_plan(prefix)
            proposals[horizon] = tuple(int(value) for value in plan.destinations_top_down)
    return proposals


class FixedHorizonActor:
    """One exact native H call; EG uses a horizon-projected group cache."""

    def __init__(
        self,
        algorithm: str,
        horizon: int,
        runtime: Mapping[str, Any],
        gp_rule: Mapping[str, Any],
    ) -> None:
        if algorithm not in ALGORITHMS or horizon not in HORIZONS:
            raise ValueError("fixed actor algorithm/horizon is outside registration")
        self.algorithm = algorithm
        self.horizon = int(horizon)
        self.runtime = runtime
        self.gp_rule = gp_rule
        self._eg_actor = runtime["eri_eg"].EG20Actor() if algorithm == "EG" else None
        if algorithm == "EG":
            self._native_actor = None
        else:
            from R.scrp.universal_r_utility_m1 import make_native_actor

            self._native_actor = make_native_actor(algorithm, runtime, gp_rule, self.horizon)

    @property
    def is_group_start(self) -> bool:
        return self._eg_actor is None or self._eg_actor._plan is None

    @property
    def cached_moves(self) -> int:
        return 0 if self._eg_actor is None else int(self._eg_actor.cached_moves)

    def reset(self) -> None:
        if self._eg_actor is not None:
            self._eg_actor.reset()

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        if self.algorithm == "EG":
            group_start = self.is_group_start
            prefix = self.runtime["truncate_public_frame"](frame, self.horizon)
            action = int(self._eg_actor.select_destination(prefix))
            return action, {
                "policy": f"FIXED{self.horizon}",
                "requested_h": self.horizon,
                "group_start": bool(group_start),
                "cached_moves": self.cached_moves,
            }
        action = int(self._native_actor(frame))
        return action, {"policy": f"FIXED{self.horizon}", "requested_h": self.horizon}


class FrozenRActor:
    """The already-frozen solver-specific R adapter used by 047/048."""

    def __init__(
        self, algorithm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError("unknown frozen-R solver adapter")
        self.algorithm = algorithm
        self.runtime = runtime
        self.gp_rule = gp_rule
        self._eri_actor = runtime["r_eri_eg"].RERIActor() if algorithm == "ERI" else None
        self._eg_actor = runtime["r_eri_eg"].REGGroupActor() if algorithm == "EG" else None

    @property
    def is_group_start(self) -> bool:
        if self.algorithm == "EG":
            return self._eg_actor is None or self._eg_actor._selected_actor is None
        return True

    @property
    def cached_moves(self) -> int:
        if self.algorithm == "EG" and self._eg_actor is not None:
            return int(self._eg_actor.cached_moves)
        return 0

    def reset(self) -> None:
        if self._eg_actor is not None:
            self._eg_actor.reset()

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        if self.algorithm == "EM":
            decision = self.runtime["select_r_action"](
                frame.observation, method="R", max_tiers=int(frame.max_tiers)
            )
            return int(decision.destination), {
                "policy": "OLD_FROZEN_R", "solver": "EM",
                "requested_h": int(decision.requested_h),
            }
        if self.algorithm == "GP":
            call = self.runtime["frozen_actor_call"](
                frame, solver="GP", frozen_solver=self.gp_rule, method="R_TRANSFER"
            )
            return int(call.action), {
                "policy": "OLD_FROZEN_R", "solver": "GP", "requested_h": int(call.selected_h),
                "solver_invocation_count": int(call.solver_invocation_count),
            }
        if self.algorithm == "ERI":
            result = self._eri_actor.select_destination(frame)
            return int(result["action"]), {
                "policy": "OLD_FROZEN_R", "solver": "ERI", "requested_h": int(result["selected_h"]),
                "native_trace": result,
            }
        if self._eg_actor is None:
            raise AssertionError("frozen EG adapter was not initialized")
        result = self._eg_actor.select_destination(frame)
        group_started = bool(result.get("group_started", False))
        return int(result["action"]), {
            "policy": "OLD_FROZEN_R", "solver": "EG", "requested_h": int(result["selected_h"]),
            "group_start": group_started, "cached_move": not group_started,
            "native_trace": result,
        }


class ROrderActor:
    """Choose a full-sequence candidate; cache EG only against full-H20 frames."""

    def __init__(
        self,
        algorithm: str,
        rates: PlackettLuceRates,
        tau: float,
        runtime: Mapping[str, Any],
        gp_rule: Mapping[str, Any],
        *,
        include_risk: bool = True,
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError("unknown proposer algorithm")
        if float(tau) not in (0.0, 0.25):
            raise ValueError("tau must be registered")
        self.algorithm = algorithm
        self.rates = rates
        self.tau = float(tau)
        self.runtime = runtime
        self.gp_rule = gp_rule
        self.include_risk = bool(include_risk)
        self._sequence: tuple[int, ...] | None = None
        self._next = 0
        self._expected_full_h20: object | None = None
        self.groups_committed = 0
        self.cached_moves = 0
        self.last_decision: ROrderDecision | None = None

    @property
    def is_group_start(self) -> bool:
        return self._sequence is None

    def reset(self) -> None:
        self._sequence = None
        self._next = 0
        self._expected_full_h20 = None
        self.groups_committed = 0
        self.cached_moves = 0
        self.last_decision = None

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        if self.algorithm != "EG":
            proposals = build_native_proposals(frame, self.algorithm, self.runtime, self.gp_rule)
            risk = build_risk_proposal(frame, self.algorithm, self.rates) if self.include_risk else None
            pool = build_candidate_pool(frame, self.algorithm, proposals, risk)
            decision = select_candidate(frame, pool, self.rates, tau=self.tau)
            self.last_decision = decision
            return decision.sequence[0], self._trace(decision, pool)

        if self._sequence is None:
            proposals = build_native_proposals(frame, self.algorithm, self.runtime, self.gp_rule)
            risk = build_risk_proposal(frame, self.algorithm, self.rates) if self.include_risk else None
            pool = build_candidate_pool(frame, self.algorithm, proposals, risk)
            decision = select_candidate(frame, pool, self.rates, tau=self.tau)
            self.last_decision = decision
            self._sequence = decision.sequence
            self._next = 1
            self.groups_committed += 1
            action = int(decision.sequence[0])
            if len(decision.sequence) > 1:
                self._expected_full_h20 = _after_move(frame, action)
            else:
                self._sequence = None
                self._next = 0
            return action, self._trace(decision, pool)

        if self._expected_full_h20 is None or not _snapshots_equal(frame, self._expected_full_h20):
            raise RuntimeError("R_ORDER EG full-H20 public cache expectation mismatch")
        sequence = self._sequence
        if sequence is None or self._next >= len(sequence):
            raise RuntimeError("R_ORDER EG committed plan cache is inconsistent")
        action = int(sequence[self._next])
        self.cached_moves += 1
        self._next += 1
        if self._next < len(sequence):
            self._expected_full_h20 = _after_move(frame, action)
        else:
            self._sequence = None
            self._next = 0
            self._expected_full_h20 = None
        return action, {
            "policy": "R_ORDER",
            "group_start": False,
            "cached_move": True,
            "committed_sequence": list(sequence),
            "destination": action,
        }

    @staticmethod
    def _trace(decision: ROrderDecision, pool: Any) -> dict[str, Any]:
        return {
            "policy": "R_ORDER",
            "group_start": True,
            "sequence": list(decision.sequence),
            "anchor_sequence": list(decision.anchor_sequence),
            "score": float(decision.score),
            "anchor_score": float(decision.anchor_score),
            "tau": float(decision.tau),
            "used_anchor": bool(decision.used_anchor),
            "candidates": [
                {
                    "sequence": list(candidate.destinations),
                    "origins": [
                        {
                            "source": origin.source,
                            "horizon": origin.horizon,
                            "proposal_id": origin.proposal_id,
                        }
                        for origin in candidate.origins
                    ],
                    "is_h20_anchor": candidate.is_h20_anchor,
                }
                for candidate in pool.candidates
            ],
            "candidate_scores": [
                {"sequence": list(sequence), "score": float(score)}
                for sequence, score in decision.scored_candidates
            ],
        }


@dataclass
class HistoryBoundController:
    """Run one public transition update before every policy decision."""

    policy: Any
    tracker: PublicHistoryTracker
    previous_frame: object | None = None
    previous_action: int | None = None

    @classmethod
    def fresh(cls, policy: Any, records: tuple[float, ...] = ()) -> "HistoryBoundController":
        return cls(policy, PublicHistoryTracker(records))

    def observe(self, frame: object) -> object:
        if self.previous_frame is not None:
            if self.previous_action is None:
                raise RuntimeError("history controller has a previous frame without an action")
            self.tracker.update(self.previous_frame, self.previous_action, frame)
        summary: PublicHistorySummary = self.tracker.summary(frame)
        return attach_history_summary(frame, summary)

    def observe_action(self, frame: object, action: int) -> object:
        """Update public history, then bind a caller-supplied committed action."""
        decorated = self.observe(frame)
        if int(action) not in tuple(int(value) for value in frame.legal_destinations):
            raise ValueError("committed action is outside the current public legal set")
        self.previous_frame = frame
        self.previous_action = int(action)
        return decorated

    def select(self, frame: object) -> tuple[int, dict[str, Any], object]:
        decorated = self.observe(frame)
        action, trace = self.policy.select(decorated)
        self.previous_frame = frame
        self.previous_action = int(action)
        return int(action), trace, decorated


__all__ = (
    "FixedHorizonActor", "FrozenRActor", "HistoryBoundController", "ROrderActor",
    "build_native_proposals",
)
