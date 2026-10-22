"""Registered UWSAA comparison policies built on the public selector core.

Historical random namespaces are preserved as exact compatibility tokens;
they determine samples and do not name the public method.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from baseline.native import SOLVERS
from environment.noise_aware_forecast import HORIZONS
from uwsaa.fast import _completion_cost_indexed
from uwsaa.public import (
    WINDOWS,
    UWSAASelectorBase,
    _candidate_sequences,
    _public_state,
    _scenarios,
    _state_seed,
    _window_sizes,
)

ABLATION_ARMS = (
    "UWSAA_NO_WINDOW_SHUFFLE_K8",
    "UWSAA_SINGLE_WINDOW_B2",
    "UWSAA_SINGLE_WINDOW_B5",
    "UWSAA_SINGLE_WINDOW_B10",
    "UWSAA_SINGLE_WINDOW_B20",
    "UWSAA_BEST_FIXED_H",
)
ROLLOUT_ARMS = (
    "UWSAA_K1_ROLLOUT",
    "UWSAA_K2_ROLLOUT",
    "UWSAA_K4_ROLLOUT",
    "UWSAA_K16_ROLLOUT",
)
LB1_ARMS = ("UWSAA_K1_LB1", "UWSAA_K8_LB1")
ALL_VARIANT_ARMS = ABLATION_ARMS + ROLLOUT_ARMS + LB1_ARMS
_K_FOR_ARM = {
    "UWSAA_K1_ROLLOUT": (1, "rollout"),
    "UWSAA_K2_ROLLOUT": (2, "rollout"),
    "UWSAA_K4_ROLLOUT": (4, "rollout"),
    "UWSAA_K16_ROLLOUT": (16, "rollout"),
    "UWSAA_K1_LB1": (1, "lb1"),
    "UWSAA_K8_LB1": (8, "lb1"),
}


def _control_scenarios(frame: object, mode: str, width: int | None = None):
    ranks, occupied, k, n, remaining, visible, token_stacks, unknown = _public_state(frame)
    future_count = n - k - 1
    base_seed = _state_seed(ranks, occupied, k)
    if mode == "no_window":
        slots = [(window, repeat) for window in WINDOWS for repeat in range(2)]
    elif mode == "single" and width in WINDOWS:
        slots = [(int(width), repeat) for repeat in range(8)]
    else:
        raise ValueError("unknown registered scenario control")
    rows = []
    for window, repeat in slots:
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, window, repeat]))
        hidden = [token_stacks[s][t] for s, t in unknown]
        tail = [hidden[i] for i in rng.permutation(len(hidden)).tolist()] if hidden else []
        future = [("r", rank) for rank in range(2, visible + 2)] + tail
        if mode == "no_window":
            shuffled = list(future)
        else:
            shuffled = []
            offset = 0
            for size in _window_sizes(future_count, k, int(width)):
                chunk = list(future[offset:offset + size])
                if size > 1:
                    chunk = [chunk[i] for i in rng.permutation(size).tolist()]
                shuffled.extend(chunk)
                offset += size
        token_rank = {("r", 1): 1}
        token_rank.update({token: i + 2 for i, token in enumerate(shuffled)})
        scenario = [[token_rank[token] for token in stack] for stack in token_stacks]
        rows.append((window, repeat, 1.0 / 8.0, scenario))
    return rows, base_seed, remaining


class UWSAAControlSelector(UWSAASelectorBase):
    """No-window or one-window public scenario ablation, with equal K=8 weight."""

    def __init__(self, solver: str, arm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]):
        if arm not in ABLATION_ARMS[:-1]:
            raise ValueError(f"unknown UWSAA scenario control {arm!r}")
        super().__init__(solver, "UWSAA", runtime, gp_rule)
        self.arm = arm
        self.width = (None if arm == "UWSAA_NO_WINDOW_SHUFFLE_K8"
                      else int(arm.removeprefix("UWSAA_SINGLE_WINDOW_B")))

    def _select_sequence(self, frame: object):
        proposals = _candidate_sequences(frame, self.solver, self.runtime, self.gp_rule)
        if set(proposals) != set(HORIZONS) or not proposals[20]:
            raise ValueError("UWSAA control is missing a registered candidate horizon")
        unique: dict[tuple[int, ...], list[int]] = {}
        for horizon in HORIZONS:
            unique.setdefault(tuple(proposals[horizon]), []).append(horizon)
        anchor = tuple(proposals[20])
        if len(unique) == 1:
            return anchor, {"solver": self.solver, "policy": self.arm, "selected_h": 20, "scenario_count": 0}
        mode = "no_window" if self.width is None else "single"
        rows, seed, remaining = _control_scenarios(frame, mode, self.width)
        scores = {
            sequence: sum(
                weight * _completion_cost_indexed(rows[i][3], sequence, int(frame.max_tiers), remaining)
                for i, (_w, _r, weight, _stacks) in enumerate(rows)
            )
            for sequence in unique
        }
        best = min(scores.values())
        tied = [sequence for sequence, score in scores.items() if abs(score - best) <= 1e-12]
        chosen = anchor if anchor in tied else min(tied, key=lambda seq: min(unique[seq]))
        return chosen, {
            "solver": self.solver,
            "policy": self.arm,
            "selected_h": min(unique[chosen]),
            "candidate_scores": [
                {"sequence": list(seq), "horizons": unique[seq], "mean_relocations": float(scores[seq])}
                for seq in sorted(scores, key=lambda item: min(unique[item]))
            ],
            "scenario_count": len(rows),
            "scenario_weight": 1.0 / 8.0,
            "scenario_windows": [int(row[0]) for row in rows],
            "public_state_seed": int(seed),
        }


def _subset_seed(frame: object) -> tuple[int, str]:
    ranks, occupied, k, *_ = _public_state(frame)
    base = _state_seed(ranks, occupied, k)
    # Preserved exactly for historical repeat compatibility; see protocol doc.
    payload = json.dumps(
        ["plan063_k_subset_v1", int(base), ranks.tolist(), occupied.astype(int).tolist(), int(k)],
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16), digest


def _scenario_ids(frame: object, k: int) -> list[tuple[int, int]]:
    if k == 8:
        return [(width, repeat) for width in WINDOWS for repeat in range(2)]
    if k == 16:
        return [(width, repeat) for width in WINDOWS for repeat in range(4)]
    seed, _digest = _subset_seed(frame)
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    width_order = [int(value) for value in rng.permutation(np.asarray(WINDOWS, dtype=np.int64))]
    repeat_order = {
        width: [int(value) for value in rng.permutation(np.asarray((0, 1), dtype=np.int64))]
        for width in WINDOWS
    }
    return [(width, repeat_order[width][0]) for width in width_order[:k]]


def _scenario_for_repeat(frame: object, width: int, repeat: int, base_seed: int):
    ranks, occupied, k, n, remaining, visible, token_stacks, unknown = _public_state(frame)
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(width), int(repeat)]))
    hidden = [token_stacks[s][t] for s, t in unknown]
    tail = [hidden[i] for i in rng.permutation(len(hidden)).tolist()] if hidden else []
    future = [("r", rank) for rank in range(2, visible + 2)] + tail
    shuffled = []
    offset = 0
    for size in _window_sizes(n - k - 1, k, int(width)):
        chunk = list(future[offset:offset + size])
        if size > 1:
            chunk = [chunk[i] for i in rng.permutation(size).tolist()]
        shuffled.extend(chunk)
        offset += size
    token_rank = {("r", 1): 1}
    token_rank.update({token: index + 2 for index, token in enumerate(shuffled)})
    return [[token_rank[token] for token in stack] for stack in token_stacks]


class UWSAAMultiKSelector(UWSAASelectorBase):
    """Registered K=1/2/4/8/16 completion or K=1/8 LB1 UWSAA variants."""

    def __init__(self, solver: str, arm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]):
        if arm not in _K_FOR_ARM:
            raise ValueError(f"unknown UWSAA multi-scenario arm {arm!r}")
        self.arm = arm
        self.k, self.cost_mode = _K_FOR_ARM[arm]
        super().__init__(solver, "UWSAA", runtime, gp_rule)

    def _select_sequence(self, frame: object):
        _ranks, _occupied, _k0, _n, remaining, _visible, _stacks, _unknown = _public_state(frame)
        proposals = _candidate_sequences(frame, self.solver, self.runtime, self.gp_rule)
        if set(proposals) != set(HORIZONS) or not proposals[20]:
            raise ValueError("UWSAA multi-scenario candidate menu is incomplete")
        unique: dict[tuple[int, ...], list[int]] = {}
        for horizon in HORIZONS:
            unique.setdefault(tuple(proposals[horizon]), []).append(horizon)
        anchor = tuple(proposals[20])
        if len(unique) == 1:
            return anchor, {"solver": self.solver, "policy": self.arm, "selected_h": 20, "scenario_count": 0}
        if self.k == 8 and self.cost_mode == "rollout":
            rows, _seed, scenario_remaining = _scenarios(frame)
            if scenario_remaining != remaining:
                raise AssertionError("public remaining count changed during UWSAA scoring")
        else:
            base_seed = _state_seed(*_public_state(frame)[:2], int(frame.retrieval_count))
            ids = _scenario_ids(frame, self.k)
            rows = [(width, repeat, 1.0 / self.k, _scenario_for_repeat(frame, width, repeat, base_seed))
                    for width, repeat in ids]
        if len(rows) != self.k:
            raise AssertionError("UWSAA registered scenario count differs from K")
        score_map = {}
        for sequence in unique:
            costs = []
            for _width, _repeat, _weight, scenario in rows:
                if self.cost_mode == "rollout":
                    cost = _completion_cost_indexed(scenario, sequence, int(frame.max_tiers), remaining)
                else:
                    cost = _lb1_after_candidate(scenario, sequence, int(frame.max_tiers))
                costs.append(float(cost) / self.k)
            score_map[sequence] = sum(costs)
        best = min(score_map.values())
        tied = [sequence for sequence, score in score_map.items() if abs(score - best) <= 1e-12]
        chosen = anchor if anchor in tied else min(tied, key=lambda seq: min(unique[seq]))
        _subset_value, subset_digest = _subset_seed(frame)
        return chosen, {
            "solver": self.solver,
            "policy": self.arm,
            "selected_h": min(unique[chosen]),
            "candidate_scores": [
                {"sequence": list(seq), "horizons": unique[seq], "mean_relocations": float(score_map[seq])}
                for seq in sorted(score_map, key=lambda item: min(unique[item]))
            ],
            "scenario_count": len(rows),
            "scenario_ids": [[int(width), int(repeat)] for width, repeat, _w, _s in rows],
            "scenario_weight": 1.0 / self.k,
            "scenario_subset_compatibility": "preserved_historical_random_namespace",
            "scenario_subset_public_state_sha256": subset_digest,
            "cost_mode": self.cost_mode,
        }


class UWSAADeterministicSingleSelector(UWSAASelectorBase):
    """One deterministic public completion used by the P0 ablation."""

    def __init__(self, solver: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]):
        super().__init__(solver, "UWSAA", runtime, gp_rule)

    def _select_sequence(self, frame: object):
        _ranks, _occupied, _k, _n, remaining, visible, token_stacks, unknown = _public_state(frame)
        proposals = _candidate_sequences(frame, self.solver, self.runtime, self.gp_rule)
        if set(proposals) != set(HORIZONS) or not proposals[20]:
            raise ValueError("deterministic UWSAA candidate menu is missing an anchor")
        unique: dict[tuple[int, ...], list[int]] = {}
        for horizon in HORIZONS:
            unique.setdefault(tuple(proposals[horizon]), []).append(horizon)
        anchor = tuple(proposals[20])
        if len(unique) == 1:
            return anchor, {"solver": self.solver, "policy": "UWSAA_DETERMINISTIC_SINGLE",
                            "selected_h": 20, "scenario_count": 0}
        visible_tokens = [("r", rank) for rank in range(2, visible + 2)]
        tail_tokens = [token_stacks[stack][tier] for stack, tier in sorted(unknown)]
        future_tokens = visible_tokens + tail_tokens
        token_rank = {("r", 1): 1}
        token_rank.update({token: index + 2 for index, token in enumerate(future_tokens)})
        scenario = [[token_rank[token] for token in stack] for stack in token_stacks]
        if len(future_tokens) != remaining - 1:
            raise ValueError("deterministic completion does not cover the public future")
        scores = {
            sequence: float(_completion_cost_indexed(scenario, sequence, int(frame.max_tiers), remaining))
            for sequence in unique
        }
        best = min(scores.values())
        tied = [sequence for sequence, score in scores.items() if abs(score - best) <= 1e-12]
        chosen = anchor if anchor in tied else min(tied, key=lambda seq: min(unique[seq]))
        return chosen, {
            "solver": self.solver,
            "policy": "UWSAA_DETERMINISTIC_SINGLE",
            "selected_h": min(unique[chosen]),
            "candidate_scores": [
                {"sequence": list(seq), "horizons": unique[seq], "deterministic_mean_relocations": scores[seq]}
                for seq in sorted(unique, key=lambda item: min(unique[item]))
            ],
            "scenario_count": 1,
            "scenario_id": "VISIBLE_ORDER_THEN_STACK_TIER_TAIL",
            "visible_count": int(visible),
            "anonymous_tail_count": len(tail_tokens),
        }


def _lb1_after_candidate(initial_stacks: Sequence[Sequence[int]], sequence: Sequence[int], max_tiers: int) -> int:
    stacks = [list(map(int, stack)) for stack in initial_stacks]
    for destination in sequence:
        source = next((index for index, stack in enumerate(stacks) if 1 in stack), None)
        if source is None or len(stacks[source]) <= stacks[source].index(1) + 1:
            raise ValueError("candidate sequence does not move a blocked current target")
        if destination == source or len(stacks[destination]) >= max_tiers:
            raise ValueError("candidate sequence has an illegal destination")
        stacks[destination].append(stacks[source].pop())
    badly = 0
    for stack in stacks:
        minimum = None
        for rank in stack:
            if minimum is not None and rank > minimum:
                badly += 1
            minimum = rank if minimum is None else min(minimum, rank)
    return len(sequence) + badly


def make_variant(solver: str, arm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]):
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}")
    if arm in ABLATION_ARMS[:-1]:
        return UWSAAControlSelector(solver, arm, runtime, gp_rule)
    if arm in _K_FOR_ARM:
        return UWSAAMultiKSelector(solver, arm, runtime, gp_rule)
    if arm == "UWSAA_DETERMINISTIC_SINGLE":
        return UWSAADeterministicSingleSelector(solver, runtime, gp_rule)
    raise ValueError(f"unsupported UWSAA variant {arm!r}")


__all__ = [
    "ABLATION_ARMS", "ALL_VARIANT_ARMS", "LB1_ARMS", "ROLLOUT_ARMS",
    "UWSAAControlSelector", "UWSAADeterministicSingleSelector", "UWSAAMultiKSelector", "make_variant",
]
