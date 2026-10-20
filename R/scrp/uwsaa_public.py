"""Public-only sample-average rollout scorer for the PLAN-056 UWSAA variants.

The scorer sees a detached public frame and candidate destination sequences.
It never receives an environment, provider, evaluator seed, condition, true
future, or box identifier. Rank-22 objects become position-derived anonymous
tokens before synthetic forecast-tail sampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


HORIZONS = (0, 2, 5, 10, 20)
WINDOWS = (2, 5, 10, 20)
SCENARIOS_PER_WINDOW = 2
NAMESPACE = "plan056_tw_saa_public_scenarios_v1"


def _public_state(frame: object) -> tuple[np.ndarray, np.ndarray, int, int, int, int, list[list[Any]], list[Any]]:
    observation = getattr(frame, "observation")
    ranks = np.asarray(getattr(observation, "rank_tensor"), dtype=np.int32)
    occupied = np.asarray(getattr(observation, "occupied_mask"), dtype=np.bool_)
    k = int(getattr(frame, "retrieval_count"))
    if ranks.ndim != 2 or ranks.shape != occupied.shape or np.any(ranks[~occupied] != 0):
        raise ValueError("UWSAA requires a valid public rank/occupancy frame")
    if k < 0 or int(getattr(frame, "max_tiers")) != int(ranks.shape[1]):
        raise ValueError("UWSAA public retrieval count or max_tiers is invalid")
    heights = occupied.sum(axis=1).astype(int)
    bottom_filled = np.arange(ranks.shape[1])[None, :] < heights[:, None]
    if not np.array_equal(occupied, bottom_filled):
        raise ValueError("UWSAA public occupancy contains holes")
    if np.any(occupied & ((ranks < 1) | (ranks > 22))):
        raise ValueError("UWSAA public occupied ranks must be in 1..22")
    n = k + int(np.count_nonzero(occupied))
    remaining_count = int(np.count_nonzero(occupied))
    future_count = n - k - 1
    if future_count < 0:
        raise ValueError("UWSAA public retrieval count exceeds the remaining layout")
    visible_count = min(20, future_count)
    visible = sorted(int(value) for value in ranks[occupied] if 2 <= int(value) <= 21)
    expected = list(range(2, visible_count + 2))
    if visible != expected:
        raise ValueError("UWSAA public forecast ranks are not the contiguous 2..21 prefix")
    current = np.argwhere(occupied & (ranks == 1))
    if current.shape != (1, 2):
        raise ValueError("UWSAA public frame must contain exactly one current target")

    unknown_positions = [
        (int(stack), int(tier))
        for stack, tier in np.argwhere(occupied & (ranks == 22))
    ]
    if len(unknown_positions) != future_count - visible_count:
        raise ValueError("UWSAA anonymous rank-22 tail count differs from public remaining items")

    stacks: list[list[Any]] = []
    for stack in range(ranks.shape[0]):
        row: list[Any] = []
        for tier in range(ranks.shape[1]):
            if not occupied[stack, tier]:
                break
            rank = int(ranks[stack, tier])
            row.append(("u", stack, tier) if rank == 22 else ("r", rank))
        stacks.append(row)
    return ranks, occupied, k, n, remaining_count, visible_count, stacks, unknown_positions


def _state_seed(ranks: np.ndarray, occupied: np.ndarray, k: int) -> int:
    payload = json.dumps(
        [NAMESPACE, ranks.tolist(), occupied.astype(int).tolist(), int(k)],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _window_sizes(future_count: int, k: int, width: int) -> list[int]:
    if future_count <= 0:
        return []
    first = min(future_count, width - (k % width))
    sizes = [first]
    remaining = future_count - first
    while remaining:
        size = min(width, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def _scenarios(frame: object) -> tuple[list[tuple[int, int, float, list[list[int]]]], int, int]:
    ranks, occupied, k, n, remaining_count, visible_count, token_stacks, unknown_positions = _public_state(frame)
    future_count = n - k - 1
    base_seed = _state_seed(ranks, occupied, k)
    scenario_rows: list[tuple[int, int, float, list[list[int]]]] = []
    for width in WINDOWS:
        sizes = _window_sizes(future_count, k, width)
        for repeat in range(SCENARIOS_PER_WINDOW):
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, width, repeat]))
            unknown_tokens = [token_stacks[stack][tier] for stack, tier in unknown_positions]
            if unknown_tokens:
                unknown_order = rng.permutation(len(unknown_tokens)).tolist()
                tail_tokens = [unknown_tokens[index] for index in unknown_order]
            else:
                tail_tokens = []
            future_tokens: list[Any] = [("r", rank) for rank in range(2, visible_count + 2)] + tail_tokens
            offset = 0
            shuffled: list[Any] = []
            for size in sizes:
                window = list(future_tokens[offset:offset + size])
                if size > 1:
                    window = [window[index] for index in rng.permutation(size).tolist()]
                shuffled.extend(window)
                offset += size
            token_to_rank = {("r", 1): 1}
            token_to_rank.update({token: index + 2 for index, token in enumerate(shuffled)})
            numeric_stacks: list[list[int]] = []
            for stack in token_stacks:
                numeric_stacks.append([token_to_rank[token] for token in stack])
            scenario_rows.append((width, repeat, 1.0 / SCENARIOS_PER_WINDOW, numeric_stacks))
    return scenario_rows, base_seed, remaining_count


def _forced_move(stacks: list[list[int]], destination: int, max_tiers: int) -> None:
    target_location = next(
        ((stack_index, tier) for stack_index, stack in enumerate(stacks)
         for tier, rank in enumerate(stack) if rank == 1),
        None,
    )
    if target_location is None:
        raise ValueError("UWSAA candidate simulation lost the current target")
    source, target_tier = target_location
    if len(stacks[source]) <= target_tier + 1:
        raise ValueError("UWSAA forced candidate moved an unblocked target")
    legal = tuple(index for index, stack in enumerate(stacks)
                  if index != source and len(stack) < max_tiers)
    if destination not in legal:
        raise ValueError("UWSAA forced candidate destination is physically illegal")
    stacks[destination].append(stacks[source].pop())


def _minmax_completion_cost(
    initial_stacks: Sequence[Sequence[int]],
    candidate_sequence: Sequence[int],
    max_tiers: int,
    remaining_count: int,
) -> int:
    from Minmax.minmax_offline import choose_minmax_destination

    stacks = [list(stack) for stack in initial_stacks]
    moves = 0
    for destination in candidate_sequence:
        _forced_move(stacks, int(destination), max_tiers)
        moves += 1

    move_limit = remaining_count * max(1, max_tiers - 1)
    for target in range(1, remaining_count + 1):
        while True:
            location = next(
                ((stack_index, tier) for stack_index, stack in enumerate(stacks)
                 for tier, rank in enumerate(stack) if rank == target),
                None,
            )
            if location is None:
                raise ValueError("UWSAA scenario completion lost a target token")
            source, target_tier = location
            if len(stacks[source]) == target_tier + 1:
                stacks[source].pop()
                break
            legal = tuple(index for index, stack in enumerate(stacks)
                          if index != source and len(stack) < max_tiers)
            if not legal:
                raise ValueError("UWSAA Minmax completion has no legal destination")
            blocker = stacks[source][-1]
            choice = choose_minmax_destination(stacks, blocker, legal, remaining_count)
            if choice is None:
                raise ValueError("UWSAA Minmax completion returned no destination")
            destination, _score = choice
            stacks[destination].append(stacks[source].pop())
            moves += 1
            if moves > move_limit:
                raise ValueError("UWSAA synthetic completion exceeded the registered physical relocation bound")
    if any(stacks):
        raise ValueError("UWSAA Minmax completion did not retrieve every anonymous token")
    return moves


def _history_weights(frame: object, n: int, k: int) -> dict[int, float]:
    hits = tuple(bool(value) for value in getattr(frame.observation, "history_hits", ()))
    if len(hits) > 8 or len(hits) > k:
        raise ValueError("UWSAA hit history exceeds the registered rolling-eight public window")
    log_weights: list[float] = []
    for width in WINDOWS:
        value = 0.0
        impossible = False
        first_event = k - len(hits) + 1
        for event, hit in zip(range(first_event, k + 1), hits):
            denominator = min(n - event, width - ((event - 1) % width))
            if denominator <= 0:
                raise ValueError("UWSAA history likelihood has a nonpositive window denominator")
            probability = 1.0 / denominator
            likelihood = probability if hit else (1.0 - probability)
            if likelihood <= 0.0:
                impossible = True
                break
            value += math.log(likelihood)
        log_weights.append(float("-inf") if impossible else value)
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        raise ValueError(
            "UWSAA public hit history has zero likelihood under every registered window width"
        )
    maximum = max(finite)
    unnormalized = [math.exp(value - maximum) if math.isfinite(value) else 0.0 for value in log_weights]
    total = sum(unnormalized)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("UWSAA public hit-history weights failed to normalize")
    return {width: value / total for width, value in zip(WINDOWS, unnormalized)}


def _candidate_sequences(
    frame: object, solver: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
) -> dict[int, tuple[int, ...]]:
    if solver in ("EM", "GP", "ERI", "EG"):
        from R.scrp.r_order_actor import build_native_proposals

        return {
            int(h): tuple(int(value) for value in sequence)
            for h, sequence in build_native_proposals(frame, solver, runtime, gp_rule).items()
        }
    if solver in ("Chain", "Minmax"):
        if solver == "Chain":
            from Chain.chain_public import propose
        else:
            from Minmax.minmax_public import propose

        return {h: (int(propose(frame, h)[0]),) for h in HORIZONS}
    raise ValueError(f"unknown PLAN-056 solver {solver!r}")


class UWSAASelectorBase:
    """One public-only TW scenario scorer, independent of Ridge and evaluator state."""

    def __init__(
        self, solver: str, method: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
    ) -> None:
        if method not in ("UWSAA", "TW_HISTORY"):
            raise ValueError("UWSAA method must be UWSAA or TW_HISTORY")
        self.solver, self.method, self.runtime, self.gp_rule = solver, method, runtime, gp_rule
        self._eg_committer: Any | None = None
        if solver == "EG":
            from R.scrp.r_five_versions_actor import _SequenceCommitter

            self._eg_committer = _SequenceCommitter("EG", self._select_sequence)

    def select(self, frame: object) -> tuple[int, dict[str, Any]]:
        from R.scrp.r_five_versions_actor import require_public_frame

        require_public_frame(frame)
        if self._eg_committer is not None:
            return self._eg_committer.select(frame)
        sequence, trace = self._select_sequence(frame)
        if len(sequence) != 1:
            raise ValueError("PLAN-056 non-EG solver must commit one public destination")
        return int(sequence[0]), trace

    def _select_sequence(self, frame: object) -> tuple[tuple[int, ...], dict[str, Any]]:
        ranks, occupied, k, original_n, remaining_count, _visible_count, _stacks, _unknown = _public_state(frame)
        proposals = _candidate_sequences(frame, self.solver, self.runtime, self.gp_rule)
        if set(proposals) != set(HORIZONS) or not proposals[20]:
            raise ValueError("PLAN-056 candidate menu is missing a registered horizon")
        unique: dict[tuple[int, ...], list[int]] = {}
        for horizon in HORIZONS:
            unique.setdefault(proposals[horizon], []).append(horizon)
        anchor = proposals[20]
        if len(unique) == 1:
            return anchor, {
                "solver": self.solver, "policy": self.method,
                "selected_h": 20, "candidate_scores": [],
                "horizons_collapsed_to_one_unique_sequence": True,
                "scenario_count": 0,
            }

        scenarios, seed, scenario_remaining_count = _scenarios(frame)
        if len(scenarios) != 8:
            raise AssertionError("PLAN-056 must generate K=2 common scenarios for all four widths")
        if self.method == "UWSAA":
            width_weights = {width: 0.25 for width in WINDOWS}
        else:
            width_weights = _history_weights(frame, original_n, k)

        max_tiers = int(getattr(frame, "max_tiers"))
        scored: dict[tuple[int, ...], float] = {}
        for sequence in unique:
            total = 0.0
            for width, _repeat, sample_weight, scenario_stacks in scenarios:
                scenario_weight = float(width_weights[width]) * sample_weight
                total += scenario_weight * _minmax_completion_cost(
                    scenario_stacks, sequence, max_tiers, scenario_remaining_count
                )
            scored[sequence] = total

        best_score = min(scored.values())
        tied = [sequence for sequence, score in scored.items() if abs(score - best_score) <= 1e-12]
        chosen = anchor if anchor in tied else min(tied, key=lambda sequence: min(unique[sequence]))
        selected_h = min(unique[chosen])
        return chosen, {
            "solver": self.solver, "policy": self.method,
            "selected_h": int(selected_h),
            "candidate_scores": [
                {"sequence": list(sequence), "horizons": unique[sequence], "weighted_saa_Rraw": score}
                for sequence, score in sorted(scored.items(), key=lambda item: min(unique[item[0]]))
            ],
            "window_weights": {str(width): float(width_weights[width]) for width in WINDOWS},
            "history_weight_status": "not_used" if self.method == "UWSAA" else "valid",
            "rolling_hit_count": len(tuple(getattr(frame.observation, "history_hits", ()))),
            "scenario_count": len(scenarios),
            "scenario_seed_public_state_sha256": hashlib.sha256(
                json.dumps([int(seed), int(k), ranks.tolist(), occupied.astype(int).tolist()],
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "continuation": "fixed Minmax rule on synthetic complete priority ranks; surrogate, not an unbiased online cost estimate",
        }
