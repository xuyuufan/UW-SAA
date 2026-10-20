"""Semantics-preserving indexed completion for the frozen public TW selector."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableSequence, Sequence

from Minmax.minmax_offline import max_in_choosestack
from R.scrp.uwsaa_public import (
    HORIZONS,
    WINDOWS,
    UWSAASelectorBase,
    _candidate_sequences,
    _public_state,
    _scenarios,
)


def _stack_min_push(prefix: list[int], rank: int) -> None:
    prefix.append(rank if not prefix else min(prefix[-1], rank))


def _completion_cost_indexed(
    initial_stacks: Sequence[Sequence[int]],
    candidate_sequence: Sequence[int],
    max_tiers: int,
    remaining_count: int,
    *,
    trace: MutableSequence[dict[str, int | str]] | None = None,
) -> int:
    """Match the frozen Minmax completion while indexing synthetic ranks."""
    stacks = [list(stack) for stack in initial_stacks]
    locations: dict[int, tuple[int, int]] = {}
    minima: list[list[int]] = []
    for stack_index, stack in enumerate(stacks):
        prefix: list[int] = []
        for tier, raw_rank in enumerate(stack):
            rank = int(raw_rank)
            locations.setdefault(rank, (stack_index, tier))
            _stack_min_push(prefix, rank)
        minima.append(prefix)

    def move_top(source: int, destination: int) -> int:
        rank = int(stacks[source].pop())
        minima[source].pop()
        destination_tier = len(stacks[destination])
        stacks[destination].append(rank)
        _stack_min_push(minima[destination], rank)
        locations[rank] = (destination, destination_tier)
        return rank

    moves = 0
    for sequence_index, raw_destination in enumerate(candidate_sequence):
        destination = int(raw_destination)
        target_location = locations.get(1)
        if target_location is None:
            raise ValueError("UWSAA candidate simulation lost the current target")
        source, target_tier = target_location
        if len(stacks[source]) <= target_tier + 1:
            raise ValueError("UWSAA forced candidate moved an unblocked target")
        legal = tuple(
            index for index, stack in enumerate(stacks)
            if index != source and len(stack) < max_tiers
        )
        if destination not in legal:
            raise ValueError("UWSAA forced candidate destination is physically illegal")
        blocker = int(stacks[source][-1])
        move_top(source, destination)
        moves += 1
        if trace is not None:
            trace.append({"kind": "forced", "index": sequence_index,
                          "source": source, "blocker": blocker,
                          "destination": destination})

    move_limit = remaining_count * max(1, max_tiers - 1)
    for target in range(1, remaining_count + 1):
        while True:
            location = locations.get(target)
            if location is None:
                raise ValueError("UWSAA scenario completion lost a target token")
            source, target_tier = location
            if len(stacks[source]) == target_tier + 1:
                stacks[source].pop()
                minima[source].pop()
                del locations[target]
                if trace is not None:
                    trace.append({"kind": "retrieve", "target": target,
                                  "source": source, "tier": target_tier})
                break

            legal = tuple(
                index for index, stack in enumerate(stacks)
                if index != source and len(stack) < max_tiers
            )
            if not legal:
                raise ValueError("UWSAA Minmax completion has no legal destination")
            blocker = int(stacks[source][-1])
            scores = [0] * len(stacks)
            for index in legal:
                scores[index] = minima[index][-1] if minima[index] else remaining_count + 1
            destination = max_in_choosestack(scores, blocker)
            move_top(source, destination)
            moves += 1
            if trace is not None:
                trace.append({"kind": "relocate", "target": target,
                              "source": source, "blocker": blocker,
                              "destination": destination})
            if moves > move_limit:
                raise ValueError(
                    "UWSAA synthetic completion exceeded the registered physical relocation bound"
                )

    if any(stacks):
        raise ValueError("UWSAA Minmax completion did not retrieve every anonymous token")
    return moves


class UWSAAFastSelector(UWSAASelectorBase):
    """Drop-in UWSAA selector with indexed synthetic Minmax completion."""

    def __init__(self, solver: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]) -> None:
        super().__init__(solver, "UWSAA", runtime, gp_rule)

    def _select_sequence(self, frame: object) -> tuple[tuple[int, ...], dict[str, Any]]:
        # Kept in the frozen selector's order, including accumulation and ties.
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
        width_weights = {width: 0.25 for width in WINDOWS}
        max_tiers = int(getattr(frame, "max_tiers"))
        scored: dict[tuple[int, ...], float] = {}
        for sequence in unique:
            total = 0.0
            for width, _repeat, sample_weight, scenario_stacks in scenarios:
                scenario_weight = float(width_weights[width]) * sample_weight
                total += scenario_weight * _completion_cost_indexed(
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
            "history_weight_status": "not_used",
            "rolling_hit_count": len(tuple(getattr(frame.observation, "history_hits", ()))),
            "scenario_count": len(scenarios),
            "scenario_seed_public_state_sha256": hashlib.sha256(
                json.dumps(
                    [int(seed), int(k), ranks.tolist(), occupied.astype(int).tolist()],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "continuation": "fixed Minmax rule on synthetic complete priority ranks; surrogate, not an unbiased online cost estimate",
        }
