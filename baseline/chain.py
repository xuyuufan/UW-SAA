"""Single-step Chain proposal using only the current public censored frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from baseline.chain_core import eq11_reverse_condition
from baseline.minmax_core import choose_minmax_destination
from environment.public_frame import (
    PUBLIC_UNKNOWN_RANK,
    max_occupied_public_rank,
    public_prefix,
    public_stacks,
)


@dataclass(frozen=True)
class _Token:
    public_rank: int | None
    token_key: tuple[str, int, int]

    @property
    def scoring_rank(self) -> int:
        return PUBLIC_UNKNOWN_RANK if self.public_rank is None else self.public_rank


@dataclass(frozen=True)
class _Next:
    target_rank: int
    token: _Token
    stacks: tuple[tuple[_Token, ...], ...]
    auto_retrieved: tuple[int, ...]


def _token_bay(frame: object) -> list[list[_Token]]:
    ranks = np.asarray(frame.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    bay: list[list[_Token]] = []
    for stack in range(ranks.shape[0]):
        row = []
        for tier in range(int(occupied[stack].sum())):
            rank = int(ranks[stack, tier])
            row.append(
                _Token(
                    public_rank=None if rank == PUBLIC_UNKNOWN_RANK else rank,
                    token_key=("unknown", stack, tier)
                    if rank == PUBLIC_UNKNOWN_RANK
                    else ("rank", rank, 0),
                )
            )
        bay.append(row)
    return bay


def _preview(
    bay_in: Sequence[Sequence[_Token]],
    *,
    target_rank: int,
    current: _Token,
    source: int,
    max_public_rank: int,
    destination: int | None,
) -> _Next | None:
    bay = [list(stack) for stack in bay_in]
    if source < 0 or source >= len(bay) or not bay[source] or bay[source][-1] != current:
        raise ValueError("current public blocker token is not on top of its source")
    bay[source].pop()
    if destination is not None:
        if destination < 0 or destination >= len(bay) or destination == source:
            raise ValueError("virtual move destination is invalid")
        bay[destination].append(current)

    auto: list[int] = []
    # max_public_rank+1 gives a finite public bound; rank 22 itself is the
    # censored tail boundary, never an ordered retrieval target.
    rank_limit = max_public_rank + 1
    for target in range(target_rank, rank_limit):
        if target == PUBLIC_UNKNOWN_RANK:
            return None
        if destination is None and current.public_rank == target:
            # The reverse path removed this known box, so skip its retrieval rank.
            continue
        locations = [
            (stack_index, tier_index)
            for stack_index, stack in enumerate(bay)
            for tier_index, token in enumerate(stack)
            if token.public_rank == target
        ]
        if len(locations) != 1:
            # A known retrieval target cannot be reconstructed from this prefix.
            return None
        target_source, target_tier = locations[0]
        if target_tier != len(bay[target_source]) - 1:
            return _Next(
                target_rank=target,
                token=bay[target_source][-1],
                stacks=tuple(tuple(stack) for stack in bay),
                auto_retrieved=tuple(auto),
            )
        bay[target_source].pop()
        auto.append(target)
    return None


def _rank_bay(bay: Sequence[Sequence[_Token]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(token.scoring_rank for token in stack) for stack in bay)


def _legal_for(bay: Sequence[Sequence[_Token]], source: int, max_tiers: int) -> tuple[int, ...]:
    return tuple(index for index, stack in enumerate(bay) if index != source and len(stack) < max_tiers)


def _locate_token(bay: Sequence[Sequence[_Token]], token: _Token) -> int:
    for index, stack in enumerate(bay):
        if token in stack:
            return index
    raise ValueError("public virtual token is absent")


def _trace_base(h: int, blocker: _Token, destination: int, reason: str, **extra: object) -> dict[str, object]:
    return {
        "solver": "Chain",
        "requested_h": int(h),
        "blocker_public_rank": blocker.scoring_rank,
        "blocker_is_unknown_tail": blocker.public_rank is None,
        "rule": "minmax_fallback" if reason else "chain_eq11_one_step",
        "fallback_reason": reason or None,
        "destination": int(destination),
        **extra,
    }


def propose(public_frame: object, h: int) -> tuple[int, dict[str, object]]:
    """Return exactly one legal destination; never cache or commit a Chain group."""
    frame = public_prefix(public_frame, h)
    ranks = np.asarray(frame.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    target_locations = np.argwhere(occupied & (ranks == 1))
    source, target_tier = (int(value) for value in target_locations[0])
    if int(occupied[source].sum()) <= target_tier + 1:
        raise ValueError("Chain proposal requires a blocked current target")

    token_bay = _token_bay(frame)
    current = token_bay[source][-1]
    ranked_bay = _rank_bay(token_bay)
    legal = tuple(int(value) for value in frame.legal_destinations)
    max_public_rank = max_occupied_public_rank(public_stacks(frame))
    direct = choose_minmax_destination(
        ranked_bay, current.scoring_rank, legal, max_public_rank
    )
    if direct is None:
        raise ValueError("Chain has no legal Minmax fallback destination")
    direct_destination, d1 = direct

    def fallback(reason: str, **extra: object) -> tuple[int, dict[str, object]]:
        return int(direct_destination), _trace_base(
            h, current, direct_destination, reason, d1=int(d1), **extra
        )

    if current.public_rank is None:
        return fallback("unknown_current_rank")

    direct_next = _preview(
        token_bay, target_rank=1, current=current, source=source,
        max_public_rank=max_public_rank, destination=direct_destination,
    )
    if direct_next is None:
        return fallback("direct_prefix_incomplete_or_no_next_blocker")
    if direct_next.token.token_key == current.token_key:
        return fallback("repeated_current_token")

    reverse_next = _preview(
        token_bay, target_rank=1, current=current, source=source,
        max_public_rank=max_public_rank, destination=None,
    )
    if reverse_next is None:
        return fallback("reverse_prefix_incomplete", next_token_match=False)
    same_next_token = direct_next.token.token_key == reverse_next.token.token_key
    if not same_next_token or direct_next.target_rank != reverse_next.target_rank:
        return fallback("next_token_identity_mismatch", next_token_match=False)
    next_token = direct_next.token
    if next_token.token_key == current.token_key:
        return fallback("repeated_current_token", next_token_match=True)

    # A known current rank is required by the paper guard. Unknown next boxes
    # remain anonymous tokens but use only the public tail score 22.
    next_score_rank = next_token.scoring_rank
    if current.public_rank >= next_score_rank:
        return fallback(
            "rank_order_precondition_failed",
            next_token_match=True,
            next_blocker_is_unknown_tail=next_token.public_rank is None,
            next_target_rank=int(direct_next.target_rank),
        )

    next_source_direct = _locate_token(direct_next.stacks, next_token)
    direct_rank_bay = _rank_bay(direct_next.stacks)
    direct_legal = _legal_for(direct_next.stacks, next_source_direct, int(frame.max_tiers))
    d_choice = choose_minmax_destination(
        direct_rank_bay, next_score_rank, direct_legal, max_public_rank
    )
    if d_choice is None:
        return fallback("no_direct_next_destination", next_token_match=True)
    d2 = int(d_choice[1])

    next_source_original = _locate_token(token_bay, next_token)
    reverse_legal = _legal_for(token_bay, next_source_original, int(frame.max_tiers))
    r_choice = choose_minmax_destination(
        ranked_bay, next_score_rank, reverse_legal, max_public_rank
    )
    if r_choice is None:
        return fallback("no_reverse_next_destination", next_token_match=True)
    reverse_destination, r2 = r_choice
    r1_choice = choose_minmax_destination(
        ranked_bay, current.scoring_rank, legal, max_public_rank,
        excluded=(reverse_destination,),
    )
    if r1_choice is None:
        return fallback("no_reverse_first_destination", next_token_match=True)
    r1 = int(r1_choice[1])
    use_reverse = eq11_reverse_condition(int(d1), d2, r1, int(r2))
    detail = {
        "next_token_match": True,
        "next_token_is_unknown_tail": next_token.public_rank is None,
        "next_target_rank": int(direct_next.target_rank),
        "direct_auto_retrieved": list(direct_next.auto_retrieved),
        "reverse_auto_retrieved": list(reverse_next.auto_retrieved),
        "d1": int(d1), "d2": d2, "r1": r1, "r2": int(r2),
        "eq11_and": bool(use_reverse),
    }
    if not use_reverse:
        return int(direct_destination), _trace_base(
            h, current, direct_destination, "", rule="chain_eq11_direct", **detail
        )
    if reverse_destination not in legal:
        return fallback("eq11_destination_illegal", **detail)
    return int(reverse_destination), _trace_base(
        h, current, int(reverse_destination), "", rule="chain_eq11_reverse_destination", **detail
    )
