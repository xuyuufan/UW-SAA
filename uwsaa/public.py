"""Public UWSAA scenario sampling and candidate-sequence support."""
from __future__ import annotations
import hashlib,json
from typing import Any,Mapping,Sequence
import numpy as np
from environment.noise_aware_forecast import HORIZONS
from baseline.native import SOLVERS
from baseline.sequence_commit import SequenceCommitter
from environment.public_frame import require_public_frame

WINDOWS=(2,5,10,20)
SCENARIOS_PER_WINDOW=2
# Historical sampling namespace is an exact compatibility token, not a method label.
NAMESPACE="plan056_tw_saa_public_scenarios_v1"

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

def _candidate_sequences(
    frame: object, solver: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any]
) -> dict[int, tuple[int, ...]]:
    if solver in ("EM", "GP", "ERI", "EG"):
        from baseline.native import build_native_proposals

        return {
            int(h): tuple(int(value) for value in sequence)
            for h, sequence in build_native_proposals(frame, solver, runtime, gp_rule).items()
        }
    if solver in ("Chain", "Minmax"):
        if solver == "Chain":
            from baseline.chain import propose
        else:
            from baseline.minmax import propose

        return {h: (int(propose(frame, h)[0]),) for h in HORIZONS}
    raise ValueError(f"unknown public UWSAA solver {solver!r}")

class UWSAASelectorBase:
    """Shared public validation and EG group commitment for the final fast selector."""
    def __init__(self,solver,method,runtime,gp_rule):
        if method!="UWSAA": raise ValueError("method must be UWSAA")
        self.solver,self.method,self.runtime,self.gp_rule=solver,method,runtime,gp_rule
        self._eg_committer=SequenceCommitter("EG",self._select_sequence) if solver=="EG" else None
    def select(self,frame):
        require_public_frame(frame)
        if self._eg_committer is not None: return self._eg_committer.select(frame)
        sequence,trace=self._select_sequence(frame)
        if len(sequence)!=1: raise ValueError("non-EG solver must commit one destination")
        return int(sequence[0]),trace
