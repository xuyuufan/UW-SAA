"""Expected-Minmax public selector used by the EM comparison."""
from __future__ import annotations
import math
import numpy as np
from environment.noise_aware_forecast import HORIZONS, PublicObservation

def horizon_ranks(observation: PublicObservation, h: int) -> np.ndarray:
    """Return an EM-h view from the same detached Hmax tensor."""
    if h < 0:
        raise ValueError("h must be non-negative")
    ranks = np.array(observation.rank_tensor, copy=True)
    occupied = observation.occupied_mask
    ranks[occupied & (ranks > h + 1)] = h + 2
    ranks[~occupied] = 0
    return ranks

def _source_and_blocker(
    observation: PublicObservation, ranks: np.ndarray
) -> tuple[int, int]:
    targets = np.argwhere(observation.occupied_mask & (ranks == 1))
    if targets.shape != (1, 2):
        raise ValueError("public state must contain exactly one current target")
    source, target_tier = (int(x) for x in targets[0])
    source_height = int(observation.occupied_mask[source].sum())
    if target_tier >= source_height - 1:
        raise ValueError("an accessible target must have been auto-retrieved")
    blocker_rank = int(ranks[source, source_height - 1])
    return source, blocker_rank

def select_em_destination(
    observation: PublicObservation, *, max_tiers: int, h: int
) -> int:
    """Choose one legal destination using the published EM rules and ties."""
    if max_tiers <= 0:
        raise ValueError("max_tiers must be positive")
    ranks = horizon_ranks(observation, h)
    _source, blocker = _source_and_blocker(observation, ranks)
    legal = tuple(int(x) for x in np.flatnonzero(observation.legal_destination_mask))
    if not legal:
        raise ValueError("EM requires at least one legal destination")

    heights = observation.occupied_mask.sum(axis=1).astype(int)

    def min_label(stack_id: int) -> float:
        labels = ranks[stack_id, observation.occupied_mask[stack_id]]
        return float(labels.min()) if labels.size else math.inf

    minima = {stack_id: min_label(stack_id) for stack_id in legal}
    safe = [stack_id for stack_id in legal if minima[stack_id] > blocker]
    if safe:
        best_min = min(minima[stack_id] for stack_id in safe)
        tied = [stack_id for stack_id in safe if minima[stack_id] == best_min]
        return min(tied, key=lambda stack_id: (-heights[stack_id], stack_id))

    best_min = max(minima.values())
    tied = [stack_id for stack_id in legal if minima[stack_id] == best_min]
    if math.isinf(best_min):
        raise RuntimeError("empty destinations should have entered EM Rule 1")
    min_count = min(
        int(np.count_nonzero(ranks[stack_id, observation.occupied_mask[stack_id]] == best_min))
        for stack_id in tied
    )
    tied = [
        stack_id
        for stack_id in tied
        if int(
            np.count_nonzero(
                ranks[stack_id, observation.occupied_mask[stack_id]] == best_min
            )
        )
        == min_count
    ]
    return min(tied, key=lambda stack_id: (-heights[stack_id], stack_id))
