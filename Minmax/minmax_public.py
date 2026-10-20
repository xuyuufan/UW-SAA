"""Single-step public-input Minmax horizon proposal."""

from __future__ import annotations

import numpy as np

from Minmax.minmax_offline import choose_minmax_destination
from R.scrp.plan052_public import max_occupied_public_rank, public_prefix, public_stacks


def propose(public_frame: object, h: int) -> tuple[int, dict[str, object]]:
    """Return one legal destination; censored 22 and empty 0 remain distinct."""
    frame = public_prefix(public_frame, h)
    stacks = public_stacks(frame)
    ranks = np.asarray(frame.observation.rank_tensor, dtype=np.int32)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    source, target_tier = (int(value) for value in np.argwhere(occupied & (ranks == 1))[0])
    if len(stacks[source]) <= target_tier + 1:
        raise ValueError("Minmax proposal requires a blocked current target")
    blocker = stacks[source][-1]
    legal = tuple(int(value) for value in frame.legal_destinations)
    maximum = max_occupied_public_rank(stacks)
    choice = choose_minmax_destination(stacks, blocker, legal, maximum)
    if choice is None:
        raise ValueError("Minmax proposal has no legal destination")
    destination, score = choice
    p_values = [min(stack) if stack else maximum + 1 for stack in stacks]
    return int(destination), {
        "solver": "Minmax",
        "requested_h": int(h),
        "blocker_public_rank": int(blocker),
        "blocker_is_unknown_tail": blocker == 22,
        "empty_p_value": maximum + 1,
        "candidate_p": [[index, p_values[index]] for index in legal],
        "selected_score": int(score),
        "destination": int(destination),
    }
