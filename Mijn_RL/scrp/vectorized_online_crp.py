"""Torch-vectorized restricted online container-relocation environment.

The scalar :class:`~scrp.online_crp.OnlineCRPEnvironment` is intentionally
simple and is useful for evaluation.  Training, however, advances many copies
of same-sized episodes.  This module keeps those copies in dense Torch tensors
so layout transitions, online revelation, legal-action masks, and relocation
costs can remain on a CPU or CUDA device.

Batch rows are episode-major: for ``E`` episodes and ``M`` rollouts the row for
episode ``e`` and rollout ``m`` is ``e * M + m``.  Physical container labels
never enter an observation.  Internally, every label is replaced by its
one-based position in that episode's hidden retrieval sequence.  Observation
values use the same convention as :class:`Online2DPriorityAdapter`: ``0`` is
an empty tier, ``-1`` is an occupied but unrevealed tier, ``1`` is the current
target, and ``2..H+1`` are the H revealed future targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .online_crp import OnlineCRPEpisode


@dataclass(frozen=True)
class VectorizedOnlineCRPState:
    """A detached-value view of one vectorized environment state.

    The tensors stay on the environment device.  Count and termination tensors
    are cloned when the state is built, so a previously returned state remains
    a valid snapshot after a later :meth:`VectorizedOnlineCRPEnvironment.step`.
    """

    priorities: torch.Tensor
    legal_destination_mask: torch.Tensor
    forbidden_mask: torch.Tensor
    retrieval_count: torch.Tensor
    relocation_count: torch.Tensor
    crane_position: torch.Tensor
    crane_time: torch.Tensor
    terminated: torch.Tensor

    @property
    def active(self) -> torch.Tensor:
        return ~self.terminated


class VectorizedOnlineCRPEnvironment:
    """Restricted online CRP transitions for a same-scale episode batch.

    Parameters
    ----------
    episodes:
        Non-empty episodes sharing ``num_stacks`` and ``max_tiers``.  Their
        container counts may differ.
    rollouts_per_episode:
        Number of initially identical rollout rows made for each episode.
    lookahead_horizon:
        Number of future retrieval targets visible in addition to the current
        target. ``0`` is OCRP_0.
    device:
        Any device accepted by :class:`torch.device`, including ``"cuda"``.
    validate_actions:
        Check the safety limit and legality of every active action before a
        transition.  Disable this in a trusted training hot path to avoid two
        host/device synchronizations per relocation step.

    Notes
    -----
    ``step`` accepts either one action for every batch row or one action for
    every currently active row.  Actions belonging to terminated rows in the
    full-batch form are ignored.  Rewards are ``-1`` for rows relocated during
    that call and ``0`` for rows which were already terminated.
    """

    def __init__(
        self,
        episodes: Sequence[OnlineCRPEpisode],
        *,
        rollouts_per_episode: int = 1,
        lookahead_horizon: int = 0,
        device: str | torch.device = "cpu",
        max_steps: int = 100_000,
        validate_actions: bool = True,
        crane_travel_time: float = 1.2,
        pick_drop_time: float = 30.0,
    ) -> None:
        episode_tuple = tuple(episodes)
        if not episode_tuple:
            raise ValueError("vectorized OCRP requires at least one episode")
        if rollouts_per_episode <= 0:
            raise ValueError("rollouts_per_episode must be positive")
        if lookahead_horizon < 0:
            raise ValueError("lookahead_horizon must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if crane_travel_time < 0 or pick_drop_time < 0:
            raise ValueError("crane time coefficients must be non-negative")

        first_shape = (
            episode_tuple[0].num_stacks,
            episode_tuple[0].max_tiers,
        )
        if any(
            (episode.num_stacks, episode.max_tiers) != first_shape
            for episode in episode_tuple
        ):
            raise ValueError(
                "all vectorized OCRP episodes must share num_stacks and max_tiers"
            )

        # Keep the complete sequences private.  They contain exactly the
        # future information which an online policy must not observe.
        self._episodes = episode_tuple
        self.episode_count = len(episode_tuple)
        self.rollouts_per_episode = int(rollouts_per_episode)
        self.batch_size = self.episode_count * self.rollouts_per_episode
        self.num_stacks, self.max_tiers = first_shape
        self.lookahead_horizon = int(lookahead_horizon)
        self.max_steps = int(max_steps)
        self.validate_actions = bool(validate_actions)
        self.crane_travel_time = float(crane_travel_time)
        self.pick_drop_time = float(pick_drop_time)
        self.device = torch.device(device)
        self._max_num_containers = max(
            episode.num_containers for episode in episode_tuple
        )

        # Container labels are opaque.  Convert each layout to one-based
        # retrieval ranks, which are compact and sufficient for all dynamics.
        layouts = torch.zeros(
            (self.episode_count, self.num_stacks, self.max_tiers),
            dtype=torch.int64,
        )
        heights = torch.zeros(
            (self.episode_count, self.num_stacks), dtype=torch.int64
        )
        locations = torch.full(
            (self.episode_count, self._max_num_containers + 1),
            -1,
            dtype=torch.int64,
        )
        container_counts = torch.empty(self.episode_count, dtype=torch.int64)
        for episode_index, episode in enumerate(episode_tuple):
            rank_by_label = {
                label: rank
                for rank, label in enumerate(episode.retrieval_sequence, start=1)
            }
            container_counts[episode_index] = episode.num_containers
            for stack_index, stack in enumerate(episode.initial_stacks):
                heights[episode_index, stack_index] = len(stack)
                for tier_index, label in enumerate(stack):
                    rank = rank_by_label[label]
                    layouts[episode_index, stack_index, tier_index] = rank
                    locations[episode_index, rank] = stack_index

        repeat = self.rollouts_per_episode
        self._initial_layouts = layouts.repeat_interleave(repeat, dim=0).to(
            self.device
        )
        self._initial_heights = heights.repeat_interleave(repeat, dim=0).to(
            self.device
        )
        self._initial_locations = locations.repeat_interleave(repeat, dim=0).to(
            self.device
        )
        self.num_containers = container_counts.repeat_interleave(
            repeat, dim=0
        ).to(self.device)
        self.episode_indices = torch.arange(
            self.episode_count, dtype=torch.int64, device=self.device
        ).repeat_interleave(repeat)
        self.rollout_indices = torch.arange(
            repeat, dtype=torch.int64, device=self.device
        ).repeat(self.episode_count)
        self._batch_indices = torch.arange(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self._stack_ids = torch.arange(
            self.num_stacks, dtype=torch.int64, device=self.device
        )[None, :]
        self._rank_ids = torch.arange(
            self._max_num_containers + 1,
            dtype=torch.int64,
            device=self.device,
        )[None, :]

        self._layouts: torch.Tensor
        self._heights: torch.Tensor
        self._locations: torch.Tensor
        self.retrieval_count: torch.Tensor
        self.relocation_count: torch.Tensor
        self.crane_position: torch.Tensor
        self.crane_time: torch.Tensor
        self.terminated: torch.Tensor
        self.reset()

    def reset(self) -> VectorizedOnlineCRPState:
        """Restore all rollout rows and perform every free initial retrieval."""

        self._layouts = self._initial_layouts.clone()
        self._heights = self._initial_heights.clone()
        self._locations = self._initial_locations.clone()
        self.retrieval_count = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.relocation_count = torch.zeros_like(self.retrieval_count)
        # Positions are one-based stack coordinates; zero is the I/O point and
        # -1 means that the first operation has not established a position yet.
        self.crane_position = torch.full_like(self.retrieval_count, -1)
        self.crane_time = torch.zeros(
            self.batch_size, dtype=torch.float64, device=self.device
        )
        self.terminated = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self._advance_free_retrievals()
        return self.state

    @property
    def active(self) -> torch.Tensor:
        return ~self.terminated

    @property
    def active_indices(self) -> torch.Tensor:
        return self._batch_indices[self.active]

    @property
    def source_stack_indices(self) -> torch.Tensor:
        """Current target stack for each row, or ``-1`` after termination."""

        target_ranks = torch.where(
            self.active,
            self.retrieval_count + 1,
            torch.zeros_like(self.retrieval_count),
        )
        return self._locations.gather(1, target_ranks[:, None]).squeeze(1)

    @property
    def priorities(self) -> torch.Tensor:
        """Return the leakage-safe ``[batch, stack, tier]`` 2D observation."""

        relative = self._layouts - self.retrieval_count[:, None, None]
        occupied = self._layouts > 0
        revealed = occupied & (relative <= self.lookahead_horizon + 1)
        unknown = torch.full_like(relative, -1)
        public = torch.where(revealed, relative, unknown)
        public = torch.where(occupied, public, torch.zeros_like(public))
        return public.to(torch.float32)

    @property
    def legal_destination_mask(self) -> torch.Tensor:
        """Boolean mask where ``True`` denotes a legal destination stack."""

        sources = self.source_stack_indices[:, None]
        return (
            self.active[:, None]
            & (self._heights < self.max_tiers)
            & (self._stack_ids != sources)
        )

    @property
    def forbidden_mask(self) -> torch.Tensor:
        """Policy-compatible mask where ``True`` denotes a forbidden stack."""

        return ~self.legal_destination_mask

    @property
    def state(self) -> VectorizedOnlineCRPState:
        legal = self.legal_destination_mask
        return VectorizedOnlineCRPState(
            priorities=self.priorities,
            legal_destination_mask=legal,
            forbidden_mask=~legal,
            retrieval_count=self.retrieval_count.clone(),
            relocation_count=self.relocation_count.clone(),
            crane_position=self.crane_position.clone(),
            crane_time=self.crane_time.clone(),
            terminated=self.terminated.clone(),
        )

    @property
    def objective_costs(self) -> torch.Tensor:
        """Return ``[batch, 2]`` costs: relocations and crane working time."""

        return torch.stack(
            (self.relocation_count.to(torch.float64), self.crane_time), dim=1
        )

    def step(
        self, actions: torch.Tensor | Sequence[int]
    ) -> tuple[VectorizedOnlineCRPState, torch.Tensor, torch.Tensor]:
        """Relocate one blocker in every active rollout and auto-retrieve.

        ``actions`` can have shape ``[batch_size]`` or ``[active_count]``.  The
        returned reward and termination tensors always have ``batch_size``
        elements.
        """

        active_before = self._apply_actions(actions)
        rewards = -active_before.to(torch.float32)
        return self.state, rewards, self.terminated.clone()

    def step_active_(self, actions: torch.Tensor | Sequence[int]) -> None:
        """Training hot path: advance active rows without building outputs."""

        self._apply_actions(actions)

    def _apply_actions(
        self, actions: torch.Tensor | Sequence[int]
    ) -> torch.Tensor:
        action_tensor = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).reshape(-1)
        active_before = self.active
        rows = self._batch_indices[active_before]
        if action_tensor.numel() == self.batch_size:
            destinations = action_tensor[active_before]
        elif action_tensor.numel() == rows.numel():
            destinations = action_tensor
        else:
            raise ValueError(
                "actions must contain one value per batch row or active row"
            )

        if self.validate_actions:
            at_limit = active_before & (self.relocation_count >= self.max_steps)
            if bool(torch.any(at_limit).item()):
                raise RuntimeError("OCRP relocation safety limit reached")
            out_of_range = (destinations < 0) | (
                destinations >= self.num_stacks
            )
            if bool(torch.any(out_of_range).item()):
                bad_rows = rows[out_of_range].detach().cpu().tolist()
                raise ValueError(f"illegal destination for batch rows {bad_rows}")
            legal = self.legal_destination_mask
            selected_legal = legal[rows, destinations]
            if bool(torch.any(~selected_legal).item()):
                bad_rows = rows[~selected_legal].detach().cpu().tolist()
                raise ValueError(f"illegal destination for batch rows {bad_rows}")

        # A fully terminated batch is a no-op.  It is useful in fixed-length
        # outer loops and mirrors the treatment of already terminated rows in
        # a partially active batch.
        if rows.numel() == 0:
            return active_before

        sources = self.source_stack_indices[active_before]
        source_positions = sources + 1
        destination_positions = destinations + 1
        previous_positions = self.crane_position[rows]
        approach_distance = torch.where(
            previous_positions >= 0,
            (previous_positions - source_positions).abs(),
            torch.zeros_like(source_positions),
        )
        relocation_distance = (source_positions - destination_positions).abs()
        self.crane_time[rows] += (
            (approach_distance + relocation_distance).to(torch.float64)
            * self.crane_travel_time
            + self.pick_drop_time
        )
        self.crane_position[rows] = destination_positions
        source_tiers = self._heights[rows, sources] - 1
        destination_tiers = self._heights[rows, destinations]
        blockers = self._layouts[rows, sources, source_tiers]

        self._layouts[rows, sources, source_tiers] = 0
        self._layouts[rows, destinations, destination_tiers] = blockers
        self._heights[rows, sources] -= 1
        self._heights[rows, destinations] += 1
        self._locations[rows, blockers] = destinations
        self.relocation_count[rows] += 1

        self._advance_free_retrievals()
        return active_before

    def _advance_free_retrievals(self) -> None:
        """Remove every consecutive accessible target in one tensor pass.

        A container with retrieval rank ``r`` blocks the sequence exactly when
        some container above it in the same stack has rank greater than ``r``.
        Containers above it with smaller ranks are retrieved first and vanish.
        Therefore the smallest such blocked rank is the next decision point;
        all ranks before it can be removed at once.  This is equivalent to the
        scalar while-loop but avoids a CUDA-to-host synchronization per free
        retrieval.
        """

        # Maximum rank strictly above each occupied tier.  Layout tiers are
        # bottom-to-top, so a flipped cumulative maximum gives the suffix max.
        suffix_max = torch.flip(
            torch.cummax(torch.flip(self._layouts, dims=(2,)), dim=2).values,
            dims=(2,),
        )
        maximum_above = torch.zeros_like(self._layouts)
        maximum_above[:, :, :-1] = suffix_max[:, :, 1:]
        occupied = self._layouts > 0
        not_already_retrieved = (
            self._layouts > self.retrieval_count[:, None, None]
        )
        blocked = (
            occupied
            & not_already_retrieved
            & (maximum_above > self._layouts)
        )

        sentinel = self._max_num_containers + 1
        blocked_ranks = torch.where(
            blocked,
            self._layouts,
            torch.full_like(self._layouts, sentinel),
        )
        next_blocked = blocked_ranks.amin(dim=(1, 2))
        new_retrieval_count = torch.minimum(
            next_blocked - 1, self.num_containers
        )

        # Account for each automatic retrieval in rank order before clearing
        # its location.  The loop length is shape-static and performs no host
        # synchronization, so it remains suitable for the CUDA hot path.
        old_retrieval_count = self.retrieval_count.clone()
        for offset in range(1, self._max_num_containers + 1):
            ranks = old_retrieval_count + offset
            retrieve = ranks <= new_retrieval_count
            safe_ranks = ranks.clamp(max=self._max_num_containers)
            source_indices = self._locations.gather(
                1, safe_ranks[:, None]
            ).squeeze(1)
            source_positions = source_indices + 1
            previous_positions = self.crane_position
            approach_distance = torch.where(
                previous_positions >= 0,
                (previous_positions - source_positions).abs(),
                torch.zeros_like(source_positions),
            )
            retrieval_time = (
                (approach_distance + source_positions).to(torch.float64)
                * self.crane_travel_time
                + self.pick_drop_time
            )
            self.crane_time += torch.where(
                retrieve, retrieval_time, torch.zeros_like(retrieval_time)
            )
            self.crane_position = torch.where(
                retrieve, torch.zeros_like(self.crane_position), self.crane_position
            )

        freely_retrieved = occupied & (
            self._layouts <= new_retrieval_count[:, None, None]
        )
        self._layouts.masked_fill_(freely_retrieved, 0)
        self._heights = occupied.logical_and(~freely_retrieved).sum(dim=2)
        retrieved_ranks = (
            (self._rank_ids > 0)
            & (self._rank_ids <= new_retrieval_count[:, None])
        )
        self._locations.masked_fill_(retrieved_ranks, -1)
        self.retrieval_count = new_retrieval_count
        self.terminated = self.retrieval_count >= self.num_containers


__all__ = [
    "VectorizedOnlineCRPEnvironment",
    "VectorizedOnlineCRPState",
]
