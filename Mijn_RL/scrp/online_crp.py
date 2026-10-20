"""Leakage-safe scalar environment for the online container relocation problem.

Only the current target and the configured H future targets enter public state.
The complete retrieval order stays private inside :class:`OnlineCRPEnvironment`.
Stack contents are always represented bottom-to-top.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _tuple_stacks(stacks: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in stack) for stack in stacks)


@dataclass(frozen=True)
class OnlineCRPEpisode:
    """A fixed bay layout and its hidden complete retrieval sequence."""

    episode_id: str
    num_stacks: int
    max_tiers: int
    initial_stacks: tuple[tuple[int, ...], ...]
    retrieval_sequence: tuple[int, ...]
    parameter_group: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        stacks = _tuple_stacks(self.initial_stacks)
        sequence = tuple(int(item) for item in self.retrieval_sequence)
        object.__setattr__(self, "initial_stacks", stacks)
        object.__setattr__(self, "retrieval_sequence", sequence)
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.episode_id or not self.parameter_group:
            raise ValueError("episode_id and parameter_group must be non-empty")
        if self.num_stacks <= 1 or self.max_tiers <= 0:
            raise ValueError("an episode needs at least two stacks and positive tiers")
        if len(stacks) != self.num_stacks:
            raise ValueError("initial_stacks length does not match num_stacks")
        if any(len(stack) > self.max_tiers for stack in stacks):
            raise ValueError("initial stack exceeds max_tiers")
        layout_ids = tuple(item for stack in stacks for item in stack)
        if not layout_ids or len(layout_ids) != len(set(layout_ids)):
            raise ValueError("layout container identifiers must be non-empty and unique")
        if len(sequence) != len(set(sequence)) or set(sequence) != set(layout_ids):
            raise ValueError("retrieval_sequence must permute the layout containers")

    @property
    def num_containers(self) -> int:
        return len(self.retrieval_sequence)

    @property
    def fill_rate(self) -> float:
        return self.num_containers / (self.num_stacks * self.max_tiers)

    @property
    def canonical_id(self) -> str:
        identity = {
            "num_stacks": self.num_stacks,
            "max_tiers": self.max_tiers,
            "initial_stacks": self.initial_stacks,
            "retrieval_sequence": self.retrieval_sequence,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "OnlineCRPEpisode":
        return cls(
            episode_id=str(record["episode_id"]),
            num_stacks=int(record["num_stacks"]),
            max_tiers=int(record["max_tiers"]),
            initial_stacks=_tuple_stacks(record["initial_stacks"]),  # type: ignore[arg-type]
            retrieval_sequence=tuple(  # type: ignore[arg-type]
                int(item) for item in record["retrieval_sequence"]
            ),
            parameter_group=str(record["parameter_group"]),
            metadata=dict(record.get("metadata", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class OnlineCRPPublicState:
    """Detached algorithm-visible state with no hidden future labels."""

    stacks: tuple[tuple[int, ...], ...]
    current_target_id: int | None
    retrieval_count: int
    relocation_count: int
    crane_position: int | None
    crane_time: float
    terminated: bool


class OnlineCRPEnvironment:
    """Restricted OCRP transition system whose action is a destination stack."""

    def __init__(
        self,
        episode: OnlineCRPEpisode,
        *,
        max_steps: int = 100_000,
        lookahead_horizon: int = 0,
        crane_travel_time: float = 1.2,
        pick_drop_time: float = 30.0,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if lookahead_horizon < 0:
            raise ValueError("lookahead_horizon must be non-negative")
        if crane_travel_time < 0 or pick_drop_time < 0:
            raise ValueError("crane time coefficients must be non-negative")
        self._episode = episode
        self.max_steps = max_steps
        self.lookahead_horizon = lookahead_horizon
        self.crane_travel_time = float(crane_travel_time)
        self.pick_drop_time = float(pick_drop_time)
        self._sequence = episode.retrieval_sequence
        self._stacks: list[list[int]] = []
        self._locations: dict[int, int | None] = {}
        self._next_index = 0
        self._relocations = 0
        self._crane_position: int | None = None
        self._crane_time = 0.0
        self._terminated = False
        self.reset()

    def reset(self) -> OnlineCRPPublicState:
        self._stacks = [list(stack) for stack in self._episode.initial_stacks]
        self._locations = {
            item: stack_id
            for stack_id, stack in enumerate(self._stacks)
            for item in stack
        }
        self._next_index = 0
        self._relocations = 0
        self._crane_position = None
        self._crane_time = 0.0
        self._terminated = False
        self._advance_free_retrievals()
        return self.state

    @property
    def state(self) -> OnlineCRPPublicState:
        # Positive public values are relative priorities, never physical IDs.
        # Zero means occupied but unrevealed; the tensor adapter maps it to -1.
        visible_end = min(
            self._next_index + self.lookahead_horizon + 1, len(self._sequence)
        )
        visible = {
            item: offset + 1
            for offset, item in enumerate(self._sequence[self._next_index:visible_end])
        }
        stacks = tuple(
            tuple(visible.get(item, 0) for item in stack) for stack in self._stacks
        )
        return OnlineCRPPublicState(
            stacks=stacks,
            current_target_id=None if self._terminated else 1,
            retrieval_count=self._next_index,
            relocation_count=self._relocations,
            crane_position=self._crane_position,
            crane_time=self._crane_time,
            terminated=self._terminated,
        )

    @property
    def current_target_id(self) -> int | None:
        return None if self._terminated else self._sequence[self._next_index]

    @property
    def relocation_count(self) -> int:
        return self._relocations

    @property
    def objective_costs(self) -> tuple[int, float]:
        return self._relocations, self._crane_time

    def legal_destinations(self) -> tuple[int, ...]:
        if self._terminated:
            return ()
        source = self._source_stack()
        return tuple(
            stack_id
            for stack_id, stack in enumerate(self._stacks)
            if stack_id != source and len(stack) < self._episode.max_tiers
        )

    def step(self, destination_stack_id: int) -> tuple[OnlineCRPPublicState, int, bool]:
        if self._terminated:
            raise RuntimeError("cannot step a terminated episode")
        if self._relocations >= self.max_steps:
            raise RuntimeError("relocation safety limit reached")
        legal = self.legal_destinations()
        if destination_stack_id not in legal:
            raise ValueError(f"illegal destination {destination_stack_id}; legal={legal}")
        source = self._source_stack()
        blocker = self._stacks[source].pop()
        source_position = source + 1
        destination_position = destination_stack_id + 1
        if self._crane_position is not None:
            self._crane_time += abs(self._crane_position - source_position) * self.crane_travel_time
        self._crane_time += (
            abs(source_position - destination_position) * self.crane_travel_time
            + self.pick_drop_time
        )
        self._crane_position = destination_position
        self._stacks[destination_stack_id].append(blocker)
        self._locations[blocker] = destination_stack_id
        self._relocations += 1
        self._advance_free_retrievals()
        return self.state, -1, self._terminated

    def _source_stack(self) -> int:
        target = self.current_target_id
        if target is None:
            raise RuntimeError("terminated episode has no source stack")
        source = self._locations[target]
        if source is None:
            raise RuntimeError("current target was already retrieved")
        if self._stacks[source][-1] == target:
            raise RuntimeError("accessible target escaped automatic retrieval")
        return source

    def _advance_free_retrievals(self) -> None:
        while self._next_index < len(self._sequence):
            target = self._sequence[self._next_index]
            source = self._locations[target]
            if source is None:
                raise RuntimeError("retrieval sequence repeated a retrieved target")
            if self._stacks[source][-1] != target:
                return
            source_position = source + 1
            if self._crane_position is not None:
                self._crane_time += abs(self._crane_position - source_position) * self.crane_travel_time
            self._crane_time += source_position * self.crane_travel_time + self.pick_drop_time
            self._crane_position = 0
            if self._stacks[source].pop() != target:
                raise RuntimeError("target/location invariant failed")
            self._locations[target] = None
            self._next_index += 1
        self._terminated = True


class LevelingBaseline:
    """Send a blocker to the lowest legal stack, breaking ties leftmost."""

    name = "paper_leveling_L"

    @staticmethod
    def select_destination(
        state: OnlineCRPPublicState, legal_destinations: Sequence[int]
    ) -> int:
        if not legal_destinations:
            raise ValueError("leveling requires a legal destination")
        return min(
            legal_destinations,
            key=lambda stack_id: (len(state.stacks[stack_id]), stack_id),
        )


def load_episode_manifest(path: str | Path) -> tuple[OnlineCRPEpisode, ...]:
    """Read a non-empty JSON manifest into validated immutable episodes."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("episode manifest must be a non-empty JSON list")
    return tuple(OnlineCRPEpisode.from_record(record) for record in records)


__all__ = [
    "LevelingBaseline",
    "OnlineCRPEpisode",
    "OnlineCRPEnvironment",
    "OnlineCRPPublicState",
    "load_episode_manifest",
]
