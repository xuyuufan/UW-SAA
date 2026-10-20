"""Small offline-only physics and result types shared by PLAN-052 solvers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Sequence


class OfflineInputError(ValueError):
    """The complete-rank offline instance violates the declared contract."""


class NoLegalDestinationError(RuntimeError):
    """A blocked target has no capacity-feasible non-source stack."""


class IllegalSolverMoveError(RuntimeError):
    """A solver selected a destination outside the legal action set."""


@dataclass(frozen=True)
class Selection:
    destination: int
    rule: str
    reason: str | None = None


@dataclass(frozen=True)
class Relocation:
    target_rank: int
    blocker_rank: int
    source: int
    destination: int
    rule: str
    reason: str | None = None


@dataclass(frozen=True)
class OfflineResult:
    relocations: tuple[Relocation, ...]
    retrieved_ranks: tuple[int, ...]
    final_stacks: tuple[tuple[int, ...], ...]

    @property
    def relocation_count(self) -> int:
        return len(self.relocations)


def normalize_instance(
    stacks: Sequence[Sequence[int]], max_tiers: int
) -> tuple[list[list[int]], int]:
    if isinstance(max_tiers, bool) or not isinstance(max_tiers, Integral):
        raise OfflineInputError("max_tiers must be a positive integer")
    max_tiers = int(max_tiers)
    if max_tiers < 1:
        raise OfflineInputError("max_tiers must be a positive integer")
    if not isinstance(stacks, Sequence) or len(stacks) < 1:
        raise OfflineInputError("stacks must contain at least one stack")

    bay: list[list[int]] = []
    seen: list[int] = []
    for index, stack in enumerate(stacks):
        if not isinstance(stack, Sequence):
            raise OfflineInputError(f"stack {index} must be a sequence")
        values: list[int] = []
        for value in stack:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise OfflineInputError("container ranks must be integers")
            values.append(int(value))
        if len(values) > max_tiers:
            raise OfflineInputError(f"stack {index} exceeds max_tiers")
        bay.append(values)
        seen.extend(values)

    n = len(seen)
    if sorted(seen) != list(range(1, n + 1)):
        raise OfflineInputError("offline ranks must be exactly the unique set 1..N")
    return bay, n


def minimum_rank(stack: Sequence[int], n: int) -> int:
    """Paper p(i): the earliest retrieval rank, or N+1 for an empty stack."""
    return min(stack) if stack else n + 1


def legal_destinations(
    bay: Sequence[Sequence[int]], source: int, max_tiers: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, stack in enumerate(bay)
        if index != source and len(stack) < max_tiers
    )


def stack_containing(bay: Sequence[Sequence[int]], rank: int) -> int:
    for index, stack in enumerate(bay):
        if rank in stack:
            return index
    raise OfflineInputError(f"rank {rank} is absent from the current bay")


Selector = Callable[
    [tuple[tuple[int, ...], ...], int, int, int, tuple[int, ...], int], Selection
]


def run_offline(
    stacks: Sequence[Sequence[int]], max_tiers: int, selector: Selector
) -> OfflineResult:
    """Run the restricted/LIFO retrieval process using a paper-specific selector."""
    bay, n = normalize_instance(stacks, max_tiers)
    relocations: list[Relocation] = []
    retrieved: list[int] = []

    for target in range(1, n + 1):
        source = stack_containing(bay, target)
        while bay[source][-1] != target:
            blocker = bay[source][-1]
            legal = legal_destinations(bay, source, max_tiers)
            if not legal:
                raise NoLegalDestinationError(
                    f"target {target} is blocked by {blocker}, with no legal destination"
                )
            public_bay = tuple(tuple(stack) for stack in bay)
            choice = selector(public_bay, target, source, blocker, legal, n)
            if choice.destination not in legal:
                raise IllegalSolverMoveError(
                    f"selector chose illegal destination {choice.destination}; legal={legal}"
                )
            bay[choice.destination].append(blocker)
            bay[source].pop()
            relocations.append(
                Relocation(
                    target_rank=target,
                    blocker_rank=blocker,
                    source=source,
                    destination=choice.destination,
                    rule=choice.rule,
                    reason=choice.reason,
                )
            )
        bay[source].pop()
        retrieved.append(target)

    if any(bay):
        raise AssertionError("all ranks were retrieved but a stack is not empty")
    return OfflineResult(
        relocations=tuple(relocations),
        retrieved_ranks=tuple(retrieved),
        final_stacks=tuple(tuple(stack) for stack in bay),
    )
