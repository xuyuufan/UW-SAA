"""Problem instances: the starting layout of a yard, and loading Caserta files.

Mirrors cpp_reference/data_io/CasertaDataSetLoader.cpp and ProblemInstance.cpp,
and CasertaProblemInstanceGenerator.cpp for new (training) instances.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

_FILE_NAME = re.compile(r"data(\d+)-(\d+)-(\d+)\.dat")

# tiers -> stack counts; the 21 Caserta classes (commented-out list in CasertaProblemSetGenerator.cpp)
CASERTA_CLASSES = {
    3: (3, 4, 5, 6, 7, 8),
    4: (4, 5, 6, 7),
    5: (4, 5, 6, 7, 8, 9, 10),
    6: (6, 10),
    10: (6, 10),
}


@dataclass(frozen=True)
class Instance:
    """Starting layout of one yard. Never changes after loading."""

    name: str
    stacks: tuple[tuple[int, ...], ...]  # one tuple per stack, IDs bottom -> top
    n_containers: int
    max_height: int  # tiers + 2


def load_instance(path: str | Path) -> Instance:
    """Read one Caserta .dat file; tiers come from the file name."""
    path = Path(path)
    match = _FILE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a Caserta file name: {path.name}")
    tiers = int(match.group(1))

    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    n_stacks, n_containers = int(rows[0][0]), int(rows[0][1])
    stacks = []
    for row in rows[1:1 + n_stacks]:
        count, ids = int(row[0]), tuple(int(x) for x in row[1:])
        if len(ids) != count:
            raise ValueError(f"{path.name}: stack says {count} containers, found {len(ids)}")
        stacks.append(ids)

    if len(stacks) != n_stacks:
        raise ValueError(f"{path.name}: expected {n_stacks} stacks, found {len(stacks)}")
    if sorted(i for s in stacks for i in s) != list(range(1, n_containers + 1)):
        raise ValueError(f"{path.name}: container IDs are not exactly 1..{n_containers}")

    return Instance(path.stem, tuple(stacks), n_containers, tiers + 2)


def load_dataset(directory: str | Path) -> list[Instance]:
    """Load every Caserta file in a directory, sorted by file name."""
    return [load_instance(p) for p in sorted(Path(directory).glob("data*-*-*.dat"))]


def generate_instance(name: str, n_stacks: int, tiers: int, rng: random.Random) -> Instance:
    """Shuffle IDs 1..N into full stacks, like CasertaProblemInstanceGenerator::GenerateInstance."""
    n = n_stacks * tiers
    ids = list(range(1, n + 1))
    rng.shuffle(ids)
    stacks = tuple(tuple(ids[i * tiers:(i + 1) * tiers]) for i in range(n_stacks))
    return Instance(name, stacks, n, tiers + 2)


def write_instance(instance: Instance, path: str | Path) -> None:
    """Write an instance in the Caserta .dat format."""
    lines = [f"{len(instance.stacks)} {instance.n_containers}"]
    lines += [" ".join(map(str, (len(s), *s))) for s in instance.stacks]
    Path(path).write_text("\n".join(lines) + "\n")
