"""GP trees the way ECF stores them: nodes in prefix order, each with its depth and subtree size.

Mirrors ECF 1.4.2 tree/Tree.cpp (Tree::update, setSize). The root has depth 0, so
max depth 5 allows 6 levels (up to 63 nodes).
"""
from __future__ import annotations

import random
from typing import Sequence

from crp.gp.expression import OPERATORS, TERMINALS

MAX_DEPTH = 5
PRIMITIVES = OPERATORS + TERMINALS  # ECF primitives_: functions and terminals


class GPTree:
    """An unchangeable tree; operators build new ones."""

    __slots__ = ("nodes", "depths", "sizes")

    def __init__(self, nodes: Sequence[str]) -> None:
        self.nodes = tuple(nodes)
        self.depths, self.sizes = _depths_and_sizes(self.nodes)

    @classmethod
    def from_text(cls, text: str) -> GPTree:
        return cls(text.split())

    def text(self) -> str:
        """Prefix formula, ready for compile_formula or the fitness cache."""
        return " ".join(self.nodes)

    def depth(self) -> int:
        return max(self.depths)

    def is_function(self, i: int) -> bool:
        return self.nodes[i] in OPERATORS

    def subtree(self, i: int) -> tuple[str, ...]:
        """The nodes of the subtree rooted at position i."""
        return self.nodes[i:i + self.sizes[i]]

    def __len__(self) -> int:
        return len(self.nodes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GPTree) and self.nodes == other.nodes

    def __repr__(self) -> str:
        return f"GPTree({self.text()!r})"


def _depths_and_sizes(nodes: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Depth of every node (root = 0) and size of the subtree under it (itself included)."""
    depths, sizes = [0] * len(nodes), [1] * len(nodes)

    def walk(i: int, depth: int) -> int:
        """Fill in the subtree starting at i; return the index right after it."""
        if i >= len(nodes):
            raise ValueError(f"formula ends too early: {' '.join(nodes)!r}")
        token = nodes[i]
        depths[i] = depth
        if token in TERMINALS:
            return i + 1
        if token not in OPERATORS:
            raise ValueError(f"unknown token {token!r}")
        end = walk(walk(i + 1, depth + 1), depth + 1)  # two children
        sizes[i] = end - i
        return end

    if walk(0, 0) != len(nodes):
        raise ValueError(f"extra tokens: {' '.join(nodes)!r}")
    return tuple(depths), tuple(sizes)


# --- initialisation: ECF Tree::initialize, fullBuild, growBuild ---

def full_tree(rng: random.Random, max_depth: int = MAX_DEPTH) -> GPTree:
    """Functions on every level above max_depth, terminals at max_depth."""
    nodes: list[str] = []

    def build(depth: int) -> None:
        if depth < max_depth:
            nodes.append(rng.choice(OPERATORS))
            build(depth + 1)
            build(depth + 1)
        else:
            nodes.append(rng.choice(TERMINALS))

    build(0)
    return GPTree(nodes)


def grow_tree(rng: random.Random, max_depth: int = MAX_DEPTH, min_depth: int = 1) -> GPTree:
    """Root is a function (min_depth 1); below it any primitive until max_depth."""
    nodes: list[str] = []

    def build(depth: int) -> None:
        if depth < min_depth:
            token = rng.choice(OPERATORS)
        elif depth < max_depth:
            token = rng.choice(PRIMITIVES)
        else:
            token = rng.choice(TERMINALS)
        nodes.append(token)
        if token in OPERATORS:
            build(depth + 1)
            build(depth + 1)

    build(0)
    return GPTree(nodes)


def random_tree(rng: random.Random) -> GPTree:
    """50% full, 50% grow, always at max depth: not ramped (ECF: "TODO: ramped")."""
    return full_tree(rng) if rng.randint(0, 1) == 0 else grow_tree(rng)


def initial_population(size: int, rng: random.Random) -> list[GPTree]:
    return [random_tree(rng) for _ in range(size)]
