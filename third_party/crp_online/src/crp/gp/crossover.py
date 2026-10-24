"""The 5 ECF tree crossovers, ported from ECF 1.4.2 (tree/TreeCrx*.cpp, Crossover.cpp).

Each operator takes two parents and returns the child, or None when it gives up
(then the caller keeps the old individual, as ECF does). With all crx.* probabilities
at their default 0, ECF picks one of the five uniformly for every mating.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from crp.gp.tree import MAX_DEPTH, GPTree

FUNC_CHOICE_PROB = 0.9  # crx.simple.functionprob default
MAX_TRIES = 5           # "nTries > 4"


def mate(male: GPTree, female: GPTree, rng: random.Random) -> Optional[GPTree]:
    """Crossover::mate: one of the 5 operators, chosen uniformly."""
    return rng.choice(OPERATORS)(male, female, rng)


def _splice(male: GPTree, m: int, female: GPTree, f: int) -> GPTree:
    """Male nodes before m, female subtree at f, rest of male after m's subtree."""
    return GPTree(male.nodes[:m] + female.subtree(f) + male.nodes[m + male.sizes[m]:])


def _fits(male: GPTree, m: int, female: GPTree, f: int) -> bool:
    """mNodeDepth + depth of female subtree <= maxDepth."""
    end = f + female.sizes[f]
    subtree_depth = max(female.depths[f:end]) - female.depths[f]
    return male.depths[m] + subtree_depth <= MAX_DEPTH


def _try_points(male: GPTree, female: GPTree, pick: Callable[[], tuple[int, int]]) -> Optional[GPTree]:
    """Up to 5 tries to find crossover points that respect the max depth."""
    for _ in range(MAX_TRIES):
        m, f = pick()
        if _fits(male, m, female, f):
            return _splice(male, m, female, f)
    return None  # "not successful": ECF leaves the child as it was


def simple(male: GPTree, female: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeCrxSimple: subtree crossover, function nodes chosen with probability 0.9."""
    def split(tree: GPTree) -> tuple[list[int], list[int]]:
        leaves = [i for i in range(len(tree)) if tree.sizes[i] == 1]
        return leaves, [i for i in range(len(tree)) if tree.sizes[i] > 1]

    m_leaves, m_inner = split(male)
    f_leaves, f_inner = split(female)

    def point(leaves: list[int], inner: list[int]) -> int:
        if rng.random() > FUNC_CHOICE_PROB or not inner:
            return rng.choice(leaves)
        return rng.choice(inner)

    return _try_points(male, female, lambda: (point(m_leaves, m_inner), point(f_leaves, f_inner)))


def context_preserved(male: GPTree, female: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeCrxContextPreserved: swap subtrees at the same position in both trees."""
    m_common, f_common = [], []
    m_skip: list[int] = []  # stacks of subtree starts to jump over; next one on top
    f_skip: list[int] = []
    im = jf = 0
    while im < len(male) and jf < len(female):
        while m_skip and im == m_skip[-1]:
            m_skip.pop()
            im += male.sizes[im]
        if im >= len(male):
            break
        while f_skip and jf == f_skip[-1]:
            f_skip.pop()
            jf += female.sizes[jf]
        if jf >= len(female):
            break
        m_common.append(im)
        f_common.append(jf)
        m_args, f_args = _arity(male, im), _arity(female, jf)
        if m_args > f_args:
            m_skip.extend(reversed(_child_starts(male, im)[f_args:]))
        if m_args < f_args:
            f_skip.extend(reversed(_child_starts(female, jf)[m_args:]))
        im += 1
        jf += 1
    if not m_common:
        return None
    k = rng.randrange(len(m_common))
    return _splice(male, m_common[k], female, f_common[k])


def uniform(male: GPTree, female: GPTree, rng: random.Random) -> GPTree:
    """TreeCrxUniform: walk both trees together, each common node from a random parent."""
    nodes: list[str] = []
    im = jf = 0
    while im < len(male) and jf < len(female):
        pick_male = rng.randint(0, 1) == 1
        if _arity(male, im) == _arity(female, jf):
            nodes.append(male.nodes[im] if pick_male else female.nodes[jf])
            im += 1
            jf += 1
        else:  # different shape here: copy a whole subtree from one parent
            nodes.extend(male.subtree(im) if pick_male else female.subtree(jf))
            im += male.sizes[im]
            jf += female.sizes[jf]
    return GPTree(nodes)


def size_fair(male: GPTree, female: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeCrxSizeFair: female subtree of about the same size as the male one."""
    by_size: dict[int, list[int]] = {}
    for i in range(len(female)):
        by_size.setdefault(female.sizes[i], []).append(i)

    def pick() -> tuple[int, int]:
        m = rng.randint(0, len(male) - 1)
        avg = male.sizes[m]
        size = avg + rng.randint(-(avg - 1), avg - 1)  # calculateSize
        while not (size <= len(female) and by_size.get(size)):
            size -= 1  # no subtree of that size: try one smaller
        return m, rng.choice(by_size[size])

    return _try_points(male, female, pick)


def one_point(male: GPTree, female: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeCrxOnePoint: random point in the common region (same shape) of both trees."""
    m_common, f_common = [], []
    im = jf = 0
    while im < len(male) and jf < len(female):
        m_common.append(im)
        f_common.append(jf)
        if _arity(male, im) != _arity(female, jf):
            im += male.sizes[im]
            jf += female.sizes[jf]
        im += 1
        jf += 1

    def pick() -> tuple[int, int]:
        k = rng.randint(0, len(m_common) - 1)
        return m_common[k], f_common[k]

    return _try_points(male, female, pick)


def _arity(tree: GPTree, i: int) -> int:
    return 2 if tree.is_function(i) else 0


def _child_starts(tree: GPTree, i: int) -> list[int]:
    """Positions of the children of node i."""
    starts, k = [], i + 1
    for _ in range(_arity(tree, i)):
        starts.append(k)
        k += tree.sizes[k]
    return starts


# same order as ECF Tree::getCrossoverOp
OPERATORS: tuple[Callable[[GPTree, GPTree, random.Random], Optional[GPTree]], ...] = (
    simple, context_preserved, uniform, size_fair, one_point,
)
