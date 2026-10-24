"""The 7 ECF tree mutations, ported from ECF 1.4.2 (tree/TreeMut*.cpp, Mutation.cpp).

Each operator returns the mutated tree, or None when it gives up; ECF ignores that
result, so the tree simply stays as it was. With all mut.* probabilities at their
default 0, ECF picks one of the seven uniformly.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from crp.gp.expression import OPERATORS as FUNCTIONS, TERMINALS
from crp.gp.tree import MAX_DEPTH, PRIMITIVES, GPTree, full_tree, grow_tree

MUTATION_PROB = 0.3  # mutation.indprob in the authors' parameter file
TRIES = 4            # "tries < 4" in every operator
COMPLEMENT = {"+": "-", "-": "+", "*": "/", "/": "*"}  # Add.h, Sub.h, Mul.h, Div.h


def mutate(tree: GPTree, rng: random.Random, prob: float = MUTATION_PROB) -> GPTree:
    """Mutation::mutation + mutate: with probability prob, apply one random operator."""
    if rng.random() > prob:  # ECF: mutate if getRandomDouble() <= indMutProb
        return tree
    return rng.choice(OPERATORS)(tree, rng) or tree  # failure: unchanged


def _replace(tree: GPTree, i: int, new: tuple[str, ...]) -> GPTree:
    """Tree with the subtree at i replaced by new nodes."""
    return GPTree(tree.nodes[:i] + new + tree.nodes[i + tree.sizes[i]:])


def permutation(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutPermutation: shuffle a function node's children (2 args: swap with prob 1/2)."""
    for _ in range(TRIES):
        i = rng.randrange(len(tree))
        if tree.is_function(i):
            break
    else:
        return None
    if rng.randint(0, 1) == 0:  # permutation[0] swapped with itself: nothing changes
        return tree
    left = i + 1
    right = left + tree.sizes[left]
    return _replace(tree, i, (tree.nodes[i],) + tree.subtree(right) + tree.subtree(left))


def gauss(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutGauss: changes a constant; set C has none, so it always gives up."""
    for _ in range(TRIES):
        rng.randrange(len(tree))  # looks for a "D_" constant node and never finds one
    return None


def node_replace(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutNodeReplace: swap one node for another primitive with the same arity."""
    i = rng.randrange(len(tree))
    old = tree.nodes[i]
    for _ in range(TRIES):
        new = rng.choice(PRIMITIVES)
        if new != old and (new in FUNCTIONS) == (old in FUNCTIONS):
            return GPTree(tree.nodes[:i] + (new,) + tree.nodes[i + 1:])
    return None


def node_complement(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutNodeComplement: + <-> -, * <-> /; terminals have no complement."""
    for _ in range(TRIES):
        i = rng.randrange(len(tree))
        if tree.nodes[i] in COMPLEMENT:
            return GPTree(tree.nodes[:i] + (COMPLEMENT[tree.nodes[i]],) + tree.nodes[i + 1:])
    return None


def hoist(tree: GPTree, rng: random.Random) -> GPTree:
    """TreeMutHoist: a random subtree becomes the whole tree (can be a single terminal)."""
    return GPTree(tree.subtree(rng.randrange(len(tree))))


def shrink(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutShrink: replace a function node's subtree by a random terminal."""
    for _ in range(TRIES):
        i = rng.randrange(len(tree))
        if tree.sizes[i] > 1:
            return _replace(tree, i, (rng.choice(TERMINALS),))
    return None


def subtree(tree: GPTree, rng: random.Random) -> Optional[GPTree]:
    """TreeMutSubtree: replace a subtree (not at max depth) by a new random one.

    ECF checks `tries == 4` after the loop, so a node found on the 4th try still
    counts as a failure: only the first 3 tries can succeed.
    """
    for tries in range(1, TRIES + 1):
        i = rng.randrange(len(tree))
        if tree.depths[i] != MAX_DEPTH:
            break
    if tries == TRIES:
        return None
    room = MAX_DEPTH - tree.depths[i]  # initMaxDepth of the new subtree, initMinDepth 0
    new = full_tree(rng, room) if rng.randint(0, 1) == 0 else grow_tree(rng, room, min_depth=0)
    return _replace(tree, i, new.nodes)


# same order as ECF Tree::getMutationOp
OPERATORS: tuple[Callable[[GPTree, random.Random], Optional[GPTree]], ...] = (
    permutation, gauss, node_replace, node_complement, hoist, shrink, subtree,
)
