"""Expression trees and ECF-style GP variation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ECF_REFERENCE_URL = "https://github.com/djakobovic/ECF"
ECF_REFERENCE_COMMIT = "818566063d693b3477c01e236bbfb8397c868873"


FUNCTION_NAMES = ("+", "-", "*", "/")
TERMINAL_NAMES = (
    "SH", "AVG", "EMP", "CUR", "RI", "DIFF",
    "DUR", "MIN", "REM", "NEXT", "EMPTY", "WL", "NL", "DSM",
    "NUNC", "NPER", "NRAT",
)
FUNCTION_COUNT = len(FUNCTION_NAMES)
TOKEN_NAMES = FUNCTION_NAMES + TERMINAL_NAMES
NAME_TO_TOKEN = {name: index for index, name in enumerate(TOKEN_NAMES)}

CORE = ("SH", "AVG", "EMP", "CUR", "RI", "DIFF")
EXTENDED = ("DUR", "MIN", "REM", "NEXT", "EMPTY", "WL", "NL", "DSM")
ONLINE = ("NUNC", "NPER", "NRAT")
TERMINAL_SETS: Mapping[str, tuple[str, ...]] = {
    "C": CORE,
    "C+E": CORE + EXTENDED,
    "C+O": CORE + ONLINE,
    "C+E+O": CORE + EXTENDED + ONLINE,
}

CROSSOVER_OPERATORS = (
    "subtree", "uniform", "context_preserving", "size_fair", "one_point"
)
MUTATION_OPERATORS = (
    "subtree", "hoist", "node_complement", "node_replacement", "permutation", "shrink"
)


@dataclass(frozen=True)
class Node:
    token: int
    children: tuple["Node", ...] = ()

    def __post_init__(self) -> None:
        expected = 2 if self.token < FUNCTION_COUNT else 0
        if self.token < 0 or self.token >= len(TOKEN_NAMES):
            raise ValueError(f"invalid token: {self.token}")
        if len(self.children) != expected:
            raise ValueError(f"token {self.token} requires {expected} children")

    @property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @property
    def depth(self) -> int:
        return 0 if not self.children else 1 + max(child.depth for child in self.children)

    def expression(self) -> str:
        if not self.children:
            return TOKEN_NAMES[self.token]
        left, right = self.children
        return f"({left.expression()} {TOKEN_NAMES[self.token]} {right.expression()})"

    def prefix(self) -> tuple[int, ...]:
        values = [self.token]
        for child in self.children:
            values.extend(child.prefix())
        return tuple(values)

    def postfix(self) -> np.ndarray:
        values: list[int] = []

        def visit(node: Node) -> None:
            for child in node.children:
                visit(child)
            values.append(node.token)

        visit(self)
        return np.asarray(values, dtype=np.int16)

    @classmethod
    def from_prefix(cls, values: Sequence[int]) -> "Node":
        position = 0

        def parse() -> Node:
            nonlocal position
            if position >= len(values):
                raise ValueError("truncated prefix expression")
            token = int(values[position])
            position += 1
            children = (parse(), parse()) if token < FUNCTION_COUNT else ()
            return cls(token, children)

        root = parse()
        if position != len(values):
            raise ValueError("prefix expression has trailing tokens")
        return root


def terminal_tokens(terminal_set: str) -> tuple[int, ...]:
    try:
        return tuple(NAME_TO_TOKEN[name] for name in TERMINAL_SETS[terminal_set])
    except KeyError as error:
        raise ValueError(f"unknown terminal set: {terminal_set}") from error


def random_tree(
    rng: np.random.Generator,
    terminals: Sequence[int],
    *,
    max_depth: int,
    min_depth: int = 1,
    full: bool = False,
    current_depth: int = 0,
) -> Node:
    if max_depth < 0 or min_depth < 0 or min_depth > max_depth:
        raise ValueError("invalid tree depth bounds")
    must_function = current_depth < min_depth
    at_limit = current_depth >= max_depth
    if must_function or (not at_limit and full):
        choose_function = True
    elif at_limit:
        choose_function = False
    else:
        # ECF growBuild chooses uniformly from the complete primitive set,
        # rather than first choosing between the function and terminal sets.
        choose_function = int(
            rng.integers(0, FUNCTION_COUNT + len(terminals))
        ) < FUNCTION_COUNT
    if choose_function:
        token = int(rng.integers(0, FUNCTION_COUNT))
        return Node(
            token,
            (
                random_tree(
                    rng, terminals, max_depth=max_depth, min_depth=min_depth,
                    full=full, current_depth=current_depth + 1,
                ),
                random_tree(
                    rng, terminals, max_depth=max_depth, min_depth=min_depth,
                    full=full, current_depth=current_depth + 1,
                ),
            ),
        )
    return Node(int(terminals[int(rng.integers(0, len(terminals)))]))


def _paths(root: Node) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def visit(node: Node, path: tuple[int, ...]) -> None:
        result.append(path)
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    visit(root, ())
    return result


def subtree_at(root: Node, path: Sequence[int]) -> Node:
    node = root
    for index in path:
        node = node.children[index]
    return node


def replace_subtree(root: Node, path: Sequence[int], replacement: Node) -> Node:
    if not path:
        return replacement
    index = int(path[0])
    children = list(root.children)
    children[index] = replace_subtree(children[index], path[1:], replacement)
    return Node(root.token, tuple(children))


def _valid_or_parent(candidate: Node, parent: Node, max_depth: int) -> Node:
    return candidate if candidate.depth <= max_depth else parent


def _random_path(
    rng: np.random.Generator, paths: Sequence[tuple[int, ...]]
) -> tuple[int, ...]:
    return paths[int(rng.integers(0, len(paths)))]


def _common_paths(parent_a: Node, parent_b: Node) -> list[tuple[int, ...]]:
    paths_b = set(_paths(parent_b))
    return [path for path in _paths(parent_a) if path in paths_b]


def _uniform_mix(
    parent_a: Node, parent_b: Node, rng: np.random.Generator
) -> Node:
    """ECF homologous uniform crossover for the binary primitive set."""

    choose_a = bool(rng.integers(0, 2))
    if bool(parent_a.children) != bool(parent_b.children):
        return parent_a if choose_a else parent_b
    chosen = parent_a if choose_a else parent_b
    if not chosen.children:
        return chosen
    return Node(
        chosen.token,
        tuple(
            _uniform_mix(child_a, child_b, rng)
            for child_a, child_b in zip(
                parent_a.children, parent_b.children, strict=True
            )
        ),
    )


def crossover(
    parent_a: Node,
    parent_b: Node,
    rng: np.random.Generator,
    *,
    operator: str,
    max_depth: int,
) -> Node:
    paths_a = _paths(parent_a)
    paths_b = _paths(parent_b)
    if operator == "uniform":
        return _valid_or_parent(_uniform_mix(parent_a, parent_b, rng), parent_a, max_depth)

    if operator == "subtree":
        functions_a = [path for path in paths_a if subtree_at(parent_a, path).children]
        terminals_a = [path for path in paths_a if not subtree_at(parent_a, path).children]
        functions_b = [path for path in paths_b if subtree_at(parent_b, path).children]
        terminals_b = [path for path in paths_b if not subtree_at(parent_b, path).children]
        for _ in range(5):
            pool_a = (
                functions_a
                if functions_a and float(rng.random()) <= 0.9
                else terminals_a
            )
            pool_b = (
                functions_b
                if functions_b and float(rng.random()) <= 0.9
                else terminals_b
            )
            path_a = _random_path(rng, pool_a)
            path_b = _random_path(rng, pool_b)
            child = replace_subtree(parent_a, path_a, subtree_at(parent_b, path_b))
            if child.depth <= max_depth:
                return child
        return parent_a
    elif operator == "context_preserving":
        common = _common_paths(parent_a, parent_b)
        path_a = path_b = _random_path(rng, common)
    elif operator == "size_fair":
        by_size: dict[int, list[tuple[int, ...]]] = {}
        for candidate in paths_b:
            by_size.setdefault(subtree_at(parent_b, candidate).size, []).append(candidate)
        for _ in range(5):
            path_a = _random_path(rng, paths_a)
            average = subtree_at(parent_a, path_a).size
            target_size = int(rng.integers(1, 2 * average))
            while target_size not in by_size:
                target_size -= 1
            path_b = _random_path(rng, by_size[target_size])
            child = replace_subtree(parent_a, path_a, subtree_at(parent_b, path_b))
            if child.depth <= max_depth:
                return child
        return parent_a
    elif operator == "one_point":
        common = _common_paths(parent_a, parent_b)
        for _ in range(5):
            path_a = path_b = _random_path(rng, common)
            child = replace_subtree(parent_a, path_a, subtree_at(parent_b, path_b))
            if child.depth <= max_depth:
                return child
        return parent_a
    else:
        raise ValueError(f"unknown crossover operator: {operator}")
    child = replace_subtree(parent_a, path_a, subtree_at(parent_b, path_b))
    return _valid_or_parent(child, parent_a, max_depth)


def mutate(
    parent: Node,
    rng: np.random.Generator,
    terminals: Sequence[int],
    *,
    operator: str,
    max_depth: int,
) -> Node:
    paths = _paths(parent)

    if operator == "subtree":
        path: tuple[int, ...] | None = None
        for _ in range(4):
            candidate = _random_path(rng, paths)
            if len(candidate) < max_depth:
                path = candidate
                break
        if path is None:
            return parent
        available_depth = max_depth - len(path)
        replacement = random_tree(
            rng,
            terminals,
            max_depth=available_depth,
            min_depth=0,
            full=bool(rng.integers(0, 2)),
        )
    elif operator == "hoist":
        return subtree_at(parent, _random_path(rng, paths))
    elif operator == "node_complement":
        complements = {0: 1, 1: 0, 2: 3, 3: 2}
        path = ()
        selected = parent
        for _ in range(4):
            path = _random_path(rng, paths)
            selected = subtree_at(parent, path)
            if selected.token in complements:
                break
        if selected.token not in complements:
            return parent
        replacement = Node(complements[selected.token], selected.children)
    elif operator == "node_replacement":
        path = _random_path(rng, paths)
        selected = subtree_at(parent, path)
        primitives = tuple(range(FUNCTION_COUNT)) + tuple(int(value) for value in terminals)
        replacement_token: int | None = None
        for _ in range(4):
            candidate = primitives[int(rng.integers(0, len(primitives)))]
            if candidate != selected.token and (candidate < FUNCTION_COUNT) == bool(selected.children):
                replacement_token = candidate
                break
        if replacement_token is None:
            return parent
        replacement = Node(replacement_token, selected.children)
    elif operator == "permutation":
        path = ()
        selected = parent
        for _ in range(4):
            path = _random_path(rng, paths)
            selected = subtree_at(parent, path)
            if selected.children:
                break
        if not selected.children or int(rng.integers(0, 2)) == 0:
            return parent
        replacement = Node(selected.token, (selected.children[1], selected.children[0]))
    elif operator == "shrink":
        path = ()
        selected = parent
        for _ in range(4):
            path = _random_path(rng, paths)
            selected = subtree_at(parent, path)
            if selected.children:
                break
        if not selected.children:
            return parent
        replacement = Node(int(terminals[int(rng.integers(0, len(terminals)))]))
    else:
        raise ValueError(f"unknown mutation operator: {operator}")
    child = replace_subtree(parent, path, replacement)
    return _valid_or_parent(child, parent, max_depth)


def evaluate_postfix(postfix: Iterable[int], features: Sequence[float]) -> float:
    stack: list[float] = []
    for raw_token in postfix:
        token = int(raw_token)
        if token >= FUNCTION_COUNT:
            stack.append(float(features[token - FUNCTION_COUNT]))
            continue
        right = stack.pop()
        left = stack.pop()
        if token == 0:
            value = left + right
        elif token == 1:
            value = left - right
        elif token == 2:
            value = left * right
        else:
            value = 1.0 if abs(right) < 1e-9 else left / right
        if not np.isfinite(value):
            value = 1e12
        stack.append(float(np.clip(value, -1e12, 1e12)))
    if len(stack) != 1:
        raise ValueError("invalid postfix expression")
    return stack[0]


__all__ = [
    "CROSSOVER_OPERATORS", "ECF_REFERENCE_COMMIT", "ECF_REFERENCE_URL",
    "EXTENDED", "FUNCTION_COUNT", "FUNCTION_NAMES", "MUTATION_OPERATORS",
    "NAME_TO_TOKEN", "Node", "ONLINE", "TERMINAL_NAMES", "TERMINAL_SETS",
    "TOKEN_NAMES", "crossover", "evaluate_postfix", "mutate", "random_tree",
    "replace_subtree", "subtree_at", "terminal_tokens",
]
