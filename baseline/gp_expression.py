"""Accepted GP expression parser and evaluator (runtime surface only)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

FUNCTION_NAMES = ("+", "-", "*", "/")

TERMINAL_NAMES = (
    "SH", "AVG", "EMP", "CUR", "RI", "DIFF",
    "DUR", "MIN", "REM", "NEXT", "EMPTY", "WL", "NL", "DSM",
    "NUNC", "NPER", "NRAT",
)

FUNCTION_COUNT = len(FUNCTION_NAMES)

TOKEN_NAMES = FUNCTION_NAMES + TERMINAL_NAMES

NAME_TO_TOKEN = {name: index for index, name in enumerate(TOKEN_NAMES)}

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
