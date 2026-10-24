"""GP expressions: parse prefix formulas like "+ SH * RI EMP" into priority functions.

Mirrors cpp_reference/gp/Evaluator.cpp (ParseExpression, Execute) with the paper's
function set: + - * and protected /.
"""
from __future__ import annotations

from itertools import count
from typing import Callable, Union

from crp.schemes import PriorityFunction

TERMINALS = ("SH", "EMP", "AVG", "CUR", "RI", "DIFF")
OPERATORS = ("+", "-", "*", "/")

Tree = Union[str, tuple[str, "Tree", "Tree"]]  # terminal name, or (operator, left, right)


def parse(text: str) -> Tree:
    """Turn a prefix formula into a tree, once."""
    tokens = text.split()
    tree, end = _parse_at(tokens, 0)
    if end != len(tokens):
        raise ValueError(f"extra tokens after position {end}: {text!r}")
    return tree


def _parse_at(tokens: list[str], i: int) -> tuple[Tree, int]:
    """Parse the subtree starting at tokens[i]; return it and the index after it."""
    if i >= len(tokens):
        raise ValueError("formula ends too early")
    token = tokens[i]
    if token in TERMINALS:
        return token, i + 1
    if token in OPERATORS:
        left, i = _parse_at(tokens, i + 1)
        right, i = _parse_at(tokens, i)
        return (token, left, right), i
    raise ValueError(f"unknown token {token!r}")


def evaluate(tree: Tree, t: dict[str, int]) -> float:
    """Compute the formula for one set of terminal values."""
    if isinstance(tree, str):
        return float(t[tree])  # the C++ stores terminal values as double
    op, left, right = tree
    a, b = evaluate(left, t), evaluate(right, t)
    if op == "+":
        return a + b
    if op == "-":
        return a - b  # C++ `Execute() - Execute()` has no defined order; we use left - right
    if op == "*":
        return a * b
    return 1.0 if abs(b) < 1e-9 else a / b  # protected division


def make_priority(text: str) -> PriorityFunction:
    """Parse once, return a function the scheme can call many times."""
    tree = parse(text)
    return lambda t: evaluate(tree, t)


def compile_formula(text: str) -> Callable[..., float]:
    """Option B: turn the formula into real Python code, once. Same results as evaluate().

    "+ SH * RI EMP" becomes:  def f(SH, EMP, AVG, CUR, RI, DIFF): return (SH + (RI * EMP))
    The caller passes the six terminal values as floats, like the C++ doubles.
    """
    names = count()

    def code(tree: Tree) -> str:
        if isinstance(tree, str):
            return tree
        op, left, right = tree
        if op == "/":  # protected division; d is computed once and reused
            d = f"_d{next(names)}"
            return f"(1.0 if abs({d} := {code(right)}) < 1e-9 else {code(left)} / {d})"
        return f"({code(left)} {op} {code(right)})"

    namespace: dict = {}
    exec(f"def f({', '.join(TERMINALS)}):\n    return {code(parse(text))}", namespace)
    return namespace["f"]


def canonical(text: str) -> str:
    """One spelling per formula: parse it, then write it back with single spaces."""

    def tokens(tree: Tree) -> list[str]:
        if isinstance(tree, str):
            return [tree]
        op, left, right = tree
        return [op, *tokens(left), *tokens(right)]

    return " ".join(tokens(parse(text)))
