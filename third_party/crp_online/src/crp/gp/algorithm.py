"""Steady-state tournament GP, ported from ECF 1.4.2 (AlgSteadyStateTournament.cpp, State::run).

One generation = population-size iterations of: pick 3 at random (repeats allowed),
the worst is replaced by a crossover child of the other two, mutate, evaluate.
Generation 0 is the initial population, so generations=50 means 1,000 + 50 x 1,000
= 51,000 evaluations with a population of 1,000.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from crp.gp import crossover, mutation
from crp.gp.tree import GPTree, initial_population

Fitness = Callable[[str], float]  # formula text -> total relocations, lower is better


@dataclass
class GPResult:
    best: GPTree                 # hall of fame: best individual seen at a generation end
    best_fitness: float
    best_generation: int
    evaluations: int
    history: list[tuple[int, int, float]] = field(default_factory=list)  # (generation, evaluations, best)


def run_gp(
    fitness: Fitness,
    rng: random.Random,
    population_size: int = 1000,
    generations: int = 50,
    tournament_size: int = 3,
    mutation_prob: float = mutation.MUTATION_PROB,
    log: Optional[Callable[[int, int, float], None]] = None,
) -> GPResult:
    population = initial_population(population_size, rng)
    scores = [fitness(tree.text()) for tree in population]
    evaluations = population_size
    best: Optional[GPTree] = None
    best_fitness, best_generation = float("inf"), 0
    history: list[tuple[int, int, float]] = []

    def end_of_generation(generation: int) -> None:
        """Population::updateDemeStats -> HallOfFame::operate."""
        nonlocal best, best_fitness, best_generation
        i = _best_index(scores)
        if best is None or scores[i] < best_fitness:  # only strictly better replaces
            best, best_fitness, best_generation = population[i], scores[i], generation
        history.append((generation, evaluations, best_fitness))
        if log:
            log(generation, evaluations, best_fitness)

    end_of_generation(0)
    for generation in range(1, generations + 1):
        for _ in range(population_size):
            tournament = [rng.randrange(population_size) for _ in range(tournament_size)]
            worst = _select_worst(tournament, scores)
            tournament.remove(worst)  # removeFrom: first occurrence only
            child = crossover.mate(population[tournament[0]], population[tournament[1]], rng)
            if child is None:  # crossover gave up: the worst stays as it was
                child = population[worst]
            child = mutation.mutate(child, rng, mutation_prob)
            population[worst] = child
            scores[worst] = fitness(child.text())  # ECF re-evaluates even an unchanged child
            evaluations += 1
        end_of_generation(generation)
    return GPResult(best, best_fitness, best_generation, evaluations, history)


def _select_worst(tournament: list[int], scores: list[float]) -> int:
    """SelWorstOp: first in tournament order, replaced only by a strictly worse one."""
    worst = tournament[0]
    for i in tournament[1:]:
        if scores[worst] < scores[i]:
            worst = i
    return worst


def _best_index(scores: list[float]) -> int:
    """SelBestOp: on ties the LAST individual with the best fitness wins."""
    best = 0
    for i in range(1, len(scores)):
        if not scores[best] < scores[i]:
            best = i
    return best
