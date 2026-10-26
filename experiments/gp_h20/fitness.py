"""Restricted clean H20 fitness and environment reference oracle for GP."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.gp import compile_priority, current_fingerprints, select_destination
from baseline.native import GPNativeActor
from environment.noise_aware_forecast import ForecastProvider
from environment.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment, load_episode_manifest
from environment.public_frame import make_public_frame, truncate_public_frame

_NUMBA_IMPORT_ERROR: str | None = None
try:
    if not os.environ.get("NUMBA_CACHE_DIR"):
        os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "uwsaa_gp_h20_numba_cache")
    import numba
    from numba import njit
except Exception as exc:  # pragma: no cover - environment-dependent fallback
    numba = None
    njit = None
    _NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

Q_PRIOR = 0.5  # history is not a GP terminal; this only satisfies the shared provider contract.
HORIZONS = (0, 2, 5, 10, 20)
MAX_RELOCATIONS = 100_000

_INT_MAX = 2_147_483_647
_TERMINAL_CODE = {"SH": 4, "EMP": 5, "AVG": 6, "CUR": 7, "RI": 8, "DIFF": 9}
_OPERATOR_CODE = {"+": 0, "-": 1, "*": 2, "/": 3}


def fitness_engine_identity() -> dict[str, str]:
    """Numerical backend fields included in diagnostic and training receipts."""
    return {
        "backend": "numba_float64_v1" if numba is not None else "source_yard_python_v1",
        "numba_version": str(numba.__version__) if numba is not None else "unavailable",
        "numpy_version": str(np.__version__),
        "llvmlite_version": str(numba.llvmlite.__version__) if numba is not None else "unavailable",
    }


def _formula_program(canonical: str) -> np.ndarray:
    """Compile the vendored source parser AST into postfix instructions."""
    from baseline.gp import bootstrap_source

    bootstrap_source()
    from crp.gp.expression import parse

    tree = parse(canonical)
    instructions: list[int] = []

    def emit(node) -> None:
        if isinstance(node, str):
            try:
                instructions.append(_TERMINAL_CODE[node])
            except KeyError as exc:
                raise ValueError(f"unsupported GP terminal: {node}") from exc
            return
        operator, left, right = node
        emit(left)
        emit(right)
        instructions.append(_OPERATOR_CODE[operator])

    emit(tree)
    return np.asarray(instructions, dtype=np.int8)


def _pack_training_bank(episodes: Sequence[OnlineCRPEpisode]):
    """Pack rank-mapped initial layouts once; target-relative values stay dynamic."""
    count = len(episodes)
    max_stacks = max(len(episode.initial_stacks) for episode in episodes)
    max_tiers = max(int(episode.max_tiers) for episode in episodes)
    stacks = np.zeros((count, max_stacks, max_tiers), dtype=np.int64)
    heights = np.zeros((count, max_stacks), dtype=np.int64)
    n_stacks = np.empty(count, dtype=np.int64)
    capacities = np.empty(count, dtype=np.int64)
    n_containers = np.empty(count, dtype=np.int64)
    for index, episode in enumerate(episodes):
        rank_by_id = {item: rank for rank, item in enumerate(episode.retrieval_sequence, start=1)}
        n_stacks[index] = len(episode.initial_stacks)
        capacities[index] = int(episode.max_tiers)
        n_containers[index] = len(episode.retrieval_sequence)
        for stack_index, stack in enumerate(episode.initial_stacks):
            heights[index, stack_index] = len(stack)
            for tier, item in enumerate(stack):
                stacks[index, stack_index, tier] = rank_by_id[item]
    return stacks, heights, n_stacks, capacities, n_containers


if numba is not None:
    @njit(cache=True, fastmath=False)
    def _public_rank_value(container: int, target: int, n_containers: int, horizon: int) -> int:
        remaining = n_containers - target + 1
        visible_count = min(remaining, horizon + 1)
        relative_rank = container - target + 1
        if 1 <= relative_rank <= visible_count:
            return relative_rank
        return visible_count + 1


    @njit(cache=True, fastmath=False)
    def _evaluate_program(program, terminals, value_stack):
        stack_size = 0
        for instruction in program:
            if instruction >= 4:
                value_stack[stack_size] = terminals[instruction - 4]
                stack_size += 1
                continue
            if stack_size < 2:
                return np.nan, False
            right = value_stack[stack_size - 1]
            left = value_stack[stack_size - 2]
            stack_size -= 1
            if instruction == 0:
                value_stack[stack_size - 1] = left + right
            elif instruction == 1:
                value_stack[stack_size - 1] = left - right
            elif instruction == 2:
                value_stack[stack_size - 1] = left * right
            elif instruction == 3:
                value_stack[stack_size - 1] = 1.0 if abs(right) < 1e-9 else left / right
            else:
                return np.nan, False
        if stack_size != 1:
            return np.nan, False
        value = value_stack[0]
        return value, math.isfinite(value)


    @njit(cache=True, fastmath=False)
    def _score_one_numba(initial_stacks, initial_heights, n_stacks: int,
                         max_tiers: int, n_containers: int, horizon: int,
                         program, action_buffer):
        stacks = initial_stacks.copy()
        heights = initial_heights.copy()
        locations = np.full(n_containers + 1, -1, dtype=np.int64)
        for stack_index in range(n_stacks):
            for tier in range(heights[stack_index]):
                locations[stacks[stack_index, tier]] = stack_index
        terminals = np.empty(6, dtype=np.float64)
        value_stack = np.empty(len(program), dtype=np.float64)
        target = 1
        relocations = 0
        action_count = 0
        while target <= n_containers:
            source = locations[target]
            if source < 0 or heights[source] <= 0:
                return relocations, action_count, 1
            while stacks[source, heights[source] - 1] != target:
                source_height = heights[source]
                current_container = stacks[source, source_height - 1]
                cur_value = _public_rank_value(current_container, target, n_containers, horizon)
                best_destination = -1
                best_priority = math.inf
                for destination in range(n_stacks):
                    if destination == source or heights[destination] >= max_tiers:
                        continue
                    destination_height = heights[destination]
                    sum_values = 0
                    smallest = _INT_MAX
                    relative_inversions = 0
                    for tier in range(destination_height):
                        value = _public_rank_value(
                            stacks[destination, tier], target, n_containers, horizon
                        )
                        sum_values += value
                        if value < smallest:
                            smallest = value
                        if value < cur_value:
                            relative_inversions += 1
                    average = int(sum_values / destination_height) if destination_height else 0
                    terminals[0] = destination_height
                    terminals[1] = max_tiers - destination_height
                    terminals[2] = average
                    terminals[3] = cur_value
                    terminals[4] = relative_inversions
                    terminals[5] = smallest - cur_value
                    score, valid = _evaluate_program(program, terminals, value_stack)
                    if not valid:
                        return relocations, action_count, 2
                    # Ascending stack indices plus strict comparison keep leftmost ties.
                    if score < best_priority:
                        best_priority = score
                        best_destination = destination
                if best_destination < 0:
                    return relocations, action_count, 3
                if action_count < len(action_buffer):
                    action_buffer[action_count] = best_destination
                action_count += 1
                if action_count > MAX_RELOCATIONS:
                    return relocations, action_count, 4
                blocker = stacks[source, heights[source] - 1]
                heights[source] -= 1
                stacks[best_destination, heights[best_destination]] = blocker
                heights[best_destination] += 1
                locations[blocker] = best_destination
                relocations += 1
            heights[source] -= 1
            locations[target] = -1
            target += 1
        return relocations, action_count, 0


    @njit(cache=True, fastmath=False)
    def _score_bank_numba(stacks, heights, n_stacks, capacities, n_containers,
                          horizon: int, program):
        no_actions = np.empty(0, dtype=np.int64)
        total = 0
        for index in range(len(n_stacks)):
            relocations, _action_count, error = _score_one_numba(
                stacks[index], heights[index], n_stacks[index], capacities[index],
                n_containers[index], horizon, program, no_actions,
            )
            if error:
                return total, error
            total += relocations
        return total, 0
else:
    _public_rank_value = None
    _evaluate_program = None
    _score_one_numba = None
    _score_bank_numba = None


def evaluate_formula_numba(formula: str, terminal_values: Sequence[float]) -> float:
    """Evaluate source-prefix formula on SH,EMP,AVG,CUR,RI,DIFF float values."""
    if len(terminal_values) != 6:
        raise ValueError("exactly six GP terminal values are required")
    canonical = " ".join(str(formula).split())
    if numba is None:
        return float(compile_priority(canonical)(*(float(value) for value in terminal_values)))
    program = _formula_program(canonical)
    values = np.asarray(terminal_values, dtype=np.float64)
    value_stack = np.empty(len(program), dtype=np.float64)
    result, valid = _evaluate_program(program, values, value_stack)
    if not valid:
        raise ValueError("GP formula returned a non-finite or invalid priority")
    return float(result)


def score_episode_numba(episode: OnlineCRPEpisode, formula: str, horizon: int,
                        actions: list[int] | None = None) -> int:
    """Run one rank-mapped episode through the sequential Numba source-semantics kernel."""
    if numba is None:
        priority = compile_priority(" ".join(formula.split()))
        return score_episode_source_yard(episode, priority, horizon, actions)
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    packed = _pack_training_bank((episode,))
    program = _formula_program(" ".join(formula.split()))
    action_buffer = np.empty(MAX_RELOCATIONS, dtype=np.int64) if actions is not None else np.empty(0, dtype=np.int64)
    relocations, action_count, error = _score_one_numba(
        packed[0][0], packed[1][0], packed[2][0], packed[3][0], packed[4][0],
        int(horizon), program, action_buffer,
    )
    if error == 1:
        raise RuntimeError("accelerated GP kernel lost the next retrieval target")
    if error == 2:
        raise ValueError("accelerated GP formula returned a non-finite priority")
    if error == 3:
        raise RuntimeError("accelerated GP kernel found no legal destination")
    if error == 4:
        raise RuntimeError("accelerated GP kernel exceeded relocation safety limit")
    if error:
        raise RuntimeError(f"accelerated GP kernel failed with code {error}")
    if actions is not None:
        actions.extend(int(value) for value in action_buffer[:action_count])
    return int(relocations)



def load_training_bank(path: Path | None = None) -> tuple[OnlineCRPEpisode, ...]:
    bank_path = path or (ROOT / "experiments" / "gp_h20" / "data" / "train_bank.json")
    episodes = load_episode_manifest(bank_path)
    fingerprints_sha = current_fingerprints()
    protocol_path = ROOT / "experiments" / "gp_h20" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if fingerprints_sha["train_bank_sha256"] != protocol["training_bank"]["sha256"]:
        raise ValueError("GP bank hash differs from the registered protocol")
    expected_rows = int(protocol["training_bank"]["rows"])
    expected_groups = int(protocol["training_bank"]["geometry_groups"])
    expected_per_group = int(protocol["training_bank"]["rows_per_geometry_group"])
    if len(episodes) != expected_rows:
        raise ValueError(f"registered GP training bank must have {expected_rows} rows, got {len(episodes)}")
    group_counts = Counter(episode.parameter_group for episode in episodes)
    if len(group_counts) != expected_groups or any(count != expected_per_group for count in group_counts.values()):
        raise ValueError("GP training bank geometry groups differ from the frozen protocol")
    fingerprints = [episode.canonical_id for episode in episodes]
    if len(set(fingerprints)) != len(episodes):
        raise ValueError("GP training bank contains duplicate layouts/target sequences")
    return episodes


def _sync_provider(provider: ForecastProvider, episode: OnlineCRPEpisode, retrieval_count: int) -> None:
    sequence = episode.retrieval_sequence
    event_target = min(int(retrieval_count), len(sequence) - 1)
    while provider.event_count < event_target:
        next_index = provider.event_count + 1
        provider.observe_reveal(sequence[next_index])


def score_episode_reference(
    episode: OnlineCRPEpisode,
    priority,
    horizon: int,
    actions: list[int] | None = None,
) -> int:
    """Slow public-frame/environment oracle retained for differential checks."""
    sequence = episode.retrieval_sequence
    provider = ForecastProvider(
        sequence,
        p=0.0,
        seed=0,
        q_prior=Q_PRIOR,
        max_horizon=20,
        forecast_future=sequence[1:],
    )
    env = OnlineCRPEnvironment(episode, lookahead_horizon=0)

    def selector(frame: object, *, horizon: int) -> int:
        return select_destination(priority, frame, horizon=horizon)

    actor = GPNativeActor(
        {"truncate_public_frame": truncate_public_frame},
        {"selector": selector},
        int(horizon),
    )
    _sync_provider(provider, episode, env.state.retrieval_count)
    while not env.state.terminated:
        state = env.state
        legal = env.legal_destinations()
        observation = provider.public_observation(
            env._stacks, legal, max_tiers=episode.max_tiers
        )
        frame = make_public_frame(observation, state, legal, max_tiers=episode.max_tiers)
        destination = actor(frame)
        if actions is not None:
            actions.append(destination)
        env.step(destination)
        _sync_provider(provider, episode, env.state.retrieval_count)
    return int(env.relocation_count)


def _source_rank_instance(episode: OnlineCRPEpisode):
    """Map physical IDs to retrieval ranks so source Yard can advance next_target."""
    from crp.instance import Instance

    rank_by_id = {item: rank for rank, item in enumerate(episode.retrieval_sequence, start=1)}
    stacks = tuple(tuple(rank_by_id[item] for item in stack) for stack in episode.initial_stacks)
    return Instance(
        name=episode.episode_id,
        stacks=stacks,
        n_containers=len(episode.retrieval_sequence),
        max_height=episode.max_tiers,
    )


def score_episode_source_yard(
    episode: OnlineCRPEpisode,
    priority,
    horizon: int,
    actions: list[int] | None = None,
) -> int:
    """Thin restricted-destination loop over source Yard transitions/terminals."""
    from baseline.gp import bootstrap_source

    bootstrap_source()
    from crp.terminals import compute_terminals
    from crp.yard import Yard

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    instance = _source_rank_instance(episode)

    class RelativeRankYard(Yard):
        def __init__(self, source_instance):
            self.public_horizon = int(horizon)
            super().__init__(source_instance, alpha=0.0)

        def value(self, container: int) -> int:
            remaining = self.n_containers - self.next_target + 1
            visible_count = min(remaining, self.public_horizon + 1)
            relative_rank = int(container) - self.next_target + 1
            if 1 <= relative_rank <= visible_count:
                return relative_rank
            return visible_count + 1

    yard = RelativeRankYard(instance)
    while not yard.is_done():
        source = yard.target_stack()
        while yard.stacks[source][-1] != yard.next_target:
            best_destination = -1
            best_priority = math.inf
            for destination in range(len(yard.stacks)):
                if destination == source or yard.is_full(destination):
                    continue
                terminals = compute_terminals(yard, source, destination)
                score = float(priority(*(float(terminals[name]) for name in
                                         ("SH", "EMP", "AVG", "CUR", "RI", "DIFF"))))
                if not math.isfinite(score):
                    raise ValueError("source GP priority returned a non-finite score")
                if score < best_priority:
                    best_priority = score
                    best_destination = destination
            if best_destination < 0:
                raise RuntimeError("restricted source Yard found no legal destination")
            if actions is not None:
                actions.append(best_destination)
            yard.relocate(source, best_destination)
            if yard.relocations > MAX_RELOCATIONS:
                raise RuntimeError("restricted source Yard exceeded the environment safety limit")
        yard.retrieve()
    return int(yard.relocations)


class FormulaFitness:
    """Memoized relocation fitness with an optional sequential Numba evaluator."""

    def __init__(
        self,
        episodes: Sequence[OnlineCRPEpisode],
        *,
        horizon: int = 20,
        use_numba: bool = True,
    ) -> None:
        if horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}")
        self.episodes = tuple(episodes)
        self.horizon = int(horizon)
        self.cache: dict[str, int] = {}
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.use_numba = bool(use_numba and numba is not None)
        self._packed = _pack_training_bank(self.episodes) if self.use_numba else None

    def _score_one(self, episode: OnlineCRPEpisode, priority) -> int:
        return score_episode_source_yard(episode, priority, self.horizon)

    def __call__(self, formula: str) -> float:
        self.calls += 1
        canonical = " ".join(str(formula).split())
        cached = self.cache.get(canonical)
        if cached is not None:
            self.cache_hits += 1
            return float(cached)
        self.cache_misses += 1
        if self.use_numba:
            program = _formula_program(canonical)
            total, error = _score_bank_numba(*self._packed, self.horizon, program)
            if error == 1:
                raise RuntimeError("accelerated GP kernel lost the next retrieval target")
            if error == 2:
                raise ValueError("accelerated GP formula returned a non-finite priority")
            if error == 3:
                raise RuntimeError("accelerated GP kernel found no legal destination")
            if error == 4:
                raise RuntimeError("accelerated GP kernel exceeded relocation safety limit")
            if error:
                raise RuntimeError(f"accelerated GP kernel failed with code {error}")
        else:
            priority = compile_priority(canonical)
            total = sum(self._score_one(episode, priority) for episode in self.episodes)
        self.cache[canonical] = int(total)
        return float(total)

    def statistics(self) -> dict[str, int | float | str]:
        return {
            "formula_calls": int(self.calls),
            "formula_cache_hits": int(self.cache_hits),
            "formula_cache_misses": int(self.cache_misses),
            "formula_cache_hit_rate": (
                float(self.cache_hits / self.calls) if self.calls else 0.0
            ),
            "unique_formula_count": int(len(self.cache)),
            **fitness_engine_identity(),
            "active_fitness_backend": "numba_float64_v1" if self.use_numba else "source_yard_python_v1",
        }


class ReferenceFormulaFitness(FormulaFitness):
    """Environment/NumPy implementation kept as a correctness oracle, not trainer."""

    def __init__(self, episodes: Sequence[OnlineCRPEpisode], *, horizon: int = 20) -> None:
        super().__init__(episodes, horizon=horizon, use_numba=False)

    def _score_one(self, episode: OnlineCRPEpisode, priority) -> int:
        return score_episode_reference(episode, priority, self.horizon)

def formula_sha256(formula: str) -> str:
    return hashlib.sha256(formula.encode("utf-8")).hexdigest()
