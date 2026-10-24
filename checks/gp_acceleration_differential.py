"""Bounded independent correctness checks for the GP acceleration kernel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.gp import bootstrap_source, compile_priority, current_fingerprints, select_destination
from environment.noise_aware_forecast import ForecastProvider
from environment.online_crp import OnlineCRPEpisode
from environment.public_frame import make_public_frame, truncate_public_frame
from experiments.gp_h20 import fitness as gp_fitness
from experiments.gp_h20.fitness import (
    HORIZONS, _sync_provider, load_training_bank, score_episode_source_yard,
)

SCHEMA = "uwsaa_gp_acceleration_differential_v1"
SAMPLE_LAYOUTS_PER_GEOMETRY = 2
RANDOM_TREE_SEEDS = (17, 20261024, 850033, 2026102401, 6102403)
COMPOSITES = (
    "+ SH EMP",
    "- RI / AVG + EMP DIFF",
    "+ * SH CUR / DIFF RI",
    "/ + SH EMP - AVG CUR",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numba_threads() -> int | None:
    try:
        import numba
        return int(numba.get_num_threads())
    except Exception:
        return None


def _formula_cases() -> dict[str, str]:
    bootstrap_source()
    from crp.gp.expression import TERMINALS
    from crp.gp.tree import random_tree

    cases = {f"terminal_{token.lower()}": token for token in TERMINALS}
    cases.update({f"composite_{i:02d}": formula for i, formula in enumerate(COMPOSITES, 1)})
    for seed in RANDOM_TREE_SEEDS:
        cases[f"seeded_source_random_tree_{seed}"] = random_tree(random.Random(seed)).text()
    result: dict[str, str] = {}
    seen: set[str] = set()
    for name, formula in cases.items():
        canonical = " ".join(formula.split())
        if canonical not in seen:
            result[name] = canonical
            seen.add(canonical)
    return result


def _tested_file_sha256s() -> dict[str, str]:
    names = (
        "baseline/gp.py",
        "experiments/gp_h20/fitness.py",
        "experiments/gp_h20/protocol.json",
        "experiments/gp_h20/data/train_bank.json",
        "third_party/crp_online/SOURCE_MANIFEST.json",
    )
    return {name: _sha256((ROOT / name).read_bytes()) for name in names}


def _sample_episodes(episodes):
    groups = {}
    for episode in episodes:
        groups.setdefault(episode.parameter_group, []).append(episode)
    if len(groups) != 21:
        raise AssertionError(f"expected 21 geometry groups, got {len(groups)}")
    selected = []
    ids = {}
    for group, rows in sorted(groups.items()):
        if len(rows) < SAMPLE_LAYOUTS_PER_GEOMETRY:
            raise AssertionError(f"geometry {group!r} has too few training rows")
        pair = rows[:SAMPLE_LAYOUTS_PER_GEOMETRY]
        if pair[0].canonical_id == pair[1].canonical_id:
            raise AssertionError(f"geometry {group!r} did not provide distinct layouts")
        selected.extend(pair)
        ids[group] = [episode.canonical_id for episode in pair]
    return selected, ids


def _require_time(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"120-second check budget exhausted during {stage}")


def _score_numba(episode, formula: str, horizon: int, actions: list[int]) -> int:
    scorer = getattr(gp_fitness, "score_episode_numba", None)
    if scorer is None:
        raise RuntimeError("frozen acceleration API missing score_episode_numba")
    return int(scorer(episode, formula, int(horizon), actions))


def _first_public_terminals(episode, horizon: int):
    sequence = episode.retrieval_sequence
    provider = ForecastProvider(
        sequence, p=0.0, seed=0, q_prior=0.5, max_horizon=20,
        forecast_future=sequence[1:],
    )
    from environment.online_crp import OnlineCRPEnvironment
    env = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    _sync_provider(provider, episode, env.state.retrieval_count)
    legal = env.legal_destinations()
    observation = provider.public_observation(env._stacks, legal, max_tiers=episode.max_tiers)
    frame = make_public_frame(observation, env.state, legal, max_tiers=episode.max_tiers)
    prefix = truncate_public_frame(frame, horizon)
    from baseline.gp import _public_yard
    from crp.terminals import compute_terminals

    yard, source = _public_yard(prefix, horizon)
    terms = {
        destination: compute_terminals(yard, source, destination)
        for destination in prefix.legal_destinations
    }
    return prefix, terms, yard


def _edge_episode_checks(deadline: float) -> dict[str, Any]:
    from crp.yard import INT_MAX
    from baseline.gp import SOURCE_TERMINALS
    from environment.online_crp import OnlineCRPEpisode

    empty_tie = OnlineCRPEpisode(
        "accel-edge-empty-tie", 3, 2, ((1, 2), (), ()), (1, 2), "edge", {},
    )
    public, terms, _yard = _first_public_terminals(empty_tie, 20)
    if terms[1]["SH"] != 0 or terms[2]["SH"] != 0:
        raise AssertionError("empty-destination SH edge did not expose a tie")
    if terms[1]["DIFF"] != INT_MAX - terms[1]["CUR"]:
        raise AssertionError("empty-stack DIFF does not use source INT_MAX semantics")
    source_choice = select_destination(compile_priority("SH"), public, horizon=20)
    source_actions, accelerated_actions = [], []
    source_total = score_episode_source_yard(
        empty_tie, compile_priority("SH"), 20, source_actions,
    )
    accelerated_total = _score_numba(empty_tie, "SH", 20, accelerated_actions)
    if source_choice != 1 or source_actions != accelerated_actions or source_total != accelerated_total:
        raise AssertionError("equal-priority episode did not preserve the leftmost legal stack")
    leftmost_accelerated_action = accelerated_actions[0]

    average = OnlineCRPEpisode(
        "accel-edge-average", 4, 3, ((1, 4), (2, 3), (), ()),
        (1, 2, 3, 4), "edge", {},
    )
    _public, average_terms, _yard = _first_public_terminals(average, 20)
    if average_terms[1]["AVG"] != 2:
        raise AssertionError(f"source AVG truncation changed: {average_terms[1]['AVG']}")
    source_actions, accelerated_actions = [], []
    source_total = score_episode_source_yard(
        average, compile_priority("AVG"), 20, source_actions,
    )
    accelerated_total = _score_numba(average, "AVG", 20, accelerated_actions)
    if (source_actions, source_total) != (accelerated_actions, accelerated_total):
        raise AssertionError("integer AVG episode differs from source action/relocation trace")

    # At remaining=H+2, one future item is censored to Maximum=H+2.
    boundary_unknown = OnlineCRPEpisode(
        "accel-edge-prefix-boundary-4", 4, 2, ((1, 4), (2,), (3,), ()),
        (1, 2, 3, 4), "edge", {},
    )
    boundary_public, _terms, boundary_yard = _first_public_terminals(boundary_unknown, 2)
    values = [int(v) for row, occupied in zip(
        boundary_public.observation.rank_tensor, boundary_public.observation.occupied_mask,
    ) for v, used in zip(row, occupied) if used]
    if 22 not in values or boundary_yard.maximum != 4 or boundary_yard.value(22) != 4:
        raise AssertionError("H=2/remaining=H+2 unknown item must map to Maximum=4")
    source_actions, accelerated_actions = [], []
    source_total = score_episode_source_yard(
        boundary_unknown, compile_priority("- DIFF RI"), 2, source_actions,
    )
    accelerated_total = _score_numba(
        boundary_unknown, "- DIFF RI", 2, accelerated_actions,
    )
    if (source_actions, source_total) != (accelerated_actions, accelerated_total):
        raise AssertionError("near-end Maximum boundary action/relocation trace differs")

    # At remaining=H+1, every rank is known, so censoring must add no unknown.
    boundary_known = OnlineCRPEpisode(
        "accel-edge-prefix-boundary-3", 3, 2, ((1, 3), (2,), ()),
        (1, 2, 3), "edge", {},
    )
    known_public, _terms, known_yard = _first_public_terminals(boundary_known, 2)
    known_values = [int(v) for row, occupied in zip(
        known_public.observation.rank_tensor, known_public.observation.occupied_mask,
    ) for v, used in zip(row, occupied) if used]
    if 22 in known_values or set(known_values) != {1, 2, 3}:
        raise AssertionError("remaining<=H+1 must keep all ranks public")
    source_actions, accelerated_actions = [], []
    source_total = score_episode_source_yard(
        boundary_known, compile_priority("+ DIFF CUR"), 2, source_actions,
    )
    accelerated_total = _score_numba(
        boundary_known, "+ DIFF CUR", 2, accelerated_actions,
    )
    if (source_actions, source_total) != (accelerated_actions, accelerated_total):
        raise AssertionError("near-end fully-visible boundary trace differs")

    _require_time(deadline, "public-terminal edge cases")
    return {
        "empty_destination": {
            "DIFF": terms[1]["DIFF"],
            "expected_INT_MAX_minus_CUR": INT_MAX - terms[1]["CUR"],
            "source_terminal_names": list(SOURCE_TERMINALS),
        },
        "leftmost_tie": {
            "legal_destinations": list(public.legal_destinations),
            "source_action": source_choice,
            "accelerated_first_action": leftmost_accelerated_action,
        },
        "integer_avg": {"raw_mean": 2.5, "source_terminal": average_terms[1]["AVG"]},
        "near_end_maximum": {
            "remaining_H_plus_2": {"horizon": 2, "maximum": boundary_yard.maximum, "unknown_value": boundary_yard.value(22)},
            "remaining_H_plus_1": {"unknown_sentinel_present": 22 in known_values, "all_remaining_ranks_public": True},
        },
        "edge_episode_traces_match": True,
    }


def _evaluate_numba(formula: str, terminals: tuple[float, ...]) -> float:
    evaluator = getattr(gp_fitness, "evaluate_formula_numba", None)
    if evaluator is None:
        raise RuntimeError("frozen acceleration API missing evaluate_formula_numba")
    return float(evaluator(formula, terminals))
def _protected_division_checks() -> dict[str, Any]:
    formula = "/ SH CUR"
    denominators = (
        -1.0000001e-9, -1.0e-9, -0.9999999e-9, 0.0,
        0.9999999e-9, 1.0e-9, 1.0000001e-9,
    )
    comparisons = []
    source_priority = compile_priority(formula)
    for denominator in denominators:
        terminals = (7.0, 3.0, 2.0, denominator, 1.0, -1.0)
        expected = float(source_priority(*terminals))
        actual = _evaluate_numba(formula, terminals)
        if actual != expected:
            raise AssertionError(
                f"protected division differs at {denominator!r}: {actual!r} != {expected!r}"
            )
        comparisons.append({
            "denominator": denominator,
            "source_priority": expected,
            "accelerated_priority": actual,
            "equal": True,
        })

    # The source compiler evaluates a protected division's numerator lazily.
    # Postfix necessarily computes it, so verify an overflowing numerator is
    # harmless when the parent denominator is protected to 1.0.
    masked_formula = "/ * DIFF DIFF AVG"
    masked_terminals = (1.0, 2.0, 0.0, 3.0, 4.0, 1e308)
    expected = float(compile_priority(masked_formula)(*masked_terminals))
    actual = _evaluate_numba(masked_formula, masked_terminals)
    if expected != 1.0 or actual != expected:
        raise AssertionError(
            f"lazy protected-division numerator differs: source={expected!r}, accelerated={actual!r}"
        )
    return {
        "formula": formula,
        "comparisons": comparisons,
        "masked_nonfinite_numerator": {
            "formula": masked_formula,
            "source_priority": expected,
            "accelerated_priority": actual,
            "source_skips_numerator_after_protected_denominator": True,
            "postfix_evaluates_but_ignores_nonfinite_intermediate": True,
        },
        "all_equal": True,
    }


def _nonfinite_check(deadline: float) -> dict[str, Any]:
    formula = " ".join(["*"] * 39 + ["DIFF"] * 40)
    episode = OnlineCRPEpisode(
        "accel-edge-nonfinite", 3, 2, ((1, 2), (), ()), (1, 2), "edge", {},
    )
    source_error = accelerated_error = None
    try:
        score_episode_source_yard(episode, compile_priority(formula), 20)
    except Exception as exc:
        source_error = type(exc).__name__
    try:
        _score_numba(episode, formula, 20, [])
    except Exception as exc:
        accelerated_error = type(exc).__name__
    direct_formula = "* DIFF DIFF"
    direct_values = (1.0, 2.0, 3.0, 4.0, 5.0, 1e308)
    direct_source = float(compile_priority(direct_formula)(*direct_values))
    direct_accelerated_error = None
    try:
        _evaluate_numba(direct_formula, direct_values)
    except Exception as exc:
        direct_accelerated_error = type(exc).__name__
    _require_time(deadline, "non-finite priority behavior")
    if source_error != "ValueError" or accelerated_error != source_error:
        raise AssertionError(
            f"non-finite episode behavior differs: source={source_error}, accelerated={accelerated_error}"
        )
    if not math.isinf(direct_source) or direct_accelerated_error != "ValueError":
        raise AssertionError(
            f"non-finite direct priority behavior differs: source={direct_source}, "
            f"accelerated_error={direct_accelerated_error}"
        )
    return {
        "episode_formula_sha256": _sha256(formula.encode("utf-8")),
        "source_episode_behavior": source_error,
        "accelerated_episode_behavior": accelerated_error,
        "nonfinite_episode_priority_rejected": True,
        "direct_source_priority": direct_source,
        "direct_accelerated_behavior": direct_accelerated_error,
        "masked_nonfinite_intermediate_checked": True,
    }


def run(max_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    budget = min(float(max_seconds), 120.0)
    deadline = started + budget
    bootstrap_source()
    if getattr(gp_fitness, "numba", None) is None:
        raise RuntimeError("Numba is unavailable; this check requires the accelerated backend")
    fingerprint_start = current_fingerprints()
    tested_files_start = _tested_file_sha256s()
    episodes = load_training_bank()
    sample, sample_ids = _sample_episodes(episodes)
    cases = _formula_cases()
    case_records = {
        name: {"formula": formula, "sha256": _sha256(formula.encode("utf-8"))}
        for name, formula in cases.items()
    }

    trace_checks = 0
    checked_by_horizon = {str(horizon): 0 for horizon in HORIZONS}
    for name, formula in cases.items():
        source_priority = compile_priority(formula)
        for horizon in HORIZONS:
            for episode in sample:
                _require_time(deadline, f"action matrix {name}/H={horizon}")
                source_actions, accelerated_actions = [], []
                source_total = score_episode_source_yard(
                    episode, source_priority, int(horizon), source_actions,
                )
                accelerated_total = _score_numba(
                    episode, formula, int(horizon), accelerated_actions,
                )
                if source_actions != accelerated_actions or source_total != accelerated_total:
                    raise AssertionError(
                        "episode differs from source oracle: "
                        f"case={name}, formula={formula!r}, episode={episode.episode_id!r}, "
                        f"horizon={horizon}, source_total={source_total}, "
                        f"accelerated_total={accelerated_total}, "
                        f"source_actions={source_actions}, accelerated_actions={accelerated_actions}"
                    )
                trace_checks += 1
                checked_by_horizon[str(horizon)] += 1

    _require_time(deadline, "sample action matrix")
    deep_cases = [
        (name, formula) for name, formula in cases.items()
        if name.startswith("seeded_source_random_tree_")
    ][:3]
    full_bank = []
    for name, formula in deep_cases:
        _require_time(deadline, f"full-bank formula {name}")
        accelerated_total = int(gp_fitness.FormulaFitness(episodes, horizon=20)(formula))
        source_priority = compile_priority(formula)
        source_total = sum(
            score_episode_source_yard(episode, source_priority, 20)
            for episode in episodes
        )
        if accelerated_total != source_total:
            raise AssertionError(
                f"full-bank total differs for {name}: {accelerated_total} != {source_total}"
            )
        full_bank.append({
            "case": name,
            "formula_sha256": case_records[name]["sha256"],
            "formula_tokens": len(formula.split()),
            "accelerated_total": accelerated_total,
            "source_yard_total": source_total,
            "equal": True,
        })

    edge_cases = _edge_episode_checks(deadline)
    protected_division = _protected_division_checks()
    nonfinite = _nonfinite_check(deadline)
    elapsed = time.monotonic() - started
    fingerprint_end = current_fingerprints()
    tested_files_end = _tested_file_sha256s()
    if tested_files_start != tested_files_end:
        raise AssertionError("scorer, oracle, protocol, bank, or source manifest changed during the differential run")
    for key in ("source_manifest_sha256", "protocol_sha256", "train_bank_sha256"):
        if fingerprint_start[key] != fingerprint_end[key]:
            raise AssertionError(f"registered input fingerprint changed during run: {key}")
    identity = gp_fitness.fitness_engine_identity()
    return {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "correctness differential only; not a GP search or timing benchmark",
        "runtime": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "numba": _package_version("numba"),
            "llvmlite": _package_version("llvmlite"),
            "numba_threads": _numba_threads(),
            "engine_identity": identity,
            "elapsed_seconds": elapsed,
            "budget_seconds": budget,
        },
        "training_bank": {
            "rows": len(episodes),
            "geometry_groups": 21,
            "sha256": _sha256((ROOT / "experiments/gp_h20/data/train_bank.json").read_bytes()),
            "sample_layouts_per_geometry": SAMPLE_LAYOUTS_PER_GEOMETRY,
            "sample_rows": len(sample),
            "sample_geometry_layout_sha256": sample_ids,
        },
        "tested_inputs": {
            "files_sha256": tested_files_start,
            "source_manifest_sha256": fingerprint_start["source_manifest_sha256"],
            "protocol_sha256": fingerprint_start["protocol_sha256"],
            "train_bank_sha256": fingerprint_start["train_bank_sha256"],
        },
        "formula_cases": case_records,
        "action_trace_differentials": {
            "counts_by_horizon": checked_by_horizon,
            "total_episode_horizon_formula_pairs": trace_checks,
            "compared": "complete destination-index action sequence and total relocations per episode",
        },
        "full_bank_formula_totals": full_bank,
        "public_terminal_edge_cases": edge_cases,
        "protected_division_boundary": protected_division,
        "nonfinite_behavior": nonfinite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "checks" / "gp_acceleration_differential_result.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        result = run(args.max_seconds)
    except Exception as exc:
        result = {
            "schema": SCHEMA,
            "status": "fail",
            "scope": "correctness differential only; not a GP search or timing benchmark",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result.get("status"),
        "output": str(output),
        "action_trace_pairs": result.get("action_trace_differentials", {}).get(
            "total_episode_horizon_formula_pairs"
        ),
        "elapsed_seconds": result.get("runtime", {}).get("elapsed_seconds"),
        "error": result.get("error"),
    }, sort_keys=True))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
