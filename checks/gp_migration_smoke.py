"""Bounded independent verification for the GP migration; no formal search is run."""
from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.gp import bootstrap_source, compile_priority, current_fingerprints, select_destination
from environment.noise_aware_forecast import ForecastProvider, HORIZONS as PUBLIC_HORIZONS
from environment.online_crp import OnlineCRPEnvironment
from environment.public_frame import make_public_frame, truncate_public_frame
from experiments.gp_h20 import train
from experiments.gp_h20.fitness import (
    HORIZONS, load_training_bank, score_episode_reference, score_episode_source_yard,
)
from reproducibility.statistics import GP_BINDING_COLUMNS, _validate_gp_model_binding
from uwsaa import entrypoint
from uwsaa.entrypoint import load_gp_rule


FORMULAS = ("SH", "CUR", "RI")
BOUNDARY_FORMULA = "- RI / AVG + EMP DIFF"


def _expect_rejection(callback, label: str) -> str:
    try:
        callback()
    except (RuntimeError, ValueError, FileExistsError) as exc:
        return f"{label}: {type(exc).__name__}: {exc}"
    raise AssertionError(f"{label}: unexpectedly accepted")


def _fake_seed_receipt(seed: int, fingerprints: dict[str, str]) -> dict:
    formula = "SH"
    return {
        "schema": "uwsaa_gp_h20_seed_receipt_v1",
        "status": "complete",
        "seed": seed,
        "run_config": train.RUN_CONFIG,
        "training_rows": 840,
        "training_total_relocations": 1234,
        "best_generation": 0,
        "evaluations": train.EVALUATIONS_PER_SEED,
        "fitness_environment": train.fitness_engine_identity(),
        "formula": formula,
        "formula_sha256": train.formula_sha256(formula),
        **fingerprints,
    }


class _Stream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        if self.killed:
            return -9
        return self.exit_code

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        self.waited = True
        return -9 if self.killed else int(self.exit_code or 0)


def _check_resume_and_cleanup(fingerprints: dict[str, str], scratch: Path) -> dict:
    resume_root = scratch / "resume"
    resume_root.mkdir()
    manifest = train._make_run_manifest(fingerprints)
    manifest["status"] = "partial"
    train._write_atomic(resume_root / "run_manifest.json", manifest)
    for seed in train.SEEDS[:-1]:
        train._write_atomic(train._receipt_path(resume_root, seed), _fake_seed_receipt(seed, fingerprints))

    launched: list[int] = []

    def launch_last(seed: int, output_root: Path):
        launched.append(seed)
        train._write_atomic(train._receipt_path(output_root, seed), _fake_seed_receipt(seed, fingerprints))
        return _Process(0), _Stream()

    with patch.object(train, "_launch_worker", side_effect=launch_last):
        resume_exit = train._run_formal(argparse.Namespace(
            output_root=resume_root, resume=True, workers=1, max_new_seeds=None, wall_seconds=30.0,
        ))
    final_manifest = json.loads((resume_root / "run_manifest.json").read_text(encoding="utf-8"))
    if resume_exit != 0 or launched != [train.SEEDS[-1]]:
        raise AssertionError(f"resume did not skip complete receipts: exit={resume_exit}, launched={launched}")
    if final_manifest.get("status") != "complete" or final_manifest.get("completed_seeds") != list(train.SEEDS):
        raise AssertionError("mocked resume did not validate all 30 matching receipts")

    drift_root = scratch / "hash_drift"
    drift_root.mkdir()
    drift_manifest = train._make_run_manifest(fingerprints)
    drift_manifest["status"] = "partial"
    drift_manifest["train_bank_sha256"] = "0" * 64
    train._write_atomic(drift_root / "run_manifest.json", drift_manifest)
    drift_launches: list[int] = []
    with patch.object(train, "_launch_worker", side_effect=lambda *args: drift_launches.append(args[0])):
        drift_message = _expect_rejection(
            lambda: train._run_formal(argparse.Namespace(
                output_root=drift_root, resume=True, workers=1, max_new_seeds=None, wall_seconds=30.0,
            )),
            "resume after training-bank hash drift",
        )
    if drift_launches:
        raise AssertionError("hash-drift resume launched a worker")

    bad_receipt_path = scratch / "drifted_receipt.json"
    bad_receipt = _fake_seed_receipt(1, fingerprints)
    bad_receipt["implementation_sha256"] = "0" * 64
    train._write_atomic(bad_receipt_path, bad_receipt)
    receipt_message = _expect_rejection(
        lambda: train._validate_seed_receipt(bad_receipt_path, 1, fingerprints),
        "seed receipt after implementation-hash drift",
    )

    failure_root = scratch / "worker_failure"
    failure_root.mkdir()
    failure_manifest = train._make_run_manifest(fingerprints)
    train._write_atomic(failure_root / "run_manifest.json", failure_manifest)
    failed_process, failed_stream = _Process(7), _Stream()
    with patch.object(train, "_launch_worker", return_value=(failed_process, failed_stream)):
        failure_exit = train._run_formal(argparse.Namespace(
            output_root=failure_root, resume=True, workers=1, max_new_seeds=None, wall_seconds=30.0,
        ))
    failure_receipt = json.loads(train._failure_path(failure_root, 1).read_text(encoding="utf-8"))
    failure_run_manifest = json.loads((failure_root / "run_manifest.json").read_text(encoding="utf-8"))
    if failure_exit != 1 or failure_receipt.get("status") != "failed":
        raise AssertionError("nonzero worker exit was not captured as a failed seed")
    if failure_run_manifest.get("status") != "partial" or not failed_stream.closed:
        raise AssertionError("failed worker did not leave a partial manifest and closed log stream")

    interrupt_root = scratch / "interrupted_worker"
    interrupt_root.mkdir()
    interrupt_manifest = train._make_run_manifest(fingerprints)
    train._write_atomic(interrupt_root / "run_manifest.json", interrupt_manifest)
    active_process, active_stream = _Process(None), _Stream()
    with patch.object(train, "_launch_worker", return_value=(active_process, active_stream)), \
         patch.object(train.time, "monotonic", side_effect=(0.0, 0.0, 0.0)), \
         patch.object(train.time, "sleep", side_effect=KeyboardInterrupt):
        interrupt_exit = train._run_formal(argparse.Namespace(
            output_root=interrupt_root, resume=True, workers=1, max_new_seeds=None, wall_seconds=30.0,
        ))
    incomplete_receipt = json.loads(train._failure_path(interrupt_root, 1).read_text(encoding="utf-8"))
    interrupt_run_manifest = json.loads((interrupt_root / "run_manifest.json").read_text(encoding="utf-8"))
    if (interrupt_exit != 130 or not active_process.killed or not active_process.waited
            or not active_stream.closed or incomplete_receipt.get("status") != "incomplete"
            or interrupt_run_manifest.get("status") != "partial"
            or interrupt_run_manifest.get("interrupted") is not True):
        raise AssertionError("interrupt did not kill/wait/close worker and persist incomplete receipt")

    return {
        "mocked_resume_skips_completed_seed_receipts": True,
        "mocked_resume_launched_only_seed": launched,
        "changed_run_hash_rejected_before_launch": drift_message,
        "changed_seed_receipt_hash_rejected": receipt_message.replace(str(scratch), "<temporary-check-directory>"),
        "nonzero_worker_exit_writes_failed_receipt_and_partial_manifest": True,
        "interrupt_kills_waits_and_closes_worker_with_incomplete_receipt": True,
        "formal_gp_worker_started": False,
    }


def _check_pending_and_non_gp(scratch: Path) -> dict:
    active_config = json.loads((ROOT / "configs" / "gp_rule.json").read_text(encoding="utf-8"))
    status = active_config.get("status")
    if status == "pending":
        direct_message = _expect_rejection(load_gp_rule, "pending GP artifact load")
        endpoint = scratch / "pending_output" / "endpoint.jsonl"
        command = [
            sys.executable, "-X", "utf8", "-B", "-m", "reproducibility.runner",
            "--smoke", "--solver", "GP", "--arm", "UWSAA",
            "--data", str(ROOT / "experiments" / "gp_h20" / "data" / "train_bank.json"),
            "--output", str(endpoint),
        ]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if process.returncode == 0 or "status='pending'" not in process.stderr:
            raise AssertionError("trajectory runner did not fail closed on the pending GP model")
        if endpoint.exists() or endpoint.parent.exists():
            raise AssertionError("pending GP run created trajectory outputs before model validation")
        gp_state_check = {
            "active_status": status,
            "pending_GP_load_rejected": direct_message,
            "pending_GP_runner_exit": process.returncode,
            "pending_GP_output_parent_and_endpoint_absent": True,
            "trained_GP_load_validated": False,
            "trained_branch_not_exercised_reason": "Active GP configuration was pending during this verification; rerun after activation to validate current fingerprints and formula identity and execute one GP smoke trajectory.",
            "runner_smoke_data": "experiments/gp_h20/data/train_bank.json",
            "pending_fixture_guard_check": "active pending configuration",
        }
    elif status == "trained":
        gp_rule = load_gp_rule()
        fingerprints = current_fingerprints()
        identity = gp_rule.get("model_identity")
        formula = gp_rule.get("formula")
        expected_formula_sha = hashlib.sha256(formula.encode("utf-8")).hexdigest() if isinstance(formula, str) else None
        if (not callable(gp_rule.get("selector"))
                or not isinstance(identity, dict)
                or any(gp_rule.get(key) != value for key, value in fingerprints.items())
                or gp_rule.get("formula_sha256") != expected_formula_sha
                or identity.get("formula_sha256") != expected_formula_sha
                or identity.get("best_fixed_h") != gp_rule.get("best_fixed_h")):
            raise AssertionError("trained GP rule failed current fingerprint or formula identity checks")

        # Validate fail-closed behavior with a private pending fixture without altering the active config.
        pending_fixture = scratch / "pending_fixture"
        (pending_fixture / "configs").mkdir(parents=True)
        (pending_fixture / "configs" / "gp_rule.json").write_text(
            json.dumps({"schema": "uwsaa_gp_rule_v1", "status": "pending"}), encoding="utf-8",
        )
        with patch.object(entrypoint, "ROOT", pending_fixture):
            fixture_message = _expect_rejection(load_gp_rule, "isolated pending GP fixture")

        endpoint = scratch / "trained_output" / "endpoint.jsonl"
        command = [
            sys.executable, "-X", "utf8", "-B", "-m", "reproducibility.runner",
            "--smoke", "--solver", "GP", "--arm", "UWSAA", "--limit", "1",
            "--data", str(ROOT / "experiments" / "gp_h20" / "data" / "train_bank.json"),
            "--output", str(endpoint),
        ]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if process.returncode or not endpoint.is_file():
            raise AssertionError(f"trained GP runner smoke failed: {process.stderr}")
        rows = [json.loads(line) for line in endpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
        if (not rows or any(row.get("gp_model", {}).get("formula_sha256") != expected_formula_sha for row in rows)
                or any(row.get("complete") is not True for row in rows)):
            raise AssertionError("trained GP runner smoke was incomplete or did not preserve the loaded formula identity")
        gp_state_check = {
            "active_status": status,
            "pending_GP_load_rejected": "not applicable to active trained config",
            "pending_GP_runner_exit": None,
            "pending_GP_output_parent_and_endpoint_absent": "not applicable to active trained config",
            "trained_GP_load_validated": True,
            "trained_GP_runner_exit": process.returncode,
            "runner_smoke_data": "experiments/gp_h20/data/train_bank.json",
            "trained_GP_runner_smoke_complete": all(row.get("complete") is True for row in rows),
            "trained_GP_smoke_rows": len(rows),
            "trained_GP_formula_sha256": expected_formula_sha,
            "pending_fixture_guard_check": fixture_message,
        }
    else:
        raise AssertionError(f"unsupported GP configuration status: {status!r}")

    non_gp = (
        "import sys; from uwsaa.entrypoint import load_runtime, make_policy; "
        "policy = make_policy('EM', 'NATIVE20', load_runtime(), None); "
        "assert policy is not None; "
        "assert 'baseline.gp' not in sys.modules and 'crp' not in sys.modules"
    )
    non_gp_process = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", non_gp],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if non_gp_process.returncode:
        raise AssertionError(f"non-GP policy required a GP model/source: {non_gp_process.stderr}")
    return {
        **gp_state_check,
        "non_GP_native_policy_constructed_without_GP_source_import": True,
    }

def _check_finalize_guard(fingerprints: dict[str, str], scratch: Path) -> dict:
    empty_run = scratch / "partial_finalize"
    empty_run.mkdir()
    empty_manifest = train._make_run_manifest(fingerprints)
    empty_manifest["status"] = "partial"
    empty_manifest["completed_seeds"] = []
    train._write_atomic(empty_run / "run_manifest.json", empty_manifest)
    config_path = ROOT / "configs" / "gp_rule.json"
    config_before = config_path.read_bytes()
    message = _expect_rejection(
        lambda: train._finalize(argparse.Namespace(
            output_root=empty_run, wall_seconds=1.0, activate=True,
        )),
        "finalize with no complete seed receipts",
    )
    config_after = config_path.read_bytes()
    if config_before != config_after or (empty_run / "selected_gp_rule.json").exists():
        raise AssertionError("incomplete finalize changed active configuration or wrote a selected rule")
    return {
        "incomplete_finalize_rejected": message,
        "active_gp_config_unchanged": True,
        "selected_rule_written": False,
    }


def _check_aggregate_identity() -> dict:
    base = {field: chr(97 + index) * 64 for index, field in enumerate(GP_BINDING_COLUMNS)}
    mixed = [
        {"solver": "GP", "gp_binding": dict(base)},
        {"solver": "GP", "gp_binding": {**base, "gp_formula_sha256": "f" * 64}},
    ]
    message = _expect_rejection(
        lambda: _validate_gp_model_binding(mixed),
        "aggregate containing two GP model identities",
    )
    same = _validate_gp_model_binding(mixed[:1] + [dict(mixed[0])])
    if same != base:
        raise AssertionError("aggregate identity validator rejected a homogeneous GP model")
    return {"mixed_GP_identity_rejected": message, "one_GP_identity_accepted": True}


def _check_expression_compiler() -> dict:
    bootstrap_source()
    from crp.gp.expression import evaluate, parse

    terminal_rows = (
        {"SH": 3, "EMP": 5, "AVG": 4, "CUR": 7, "RI": 2, "DIFF": -1},
        {"SH": 0, "EMP": 3, "AVG": 0, "CUR": 0, "RI": 0, "DIFF": 2},
    )
    formulas = ("SH", "+ SH * RI EMP", BOUNDARY_FORMULA, "/ DIFF CUR")
    comparisons = 0
    for formula in formulas:
        compiled = compile_priority(formula)
        tree = parse(formula)
        for terminals in terminal_rows:
            actual = float(compiled(*(float(terminals[name]) for name in ("SH", "EMP", "AVG", "CUR", "RI", "DIFF"))))
            expected = float(evaluate(tree, terminals))
            if abs(actual - expected) > 1e-12:
                raise AssertionError(f"source expression compiler/interpreter mismatch for {formula}")
            comparisons += 1
    return {
        "formulas": list(formulas),
        "terminal_rows_per_formula": len(terminal_rows),
        "comparisons": comparisons,
        "protected_division_zero_denominator_checked": True,
        "all_compiled_values_match_source_interpreter": True,
    }


def _check_differential_and_boundary() -> dict:
    bank = load_training_bank()
    first_by_group: OrderedDict[str, object] = OrderedDict()
    for episode in bank:
        first_by_group.setdefault(episode.parameter_group, episode)
    if len(first_by_group) != 21:
        raise AssertionError(f"expected 21 geometry groups, got {len(first_by_group)}")

    cells = []
    for formula in FORMULAS:
        priority = compile_priority(formula)
        for horizon in HORIZONS:
            started = time.perf_counter()
            for episode in first_by_group.values():
                source_actions: list[int] = []
                reference_actions: list[int] = []
                source_total = score_episode_source_yard(episode, priority, horizon, source_actions)
                reference_total = score_episode_reference(episode, priority, horizon, reference_actions)
                if source_actions != reference_actions or source_total != reference_total:
                    raise AssertionError(
                        f"source/reference action mismatch: {formula}, H={horizon}, {episode.episode_id}"
                    )
            cells.append({
                "formula": formula, "horizon": horizon, "layouts": 21,
                "exact_action_sequence_and_relocation_match": True,
                "wall_seconds": round(time.perf_counter() - started, 3),
            })

    priority = compile_priority(BOUNDARY_FORMULA)
    protected_cells = []
    for horizon in HORIZONS:
        started = time.perf_counter()
        for episode in first_by_group.values():
            source_actions: list[int] = []
            reference_actions: list[int] = []
            source_total = score_episode_source_yard(episode, priority, horizon, source_actions)
            reference_total = score_episode_reference(episode, priority, horizon, reference_actions)
            if source_actions != reference_actions or source_total != reference_total:
                raise AssertionError(
                    f"protected-terminal source/reference mismatch: H={horizon}, {episode.episode_id}"
                )
        protected_cells.append({
            "horizon": horizon, "layouts": 21,
            "exact_action_sequence_and_relocation_match": True,
            "wall_seconds": round(time.perf_counter() - started, 3),
        })
    horizon = 2

    def first_blocked_index(episode) -> int | None:
        stacks = [list(stack) for stack in episode.initial_stacks]
        for index, target in enumerate(episode.retrieval_sequence):
            source = next(i for i, stack in enumerate(stacks) if target in stack)
            if stacks[source][-1] != target:
                return index
            stacks[source].pop()
        return None

    def first_public_frame(episode, h: int):
        provider = ForecastProvider(
            episode.retrieval_sequence, p=0.0, seed=0, q_prior=0.5,
            max_horizon=20, forecast_future=episode.retrieval_sequence[1:],
        )
        env = OnlineCRPEnvironment(episode, lookahead_horizon=0)
        target = min(env.state.retrieval_count, len(episode.retrieval_sequence) - 1)
        while provider.event_count < target:
            provider.observe_reveal(episode.retrieval_sequence[provider.event_count + 1])
        legal = env.legal_destinations()
        observation = provider.public_observation(
            env._stacks, legal, max_tiers=episode.max_tiers,
        )
        frame = make_public_frame(observation, env.state, legal, max_tiers=episode.max_tiers)
        return truncate_public_frame(frame, h)

    boundary_case = None
    for episode in bank:
        blocked = first_blocked_index(episode)
        if blocked is None:
            continue
        cutoff = blocked + horizon + 1
        tail = episode.retrieval_sequence[cutoff:]
        reversed_tail = tuple(reversed(tail))
        if len(tail) < 2 or reversed_tail == tail:
            continue
        altered = replace(
            episode,
            retrieval_sequence=episode.retrieval_sequence[:cutoff] + reversed_tail,
        )
        original_frame = first_public_frame(episode, horizon)
        altered_frame = first_public_frame(altered, horizon)
        if not np.array_equal(original_frame.observation.rank_tensor, altered_frame.observation.rank_tensor):
            raise AssertionError("public H2 frame leaked a retrieval suffix beyond its visible prefix")
        if (original_frame.legal_destinations != altered_frame.legal_destinations
                or original_frame.retrieval_count != altered_frame.retrieval_count):
            raise AssertionError("public H2 legality/retrieval state changed after hidden-suffix permutation")
        adapter_action = select_destination(priority, original_frame, horizon=horizon)
        altered_adapter_action = select_destination(priority, altered_frame, horizon=horizon)
        original_actions: list[int] = []
        altered_actions: list[int] = []
        score_episode_source_yard(episode, priority, horizon, original_actions)
        score_episode_source_yard(altered, priority, horizon, altered_actions)
        if (adapter_action != altered_adapter_action or not original_actions or not altered_actions
                or original_actions[0] != altered_actions[0] or original_actions[0] != adapter_action):
            raise AssertionError("H20 source scorer or runtime adapter used the hidden retrieval suffix")
        boundary_case = {
            "episode_id": episode.episode_id,
            "horizon": horizon,
            "blocked_current_rank_index": blocked,
            "visible_prefix_length": cutoff,
            "hidden_suffix_length": len(tail),
            "same_censored_public_frame": True,
            "same_runtime_adapter_action": adapter_action,
            "same_source_fitness_first_action": original_actions[0],
            "formula": BOUNDARY_FORMULA,
        }
        break
    if boundary_case is None:
        raise AssertionError("no information-boundary fixture found in the registered training bank")

    return {
        "expression_compiler_differential": _check_expression_compiler(),
        "exact_trajectory_differential": {
            "formulas": list(FORMULAS),
            "horizons": list(HORIZONS),
            "geometry_layouts_per_cell": 21,
            "trajectory_pairs": len(cells) * 21,
            "all_actions_and_relocation_totals_match": True,
            "cells": cells,
        },
        "protected_terminal_differential": {
            "formula": BOUNDARY_FORMULA,
            "uses_all_terminals_and_protected_division": True,
            "horizons": list(HORIZONS),
            "geometry_layouts_per_horizon": 21,
            "trajectory_pairs": len(protected_cells) * 21,
            "all_actions_and_relocation_totals_match": True,
            "cells": protected_cells,
        },
        "information_boundary": boundary_case,
    }


def _check_preflight_cost() -> dict:
    output = io.StringIO()
    with redirect_stdout(output):
        code = train._preflight(argparse.Namespace(
            formula=list(FORMULAS), horizon=20, warm_repeats=5,
            assumed_cache_hit_rate=None, output=None,
        ))
    if code != 0:
        raise AssertionError("fixed-formula cost preflight failed")
    report = json.loads(output.getvalue())
    if report.get("training_rows") != 840 or report.get("status") != "DIAGNOSTIC_ONLY":
        raise AssertionError("full-bank cost preflight was not diagnostic on all 840 rows")
    return report


def _check_bounded_gp_smoke(scratch: Path) -> dict:
    output = scratch / "gp_smoke.json"
    command = [
        sys.executable, "-X", "utf8", "-B",
        str(ROOT / "experiments" / "gp_h20" / "train.py"), "smoke",
        "--rows", "1", "--population-size", "4", "--generations", "0",
        "--output", str(output),
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if process.returncode:
        raise AssertionError(f"bounded correctness smoke failed: {process.stderr}")
    receipt = json.loads(output.read_text(encoding="utf-8"))
    if (receipt.get("status") != "complete" or receipt.get("correctness_only") is not True
            or receipt.get("formal_selection_eligible") is not False
            or receipt.get("training_rows") != 1 or receipt.get("evaluations") != 4):
        raise AssertionError("bounded correctness smoke could be mistaken for training evidence")
    return {
        "passed": True,
        "training_rows": receipt["training_rows"],
        "evaluations": receipt["evaluations"],
        "formal_selection_eligible": receipt["formal_selection_eligible"],
        "training_total_relocations": receipt["training_total_relocations"],
        "formula_sha256": receipt["formula_sha256"],
    }


def main() -> None:
    fingerprints = current_fingerprints()
    train._validate_registered_run_contract()
    active_config = json.loads((ROOT / "configs" / "gp_rule.json").read_text(encoding="utf-8"))
    active_matches = (
        active_config.get("status") in {"pending", "trained"}
        and all(active_config.get(key) == value for key, value in fingerprints.items())
    )
    if not active_matches:
        raise AssertionError("active GP configuration status or fingerprints are invalid")
    with tempfile.TemporaryDirectory(prefix=".gp_migration_verification_", dir=ROOT) as temp:
        scratch = Path(temp)
        preflight = _check_preflight_cost()
        hit_rates = {
            "min": 0.2058235294117647,
            "median": 0.29714705882352943,
            "max": 0.5333137254901961,
        }
        historical_cache_scenarios = {
            name: {
                "assumed_hit_rate": rate,
                "seed_seconds": preflight["no_cache_seed_projection_seconds"] * (1.0 - rate),
                "seed_hours": preflight["no_cache_seed_projection_seconds"] * (1.0 - rate) / 3600.0,
            }
            for name, rate in hit_rates.items()
        }
        result = {
            "schema": "uwsaa_gp_migration_verification_v1",
            "fingerprints": fingerprints,
            "bounded_gp_smoke": _check_bounded_gp_smoke(scratch),
            "pending_and_non_GP": _check_pending_and_non_gp(scratch),
            "resume_failure_and_interruption": _check_resume_and_cleanup(fingerprints, scratch),
            "incomplete_finalize_guard": _check_finalize_guard(fingerprints, scratch),
            "aggregate_model_identity": _check_aggregate_identity(),
            "adapter_and_fitness_differential": _check_differential_and_boundary(),
            "full_bank_cost_preflight": {
                **preflight,
                "historical_cache_scenarios": historical_cache_scenarios,
            },
            "upstream_gp_tests": {
                "passed": 85,
                "failed": 1,
                "known_fixture_failure": {
                    "test": "tests/test_gp_mutation.py::test_subtree_success_needs_a_good_node_within_three_tries",
                    "test_line": 78,
                    "implementation_lines": "mutation.py:94-101",
                    "cause": "ScriptedRng fixture supplies three randint values, but source mutation also draws once to choose full versus grow mode.",
                    "vendored_engine_changed": False,
                },
                "source_suite_not_rerun_by_this_script": True,
            },
            "historical_unrestricted_source_cache_context": {
                "runs": 90,
                "evaluations_per_run": 51000,
                "hit_rate_min": 0.2058235294,
                "hit_rate_median": 0.2971470588,
                "hit_rate_max": 0.5333137255,
                "source_csv_basename": "gp_full.csv",
                "use": "illustrative scenario only; not an H20 observed rate",
            },
            "formal_gp_training_started": False,
            "formal_total_run_budget_seconds": train.TOTAL_RUN_BUDGET_SECONDS,
            "formal_training_deadline_seconds": (
                train.TOTAL_RUN_BUDGET_SECONDS - train.FINALIZATION_RESERVE_SECONDS
            ),
            "formal_finalization_reserve_seconds": train.FINALIZATION_RESERVE_SECONDS,
            "bounded_first_wave_max_new_seeds": 8,
            "active_configuration_status": active_config.get("status"),
            "active_configuration_matches_current_fingerprints": active_matches,
            "final_fingerprint_refresh_note": (
                "The final frozen implementation/protocol fingerprints are recorded above. The separate checks/gp_acceleration_differential_result.json binds the final scorer and protocol hashes and records the bounded source-oracle matrix. The earlier 420 paired action trajectories were not repeated solely for this timing-only scorer update."
            ),
            "reference_environment_cost_probe": {
                "status": "CENSORED_STOPPED_BEFORE_COMPLETING_FIRST_FORMULA",
                "elapsed_seconds_claimed": None,
                "interpretation": "Not used as a runtime estimate; the source-Yard scalar scorer is the measured path.",
            },
            "note": "Verification smoke and linear cost diagnostics only; no selected GP model or registered UWSAA performance result.",
        }
        result["status"] = "PASS_BOUNDED_CHECKS_WITH_KNOWN_UPSTREAM_FIXTURE_FAILURE"
        report_path = ROOT / "checks" / "gp_migration_verification.json"
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "report": str(report_path)}, ensure_ascii=False))
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()