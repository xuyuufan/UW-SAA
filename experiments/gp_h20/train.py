"""Seed-resumable H20 GP training, preflight, smoke, and training-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.gp import current_fingerprints
from experiments.gp_h20.fitness import (
    FormulaFitness, fitness_engine_identity, formula_sha256, load_training_bank,
)

POPULATION_SIZE = 1000
GENERATIONS = 50
EVALUATIONS_PER_SEED = POPULATION_SIZE * (GENERATIONS + 1)
SEEDS = tuple(range(1, 31))
TOTAL_RUN_BUDGET_SECONDS = 24 * 60 * 60
FINALIZATION_RESERVE_SECONDS = 60 * 60
HORIZONS = (0, 2, 5, 10, 20)
RUN_CONFIG = {
    "population_size": POPULATION_SIZE,
    "generations": GENERATIONS,
    "seeds": list(SEEDS),
    "horizon": 20,
    "fitness": "sum of restricted clean H20 relocation counts over all 840 training rows",
    "fitness_cache": "canonical prefix formula memoization within each seed only",
    "source_evaluations_per_seed": EVALUATIONS_PER_SEED,
    "selection": "minimum training total relocations; ties choose smaller seed",
}
DEFAULT_RUN_ROOT = ROOT / "experiments" / "gp_h20" / "runs" / "formal"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_registered_run_contract() -> dict[str, str]:
    protocol = _read_json(ROOT / "experiments" / "gp_h20" / "protocol.json")
    search = protocol.get("gp_search", {})
    if search.get("seeds") != [1, 30]:
        raise ValueError("protocol seed range differs from the fixed 1..30 GP contract")
    if search.get("population_size") != POPULATION_SIZE or search.get("generations") != GENERATIONS:
        raise ValueError("protocol population/generation count differs from the frozen GP contract")
    if search.get("source_engine_evaluations_per_seed") != EVALUATIONS_PER_SEED:
        raise ValueError("protocol evaluation budget differs from the source run_gp contract")
    if protocol.get("fitness_protocol", {}).get("policy_horizon") != 20:
        raise ValueError("protocol does not register fixed public horizon 20 for GP fitness")
    if protocol.get("fixed_h_selection", {}).get("horizons") != list(HORIZONS):
        raise ValueError("protocol fixed-h grid differs from the registered horizon set")
    fingerprints = current_fingerprints()
    bank_protocol = protocol.get("training_bank", {})
    if fingerprints["train_bank_sha256"] != bank_protocol.get("sha256"):
        raise ValueError("training bank SHA-256 differs from the frozen GP protocol")
    if bank_protocol.get("rows") != 840 or bank_protocol.get("geometry_groups") != 21 or bank_protocol.get("rows_per_geometry_group") != 40:
        raise ValueError("protocol training-bank dimensions differ from the registered 840-row bank")
    return fingerprints


def _failure_path(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:02d}.failed.json"


def _write_failure_receipt(root: Path, seed: int, status: str, reason: str,
                           fingerprints: Mapping[str, str]) -> None:
    _write_atomic(_failure_path(root, seed), {
        "schema": "uwsaa_gp_h20_incomplete_seed_v1",
        "status": status,
        "seed": int(seed),
        "reason": str(reason),
        **fingerprints,
    })


def _receipt_path(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:02d}.json"


def _validate_seed_receipt(path: Path, seed: int, fingerprints: Mapping[str, str]) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("status") != "complete" or receipt.get("seed") != seed:
        raise ValueError(f"seed receipt is incomplete or mislabeled: {path}")
    if receipt.get("run_config") != RUN_CONFIG:
        raise ValueError(f"seed receipt has a different formal GP contract: {path}")
    if receipt.get("fitness_environment") != fitness_engine_identity():
        raise ValueError(f"seed receipt fitness environment differs from this worker: {path}")
    for field, value in fingerprints.items():
        if receipt.get(field) != value:
            raise ValueError(f"seed receipt {field} drifted: {path}")
    formula = receipt.get("formula")
    if not isinstance(formula, str) or receipt.get("formula_sha256") != formula_sha256(formula):
        raise ValueError(f"seed receipt formula hash is invalid: {path}")
    if receipt.get("evaluations") != EVALUATIONS_PER_SEED:
        raise ValueError(f"seed receipt has an unexpected evaluation count: {path}")
    if int(receipt.get("training_total_relocations", -1)) < 0:
        raise ValueError(f"seed receipt has an invalid fitness total: {path}")
    return receipt


def _worker(seed: int, output_root: Path, *, row_limit: int | None = None,
            population_size: int = POPULATION_SIZE, generations: int = GENERATIONS,
            smoke: bool = False) -> dict[str, Any]:
    worker_started_unix_ns = time.time_ns()
    worker_started_perf_ns = time.perf_counter_ns()
    generation_started_perf_ns = worker_started_perf_ns
    from baseline.gp import bootstrap_source

    bootstrap_source()
    from crp.gp.algorithm import run_gp

    fingerprints = _validate_registered_run_contract()
    episodes = load_training_bank()
    if row_limit is not None:
        if row_limit <= 0 or row_limit > len(episodes):
            raise ValueError("row limit must be in the training-bank range")
        episodes = episodes[:row_limit]
    fitness = FormulaFitness(episodes, horizon=20)

    def log(generation: int, evaluations: int, best: float) -> None:
        nonlocal generation_started_perf_ns
        now_ns = time.perf_counter_ns()
        row = {
            "seed": seed, "worker_pid": os.getpid(), "generation": generation,
            "evaluations": evaluations, "best_training_total": int(best),
            "generation_wall_seconds": (now_ns - generation_started_perf_ns) / 1_000_000_000,
            "elapsed_wall_seconds": (now_ns - worker_started_perf_ns) / 1_000_000_000,
            "fitness_cache": fitness.statistics(),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        generation_started_perf_ns = now_ns

    result = run_gp(
        fitness,
        random.Random(seed),
        population_size=population_size,
        generations=generations,
        log=log,
    )
    formula = " ".join(result.best.text().split())
    expected_evaluations = population_size * (generations + 1)
    if result.evaluations != expected_evaluations:
        raise RuntimeError(
            f"source GP reported {result.evaluations} evaluations, expected {expected_evaluations}"
        )
    worker_finished_unix_ns = time.time_ns()
    elapsed_wall_seconds = (time.perf_counter_ns() - worker_started_perf_ns) / 1_000_000_000
    if smoke:
        receipt = {
            "schema": "uwsaa_gp_h20_smoke_v1",
            "status": "complete",
            "correctness_only": True,
            "formal_selection_eligible": False,
            "seed": int(seed),
            "worker_pid": os.getpid(),
            "worker_started_unix_ns": worker_started_unix_ns,
            "worker_finished_unix_ns": worker_finished_unix_ns,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "fitness_environment": fitness_engine_identity(),
            "population_size": population_size,
            "generations": generations,
            "evaluations": result.evaluations,
            "training_rows": len(episodes),
            "training_total_relocations": int(result.best_fitness),
            "formula": formula,
            "formula_sha256": formula_sha256(formula),
            "history": [list(row) for row in result.history],
            "fitness_statistics": fitness.statistics(),
            **fingerprints,
        }
        _write_atomic(output_root, receipt)
        return receipt

    if seed not in SEEDS or population_size != POPULATION_SIZE or generations != GENERATIONS or len(episodes) != 840:
        raise ValueError("formal worker does not match the frozen 30-seed H20 contract")
    receipt = {
        "schema": "uwsaa_gp_h20_seed_receipt_v1",
        "status": "complete",
        "seed": int(seed),
        "worker_pid": os.getpid(),
        "worker_started_unix_ns": worker_started_unix_ns,
        "worker_finished_unix_ns": worker_finished_unix_ns,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "fitness_environment": fitness_engine_identity(),
        "run_config": RUN_CONFIG,
        "training_rows": len(episodes),
        "training_total_relocations": int(result.best_fitness),
        "best_generation": int(result.best_generation),
        "evaluations": int(result.evaluations),
        "formula": formula,
        "formula_sha256": formula_sha256(formula),
        "history": [list(row) for row in result.history],
        "fitness_statistics": fitness.statistics(),
        **fingerprints,
    }
    _write_atomic(_receipt_path(output_root, seed), receipt)
    return receipt


def _make_run_manifest(fingerprints: Mapping[str, str]) -> dict[str, Any]:
    started_ns = time.time_ns()
    total_deadline_ns = started_ns + TOTAL_RUN_BUDGET_SECONDS * 1_000_000_000
    training_deadline_ns = total_deadline_ns - FINALIZATION_RESERVE_SECONDS * 1_000_000_000
    return {
        "schema": "uwsaa_gp_h20_run_manifest_v1",
        "status": "running",
        "run_config": RUN_CONFIG,
        "fitness_environment": fitness_engine_identity(),
        "total_budget_seconds": TOTAL_RUN_BUDGET_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "run_started_unix_ns": started_ns,
        "training_deadline_unix_ns": training_deadline_ns,
        "total_deadline_unix_ns": total_deadline_ns,
        **fingerprints,
    }


def _ensure_run_root(run_root: Path, resume: bool, fingerprints: Mapping[str, str]) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires an existing run_manifest.json")
        manifest = _read_json(manifest_path)
        expected = _make_run_manifest(fingerprints)
        for dynamic in ("status", "run_started_unix_ns", "training_deadline_unix_ns", "total_deadline_unix_ns"):
            expected.pop(dynamic)
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"resume rejected because run manifest {key} changed")
        if manifest.get("status") not in ("running", "partial", "complete"):
            raise ValueError("resume rejected because run manifest status is invalid")
        if not isinstance(manifest.get("run_started_unix_ns"), int) or not isinstance(manifest.get("training_deadline_unix_ns"), int) or not isinstance(manifest.get("total_deadline_unix_ns"), int):
            raise ValueError("resume rejected because persistent total-budget deadlines are missing")
        if manifest["training_deadline_unix_ns"] > manifest["total_deadline_unix_ns"]:
            raise ValueError("resume rejected because the persisted deadlines are inconsistent")
        return manifest
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run directory is non-empty; use --resume: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = _make_run_manifest(fingerprints)
    _write_atomic(manifest_path, manifest)
    return manifest

def _hold_system_awake():
    """Prevent system sleep for this runner thread, without keeping the display on."""
    if os.name != "nt":
        return None
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_state = kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint32]
    set_state.restype = ctypes.c_uint32
    previous_state = int(set_state(0x80000001))  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if previous_state == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, set_state, previous_state


def _restore_system_awake(state) -> None:
    if state is None:
        return
    _kernel32, set_state, previous_state = state
    if int(set_state(previous_state)) == 0:
        print(json.dumps({
            "warning": "SetThreadExecutionState restore failed",
            "winerror": int(__import__("ctypes").get_last_error()),
        }), file=sys.stderr, flush=True)


def _launch_worker(seed: int, run_root: Path) -> tuple[subprocess.Popen, Any]:
    log_path = run_root / f"seed_{seed:02d}.log"
    log_stream = log_path.open("ab")
    command = [
        sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
        "worker", "--seed", str(seed), "--output-root", str(run_root),
    ]
    try:
        worker_env = os.environ.copy()
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            worker_env[variable] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=worker_env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        log_stream.close()
        raise
    return process, log_stream


def _warm_numba_kernel() -> float:
    """Compile/cache the scoring kernel once before spawning seed workers."""
    if fitness_engine_identity().get("backend") != "numba_float64_v1":
        return 0.0
    episodes = load_training_bank()
    if not episodes:
        raise RuntimeError("cannot warm GP fitness without a training episode")
    started_ns = time.perf_counter_ns()
    warmup = FormulaFitness(episodes[:1], horizon=20)
    score = float(warmup("SH"))
    if not math.isfinite(score):
        raise RuntimeError("Numba GP fitness warm-up returned a non-finite score")
    return (time.perf_counter_ns() - started_ns) / 1_000_000_000

def _run_formal(args: argparse.Namespace) -> int:
    fingerprints = _validate_registered_run_contract()
    run_root: Path = args.output_root.resolve()
    run_manifest = _ensure_run_root(run_root, args.resume, fingerprints)
    completed = set()
    for seed in SEEDS:
        path = _receipt_path(run_root, seed)
        if path.exists():
            _validate_seed_receipt(path, seed, fingerprints)
            completed.add(seed)
    all_pending = [seed for seed in SEEDS if seed not in completed]
    if args.max_new_seeds is not None and args.max_new_seeds <= 0:
        raise ValueError("--max-new-seeds must be positive")
    pending = all_pending[:args.max_new_seeds] if args.max_new_seeds is not None else list(all_pending)
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    if workers < 1 or workers > 8:
        raise ValueError("formal GP runner supports one through eight CPU workers")
    if args.wall_seconds <= 0:
        raise ValueError("wall-clock limit must be positive")
    now_unix_ns = time.time_ns()
    training_remaining_seconds = max(
        0.0, (run_manifest["training_deadline_unix_ns"] - now_unix_ns) / 1_000_000_000
    )
    batch_wall_seconds = min(float(args.wall_seconds), training_remaining_seconds)
    deadline = time.monotonic() + batch_wall_seconds
    warmup_seconds = _warm_numba_kernel() if pending and batch_wall_seconds > 0 else 0.0
    awake_state = _hold_system_awake() if pending and batch_wall_seconds > 0 else None
    running: dict[int, tuple[subprocess.Popen, Any]] = {}
    failed: list[int] = []
    timed_out = False
    interrupted = False
    abort_reason: str | None = None
    print(json.dumps({
        "status": "STARTED", "completed_seeds": sorted(completed),
        "pending_seeds": pending, "all_remaining_seeds": all_pending,
        "max_new_seeds": args.max_new_seeds, "workers": workers,
        "requested_wall_seconds": args.wall_seconds,
        "remaining_training_budget_seconds": training_remaining_seconds,
        "effective_batch_wall_seconds": batch_wall_seconds,
        "numba_warmup_seconds": warmup_seconds,
        "fitness_environment": fitness_engine_identity(),
        "system_sleep_prevented": awake_state is not None,
        "output_root": str(run_root), "resume": args.resume,
    }, indent=2), flush=True)
    try:
        while pending or running:
            while pending and len(running) < workers and time.monotonic() < deadline:
                seed = pending[0]
                failure_path = _failure_path(run_root, seed)
                if failure_path.exists():
                    failure_path.unlink()
                try:
                    process, stream = _launch_worker(seed, run_root)
                except BaseException as exc:
                    failed.append(seed)
                    _write_failure_receipt(
                        run_root, seed, "failed",
                        f"worker launch failed: {type(exc).__name__}: {exc}", fingerprints,
                    )
                    raise
                pending.pop(0)
                running[seed] = (process, stream)
            finished = []
            for seed, (process, _stream) in running.items():
                code = process.poll()
                if code is not None:
                    finished.append((seed, code))
            for seed, code in finished:
                process, stream = running.pop(seed)
                stream.close()
                if code == 0:
                    try:
                        _validate_seed_receipt(_receipt_path(run_root, seed), seed, fingerprints)
                        completed.add(seed)
                        print(f"seed {seed:02d} complete", flush=True)
                    except Exception as exc:
                        failed.append(seed)
                        abort_reason = f"seed {seed} receipt rejected: {exc}"
                        _write_failure_receipt(run_root, seed, "failed", abort_reason, fingerprints)
                        print(abort_reason, flush=True)
                else:
                    failed.append(seed)
                    abort_reason = f"seed {seed} worker exited with code {code}"
                    _write_failure_receipt(run_root, seed, "failed", abort_reason, fingerprints)
                    print(abort_reason, flush=True)
            if abort_reason:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            if not finished:
                time.sleep(0.25)
    except KeyboardInterrupt:
        interrupted = True
        abort_reason = "keyboard interrupt; active seed workers were stopped"
        print(abort_reason, flush=True)
    except BaseException as exc:
        abort_reason = f"runner error: {type(exc).__name__}: {exc}"
        print(abort_reason, flush=True)
    finally:
        try:
            stopped_seeds = []
            for seed, (process, stream) in list(running.items()):
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                finally:
                    stream.close()
                stopped_seeds.append(seed)
                _write_failure_receipt(
                    run_root, seed, "incomplete",
                    abort_reason or "batch wall-clock limit reached", fingerprints,
                )
                print(f"seed {seed:02d} worker stopped during batch cleanup", flush=True)
            for seed in SEEDS:
                path = _receipt_path(run_root, seed)
                if path.exists():
                    try:
                        _validate_seed_receipt(path, seed, fingerprints)
                        completed.add(seed)
                        failure_path = _failure_path(run_root, seed)
                        if failure_path.exists():
                            failure_path.unlink()
                    except Exception as exc:
                        failed.append(seed)
                        _write_failure_receipt(run_root, seed, "failed", str(exc), fingerprints)
            manifest_path = run_root / "run_manifest.json"
            manifest = _read_json(manifest_path)
            manifest["status"] = "complete" if len(completed) == len(SEEDS) else "partial"
            manifest["completed_seeds"] = sorted(completed)
            manifest["failed_seeds"] = sorted(set(failed))
            manifest["timed_out"] = bool(timed_out)
            manifest["interrupted"] = bool(interrupted)
            manifest["abort_reason"] = abort_reason
            _write_atomic(manifest_path, manifest)
        finally:
            _restore_system_awake(awake_state)
    remaining_training_seconds = max(
        0, int((manifest["training_deadline_unix_ns"] - time.time_ns()) / 1_000_000_000)
    )
    print(json.dumps({
        "status": "COMPLETE" if len(completed) == len(SEEDS) else "PARTIAL",
        "completed_seeds": sorted(completed),
        "missing_seeds": [seed for seed in SEEDS if seed not in completed],
        "failed_seeds": sorted(set(failed)),
        "timed_out": bool(timed_out),
        "interrupted": bool(interrupted),
        "abort_reason": abort_reason,
        "remaining_training_budget_seconds": remaining_training_seconds,
        "resume_command": f"python -X utf8 -B experiments/gp_h20/train.py run --resume --wall-seconds {remaining_training_seconds} --output-root \"{run_root}\"",
    }, indent=2), flush=True)
    if len(completed) == len(SEEDS):
        return 0
    if interrupted:
        return 130
    if failed or abort_reason:
        return 1
    return 2


def _validate_score_receipt(path: Path, horizon: int, formula: str,
                            fingerprints: Mapping[str, str]) -> dict[str, Any]:
    receipt = _read_json(path)
    if receipt.get("status") != "complete" or receipt.get("horizon") != horizon:
        raise ValueError(f"fixed-horizon score receipt is incomplete or mislabeled: {path}")
    if receipt.get("formula_sha256") != formula_sha256(formula):
        raise ValueError(f"fixed-horizon score formula changed: {path}")
    for field, value in fingerprints.items():
        if receipt.get(field) != value:
            raise ValueError(f"fixed-horizon score receipt {field} drifted: {path}")
    if receipt.get("training_rows") != 840:
        raise ValueError(f"fixed-horizon score receipt row count is invalid: {path}")
    return receipt


def _score_worker(formula: str, horizon: int, output: Path) -> dict[str, Any]:
    fingerprints = _validate_registered_run_contract()
    episodes = load_training_bank()
    fitness = FormulaFitness(episodes, horizon=horizon)
    total = int(fitness(formula))
    receipt = {
        "schema": "uwsaa_gp_h20_fixed_h_score_v1",
        "status": "complete",
        "horizon": int(horizon),
        "training_rows": len(episodes),
        "training_total_relocations": total,
        "formula_sha256": formula_sha256(formula),
        "fitness_statistics": fitness.statistics(),
        **fingerprints,
    }
    _write_atomic(output, receipt)
    return receipt


def _run_score_subprocess(formula: str, horizon: int, output: Path,
                          run_root: Path, deadline: float) -> bool:
    command = [
        sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
        "score", "--formula", formula, "--horizon", str(horizon), "--output", str(output),
    ]
    log_path = run_root / f"fixed_h_{horizon}.log"
    remaining = max(0.0, deadline - time.monotonic())
    if remaining <= 0:
        return False
    with log_path.open("ab") as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return False
    return code == 0


def _finalize(args: argparse.Namespace) -> int:
    run_root: Path = args.output_root.resolve()
    fingerprints = _validate_registered_run_contract()
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("cannot finalize without the persistent run manifest")
    run_manifest = _read_json(manifest_path)
    if run_manifest.get("status") != "complete":
        raise RuntimeError("cannot finalize before all 30 registered seed receipts are complete")
    total_remaining_seconds = max(
        0.0, (run_manifest["total_deadline_unix_ns"] - time.time_ns()) / 1_000_000_000
    )
    effective_finalize_seconds = min(float(args.wall_seconds), total_remaining_seconds)
    if effective_finalize_seconds <= 0:
        raise RuntimeError("the persistent 24-hour run budget has expired")
    receipts = []
    for seed in SEEDS:
        path = _receipt_path(run_root, seed)
        if not path.is_file():
            raise RuntimeError(f"cannot select a GP rule: seed {seed} is incomplete")
        receipts.append(_validate_seed_receipt(path, seed, fingerprints))
    selected = min(receipts, key=lambda row: (int(row["training_total_relocations"]), int(row["seed"])))
    formula = selected["formula"]
    deadline = time.monotonic() + effective_finalize_seconds
    totals: dict[int, int] = {20: int(selected["training_total_relocations"])}
    for horizon in HORIZONS:
        if horizon == 20:
            continue
        score_path = run_root / f"fixed_h_{horizon}.json"
        if score_path.exists():
            row = _validate_score_receipt(score_path, horizon, formula, fingerprints)
            totals[horizon] = int(row["training_total_relocations"])
            continue
        if not _run_score_subprocess(formula, horizon, score_path, run_root, deadline):
            print(json.dumps({
                "status": "PARTIAL_SELECTION", "selected_seed": selected["seed"],
                "completed_fixed_horizons": sorted(totals), "missing_horizons": [
                    h for h in HORIZONS if h not in totals
                ], "activation_written": False,
            }, indent=2), flush=True)
            return 2
        row = _validate_score_receipt(score_path, horizon, formula, fingerprints)
        totals[horizon] = int(row["training_total_relocations"])
    best_fixed_h = min(HORIZONS, key=lambda h: (totals[h], h))
    seed_receipt_path = _receipt_path(run_root, int(selected["seed"]))
    model_identity = {
        **fingerprints,
        "fitness_environment": fitness_engine_identity(),
        "selected_seed": int(selected["seed"]),
        "training_total_relocations": int(selected["training_total_relocations"]),
        "horizon_training_totals": {str(h): int(totals[h]) for h in HORIZONS},
        "best_fixed_h": int(best_fixed_h),
        "seed_receipt_sha256": _sha256(seed_receipt_path.read_bytes()),
        "selection_basis": "training_bank_total_relocations",
        "validation_status": "training_selection_not_independent_validation",
        "formula_sha256": formula_sha256(formula),
        "evaluations_per_seed": EVALUATIONS_PER_SEED,
        "selected_seed_cache_statistics": selected["fitness_statistics"],
    }
    candidate = {
        "schema": "uwsaa_gp_rule_v1",
        "status": "trained",
        **fingerprints,
        "fitness_environment": fitness_engine_identity(),
        "formula": formula,
        "formula_sha256": formula_sha256(formula),
        "best_fixed_h": int(best_fixed_h),
        "model_identity": model_identity,
    }
    _write_atomic(run_root / "selected_gp_rule.json", candidate)
    _write_atomic(run_root / "selection_receipt.json", {
        "schema": "uwsaa_gp_h20_selection_receipt_v1",
        "status": "complete",
        "selected_seed": int(selected["seed"]),
        "selected_training_total_relocations": int(selected["training_total_relocations"]),
        "horizon_training_totals": {str(h): int(totals[h]) for h in HORIZONS},
        "best_fixed_h": int(best_fixed_h),
        "selection_basis": "training_bank_only",
        "validation_status": "training_selection_not_independent_validation",
        "model_identity": model_identity,
    })
    if args.activate:
        config_path = ROOT / "configs" / "gp_rule.json"
        _write_atomic(config_path, candidate)
    print(json.dumps({
        "status": "ACTIVATED" if args.activate else "CANDIDATE_READY",
        "selected_seed": int(selected["seed"]),
        "training_total_relocations": int(selected["training_total_relocations"]),
        "horizon_training_totals": {str(h): totals[h] for h in HORIZONS},
        "best_fixed_h": int(best_fixed_h),
        "activation_written": bool(args.activate),
        "candidate": str(run_root / "selected_gp_rule.json"),
    }, indent=2), flush=True)
    return 0


def _preflight(args: argparse.Namespace) -> int:
    fingerprints = _validate_registered_run_contract()
    episodes = load_training_bank()
    formulas = args.formula or ["SH", "CUR", "RI"]
    warm_repeats = max(1, int(args.warm_repeats))
    fitness_rows = []
    projection_rates = []
    for index, formula in enumerate(formulas, start=1):
        fitness = FormulaFitness(episodes, horizon=args.horizon)
        started_ns = time.perf_counter_ns()
        total = int(fitness(formula))
        first_pass_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        warm_samples = []
        for _ in range(warm_repeats):
            fitness.cache.clear()
            warm_started_ns = time.perf_counter_ns()
            warm_total = int(fitness(formula))
            warm_samples.append((time.perf_counter_ns() - warm_started_ns) / 1_000_000_000)
            if warm_total != total:
                raise RuntimeError("repeated no-cache fitness score changed within one preflight")
        ordered = sorted(warm_samples)
        median_seconds = float(ordered[len(ordered) // 2])
        p90_index = max(0, math.ceil(0.90 * len(ordered)) - 1)
        p90_seconds = float(ordered[p90_index])
        projection_rates.append(p90_seconds)
        print(
            f"preflight formula {index}/{len(formulas)} complete: "
            f"{formula!r} first_pass_seconds={first_pass_seconds:.6f} "
            f"warm_median_seconds={median_seconds:.6f} warm_p90_seconds={p90_seconds:.6f}",
            file=sys.stderr,
            flush=True,
        )
        fitness_rows.append({
            "formula": " ".join(formula.split()),
            "formula_sha256": formula_sha256(" ".join(formula.split())),
            "total_relocations": total,
            "bank_rows": len(episodes),
            "first_pass_including_cold_specialization_seconds": first_pass_seconds,
            "warm_no_cache_median_seconds": median_seconds,
            "warm_no_cache_p90_seconds": p90_seconds,
            "warm_no_cache_samples_seconds": warm_samples,
            "warm_repeats": warm_repeats,
        })
    no_cache_projection = max(projection_rates) * EVALUATIONS_PER_SEED
    estimated = None
    if args.assumed_cache_hit_rate is not None:
        if not 0.0 <= args.assumed_cache_hit_rate < 1.0:
            raise ValueError("assumed cache hit rate must be in [0, 1)")
        estimated = no_cache_projection * (1.0 - args.assumed_cache_hit_rate)
    report = {
        "schema": "uwsaa_gp_h20_preflight_v1",
        "status": "DIAGNOSTIC_ONLY",
        "training_rows": len(episodes),
        "horizon": int(args.horizon),
        "timed_formulas": fitness_rows,
        "max_warm_no_cache_p90_seconds_per_formula": max(projection_rates),
        "no_cache_seed_projection_seconds": no_cache_projection,
        "assumed_cache_hit_rate": args.assumed_cache_hit_rate,
        "cache_scenario_seed_projection_seconds": estimated,
        "cache_scenario_basis": "user-supplied observed cache-hit rate; illustrative linear scenario only",
        "fitness_environment": fitness_engine_identity(),
        "interpretation": "warm-cache-miss samples are repeated after clearing per-seed memoization; first-pass compilation is reported separately and excluded from the sample-based linear projection. Formula complexity, trajectory behavior, and actual cache-hit rates may differ; this is not an upper/lower bound or a training performance result",
        **fingerprints,
    }
    if args.output:
        _write_atomic(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0

def _smoke(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    receipt = _worker(
        args.seed,
        output,
        row_limit=args.rows,
        population_size=args.population_size,
        generations=args.generations,
        smoke=True,
    )
    print(json.dumps({
        "status": "SMOKE_CORRECTNESS_ONLY",
        "training_rows": receipt["training_rows"],
        "evaluations": receipt["evaluations"],
        "best_training_total_relocations": receipt["training_total_relocations"],
        "output": str(output),
        "formal_selection_eligible": False,
    }, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run/resume the fixed 30-seed formal search")
    run.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--workers", type=int, help="1 through 8; default is min(8, CPU count)")
    run.add_argument("--max-new-seeds", type=int,
                     help="optional bounded first wave; resume later for remaining registered seeds")
    run.add_argument("--wall-seconds", type=float, default=TOTAL_RUN_BUDGET_SECONDS,
                     help="per-command cap, clamped to the persistent 24-hour run deadline with a 1-hour finalization reserve")

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)

    smoke = commands.add_parser("smoke", help="small correctness-only source-engine smoke")
    smoke.add_argument("--seed", type=int, default=1)
    smoke.add_argument("--rows", type=int, default=8)
    smoke.add_argument("--population-size", type=int, default=12)
    smoke.add_argument("--generations", type=int, default=2)
    smoke.add_argument("--output", type=Path,
                       default=ROOT / "experiments" / "gp_h20" / "smoke" / "smoke_seed_01.json")

    preflight = commands.add_parser("preflight", help="time fixed formulas over the training bank")
    preflight.add_argument("--formula", action="append", help="source prefix formula; repeatable")
    preflight.add_argument("--horizon", type=int, choices=HORIZONS, default=20)
    preflight.add_argument("--warm-repeats", type=int, default=5,
                           help="number of cache-miss repeat timings per formula after the first pass")
    preflight.add_argument("--assumed-cache-hit-rate", type=float,
                           help="optional scenario only; does not change the no-cache projection")
    preflight.add_argument("--output", type=Path)

    score = commands.add_parser("score", help=argparse.SUPPRESS)
    score.add_argument("--formula", required=True)
    score.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    score.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser("finalize", help="select across 30 seeds and choose best fixed H")
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    finalize.add_argument("--wall-seconds", type=float, default=FINALIZATION_RESERVE_SECONDS)
    finalize.add_argument("--activate", action="store_true",
                          help="write configs/gp_rule.json only after complete 30-seed selection")
    args = parser.parse_args()

    if args.command == "run":
        return _run_formal(args)
    if args.command == "worker":
        _worker(args.seed, args.output_root.resolve())
        return 0
    if args.command == "smoke":
        if args.rows <= 0 or args.population_size <= 0 or args.generations < 0:
            parser.error("smoke rows/population must be positive and generations nonnegative")
        return _smoke(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "score":
        _score_worker(args.formula, args.horizon, args.output.resolve())
        return 0
    if args.command == "finalize":
        if args.wall_seconds <= 0:
            parser.error("finalize wall-clock limit must be positive")
        return _finalize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
