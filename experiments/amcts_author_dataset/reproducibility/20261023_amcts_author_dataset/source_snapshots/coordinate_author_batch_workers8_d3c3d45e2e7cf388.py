"""Prepare or run the root-released, bounded AMCTS author-dataset batch.

Default behavior is preparation only. Execution requires an explicit root
release JSON bound to the exact task list and source hashes.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = TASK_DIR / "data"
INSTANCE_DIR = DATA_DIR / "verified_instances"
AUDIT_PATH = DATA_DIR / "dataset_audit.json"
ARCHIVE_PATH = DATA_DIR / "mendeley_3j52cjrm52_v3_downloadall.zip"
INDEX_PATH = REPO_ROOT / "reproducibility" / "20261023_amcts_paper_alignment_v2" / "data" / "instances_metadata_index.json"
TASK_LIST_PATH = DATA_DIR / "task_list_1680_v2.jsonl"
RESULTS_DIR = TASK_DIR / "results_v2"
CASE_LOG_DIR = RESULTS_DIR / "cases"
STDOUT_DIR = RESULTS_DIR / "stdout"
PREP_RECEIPT = TASK_DIR / "receipts" / "batch_preparation_v2.json"
MANIFEST_PATH = RESULTS_DIR / "batch_manifest.json"
COORDINATOR_EVENTS = RESULTS_DIR / "coordinator_events.jsonl"
RUNNER_PATH = TASK_DIR / "run_author_case.py"
GATE_LOG = TASK_DIR / "receipts" / "six_container_full_budget_v2.jsonl"
GATE_AUDIT = TASK_DIR / "receipts" / "six_container_gate_audit_v2.json"

EXPECTED_CASES = 1680
MAX_WORKERS = 8
WORKER_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
MAX_BATCH_WALL_SECONDS = 24 * 60 * 60
MAX_BATCH_PROCESS_CPU_SECONDS = 192 * 60 * 60


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _case_source_hashes() -> dict[str, str]:
    from run_author_case import source_hashes

    return source_hashes()


def _batch_source_hashes() -> dict[str, str]:
    return {**_case_source_hashes(), "coordinate_author_batch.py": _hash_file(Path(__file__).resolve())}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{number}")
        result.append(row)
    return result


def _task_rows() -> list[dict[str, Any]]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_BYTES_HASH_AND_SCHEMA" or audit.get("instance_count") != EXPECTED_CASES:
        raise ValueError("author dataset audit is not a passing 1680-case audit")
    source_hashes = _case_source_hashes()
    by_key = {Path(row["filename"]).stem: row for row in audit["files"]}
    if len(by_key) != EXPECTED_CASES:
        raise ValueError("dataset audit does not contain 1680 unique case keys")
    configs = sorted(
        {str(row["configuration"]) for row in audit["files"]},
        key=lambda value: tuple(int(part) for part in value.split("-")),
    )
    rows_by_config_seed = {
        (str(row["configuration"]), int(row["seed"])): row for row in audit["files"]
    }
    if len(configs) != 56 or len(rows_by_config_seed) != EXPECTED_CASES:
        raise ValueError("expected 56 configurations and 1680 unique configuration/seed pairs")

    ordered = []
    for seed in range(30):
        for config in configs:
            try:
                source = rows_by_config_seed[(config, seed)]
            except KeyError as exc:
                raise ValueError(f"missing configured case seed: {config}, {seed}") from exc
            key = Path(str(source["filename"])).stem
            ordered.append({
                "task_index": len(ordered), "case_key": key,
                "configuration": config, "seed": seed,
                "stack_count_W": int(source["stack_count"]),
                "max_tier_H": int(source["max_tier"]),
                "container_count_N": int(source["container_count"]),
                "input_sha256": str(source["sha256"]),
                "case_source_hashes": source_hashes,
                "origin": "pre_release_gate" if key == "4-3-6-0" else "not_run",
            })
    if len(ordered) != EXPECTED_CASES or len({row["case_key"] for row in ordered}) != EXPECTED_CASES:
        raise ValueError("frozen task table must have 1680 unique case keys")
    if [row["task_index"] for row in ordered] != list(range(EXPECTED_CASES)):
        raise AssertionError("task indexes are not contiguous")
    return ordered


def _verify_gate_and_reuse_case(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gate_audit = json.loads(GATE_AUDIT.read_text(encoding="utf-8"))
    if gate_audit.get("status") != "PASS_COMPLETE_TRAJECTORY_AND_REPLAY_AUDIT":
        raise ValueError("six-container full-budget replay audit has not passed")
    task = next(row for row in task_rows if row["case_key"] == "4-3-6-0")
    if _hash_file(INSTANCE_DIR / "4-3-6-0.json") != task["input_sha256"]:
        raise ValueError("six-container case input no longer matches frozen task table")
    if task["case_source_hashes"] != _case_source_hashes():
        raise ValueError("six-container gate source map is no longer compatible")
    gate_rows = _read_jsonl(GATE_LOG)
    terminal = [row for row in gate_rows if row.get("event") == "case_terminal" and row.get("case_key") == "4-3-6-0"]
    if len(terminal) != 1 or terminal[0].get("status") != "complete":
        raise ValueError("gate run must have exactly one complete terminal row")
    if terminal[0].get("input_sha256") != task["input_sha256"] or terminal[0].get("source_hashes") != task["case_source_hashes"]:
        raise ValueError("gate terminal input/source hashes do not match frozen task")
    decisions = [row for row in gate_rows if row.get("event") == "decision"]
    if any(int(row.get("playout_budget", -1)) != 2500 * task["stack_count_W"] for row in decisions):
        raise ValueError("gate decisions do not match the frozen per-decision budget")

    destination = CASE_LOG_DIR / "4-3-6-0.jsonl"
    origin = CASE_LOG_DIR / "4-3-6-0.origin.json"
    CASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    gate_bytes = GATE_LOG.read_bytes()
    if destination.exists() and destination.read_bytes() != gate_bytes:
        raise ValueError("existing formal case log differs from the reviewed pre-release gate")
    if not destination.exists():
        shutil.copy2(GATE_LOG, destination)
    origin_record = {
        "case_key": "4-3-6-0", "origin": "pre_release_gate",
        "source_log": str(GATE_LOG.resolve()),
        "case_log": str(destination.resolve()),
        "copied_log_sha256": _hash_file(destination),
        "input_sha256": task["input_sha256"],
        "source_hashes": task["case_source_hashes"],
        "new_scored_run": False,
    }
    if origin.exists():
        current = json.loads(origin.read_text(encoding="utf-8"))
        if current != origin_record:
            raise ValueError("existing gate-origin sidecar differs from verified provenance")
    else:
        _write_json_atomic(origin, origin_record)
    return origin_record


def prepare() -> dict[str, Any]:
    task_rows = _task_rows()
    gate_origin = _verify_gate_and_reuse_case(task_rows)
    lines = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in task_rows
    )
    TASK_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TASK_LIST_PATH.exists() and TASK_LIST_PATH.read_bytes() != lines:
        raise ValueError("frozen task list already exists with different content")
    if not TASK_LIST_PATH.exists():
        TASK_LIST_PATH.write_bytes(lines)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    prep = {
        "schema": "amcts_author_batch_preparation_v2",
        "status": "PREPARED_NOT_RELEASED",
        "dataset_version": 3,
        "archive_sha256": audit["archive_sha256"],
        "metadata_index_sha256": audit["metadata_index_sha256"],
        "task_list_path": str(TASK_LIST_PATH.resolve()),
        "task_list_sha256": _hash_file(TASK_LIST_PATH),
        "full_case_count": len(task_rows),
        "configuration_count": 56,
        "cases_per_configuration": 30,
        "deterministic_interleave": "seed-major 0..29, numeric configuration order (W,H,N)",
        "case_log_layout": "one JSONL per case under results_v2/cases/",
        "pre_release_gate_reused": gate_origin,
        "planned_workers_maximum": MAX_WORKERS,
        "proposed_batch_wall_ceiling_seconds": MAX_BATCH_WALL_SECONDS,
        "proposed_batch_process_cpu_budget_estimate_seconds": MAX_BATCH_PROCESS_CPU_SECONDS,
        "process_cpu_budget_is_estimate": True,
        "process_cpu_metric_note": "Completed rows use episode CPU and exclude process startup/import; active rows use process CPU. This is a budget estimate, not an exact process-CPU ceiling.",
        "strict_wall_time_ceiling_seconds": MAX_BATCH_WALL_SECONDS,
        "worker_thread_environment": WORKER_THREAD_ENV,
        "per_case_timeout_seconds": None,
        "execution_started": False,
        "root_release_required": True,
        "case_source_hashes": _case_source_hashes(),
        "coordinator_source_hashes": _batch_source_hashes(),
    }
    _write_json_atomic(PREP_RECEIPT, prep)
    return prep


def _validate_release(release_path: Path, prep: dict[str, Any]) -> dict[str, Any]:
    if not release_path.is_file():
        raise ValueError("no release file exists; full author queue remains stopped")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "ROOT_RELEASED":
        raise ValueError("release status must be ROOT_RELEASED")
    if release.get("task_list_sha256") != prep["task_list_sha256"]:
        raise ValueError("release task-list hash does not match frozen 1680-case task table")
    if release.get("archive_sha256") != prep["archive_sha256"] or release.get("metadata_index_sha256") != prep["metadata_index_sha256"]:
        raise ValueError("release dataset hashes do not match audited inputs")
    if release.get("case_source_hashes") != _case_source_hashes():
        raise ValueError("release case-source hashes do not match current code")
    if release.get("coordinator_source_hashes") != _batch_source_hashes():
        raise ValueError("release coordinator-source hashes do not match current code")
    if int(release.get("max_workers", 0)) != MAX_WORKERS:
        raise ValueError(f"release must specify exactly {MAX_WORKERS} workers")
    wall = int(release.get("max_wall_seconds", 0))
    cpu = int(release.get("max_process_cpu_seconds", 0))
    if not 0 < wall <= MAX_BATCH_WALL_SECONDS or not 0 < cpu <= MAX_BATCH_PROCESS_CPU_SECONDS:
        raise ValueError("release wall ceiling and CPU estimate trigger must be positive and within the reviewed limits")
    if release.get("per_case_timeout_seconds") not in (None, 0):
        raise ValueError("per-case timeouts are not permitted by this registered protocol")
    return release


def _process_cpu_seconds(pid: int) -> float:
    if os.name != "nt":
        raise RuntimeError("global batch CPU cap accounting is implemented for Windows only")

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x0400, 0, int(pid))  # PROCESS_QUERY_INFORMATION
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess failed for PID {pid}")
    try:
        created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user),
        ):
            raise OSError(ctypes.get_last_error(), f"GetProcessTimes failed for PID {pid}")
        def ticks(value: FILETIME) -> int:
            return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
        return (ticks(kernel) + ticks(user)) / 10_000_000.0
    finally:
        kernel32.CloseHandle(handle)


@dataclass
class Worker:
    task: dict[str, Any]
    process: subprocess.Popen
    stdout_stream: Any
    started_monotonic: float


def _terminal_for_case(case_log: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    rows = _read_jsonl(case_log)
    terminals = [row for row in rows if row.get("event") == "case_terminal"]
    if len(terminals) > 1:
        raise ValueError(f"duplicate terminal case rows in {case_log}")
    if not terminals:
        return None
    terminal = terminals[0]
    if terminal.get("case_key") != task["case_key"] or terminal.get("status") != "complete":
        raise ValueError(f"unexpected terminal status/key in {case_log}")
    if terminal.get("input_sha256") != task["input_sha256"]:
        raise ValueError(f"input hash mismatch in {case_log}")
    if terminal.get("source_hashes") != task["case_source_hashes"] or terminal.get("source_hashes_at_end") != task["case_source_hashes"]:
        raise ValueError(f"source hash mismatch in {case_log}")
    return terminal


def _verify_all_inputs(task_rows: list[dict[str, Any]], prep: dict[str, Any]) -> None:
    if _hash_file(ARCHIVE_PATH) != prep["archive_sha256"] or _hash_file(INDEX_PATH) != prep["metadata_index_sha256"]:
        raise ValueError("dataset archive or metadata index changed after preparation")
    for row in task_rows:
        path = INSTANCE_DIR / f"{row['case_key']}.json"
        if _hash_file(path) != row["input_sha256"]:
            raise ValueError(f"input changed after preparation: {row['case_key']}")


def execute(release_path: Path) -> dict[str, Any]:
    prep = prepare()
    release = _validate_release(release_path, prep)
    if _batch_source_hashes() != prep["coordinator_source_hashes"]:
        raise ValueError("coordinator source changed after preparation")
    if _case_source_hashes() != prep["case_source_hashes"]:
        raise ValueError("case source changed after preparation")
    task_rows = [json.loads(line) for line in TASK_LIST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    _verify_all_inputs(task_rows, prep)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_DIR.mkdir(parents=True, exist_ok=True)
    start_wall = time.monotonic()
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    case_sources_start = _case_source_hashes()
    batch_sources_start = _batch_source_hashes()
    complete_by_key: dict[str, dict[str, Any]] = {}
    partial_by_key: set[str] = set()
    for row in task_rows:
        case_log = CASE_LOG_DIR / f"{row['case_key']}.jsonl"
        if not case_log.exists():
            continue
        terminal = _terminal_for_case(case_log, row)
        if terminal is not None:
            complete_by_key[row["case_key"]] = terminal
        elif _read_jsonl(case_log):
            partial_by_key.add(row["case_key"])

    tasks_pending = [row for row in task_rows if row["case_key"] not in complete_by_key]
    active: dict[str, Worker] = {}
    total_completed_cpu = sum(float(row["episode_process_cpu_seconds"]) for row in complete_by_key.values())
    fatal: str | None = None
    stop_reason: str | None = None
    final_input_verification: dict[str, Any] = {"status": "NOT_RUN"}

    def active_cpu_seconds() -> float:
        total = 0.0
        for worker in active.values():
            if worker.process.poll() is not None:
                continue
            try:
                total += _process_cpu_seconds(worker.process.pid)
            except OSError:
                # A worker can exit after poll() and before GetProcessTimes().
                if worker.process.poll() is None:
                    raise
        return total

    def manifest(status: str) -> dict[str, Any]:
        done = len(complete_by_key)
        active_keys = set(active)
        not_started_keys = {
            str(row["case_key"]) for row in task_rows
            if row["case_key"] not in complete_by_key
            and row["case_key"] not in active_keys
            and row["case_key"] not in partial_by_key
        }
        return {
            "schema": "amcts_author_dataset_batch_manifest_v2",
            "status": status,
            "start_utc": start_utc,
            "elapsed_wall_seconds": time.monotonic() - start_wall,
            "max_workers": MAX_WORKERS,
            "released_max_wall_seconds": int(release["max_wall_seconds"]),
            "released_process_cpu_budget_estimate_seconds": int(release["max_process_cpu_seconds"]),
            "process_cpu_budget_is_estimate": True,
            "process_cpu_metric_note": "Completed rows use episode CPU and exclude process startup/import; active rows use process CPU. This is a budget estimate, not an exact process-CPU ceiling.",
            "strict_wall_time_ceiling": True,
            "worker_thread_environment": WORKER_THREAD_ENV,
            "final_input_verification": final_input_verification,
            "per_case_timeout_seconds": None,
            "completed_count": done,
            "remaining_not_complete_count": max(0, EXPECTED_CASES - done),
            "partial_incomplete_count": len(partial_by_key),
            "remaining_unstarted_count": len(not_started_keys),
            "active_count": len(active),
            "completed_episode_cpu_seconds": total_completed_cpu,
            "current_active_process_cpu_seconds": active_cpu_seconds(),
            "case_keys_complete": sorted(complete_by_key),
            "case_keys_active": sorted(active),
            "case_keys_partial_incomplete": sorted(partial_by_key),
            "case_keys_not_started": sorted(not_started_keys),
            "fatal_error": fatal,
            "stop_reason": stop_reason,
            "case_source_hashes_at_start": case_sources_start,
            "case_source_hashes_at_end": _case_source_hashes(),
            "coordinator_source_hashes_at_start": batch_sources_start,
            "coordinator_source_hashes_at_end": _batch_source_hashes(),
            "archive_sha256": prep["archive_sha256"],
            "metadata_index_sha256": prep["metadata_index_sha256"],
            "task_list_sha256": prep["task_list_sha256"],
            "release_file_sha256": _hash_file(release_path),
        }

    def save_manifest(status: str) -> None:
        _write_json_atomic(MANIFEST_PATH, manifest(status))

    _append_jsonl(COORDINATOR_EVENTS, {
        "event": "batch_start", "start_utc": start_utc,
        "precompleted_case_count": len(complete_by_key),
        "scheduled_case_count": len(tasks_pending),
        "workers": MAX_WORKERS, "per_case_timeout_seconds": None,
        "worker_thread_environment": WORKER_THREAD_ENV,
        "release_file_sha256": _hash_file(release_path),
    })
    save_manifest("RUNNING")

    def terminate_active(reason: str) -> None:
        nonlocal total_completed_cpu, fatal
        for key, worker in list(active.items()):
            code = worker.process.poll()
            if code is None:
                worker.process.terminate()
                try:
                    code = worker.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    code = worker.process.wait(timeout=5)
            worker.stdout_stream.close()
            case_log = CASE_LOG_DIR / f"{key}.jsonl"
            if code == 0:
                try:
                    terminal = _terminal_for_case(case_log, worker.task)
                except Exception as exc:
                    terminal = None
                    fatal = fatal or f"terminal_integrity_error:{key}:{type(exc).__name__}:{exc}"
                if terminal is not None and key not in complete_by_key:
                    complete_by_key[key] = terminal
                    partial_by_key.discard(key)
                    total_completed_cpu += float(terminal["episode_process_cpu_seconds"])
                    _append_jsonl(COORDINATOR_EVENTS, {
                        "event": "worker_complete_at_stop_boundary", "case_key": key,
                        "pid": worker.process.pid,
                        "episode_wall_seconds": terminal["episode_wall_seconds"],
                        "episode_process_cpu_seconds": terminal["episode_process_cpu_seconds"],
                        "RR": terminal["RR"], "relocations_R": terminal["relocations_R"],
                    })
                    continue
            _append_jsonl(CASE_LOG_DIR / f"{key}.jsonl", {
                "event": "coordinator_interruption", "case_key": key,
                "status": "partial_not_complete", "reason": reason,
                "input_sha256": worker.task["input_sha256"],
                "source_hashes": worker.task["case_source_hashes"],
            })
            _append_jsonl(COORDINATOR_EVENTS, {
                "event": "worker_interrupted", "case_key": key,
                "pid": worker.process.pid, "reason": reason,
            })
            partial_by_key.add(key)
        active.clear()

    pending_index = 0
    try:
        while pending_index < len(tasks_pending) or active:
            for key, worker in list(active.items()):
                code = worker.process.poll()
                if code is None:
                    continue
                worker.stdout_stream.close()
                case_log = CASE_LOG_DIR / f"{key}.jsonl"
                try:
                    terminal = _terminal_for_case(case_log, worker.task)
                except Exception as exc:
                    terminal = None
                    fatal = f"terminal_integrity_error:{key}:{type(exc).__name__}:{exc}"
                if code != 0 or terminal is None:
                    fatal = fatal or f"worker_failed:{key}:exit={code}"
                    partial_by_key.add(key)
                    _append_jsonl(COORDINATOR_EVENTS, {
                        "event": "worker_failed", "case_key": key, "pid": worker.process.pid,
                        "exit_code": code, "error": fatal,
                    })
                    active.pop(key)
                    break
                complete_by_key[key] = terminal
                partial_by_key.discard(key)
                total_completed_cpu += float(terminal["episode_process_cpu_seconds"])
                _append_jsonl(COORDINATOR_EVENTS, {
                    "event": "worker_complete", "case_key": key,
                    "pid": worker.process.pid,
                    "episode_wall_seconds": terminal["episode_wall_seconds"],
                    "episode_process_cpu_seconds": terminal["episode_process_cpu_seconds"],
                    "RR": terminal["RR"], "relocations_R": terminal["relocations_R"],
                })
                active.pop(key)
                save_manifest("RUNNING")

            if fatal is not None:
                stop_reason = "fatal_worker_or_integrity_error"
                terminate_active(stop_reason)
                break

            elapsed = time.monotonic() - start_wall
            active_cpu = active_cpu_seconds()
            if elapsed >= int(release["max_wall_seconds"]):
                stop_reason = "released_global_wall_cap"
                terminate_active(stop_reason)
                break
            if total_completed_cpu + active_cpu >= int(release["max_process_cpu_seconds"]):
                stop_reason = "released_global_process_cpu_estimate_reached"
                terminate_active(stop_reason)
                break

            while pending_index < len(tasks_pending) and len(active) < MAX_WORKERS and fatal is None:
                task = tasks_pending[pending_index]
                pending_index += 1
                key = str(task["case_key"])
                case_log = CASE_LOG_DIR / f"{key}.jsonl"
                stdout_path = STDOUT_DIR / f"{key}.stdout.txt"
                stream = stdout_path.open("a", encoding="utf-8")
                command = [
                    sys.executable, str(RUNNER_PATH), "--case-key", key,
                    "--case-dir", str(INSTANCE_DIR), "--log", str(case_log),
                    "--playout-multiplier", "2500",
                ]
                worker_env = os.environ.copy()
                worker_env.update(WORKER_THREAD_ENV)
                try:
                    process = subprocess.Popen(
                        command, stdout=stream, stderr=subprocess.STDOUT,
                        env=worker_env,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                except Exception:
                    stream.close()
                    raise
                active[key] = Worker(task, process, stream, time.monotonic())
                _append_jsonl(COORDINATOR_EVENTS, {
                    "event": "worker_start", "case_key": key, "task_index": task["task_index"],
                    "pid": process.pid, "input_sha256": task["input_sha256"],
                    "source_hashes": task["case_source_hashes"],
                    "worker_thread_environment": WORKER_THREAD_ENV,
                })
            if active:
                time.sleep(1.0)

    except BaseException as exc:
        fatal = f"coordinator_exception:{type(exc).__name__}:{exc}"
        stop_reason = "coordinator_exception"
        terminate_active(stop_reason)
    try:
        _verify_all_inputs(task_rows, prep)
        final_input_verification = {
            "status": "PASS", "archive_sha256": _hash_file(ARCHIVE_PATH),
            "metadata_index_sha256": _hash_file(INDEX_PATH),
            "instance_count_rehashed": len(task_rows),
        }
    except Exception as exc:
        final_input_verification = {"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}
        fatal = fatal or f"final_input_integrity_error:{type(exc).__name__}:{exc}"
        stop_reason = stop_reason or "final_input_integrity_error"
    sources_end = _case_source_hashes()
    batch_sources_end = _batch_source_hashes()
    if sources_end != case_sources_start or batch_sources_end != batch_sources_start:
        fatal = fatal or "source_hash_drift_during_batch"
        stop_reason = stop_reason or "source_hash_drift"
    if fatal:
        status = "STOPPED_FATAL"
    elif stop_reason:
        status = "STOPPED_AT_GLOBAL_BUDGET"
    elif len(complete_by_key) == EXPECTED_CASES:
        status = "COMPLETE_1680"
    else:
        status = "INCOMPLETE_RESUMABLE"
    final = manifest(status)
    final["completed_count"] = len(complete_by_key)
    final["active_count"] = 0
    final["completed_episode_cpu_seconds"] = total_completed_cpu
    final["case_source_hashes_at_end"] = sources_end
    final["coordinator_source_hashes_at_end"] = batch_sources_end
    _write_json_atomic(MANIFEST_PATH, final)
    _append_jsonl(COORDINATOR_EVENTS, {
        "event": "batch_terminal", "status": status,
        "completed_count": len(complete_by_key),
        "remaining_count": EXPECTED_CASES - len(complete_by_key),
        "total_completed_episode_cpu_seconds": total_completed_cpu,
        "process_cpu_total_is_estimate": True,
        "fatal_error": fatal, "stop_reason": stop_reason,
    })
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true", help="prepare frozen table and gate reuse; does not execute cases")
    group.add_argument("--execute", action="store_true", help="execute only with an explicit matching ROOT_RELEASE.json")
    parser.add_argument("--release-file", type=Path)
    args = parser.parse_args()
    if args.execute:
        if args.release_file is None:
            raise SystemExit("--execute requires --release-file from root")
        result = execute(args.release_file)
    else:
        result = prepare()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
