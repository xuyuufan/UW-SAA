"""Linux coordinator for the hash-bound AMCTS author dataset batch."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from run_author_case import source_hashes

DATA_DIR = TASK_DIR / "data"
INSTANCE_DIR = DATA_DIR / "verified_instances"
ARCHIVE = DATA_DIR / "mendeley_3j52cjrm52_v3_downloadall.zip"
INDEX = DATA_DIR / "instances_metadata_index.json"
AUDIT = DATA_DIR / "dataset_audit.json"
TASK_LIST = DATA_DIR / "task_list_1680_v2.jsonl"
CASE_DIR = TASK_DIR / "results_v2" / "cases"
RUNS_DIR = TASK_DIR / "results_v2" / "cloud_batches"
RECEIPTS = TASK_DIR / "receipts"
PREVIOUS_PREP = RECEIPTS / "batch_preparation_v2.json"
CLOUD_PREP = RECEIPTS / "cloud_preparation.json"
EXPECTED = {
    "task_list_sha256": "998afed4e3b44b5ce80ece349689ba0a6e8b579695ed429af68b9ccae73d49bc",
    "archive_sha256": "7208263dcbb20c825ab637dad18aed61fd62a5d9daeb90416811a4974ca9e885",
    "metadata_index_sha256": "037382b1285624db0f8deededc2c7317703d2b79001aa60df98d9301704f4dc7",
    "case_source_hashes": {
        "paper_zhang.py": "c084626d0b2f5c90d85f72cb7b2fac7bb7bce30d942e700466f6edc670f8ac1f",
        "zhang_author_data.py": "0121bcb4466ed0d49e3c52a2845c391546bada6a18c5ade62739e90b48460639",
        "run_author_case.py": "969df0793384f43ea45342f668bce985fbb5a38061e302837ff6741830b6fb36",
    },
}
EXPECTED_CASES = 1680
MAX_WORKERS = 4
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON row at {path}:{line_number}")
            rows.append(row)
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def tasks() -> list[dict[str, Any]]:
    raw = TASK_LIST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED["task_list_sha256"]:
        raise ValueError("frozen 1680-case task list hash mismatch")
    result = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    keys = [row["case_key"] for row in result]
    if len(result) != EXPECTED_CASES or len(set(keys)) != EXPECTED_CASES:
        raise ValueError("task list must contain 1680 unique cases")
    if {int(row["task_index"]) for row in result} != set(range(EXPECTED_CASES)):
        raise ValueError("task indexes are not the frozen contiguous range")
    return result


def verify_terminal(path: Path, task: dict[str, Any], sources: dict[str, str]) -> dict[str, Any] | None:
    rows = read_jsonl(path)
    terminals = [row for row in rows if row.get("event") == "case_terminal"]
    if len(terminals) > 1:
        raise ValueError(f"duplicate terminal rows in {path}")
    for row in rows:
        if row.get("input_sha256") not in (None, task["input_sha256"]):
            raise ValueError(f"input hash mismatch in {path}")
        if row.get("source_hashes") not in (None, task["case_source_hashes"]):
            raise ValueError(f"method source hash mismatch in {path}")
    if not terminals:
        return None
    terminal = terminals[0]
    if (terminal.get("case_key") != task["case_key"] or terminal.get("status") != "complete"
            or terminal.get("input_sha256") != task["input_sha256"]
            or terminal.get("source_hashes") != sources
            or terminal.get("source_hashes_at_end") != sources):
        raise ValueError(f"complete terminal is not bound to current inputs/sources: {path}")
    attempt_id = terminal.get("attempt_id")
    starts = [row for row in rows if row.get("event") == "case_start" and row.get("attempt_id") == attempt_id]
    decisions = [row for row in rows if row.get("event") == "decision" and row.get("attempt_id") == attempt_id]
    if len(starts) != 1 or starts[0].get("future_retrieval_sequence_passed_to_policy") is not False:
        raise ValueError(f"public information boundary failed in {path}")
    if len(decisions) != int(terminal["decision_count"]):
        raise ValueError(f"decision count mismatch in {path}")
    budget = 2500 * int(task["stack_count_W"])
    if any(int(row.get("playout_budget", -1)) != budget or int(row.get("playouts_added", -1)) != budget
           or row.get("policy_received_only_current_target") is not True for row in decisions):
        raise ValueError(f"decision budget or public-input boundary failed in {path}")
    n, r = int(terminal["container_count_N"]), int(terminal["relocations_R"])
    if n != int(task["container_count_N"]) or r < 0 or float(terminal["RR"]) != r / n:
        raise ValueError(f"invalid relocation/RR metrics in {path}")
    if int(terminal["complete_retrievals"]) != n:
        raise ValueError(f"incomplete retrieval count in {path}")
    return terminal


def verify_inputs(task_rows: list[dict[str, Any]], previous_prep: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(ARCHIVE) != EXPECTED["archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    if sha256_file(INDEX) != EXPECTED["metadata_index_sha256"]:
        raise ValueError("Mendeley metadata-index hash mismatch")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_BYTES_HASH_AND_SCHEMA" or audit.get("instance_count") != EXPECTED_CASES:
        raise ValueError("dataset audit does not certify the full 1680-case dataset")
    if audit.get("archive_sha256") != EXPECTED["archive_sha256"] or audit.get("metadata_index_sha256") != EXPECTED["metadata_index_sha256"]:
        raise ValueError("dataset audit source hashes mismatch")
    if previous_prep.get("task_list_sha256") != EXPECTED["task_list_sha256"]:
        raise ValueError("prior preparation receipt does not match the frozen task list")
    sources = source_hashes()
    if sources != EXPECTED["case_source_hashes"] or previous_prep.get("case_source_hashes") != sources:
        raise ValueError("frozen method source hashes differ from the reviewed source map")
    complete: dict[str, dict[str, Any]] = {}
    partial: list[str] = []
    for task in task_rows:
        case_path = INSTANCE_DIR / f"{task['case_key']}.json"
        if not case_path.is_file() or sha256_file(case_path) != task["input_sha256"]:
            raise ValueError(f"author input missing or modified: {task['case_key']}")
        log = CASE_DIR / f"{task['case_key']}.jsonl"
        terminal = verify_terminal(log, task, sources)
        if terminal is not None:
            complete[task["case_key"]] = terminal
        elif log.exists() and log.stat().st_size:
            partial.append(task["case_key"])
    known = {row["case_key"] for row in task_rows}
    extras = {p.stem for p in CASE_DIR.glob("*.jsonl")} - known
    if extras:
        raise ValueError(f"unexpected case logs: {sorted(extras)[:5]}")
    return {"complete": complete, "partial": sorted(partial), "sources": sources}


def resource_snapshot() -> dict[str, Any]:
    mem_available_mib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_mib = int(line.split()[1]) // 1024
                break
    return {"logical_cpu_count": os.cpu_count(), "available_memory_mib": mem_available_mib}


def prepare() -> dict[str, Any]:
    task_rows = tasks()
    previous = json.loads(PREVIOUS_PREP.read_text(encoding="utf-8"))
    state = verify_inputs(task_rows, previous)
    not_run = EXPECTED_CASES - len(state["complete"]) - len(state["partial"])
    receipt = {
        "schema": "amcts_author_dataset_cloud_preparation_v1",
        "status": "PASS_PREPARED_NOT_STARTED",
        "dataset": "Mendeley 3j52cjrm52 v3, 1680 scored JSON inputs, 56 configurations x 30 seeds",
        "task_list_sha256": EXPECTED["task_list_sha256"],
        "archive_sha256": EXPECTED["archive_sha256"],
        "metadata_index_sha256": EXPECTED["metadata_index_sha256"],
        "case_source_hashes": state["sources"],
        "cloud_coordinator_sha256": sha256_file(Path(__file__).resolve()),
        "counts": {"expected": EXPECTED_CASES, "complete": len(state["complete"]),
                   "partial": len(state["partial"]), "not_run": not_run},
        "partial_case_keys": state["partial"],
        "resource_snapshot": resource_snapshot(),
        "per_case_timeout_seconds": None,
        "playouts_per_decision": "2500 x W",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_started": False,
    }
    atomic_json(CLOUD_PREP, receipt)
    return receipt


def process_cpu_seconds(pid: int) -> float:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    fields = stat[stat.rfind(")") + 2:].split()  # starts at field 3 (state)
    ticks = int(fields[11]) + int(fields[12])  # utime + stime, fields 14 and 15
    return ticks / float(os.sysconf("SC_CLK_TCK"))


@dataclass
class Worker:
    task: dict[str, Any]
    process: subprocess.Popen
    stdout: Any
    started: float


def run(workers: int, batch_hours: float) -> dict[str, Any]:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1..{MAX_WORKERS}")
    if os.cpu_count() is not None and workers > os.cpu_count():
        raise ValueError("worker count exceeds logical CPUs visible to this process")
    if not 0 < batch_hours <= 24:
        raise ValueError("each batch is capped at 24 wall-clock hours")
    prep = prepare()
    task_rows = tasks()
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    batch_dir = RUNS_DIR / batch_id
    stdout_dir = batch_dir / "stdout"
    batch_dir.mkdir(parents=True, exist_ok=False)
    stdout_dir.mkdir()
    events_path = batch_dir / "events.jsonl"
    manifest_path = batch_dir / "manifest.json"
    max_wall = int(batch_hours * 3600)
    max_cpu_estimate = max_wall * workers
    sources_start = source_hashes()
    state = verify_inputs(task_rows, json.loads(PREVIOUS_PREP.read_text(encoding="utf-8")))
    completed = dict(state["complete"])
    partial = set(state["partial"])
    pending = [row for row in task_rows if row["case_key"] not in completed]
    active: dict[str, Worker] = {}
    batch_episode_cpu = 0.0
    started_monotonic = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stop_requested = False
    fatal: str | None = None

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}

    def active_cpu() -> float:
        total = 0.0
        for worker in active.values():
            if worker.process.poll() is not None:
                continue
            try:
                total += process_cpu_seconds(worker.process.pid)
            except (FileNotFoundError, ProcessLookupError):
                if worker.process.poll() is None:
                    raise
        return total

    def write_manifest(status: str, stop_reason: str | None = None) -> dict[str, Any]:
        elapsed = time.monotonic() - started_monotonic
        data = {
            "schema": "amcts_author_dataset_cloud_batch_v1", "batch_id": batch_id,
            "status": status, "start_utc": started_utc,
            "elapsed_wall_seconds": elapsed, "workers": workers,
            "max_wall_seconds": max_wall,
            "process_cpu_budget_estimate_seconds": max_cpu_estimate,
            "process_cpu_budget_is_estimate": True,
            "completed_count": len(completed), "completed_this_batch": len(completed) - len(state["complete"]),
            "partial_count": len(partial), "not_run_count": EXPECTED_CASES - len(completed) - len(partial),
            "active_case_keys": sorted(active), "case_source_hashes": sources_start,
            "cloud_coordinator_sha256": sha256_file(Path(__file__).resolve()),
            "task_list_sha256": EXPECTED["task_list_sha256"],
            "archive_sha256": EXPECTED["archive_sha256"],
            "metadata_index_sha256": EXPECTED["metadata_index_sha256"],
            "episode_cpu_seconds_this_batch": batch_episode_cpu,
            "active_worker_cpu_seconds": active_cpu(),
            "per_case_timeout_seconds": None, "stop_reason": stop_reason,
        }
        atomic_json(manifest_path, data)
        atomic_json(TASK_DIR / "results_v2" / "cloud_latest_manifest.json", data)
        return data

    append_jsonl(events_path, {"event": "batch_start", "batch_id": batch_id,
                               "start_utc": started_utc, "precompleted": len(completed),
                               "pending": len(pending), "workers": workers,
                               "per_case_timeout_seconds": None})
    write_manifest("RUNNING")
    pending_index = 0
    last_manifest = time.monotonic()
    stop_reason: str | None = None
    try:
        while pending_index < len(pending) or active:
            for key, worker in list(active.items()):
                code = worker.process.poll()
                if code is None:
                    continue
                worker.stdout.close()
                log = CASE_DIR / f"{key}.jsonl"
                terminal = verify_terminal(log, worker.task, sources_start)
                if code != 0 or terminal is None:
                    fatal = f"worker_failed:{key}:exit={code}"
                    append_jsonl(events_path, {"event": "worker_failed", "case_key": key,
                                               "exit_code": code, "fatal": fatal})
                    partial.add(key)
                    active.pop(key)
                    break
                completed[key] = terminal
                partial.discard(key)
                batch_episode_cpu += float(terminal["episode_process_cpu_seconds"])
                append_jsonl(events_path, {"event": "worker_complete", "case_key": key,
                                           "RR": terminal["RR"], "relocations_R": terminal["relocations_R"],
                                           "episode_wall_seconds": terminal["episode_wall_seconds"],
                                           "episode_process_cpu_seconds": terminal["episode_process_cpu_seconds"]})
                active.pop(key)
                write_manifest("RUNNING")
            if fatal:
                stop_reason = "fatal_worker_or_integrity_error"
                break
            elapsed = time.monotonic() - started_monotonic
            if stop_requested:
                stop_reason = "operator_or_platform_stop"
                break
            if elapsed >= max_wall:
                stop_reason = "24h_batch_wall_cap"
                break
            if batch_episode_cpu + active_cpu() >= max_cpu_estimate:
                stop_reason = "batch_process_cpu_estimate_cap"
                break
            while pending_index < len(pending) and len(active) < workers:
                task = pending[pending_index]
                pending_index += 1
                key = task["case_key"]
                log_path = CASE_DIR / f"{key}.jsonl"
                stdout_path = stdout_dir / f"{key}.stdout.txt"
                stdout = stdout_path.open("ab")
                env = os.environ.copy()
                env.update(THREAD_ENV)
                command = [sys.executable, str(TASK_DIR / "run_author_case.py"),
                           "--case-key", key, "--case-dir", str(INSTANCE_DIR),
                           "--log", str(log_path), "--playout-multiplier", "2500"]
                process = subprocess.Popen(command, cwd=EXPERIMENT_ROOT, stdout=stdout,
                                           stderr=subprocess.STDOUT, env=env)
                active[key] = Worker(task, process, stdout, time.monotonic())
                append_jsonl(events_path, {"event": "worker_start", "case_key": key,
                                           "pid": process.pid, "task_index": task["task_index"],
                                           "input_sha256": task["input_sha256"],
                                           "source_hashes": task["case_source_hashes"],
                                           "thread_environment": THREAD_ENV})
            if active:
                time.sleep(1)
            if time.monotonic() - last_manifest >= 30:
                write_manifest("RUNNING")
                last_manifest = time.monotonic()
    except BaseException as exc:
        fatal = f"coordinator_error:{type(exc).__name__}:{exc}"
        stop_reason = "coordinator_error"
    finally:
        for key, worker in list(active.items()):
            try:
                if worker.process.poll() is None:
                    worker.process.terminate()
                try:
                    worker.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    worker.process.kill()
                    worker.process.wait()
            except ProcessLookupError:
                pass
            try:
                worker.stdout.close()
            except Exception as exc:
                fatal = fatal or f"worker_stdout_close_error:{key}:{type(exc).__name__}:{exc}"
            try:
                terminal = verify_terminal(CASE_DIR / f"{key}.jsonl", worker.task, sources_start)
            except Exception as exc:
                terminal = None
                fatal = fatal or f"terminal_integrity_error_at_stop:{key}:{type(exc).__name__}:{exc}"
            if terminal is not None:
                completed[key] = terminal
                partial.discard(key)
                batch_episode_cpu += float(terminal["episode_process_cpu_seconds"])
                append_jsonl(events_path, {"event": "worker_completed_at_stop", "case_key": key})
            else:
                try:
                    append_jsonl(CASE_DIR / f"{key}.jsonl", {
                        "event": "coordinator_interruption", "case_key": key,
                        "status": "partial_not_complete", "reason": stop_reason or "coordinator_stop",
                        "input_sha256": worker.task["input_sha256"],
                        "source_hashes": worker.task["case_source_hashes"],
                    })
                except Exception as exc:
                    fatal = fatal or f"worker_interruption_receipt_error:{key}:{type(exc).__name__}:{exc}"
                partial.add(key)
                append_jsonl(events_path, {"event": "worker_interrupted", "case_key": key,
                                           "reason": stop_reason or "coordinator_stop"})
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    sources_end = source_hashes()
    if sources_end != sources_start:
        fatal = fatal or "method_source_changed_during_batch"
        stop_reason = stop_reason or "method_source_hash_drift"
    try:
        verify_inputs(task_rows, json.loads(PREVIOUS_PREP.read_text(encoding="utf-8")))
    except Exception as exc:
        fatal = fatal or f"final_input_hash_failure:{type(exc).__name__}:{exc}"
        stop_reason = stop_reason or "final_input_hash_failure"
    if fatal:
        status = "STOPPED_FATAL"
    elif len(completed) == EXPECTED_CASES:
        status = "COMPLETE_1680"
    elif stop_reason:
        status = "CHECKPOINT_RESUMABLE"
    else:
        status = "INCOMPLETE_RESUMABLE"
    final = write_manifest(status, stop_reason)
    final["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    final["case_source_hashes_at_end"] = sources_end
    final["fatal_error"] = fatal
    final["resume_policy"] = "Only hash-matching complete cases are skipped; partial attempts are retained and rerun from the initial state."
    atomic_json(manifest_path, final)
    atomic_json(TASK_DIR / "results_v2" / "cloud_latest_manifest.json", final)
    append_jsonl(events_path, {"event": "batch_terminal", "status": status,
                               "completed_count": len(completed), "partial_count": len(partial),
                               "not_run_count": EXPECTED_CASES - len(completed) - len(partial),
                               "stop_reason": stop_reason, "fatal_error": fatal})
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="hash-check all inputs/results and write a preparation receipt")
    parser.add_argument("--execute", action="store_true", help="run/resume the dataset after prepare validation")
    parser.add_argument("--workers", type=int, default=2, help="choose 1..4 based on visible RAM/CPU; default 2")
    parser.add_argument("--batch-hours", type=float, default=24, help="strict maximum wall time for this batch (default/max 24h)")
    args = parser.parse_args()
    if args.execute and args.prepare:
        parser.error("choose only one of --prepare or --execute")
    result = run(args.workers, args.batch_hours) if args.execute else prepare()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
