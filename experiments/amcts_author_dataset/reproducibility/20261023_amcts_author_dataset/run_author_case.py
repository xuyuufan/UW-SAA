"""Run one complete author instance with AMCTS and append resumable JSONL events."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = TASK_DIR / "data" / "verified_instances"
DEFAULT_LOG = TASK_DIR / "receipts" / "author_runs.jsonl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.paper_zhang import AMCTSPolicy
from baseline.zhang_author_data import load_zhang_author_inputs


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = {
        "paper_zhang.py": REPO_ROOT / "baseline" / "paper_zhang.py",
        "zhang_author_data.py": REPO_ROOT / "baseline" / "zhang_author_data.py",
        "run_author_case.py": Path(__file__).resolve(),
    }
    return {name: sha256_bytes(path.read_bytes()) for name, path in paths.items()}


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_case_log(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def resume_decision(
    rows: list[dict[str, object]], *, case_key: str, input_sha256: str,
    current_sources: dict[str, str],
) -> tuple[str, list[str]]:
    terminal = [row for row in rows if row.get("event") == "case_terminal" and row.get("case_key") == case_key]
    if len(terminal) > 1:
        raise ValueError(f"duplicate terminal records for case {case_key}")
    if terminal:
        row = terminal[0]
        if row.get("status") != "complete":
            raise ValueError(f"case {case_key} has a terminal non-complete record; manual review required")
        if row.get("input_sha256") != input_sha256 or row.get("source_hashes") != current_sources:
            raise ValueError(f"refusing resume: input/source hash mismatch for completed case {case_key}")
        return "skip_completed", []
    partial_rows = [
        row for row in rows
        if row.get("case_key") == case_key
        and row.get("event") in {"case_start", "decision", "case_error", "coordinator_interruption"}
    ]
    for row in partial_rows:
        row_input = row.get("input_sha256")
        row_sources = row.get("source_hashes")
        if row_input is not None and row_input != input_sha256:
            raise ValueError(f"refusing resume: partial-attempt input hash mismatch for case {case_key}")
        if row_sources is not None and row_sources != current_sources:
            raise ValueError(f"refusing resume: partial-attempt source hash mismatch for case {case_key}")
    partial = sorted({
        str(row.get("attempt_id")) for row in partial_rows
        if row.get("attempt_id") is not None
    })
    return ("restart_from_initial_after_partial", partial) if partial else ("run_new", [])


def _find_item(stacks: tuple[tuple[int, ...], ...], item: int) -> tuple[int, int]:
    for stack_index, stack in enumerate(stacks):
        for tier, value in enumerate(stack):
            if value == item:
                return stack_index, tier
    raise ValueError(f"target item {item} is absent from the current bay")


def _stack_hash(stacks: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(stacks, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def _validate_bay(stacks: tuple[tuple[int, ...], ...], all_ids: set[int], max_tiers: int, retrieved: set[int]) -> None:
    active = [item for stack in stacks for item in stack]
    if any(len(stack) > max_tiers for stack in stacks):
        raise AssertionError("capacity violation")
    if len(active) != len(set(active)):
        raise AssertionError("duplicate live item")
    if set(active).intersection(retrieved):
        raise AssertionError("retrieved item remains in bay")
    if set(active).union(retrieved) != all_ids:
        raise AssertionError("container conservation violation")


def run_case(
    case_path: Path, *, log_path: Path = DEFAULT_LOG, playout_multiplier: int = 2500,
    reuse_namespace_prefix: str = "mendeley-v3",
) -> dict[str, object]:
    if playout_multiplier != 2500:
        raise ValueError("registered author-set budget is fixed at 2500 × W")
    raw_bytes = case_path.read_bytes()
    input_sha = sha256_bytes(raw_bytes)
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("author case must be a JSON object")
    if "retrieval_sequence" not in raw:
        raise ValueError("outer evaluator requires retrieval_sequence")
    case_key = case_path.stem
    inputs = load_zhang_author_inputs(case_path)
    sequence = raw["retrieval_sequence"]  # private to this evaluator; never forwarded to the policy
    all_ids = {item for stack in inputs.stacks for item in stack}
    if (not isinstance(sequence, list) or set(sequence) != all_ids or len(sequence) != len(all_ids)):
        raise ValueError("retrieval_sequence must be a permutation of the instance's item IDs")
    sources_at_start = source_hashes()
    history = read_case_log(log_path)
    resume_status, prior_attempts = resume_decision(
        history, case_key=case_key, input_sha256=input_sha, current_sources=sources_at_start,
    )
    if resume_status == "skip_completed":
        completed = next(row for row in history if row.get("event") == "case_terminal" and row.get("case_key") == case_key)
        return {**completed, "resume_action": "skip_completed_without_append"}

    attempt_id = str(uuid.uuid4())
    namespace = f"{reuse_namespace_prefix}:{case_key}"
    policy = AMCTSPolicy(playout_multiplier=playout_multiplier)
    stacks = inputs.stacks
    retrieved: set[int] = set()
    relocation_count = 0
    decisions: list[dict[str, object]] = []
    episode_wall_start = time.perf_counter()
    episode_cpu_start = time.process_time()
    append_event(log_path, {
        "event": "case_start", "case_key": case_key, "attempt_id": attempt_id,
        "resume_status": resume_status, "prior_partial_attempt_ids": prior_attempts,
        "input_sha256": input_sha, "source_hashes": sources_at_start,
        "namespace": namespace, "retrieval_count_N": len(sequence),
        "policy_input_contract": ["current_stacks", "author_rpm", "max_tiers", "current_target_id"],
        "future_retrieval_sequence_passed_to_policy": False,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    try:
        for retrieval_index, target_id in enumerate(sequence):
            if target_id in retrieved:
                raise AssertionError("target was already retrieved")
            action_index = 0
            while True:
                source, tier = _find_item(stacks, target_id)
                if tier == len(stacks[source]) - 1:
                    mutable = [list(stack) for stack in stacks]
                    popped = mutable[source].pop()
                    if popped != target_id:
                        raise AssertionError("retrieval popped a different item")
                    stacks = tuple(tuple(stack) for stack in mutable)
                    retrieved.add(target_id)
                    _validate_bay(stacks, all_ids, inputs.max_tiers, retrieved)
                    break

                legal = tuple(
                    destination for destination, stack in enumerate(stacks)
                    if destination != source and len(stack) < inputs.max_tiers
                )
                if not legal:
                    raise RuntimeError("blocked target has no legal destination")
                decision_index = len(decisions)
                state_hash = _stack_hash(stacks)
                decision_wall_start = time.perf_counter()
                decision_cpu_start = time.process_time()
                # The only retrieval-order information passed here is target_id.
                destination, details = policy.select_state(
                    stacks, inputs.rpm, inputs.max_tiers, int(target_id),
                    reuse_namespace=namespace,
                )
                decision_wall = time.perf_counter() - decision_wall_start
                decision_cpu = time.process_time() - decision_cpu_start
                expected_budget = 2500 * len(stacks)
                if details["playouts"] != expected_budget or details["playouts_added"] != expected_budget:
                    raise AssertionError("AMCTS did not use the registered 2500 × W fresh budget")
                if details["root_visits_after"] - details["root_visits_before"] != expected_budget:
                    raise AssertionError("root visit delta differs from registered fresh budget")
                if destination not in legal:
                    raise AssertionError("AMCTS returned an illegal destination")

                blocker = stacks[source][-1]
                mutable = [list(stack) for stack in stacks]
                moved = mutable[source].pop()
                if moved != blocker:
                    raise AssertionError("only the top blocker may be relocated")
                mutable[destination].append(moved)
                stacks = tuple(tuple(stack) for stack in mutable)
                relocation_count += 1
                _validate_bay(stacks, all_ids, inputs.max_tiers, retrieved)
                event = {
                    "event": "decision", "case_key": case_key, "attempt_id": attempt_id,
                    "decision_index": decision_index, "retrieval_index": retrieval_index,
                    "action_within_retrieval": action_index, "current_target_id": int(target_id),
                    "state_sha256": state_hash, "legal_destinations": legal,
                    "action": [source, int(destination)], "moved_blocker_id": blocker,
                    "playout_budget": expected_budget,
                    "root_visits_before": details["root_visits_before"],
                    "root_visits_after": details["root_visits_after"],
                    "playouts_added": details["playouts_added"],
                    "tree_reuse": details["tree_reuse"],
                    "tree_reuse_status": details["tree_reuse_status"],
                    "reuse_root_parent_detached": details["reuse_root_parent_detached"],
                    "decision_wall_seconds": decision_wall,
                    "decision_process_cpu_seconds": decision_cpu,
                    "policy_received_only_current_target": True,
                }
                decisions.append(event)
                append_event(log_path, event)
                action_index += 1

        if stacks != tuple(() for _ in stacks) or retrieved != all_ids:
            raise AssertionError("case did not end with all containers retrieved and bay empty")
        sources_at_end = source_hashes()
        if sources_at_end != sources_at_start:
            raise RuntimeError("source hash drift during case")
        episode_wall = time.perf_counter() - episode_wall_start
        episode_cpu = time.process_time() - episode_cpu_start
        terminal = {
            "event": "case_terminal", "case_key": case_key, "attempt_id": attempt_id,
            "status": "complete", "input_sha256": input_sha,
            "source_hashes": sources_at_start, "source_hashes_at_end": sources_at_end,
            "container_count_N": len(sequence), "relocations_R": relocation_count,
            "total_movements_R_plus_N": relocation_count + len(sequence),
            "RR": relocation_count / len(sequence),
            "complete_retrievals": len(retrieved), "decision_count": len(decisions),
            "tree_reuse_calls": sum(bool(row["tree_reuse"]) for row in decisions),
            "cross_retrieval_reuse_calls": sum(
                bool(row["tree_reuse"]) and i > 0 and row["retrieval_index"] != decisions[i - 1]["retrieval_index"]
                for i, row in enumerate(decisions)
            ),
            "episode_wall_seconds": episode_wall, "episode_process_cpu_seconds": episode_cpu,
            "resume_status": resume_status,
            "resume_action": "executed_new_attempt",
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_event(log_path, terminal)
        return terminal
    except BaseException as exc:
        append_event(log_path, {
            "event": "case_error", "case_key": case_key, "attempt_id": attempt_id,
            "status": "incomplete", "error_type": type(exc).__name__, "error": str(exc),
            "partial_relocations": relocation_count, "partial_retrievals": len(retrieved),
            "input_sha256": input_sha, "source_hashes": sources_at_start,
            "episode_wall_seconds": time.perf_counter() - episode_wall_start,
            "episode_process_cpu_seconds": time.process_time() - episode_cpu_start,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-key", default="4-3-6-0")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--playout-multiplier", type=int, default=2500)
    args = parser.parse_args()
    case_path = args.case_dir / f"{args.case_key}.json"
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    print(json.dumps(run_case(
        case_path, log_path=args.log, playout_multiplier=args.playout_multiplier,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
