"""Replay the six-container author run and verify full-budget/legal transitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from baseline.zhang_author_data import load_zhang_author_inputs
from run_author_case import source_hashes


CASE_KEY = "4-3-6-0"
CASE_PATH = TASK_DIR / "data" / "verified_instances" / f"{CASE_KEY}.json"
LOG_PATH = TASK_DIR / "receipts" / "six_container_full_budget_v2.jsonl"


def _find(stacks, target):
    for source, stack in enumerate(stacks):
        for tier, item in enumerate(stack):
            if item == target:
                return source, tier
    raise AssertionError(f"target {target} missing")


def _hash_stacks(stacks):
    return hashlib.sha256(json.dumps(stacks, separators=(",", ":")).encode()).hexdigest()


def _pop_target(stacks, target, retrieved, all_ids, max_tier):
    source, tier = _find(stacks, target)
    assert tier == len(stacks[source]) - 1, f"retrieval target {target} was blocked"
    mutable = [list(stack) for stack in stacks]
    assert mutable[source].pop() == target
    updated = tuple(tuple(stack) for stack in mutable)
    retrieved.add(target)
    active = [item for stack in updated for item in stack]
    assert all(len(stack) <= max_tier for stack in updated)
    assert len(active) == len(set(active))
    assert not (set(active) & retrieved)
    assert set(active) | retrieved == all_ids
    return updated


def main() -> None:
    raw_bytes = CASE_PATH.read_bytes()
    raw = json.loads(raw_bytes)
    inputs = load_zhang_author_inputs(CASE_PATH)
    all_ids = {item for stack in inputs.stacks for item in stack}
    retrieved: set[int] = set()
    stacks = inputs.stacks
    rows = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    starts = [row for row in rows if row.get("event") == "case_start"]
    decisions = [row for row in rows if row.get("event") == "decision"]
    terminals = [row for row in rows if row.get("event") == "case_terminal"]
    assert len(starts) == 1 and len(terminals) == 1
    terminal = terminals[0]
    assert terminal["status"] == "complete"
    assert terminal["source_hashes"] == source_hashes() == terminal["source_hashes_at_end"]
    assert terminal["input_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert starts[0]["future_retrieval_sequence_passed_to_policy"] is False
    assert starts[0]["policy_input_contract"] == [
        "current_stacks", "author_rpm", "max_tiers", "current_target_id",
    ]
    sequence = raw["retrieval_sequence"]
    expected_reuses = []
    retrieval_index = 0
    relocation_count = 0
    for index, row in enumerate(decisions):
        row_retrieval_index = int(row["retrieval_index"])
        while retrieval_index < row_retrieval_index:
            stacks = _pop_target(
                stacks, sequence[retrieval_index], retrieved, all_ids, inputs.max_tiers,
            )
            retrieval_index += 1
        assert retrieval_index == row_retrieval_index
        target = sequence[retrieval_index]
        assert int(row["current_target_id"]) == target
        source, tier = _find(stacks, target)
        assert tier < len(stacks[source]) - 1
        assert row["state_sha256"] == _hash_stacks(stacks)
        legal = tuple(
            destination for destination, stack in enumerate(stacks)
            if destination != source and len(stack) < inputs.max_tiers
        )
        assert tuple(row["legal_destinations"]) == legal
        action_source, destination = map(int, row["action"])
        assert action_source == source and destination in legal
        blocker = stacks[source][-1]
        assert int(row["moved_blocker_id"]) == blocker
        assert int(row["playout_budget"]) == 2500 * len(stacks)
        assert int(row["playouts_added"]) == 2500 * len(stacks)
        assert int(row["root_visits_after"]) - int(row["root_visits_before"]) == 2500 * len(stacks)
        assert row["policy_received_only_current_target"] is True
        if index and row["tree_reuse"] and int(row["retrieval_index"]) != int(decisions[index - 1]["retrieval_index"]):
            expected_reuses.append({
                "from_retrieval_index": int(decisions[index - 1]["retrieval_index"]),
                "to_retrieval_index": int(row["retrieval_index"]),
                "target_id": target,
                "status": row["tree_reuse_status"],
            })
        mutable = [list(stack) for stack in stacks]
        assert mutable[source].pop() == blocker
        mutable[destination].append(blocker)
        stacks = tuple(tuple(stack) for stack in mutable)
        relocation_count += 1
        active = [item for stack in stacks for item in stack]
        assert all(len(stack) <= inputs.max_tiers for stack in stacks)
        assert len(active) == len(set(active))
        assert not (set(active) & retrieved)
        assert set(active) | retrieved == all_ids
        source, tier = _find(stacks, target)
        if tier == len(stacks[source]) - 1:
            stacks = _pop_target(stacks, target, retrieved, all_ids, inputs.max_tiers)
            retrieval_index += 1

    while retrieval_index < len(sequence):
        stacks = _pop_target(stacks, sequence[retrieval_index], retrieved, all_ids, inputs.max_tiers)
        retrieval_index += 1

    assert not any(stacks) and retrieved == all_ids
    assert relocation_count == int(terminal["relocations_R"])
    assert len(retrieved) == int(terminal["complete_retrievals"]) == 6
    assert len(decisions) == int(terminal["decision_count"]) == 3
    assert int(terminal["total_movements_R_plus_N"]) == relocation_count + 6
    assert terminal["RR"] == relocation_count / 6
    assert len(expected_reuses) >= 1, "no exact reuse was observed across a real retrieval transition"

    receipt = {
        "schema": "author_six_container_full_budget_gate_v1",
        "status": "PASS_COMPLETE_TRAJECTORY_AND_REPLAY_AUDIT",
        "case_key": CASE_KEY,
        "input_sha256": terminal["input_sha256"],
        "source_hashes": terminal["source_hashes"],
        "N": 6,
        "R": relocation_count,
        "RR": relocation_count / 6,
        "decision_count": len(decisions),
        "budget_per_decision": 2500 * len(inputs.stacks),
        "retrieval_sequence_exactly_replayed": True,
        "legal_actions_and_conservation_verified": True,
        "outer_policy_input_contains_only_current_target": True,
        "cross_retrieval_exact_branch_reuse": expected_reuses,
        "episode_wall_seconds": terminal["episode_wall_seconds"],
        "episode_process_cpu_seconds": terminal["episode_process_cpu_seconds"],
        "full_1680_case_queue_started": False,
    }
    out = TASK_DIR / "receipts" / "six_container_gate_audit_v2.json"
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
