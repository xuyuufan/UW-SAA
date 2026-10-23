"""Correctness-only checks for append-only author-run resume semantics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from run_author_case import resume_decision, source_hashes


def main() -> None:
    current = source_hashes()
    key = "4-3-6-0"
    input_hash = "a" * 64
    partial_id = "attempt-partial"
    partial = [
        {"event": "case_start", "case_key": key, "attempt_id": partial_id},
        {"event": "decision", "case_key": key, "attempt_id": partial_id},
    ]
    decision, attempts = resume_decision(
        partial, case_key=key, input_sha256=input_hash, current_sources=current,
    )
    assert decision == "restart_from_initial_after_partial" and attempts == [partial_id]
    assert not any(row.get("event") == "case_terminal" and row.get("status") == "complete" for row in partial)

    partial_input_mismatch_rejected = False
    try:
        resume_decision(
            [dict(partial[0], input_sha256="b" * 64, source_hashes=current)],
            case_key=key, input_sha256=input_hash, current_sources=current,
        )
    except ValueError as exc:
        partial_input_mismatch_rejected = "partial-attempt input hash mismatch" in str(exc)
    assert partial_input_mismatch_rejected

    partial_source_mismatch_rejected = False
    try:
        resume_decision(
            [dict(partial[0], input_sha256=input_hash, source_hashes={"paper_zhang.py": "0" * 64})],
            case_key=key, input_sha256=input_hash, current_sources=current,
        )
    except ValueError as exc:
        partial_source_mismatch_rejected = "partial-attempt source hash mismatch" in str(exc)
    assert partial_source_mismatch_rejected

    completed = {
        "event": "case_terminal", "case_key": key, "attempt_id": "attempt-complete",
        "status": "complete", "input_sha256": input_hash, "source_hashes": current,
    }
    decision, attempts = resume_decision(
        partial + [completed], case_key=key, input_sha256=input_hash, current_sources=current,
    )
    assert decision == "skip_completed" and attempts == []

    mismatch_rejected = False
    try:
        resume_decision(
            [completed], case_key=key, input_sha256="b" * 64, current_sources=current,
        )
    except ValueError as exc:
        mismatch_rejected = "hash mismatch" in str(exc)
    assert mismatch_rejected

    duplicate_rejected = False
    try:
        resume_decision(
            [completed, dict(completed, attempt_id="duplicate")],
            case_key=key, input_sha256=input_hash, current_sources=current,
        )
    except ValueError as exc:
        duplicate_rejected = "duplicate terminal" in str(exc)
    assert duplicate_rejected

    source_mismatch = dict(completed, source_hashes={"paper_zhang.py": "0" * 64})
    source_mismatch_rejected = False
    try:
        resume_decision(
            [source_mismatch], case_key=key, input_sha256=input_hash, current_sources=current,
        )
    except ValueError as exc:
        source_mismatch_rejected = "hash mismatch" in str(exc)
    assert source_mismatch_rejected

    receipt = {
        "schema": "author_run_resume_integrity_v2",
        "status": "PASS_CORRECTNESS_ONLY",
        "source_hashes_used": current,
        "partial_attempt_restarts_from_initial": True,
        "completed_attempt_skipped_exactly_once": True,
        "input_mismatch_refused": True,
        "source_mismatch_refused": True,
        "partial_attempt_input_mismatch_refused": True,
        "partial_attempt_source_mismatch_refused": True,
        "duplicate_terminal_key_refused": True,
        "partial_attempt_not_counted_as_complete": True,
        "author_dataset_run_started": False,
    }
    out = TASK_DIR / "receipts" / "resume_integrity_v2_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
