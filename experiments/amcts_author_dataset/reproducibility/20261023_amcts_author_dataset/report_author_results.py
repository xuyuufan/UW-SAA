"""Validate per-case logs and generate ARR tables and figures after a run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_LIST = TASK_DIR / "data" / "task_list_1680_v2.jsonl"
CASE_LOG_DIR = TASK_DIR / "results_v2" / "cases"
PREP_RECEIPT = TASK_DIR / "receipts" / "batch_preparation_v2.json"
LOCAL_STOP_RECEIPT = TASK_DIR / "receipts" / "local_stop_attempt03_final.json"
DEFAULT_OUT = TASK_DIR / "results_v2" / "analysis"
RESULTS_DIR = TASK_DIR / "results_v2"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from run_author_case import source_hashes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object row at {path}:{number}")
        result.append(value)
    return result


def _resource_phases() -> list[tuple[str, str]]:
    """Return local and cloud phase starts, keeping their timings distinct."""
    phases = [
        ("attempt01_workers8_resource_pressure", RESULTS_DIR / "launch_attempt01.json"),
        ("attempt02_worker1", RESULTS_DIR / "launch_attempt02.json"),
        ("attempt03_workers2", RESULTS_DIR / "launch_attempt03.json"),
    ]
    starts: list[tuple[str, str]] = []
    for name, path in phases:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            started = payload.get("started_utc")
            if started:
                starts.append((started, name))
    cloud_dir = RESULTS_DIR / "cloud_batches"
    if cloud_dir.exists():
        for manifest_path in sorted(cloud_dir.glob("*/manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            started = payload.get("start_utc")
            if not started:
                continue
            batch_id = str(payload.get("batch_id") or manifest_path.parent.name)
            workers = payload.get("workers", "unknown")
            starts.append((started, f"cloud_batch_{batch_id}_workers{workers}"))
    return sorted(starts)


def _phase_for_time(started_utc: str | None, origin: str, phases: list[tuple[str, str]]) -> str:
    if origin == "pre_release_gate":
        return origin
    if not started_utc:
        return "unknown"
    applicable = [name for start, name in phases if start <= started_utc]
    stop_cutoff = None
    if LOCAL_STOP_RECEIPT.exists():
        stop_payload = json.loads(LOCAL_STOP_RECEIPT.read_text(encoding="utf-8"))
        stop_cutoff = stop_payload.get("captured_utc") or stop_payload.get("stop_requested_utc")
    if stop_cutoff and started_utc > stop_cutoff:
        if not applicable or not applicable[-1].startswith("cloud_batch_"):
            return "unknown_cloud"
    return applicable[-1] if applicable else "pre_queue_or_gate"


def _load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    task_bytes = TASK_LIST.read_bytes()
    prep = json.loads(PREP_RECEIPT.read_text(encoding="utf-8"))
    if _sha256_bytes(task_bytes) != prep.get("task_list_sha256"):
        raise ValueError("task-list hash does not match preparation receipt")
    tasks = [json.loads(line) for line in task_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(tasks) != 1680 or len({row["case_key"] for row in tasks}) != 1680:
        raise ValueError("expected exactly 1680 unique frozen task rows")
    expected_sources = prep["case_source_hashes"]
    if source_hashes() != expected_sources:
        raise ValueError("current case source hashes differ from the frozen preparation receipt")
    results = []
    attempts: list[dict[str, Any]] = []
    phases = _resource_phases()
    for task in tasks:
        case_log = CASE_LOG_DIR / f"{task['case_key']}.jsonl"
        log_rows = _read_jsonl(case_log)
        terminals = [row for row in log_rows if row.get("event") == "case_terminal"]
        if len(terminals) > 1:
            raise ValueError(f"duplicate terminal rows for {task['case_key']}")
        terminal = terminals[0] if terminals else None
        terminal_attempt_id = terminal.get("attempt_id") if terminal else None
        starts = {row.get("attempt_id"): row for row in log_rows
                  if row.get("event") == "case_start" and row.get("attempt_id")}
        attempt_ids = list(dict.fromkeys(row.get("attempt_id") for row in log_rows if row.get("attempt_id")))
        logged_partial_wall = 0.0
        logged_partial_cpu = 0.0
        logged_partial_decisions = 0
        prior_partial_phases: list[str] = []
        for attempt_id in attempt_ids:
            attempt_rows = [row for row in log_rows if row.get("attempt_id") == attempt_id]
            decisions = [row for row in attempt_rows if row.get("event") == "decision"]
            start_row = starts.get(attempt_id, {})
            phase = _phase_for_time(start_row.get("time_utc"), task.get("origin", ""), phases)
            is_canonical = attempt_id == terminal_attempt_id
            attempt_wall = sum(float(row.get("decision_wall_seconds", 0.0)) for row in decisions)
            attempt_cpu = sum(float(row.get("decision_process_cpu_seconds", 0.0)) for row in decisions)
            attempts.append({
                "case_key": task["case_key"], "configuration": task["configuration"],
                "seed": int(task["seed"]), "W": int(task["stack_count_W"]),
                "H": int(task["max_tier_H"]), "N": int(task["container_count_N"]),
                "attempt_id": attempt_id, "canonical_scoring_attempt": is_canonical,
                "status": "complete_canonical" if is_canonical else "incomplete_noncanonical",
                "resource_phase": phase, "logged_decision_count": len(decisions),
                "logged_decision_wall_seconds": attempt_wall,
                "logged_decision_process_cpu_seconds": attempt_cpu,
            })
            if not is_canonical:
                logged_partial_wall += attempt_wall
                logged_partial_cpu += attempt_cpu
                logged_partial_decisions += len(decisions)
                prior_partial_phases.append(phase)
        if terminal is not None:
            if terminal.get("status") != "complete":
                raise ValueError(f"non-complete terminal row for {task['case_key']}")
            if terminal.get("input_sha256") != task["input_sha256"]:
                raise ValueError(f"input hash mismatch for {task['case_key']}")
            if terminal.get("source_hashes") != expected_sources or terminal.get("source_hashes_at_end") != expected_sources:
                raise ValueError(f"source hash mismatch for {task['case_key']}")
            n = int(terminal["container_count_N"])
            r = int(terminal["relocations_R"])
            if n != int(task["container_count_N"]) or r < 0 or float(terminal["RR"]) != r / n:
                raise ValueError(f"invalid N/R/RR metrics for {task['case_key']}")
            canonical_decisions = [row for row in log_rows
                                   if row.get("event") == "decision" and row.get("attempt_id") == terminal_attempt_id]
            if len(canonical_decisions) != int(terminal["decision_count"]):
                raise ValueError(f"canonical decision count mismatch for {task['case_key']}")
            expected_budget = 2500 * int(task["stack_count_W"])
            if any(int(row.get("playout_budget", -1)) != expected_budget or
                   int(row.get("playouts_added", -1)) != expected_budget for row in canonical_decisions):
                raise ValueError(f"canonical decision budget mismatch for {task['case_key']}")
            canonical_start = next((row for row in log_rows
                                    if row.get("event") == "case_start" and row.get("attempt_id") == terminal_attempt_id), None)
            if canonical_start is None or canonical_start.get("future_retrieval_sequence_passed_to_policy") is not False:
                raise ValueError(f"future retrieval sequence leakage boundary failed for {task['case_key']}")
            if any(row.get("policy_received_only_current_target") is not True for row in canonical_decisions):
                raise ValueError(f"policy input boundary failed for {task['case_key']}")
            status = "complete"
        elif any(row.get("event") in {"case_error", "coordinator_interruption"} for row in log_rows):
            status = "partial_or_failed"
        elif log_rows:
            status = "partial"
        else:
            status = "not_run"
        results.append({
            "task_index": int(task["task_index"]), "case_key": task["case_key"],
            "configuration": task["configuration"], "seed": int(task["seed"]),
            "W": int(task["stack_count_W"]), "H": int(task["max_tier_H"]),
            "N": int(task["container_count_N"]), "status": status,
            "canonical_attempt_id": terminal_attempt_id,
            "canonical_resource_phase": _phase_for_time(
                next((row.get("time_utc") for row in log_rows
                      if row.get("event") == "case_start" and row.get("attempt_id") == terminal_attempt_id), None),
                task.get("origin", ""), phases),
            "attempt_count": len(attempt_ids),
            "prior_partial_attempt_count": sum(attempt["case_key"] == task["case_key"] and not attempt["canonical_scoring_attempt"] for attempt in attempts[-len(attempt_ids):]) if attempt_ids else 0,
            "prior_partial_resource_phases": ";".join(dict.fromkeys(prior_partial_phases)),
            "prior_partial_logged_decision_count": logged_partial_decisions,
            "prior_partial_logged_decision_wall_seconds": logged_partial_wall,
            "prior_partial_logged_decision_process_cpu_seconds": logged_partial_cpu,
            "R": int(terminal["relocations_R"]) if terminal else None,
            "RR": float(terminal["RR"]) if terminal else None,
            "total_movements_R_plus_N": int(terminal["total_movements_R_plus_N"]) if terminal else None,
            "episode_wall_seconds": float(terminal["episode_wall_seconds"]) if terminal else None,
            "episode_process_cpu_seconds": float(terminal["episode_process_cpu_seconds"]) if terminal else None,
            "decision_count": int(terminal["decision_count"]) if terminal else None,
            "tree_reuse_calls": int(terminal["tree_reuse_calls"]) if terminal else None,
            "origin": task.get("origin", "not_run"),
        })
    return results, attempts, prep, {"task_list_sha256": _sha256_bytes(task_bytes), "source_hashes": expected_sources}


def build_report(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    rows, attempts, prep, provenance = _load_inputs()
    cloud_manifest_path = RESULTS_DIR / "cloud_latest_manifest.json"
    local_stop_path = LOCAL_STOP_RECEIPT
    batch_manifest_path = cloud_manifest_path
    if cloud_manifest_path.exists():
        batch_manifest = json.loads(cloud_manifest_path.read_text(encoding="utf-8"))
        manifest_origin = "cloud_batch"
    elif local_stop_path.exists():
        batch_manifest = json.loads(local_stop_path.read_text(encoding="utf-8"))
        batch_manifest = {"status": "LOCAL_WINDOWS_RUN_INTERRUPTED", **batch_manifest}
        manifest_origin = "local_stop_receipt"
    else:
        batch_manifest = {}
        manifest_origin = "none"
    complete = [row for row in rows if row["status"] == "complete"]
    partial = [row for row in rows if row["status"] in {"partial", "partial_or_failed"}]
    not_run = [row for row in rows if row["status"] == "not_run"]
    rr_values = [float(row["RR"]) for row in complete]
    full_arr = sum(rr_values) / len(rr_values) if len(complete) == 1680 else None
    complete_mean = sum(rr_values) / len(rr_values) if rr_values else None

    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / "case_results_1680.csv"
    with case_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    configs = sorted({row["configuration"] for row in rows}, key=lambda value: tuple(int(x) for x in value.split("-")))
    config_summaries = []
    for config in configs:
        group = [row for row in rows if row["configuration"] == config]
        group_complete = [row for row in group if row["status"] == "complete"]
        config_summaries.append({
            "configuration": config,
            "expected_cases": len(group),
            "complete_cases": len(group_complete),
            "partial_cases": sum(row["status"] in {"partial", "partial_or_failed"} for row in group),
            "not_run_cases": sum(row["status"] == "not_run" for row in group),
            "mean_relocations_R": sum(row["R"] for row in group_complete) / len(group_complete) if group_complete else None,
            "mean_RR": sum(row["RR"] for row in group_complete) / len(group_complete) if group_complete else None,
            "mean_RR_percent": 100.0 * sum(row["RR"] for row in group_complete) / len(group_complete) if group_complete else None,
            "confidence_interval": None,
        })
    config_csv = out_dir / "configuration_summary_56.csv"
    with config_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(config_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(config_summaries)

    partial_attempts = [row for row in attempts if not row["canonical_scoring_attempt"]]
    partial_attempt_csv = out_dir / "partial_attempt_costs.csv"
    partial_fields = list(partial_attempts[0].keys()) if partial_attempts else [
        "case_key", "configuration", "seed", "W", "H", "N", "attempt_id",
        "canonical_scoring_attempt", "status", "resource_phase", "logged_decision_count",
        "logged_decision_wall_seconds", "logged_decision_process_cpu_seconds",
    ]
    with partial_attempt_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=partial_fields)
        writer.writeheader()
        writer.writerows(partial_attempts)

    phase_names = list(dict.fromkeys([row["canonical_resource_phase"] for row in complete] +
                                     [row["resource_phase"] for row in partial_attempts]))
    phase_summaries = []
    for phase in phase_names:
        phase_complete = [row for row in complete if row["canonical_resource_phase"] == phase]
        phase_partial = [row for row in partial_attempts if row["resource_phase"] == phase]
        phase_summaries.append({
            "resource_phase": phase,
            "complete_case_count": len(phase_complete),
            "complete_episode_wall_seconds_total": sum(row["episode_wall_seconds"] for row in phase_complete),
            "complete_episode_wall_seconds_mean": sum(row["episode_wall_seconds"] for row in phase_complete) / len(phase_complete) if phase_complete else None,
            "complete_episode_process_cpu_seconds_total": sum(row["episode_process_cpu_seconds"] for row in phase_complete),
            "noncanonical_partial_attempt_count": len(phase_partial),
            "partial_logged_decision_wall_seconds_total": sum(row["logged_decision_wall_seconds"] for row in phase_partial),
            "partial_logged_decision_process_cpu_seconds_total": sum(row["logged_decision_process_cpu_seconds"] for row in phase_partial),
            "partial_logged_decision_count": sum(row["logged_decision_count"] for row in phase_partial),
            "partial_cost_note": "Logged completed decisions only; an interrupted in-progress decision may not have been flushed.",
        })
    phase_csv = out_dir / "resource_phase_summary.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_summaries[0].keys()) if phase_summaries else ["resource_phase"])
        writer.writeheader()
        writer.writerows(phase_summaries)

    summary = {
        "schema": "amcts_author_dataset_results_v1",
        "status": "COMPLETE_1680" if len(complete) == 1680 else "PARTIAL_DESCRIPTIVE",
        "batch_manifest_status": batch_manifest.get("status", "missing"),
        "batch_manifest_origin": manifest_origin,
        "report_is_live_snapshot": batch_manifest.get("status") == "RUNNING",
        "dataset_version": 3,
        "expected_cases": 1680,
        "complete_cases": len(complete),
        "partial_or_failed_cases": len(partial),
        "not_run_cases": len(not_run),
        "primary_metric": "ARR = mean(RR), RR = relocations R / container count N",
        "playout_budget_validation": "Every canonical complete decision must record exactly 2500 x W added playouts.",
        "information_boundary_validation": "Each canonical case start marks the full retrieval sequence as evaluator-only; each policy decision receives only the current target.",
        "ARR_full_dataset": full_arr,
        "ARR_full_dataset_percent": 100.0 * full_arr if full_arr is not None else None,
        "complete_case_subset_mean_RR_descriptive_only": complete_mean,
        "complete_case_subset_mean_RR_percent_descriptive_only": 100.0 * complete_mean if complete_mean is not None else None,
        "mean_absolute_relocations_complete_cases": sum(row["R"] for row in complete) / len(complete) if complete else None,
        "all_56_configurations_reported": len(config_summaries) == 56,
        "confidence_intervals": None,
        "paper_reported_full_set_AMCTS_ARR_percent_reference_only": 45.52,
        "paper_timing_comparison_claim": False,
        "task_list_sha256": provenance["task_list_sha256"],
        "case_source_hashes": provenance["source_hashes"],
        "coordinator_source_hashes_at_start": batch_manifest.get("coordinator_source_hashes_at_start"),
        "coordinator_source_hashes_at_end": batch_manifest.get("coordinator_source_hashes_at_end"),
        "root_release_file_sha256": batch_manifest.get("release_file_sha256"),
        "final_input_verification": batch_manifest.get("final_input_verification"),
        "archive_sha256": prep["archive_sha256"],
        "metadata_index_sha256": prep["metadata_index_sha256"],
        "case_results_csv": str(case_csv.relative_to(TASK_DIR)),
        "configuration_summary_csv": str(config_csv.relative_to(TASK_DIR)),
        "partial_attempt_costs_csv": str(partial_attempt_csv.relative_to(TASK_DIR)),
        "resource_phase_summary_csv": str(phase_csv.relative_to(TASK_DIR)),
        "resource_phase_summary": phase_summaries,
        "resource_timing_interpretation": "Descriptive operational timing only; resource phases used different case mixes and are not a controlled speed comparison.",
        "partial_cost_accounting": "Noncanonical attempts are reported separately from scoring rows; only logged completed decisions are summed, so interrupted in-progress work may be absent.",
        "no_costs_imputed_for_incomplete_cases": True,
    }
    summary_path = out_dir / "results_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the registered PNG/PDF figure") from exc

    fig, ax = plt.subplots(figsize=(22, 8))
    positions = []
    values = []
    for position, config in enumerate(configs, 1):
        group_values = [float(row["RR"]) for row in rows if row["configuration"] == config and row["status"] == "complete"]
        if group_values:
            positions.append(position)
            values.append(group_values)
    if values:
        artists = ax.boxplot(values, positions=positions, widths=0.62, showfliers=False, patch_artist=True)
        for box in artists["boxes"]:
            box.set_facecolor("#9ecae1")
            box.set_edgecolor("#1f4e79")
        for median in artists["medians"]:
            median.set_color("#a50f15")
            median.set_linewidth(1.5)
        if complete_mean is not None:
            ax.axhline(
                complete_mean, color="#238b45", linestyle="--", linewidth=1.4,
                label=f"complete-case mean RR = {complete_mean:.3f} (n={len(complete)})",
            )
        if len(complete) == 1680:
            ax.axhline(
                0.4552, color="#636363", linestyle=":", linewidth=1.4,
                label="paper-reported all-case AMCTS ARR = 45.52% (literature reference)",
            )
    ax.set_xticks(range(1, len(configs) + 1))
    ax.set_xticklabels(configs, rotation=90, fontsize=6)
    ax.set_ylabel("RR = relocations / N (lower is better)")
    ax.set_xlabel("Published configuration W-H-N; see CSV for per-configuration n")
    ax.set_title(f"AMCTS on public author instances: {len(complete)}/1680 complete; no confidence intervals")
    ax.grid(axis="y", alpha=0.2)
    if values:
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    png_path = out_dir / "author_dataset_rr_by_configuration.png"
    pdf_path = out_dir / "author_dataset_rr_by_configuration.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    timing_fig, axes = plt.subplots(3, 1, figsize=(22, 13), gridspec_kw={"height_ratios": [1.2, 1.2, 0.8]})
    x = list(range(len(configs)))
    complete_counts = [next(item["complete_cases"] for item in config_summaries if item["configuration"] == config) for config in configs]
    partial_counts = [next(item["partial_cases"] for item in config_summaries if item["configuration"] == config) for config in configs]
    not_run_counts = [next(item["not_run_cases"] for item in config_summaries if item["configuration"] == config) for config in configs]
    axes[0].bar(x, complete_counts, label="complete", color="#238b45")
    axes[0].bar(x, partial_counts, bottom=complete_counts, label="partial / failed", color="#fd8d3c")
    axes[0].bar(x, not_run_counts, bottom=[a + b for a, b in zip(complete_counts, partial_counts)], label="not run", color="#bdbdbd")
    axes[0].set_ylabel("cases (n=30/config)")
    axes[0].set_title("Coverage across all 56 published configurations")
    axes[0].set_ylim(0, 30)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)

    mean_wall_by_config = []
    for config in configs:
        vals = [row["episode_wall_seconds"] for row in complete if row["configuration"] == config]
        mean_wall_by_config.append(sum(vals) / len(vals) if vals else None)
    wall_positions = [i for i, value in enumerate(mean_wall_by_config) if value is not None and value > 0]
    axes[1].bar(wall_positions, [mean_wall_by_config[i] for i in wall_positions], color="#3182bd")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("mean complete-case wall time (s, log; lower cost is better)")
    axes[1].set_title("Per-case time by configuration; canonical complete attempts only")

    phase_partial_wall = [item["partial_logged_decision_wall_seconds_total"] for item in phase_summaries]
    axes[2].bar(range(len(phase_summaries)), phase_partial_wall, color="#9e9ac8")
    axes[2].set_xticks(range(len(phase_summaries)))
    axes[2].set_xticklabels([item["resource_phase"] for item in phase_summaries], rotation=15, ha="right")
    axes[2].set_ylabel("logged partial decision wall seconds")
    axes[2].set_title("Noncanonical partial computation, reported separately")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    axes[1].set_xticks(range(len(configs)))
    axes[1].set_xticklabels(configs, rotation=90, fontsize=5)
    timing_fig.suptitle("AMCTS author-set coverage and operational timing (descriptive; no CI; resource phases are not controlled comparisons)")
    timing_fig.tight_layout(rect=(0, 0, 1, 0.97))
    timing_png = out_dir / "author_dataset_coverage_timing.png"
    timing_pdf = out_dir / "author_dataset_coverage_timing.pdf"
    timing_fig.savefig(timing_png, dpi=200)
    timing_fig.savefig(timing_pdf)
    plt.close(timing_fig)

    summary["figure_png"] = str(png_path.relative_to(TASK_DIR))
    summary["figure_pdf"] = str(pdf_path.relative_to(TASK_DIR))
    summary["coverage_timing_figure_png"] = str(timing_png.relative_to(TASK_DIR))
    summary["coverage_timing_figure_pdf"] = str(timing_pdf.relative_to(TASK_DIR))
    summary["analysis_script_sha256"] = _sha256_bytes(Path(__file__).read_bytes())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build_report(args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
