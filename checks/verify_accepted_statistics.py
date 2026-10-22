"""Verify registered suite-level estimates against the frozen accepted tables.

This reads endpoint records and recomputes only registered primary contrasts and
the P1/P3/P4 rollout/LB1 interaction. It does not launch trajectories or
recompute per-condition summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from reproducibility import statistics as stats
from reproducibility.protocol import SOLVERS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "results" / "uwsaa-benchmark-evaluation" / "data" / "accepted_primary_contrasts_all_solvers.csv"
DEFAULT_SCENARIO = ROOT / "results" / "uwsaa-benchmark-evaluation" / "data" / "scenario_ablation_primary_contrasts_all_solvers.csv"
DEFAULT_OUTPUT = ROOT / "checks" / "accepted_statistics_verification.json"
TOLERANCE = 2e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _comparison_has_baseline(comparison: str, baseline: str) -> bool:
    token = _norm(baseline.removeprefix("UWSAA_"))
    suffix = r"(?![0-9])" if token[-1:].isdigit() else r""
    return re.search(re.escape(token) + suffix, _norm(comparison)) is not None


def _num(row: Mapping[str, str], field: str, scale: float = 1.0) -> float | None:
    value = row.get(field, "")
    if value in (None, ""):
        return None
    return float(value) / scale


def _first_num(row: Mapping[str, str], fields: Sequence[str], scale: float = 1.0) -> float | None:
    for field in fields:
        if row.get(field, "") not in (None, ""):
            return _num(row, field, scale)
    return None


def _primary_table_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], Mapping[str, str]]:
    index: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        stage = _norm(str(row.get("stage", "")))
        comparison = str(row.get("comparison", ""))
        metric = str(row.get("metric", ""))
        solver = str(row.get("solver", ""))
        if metric != "relocations" or not solver:
            continue
        if "MAIN" in stage:
            category = "main"
            baseline = "NATIVE20"
        elif "ABLATION" in stage:
            category = "ablation"
            baseline = next((arm for arm in stats.ABLATION_VARIANTS
                             if _comparison_has_baseline(comparison, arm)), "")
        elif "DETERMINISTIC" in stage or "P0" in stage:
            category = "deterministic"
            baseline = "UWSAA_DETERMINISTIC_SINGLE"
        else:
            continue
        if not baseline or not _comparison_has_baseline(comparison, baseline):
            raise ValueError(f"accepted comparison label does not identify its registered baseline: {comparison}")
        key = (category, baseline, solver)
        if key in index:
            raise ValueError(f"duplicate accepted primary row: {key}")
        index[key] = row
    return index


def _compare_primary(index: Mapping[tuple[str, str, str], Mapping[str, str]],
                     category: str, baseline: str, computed: Sequence[Mapping[str, Any]],
                     *, corrected_level: float | None = None,
                     error_rows: list[dict[str, Any]]) -> None:
    for item in computed:
        solver = str(item["solver"])
        key = (category, baseline, solver)
        if key not in index:
            raise ValueError(f"accepted primary table lacks row: {key}")
        source = index[key]
        # Older accepted tables use percent columns; normalized tables use proportions.
        percent_estimate = bool(source.get("relative_reduction_percent", ""))
        accepted_fields = (
            ("estimate", float(item["estimate"]), ("relative_reduction_percent", "estimate", "effect")),
            ("ordinary_ci_low", float(item["ci95_low"]), ("ordinary_ci_low_percent", "ordinary_ci_low", "ordinary_ci_lower", "ci95_low")),
            ("ordinary_ci_high", float(item["ci95_high"]), ("ordinary_ci_high_percent", "ordinary_ci_high", "ordinary_ci_upper", "ci95_high")),
        )
        for label, calculated, fields in accepted_fields:
            accepted = _first_num(source, fields, 100.0 if percent_estimate else 1.0)
            if accepted is None:
                raise ValueError(f"accepted primary value missing: {key}/{label}")
            error_rows.append({"suite": category, "baseline": baseline, "solver": solver,
                               "field": label, "absolute_difference": abs(calculated - accepted)})
        if corrected_level is not None:
            accepted_level = _num(source, "adjusted_ci_level")
            if accepted_level is None or not math.isclose(accepted_level, corrected_level, abs_tol=TOLERANCE):
                raise ValueError(f"adjusted confidence level differs for {key}")
            for label, calculated, fields in (
                ("adjusted_ci_low", float(item["adjusted_ci_low"]), ("adjusted_ci_low_percent", "adjusted_ci_low", "adjusted_ci_lower")),
                ("adjusted_ci_high", float(item["adjusted_ci_high"]), ("adjusted_ci_high_percent", "adjusted_ci_high", "adjusted_ci_upper")),
            ):
                accepted = _first_num(source, fields, 100.0 if percent_estimate else 1.0)
                if accepted is None:
                    raise ValueError(f"accepted adjusted interval missing: {key}/{label}")
                error_rows.append({"suite": category, "baseline": baseline, "solver": solver,
                                   "field": label, "absolute_difference": abs(calculated - accepted)})


def _scenario_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    index: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        contrast = str(row.get("contrast_id", ""))
        solver = str(row.get("solver", ""))
        if not contrast or not solver:
            continue
        key = (contrast, solver)
        if key in index:
            raise ValueError(f"duplicate accepted scenario row: {key}")
        index[key] = row
    return index


def _compare_scenario(index: Mapping[tuple[str, str], Mapping[str, str]], contrast: str,
                      solver_rows: Sequence[Mapping[str, Any]], *, corrected_level: float | None,
                      absolute: bool, error_rows: list[dict[str, Any]]) -> None:
    for item in solver_rows:
        solver = str(item["solver"])
        key = (contrast, solver)
        if key not in index:
            raise ValueError(f"accepted scenario table lacks row: {key}")
        source = index[key]
        accepted_unit = str(source.get("unit", source.get("metric", "")))
        if absolute and accepted_unit not in {"relocations", "relocation_count"}:
            raise ValueError(f"interaction unit is not relocations: {key}={accepted_unit}")
        expected = float(item["estimate"])
        for label, fields in (("estimate", ("estimate", "relative_reduction_fraction", "effect")),
                              ("ordinary_ci_low", ("ordinary_ci_lower", "ordinary_ci_low", "ci95_low")),
                              ("ordinary_ci_high", ("ordinary_ci_upper", "ordinary_ci_high", "ci95_high"))):
            accepted = _first_num(source, fields)
            calculated = expected if label == "estimate" else float(item[label])
            if accepted is None:
                raise ValueError(f"accepted scenario value missing: {key}/{label}")
            error_rows.append({"suite": "scenario_ablation", "contrast": contrast, "solver": solver,
                               "field": label, "absolute_difference": abs(calculated - accepted)})
        if corrected_level is not None:
            level = _num(source, "adjusted_ci_level")
            if level is None or not math.isclose(level, corrected_level, abs_tol=TOLERANCE):
                raise ValueError(f"adjusted scenario confidence level differs for {key}")
            for label, fields in (("adjusted_ci_low", ("adjusted_ci_lower", "adjusted_ci_low")),
                                 ("adjusted_ci_high", ("adjusted_ci_upper", "adjusted_ci_high"))):
                accepted = _first_num(source, fields)
                calculated = float(item[label])
                if accepted is None:
                    raise ValueError(f"accepted adjusted scenario interval missing: {key}/{label}")
                error_rows.append({"suite": "scenario_ablation", "contrast": contrast, "solver": solver,
                                   "field": label, "absolute_difference": abs(calculated - accepted)})


def _as_rows(summary: Mapping[str, Any], corrected: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for solver in (*SOLVERS, "MEAN6"):
        if solver == "MEAN6":
            estimate = float(summary["estimate_mean6_relative_reduction"])
            ordinary = summary["mean6_interval"]
            adjusted = corrected["mean6_interval"] if corrected else None
        else:
            estimate = float(summary["solver_estimates"][solver])
            ordinary = summary["solver_intervals"][solver]
            adjusted = corrected["solver_intervals"][solver] if corrected else None
        row: dict[str, Any] = {"solver": solver, "estimate": estimate,
                               "ci95_low": float(ordinary[0]), "ci95_high": float(ordinary[1])}
        if adjusted:
            row["adjusted_ci_low"], row["adjusted_ci_high"] = map(float, adjusted)
        rows.append(row)
    return rows


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    endpoint_path = args.endpoints
    parts_index: Path | None = None
    if not endpoint_path.is_file() and endpoint_path.resolve() == stats.DEFAULT_ENDPOINTS.resolve():
        if not stats.DEFAULT_PARTS_INDEX.is_file():
            raise FileNotFoundError(f"endpoint payload and parts index are missing: {endpoint_path}")
        from reproducibility.data.reassemble_endpoints import assemble_endpoint_gzip
        endpoint_path = assemble_endpoint_gzip(stats.DEFAULT_PARTS_INDEX)
        parts_index = stats.DEFAULT_PARTS_INDEX
    if not endpoint_path.is_file():
        raise FileNotFoundError(endpoint_path)
    for path in (args.accepted_primary, args.accepted_scenario):
        if not path.is_file():
            raise FileNotFoundError(path)

    endpoint_sha = stats._sha256(endpoint_path)
    primary_sha = _sha256(args.accepted_primary)
    scenario_sha = _sha256(args.accepted_scenario)
    errors: list[dict[str, Any]] = []
    accepted_primary = _primary_table_index(_read_csv(args.accepted_primary))
    accepted_scenario = _scenario_index(_read_csv(args.accepted_scenario))
    if primary_sha != _sha256(args.accepted_primary) or scenario_sha != _sha256(args.accepted_scenario):
        raise ValueError("an accepted comparison table changed during initial verification setup")
    expected_primary = {
        ("main", "NATIVE20", solver) for solver in (*SOLVERS, "MEAN6")
    } | {
        ("ablation", arm, solver) for arm in stats.ABLATION_VARIANTS for solver in (*SOLVERS, "MEAN6")
    } | {
        ("deterministic", "UWSAA_DETERMINISTIC_SINGLE", solver) for solver in (*SOLVERS, "MEAN6")
    }
    if set(accepted_primary) != expected_primary:
        missing = sorted(expected_primary - set(accepted_primary))
        extra = sorted(set(accepted_primary) - expected_primary)
        raise ValueError(f"accepted primary table keyset mismatch; missing={missing}, extra={extra}")
    scenario_ids = ("P1_K1_vs_K8", "P1_K2_vs_K8", "P1_K4_vs_K8", "P1_K16_vs_K8",
                    "P3_LB1_vs_K8_rollout", "P4_LB1_vs_K1_rollout", "absolute_interaction")
    expected_scenario = {(contrast, solver) for contrast in scenario_ids for solver in (*SOLVERS, "MEAN6")}
    if set(accepted_scenario) != expected_scenario:
        missing = sorted(expected_scenario - set(accepted_scenario))
        extra = sorted(set(accepted_scenario) - expected_scenario)
        raise ValueError(f"accepted scenario table keyset mismatch; missing={missing}, extra={extra}")
    canonical = stats._canonical_rows(endpoint_path)
    matrices: dict[str, Any] = {}

    for suite, category in (("main", "main"), ("ablation", "ablation"), ("deterministic", "deterministic")):
        selected, _definitions, primary, _conditions, metadata = stats._suite_spec(suite, canonical)
        indexes = stats._bootstrap_indices(selected, stats.DEFAULT_SEED, stats.DEFAULT_DRAWS)
        matrices[suite] = {**metadata["matrix"], "selected_endpoint_rows": len(selected)}
        for contrast in primary:
            result_rows, _ = stats._run_effect(
                selected, contrast["baseline"], contrast["candidate"], contrast["groups"],
                indexes, stats.DEFAULT_DRAWS, contrast["alpha"], metrics=("relocations",)
            )
            _compare_primary(accepted_primary, category, contrast["baseline"], result_rows,
                             corrected_level=(1 - contrast["alpha"]) if contrast["alpha"] else None,
                             error_rows=errors)

    main_rows, _definitions, _primary, _conditions, _metadata = stats._suite_spec("main", canonical)
    main_indexes = stats._bootstrap_indices(main_rows, stats.DEFAULT_SEED, stats.DEFAULT_DRAWS)
    clean_ni = stats._clean_noninferiority(main_rows, main_indexes, stats.DEFAULT_DRAWS)

    selected, definitions, primary, _conditions, metadata = stats._suite_spec("scenario-ablation", canonical)
    indexes = stats._bootstrap_indices(selected, stats.DEFAULT_SEED, stats.DEFAULT_DRAWS)
    matrices["scenario-ablation"] = {**metadata["matrix"], "selected_endpoint_rows": len(selected)}
    contrast_ids = {
        ("UWSAA_K8_ROLLOUT", "UWSAA_K1_ROLLOUT"): "P1_K1_vs_K8",
        ("UWSAA_K8_ROLLOUT", "UWSAA_K2_ROLLOUT"): "P1_K2_vs_K8",
        ("UWSAA_K8_ROLLOUT", "UWSAA_K4_ROLLOUT"): "P1_K4_vs_K8",
        ("UWSAA_K8_ROLLOUT", "UWSAA_K16_ROLLOUT"): "P1_K16_vs_K8",
        ("UWSAA_K8_LB1", "UWSAA_K8_ROLLOUT"): "P3_LB1_vs_K8_rollout",
        ("UWSAA_K1_LB1", "UWSAA_K1_ROLLOUT"): "P4_LB1_vs_K1_rollout",
    }
    for contrast in primary:
        result_rows, summary = stats._run_effect(
            selected, contrast["baseline"], contrast["candidate"], contrast["groups"],
            indexes, stats.DEFAULT_DRAWS, contrast["alpha"], metrics=("relocations",)
        )
        contrast_id = contrast_ids[(contrast["candidate"], contrast["baseline"])]
        _compare_scenario(accepted_scenario, contrast_id, _scenario_rows_from_effect_rows(result_rows),
                          corrected_level=1 - contrast["alpha"], absolute=False,
                          error_rows=errors)

    noisy_groups = stats._family_groups(definitions, stats.PURE_FAMILIES)
    ordinary = stats._interaction(selected, indexes, noisy_groups, draws=stats.DEFAULT_DRAWS,
                                  relative=False, alpha=0.05)
    adjusted = stats._interaction(selected, indexes, noisy_groups, draws=stats.DEFAULT_DRAWS,
                                  relative=False, alpha=1 / 60)
    scenario_rows = []
    for solver in (*SOLVERS, "MEAN6"):
        estimate = ordinary["mean6_estimate"] if solver == "MEAN6" else ordinary["solver_estimates"][solver]
        interval = ordinary["mean6_interval"] if solver == "MEAN6" else ordinary["solver_intervals"][solver]
        adjusted_interval = adjusted["mean6_interval"] if solver == "MEAN6" else adjusted["solver_intervals"][solver]
        scenario_rows.append({"solver": solver, "estimate": estimate,
                              "ordinary_ci_low": interval[0], "ordinary_ci_high": interval[1],
                              "adjusted_ci_low": adjusted_interval[0], "adjusted_ci_high": adjusted_interval[1]})
    _compare_scenario(accepted_scenario, "absolute_interaction", scenario_rows,
                      corrected_level=1 - 1 / 60, absolute=True, error_rows=errors)

    max_error = max((item["absolute_difference"] for item in errors), default=0.0)
    if max_error > TOLERANCE:
        worst = max(errors, key=lambda item: item["absolute_difference"])
        raise ValueError(f"accepted statistics differ by {max_error:.6g}, exceeding tolerance: {worst}")
    expected_primary = 56
    if len(accepted_primary) != expected_primary * 1:
        raise ValueError(f"expected 56 accepted main/ablation/P0 solver rows, found {len(accepted_primary)}")
    if len(accepted_scenario) != 49:
        raise ValueError(f"expected 49 accepted 063 primary solver rows, found {len(accepted_scenario)}")
    return {
        "schema": "uwsaa_accepted_statistics_verification_v1",
        "status": "PASS",
        "recomputation_scope": "registered suite-level primary effects and the rollout/LB1 absolute interaction only; no trajectories were run and per-condition summaries were not recomputed",
        "inputs": {
            "endpoint_payload": {"repository_path": "reproducibility/data/uwsaa_trajectory_endpoints.csv.gz (reassembled from listed parts when needed)",
                                  "sha256": endpoint_sha, "compressed_bytes": endpoint_path.stat().st_size,
                                  "accepted_and_gate_rows_parsed": len(canonical)},
            "endpoint_parts_index_sha256": _sha256(parts_index) if parts_index else _sha256(stats.DEFAULT_PARTS_INDEX),
            "accepted_primary_table": {"repository_path": args.accepted_primary.relative_to(ROOT).as_posix(),
                                        "sha256": primary_sha, "rows_compared": len(accepted_primary)},
            "accepted_scenario_table": {"comparison_source_id": args.scenario_source_id,
                                         "sha256": scenario_sha, "rows_compared": len(accepted_scenario)},
        },
        "protocol": {"bootstrap_draws": stats.DEFAULT_DRAWS, "bootstrap_seed": stats.DEFAULT_SEED,
                     "interval_tolerance": TOLERANCE,
                     "aggregation_order": "conditions within layout; layouts within geometry; geometries equally; relative family effects equally; solver effects equally",
                     "bootstrap_unit": "paired layout fingerprint sampled within geometry"},
        "matrices": matrices,
        "comparisons": {"main_ablation_p0_solver_rows": len(accepted_primary),
                        "scenario_ablation_solver_rows_including_interaction": len(accepted_scenario),
                        "total_solver_rows": len(accepted_primary) + len(accepted_scenario),
                        "numeric_fields_compared": len(errors),
                        "maximum_absolute_difference": max_error,
                        "maximum_by_field": {
                            field: max((item["absolute_difference"] for item in errors if item["field"] == field), default=0.0)
                            for field in sorted({item["field"] for item in errors})
                        }},
        "clean_noninferiority": clean_ni,
        "limitations": [
            "first_forecast_row_sha256 is a hash of the source first forecast row, not the entire forecast table",
            "full action traces and original recovery logs remain in the local archive; this check uses endpoint records",
            "historical partial and correctness-smoke records are excluded from accepted matrices",
            "this check does not rerun performance experiments or recompute condition-level result tables",
        ],
    }


def _scenario_rows_from_effect_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"solver": row["solver"], "estimate": row["estimate"],
             "ordinary_ci_low": row["ci95_low"], "ordinary_ci_high": row["ci95_high"],
             "adjusted_ci_low": row["adjusted_ci_low"], "adjusted_ci_high": row["adjusted_ci_high"]}
            for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", type=Path, default=stats.DEFAULT_ENDPOINTS)
    parser.add_argument("--accepted-primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--accepted-scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--scenario-source-id", default="registered_scenario_primary_contrasts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = _verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": args.output.name,
                      "rows_compared": result["comparisons"]["total_solver_rows"],
                      "numeric_fields_compared": result["comparisons"]["numeric_fields_compared"],
                      "maximum_absolute_difference": result["comparisons"]["maximum_absolute_difference"]}))


if __name__ == "__main__":
    main()
