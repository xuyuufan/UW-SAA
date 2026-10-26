"""Recompute registered paired relocation effects from canonical endpoints.

This reads trajectory endpoint rows; it never launches an evaluation. Estimates
follow the source macro order: conditions within layout, layouts within geometry,
geometries equally, family-specific relative reductions, then six-solver mean.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reproducibility.protocol import ABLATION_CONDITIONS, SOLVERS, main_conditions

ROOT = Path(__file__).resolve().parents[1]
# This path is populated only after a new trained GP run is merged with the
# retained non-GP evidence. It deliberately does not point at the retired mixed
# endpoint payload or its former parts index.
DEFAULT_ENDPOINTS = ROOT / "reproducibility" / "results" / "gp_migration" / "canonical_six_solver_endpoints.csv.gz"
GP_STATUS_COLUMN = "gp_status"
GP_BINDING_COLUMNS = (
    "gp_model_identity_sha256", "gp_formula_sha256", "gp_source_manifest_sha256",
    "gp_protocol_sha256", "gp_train_bank_sha256", "gp_implementation_sha256",
    "gp_seed_receipt_sha256",
)
DEFAULT_SEED = 202610195903
DEFAULT_DRAWS = 10_000
PURE_FAMILIES = ("subset_shuffle", "time_window_shuffle", "delayed_positions")
ABLATION_VARIANTS = (
    "UWSAA_NO_WINDOW_SHUFFLE_K8", "UWSAA_BEST_FIXED_H", "UWSAA_SINGLE_WINDOW_B2",
    "UWSAA_SINGLE_WINDOW_B5", "UWSAA_SINGLE_WINDOW_B10", "UWSAA_SINGLE_WINDOW_B20",
)
MULTIK_VARIANTS = (
    "UWSAA_K1_ROLLOUT", "UWSAA_K2_ROLLOUT", "UWSAA_K4_ROLLOUT",
    "UWSAA_K16_ROLLOUT", "UWSAA_K1_LB1", "UWSAA_K8_LB1",
)
CORRECTED_ALPHA = {
    "UWSAA_NO_WINDOW_SHUFFLE_K8": 0.025,
    "UWSAA_BEST_FIXED_H": 0.025,
    "UWSAA_SINGLE_WINDOW_B2": 0.0125,
    "UWSAA_SINGLE_WINDOW_B5": 0.0125,
    "UWSAA_SINGLE_WINDOW_B10": 0.0125,
    "UWSAA_SINGLE_WINDOW_B20": 0.0125,
}
METRICS = ("relocations", "relocations_per_container")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> Iterable[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "study_group", "evidence_status", "episode_id", "layout_fingerprint",
            "condition_id", "family", "repeat", "solver", "arm", "geometry",
            "parameter_group", "relocations", "relocations_per_container",
            "first_forecast_row_sha256", "num_containers", GP_STATUS_COLUMN, *GP_BINDING_COLUMNS,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"canonical endpoint CSV lacks required columns: {sorted(missing)}")
        yield from reader


def _canonical_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_statuses = {"accepted", "failed_performance_gate", "failed_clean_gate"}
    for line_number, source in enumerate(_csv_rows(path), start=2):
        if source["evidence_status"] not in accepted_statuses:
            continue
        try:
            count = int(source["num_containers"])
            relocations = int(source["relocations"])
            ratio = float(source["relocations_per_container"])
            repeat = int(source["repeat"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid endpoint value at CSV row {line_number}") from exc
        if count <= 0 or relocations < 0 or not math.isfinite(ratio):
            raise ValueError(f"invalid relocation endpoint at CSV row {line_number}")
        if not math.isclose(ratio, relocations / count, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"relocations_per_container mismatch at CSV row {line_number}")
        fingerprint = source["layout_fingerprint"]
        if len(fingerprint) != 64 or any(c not in "0123456789abcdefABCDEF" for c in fingerprint):
            raise ValueError(f"invalid layout fingerprint at CSV row {line_number}")
        forecast_hash = source["first_forecast_row_sha256"]
        if len(forecast_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in forecast_hash):
            raise ValueError(f"invalid first-forecast-row hash at CSV row {line_number}")
        if source["geometry"] != source["parameter_group"]:
            raise ValueError(f"geometry and parameter_group disagree at CSV row {line_number}")
        gp_binding = {field: source[field].strip() for field in GP_BINDING_COLUMNS}
        if source["solver"] == "GP":
            if source[GP_STATUS_COLUMN] != "trained":
                raise ValueError(f"GP endpoint is not marked trained at CSV row {line_number}")
            if any(len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value)
                   for value in gp_binding.values()):
                raise ValueError(f"GP endpoint lacks complete model/source/protocol SHA bindings at CSV row {line_number}")
        elif source[GP_STATUS_COLUMN].strip() or any(gp_binding.values()):
            raise ValueError(f"non-GP endpoint unexpectedly carries GP model identity at CSV row {line_number}")
        rows.append({
            "gp_binding": gp_binding if source["solver"] == "GP" else None,
            "study_group": source["study_group"], "evidence_status": source["evidence_status"],
            "episode_id": source["episode_id"], "layout_fingerprint": fingerprint,
            "condition_id": source["condition_id"], "family": source["family"], "repeat": repeat,
            "solver": source["solver"], "arm": source["arm"], "geometry": source["geometry"],
            "relocations": relocations, "relocations_per_container": ratio,
            "num_containers": count, "first_forecast_row_sha256": forecast_hash,
            "source_id": source.get("source_id", ""), "source_sha256": source.get("source_sha256", ""),
            "source_row": source.get("source_row", ""),
        })
    return rows


def _validate_gp_model_binding(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    bindings = {tuple(sorted(dict(row["gp_binding"]).items())) for row in rows if row.get("solver") == "GP"}
    if not bindings:
        raise ValueError("no trained GP endpoints are present; five-solver data cannot produce six-solver statistics")
    if len(bindings) != 1:
        raise ValueError("canonical endpoints mix GP model/source/protocol identities; aggregate is refused")
    return dict(next(iter(bindings)))


def _merge_unique(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    episode_geometry: dict[str, str] = {}
    episode_fingerprint: dict[str, str] = {}
    episode_counts: dict[str, int] = {}
    condition_metadata: dict[str, tuple[str, int]] = {}
    forecast_by_cell: dict[tuple[str, str, str], str] = {}
    for source in rows:
        row = dict(source)
        episode_id = str(row["episode_id"])
        fingerprint = str(row["layout_fingerprint"])
        geometry = str(row["geometry"])
        if episode_id in episode_geometry and episode_geometry[episode_id] != geometry:
            raise ValueError(f"episode {episode_id} changes geometry")
        if episode_id in episode_fingerprint and episode_fingerprint[episode_id] != fingerprint:
            raise ValueError(f"episode {episode_id} changes layout fingerprint")
        count = int(row["num_containers"])
        if episode_id in episode_counts and episode_counts[episode_id] != count:
            raise ValueError(f"episode {episode_id} changes container count")
        condition, family, repeat = str(row["condition_id"]), str(row["family"]), int(row["repeat"])
        if condition in condition_metadata and condition_metadata[condition] != (family, repeat):
            raise ValueError(f"condition {condition} changes family or repeat metadata")
        episode_geometry[episode_id] = geometry
        episode_fingerprint[episode_id] = fingerprint
        episode_counts[episode_id] = count
        condition_metadata[condition] = (family, repeat)
        cell_key = (fingerprint, condition, str(row["solver"]))
        forecast_hash = str(row["first_forecast_row_sha256"]).lower()
        old_forecast = forecast_by_cell.setdefault(cell_key, forecast_hash)
        if old_forecast != forecast_hash:
            raise ValueError(f"paired arms have different first forecast rows: {cell_key}")
        key = (fingerprint, condition, str(row["solver"]), str(row["arm"]))
        if key in by_key:
            raise ValueError(f"duplicate endpoint pairing key: {key}")
        by_key[key] = row
    return list(by_key.values())


def _matrix(rows: Sequence[Mapping[str, Any]], conditions: Sequence[str], arms: Sequence[str],
            *, condition_definitions: Sequence[Mapping[str, Any]], expected_layouts: int = 840) -> dict[str, Any]:
    selected = [row for row in rows if row["condition_id"] in set(conditions)]
    condition_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        condition_rows[str(row["condition_id"])].append(row)
    if set(condition_rows) != set(conditions):
        missing = sorted(set(conditions) - set(condition_rows))
        raise ValueError(f"registered endpoint conditions are missing: {missing}")
    fingerprints_by_condition: dict[str, set[str]] = {}
    for condition in conditions:
        cells = condition_rows[condition]
        definition = next(item for item in condition_definitions if item["condition_id"] == condition)
        if any(row["family"] != definition["family"]
               or int(row["repeat"]) != int(definition.get("repeat", 0)) for row in cells):
            raise ValueError(f"{condition} family or repeat differs from the registered condition")
        fingerprints = {str(row["layout_fingerprint"]) for row in cells}
        fingerprints_by_condition[condition] = fingerprints
        if len(fingerprints) != expected_layouts:
            raise ValueError(f"{condition} has {len(fingerprints)} layouts; expected {expected_layouts}")
        if len(cells) != expected_layouts * len(SOLVERS) * len(arms):
            raise ValueError(f"{condition} row count does not match its complete registered matrix")
        observed = {(str(row["layout_fingerprint"]), str(row["solver"]), str(row["arm"])) for row in cells}
        expected = {(fp, solver, arm) for fp in fingerprints for solver in SOLVERS for arm in arms}
        if observed != expected:
            raise ValueError(f"{condition} has missing or unexpected solver/arm endpoints")
        geom_counts: dict[str, set[str]] = defaultdict(set)
        for row in cells:
            geom_counts[str(row["geometry"])].add(str(row["layout_fingerprint"]))
        if len(geom_counts) != 21 or any(len(fps) != 40 for fps in geom_counts.values()):
            raise ValueError(f"{condition} is not the registered 21-geometry × 40-layout design")
    first = fingerprints_by_condition[conditions[0]]
    if any(fingerprints != first for fingerprints in fingerprints_by_condition.values()):
        raise ValueError("registered conditions do not share the same paired layout set")
    return {"layout_count": len(first), "condition_count": len(conditions),
            "geometry_count": 21, "layouts_per_geometry": 40,
            "rows_per_condition": expected_layouts * len(SOLVERS) * len(arms)}


def _bootstrap_indices(rows: Sequence[Mapping[str, Any]], seed: int, draws: int) -> dict[str, np.ndarray]:
    layouts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        layouts[str(row["geometry"])].add(str(row["layout_fingerprint"]))
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return {
        geometry: rng.integers(0, len(fingerprints), size=(draws, len(fingerprints)), dtype=np.int32)
        for geometry, fingerprints in sorted(layouts.items())
    }


def _paired_effect(rows: Sequence[Mapping[str, Any]], *, baseline: str, candidate: str,
                   condition_groups: Sequence[tuple[str, Sequence[str]]],
                   bootstrap_indices: Mapping[str, np.ndarray], draws: int,
                   metric: str, alpha: float = 0.05,
                   one_sided_lower: float | None = None) -> dict[str, Any]:
    """Source-compatible family-first macro effect with paired stratified bootstrap."""
    cell: dict[tuple[str, str, str, str], float] = {}
    fingerprints_by_geometry: dict[str, set[str]] = defaultdict(set)
    required_conditions = [str(cid) for _name, ids in condition_groups for cid in ids]
    if not condition_groups or any(not ids for _name, ids in condition_groups):
        raise ValueError("paired effect requires nonempty registered condition groups")
    for row in rows:
        geometry, fingerprint = str(row["geometry"]), str(row["layout_fingerprint"])
        condition, solver, arm = str(row["condition_id"]), str(row["solver"]), str(row["arm"])
        fingerprints_by_geometry[geometry].add(fingerprint)
        key = (fingerprint, condition, solver, arm)
        if key in cell:
            raise ValueError(f"duplicate paired outcome row: {key}")
        cell[key] = float(row[metric])
    family_names = [str(name) for name, _ids in condition_groups]
    if len(set(family_names)) != len(family_names):
        raise ValueError("paired effect group labels must be unique")
    layouts_by_geometry = {
        geometry: sorted(fp for fp in fingerprints
                         if all((fp, condition, solver, arm) in cell
                                for condition in required_conditions for solver in SOLVERS
                                for arm in (baseline, candidate)))
        for geometry, fingerprints in sorted(fingerprints_by_geometry.items())
    }
    layouts_by_geometry = {geometry: fps for geometry, fps in layouts_by_geometry.items() if fps}
    geometries = tuple(sorted(layouts_by_geometry))
    if not geometries:
        raise ValueError("cannot bootstrap an empty trajectory set")
    macro_means: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for geometry in geometries:
        fingerprints = layouts_by_geometry[geometry]
        if geometry not in bootstrap_indices or bootstrap_indices[geometry].shape != (draws, len(fingerprints)):
            raise ValueError(f"paired bootstrap index count mismatch for {geometry}")
        for solver in SOLVERS:
            for family, conditions in condition_groups:
                baseline_matrix = np.asarray([
                    [cell[(fp, str(cid), solver, baseline)] for cid in conditions] for fp in fingerprints
                ], dtype=np.float64)
                candidate_matrix = np.asarray([
                    [cell[(fp, str(cid), solver, candidate)] for cid in conditions] for fp in fingerprints
                ], dtype=np.float64)
                macro_means.setdefault(geometry, {}).setdefault(solver, {})[str(family)] = {
                    "baseline": float(baseline_matrix.mean(axis=1).mean()),
                    "candidate": float(candidate_matrix.mean(axis=1).mean()),
                }
    solver_draws = np.zeros((draws, len(SOLVERS)), dtype=np.float64)
    solver_estimates: dict[str, float] = {}
    family_estimates: dict[str, dict[str, float]] = {solver: {} for solver in SOLVERS}
    for solver_index, solver in enumerate(SOLVERS):
        point_families: list[float] = []
        family_draws_all: list[np.ndarray] = []
        for family, conditions in condition_groups:
            base_macro = float(np.mean([macro_means[g][solver][str(family)]["baseline"] for g in geometries]))
            candidate_macro = float(np.mean([macro_means[g][solver][str(family)]["candidate"] for g in geometries]))
            if base_macro <= 0:
                raise ValueError(f"aggregated baseline denominator is zero for {solver}/{family}")
            point = (base_macro - candidate_macro) / base_macro
            family_estimates[solver][str(family)] = point
            point_families.append(point)
            base_draws_by_geometry: list[np.ndarray] = []
            candidate_draws_by_geometry: list[np.ndarray] = []
            for geometry in geometries:
                fingerprints = layouts_by_geometry[geometry]
                indexes = bootstrap_indices[geometry]
                base_layout = np.asarray([
                    np.mean([cell[(fp, str(cid), solver, baseline)] for cid in conditions])
                    for fp in fingerprints
                ], dtype=np.float64)
                candidate_layout = np.asarray([
                    np.mean([cell[(fp, str(cid), solver, candidate)] for cid in conditions])
                    for fp in fingerprints
                ], dtype=np.float64)
                base_draws_by_geometry.append(base_layout[indexes].mean(axis=1))
                candidate_draws_by_geometry.append(candidate_layout[indexes].mean(axis=1))
            base_draws = np.stack(base_draws_by_geometry, axis=1).mean(axis=1)
            candidate_draws = np.stack(candidate_draws_by_geometry, axis=1).mean(axis=1)
            valid = base_draws > 0
            family_draws = np.full(draws, np.nan, dtype=np.float64)
            family_draws[valid] = (base_draws[valid] - candidate_draws[valid]) / base_draws[valid]
            family_draws_all.append(family_draws)
        solver_estimates[solver] = float(np.mean(point_families))
        solver_draws[:, solver_index] = np.mean(np.stack(family_draws_all, axis=1), axis=1)
    mean6_draws = np.mean(solver_draws, axis=1)
    mean6_draws[~np.isfinite(solver_draws).all(axis=1)] = np.nan
    if not np.isfinite(mean6_draws).any():
        raise ValueError("paired bootstrap has no defined draw")
    lower_q, upper_q = alpha / 2, 1 - alpha / 2
    result = {
        "baseline": baseline, "candidate": candidate, "metric": metric,
        "estimate_mean6_relative_reduction": float(np.mean(list(solver_estimates.values()))),
        "mean6_interval": [float(x) for x in np.nanquantile(mean6_draws, [lower_q, upper_q])],
        "solver_estimates": solver_estimates,
        "solver_intervals": {
            solver: [float(x) for x in np.nanquantile(values, [lower_q, upper_q])]
            for solver, values in zip(SOLVERS, solver_draws.T, strict=True)
        },
        "family_estimates_by_solver": family_estimates,
        "layout_count_by_geometry": {geometry: len(fps) for geometry, fps in layouts_by_geometry.items()},
        "bootstrap_defined_draw_fraction_mean6": float(np.isfinite(mean6_draws).mean()),
        "confidence_level": 1.0 - alpha,
        "bootstrap_draws": draws,
        "bootstrap_seed": None,
        "direction": "positive favors candidate; lower relocations are better",
    }
    if one_sided_lower is not None:
        if not 0.0 < one_sided_lower < 1.0:
            raise ValueError("one-sided lower quantile must be between zero and one")
        result["one_sided_lower_quantile"] = one_sided_lower
        result["mean6_one_sided_lower"] = float(np.nanquantile(mean6_draws, one_sided_lower))
        result["solver_one_sided_lowers"] = {
            solver: float(np.nanquantile(values, one_sided_lower))
            for solver, values in zip(SOLVERS, solver_draws.T, strict=True)
        }
    return result


def _interaction(rows: Sequence[Mapping[str, Any]], bootstrap_indices: Mapping[str, np.ndarray],
                 condition_groups: Sequence[tuple[str, Sequence[str]]], *,
                 draws: int, relative: bool, alpha: float) -> dict[str, Any]:
    """Paired rollout/LB1 difference-in-differences from the registered source analysis."""
    k1, k8, p3, p4 = "UWSAA_K1_ROLLOUT", "UWSAA_K8_ROLLOUT", "UWSAA_K8_LB1", "UWSAA_K1_LB1"
    arms = (k1, k8, p3, p4)
    cell = {
        (str(row["layout_fingerprint"]), str(row["condition_id"]), str(row["solver"]), str(row["arm"])):
        float(row["relocations"])
        for row in rows
    }
    layouts_by_geometry: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        layouts_by_geometry[str(row["geometry"])].add(str(row["layout_fingerprint"]))
    layouts = {geometry: sorted(fps) for geometry, fps in sorted(layouts_by_geometry.items())}
    family_points: dict[str, dict[str, float | None]] = {solver: {} for solver in SOLVERS}
    solver_draw_matrix: dict[str, np.ndarray] = {}
    solver_estimates: dict[str, float | None] = {}
    for solver in SOLVERS:
        family_draws: list[np.ndarray] = []
        family_point_values: list[float] = []
        for family, conditions in condition_groups:
            arm_geometry_means: dict[str, list[float]] = {arm: [] for arm in arms}
            arm_geometry_draws: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
            for geometry, fingerprints in layouts.items():
                index = bootstrap_indices[geometry]
                arm_layout_values = {
                    arm: np.asarray([
                        np.mean([cell[(fp, str(condition), solver, arm)] for condition in conditions])
                        for fp in fingerprints
                    ], dtype=np.float64)
                    for arm in arms
                }
                for arm, values in arm_layout_values.items():
                    arm_geometry_means[arm].append(float(values.mean()))
                    arm_geometry_draws[arm].append(values[index].mean(axis=1))
            macro = {arm: float(np.mean(values)) for arm, values in arm_geometry_means.items()}
            if relative:
                defined = macro[k1] > 0 and macro[k8] > 0
                point = ((macro[p4] / macro[k1] - 1.0) - (macro[p3] / macro[k8] - 1.0)) if defined else None
            else:
                defined = True
                point = (macro[p4] - macro[k1]) - (macro[p3] - macro[k8])
            family_points[solver][str(family)] = point
            if point is not None:
                family_point_values.append(float(point))
            arm_draws = {arm: np.stack(values, axis=1).mean(axis=1)
                         for arm, values in arm_geometry_draws.items()}
            if relative:
                draw_defined = (arm_draws[k1] > 0) & (arm_draws[k8] > 0)
                values = np.full(draws, np.nan, dtype=np.float64)
                values[draw_defined] = ((arm_draws[p4][draw_defined] / arm_draws[k1][draw_defined] - 1.0)
                                        - (arm_draws[p3][draw_defined] / arm_draws[k8][draw_defined] - 1.0))
            else:
                values = (arm_draws[p4] - arm_draws[k1]) - (arm_draws[p3] - arm_draws[k8])
            family_draws.append(values)
        solver_estimates[solver] = (
            float(np.mean(family_point_values)) if not relative or len(family_point_values) == len(condition_groups)
            else None
        )
        stacked = np.stack(family_draws, axis=1)
        valid = np.isfinite(stacked).all(axis=1)
        solver_draws = np.full(draws, np.nan, dtype=np.float64)
        solver_draws[valid] = stacked[valid].mean(axis=1)
        solver_draw_matrix[solver] = solver_draws
    solver_values = np.stack([solver_draw_matrix[solver] for solver in SOLVERS], axis=1)
    valid_mean6 = np.isfinite(solver_values).all(axis=1)
    mean6_draws = np.full(draws, np.nan, dtype=np.float64)
    mean6_draws[valid_mean6] = solver_values[valid_mean6].mean(axis=1)
    defined_mean6 = mean6_draws[np.isfinite(mean6_draws)]
    if defined_mean6.size:
        qlo, qhi = np.quantile(defined_mean6, [alpha / 2, 1 - alpha / 2]).tolist()
        mean6_interval: list[float | None] = [float(qlo), float(qhi)]
    else:
        mean6_interval = [None, None]
    solver_intervals = {}
    for solver, values in solver_draw_matrix.items():
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2]).tolist()
            solver_intervals[solver] = [float(low), float(high)]
        else:
            solver_intervals[solver] = [None, None]
    point_defined = all(value is not None for value in solver_estimates.values())
    mean6_estimate = float(np.mean([solver_estimates[s] for s in SOLVERS])) if point_defined else None
    return {
        "estimand": "(P4_K1_LB1 - P1_K1_ROLLOUT) - (P3_K8_LB1 - K8_ROLLOUT)" if not relative
                     else "(P4_K1_LB1 / P1_K1_ROLLOUT - 1) - (P3_K8_LB1 / K8_ROLLOUT - 1)",
        "unit": "proportion" if relative else "relocations",
        "family_estimates_by_solver": family_points, "solver_estimates": solver_estimates,
        "solver_intervals": solver_intervals,
        "mean6_estimate": mean6_estimate, "mean6_interval": mean6_interval,
        "confidence_level": 1.0 - alpha,
        "bootstrap_defined_draw_fraction_mean6": float(valid_mean6.mean()),
        "direction": "secondary relative interaction; dimensionless proportion" if relative else
                     "positive means the LB1 relocation penalty is larger at K1 than at K8; negative means the rollout advantage is larger at K8",
    }


def _family_groups(conditions: Sequence[Mapping[str, Any]], families: Sequence[str]) -> list[tuple[str, list[str]]]:
    return [
        (family, sorted(str(row["condition_id"]) for row in conditions if row["family"] == family))
        for family in families
    ]


def _selected_rows(rows: Sequence[Mapping[str, Any]], group: str, statuses: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["study_group"] == group and row["evidence_status"] in statuses]


def _contrast_rows(summary: Mapping[str, Any], report_group: str, alpha_corrected: float | None) -> list[dict[str, Any]]:
    output = []
    for solver in (*SOLVERS, "MEAN6"):
        if solver == "MEAN6":
            estimate = float(summary["estimate_mean6_relative_reduction"])
            interval = summary["mean6_interval"]
        else:
            estimate = float(summary["solver_estimates"][solver])
            interval = summary["solver_intervals"][solver]
        output.append({
            "report_group": report_group, "baseline": summary["baseline"],
            "candidate": summary["candidate"], "metric": summary["metric"], "solver": solver,
            "estimand": "relative_reduction_of_macro_outcome", "unit": "proportion",
            "direction": "positive favors candidate; lower relocations are better",
            "estimate": estimate,
            "ci95_low": interval[0] if summary["confidence_level"] == 0.95 else "",
            "ci95_high": interval[1] if summary["confidence_level"] == 0.95 else "",
            "adjusted_alpha": alpha_corrected or "",
            "adjusted_ci_low": "", "adjusted_ci_high": "",
            "confidence_level": summary["confidence_level"],
            "bootstrap_defined_draw_fraction": summary["bootstrap_defined_draw_fraction_mean6"],
            "layout_count": sum(summary["layout_count_by_geometry"].values()),
        })
    return output


def _run_effect(rows: Sequence[Mapping[str, Any]], baseline: str, candidate: str,
                groups: Sequence[tuple[str, Sequence[str]]], indexes: Mapping[str, np.ndarray],
                draws: int, alpha_corrected: float | None,
                metrics: Sequence[str] = METRICS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contrast_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for metric in metrics:
        ordinary = _paired_effect(rows, baseline=baseline, candidate=candidate,
                                  condition_groups=groups, bootstrap_indices=indexes,
                                  draws=draws, metric=metric, alpha=0.05)
        summaries[metric] = ordinary
        contrast_rows.extend(_contrast_rows(ordinary, "PURE_BALANCED" if len(groups) > 1 else "CONDITION", None))
        if metric == "relocations" and alpha_corrected is not None:
            corrected = _paired_effect(rows, baseline=baseline, candidate=candidate,
                                       condition_groups=groups, bootstrap_indices=indexes,
                                       draws=draws, metric=metric, alpha=alpha_corrected)
            for row in contrast_rows[-7:]:
                solver = row["solver"]
                interval = corrected["mean6_interval"] if solver == "MEAN6" else corrected["solver_intervals"][solver]
                row["adjusted_ci_low"], row["adjusted_ci_high"] = interval
                row["adjusted_alpha"] = alpha_corrected
            summaries[metric]["adjusted_interval"] = corrected["mean6_interval"]
    return contrast_rows, summaries


def _clean_noninferiority(rows: Sequence[Mapping[str, Any]], indexes: Mapping[str, np.ndarray],
                          draws: int) -> dict[str, Any]:
    """Reproduce the registered MAIN CLEAN NI summaries and strict pass rules."""
    group = [("CLEAN", ["CLEAN"])]
    ordinary = _paired_effect(rows, baseline="NATIVE20", candidate="UWSAA",
                              condition_groups=group, bootstrap_indices=indexes,
                              draws=draws, metric="relocations", alpha=0.05)
    mean6 = _paired_effect(rows, baseline="NATIVE20", candidate="UWSAA",
                           condition_groups=group, bootstrap_indices=indexes,
                           draws=draws, metric="relocations", one_sided_lower=0.05)
    all_six = _paired_effect(rows, baseline="NATIVE20", candidate="UWSAA",
                             condition_groups=group, bootstrap_indices=indexes,
                             draws=draws, metric="relocations", one_sided_lower=0.05 / 6.0)
    margin = 0.01
    return {
        "status": "RECOMPUTED_FROM_COMPLETE_MAIN_ENDPOINTS",
        "baseline": "NATIVE20", "candidate": "UWSAA",
        "estimand": "relative reduction in macro mean relocations: (baseline - candidate) / baseline",
        "unit": "proportion", "positive_direction": "positive favors UWSAA",
        "relative_loss_margin": margin,
        "noninferiority_rule": "lower bound must be strictly greater than -0.01",
        "ordinary_two_sided_95": {
            "mean6_estimate": ordinary["estimate_mean6_relative_reduction"],
            "mean6_interval": ordinary["mean6_interval"],
            "solver_estimates": ordinary["solver_estimates"],
            "solver_intervals": ordinary["solver_intervals"],
        },
        "mean6_one_sided_95_lower": {
            "quantile": mean6["one_sided_lower_quantile"],
            "lower_bound": mean6["mean6_one_sided_lower"],
            "pass": mean6["mean6_one_sided_lower"] > -margin,
        },
        "all_six_bonferroni_one_sided_lower": {
            "per_solver_quantile": all_six["one_sided_lower_quantile"],
            "solver_lower_bounds": all_six["solver_one_sided_lowers"],
            "pass": all(value > -margin for value in all_six["solver_one_sided_lowers"].values()),
        },
        "bootstrap_draws": draws,
        "bootstrap_defined_draw_fraction_mean6": ordinary["bootstrap_defined_draw_fraction_mean6"],
        "layout_count_by_geometry": ordinary["layout_count_by_geometry"],
    }


def _suite_spec(suite: str, canonical_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    statuses = {"accepted", "failed_performance_gate", "failed_clean_gate"}
    main_rows = _selected_rows(canonical_rows, "caserta_main", statuses)
    main_defs = main_conditions()
    ablation_defs = list(ABLATION_CONDITIONS)
    metadata: dict[str, Any] = {"source_groups": [], "evidence_statuses": []}
    if suite == "main":
        selected = _merge_unique(main_rows)
        conditions = [str(item["condition_id"]) for item in main_defs]
        matrix = _matrix(selected, conditions, ("UWSAA", "NATIVE20"), condition_definitions=main_defs)
        groups = _family_groups(main_defs, PURE_FAMILIES)
        metadata.update(source_groups=["caserta_main"], logical_arms=["UWSAA", "NATIVE20"], matrix=matrix)
        primary = [{"baseline": "NATIVE20", "candidate": "UWSAA", "groups": groups, "alpha": None}]
        condition_contrasts = [(cid, "NATIVE20", "UWSAA") for cid in conditions]
        return selected, main_defs, primary, condition_contrasts, metadata
    if suite == "ablation":
        extra = _selected_rows(canonical_rows, "caserta_ablation", statuses)
        condition_ids = [str(item["condition_id"]) for item in ablation_defs]
        references = [row for row in main_rows if row["condition_id"] in condition_ids
                      and row["arm"] in ("UWSAA", "NATIVE20")]
        selected = _merge_unique([*references, *extra])
        arms = ("UWSAA", "NATIVE20", *ABLATION_VARIANTS)
        matrix = _matrix(selected, condition_ids, arms, condition_definitions=ablation_defs)
        groups = _family_groups(ablation_defs, PURE_FAMILIES)
        metadata.update(source_groups=["caserta_main", "caserta_ablation"], logical_arms=list(arms), matrix=matrix)
        primary = [{"baseline": arm, "candidate": "UWSAA", "groups": groups,
                    "alpha": CORRECTED_ALPHA[arm]} for arm in ABLATION_VARIANTS]
        condition_contrasts = [(cid, arm, "UWSAA") for cid in condition_ids for arm in ABLATION_VARIANTS]
        return selected, ablation_defs, primary, condition_contrasts, metadata
    if suite == "deterministic":
        condition_ids = [str(item["condition_id"]) for item in ablation_defs]
        deterministic_rows = _selected_rows(canonical_rows, "deterministic_single", statuses)
        shared_uwsaa = [row for row in main_rows
                        if row["condition_id"] in condition_ids and row["arm"] == "UWSAA"]
        selected = _merge_unique([*shared_uwsaa, *deterministic_rows])
        arms = ("UWSAA", "UWSAA_DETERMINISTIC_SINGLE")
        matrix = _matrix(selected, condition_ids, arms, condition_definitions=ablation_defs)
        groups = _family_groups(ablation_defs, PURE_FAMILIES)
        metadata.update(source_groups=["caserta_main", "deterministic_single"],
                        shared_reference="UWSAA arm from the seven matching MAIN conditions, stored once",
                        logical_arms=list(arms), matrix=matrix)
        primary = [{"baseline": "UWSAA_DETERMINISTIC_SINGLE", "candidate": "UWSAA", "groups": groups, "alpha": None}]
        condition_contrasts = [(cid, "UWSAA_DETERMINISTIC_SINGLE", "UWSAA") for cid in condition_ids]
        return selected, ablation_defs, primary, condition_contrasts, metadata
    if suite == "scenario-ablation":
        condition_ids = [str(item["condition_id"]) for item in ablation_defs]
        scenario_rows = _selected_rows(canonical_rows, "scenario_ablation", statuses)
        extra = [row for row in scenario_rows if row["arm"] in MULTIK_VARIANTS]
        k8_rollout = [row for row in scenario_rows if row["arm"] == "UWSAA_K8_ROLLOUT"]
        selected = _merge_unique([*k8_rollout, *extra])
        arms = (*MULTIK_VARIANTS, "UWSAA_K8_ROLLOUT")
        matrix = _matrix(selected, condition_ids, arms, condition_definitions=ablation_defs)
        groups = _family_groups(ablation_defs, PURE_FAMILIES)
        source_groups = sorted({str(row["study_group"]) for row in (*k8_rollout, *extra)})
        metadata.update(source_groups=source_groups, logical_arms=list(arms), matrix=matrix)
        primary = [
            {"baseline": arm, "candidate": "UWSAA_K8_ROLLOUT", "groups": groups, "alpha": 0.0125}
            for arm in MULTIK_VARIANTS[:4]
        ]
        primary.extend((
            {"baseline": "UWSAA_K8_ROLLOUT", "candidate": "UWSAA_K8_LB1", "groups": groups, "alpha": 1 / 60},
            {"baseline": "UWSAA_K1_ROLLOUT", "candidate": "UWSAA_K1_LB1", "groups": groups, "alpha": 1 / 60},
        ))
        condition_contrasts = [
            *((cid, arm, "UWSAA_K8_ROLLOUT") for cid in condition_ids for arm in MULTIK_VARIANTS[:4]),
            *((cid, "UWSAA_K8_ROLLOUT", "UWSAA_K8_LB1") for cid in condition_ids),
            *((cid, "UWSAA_K1_ROLLOUT", "UWSAA_K1_LB1") for cid in condition_ids),
        ]
        return selected, ablation_defs, primary, condition_contrasts, metadata
    raise ValueError(f"unknown registered analysis suite {suite!r}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "ablation", "deterministic", "scenario-ablation"), required=True)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.draws != DEFAULT_DRAWS:
        parser.error("registered protocol requires exactly 10,000 bootstrap draws")
    if args.seed != DEFAULT_SEED:
        parser.error(f"registered protocol requires seed {DEFAULT_SEED}")
    endpoint_path = args.endpoints
    if not endpoint_path.is_file():
        parser.error(
            f"new six-solver canonical endpoint file is not available: {endpoint_path}; "
            "restore/filter the retained non-GP endpoints, train and freeze GP, then rerun and merge registered suites"
        )
    canonical_rows = _canonical_rows(endpoint_path)
    gp_model_binding = _validate_gp_model_binding(canonical_rows)
    selected, definitions, primary, condition_contrasts, metadata = _suite_spec(args.suite, canonical_rows)
    if set(row["solver"] for row in selected) != set(SOLVERS):
        parser.error("registered statistics require complete six-solver data; five-solver MEAN6 is forbidden")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output directory already exists; choose a new path")
    indexes = _bootstrap_indices(selected, args.seed, args.draws)
    comparison_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for contrast in primary:
        name = f"{contrast['candidate']}_vs_{contrast['baseline']}"
        result_rows, effect = _run_effect(selected, contrast["baseline"], contrast["candidate"],
                                          contrast["groups"], indexes, args.draws, contrast["alpha"],
                                          metrics=("relocations",) if args.suite == "scenario-ablation" else METRICS)
        for row in result_rows:
            row["report_group"] = "PURE_BALANCED"
            row["comparison"] = name
            row["corrected_confidence_level"] = (1 - contrast["alpha"]) if contrast["alpha"] else ""
        comparison_rows.extend(result_rows)
        summaries[name] = effect
    definitions_by_id = {str(item["condition_id"]): item for item in definitions}
    for condition_id, baseline, candidate in condition_contrasts:
        group = [(f"condition:{condition_id}", [condition_id])]
        alpha = CORRECTED_ALPHA.get(baseline) if args.suite == "ablation" else None
        result_rows, effect = _run_effect(selected, baseline, candidate, group, indexes, args.draws, alpha)
        for row in result_rows:
            row["condition_id"] = condition_id
            row["family"] = str(definitions_by_id[condition_id]["family"])
            row["repeat"] = int(definitions_by_id[condition_id].get("repeat", 0))
            row["comparison"] = f"{candidate}_vs_{baseline}"
            row["report_group"] = f"condition:{condition_id}"
            row["corrected_confidence_level"] = (1 - alpha) if alpha else ""
        condition_rows.extend(result_rows)

    clean_ni = _clean_noninferiority(selected, indexes, args.draws) if args.suite == "main" else None
    clean_rows: list[dict[str, Any]] = []
    interaction_details: dict[str, Any] = {}
    if args.suite == "scenario-ablation":
        clean_group = [("CLEAN", ["CLEAN"])]
        for contrast in primary:
            result_rows, _clean_effect = _run_effect(
                selected, contrast["baseline"], contrast["candidate"], clean_group,
                indexes, args.draws, None, metrics=("relocations",)
            )
            for row in result_rows:
                row["report_group"] = "CLEAN_SECONDARY"
                row["comparison"] = f"{contrast['candidate']}_vs_{contrast['baseline']}"
                row["condition_id"] = "CLEAN"
                row["family"] = "CLEAN"
                row["repeat"] = 0
                row["corrected_confidence_level"] = ""
            clean_rows.extend(result_rows)
        noisy_groups = _family_groups(definitions, PURE_FAMILIES)
        interaction_details = {
            "absolute_interaction": {
                "ordinary_95": _interaction(selected, indexes, noisy_groups,
                                              draws=args.draws, relative=False, alpha=0.05),
                "bonferroni_98_33": _interaction(selected, indexes, noisy_groups,
                                                  draws=args.draws, relative=False, alpha=1 / 60),
                "unit": "relocations",
                "direction": "positive means the LB1 relocation penalty is larger at K1 than at K8; negative means the rollout advantage is larger at K8",
            },
            "relative_interaction_secondary_95": _interaction(
                selected, indexes, noisy_groups, draws=args.draws, relative=True, alpha=0.05
            ),
            "CLEAN_secondary": {
                "absolute_interaction_95": _interaction(
                    selected, indexes, clean_group, draws=args.draws, relative=False, alpha=0.05
                ),
                "relative_interaction_secondary_95": _interaction(
                    selected, indexes, clean_group, draws=args.draws, relative=True, alpha=0.05
                ),
            },
        }
        interaction_rows = []
        ordinary = interaction_details["absolute_interaction"]["ordinary_95"]
        adjusted = interaction_details["absolute_interaction"]["bonferroni_98_33"]
        for solver in (*SOLVERS, "MEAN6"):
            if solver == "MEAN6":
                estimate = ordinary["mean6_estimate"]
                ci = ordinary["mean6_interval"]
                adjusted_ci = adjusted["mean6_interval"]
            else:
                estimate = ordinary["solver_estimates"][solver]
                ci = ordinary["solver_intervals"][solver]
                adjusted_ci = adjusted["solver_intervals"][solver]
            interaction_rows.append({
                "report_group": "PURE_BALANCED", "comparison": "ABSOLUTE_INTERACTION_P4_K1_MINUS_P1_K1_MINUS_P3_K8_MINUS_K8",
                "baseline": "paired rollout arms", "candidate": "paired LB1 arms",
                "metric": "relocations", "estimand": ordinary["estimand"], "unit": "relocations",
                "direction": ordinary["direction"], "solver": solver, "estimate": estimate,
                "ci95_low": ci[0], "ci95_high": ci[1], "adjusted_alpha": 1 / 60,
                "adjusted_ci_low": adjusted_ci[0], "adjusted_ci_high": adjusted_ci[1],
                "confidence_level": 0.95, "corrected_confidence_level": 1 - 1 / 60,
                "bootstrap_defined_draw_fraction": ordinary["bootstrap_defined_draw_fraction_mean6"],
                "layout_count": sum(metadata["matrix"]["layouts_per_geometry"] for _ in range(21)),
            })
        comparison_rows.extend(interaction_rows)

    common_fields = ["report_group", "comparison", "baseline", "candidate", "metric", "estimand",
                     "unit", "direction", "solver", "estimate", "ci95_low", "ci95_high", "adjusted_alpha",
                     "adjusted_ci_low", "adjusted_ci_high", "confidence_level",
                     "corrected_confidence_level", "bootstrap_defined_draw_fraction", "layout_count"]
    _write_csv(args.output_dir / "registered_contrasts.csv", comparison_rows, common_fields)
    condition_fields = ["condition_id", "family", "repeat", *common_fields]
    _write_csv(args.output_dir / "condition_effects.csv", condition_rows, condition_fields)
    if clean_rows:
        _write_csv(args.output_dir / "clean_secondary.csv", clean_rows, ["condition_id", "family", "repeat", *common_fields])
    manifest = {
        "schema": "uwsaa_registered_statistics_v1", "status": "PASS_COMPLETE_REGISTERED_MATRIX",
        "suite": args.suite, "source_endpoint_file": endpoint_path.name,
        "source_endpoint_sha256": _sha256(endpoint_path),
        "gp_model_binding": gp_model_binding,
        "endpoint_rows_read": len(canonical_rows),
        "selected_rows": len(selected), "source_groups": metadata["source_groups"],
        "logical_arms": metadata["logical_arms"], "evidence_statuses": sorted({row["evidence_status"] for row in selected}),
        "matrix": metadata["matrix"], "bootstrap_draws": args.draws, "bootstrap_seed": args.seed,
        "bootstrap_rng": "NumPy PCG64; canonical geometry and layout-fingerprint order",
        "paired_unit": "layout fingerprint sampled within geometry; identical sampled layout indices for paired arms",
        "aggregation_order": "conditions averaged within each layout; layouts averaged within geometry; 21 geometries equally weighted; family-specific relative reductions averaged equally; six solver effects averaged equally",
        "metrics": {"relocations": "relocation count (source R_raw)",
                    "relocations_per_container": "relocations divided by initial container count (source R_over_N)"},
        "forecast_pairing_check": "first_forecast_row_sha256 must match across arms within episode-condition-solver; this hash is not a full forecast-table hash",
        "status_boundary": "only accepted and explicitly failed-gate complete records are analyzed; partial, superseded, and smoke rows are excluded",
        "condition_effect_rows": len(condition_rows), "registered_contrast_rows": len(comparison_rows),
        "files": {"registered_contrasts.csv": "suite-level registered pure-family comparisons",
                  "condition_effects.csv": "each registered condition, with CLEAN retained separately",
                  "clean_noninferiority.json": "MAIN CLEAN ordinary intervals and registered noninferiority bounds" if clean_ni else None,
                  "clean_secondary.csv": "registered 063 CLEAN contrasts, separate from noisy primary" if clean_rows else None,
                  "interactions.json": "registered rollout/LB1 interaction with absolute relocation and secondary relative estimands" if interaction_details else None},
    }
    with (args.output_dir / "statistics_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (args.output_dir / "contrast_details.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summaries, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    if clean_ni is not None:
        with (args.output_dir / "clean_noninferiority.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(clean_ni, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    if interaction_details:
        with (args.output_dir / "interactions.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(interaction_details, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps({"status": manifest["status"], "suite": args.suite,
                      "selected_rows": len(selected), "condition_effect_rows": len(condition_rows),
                      "registered_contrast_rows": len(comparison_rows),
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
