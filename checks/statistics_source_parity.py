"""Compare the public paired estimator with its frozen source implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.protocol import SOLVERS
from reproducibility.statistics import _bootstrap_indices, _paired_effect


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture() -> tuple[list[dict], list[dict], dict[str, list[str]], list[tuple[str, list[str]]]]:
    groups = [("subset_shuffle", ["SS0", "SS1"]),
              ("time_window_shuffle", ["TW0"]),
              ("delayed_positions", ["DP0", "DP1"])]
    rows, source_rows = [], []
    for g in range(3):
        geometry = f"G{g + 1}"
        for j in range(4):
            fingerprint = hashlib.sha256(f"fixture:{g}:{j}".encode()).hexdigest()
            for family, conditions in groups:
                for condition_index, condition in enumerate(conditions):
                    for solver_index, solver in enumerate(SOLVERS):
                        base = 18 + g * 2.5 + j * 0.7 + solver_index * 0.31 + condition_index * 0.23
                        for arm, value in (("NATIVE20", base), ("UWSAA", base * (0.91 + g * 0.002))):
                            rows.append({
                                "geometry": geometry, "layout_fingerprint": fingerprint,
                                "condition_id": condition, "solver": solver, "arm": arm,
                                "relocations": value, "relocations_per_container": value / 24,
                            })
                            source_rows.append({
                                "geometry": geometry, "layout_fingerprint": fingerprint,
                                "condition_id": condition, "solver": solver, "method": arm,
                                "R_raw": value, "R_over_N": value / 24,
                            })
            for solver_index, solver in enumerate(SOLVERS):
                base = 18 + g * 2.5 + j * 0.7 + solver_index * 0.31
                for arm, value in (("NATIVE20", base), ("UWSAA", base * (0.995 + g * 0.001))):
                    rows.append({
                        "geometry": geometry, "layout_fingerprint": fingerprint,
                        "condition_id": "CLEAN", "solver": solver, "arm": arm,
                        "relocations": value, "relocations_per_container": value / 24,
                    })
                    source_rows.append({
                        "geometry": geometry, "layout_fingerprint": fingerprint,
                        "condition_id": "CLEAN", "solver": solver, "method": arm,
                        "R_raw": value, "R_over_N": value / 24,
                    })
    layouts = {f"G{g + 1}": [hashlib.sha256(f"fixture:{g}:{j}".encode()).hexdigest() for j in range(4)]
               for g in range(3)}
    return rows, source_rows, layouts, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "checks" / "statistics_source_parity_result.json")
    args = parser.parse_args()
    source_file = args.source_root / "uw_saa" / "experiments" / "uw_saa_followup.py"
    if not source_file.is_file():
        parser.error("--source-root does not contain the frozen paired estimator")
    sys.path.insert(0, str(args.source_root.resolve()))
    from uw_saa.experiments import uw_saa_followup as source

    public_rows, source_rows, layouts, groups = _fixture()
    seed, draws = 202610195903, 10_000
    source_indices = source._bootstrap_indices(layouts, seed)
    public_indices = _bootstrap_indices(public_rows, seed, draws)
    if any(not (source_indices[g] == public_indices[g]).all() for g in layouts):
        raise AssertionError("public bootstrap indices differ from the frozen source ordering")
    expected = source._paired_effect(
        source_rows, baseline="NATIVE20", candidate="UWSAA", condition_groups=groups,
        bootstrap_idx=source_indices, outcome_field="R_raw", two_sided_alpha=0.05,
    )
    actual = _paired_effect(
        public_rows, baseline="NATIVE20", candidate="UWSAA", condition_groups=groups,
        bootstrap_indices=public_indices, draws=draws, metric="relocations", alpha=0.05,
    )
    checks = [
        (expected["estimate_mean6_relative_reduction"], actual["estimate_mean6_relative_reduction"]),
        (expected["mean6_interval"][0], actual["mean6_interval"][0]),
        (expected["mean6_interval"][1], actual["mean6_interval"][1]),
    ]
    checks.extend((expected["solver_estimates"][solver], actual["solver_estimates"][solver]) for solver in SOLVERS)
    checks.extend((expected["family_estimates_by_solver"][solver][family],
                   actual["family_estimates_by_solver"][solver][family])
                  for solver in SOLVERS for family, _conditions in groups)
    if any(not math.isclose(left, right, rel_tol=0, abs_tol=1e-14) for left, right in checks):
        raise AssertionError("public point estimate, bootstrap interval, or family/solver effect differs from source")
    clean_groups = [("CLEAN", ["CLEAN"])]
    source_clean = [row for row in source_rows if row["condition_id"] == "CLEAN"]
    public_clean = [row for row in public_rows if row["condition_id"] == "CLEAN"]
    for tail in (0.05, 0.05 / 6.0):
        expected_lower = source._paired_effect(
            source_clean, baseline="NATIVE20", candidate="UWSAA", condition_groups=clean_groups,
            bootstrap_idx=source_indices, outcome_field="R_raw", one_sided_lower=tail,
        )
        actual_lower = _paired_effect(
            public_clean, baseline="NATIVE20", candidate="UWSAA", condition_groups=clean_groups,
            bootstrap_indices=public_indices, draws=draws, metric="relocations", one_sided_lower=tail,
        )
        one_sided_checks = [(expected_lower["mean6_interval"][0], actual_lower["mean6_one_sided_lower"])]
        one_sided_checks.extend((expected_lower["solver_intervals"][solver][0],
                                 actual_lower["solver_one_sided_lowers"][solver]) for solver in SOLVERS)
        if any(not math.isclose(left, right, rel_tol=0, abs_tol=1e-14)
               for left, right in one_sided_checks):
            raise AssertionError(f"public CLEAN one-sided lower bounds differ from source at quantile {tail}")
    receipt = {
        "schema": "uwsaa_statistics_source_parity_v1", "status": "PASS",
        "scope": "synthetic paired matrix; exact estimator and bootstrap-index comparison; no experiment run",
        "source_file_ids_sha256": {"registered_paired_macro_estimator": _sha(source_file)},
        "public_file_sha256": {"reproducibility/statistics.py": _sha(ROOT / "reproducibility" / "statistics.py")},
        "bootstrap_seed": seed, "bootstrap_draws": draws, "bootstrap_indices_match": True,
        "solver_effects_checked": len(SOLVERS), "family_solver_effects_checked": len(checks) - 3 - len(SOLVERS),
        "clean_one_sided_quantiles_checked": [0.05, 0.05 / 6.0],
        "estimate_mean6": actual["estimate_mean6_relative_reduction"],
        "mean6_interval": actual["mean6_interval"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
