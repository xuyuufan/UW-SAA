#!/usr/bin/env python3
"""Generate the figures for the accepted benchmark result tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
FIGURES = PACKAGE / "figures"

# Okabe-Ito colors for colorblind-safe categorical plots.
COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#777777",
}
SOLVER_COLORS = {
    "EM": COLORS["blue"],
    "GP": COLORS["orange"],
    "ERI": COLORS["green"],
    "EG": COLORS["vermillion"],
    "Chain": COLORS["sky"],
    "Minmax": COLORS["pink"],
}
SOLVER_MARKERS = {
    "EM": "o",
    "GP": "s",
    "ERI": "^",
    "EG": "D",
    "Chain": "v",
    "Minmax": "P",
}
SOLVER_ORDER = ["EM", "GP", "ERI", "EG", "Chain", "Minmax"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)


def read_csv(name: str) -> list[dict[str, str]]:
    path = DATA / name
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value: str) -> float:
    if value == "":
        raise ValueError("A required numeric field is empty.")
    return float(value)


def optional_number(value: str) -> float | None:
    return None if value == "" else float(value)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accepted_mean6() -> None:
    rows = read_csv("accepted_primary_contrasts_all_solvers.csv")
    mean6 = [row for row in rows if row["solver"] == "MEAN6"]
    labels_by_comparison = {
        "UWSAA_vs_NATIVE20": "MAIN: UWSAA vs NATIVE20",
        "UWSAA_vs_BEST_FIXED_H": "Ablation: best fixed horizon",
        "UWSAA_vs_NO_WINDOW_SHUFFLE_K8": "Ablation: no-window shuffle",
        "UWSAA_vs_SINGLE_WINDOW_B2": "Ablation: single window B=2",
        "UWSAA_vs_SINGLE_WINDOW_B5": "Ablation: single window B=5",
        "UWSAA_vs_SINGLE_WINDOW_B10": "Ablation: single window B=10",
        "UWSAA_vs_SINGLE_WINDOW_B20": "Ablation: single window B=20",
        "UWSAA_K8_vs_DETERMINISTIC_SINGLE": "UWSAA deterministic single",
    }
    order = list(labels_by_comparison)
    by_comparison = {row["comparison"]: row for row in mean6}
    if len(mean6) != 8 or set(by_comparison) != set(order):
        raise ValueError("Expected the eight accepted MEAN6 comparisons.")
    ordered = [by_comparison[key] for key in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    y = np.arange(len(ordered))
    estimates = np.array([number(row["relative_reduction_percent"]) for row in ordered])
    lows = np.array([number(row["ordinary_ci_low_percent"]) for row in ordered])
    highs = np.array([number(row["ordinary_ci_high_percent"]) for row in ordered])
    ax.errorbar(
        estimates,
        y,
        xerr=np.vstack((estimates - lows, highs - estimates)),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=3,
        elinewidth=1.7,
        markersize=5,
        label="Ordinary 95% CI",
        zorder=3,
    )

    for pos, row, estimate in zip(y, ordered, estimates):
        adj_low = optional_number(row["adjusted_ci_low_percent"])
        adj_high = optional_number(row["adjusted_ci_high_percent"])
        if adj_low is None or adj_high is None:
            continue
        ax.hlines(
            pos + 0.18,
            adj_low,
            adj_high,
            color=COLORS["vermillion"],
            linewidth=2.1,
            zorder=2,
        )
        ax.plot(
            estimate,
            pos + 0.18,
            marker="D",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=COLORS["vermillion"],
            linestyle="none",
            zorder=4,
        )

    ax.axvline(0, color=COLORS["gray"], linewidth=1.1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels_by_comparison[key] for key in order])
    ax.invert_yaxis()
    ax.set_xlabel("Relative reduction in relocations (%)  ·  positive favors the candidate")
    ax.set_title("Accepted comparisons on the public Caserta benchmark")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["blue"],
            marker="o",
            linestyle="-",
            markersize=5,
            label="Ordinary 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["vermillion"],
            marker="D",
            markerfacecolor="white",
            linestyle="-",
            markersize=4.5,
            label="Adjusted 97.5% CI",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["vermillion"],
            marker="D",
            markerfacecolor="white",
            linestyle="-",
            markersize=4.5,
            label="Adjusted 98.75% CI",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
    )
    fig.text(
        0.11,
        0.015,
        "Lower relocation count is better. B=5: ordinary 95% CI [-0.328%, -0.026%]; "
        "adjusted 98.75% CI [-0.370%, +0.012%] crosses zero.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.31, right=0.98, top=0.88, bottom=0.27)
    save_figure(fig, "uwsaa_accepted_mean6_comparisons")


def plot_p0_conditions() -> None:
    rows = read_csv("deterministic_single_condition_solver_results.csv")
    solver_rows = [row for row in rows if row["solver"] != "MEAN6"]
    condition_order = [
        "subset_shuffle|p=0.3|repeat=0",
        "subset_shuffle|p=0.3|repeat=1",
        "time_window_shuffle|B=5|repeat=0",
        "time_window_shuffle|B=5|repeat=1",
        "delayed_positions|p=0.3|repeat=0",
        "delayed_positions|p=0.3|repeat=1",
        "CLEAN",
    ]
    display = {
        "subset_shuffle|p=0.3|repeat=0": "subset shuffle, p=0.3, repeat=0",
        "subset_shuffle|p=0.3|repeat=1": "subset shuffle, p=0.3, repeat=1",
        "time_window_shuffle|B=5|repeat=0": "time-window shuffle, B=5, repeat=0",
        "time_window_shuffle|B=5|repeat=1": "time-window shuffle, B=5, repeat=1",
        "delayed_positions|p=0.3|repeat=0": "delayed positions, p=0.3, repeat=0",
        "delayed_positions|p=0.3|repeat=1": "delayed positions, p=0.3, repeat=1",
        "CLEAN": "CLEAN",
    }
    if len(solver_rows) != 42:
        raise ValueError("Expected 7 registered conditions × 6 solvers.")
    if {row["condition_id"] for row in solver_rows} != set(condition_order):
        raise ValueError("Deterministic-single condition table does not match the seven registered conditions.")

    centers = {
        "subset_shuffle|p=0.3|repeat=0": 7.0,
        "subset_shuffle|p=0.3|repeat=1": 6.0,
        "time_window_shuffle|B=5|repeat=0": 5.0,
        "time_window_shuffle|B=5|repeat=1": 4.0,
        "delayed_positions|p=0.3|repeat=0": 3.0,
        "delayed_positions|p=0.3|repeat=1": 2.0,
        "CLEAN": 0.0,
    }
    offsets = np.linspace(-0.28, 0.28, len(SOLVER_ORDER))
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    clean_center = centers["CLEAN"]
    ax.axhspan(
        clean_center - 0.45,
        clean_center + 0.45,
        color="#F1F1EE",
        zorder=0,
    )
    for condition in condition_order:
        condition_rows = [row for row in solver_rows if row["condition_id"] == condition]
        by_solver = {row["solver"]: row for row in condition_rows}
        if set(by_solver) != set(SOLVER_ORDER):
            raise ValueError(f"Incomplete solver coverage for {condition}.")
        for solver_index, solver in enumerate(SOLVER_ORDER):
            row = by_solver[solver]
            estimate = number(row["contrast_1_minus_uwsaa_k8_over_deterministic_single"]) * 100
            low = number(row["ci95_lower"]) * 100
            high = number(row["ci95_upper"]) * 100
            ax.errorbar(
                estimate,
                centers[condition] + offsets[solver_index],
                xerr=[[estimate - low], [high - estimate]],
                fmt=SOLVER_MARKERS[solver],
                color=SOLVER_COLORS[solver],
                ecolor=SOLVER_COLORS[solver],
                capsize=2,
                elinewidth=1.15,
                markersize=4.3,
                zorder=3,
            )

    ax.axvline(0, color=COLORS["gray"], linewidth=1.1, linestyle="--", zorder=1)
    ax.set_yticks([centers[key] for key in condition_order])
    ax.set_yticklabels([display[key] for key in condition_order])
    ax.set_ylim(-0.75, 7.55)
    ax.set_xlim(
        min(number(row["ci95_lower"]) for row in solver_rows) * 100 - 0.8,
        max(number(row["ci95_upper"]) for row in solver_rows) * 100 + 0.8,
    )
    ax.set_xlabel("1 − relocations(UWSAA K8) / relocations(deterministic single) (%)  ·  positive favors UWSAA K8")
    ax.set_title("UWSAA K8 vs deterministic single by registered condition and solver")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D(
            [0],
            [0],
            color=SOLVER_COLORS[solver],
            marker=SOLVER_MARKERS[solver],
            linestyle="none",
            markersize=5,
            label=solver,
        )
        for solver in SOLVER_ORDER
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    fig.text(
        0.13,
        0.018,
        "CLEAN is shown separately. No noninferiority test was registered for the deterministic-single study.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.9, bottom=0.18)
    save_figure(fig, "deterministic_single_condition_solver_effects")


def plot_descriptive_timing() -> None:
    rows = read_csv("uwsaa_rollout_vs_deterministic_timing.csv")
    solver_order = ["EM", "GP", "ERI", "EG", "Chain", "Minmax", "MEAN6"]
    arm_order = ["UWSAA_K8_ROLLOUT", "UWSAA_DETERMINISTIC_SINGLE"]
    arm_labels = {
        "UWSAA_K8_ROLLOUT": "UWSAA K8 rollout",
        "UWSAA_DETERMINISTIC_SINGLE": "UWSAA deterministic single",
    }
    arm_colors = {
        "UWSAA_K8_ROLLOUT": COLORS["blue"],
        "UWSAA_DETERMINISTIC_SINGLE": COLORS["vermillion"],
    }
    grouped = {(row["arm"], row["solver"]): row for row in rows}
    if len(rows) != 14 or any((arm, solver) not in grouped for arm in arm_order for solver in solver_order):
        raise ValueError("Expected two timing arms and six solvers plus MEAN6.")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True)
    y = np.arange(len(solver_order))
    arm_offsets = {"UWSAA_K8_ROLLOUT": -0.14, "UWSAA_DETERMINISTIC_SINGLE": 0.14}
    panels = [
        ("actor_select_mean_ms", "Actor selection mean (ms)"),
        ("complete_trajectory_wall_mean_ms", "Trajectory wall mean (ms)"),
    ]
    for ax, (field, title) in zip(axes, panels):
        for arm in arm_order:
            values = [
                number(grouped[(arm, solver)][field])
                for solver in solver_order
            ]
            ax.scatter(
                values,
                y + arm_offsets[arm],
                color=arm_colors[arm],
                marker="o",
                s=28,
                label=arm_labels[arm],
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel("Milliseconds")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(solver_order)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("Descriptive on-policy timing by solver", y=1.02, fontsize=11, fontweight="bold")
    fig.text(
        0.11,
        0.01,
        "Means only; no confidence intervals. P95 values and counts are in the CSV. "
        "This is not a matched-state speedup comparison.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.19, wspace=0.24)
    save_figure(fig, "uwsaa_descriptive_timing_by_solver")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_accepted_mean6()
    plot_p0_conditions()
    plot_descriptive_timing()
    print(f"Figures written to {FIGURES}")


if __name__ == "__main__":
    main()
