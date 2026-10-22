#!/usr/bin/env python3
"""Generate UWSAA scenario-ablation and descriptive timing figures from published CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
FIGURES = PACKAGE / "figures"
SOLVER_ORDER = ["EM", "GP", "ERI", "EG", "Chain", "Minmax", "MEAN6"]
SOLVERS = ["EM", "GP", "ERI", "EG", "Chain", "Minmax"]
COLORS = {"EM": "#0072B2", "GP": "#E69F00", "ERI": "#009E73", "EG": "#D55E00", "Chain": "#56B4E9", "Minmax": "#CC79A7", "MEAN6": "#333333"}
PAIR_LABELS = [
    "UWSAA K=8 rollout vs K=1 rollout", "UWSAA K=8 rollout vs K=2 rollout",
    "UWSAA K=8 rollout vs K=4 rollout", "UWSAA K=8 rollout vs K=16 rollout",
    "UWSAA K=8 LB1 vs K=8 rollout", "UWSAA K=1 LB1 vs K=1 rollout",
]
ARM_ORDER = [
    "NATIVE20", "UWSAA_BEST_FIXED_H", "UWSAA_K1_ROLLOUT", "UWSAA_K2_ROLLOUT",
    "UWSAA_K4_ROLLOUT", "UWSAA_K8_ROLLOUT", "UWSAA_K16_ROLLOUT", "UWSAA_K8_LB1", "UWSAA_K1_LB1",
]
ARM_LABEL = {
    "NATIVE20": "Native20", "UWSAA_BEST_FIXED_H": "UWSAA best fixed horizon",
    "UWSAA_K1_ROLLOUT": "UWSAA K=1 rollout", "UWSAA_K2_ROLLOUT": "UWSAA K=2 rollout",
    "UWSAA_K4_ROLLOUT": "UWSAA K=4 rollout", "UWSAA_K8_ROLLOUT": "UWSAA K=8 rollout",
    "UWSAA_K16_ROLLOUT": "UWSAA K=16 rollout", "UWSAA_K8_LB1": "UWSAA K=8 LB1",
    "UWSAA_K1_LB1": "UWSAA K=1 LB1",
}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18, "savefig.bbox": "tight"})


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def limits(rows: list[dict[str, str]], scale: float) -> tuple[float, float]:
    endpoints = [scale*f(r, k) for r in rows for k in ("ordinary_ci_lower", "ordinary_ci_upper", "adjusted_ci_lower", "adjusted_ci_upper")]
    lo, hi = min(endpoints), max(endpoints)
    pad = max((hi-lo)*0.08, 0.05 if scale == 1 else 0.15)
    return lo-pad, hi+pad


def plot_contrast_panel(ax: plt.Axes, rows: list[dict[str, str]], title: str, scale: float, x_label: str) -> None:
    by_solver = {r["solver"]: r for r in rows}
    if set(by_solver) != set(SOLVER_ORDER):
        raise ValueError(f"Expected six solvers plus MEAN6 for {title}; got {sorted(by_solver)}")
    adjusted_levels = {round(100*f(r, "adjusted_ci_level"), 2) for r in rows}
    if len(adjusted_levels) != 1:
        raise ValueError(f"Expected one adjusted CI level per panel for {title}: {adjusted_levels}")
    adjusted_label = adjusted_levels.pop()
    for y, solver in enumerate(SOLVER_ORDER):
        row = by_solver[solver]
        estimate = scale*f(row, "estimate")
        low = scale*f(row, "ordinary_ci_lower")
        high = scale*f(row, "ordinary_ci_upper")
        adj_low = scale*f(row, "adjusted_ci_lower")
        adj_high = scale*f(row, "adjusted_ci_upper")
        color = COLORS[solver]
        ax.errorbar(estimate, y - 0.08, xerr=[[estimate-low], [high-estimate]], fmt="o", color=color, ecolor=color, capsize=2.1, elinewidth=1.2, markersize=3.8, zorder=3)
        ax.hlines(y + 0.14, adj_low, adj_high, color=color, linewidth=1.4, zorder=2)
        ax.plot(estimate, y + 0.14, marker="D", markersize=3.1, markerfacecolor="white", markeredgecolor=color, linestyle="none", zorder=4)
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(np.arange(len(SOLVER_ORDER)))
    ax.set_yticklabels(SOLVER_ORDER)
    ax.invert_yaxis()
    ax.set_title(f"{title}\nBonferroni-adjusted {adjusted_label:.2f}% CI", fontsize=8.5)
    ax.set_xlabel(x_label)
    ax.grid(axis="y", visible=False)


def plot_scenario() -> None:
    rows = read_csv("scenario_ablation_primary_contrasts_all_solvers.csv")
    if len(rows) != 49:
        raise ValueError(f"Expected 49 registered rows, found {len(rows)}")
    pair_rows = [r for r in rows if r["contrast_label"] in PAIR_LABELS]
    pair_bounds = limits(pair_rows, 100.0)
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.0), sharex=True)
    for ax, label in zip(axes.flat, PAIR_LABELS):
        panel = [r for r in pair_rows if r["contrast_label"] == label]
        plot_contrast_panel(ax, panel, label, 100.0, "Relative reduction in relocations (%)")
        ax.set_xlim(*pair_bounds)
    fig.suptitle("UWSAA scenario and rollout ablations", fontsize=13, fontweight="bold", y=1.005)
    handles = [Line2D([0], [0], color="#555555", marker="o", markersize=4, linewidth=1.2, label="Ordinary 95% CI"), Line2D([0], [0], color="#555555", marker="D", markerfacecolor="white", markersize=4, linewidth=1.4, label="Bonferroni-adjusted CI (panel-specific level)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.035, "Positive values favor the candidate (fewer relocations); all seven registered solver summaries are shown.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    save(fig, "uwsaa_scenario_ablation_pairwise")

    interaction = [r for r in rows if r["unit"] == "relocations"]
    if len(interaction) != 7:
        raise ValueError(f"Expected seven absolute-interaction rows in relocations; found {len(interaction)}")
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    plot_contrast_panel(ax, interaction, "Absolute interaction: LB1 effect at K=1 minus K=8", 1.0, "Absolute interaction (relocations)")
    ax.set_xlim(*limits(interaction, 1.0))
    ax.text(0.01, -0.30, interaction[0]["direction"], transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.suptitle("UWSAA absolute rollout interaction", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "uwsaa_scenario_ablation_absolute_interaction")


def plot_cost() -> None:
    rows = read_csv("uwsaa_cost_summary_by_solver_arm.csv")
    if len(rows) != 54:
        raise ValueError(f"Expected 54 solver-arm timing summaries, found {len(rows)}")
    by = {(r["solver"], r["arm"]): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.6), sharex=True)
    for ax, solver in zip(axes.flat, SOLVERS):
        values = [f(by[(solver, arm)], "full_wall_mean_multiple_vs_native20") for arm in ARM_ORDER]
        y = np.arange(len(ARM_ORDER))
        ax.barh(y, values, color="#4C78A8", alpha=0.9)
        ax.axvline(1.0, color="#555555", linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(solver)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean wall time ratio vs Native20")
    fig.suptitle("Descriptive on-policy wall-time ratios by solver and arm", fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.015, "Per-trajectory means; no confidence intervals or hardware-load control. Ratios describe this recorded run only.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    save(fig, "uwsaa_cost_wall_time_ratio")


def main() -> None:
    plot_scenario()
    plot_cost()
    print("Generated scenario and descriptive wall-time figures from published CSVs.")


if __name__ == "__main__":
    main()
