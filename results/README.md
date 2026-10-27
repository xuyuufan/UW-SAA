# Current UW-SAA results

The active result package is [replacement-GP Caserta MAIN, 2026-10-24](gp-caserta-main-20261024/README.md). It contains all 68,880 formal GP endpoints in a verified ZIP, paired statistics, and newly regenerated PNG/PDF figures. Independent numerical review passed on 2026-10-25: all 100 published rows and 894 numeric fields agree within 1e-12 using a separately written implementation. The original `pending_review` CSV tags are preserved; this review does not claim external human peer review. See [the review record](gp-caserta-main-20261024/REVIEW.md).

| Study | Current status | Published output |
| --- | --- | --- |
| GP MAIN: UWSAA vs NATIVE20 | Complete: 41 conditions, 840 layouts, 68,880 endpoints | [Tables, figures, raw data](gp-caserta-main-20261024/README.md) |
| GP ablation | Canceled by project owner, 2026-10-25; no replacement run planned | None |
| GP deterministic single (P0) | Canceled with the ablation controls, 2026-10-25 | None |
| GP multi-K / LB1 | Canceled with the ablation controls, 2026-10-25 | None |
| GP selection-time / trajectory-cost studies | Replacement measurements pending | None |
| MAIN six-solver merge and MEAN6 | Identity-checked MAIN merge pending; canceled studies are excluded | None |

The retired GP code/configuration, mixed old endpoints, result tables, plots, plotting scripts, old verification receipts, and obsolete bootstrap artifacts are removed from the active tree. No old GP values are retained as current results. Other five-solver history can be reconstructed with [the pinned recovery workflow](../reproducibility/data/README.md); it must retain its original evidence status and cannot stand in for a new six-solver result.

Regenerate the two current figure pairs with `python results/gp-caserta-main-20261024/scripts/generate_figures.py --output-dir work/gp-figures`. Recompute and verify the published statistics with `python -B -m checks.verify_gp_results`. The [repository guide](../README.md) documents dependencies and the [protocol](../reproducibility/PROTOCOL.md) preserves the statistical contract.

The replacement-GP ablation cancellation applies to the ablation, P0 and multi-K/LB1 controls. The historical design and code are retained for reference; canceled runs are not prerequisites for reviewing MAIN. See [the MAIN review guide](gp-caserta-main-20261024/REVIEW.md).
