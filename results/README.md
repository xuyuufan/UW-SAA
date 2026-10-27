# Results inventory

| Package | Scope | Status |
| --- | --- | --- |
| [Six-solver MAIN](best-h-main-20261028/README.md) | Preselected best H versus UWSAA; 41 conditions, 840 layouts, six solvers | Complete, 413,280 endpoints; ERI H10, all other fixed baselines H20 |
| [Current timing](best-h-main-20261028/tables/timing_current_six_solvers.pdf) | One layout per size, seven auxiliary conditions, same current runtime | Complete sample, 1,764 scored trajectories; not full-MAIN timing |
| Replacement-GP ablation/P0/multi-K | New runs using replacement GP | Canceled; not replaced with historical GP results |

The six-solver MAIN table averages all 41 conditions equally for its descriptive absolute-cost comparison. The primary noisy effect averages SS, TW and DP family-specific relative reductions equally. CLEAN, mixed-noise conditions and negative values remain separately available. Do not reuse historical MEAN6 or old GP ablation values as current results.

Earlier GP-only MAIN, historical Ablation and their audit/probe records are preserved in the [external archive](../reproducibility/HISTORICAL_ARCHIVE.md).
