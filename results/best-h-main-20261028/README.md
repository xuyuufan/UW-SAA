# Six-solver MAIN: preselected best H versus UWSAA

The fixed baselines are EM=20, GP=20, ERI=10, EG=20, Chain=20 and Minmax=20, frozen before this evaluation. The complete matrix contains 413,280 endpoints: 41 conditions × 840 layouts × six solvers × two arms. Each of 21 size classes has 40 layouts; size means initial height × number of stacks.

## Tables and figures

- `tables/table_main41_best_h_vs_uwsaa.*`: all 41 conditions averaged equally; paired best-H/UWSAA columns, smaller value bold within each algorithm pair. Display-precision ties are both bold.
- `tables/table_noisy_best_h_vs_uwsaa.*`: absolute means with equal SS/TW/DP family weights.
- `tables/table_clean_best_h_vs_uwsaa.*`: CLEAN only.
- Corresponding reduction tables, full-precision CSV, LaTeX, vector PDF/SVG and PNG are included. LaTeX requires booktabs.
- `figures/absolute_relocations_wide_main41.*`: wider absolute relocation plot.
- `figures/absolute_relocations_local_zoom_main41.*`: local x ranges by initial height. Axes differ between panels; use actual ticks when comparing gaps.
- `figures/family_effects_best_h_ci.*`: primary, CLEAN and mixed-noise family effects with pointwise paired 95% intervals.

## Full data and provenance

`data/six_solver_best_h_main_endpoints.csv.gz` contains all 413,280 keyed endpoint rows. `data/validated_best_h_main_arrays.npz` provides the same outcomes with solver/condition/size/layout/arm axes and container counts. `data/size_solver_effects.csv`, `overall_solver_effects.csv` and `all41_condition_size_means.csv` retain the detailed effects.

Existing accepted outcomes are reused without retuning. ERI H10 had 5,880 historical endpoints on seven conditions; the missing 28,560 endpoints were completed on 2026-10-28. All 147 historical probe cases matched. New full forecast hashes match the current GP MAIN forecasts for all 28,560 new endpoints. The other five best-H baselines are identical to their H20/Native20 reference. All UWSAA outcomes are unchanged. Compressed new ERI endpoint records and audit receipts are included.

The primary relative effect averages SS, TW and DP family-specific relative reductions equally. Each family averages conditions and layouts; overall results weight size classes equally before forming family ratios. This differs from averaging all 41 conditions or averaging size-specific percentages. Bootstrap intervals use 10,000 within-size layout resamples (seed 202610195903), paired across arms, conditions and solvers. They are pointwise, not simultaneous or multiplicity-adjusted.

## Timing sample

`tables/timing_current_six_solvers.*` and `timing_current_by_size.*` describe current-runtime timing of all six solvers. The sample has one fixed fingerprint-sorted layout per size, seven auxiliary conditions, 147 trajectories per arm per solver, and 1,764 scored trajectories total. One serial process interleaves solver/arm combinations in seeded random order within each layout-condition cell; 12 warmups are excluded. Every relocation outcome matches MAIN. Raw records, per-decision durations and the timing audit are in `data/current_six_solver_timing_sample.jsonl.gz` and `timing_sample_audit.json`.

Decision time averages measured selection calls; actor time sums selection calls per trajectory; wall time includes forecast/environment work. Ratios divide UWSAA by the best-H baseline. This is descriptive on-policy timing of a stratified sample, not a full-MAIN timing study or controlled-hardware speedup claim. There are no timing confidence intervals.

## Reproduce

From the repository root, run `python checks/verify_six_solver_results.py` to check the matrix, CSVs, statistics, timing records and raw new-baseline bindings. Run `python results/best-h-main-20261028/scripts/generate_figures.py --output-dir work/main-figures` to regenerate figures from the distributed data. These commands do not train or rerun trajectories.

Caserta is held out from UWSAA development; this is not OOD evidence. GP training selection is not independent validation. Detailed GP training-process artifacts are maintained outside the public repository; the executable frozen model and its runtime identity inputs remain distributed.
