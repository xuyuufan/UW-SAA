# GP-H20 training pipeline

This pipeline uses the byte-preserved third-party Python reproduction's GP search, expression compiler, source terminals, and tree operators. It is not an exact reproduction of the original authors' implementation. The restricted H20 adapter changes only fitness simulation and public-rank mapping; the GP source engine remains unchanged.

## Frozen fit contract

Each formula is scored by total relocation count over the 840-row copied training bank, under the restricted-destination environment, clean forecast, and public relative-rank horizon 20. The accelerated evaluator is a sequential Numba CPU kernel using float64 with fastmath disabled. The public-frame bridge, source Yard transitions, source terminal computation and formula compiler remain correctness oracles. Source IDs are remapped to retrieval-order ranks for the registered H20 protocol; the evaluator uses the retrieval sequence only to construct those ranks and simulate training fitness. Runtime actors receive only a detached public frame and explicit horizon.

The bank has 21 geometry groups with 40 episodes each and SHA-256 `9dfaf66cce987d70cad2673089c27aea75af1a82e060d043f351218b665384c6`. It is unique and disjoint from the 840-row Caserta evaluation set. It is a training and selection bank, not an independent validation set. Formula selection is minimum training total across seeds 1–30, with ties resolved by lower seed. For the selected formula, fixed horizons `0, 2, 5, 10, 20` are scored on the same bank; ties choose the lower horizon. Caserta is never used for selection or tuning.

The 2026 paper reports 30 independent executions per scenario but does not prescribe seed IDs. The registered seeds are 1–30. Source `run_gp` uses population 1,000 plus 50 generations, or 51,000 fitness evaluations per seed; this is 1,000 more than the paper's Table 3 budget of 50,000. The source engine was kept unchanged, and the difference is recorded in the protocol.

No retired rule was warm-started. The 30-seed run and fixed-horizon selection are complete, and `configs/gp_rule.json` contains the activated hash-bound rule. Seed 5 was selected at 24,421 training relocations; H=20 was selected on the same bank. The five horizon totals and all receipts are in [the byte-preserved training artifact bundle](../../artifacts/gp_h20/20261024_formal_30seed/README.md). Selection used training data only and is not independent validation; Caserta was not read.

## Training environment and measured diagnostics

Use the training-only pins in [requirements-training.txt](requirements-training.txt) in a dedicated Python 3.12 environment. The validated combination is CPython 3.12.4, NumPy 1.26.4, Numba 0.59.1, and llvmlite 0.42.0. The public inference bundle keeps its separate NumPy dependency. Each child worker sets BLAS/OpenMP/Numba thread counts to one; GP seeds use independent `random.Random(seed)` instances.

The previous pure-Python scorer made a three-formula, 840-row timing project to 44.94 hours per seed. After replacing only the scorer with the checked Numba kernel, a bounded source-`run_gp` diagnostic used the full bank, population 40, eight generations (360 evaluations per seed), and seeds 501–508. Serial execution took 55.84 seconds; eight genuinely overlapping worker processes took 12.50 seconds, a 4.47x speedup. All eight paired seeds had identical formulas, fitness histories, and best scores. Peak summed worker RSS was about 1.06 GB and minimum available memory during that diagnostic was about 3.15 GB.

The bounded pilot is diagnostic only and does not bound full-search runtime. The complete formal run has now finished: wave 1 used 779.16 seconds and the remaining 22 seeds used 2,791.07 seconds, for 3,570.23 seconds of seed compute (59m30s). Elapsed wall time from start through final review was 7,847.67 seconds (2h10m48s), including monitoring pauses. All 30 receipts match the frozen 51,000-evaluation contract and training environment. These are training diagnostics, not benchmark-performance results.

## Completed run and next evidence gate

The formal run used seeds 1–30, 51,000 evaluations per seed, and the frozen Python 3.12.4 / NumPy 1.26.4 / Numba 0.59.1 / llvmlite 0.42.0 training environment. The selected rule uses seed 5 and H=20. An independent source-Yard re-score over all 840 training episodes reproduced the selected total of 24,421 relocations, and a complete one-row GP runner smoke plus strict identity-aware aggregate passed on the explicit training-bank path.

The activated rule is not a performance result. GP evaluation for MAIN, ablation, P0, and multi-K; canonical six-solver merging; registered statistics; MEAN6; and figures remain pending. Follow [the evaluation handoff](../../reproducibility/CLOUD_GP_MIGRATION_PROMPT.md) for that separate stage. Do not read Caserta for training or selection.
