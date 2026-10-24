# UW-SAA reproduction

This repository contains a CPU-capable Python reproduction of Uniform-Width Sample-Average Approximation (UW-SAA), selected solver adapters, a registered trajectory runner, and the GP migration workflow. The GP component uses a third-party Python reproduction attributed to Narek Sergeyan under the MIT license; it is not the original authors' implementation.

## Current evidence status

The earlier all-six-solver endpoint bundle and figures used a retired GP rule. Those files and the mixed raw endpoint payload have been removed from the active tree and preserved in a hash-verified archive. The replacement GP rule is trained and active. Its GP-only Caserta MAIN comparison against NATIVE20 has started. The local shard is running on registered conditions 1–14 (six SS and eight TW conditions; 23,520 endpoints). The non-overlapping cloud shard is assigned to conditions 15–41 (six DP, two overlay, 18 regime-mixture, and CLEAN; 45,360 endpoints) and starts after its cross-host probe passes. Together the shards cover 68,880 endpoints. This comparison does not rerun the other five solvers. Six-solver tables, figures, MEAN6, and the GP ablation/P0/multi-K matrices remain pending.

The previous non-GP endpoints can be reconstructed from the pinned pre-migration Git tree with [the recovery script](reproducibility/data/recover_non_gp_endpoints.py). It verifies the source index, all source parts, and the combined payload before filtering GP and MEAN6 rows. Its filtered output is written outside the checkout; the receipt in reproducibility/data/retained_non_gp_endpoints.receipt.json records the exact hashes and row counts. The five-solver payload is not a six-solver result and cannot be used to compute MEAN6.

## GP training contract

The replacement model was selected only on the separate 840-episode training bank in `experiments/gp_h20/data/train_bank.json`. The 840 Caserta layouts remain the sealed evaluation benchmark for UW-SAA and were not used for GP training, formula selection, fixed-horizon selection, or tuning.

The registered source engine ran seeds 1 through 30, with 51,000 fitness evaluations per seed (1,000 initial individuals plus 50 generations of 1,000). The 2026 paper reports 30 independent executions per scenario (p. 9, section 5.1) and means across 30 (p. 16), but does not publish seed IDs. The registered IDs are reproducible seeds, not recovered paper seeds. The source count of 51,000 differs from Table 3's reported 50,000 evaluations. Formula selection minimizes training-bank relocations, with lower seed breaking ties. Fixed-horizon selection compares H=0,2,5,10,20 on the same training bank, with lower horizon breaking ties.

Seed 5 was selected at 24,421 relocations; H=20 was the selected fixed horizon. The active rule is hash-bound to the vendored source, protocol, training bank, implementation, and selected seed receipt. This is training-bank selection, not independent validation. All 30 seeds used the pinned training environment (Python 3.12.4, NumPy 1.26.4, Numba 0.59.1, llvmlite 0.42.0). Seed compute totaled 3,570.23 seconds; elapsed wall time including monitoring pauses was 7,847.67 seconds. The full receipts and correctness checks are in [the formal training artifact bundle](artifacts/gp_h20/20261024_formal_30seed/README.md).

A bounded eight-worker diagnostic on a smaller population had matching serial/parallel formulas, histories, and scores (55.84 seconds serial, 12.50 seconds parallel). It is not a full-search performance result or runtime bound. No GP performance conclusion has been produced. The active model is not evidence of benchmark performance. The cloud shard must first reproduce the 252-endpoint cross-host probe and match each keyed result's relocations, N, decision/retrieval/reveal counts, forecast hash, and model identity. Probe rows are diagnostic and are not counted again in the formal cloud shard.

## Install and bounded checks

The public runtime uses requirements.txt with NumPy 2.4.4. GP training uses a separate Python 3.12.4 environment and [the training extras](experiments/gp_h20/requirements-training.txt), pinned to NumPy 1.26.4, Numba 0.59.1, and llvmlite 0.42.0. Keep these environments separate; the training pins do not alter the public runtime dependency set.

Use Python 3.12 and install the pinned public-runtime dependency:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -B checks/public_smoke.py
    python -B checks/environment_smoke.py
    python -B checks/trajectory_runner_smoke.py
    python -B checks/aggregate_validation_smoke.py
    python -B checks/gp_migration_smoke.py
    python -B checks/gp_acceleration_differential.py

These checks cover interfaces and bounded correctness. They do not run the registered performance matrices.

## Registered trajectory runner

List a suite before running it:

    python -B -m reproducibility.runner --suite main --list
    python -B -m reproducibility.runner --suite ablation --list
    python -B -m reproducibility.runner --suite p0 --list
    python -B -m reproducibility.runner --suite multik --list

A non-GP smoke trajectory does not load the GP rule:

    python -B -m reproducibility.runner --suite main --smoke --solver EM --condition CLEAN --output reproducibility/results/em-smoke.jsonl

Selecting GP requires a complete trained configs/gp_rule.json; the runner checks this before creating endpoint or manifest files. It records the rule, formula, source, protocol, training-bank, implementation, and seed-receipt identities in every GP endpoint and the run manifest.

The registered evaluation sets contain 41 MAIN conditions and seven conditions each for ablation, P0, and multi-K. The current GP-only MAIN comparison uses 840 layouts and two GP arms, UWSAA and NATIVE20. Ablation reuses matching MAIN UWSAA and NATIVE20 rows; P0 and multi-K reuse matching MAIN UWSAA rows. In multi-K analyses, the standard UWSAA rows are the K=8 rollout reference. The detailed pairing and merge boundary are in [the protocol](reproducibility/PROTOCOL.md).

## Statistics and results

The statistics command reads only the new canonical six-solver endpoint path under reproducibility/results/gp_migration/. It requires complete GP model bindings and rejects mixed GP identities or incomplete solver matrices before creating an output directory. The five-solver recovery payload is not a valid input to registered statistics.

The active [results inventory](results/uwsaa-benchmark-evaluation/README.md) explains what remains pending. It contains no retired GP values or figures. Caserta is used only as the registered evaluation benchmark; it remains held out from UW-SAA method development and from GP training or selection.
