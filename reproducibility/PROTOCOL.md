# UW-SAA trajectory and statistics protocol

## Migration status

The replacement GP rule is trained and active. Its GP-only Caserta MAIN comparison against NATIVE20 is complete and published: local conditions 1–14 supply 23,520 endpoints and cloud conditions 15–41 supply 45,360, for all 68,880 formal endpoints. The 252-endpoint cross-host probe passed and is excluded from formal statistics. Tables, raw data and regenerated figures are in [the current results package](../results/gp-caserta-main-20261024/README.md). Independent numerical review passed on 2026-10-25 using a separately written statistical implementation; its record is in checks/gp_main_independent_review.json. The original `pending_review` CSV tags remain frozen source metadata. Implementation independence by the same Codex session is distinct from external human peer review. The retired GP implementation, mixed historical payload, old tables and figures are removed from the active tree. Replacement-GP ablation and the P0/multi-K controls were canceled by the project owner on 2026-10-25 and are not scheduled for rerun. Historical definitions below remain for reference and do not constitute a pending task. Timing measurements and the MAIN six-solver merge/MEAN6 remain separate unfinished work.

The GP runtime uses a third-party Python reproduction attributed to Narek Sergeyan under the MIT license; it is not the original authors' implementation. The source engine performs 51,000 fitness calls per seed (1,000 initial individuals plus 50 generations of 1,000), compared with Table 3's reported 50,000 evaluations. The replacement rule is selected over 30 seeds on the separate 840-row training bank. The best fixed horizon is selected on that same bank and is not independent validation. Caserta remains the sealed evaluation benchmark for UW-SAA and is never read for GP training or model selection.

## Registered evaluation matrices

The MAIN suite has 41 conditions: 20 pure-noise SS, TW, and DP conditions, two TW-then-SS overlay conditions, 18 regime-mixture conditions across nine profiles, and CLEAN. Every condition runs over 840 layouts, six solvers, and two arms (UWSAA, NATIVE20): 413,280 endpoints.

The ablation suite has seven conditions, 840 layouts, six solvers, and eight logical arms. Its UWSAA and NATIVE20 references are the matching MAIN rows; six ablation arms are newly evaluated. P0 has seven conditions and two logical arms (UWSAA, UWSAA_DETERMINISTIC_SINGLE), with UWSAA reused from MAIN. Multi-K has seven conditions and seven logical arms: the UWSAA K=8 rollout reference plus K=1/2/4/16 rollout and K=1/K=8 LB1 variants. Its standard UWSAA rows are reused from MAIN for matched conditions.

The current MAIN comparison evaluates only GP, in the UWSAA and NATIVE20 arms, over all 840 Caserta layouts. Registered condition indices 1–14 (SS and TW) form the local shard; indices 15–41 (DP, overlay, regime-mixture, and CLEAN) form the cloud shard. These partitions are disjoint and cover all 41 MAIN conditions. Each condition has 840 layouts × two arms. Every endpoint must bind to the same frozen GP rule identity. The separate replacement-GP ablation, P0, and multi-K evaluations are canceled; the other five solvers and six-solver statistics are outside this run. Join or combine endpoints by layout_fingerprint, condition, solver, and arm; do not rely on episode_id alone.

For every forecast condition, paired GP arms receive the same forecast table and must have matching forecast_table_sha256 values. The runner records a full forecast-table hash. The retained historical non-GP CSV stores only first_forecast_row_sha256, a source-defined hash over the first forecast row. A future six-solver merge must recompute that source-compatible first-row hash from the runner's forecast table and verify it against the retained non-GP row. A full-table digest must never be placed in the first-row field.

## Training boundary

experiments/gp_h20/protocol.json binds the search source, 840-row training bank, evaluator, and training-only selection criteria. The 2026 paper reports 30 independent executions per scenario (p. 9, section 5.1) and means across 30 executions (p. 16), but does not publish seed IDs; this package registers reproducible seeds 1–30. The unchanged source engine performs 51,000 fitness calls per seed, compared with Table 3's 50,000. The selected formula minimizes total training relocations; ties choose the lower seed. Fixed horizons are 0, 2, 5, 10, and 20, scored on the same training bank; ties choose the lower horizon. These are selection scores, not independent validation. All 30 complete seed receipts and five horizon totals are present. Seed 5 was selected at 24,421 training relocations; H=20 was selected. The selection is on the training bank, not independent validation. Full receipts and verification files are in artifacts/gp_h20/20261024_formal_30seed. The selected rule is active. GP-only MAIN is complete. The replacement-GP ablation/P0/multi-K follow-ups are canceled; timing and MAIN six-solver reporting remain unfinished.

Training used the separate, pinned Python 3.12 environment in experiments/gp_h20/requirements-training.txt, distinct from the public inference requirements. All 30 seeds completed within the registered 24-hour total budget. Seed compute was 3,570.23 seconds; elapsed wall time including monitoring pauses was 7,847.67 seconds. The bounded small-population pilot remains diagnostic and does not bound full-seed runtime. See experiments/gp_h20/CLOUD_TASK.md for the completed run record.

## Forecast and policy semantics

The runner uses data/caserta_table2_840.json, which contains 21 geometries and 40 layouts per geometry. Layout fingerprints are label-invariant and provide the stable join key across old canonical rows and new endpoint JSONL.

The MAIN suite uses stable per-layout random seeds for registered pure conditions. Overlay applies time-window shuffle first, then subset shuffle. The P0 control is a deterministic single completion and is not the K=1 rollout arm. Multi-K samples use deterministic public-state seeds and nested sets; K8 rollout is the standard UWSAA reference. Candidate actions use the detached public frame. Evaluator-side retrieval order stays with the environment and forecast provider.

Each trajectory has a 120-second wall-clock cap. Failed trajectories remain endpoint rows and make a run incomplete. The runner never overwrites output paths. A GP run fails before output creation while the model config is pending or its source/protocol/training fingerprints do not match.

## Aggregate gates

reproducibility.aggregate validates JSONL endpoints against their run manifest. For GP it requires every endpoint's model identity to match a trained, hash-bound model record in that manifest. Non-GP rows must not carry a GP identity.

reproducibility.statistics accepts only the new canonical six-solver CSV produced after GP training and evaluation work. It requires all six solver cells, complete GP formula/source/protocol/training/implementation/seed-receipt bindings, and one consistent GP identity. Five-solver recovery data, a mixture of old/new GP rows, and incomplete matrices fail before statistics output is created. No MEAN6 is reported from five solvers.

## Registered six-solver statistical procedure (unchanged; merge pending)


The primary outcome is `relocations`, a count where lower is better. `relocations_per_container` is the count divided by the initial container count. Older outputs used `R_raw` and `R_over_N`; those names are retained only in source mappings and are normalized to the public metric names. Positive relative reductions favor the candidate named in a comparison.

MAIN's pure-noise primary groups contain all registered conditions within each family: six SS conditions, eight TW conditions, and six DP conditions. Overlay, regime-mixture, and CLEAN estimates remain separate condition-level results. The seven-condition ablation and P0 suites use the two repeats for each noisy family and keep CLEAN separate.

MAIN CLEAN also has a registered noninferiority assessment alongside its ordinary two-sided 95% interval. The estimand is the relative reduction in macro mean `relocations`, `(NATIVE20 - UWSAA) / NATIVE20`, with a 0.01 relative-loss margin. The MEAN6 one-sided 95% lower bootstrap bound must be strictly above -0.01. The all-six-solver rule uses one-sided lower quantile 0.05/6 for each solver and passes only when every lower bound is strictly above -0.01. These are the existing registered criteria, not added gates. The former result flags came from the retired GP rule and are withdrawn with those results. Recompute the bounds and flags with the replacement GP before drawing any conclusion; the `main` statistics run writes them to `clean_noninferiority.json`.

Within each geometry and solver, the analysis first averages all registered conditions within a family at each layout, including the family's intensity or width levels and repeats, then averages layouts within that geometry. It equally weights the 21 geometry means to form baseline and candidate macro means, computes the family-specific relative reduction only after those means, averages the three family reductions equally, then equally averages the six solver effects. It does not pool costs across families before forming one ratio. The MEAN6 result is a macro-average across solver effects, not six repeated observations.

The paired bootstrap uses 10,000 draws with NumPy PCG64 and seed `202610195903`. It samples 40 layout fingerprints with replacement within each geometry, using canonical sorted geometry and fingerprint order. The same sampled layout indices are applied to every arm and condition in a pair. Confidence intervals are percentile intervals over the six-solver mean and each solver estimate. Undefined nonpositive baseline denominators remain flagged; the calculation does not renormalize family or solver weights. Registered corrected intervals use the alpha recorded with each comparison.

For the multi-K follow-up, the seven-row-per-solver primary table consists of four K=1/2/4/16 rollout-versus-K8 contrasts, K8 rollout versus K8 LB1, K1 rollout versus K1 LB1, and the absolute rollout/LB1 interaction. The four P1 comparisons use two-sided alpha 0.0125. P3 and P4 use two-sided alpha 1/60. The absolute interaction is measured in relocations:

```text
(P4_K1_LB1 - P1_K1_ROLLOUT) - (P3_K8_LB1 - K8_ROLLOUT)
```

Positive interaction values mean the LB1 relocation penalty is larger at K1 than K8; a negative value means the rollout advantage is larger at K8. Its ordinary 95% and corrected 98.33% intervals are primary. The relative interaction is dimensionless and secondary. CLEAN comparisons and interactions are reported separately from the noisy primary.


## GP-only MAIN comparison and analysis

The current Caserta comparison evaluates the active GP policy in two arms only: UWSAA and NATIVE20. Registry indices 1–14 (six subset-shuffle and eight time-window conditions) form the local shard; indices 15–41 (six delayed-position, two overlay, 18 regime-mixture, and CLEAN conditions) form the cloud shard. Each condition has 840 layouts and two arms, yielding 23,520 local and 45,360 cloud endpoints. The non-overlapping shards cover 68,880 GP endpoints. The registered model formula SHA-256 is `e3cbbd12defb760be3e6afe57df7004980b1660be73b6f214bad1817473f9845`; the Caserta dataset SHA-256 is `bb471dc334ad186523bcbd436bbe103dd81e517554999d35132a5e0692c98448`.

Before its formal partition, the cloud host reproduces the 252-endpoint cross-host probe: one layout per each of 21 geometries, six representative conditions, and both arms. It matches each row against the published local reference by condition, layout fingerprint, and arm, including relocations, N, decisions, retrievals, reveals, forecast-table hash, and GP model identity. Probe rows are diagnostic and are excluded from the formal cloud endpoint shard. Any mismatch stops the cloud run.

The GP-only analysis is separate from `reproducibility.statistics` and never emits MEAN6. Its primary metric is `relocations`; `relocations_per_container` is secondary. It reports SS, TW, and DP family effects separately, with each family effect formed after averaging conditions within layout, layouts within geometry, and the 21 geometry means equally. The combined effect is the equal mean of the three family-specific relative reductions, `(NATIVE20 - UWSAA) / NATIVE20`. Overlay, regime-mixture, and CLEAN remain condition-level results. Paired 95% percentile intervals use 10,000 geometry-stratified layout-cluster draws from NumPy PCG64, seed `202610195903`; each sampled layout index is shared by both arms and all conditions. CLEAN's GP-only corrected lower quantile is descriptive and is not the registered all-six-solver noninferiority gate. The complete 41-condition input has passed integrity validation. The original output retains its frozen `pending_review` metadata; the separate independent-implementation numerical review has passed. This review does not claim external human peer review.

## Non-GP endpoint recovery

The mixed historical endpoint parts are no longer in the active tree. reproducibility/data/recover_non_gp_endpoints.py reads their exact bytes from pinned Git commit a39d0ac0e454f48d591d573868afa21531db34e7, verifies the historical index and source payload, removes GP and MEAN6 records, then writes a filtered gzip and standard parts index outside the checkout. See reproducibility/data/README.md and its receipt for source hashes, retained solver counts, evidence status, and coverage.

## Reproducibility

The runner, condition registry, and statistical aggregation are in reproducibility/runner.py, reproducibility/protocol.py, and reproducibility/statistics.py. The public file inventory is generated from the explicit reproducibility/published_files.json list. The explicit inventory includes the completed GP MAIN tables, figures, raw ZIP and figure script. Scratch files and newly generated runs are excluded.
