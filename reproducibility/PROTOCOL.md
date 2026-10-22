# UWSAA trajectory and statistics protocol

This document describes the public trajectory endpoint runner and the frozen registered comparisons. It separates previously executed results from the bounded correctness checks run while packaging this repository. The packaging checks did not execute a full performance matrix.

## Inputs and forecast conditions

The runner uses `data/caserta_table2_840.json`, the frozen runtime files under `configs/`, and the policy/solver modules in this repository. The 840 layouts form 21 geometries with 40 layouts each. A layout fingerprint is computed from the ranked initial stacks, not the original container labels. The same fingerprint identifies paired conditions and solver arms.

The MAIN suite contains 41 conditions: 20 pure-noise SS, TW, and DP conditions, two TW-then-SS overlay conditions, 18 regime-mixture conditions across nine registered profiles, and CLEAN. Pure conditions use two registered repeats. SS and DP use probabilities 0.1, 0.3, and 0.5; TW uses window widths 2, 5, 10, and 20. Each pure family/repeat combination is derived from a stable layout-specific random seed. Overlay applies the time-window shuffle first, then subset shuffle. Regime mixtures use largest-remainder allocation for each layout and the profile weights in `reproducibility/protocol.py`.

The ablation, deterministic-single, and multi-K suites use seven conditions: SS with probability 0.3 at repeats 0 and 1; TW with width 5 at repeats 0 and 1; DP with probability 0.3 at repeats 0 and 1; and CLEAN. In these IDs, `repeat` names independent registered random draws. It is not a relocation count or an arm label. P0 is a deterministic single completion with no random resampling; it must not be described as the K=1 rollout arm. Its scenario orders visible future ranks first, then the anonymous forecast tail in sorted stack/tier order, and assigns the current target rank 1 without RNG or shuffling.

For every forecast condition, all arms paired on a layout, condition, and solver receive the same forecast table. The runner records a SHA-256 digest of the complete table in its endpoint. The canonical historical endpoint file preserves the original first-row digest in `first_forecast_row_sha256`; that digest covers only the first forecast row and is not a full-table hash.

UWSAA's sampler uses two synthetic scenarios for each of four window widths, with equal total weight per width. Identical candidate sequences are merged, and each remaining sequence is scored using the fixed Minmax continuation. A historical random namespace is retained as an exact compatibility token so registered scenarios remain stable; it does not name the public method. Candidate actions use the detached public frame, and the evaluator-side retrieval sequence is not passed to a policy.

The multi-K arms use deterministic public-state seeds to order widths and then repeats. The sampled sets are nested: K1 is a subset of K2, K2 of K4, and K4 of K8. K8 contains repeats 0 and 1 for each width; K16 contains repeats 0 through 3 for each width. K1 is one equally weighted scenario from the eight K8 scenarios. The no-window control keeps eight anonymous-tail draws but omits within-window permutations. Each single-width control keeps eight draws at its selected width. The rollout arms commit the selected candidate sequence and apply the configured rollout continuation; LB1 evaluates the complete candidate moves, then counts each stack box whose rank exceeds a smaller rank below it. LB1 adds no further rollout after the candidate moves.

The nested-set sampler retains its historical random-seed compatibility token so the registered scenario assignments remain fixed. That token identifies a random-number contract, not the public method.

## Registered matrices and arms

The complete default MAIN matrix is 41 conditions × 840 layouts × six solvers × two arms (`UWSAA`, `NATIVE20`), or 413,280 trajectory endpoints.

The complete ABLATION matrix is seven conditions × 840 layouts × six solvers × eight logical arms (`UWSAA`, `NATIVE20`, `UWSAA_NO_WINDOW_SHUFFLE_K8`, `UWSAA_SINGLE_WINDOW_B2`, `UWSAA_SINGLE_WINDOW_B5`, `UWSAA_SINGLE_WINDOW_B10`, `UWSAA_SINGLE_WINDOW_B20`, and `UWSAA_BEST_FIXED_H`), or 282,240 endpoints. The two common reference arms are represented once in the canonical endpoint file and mapped from MAIN rather than duplicated.

P0 uses `UWSAA` and `UWSAA_DETERMINISTIC_SINGLE`. The multi-K suite uses `UWSAA` plus `UWSAA_K1_ROLLOUT`, `UWSAA_K2_ROLLOUT`, `UWSAA_K4_ROLLOUT`, `UWSAA_K16_ROLLOUT`, `UWSAA_K1_LB1`, and `UWSAA_K8_LB1`. The P1 and LB1 comparisons are paired on the same seven conditions, 840 layouts, and six solvers. K8 rollout rows are referenced once from their executed baseline run.

Each trajectory can run for at most 120 seconds. A failure row retains the episode-condition-solver-arm identity and available endpoint audit counts; the runner continues the registered matrix and exits nonzero when any row fails. Existing output files are never overwritten. Successful runs must complete every retrieval and the provider must reveal each forecast target exactly once.

The paired semantic key is the episode ID, label-invariant layout fingerprint, full condition parameters and repeat, solver, and arm. The aggregate verifier reconstructs that key from row fields and the run manifest, checks the condition and episode metadata, validates the complete matrix, and requires identical forecast hashes across arms. It rejects failed/incomplete rows, duplicate pairs, missing arms, invalid relocation accounting, or mislabeled endpoint rows.

## Statistical procedure

The primary outcome is `relocations`, a count where lower is better. `relocations_per_container` is the count divided by the initial container count. Older outputs used `R_raw` and `R_over_N`; those names are retained only in source mappings and are normalized to the public metric names. Positive relative reductions favor the candidate named in a comparison.

MAIN's pure-noise primary groups contain all registered conditions within each family: six SS conditions, eight TW conditions, and six DP conditions. Overlay, regime-mixture, and CLEAN estimates remain separate condition-level results. The seven-condition ablation and P0 suites use the two repeats for each noisy family and keep CLEAN separate.

MAIN CLEAN also has a registered noninferiority assessment alongside its ordinary two-sided 95% interval. The estimand is the relative reduction in macro mean `relocations`, `(NATIVE20 - UWSAA) / NATIVE20`, with a 0.01 relative-loss margin. The MEAN6 one-sided 95% lower bootstrap bound must be strictly above -0.01. The all-six-solver rule uses one-sided lower quantile 0.05/6 for each solver and passes only when every lower bound is strictly above -0.01. These are the existing registered criteria, not added gates. The accepted result passed MEAN6 and failed the all-six rule; the recomputed bounds and flags are written to `clean_noninferiority.json` by the `main` statistics run.

Within each geometry and solver, the analysis first averages all registered conditions within a family at each layout, including the family's intensity or width levels and repeats, then averages layouts within that geometry. It equally weights the 21 geometry means to form baseline and candidate macro means, computes the family-specific relative reduction only after those means, averages the three family reductions equally, then equally averages the six solver effects. It does not pool costs across families before forming one ratio. The MEAN6 result is a macro-average across solver effects, not six repeated observations.

The paired bootstrap uses 10,000 draws with NumPy PCG64 and seed `202610195903`. It samples 40 layout fingerprints with replacement within each geometry, using canonical sorted geometry and fingerprint order. The same sampled layout indices are applied to every arm and condition in a pair. Confidence intervals are percentile intervals over the six-solver mean and each solver estimate. Undefined nonpositive baseline denominators remain flagged; the calculation does not renormalize family or solver weights. Registered corrected intervals use the alpha recorded with each comparison.

For the multi-K follow-up, the seven-row-per-solver primary table consists of four K=1/2/4/16 rollout-versus-K8 contrasts, K8 rollout versus K8 LB1, K1 rollout versus K1 LB1, and the absolute rollout/LB1 interaction. The four P1 comparisons use two-sided alpha 0.0125. P3 and P4 use two-sided alpha 1/60. The absolute interaction is measured in relocations:

```text
(P4_K1_LB1 - P1_K1_ROLLOUT) - (P3_K8_LB1 - K8_ROLLOUT)
```

Positive interaction values mean the LB1 relocation penalty is larger at K1 than K8; a negative value means the rollout advantage is larger at K8. Its ordinary 95% and corrected 98.33% intervals are primary. The relative interaction is dimensionless and secondary. CLEAN comparisons and interactions are reported separately from the noisy primary.

## Endpoint records and evidence limits

The public archive contains provenance-indexed endpoint records with terminal outcomes, source-row hashes, first-forecast-row hashes, source mapping, run/evidence status, solver, arm, condition, geometry, and container count. It retains separately labelled historical records, including records whose episodes overlap across earlier outputs. Accepted matrices use complete deduplicated endpoint references; common baselines are stored once and reused through membership mappings. Accepted results, executed negative results, development and follow-up records, partial records, failed-gate records, and correctness-only checks have distinct status labels. Partial or superseded rows are not silently pooled into accepted statistics.

Terminal rows do not contain every intermediate action or original recovery log. Partial endpoint prefixes are retained and labeled with their historical status, but they do not enter complete accepted matrices. Full per-decision traces are not part of the public distribution, so this package supports reproduction of terminal endpoints and registered endpoint statistics, not reconstruction of every action. Published selection-time summaries are descriptive and do not reproduce hardware load or establish a causal speedup. The existing result figures are descriptive or paired-bootstrap summaries of the included endpoints. Caserta is the sealed evaluation benchmark for UW-SAA, held out from UW-SAA method development; its public availability and use in other project methods do not change that method-specific evaluation role.

The full condition/arm registry is executable in `reproducibility/protocol.py`. Statistical aggregation and reports are in `reproducibility/aggregate.py` and `reproducibility/statistics.py`. The data source-to-public mapping is documented in `reproducibility/data/README.md` and `reproducibility/data_inventory.json`.
