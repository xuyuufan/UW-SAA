# UW-SAA endpoint data

`uwsaa_trajectory_endpoints.csv.gz` is the canonical normalized endpoint table. Its final gzip SHA-256 is `82f53e12cb4a7be109a504485267373a0032c45a5c0964a6153f07ec61616af4`, compressed size 174,138,724 bytes, and it contains 1,722,662 provenance-indexed endpoint records, including separately labelled overlapping historical and partial records. Repository distribution uses four ordered binary parts under 50 MiB each. Reassemble and verify them with the standard-library helper:

```text
python reproducibility/data/reassemble_endpoints.py
```

By default, the helper writes the verified gzip to the operating system temporary directory. You can choose a different output with `--output PATH`. The parts index records each part's byte count and SHA-256 and the assembled payload SHA. Do not decompress each part separately; concatenate them in index order, then decompress the single resulting gzip CSV.

The table preserves every source endpoint column across the source union, normalized public headers and labels, source row number, source SHA-256, neutral source ID, and a neutral episode ID. Numeric source cells remain textual values. `geometry` and `layout_fingerprint` are preserved unchanged because they determine the registered geometry grouping and bootstrap order. `first_forecast_row_sha256` is the source hash of the first forecast row; it is not a full forecast-table hash.

## Public names

The source metric `R_raw` is named `relocations`; `R_over_N` is named `relocations_per_container`; `N` is `num_containers`. The `r=0/1` suffix in older condition IDs denoted the repeat replicate, and is published as `repeat=0/1`. It is independent of the relocation metric. The historical `TW` family is published as `time_window_shuffle`; overlay and regime conditions use their registered full public IDs. Original byte-level artifacts and old field labels remain in the local archive; `data_inventory.json` records the minimal compatibility mapping and source SHA-to-public-payload mapping without private source paths.

## Coverage and evidence status

| Study group | Public rows | Status and interpretation |
| --- | ---: | --- |
| `caserta_main` | 413,280 | Accepted: 41 conditions × 6 solvers × 2 arms × 840 layouts |
| `caserta_ablation` | 211,680 | Accepted new arms: 7 conditions × 6 solvers × 6 arms × 840; 70,560 MAIN reference rows are shared via membership |
| `deterministic_single` | 35,280 | Accepted: 7 conditions × 6 solvers × 840; separate from K=1 rollout; 35,280 MAIN UWSAA reference rows are shared |
| `scenario_ablation` | 246,960 | Accepted: six newly executed arms (211,680) plus reused K=8 rollout reference (35,280) |
| `followup_a` | 28,800 | Accepted; used the hash-bound 40-layout synthetic development/evaluation bank, not an independent test or OOD set |
| `followup_bdev` | 24,000 | Accepted development follow-up |
| `followup_btest` | 39,360 | Accepted follow-up BTEST |
| `followup_c` | 39,360 | Accepted follow-up C |
| `speedup_validation` | 756 | Retained negative evidence: registered speed gate failed |
| `dev100` | 7,560 | Retained negative evidence: registered clean gate failed |
| `correctness_smoke` | 636 | Correctness/profile smoke rows; not performance evidence |
| `historical_partial` | 674,990 | Earlier or interrupted partial matrices; not additional accepted full experiments |

The table stores shared accepted main rows once; `study_membership` makes their use by ablation and deterministic-single analyses explicit. The byte-identical second copy of the accepted main aggregate is represented as a source alias and does not duplicate endpoint rows. An interrupted 61,745-row prefix before the completed scenario run is registered as historical partial data, not as a second experiment. Unexecuted earlier variants are not inferred to have run from these records.

The complete trajectories and per-step action traces remain in `local-archive/outputs` for the local distribution. Public endpoint rows retain the source's available per-trajectory timing, binding hashes, horizon, status, and other endpoint fields; they are not a substitute for full action traces. See `../data_inventory.json` for the row-level source inventory, evidence details, aliases, public schema mapping, and part metadata.
Canonical endpoint validation is recorded in [`validation.json`](validation.json). The byte-preserved MAIN, accepted ablation, and deterministic-single bootstrap inputs and their SHA-256 values are listed in [`../artifacts/bootstrap_manifest.json`](../artifacts/bootstrap_manifest.json). The public result-table normalization receipt is [`../artifacts/result-table-normalization-validation.json`](../artifacts/result-table-normalization-validation.json).


## Scenario-ablation and descriptive cost artifacts

The accepted scenario-ablation tables, bootstrap evidence, figures, and source-to-public hash mapping are indexed by [`uwsaa_scenario_ablation_manifest.json`](../artifacts/scenario-ablation/uwsaa_scenario_ablation_manifest.json). It records 211,680 newly executed rows plus 35,280 reused K=8 reference rows and the corresponding validation.

Public P5 cost outputs are descriptive on-policy timing data. They include 317,520 trajectory records and 11,325,741 decision-selection timing records, with no confidence intervals or controlled-hardware speedup interpretation. The latter gzip is distributed as five ordered parts. Reassemble and verify it with:

```text
python reproducibility/data/reassemble_endpoints.py --index results/uwsaa-benchmark-evaluation/data/uwsaa_cost_decision_select_seconds.parts.json --output <path>/uwsaa_cost_decision_select_seconds.csv.gz
```

Full per-step action traces remain in the local archive; the public cost tables are separate trajectory/decision timing summaries and do not include those traces.
