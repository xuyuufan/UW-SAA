# Endpoint data

The current GP-only MAIN archive is [`gp_caserta_full_results.zip`](../../results/gp-caserta-main-20261024/gp_caserta_full_results.zip). It contains 41 JSONL condition shards and companion manifests: local conditions 1–14 (23,520 rows) and cloud conditions 15–41 (45,360 rows). All 68,880 formal rows use the active trained GP and the same Caserta dataset. Diagnostic probe rows are stored separately under `integrity/probe_diagnostic_only/` and are excluded from the formal matrix.

Run `python -B -m checks.verify_gp_results` from the repository root to verify every archive hash and recompute the registered tables. It extracts only formal shards to a temporary directory under ignored `work/`, validates model identity, layout and forecast pairing, and checks the recomputed tables and manifest exactly. It removes the temporary endpoints after verification and writes `work/gp_results_verification.json`. The original statistics retain their frozen `pending_review` tags. A separately written implementation has since passed independent numerical verification; see [the review record](../../results/gp-caserta-main-20261024/REVIEW.md). It does not claim external human peer review.

See [the current results guide](../../results/gp-caserta-main-20261024/README.md) for tables, plots and download links. A complete replacement six-solver canonical CSV is not yet published. Replacement-GP ablation/P0/multi-K reruns were canceled on 2026-10-25; no replacement data for them are planned. Timing data remain separate unfinished work.

## Historical non-GP endpoint recovery

The former canonical mixed endpoint payload is not present in the active checkout. Its exact bytes remain available in the pinned Git history. This repository carries only a recovery script and a compact receipt so the other five solvers' source records can be restored when needed.

The recovery script reads the source parts and index from pinned Git commit a39d0ac0e454f48d591d573868afa21531db34e7. It verifies the index SHA, each part's byte count and SHA, the combined source payload SHA, and row accounting. It streams the CSV while removing rows whose solver is GP or MEAN6; all retained CSV cell text, source/provenance columns, and evidence_status values are preserved. The output gzip is split into parts of at most 40,000,000 bytes, and the script writes a standard parts index and receipt.

Run it in a checkout whose Git object database contains the pinned commit. Choose an empty output directory outside the checkout:

    python -X utf8 -B reproducibility/data/recover_non_gp_endpoints.py --output-dir /tmp/uwsaa-retained-non-gp
    python -X utf8 -B reproducibility/data/reassemble_endpoints.py --index /tmp/uwsaa-retained-non-gp/retained_non_gp_endpoints.parts.json --output /tmp/uwsaa-retained-non-gp/retained_non_gp_endpoints.csv.gz

The payload is intentionally not uploaded with this repository because it is approximately 147 MB. The checked-in receipt at retained_non_gp_endpoints.receipt.json records the source and filtered payload SHA-256 values, part hashes, row counts, solver counts, evidence-status counts, and study coverage.

| Record | Rows or bytes |
| --- | ---: |
| Pinned source endpoint rows | 1,722,662 |
| Removed GP rows | 287,113 |
| Retained non-GP rows | 1,435,549 |
| Filtered gzip bytes | 147,364,265 |
| Filtered gzip SHA-256 | 100b385d7a6f2b8424a245e3b37a3f4ecace12a2536344b983094ad1fa398a4a |

Retained solver counts are Chain 287,106; EG 287,113; EM 287,113; ERI 287,113; and Minmax 287,104. These include historical, partial, correctness-only, and failed-gate records as separately tagged in the receipt. They are not all accepted performance results.

The retained five-solver file must not be passed to registered statistics. Statistics require a new canonical six-solver payload with complete replacement-GP identity fields. The merge must join by layout fingerprint and registered condition metadata. It must recompute the historical source-compatible hash of the first forecast row and compare it with first_forecast_row_sha256; the runner's full forecast_table_sha256 must remain a separate field. The standard UWSAA arm supplies the K=8 rollout reference in multi-K analyses.