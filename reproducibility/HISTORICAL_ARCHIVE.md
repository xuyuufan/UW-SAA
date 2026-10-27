# External historical archive

On 2026-10-28 the older GP-only MAIN, historical Ablation, old audit/review outputs, cross-host probe reference and completed GP partition/analysis workflow were removed from the active repository after byte verification.

The local archive is `D:\DSAI\UWSAA\local-archive\repo-cleanup-20261028\a5ee84635409b8d7166d08e147e248e682e3c19b`. Original repository-relative paths are preserved inside it. A complete snapshot of all 183 files at source commit `a5ee84635409b8d7166d08e147e248e682e3c19b` is retained there, including the old documentation and checkers. The external `source-snapshot-verification.json` records all snapshot hashes. Existing local archive versions were preserved.

The [removal receipt](historical_archive_receipt.json) records SHA-256, Git blob IDs, byte sizes and original paths for all 50 removed files (31,095,726 bytes). To restore a historical package, copy its path from that snapshot into a separate historical checkout. Archived checkers should be run with their matching snapshot rather than current documentation/manifests. This is a local archive, not a public download; original files also remain in Git history at the pinned source commit. Git history was not rewritten.

## Evidence boundaries

The earlier GP-only MAIN has 68,880 formal endpoints; its GP outcomes are retained in current six-solver MAIN. Diagnostic probe rows are excluded from formal matrices. The historical Ablation GP and MEAN6 values use the retired GP version and do not describe replacement-GP ablation. Replacement-GP ablation/P0/multi-K remain canceled. The earlier numerical review used a separate implementation in the same Codex session and does not claim external human peer review.

The current `results/best-h-main-20261028` package remains byte-identical, including its integrity receipts, ERI H10 completion data and current-runtime timing sample. These inputs are required by the current six-solver verifier. Frozen runtime/model/protocol/training-bank identity inputs remain public. Full-MAIN timing and OOD generalization are not established.
