# Formal GP-H20 training evidence

Run ID: `20261024_formal_30seed`. This directory contains byte-preserved evidence for the completed 30-seed training-bank search and rule selection. It includes the complete run manifest, all 30 complete seed receipts, the available fixed-horizon score receipts, selected rule, selection receipt, and correctness verification receipts. Logs, caches, partial runs, and raw smoke endpoints are excluded.

All 30 seeds completed with 51,000 fitness evaluations each. The frozen runtime identity is `numba_float64_v1`, NumPy 1.26.4, Numba 0.59.1, llvmlite 0.42.0. Total measured seed compute time was 3,570.23 seconds (59m30s); the first wave took 779.16 seconds and the remaining 22 took 2,791.07 seconds. Elapsed wall time including scheduled monitoring pauses was 7,847.67 seconds (2h10m48s).

The selection rule is minimum total restricted-clean H20 relocations on the 840-row training bank, with ties broken by lower seed. Seed 5 was selected at 24,421 relocations. Its formula SHA-256 is `e3cbbd12defb760be3e6afe57df7004980b1660be73b6f214bad1817473f9845`. Fixed-horizon training totals for H=0,2,5,10,20 are 37,067; 32,237; 28,859; 26,320; and 24,421. The H20 value is the selected seed's registered H20 training score; separate fixed-H receipts are present for H=0,2,5,10.

Selection used the training bank only and is not independent validation. Caserta remains the sealed evaluation benchmark for UW-SAA and was not read for training or selection. The full GP MAIN/ablation/P0/multi-K evaluation matrices, all-solver statistics, MEAN6 values, and figures remain pending. This training record is not performance-evaluation evidence.

`pretraining_migration_verification.json` and `acceleration_differential_verification.json` are preserved as historical correctness/diagnostic receipts. Their recorded scopes and input fingerprints are authoritative; they must not be rewritten to imply that every diagnostic used the final composite training implementation identity. The formal run, selection, and trained runner smoke are separately bound to implementation SHA `5f29b4d90403df9cb1ead3ba44a2306927d66990b291b4dffcb7827fe493e0ee`.

`SHA256SUMS.txt` hashes every other file in this directory byte-for-byte; the public bundle manifest also records the checksum file itself.
