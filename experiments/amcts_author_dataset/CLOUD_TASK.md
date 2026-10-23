# Codex Cloud task prompt: finish the author-data AMCTS evaluation

Paste the prompt below into a Codex Cloud task using the branch `feat/amcts-author-dataset-cloud-20261023` (or the exact reviewed PR head commit). It is intended to run in the repository workspace; it does not require a separate environment ID. Do not switch to `main` or use a different commit.

---

Run the Zhang et al. AMCTS experiment on the bundled Mendeley Data v3 author dataset from this repository branch. First verify the branch/commit and read `experiments/amcts_author_dataset/README.md`. The scope is exactly the 1,680 released instances with their author-provided fixed RPMs. This is original-RPM AMCTS only; do not run UWSAA, H20, B&SDRL, MLP training, or any Caserta test.

Use the included code and protocol without edits: 2,500 × W playouts at every decision, paper parameters unchanged, no per-case timeout. Do not add hidden/future retrieval information to the policy. The evaluator may reveal only the current target at each step. The complete retrieval sequence must remain outside the policy.

Install the tested dependencies:

```bash
python3 -m pip install -r experiments/amcts_author_dataset/requirements-amcts-author-dataset.txt
```

Before running, inspect visible logical CPUs and available memory. Run the full hash/schema/resume preflight and show the receipt:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/coordinate_author_batch_linux.py --prepare
```

The initial packaged snapshot should report 37 complete, 2 partial, and 1,641 not run. If the counts have changed because a prior batch progressed, report the new verified counts instead. Stop if any archive, metadata, task-list, method-source, input, terminal, budget, or information-boundary hash check fails.

Choose 2 workers by default; use up to 4 only if the environment's visible CPU/RAM can support them without memory pressure. Each execution call is capped at 24 wall-clock hours and writes checkpoints throughout. Execute/resume with:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/coordinate_author_batch_linux.py --execute --workers 2 --batch-hours 24
```

Continue with additional 24-hour-or-shorter batches until all 1,680 unique cases are complete, preserving each batch manifest/events and every per-case JSONL history. On each batch start, revalidate the full input set. Only skip a complete case when exactly one complete terminal matches its frozen input and method hashes; retain partial attempts and restart those cases from the initial layout. Do not merge partial costs into the canonical relocation metric. Do not overwrite earlier batch receipts.

After each batch, run:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/report_author_results.py --out-dir experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/results_v2/analysis
```

At completion, verify there are exactly 1,680 unique complete cases, zero partial/not-run cases, no fatal worker errors, valid conservation/legal-action checks, and unchanged source/input hashes. Report ARR = mean(relocations/N) and 100 × ARR in percent, alongside the 56-configuration breakdown, absolute relocations, completion count, wall/process CPU accounting, and the included PNG/PDF figures. Keep the paper-reported 45.52% ARR separate from measured values; do not claim exact numerical reproduction or compare our timing to the paper's timing table. The local Windows run timing is historical and must not be mixed with cloud timing.

Keep all artifacts in the workspace so they can be reviewed/downloaded. Do not push, merge, or change repository code unless separately requested. If a 24-hour cap or platform limit ends a call, report the saved checkpoint and exact resume command. Continue in the same existing cloud task/workspace when possible; a new cloud task does not automatically inherit uncommitted files, so first transfer its checkpoint archive or use a committed checkpoint. To package the current result and resume state for download/transfer, run from the repository root:

```bash
tar -czf /tmp/amcts-author-dataset-checkpoint.tar.gz -C experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset results_v2 receipts
```

Provide `/tmp/amcts-author-dataset-checkpoint.tar.gz` as a downloadable artifact if the workspace supports it, or leave it in the workspace and give its path and SHA-256.

---
