# AMCTS on the Zhang et al. author dataset

This isolated experiment bundle runs the Zhang et al. AMCTS method against its released instance set and per-instance retrieval-probability matrices. The method receives only the current bay, the corresponding fixed RPM matrix, capacity, and currently revealed target; the complete retrieval sequence remains in the evaluator. It uses the fixed-RPM tree-reuse implementation and the paper-aligned 2,500 × W playout budget for every decision.

The package does not use UWSAA, the H20 forecast adapter, B&SDRL, or a newly trained MLP. Results from this package are a single-method author-dataset evaluation, not a comparison against the Caserta/UWSAA experiment.

## Data and provenance

Data attribution: Zhanluo Zhang, [Data for “A data-driven approach to solving the container relocation problem with uncertainties”](https://data.mendeley.com/datasets/3j52cjrm52/3), Mendeley Data, version 3, DOI [10.17632/3j52cjrm52.3](https://doi.org/10.17632/3j52cjrm52.3), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The archive contains 1,680 scored instance JSON files (56 configurations × 30 seeds), a non-scored example, and a shared retrieval-probability file. The original archive, metadata index, per-file audit, and extracted instance bytes are retained under `reproducibility/20261023_amcts_author_dataset/data/`.

See [ATTRIBUTION.md](ATTRIBUTION.md) for the redistribution notice.

The included results are the preserved local run at transfer: 37 complete cases, 2 partial case histories, and 1,641 cases not yet run. The report treats only hash-matching complete terminal attempts as scored observations. Partial decisions remain separately auditable and are never converted to relocations or zeros. Local Windows timing is historical provenance and is not directly comparable with cloud timing.

The paper reports AMCTS ARR of 45.52% for the full dataset. That value is shown only as a paper-reported reference; this partial set does not reproduce or validate it. The paper's timing table and our cloud timing are not compared because the paper's per-instance timing scope is unspecified and the released solved subset differs.

The primary metric is `ARR = mean(R/N)` across instances, where `R` is the relocation count and `N` is the number of containers. The report includes both the ratio and `100 × ARR` as a percentage; lower values indicate fewer relocations per container.

## Runtime

The verified local runtime was Python 3.12.3, NumPy 2.4.4, and Matplotlib 3.11.2. The execution path needs NumPy; Matplotlib is used to generate report figures. Install the exact tested versions with:

```bash
python3 -m pip install -r experiments/amcts_author_dataset/requirements-amcts-author-dataset.txt
```

From the repository root, verify inputs/results and create a cloud preparation receipt:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/coordinate_author_batch_linux.py --prepare
```

Inspect the receipt and available CPU/RAM before execution. Each execution batch has a strict 24-hour wall-clock cap, no per-case timeout, and retains per-decision JSONL checkpoints. Choose 1–4 workers based on actual cloud resources (2 is the conservative default); thread pools inside each worker are pinned to one thread. Run or resume one batch with:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/coordinate_author_batch_linux.py --execute --workers 2 --batch-hours 24
```

A later invocation first rechecks all 1,680 inputs and method hashes. It skips only a unique valid complete case with matching input and source hashes, and restarts partial cases from their initial states while preserving prior JSONL events. Do not lower the playout budget or add per-case timeouts.

After a batch, generate the complete/partial/not-run tables and figures:

```bash
python3 experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/report_author_results.py --out-dir experiments/amcts_author_dataset/reproducibility/20261023_amcts_author_dataset/results_v2/analysis
```

The report includes per-case CSV, 56-configuration CSV, partial-attempt accounting, PNG/PDF figures, ARR when all 1,680 cases are complete, and no confidence interval unless one is registered and implemented. See `CLOUD_TASK.md` for a copyable Codex Cloud instruction.
