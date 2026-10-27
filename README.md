# Uniform-Width Sample-Average Approximation (UW-SAA)

This repository contains the public `UWSAA` decision method, six solver adapters (EM, GP, ERI, EG, Chain, Minmax), registered trajectory runners, and the completed replacement-GP Caserta MAIN results. The active GP is the trained seed-5, H=20 rule in [`configs/gp_rule.json`](configs/gp_rule.json). Its engine is the MIT-licensed Python reproduction attributed to Narek Sergeyan, rather than the original authors' implementation.

## Current results

**GP MAIN status: COMPLETE (68,880 / 68,880 endpoints). Numerical verification: PASS. GP ablation/P0/multi-K: CANCELED.**

The [GP MAIN results](results/gp-caserta-main-20261024/README.md) cover **41 conditions × 840 layouts × two arms = 68,880 complete endpoints**. Local conditions 1–14 contribute 23,520 rows; cloud conditions 15–41 contribute 45,360 rows. The 252-row cross-host probe is diagnostic and excluded from the formal matrix.

UWSAA reduces relocations against NATIVE20 by **6.63%** across the equally weighted SS/TW/DP family effects (paired 95% CI **6.38%–6.87%**). On CLEAN it increases relocations by **1.74%** (95% CI **1.14%–2.36%**). A separately written statistical implementation verified all 100 published rows and 894 numeric fields on 2026-10-25, including the registered bootstrap intervals; the maximum absolute difference was 1.43e-14 or less. See [the review record](results/gp-caserta-main-20261024/REVIEW.md). The original CSV `pending_review` tags are preserved as source metadata. This is implementation-independent numerical review by the same Codex session, not external human peer review.

![GP MAIN results](results/gp-caserta-main-20261024/gp_caserta_summary.png)

- [All 41 conditions](results/gp-caserta-main-20261024/gp_caserta_all_conditions.png) · [Detailed results in Chinese](results/gp-caserta-main-20261024/结果说明.md)
- [Family statistics](results/gp-caserta-main-20261024/family_effects.csv) · [Condition statistics](results/gp-caserta-main-20261024/condition_effects.csv) · [Raw endpoint archive](results/gp-caserta-main-20261024/gp_caserta_full_results.zip)
- [Results inventory and remaining studies](results/README.md) · [Data inventory](reproducibility/data_inventory.json) · [Registered protocol](reproducibility/PROTOCOL.md)

The retired GP implementation, configuration, mixed endpoint parts, old result tables, old figures, and obsolete verification receipts have been removed from the active tree. Replacement-GP ablation and its P0/multi-K controls were canceled on 2026-10-25 at the project owner's request and are no longer scheduled. Their historical protocol and runnable code remain for reference; they have no active numerical results or figures. Timing studies and the MAIN six-solver merge remain separate unfinished work. The other five solvers' historical endpoints remain recoverable from the pinned Git history through [the non-GP recovery guide](reproducibility/data/README.md).

## Install and check

Use Python 3.12. The public runtime and result plotting use separate requirement files; training has its own pinned environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-results.txt
python -B checks/verify_published_files.py
python -B -m checks.verify_gp_results
```

The final command verifies the raw archive and every member hash, validates all formal endpoints and their model/data bindings, recomputes the registered 10,000-draw paired bootstrap, and requires exact matches for both tables and the analysis manifest. It also checks the summary against the family/CLEAN rows. It writes its receipt under ignored `work/` and removes temporary extracted endpoints. This command recomputes statistics without rerunning trajectories or training. The original published CSV tags remain `pending_review`; the independent numerical review is recorded separately.

The independent checker imports no repository code. It reads original JSONL, reconstructs the registered design and layout fingerprints, and uses layout multiplicity matrices plus explicit percentile interpolation rather than the frozen analyzer's implementation:

```bash
OPENBLAS_NUM_THREADS=1 python -B checks/independent_gp_main_review.py
```

The receipt and per-field comparison are in `checks/gp_main_independent_review.json` and `checks/gp_main_independent_comparison.csv`; new runs default to ignored `work/`. The check also verifies all 252 probe records against both the local reference and the formal matrix, all 30 seed selection receipts, zero training/evaluation layout overlap, and the recorded runtime file hashes.

To run the statistical regression tests, install `pytest==8.4.2` and use `python -m pytest -q checks/test_independent_gp_main_review.py`. Historical training/source differential receipts are preserved in [the formal training evidence](artifacts/gp_h20/20261024_formal_30seed/README.md).

## Regenerate figures

Install Noto Sans CJK (`fonts-noto-cjk` on Debian/Ubuntu) or pass an installed Chinese font family with `--font`.

```bash
python -B results/gp-caserta-main-20261024/scripts/generate_figures.py --output-dir work/gp-figures
```

This regenerates the summary and all-condition plots as PNG and vector PDF from the published CSVs. It uses repository-relative paths and writes only the requested output directory. The checked-in figures are regenerated from the same script; they contain no retired-GP values.

## Method and training boundary

UWSAA obtains destination sequences at horizons 0, 2, 5, 10, and 20, merges identical sequences, and scores the candidates against eight synthetic completion scenarios: two at each width 2, 5, 10, and 20. Each width has equal total weight. All candidates share the scenarios and the fixed Minmax continuation. Policies receive detached public decision frames; future retrieval order stays with the environment and forecast provider.

The replacement GP was selected only on the separate [840-episode training bank](experiments/gp_h20/data/train_bank.json), across 30 registered seeds with 51,000 fitness calls per seed. Seed 5 achieved 24,421 training relocations; H=20 was selected on the same bank. This is training selection. Caserta is the sealed evaluation benchmark for UW-SAA and was not used for training, formula selection, horizon selection, or tuning. The [formal training evidence](artifacts/gp_h20/20261024_formal_30seed/README.md) preserves all 30 seed receipts and model fingerprints.

Training uses Python 3.12.4, NumPy 1.26.4, Numba 0.59.1, and llvmlite 0.42.0 from [the training requirements](experiments/gp_h20/requirements-training.txt). Keep that environment separate from the NumPy 2.4.4 public runtime. The 51,000 source-engine calls differ from the paper's reported 50,000 budget; the source engine and this difference remain documented.

## Run trajectories

```bash
python -B -m reproducibility.runner --suite main --list
python -B -m reproducibility.runner --suite ablation --list
python -B -m reproducibility.runner --suite p0 --list
python -B -m reproducibility.runner --suite multik --list
python -B -m reproducibility.runner --suite main --smoke --solver GP --condition CLEAN --output work/gp-smoke.jsonl
```

Use `--run` and a new output path for a full matrix. Each trajectory has a 120-second wall-clock cap; failures remain recorded and make a run incomplete. Outputs are never overwritten. Every GP endpoint binds the selected formula, source, protocol, training bank, implementation, and seed receipt. The [partition runner](reproducibility/gp_caserta_partition_runner.py) preserves the completed local/cloud design; [its handoff](reproducibility/CLOUD_GP_MIGRATION_PROMPT.md) documents the original execution procedure.

The six-solver `reproducibility.statistics` command requires a complete identity-checked six-solver canonical payload. The completed GP-only MAIN archive uses its separate paired analyzer and cannot supply MEAN6 or results for canceled studies. See [the MAIN review guide](results/gp-caserta-main-20261024/REVIEW.md) for completed checks and the suggested independent review.

## Repository map

| Directory | Contents |
| --- | --- |
| `uwsaa/` | Public UWSAA policy and registered comparison arms |
| `baseline/` | Six solver adapters and active GP runtime |
| `environment/` | Scalar online environment and public-frame contracts |
| `configs/` | Active trained GP and frozen runtime settings |
| `data/` | Caserta evaluation layouts |
| `experiments/gp_h20/` | Training code, protocol, and separate training bank |
| `artifacts/gp_h20/` | Formal training receipts and correctness evidence |
| `third_party/crp_online/` | Vendored GP engine, attribution, and MIT license |
| `reproducibility/` | Runners, registered statistics, data and publication inventories |
| `checks/` | Newly published MAIN numerical review, result-integrity checks, and review receipts |
| `results/` | Current GP MAIN tables, figures, raw archive, and regeneration script |
| `work/` | Ignored scratch files and regenerated analysis outputs |
