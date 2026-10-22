# UW-SAA trajectory benchmark

This repository contains the public UW-SAA decision method, its six solver baselines, the Caserta input set, a complete trajectory endpoint runner, the registered evaluation protocol, and endpoint-level results. The public method identifier is `UWSAA`; all result metrics use the names `relocations` and `relocations_per_container`.

The public results package is [UWSAA benchmark evaluation](results/uwsaa-benchmark-evaluation/README.md). It includes condition-level outcomes, accepted comparisons, figures in PNG and PDF, and the scripts used to regenerate result tables and figures. The canonical trajectory endpoints and source mapping are described in [the endpoint data guide](reproducibility/data/README.md) and [the data inventory](reproducibility/data_inventory.json). The [public bundle manifest](reproducibility/public_bundle_manifest.json) records SHA-256 hashes and byte sizes for every published file.

The registered matrix, pairing, timeout, and confidence-interval rules are described in [the public protocol](reproducibility/PROTOCOL.md).

## Method and solver labels

For the selected solver, UWSAA obtains destination sequences at horizons 0, 2, 5, 10, and 20. It merges identical sequences and evaluates the remaining candidates against eight synthetic completion scenarios: two scenarios at each window width 2, 5, 10, and 20. Each width has equal total weight. The same scenarios score every candidate, using the fixed Minmax continuation. Policies receive a detached public decision frame; the future retrieval sequence stays with the environment and forecast provider.

The registered solver labels are EM, GP, ERI, EG, Chain, and Minmax. `relocations` is the total relocation count, and `relocations_per_container` is that count divided by the initial number of containers. These names map to historical source fields `R_raw` and `R_over_N`. The solver label `ERI` is an algorithm name; it is unrelated to the historical letter used in metric field names.

## Install and check

The package uses Python 3.12 and the pinned dependency in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B checks/public_smoke.py
python -B checks/environment_smoke.py
python -B checks/trajectory_runner_smoke.py
python -B checks/aggregate_validation_smoke.py
python -B checks/verify_bundle.py
```

The smoke checks exercise interfaces, policy behavior on small fixtures, and strict endpoint validation. They do not run a performance benchmark. The detached-frame source differential receipt and statistics implementation check are optional compatibility checks; the first needs an externally supplied frozen implementation:

```bash
python -B checks/source_differential_smoke.py --source-root /path/to/frozen-source
python -B checks/statistics_source_parity.py --source-root /path/to/frozen-source
```

## Run registered trajectories

Use `--list` to inspect the conditions and arms before running a suite:

```bash
python -B -m reproducibility.runner --suite main --list
python -B -m reproducibility.runner --suite ablation --list
python -B -m reproducibility.runner --suite p0 --list
python -B -m reproducibility.runner --suite multik --list
```

For a complete single-episode smoke trajectory, select a registered condition and save to a new output path:

```bash
python -B -m reproducibility.runner --suite main --smoke --condition CLEAN --output reproducibility/results/smoke.jsonl
```

For a full registered matrix, use `--run` and a new output path. The main suite has 41 conditions × 840 layouts × six solvers × two arms (413,280 endpoint rows). The ablation suite has seven conditions × 840 layouts × six solvers × eight logical arms (282,240 rows); its two common reference arms are shared with MAIN in the canonical result data. P0 uses its deterministic single-completion control, while the multi-K suite has UWSAA plus the six registered rollout/LB1 variants. Each trajectory has a 120-second wall-clock elapsed cap, including policy calls; failures produce endpoint rows and make the command exit nonzero. Output paths are opened exclusively so an existing run is not overwritten.

```bash
python -B -m reproducibility.runner --suite main --run --output reproducibility/results/main.jsonl
```

The runner emits one terminal endpoint per episode-condition-solver-arm cell and a manifest that binds the selected matrix, layout metadata, dataset hash, and endpoint count. It does not emit per-decision action traces, timing summaries, or performance claims. The public archive also preserves partial endpoint prefixes with their historical status labels. Full action traces and original recovery logs are outside the public distribution.

Validate and describe an endpoint run before using it:

```bash
python -B -m reproducibility.aggregate \
  --endpoints reproducibility/results/main.jsonl \
  --manifest reproducibility/results/main.jsonl.manifest.json \
  --output reproducibility/results/main_descriptive.csv
```

The aggregate script rejects partial or failed runs, duplicate or missing pair keys, row/manifest label mismatches, incomplete retrieval accounting, and forecast hashes that differ across paired arms.

## Recompute registered statistics

The provenance-indexed endpoint records are distributed as ordered gzip parts. The standard-library [reassembly helper](reproducibility/data/reassemble_endpoints.py) verifies every part and writes a reconstructed gzip into the operating-system temporary directory. The statistics runner uses this helper automatically when the assembled file is absent; an explicitly supplied missing `--endpoints` path is an error. It checks complete registered matrices and paired first-forecast-row hashes before writing any table. Historical records with overlapping episodes remain separately labeled in the archive. Accepted matrices use deduplicated endpoint references, with shared baseline records stored once and identified by membership and source mappings:

```bash
python -B -m reproducibility.statistics --suite main --output-dir analysis/main
python -B -m reproducibility.statistics --suite ablation --output-dir analysis/ablation
python -B -m reproducibility.statistics --suite deterministic --output-dir analysis/deterministic
python -B -m reproducibility.statistics --suite scenario-ablation --output-dir analysis/scenario-ablation
```

Each output directory must be new. The paired bootstrap uses 10,000 draws, NumPy PCG64, seed `202610195903`, sorted geometry and layout-fingerprint order, and matched within-geometry layout resampling. It averages registered repeats within each layout, layouts within geometry, and the 21 geometries equally. The noisy primary first forms one relative reduction per family, then averages the three families equally, then averages the six solver effects equally. CLEAN is reported separately. Registered multiplicity adjustments are kept with the comparison that specified them.

To compare the registered primary estimates and intervals against the accepted result tables without running trajectories or per-condition analyses, run:

```bash
python -B -m checks.verify_accepted_statistics
```

This verifies all 56 MAIN, Caserta ablation, and deterministic solver rows plus the 49 multi-K primary and absolute-interaction rows. Its receipt at `checks/accepted_statistics_verification.json` records input hashes and the maximum numeric difference.

For the multi-K follow-up, the primary table has seven rows per solver: four K=1/2/4/16 rollout comparisons against K=8, the K=8 rollout/LB1 comparison, the K=1 rollout/LB1 comparison, and the absolute rollout/LB1 interaction. The interaction is reported in relocations as `(P4 − K1) − (P3 − K8)`; positive values mean the LB1 relocation penalty is larger at K=1 than K=8. The relative interaction is secondary and dimensionless. CLEAN remains a separate secondary comparison.

## Evidence status and scope

The result inventory distinguishes accepted benchmark outcomes, development or follow-up runs, partial records, failed registered gates, and correctness smoke checks. A stopped earlier expansion remains recorded as not run; later K-based results are reported as their own executed study. The packaging work adds a runnable endpoint runner and rechecks small trajectories, source-to-public policy parity, and existing endpoint statistics. It does not rerun the full performance matrices.

Trajectory endpoint data are sufficient to reproduce the registered terminal-outcome tables and paired statistics. They do not reconstruct every action or hidden forecast observation. Published selection-time summaries are descriptive; they do not reproduce hardware load or establish a causal speedup. The report is evidence on the public Caserta layouts; it is not sealed or out-of-distribution evidence.

## Repository map

```text
uwsaa/                 Public UWSAA policy and registered comparison arms
baseline/              EM, GP, ERI, EG, Chain, Minmax, and Native20 adapters
environment/           Online environment and detached public-frame contracts
data/                  Caserta input layouts and retrieval sequences
configs/               Frozen runtime and solver settings
reproducibility/        Runner, forecast registry, strict aggregation, and statistics
checks/                 Bounded policy, runner, and package-integrity checks
results/                Endpoint-level accepted tables, figures, and provenance guide
```
