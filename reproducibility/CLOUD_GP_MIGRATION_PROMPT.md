# Cloud task: run the assigned GP-only Caserta MAIN partition

Current status (2026-10-24): GP-only MAIN has completed across all 41 conditions and 68,880 endpoints. The 252-row cross-host probe passed. See [the published GP results](../results/gp-caserta-main-20261024/README.md). The execution instructions below document the completed run; Replacement-GP ablation/P0/multi-K follow-ups were canceled on 2026-10-25; MAIN six-solver reporting remains unfinished.

## Objective and scope

Run only registered MAIN condition indices 15–41 for the paired comparison of the frozen GP policy in the UWSAA and NATIVE20 arms. Condition indices are 1-based in `reproducibility.protocol.main_conditions()`. The local partition owns indices 1–14 (six SS and eight TW conditions); the cloud partition owns indices 15–41 (six DP, two overlay, 18 regime-mixture, and CLEAN conditions). The partitions are disjoint. The cloud shard must contain exactly 27 × 840 × 2 = 45,360 complete GP endpoints. Together with the local shard, the design has 68,880 endpoints.

This is a GP-only method comparison. Do not restore historical mixed endpoints, rerun other solvers, produce MEAN6, or claim any six-solver result. GP ablation, P0, multi-K, and six-solver statistics remain outside this task. Do not train, retune, select a formula or horizon, or use Caserta to change the method. Caserta is being used only for the registered held-out evaluation.

## Checkout and immutable inputs

Use the exact head commit of the GP migration PR supplied with this handoff. Do not use the base branch's older GP rule. Before running, record `git rev-parse HEAD` and verify it matches the supplied PR head.

The expected frozen identities are:

- GP formula SHA-256: `e3cbbd12defb760be3e6afe57df7004980b1660be73b6f214bad1817473f9845`
- GP implementation SHA-256: `5f29b4d90403df9cb1ead3ba44a2306927d66990b291b4dffcb7827fe493e0ee`
- GP protocol SHA-256: `059f340980a1da0eb1af2c40a1ef17f02696f3701383a516a8593f81858f38db`
- GP training-bank SHA-256: `9dfaf66cce987d70cad2673089c27aea75af1a82e060d043f351218b665384c6`
- GP source-manifest SHA-256: `bdcab4b2c56c73332cd47545608416ea8de1f21168ceb0099a686e652d10bd8d`
- Caserta evaluation data SHA-256: `bb471dc334ad186523bcbd436bbe103dd81e517554999d35132a5e0692c98448` (840 episodes, 21 geometries × 40 layouts)

The selected model is seed 5, H=20, with 24,421 training-bank relocations. Training used CPython 3.12.4, NumPy 1.26.4, Numba 0.59.1, and llvmlite 0.42.0. Install the pinned training extras in a separate environment from the repository's NumPy 2.4.4 public-runtime environment. If the image's default Python differs, first provision CPython 3.12.4 with `uv python install 3.12.4` and create the environment with `uv venv --python 3.12.4 --seed .venv-gp-eval`. If `uv` is unavailable, use the cloud image's managed Python installer or an already installed exact 3.12.4 interpreter. Do not silently evaluate with a different Python, model, or scorer; if the exact interpreter cannot be obtained, report that blocker before running the probe.

```bash
uv python install 3.12.4
uv venv --python 3.12.4 --seed .venv-gp-eval
source .venv-gp-eval/bin/activate
python --version  # Must report 3.12.4.
python -m pip install -r experiments/gp_h20/requirements-training.txt
python -B -m checks.verify_bundle
python -B - <<'PY'
from uwsaa.entrypoint import load_gp_rule

rule = load_gp_rule()
identity = rule.get("model_identity") or rule.get("identity")
assert identity["selected_seed"] == 5
assert identity["best_fixed_h"] == 20
assert identity["formula_sha256"] == "e3cbbd12defb760be3e6afe57df7004980b1660be73b6f214bad1817473f9845"
assert identity["implementation_sha256"] == "5f29b4d90403df9cb1ead3ba44a2306927d66990b291b4dffcb7827fe493e0ee"
assert identity["protocol_sha256"] == "059f340980a1da0eb1af2c40a1ef17f02696f3701383a516a8593f81858f38db"
assert identity["train_bank_sha256"] == "9dfaf66cce987d70cad2673089c27aea75af1a82e060d043f351218b665384c6"
assert identity["source_manifest_sha256"] == "bdcab4b2c56c73332cd47545608416ea8de1f21168ceb0099a686e652d10bd8d"
print({key: identity[key] for key in (
    "selected_seed", "best_fixed_h", "formula_sha256", "implementation_sha256",
    "protocol_sha256", "train_bank_sha256", "source_manifest_sha256",
)})
PY
sha256sum data/caserta_table2_840.json experiments/gp_h20/data/train_bank.json
```

These are read-only preflight checks; they do not invoke training or write a receipt. Confirm the dataset SHA printed by `sha256sum` matches the expected Caserta value above and the loaded trained identity matches the listed fingerprints before the probe.

The PR contains `reproducibility/gp_caserta_partition_runner.py` and the 252-row cross-host probe reference at `reproducibility/references/gp_caserta_crosshost_probe_reference.json`. Use Python 3.12 with the pinned training extras. Do not invoke the `all` partition or create formal rows for local indices 1–14. The commands below use Linux bash and write generated output under the ignored `reproducibility/results/` directory.

## Phase 1: cross-host probe

First measure the effective cgroup CPU quota and CPU affinity. Set `WORKERS` to `min(4, max(1, floor(quota)), affinity_count)`, record those inputs, and set each math-library thread count to one. Use the same worker count for the probe and formal cloud shard so the timing projection reflects the planned run.

Run the 252-endpoint probe: one registered layout per each of 21 geometries, six representative conditions (SS p=.3 repeat 0, TW B=5 repeat 0, DP p=.3 repeat 0, overlay p=.3 B=5 repeat 0, regime profile C_EQ repeat 0, and CLEAN), and both arms. Match each endpoint against `reproducibility/references/gp_caserta_crosshost_probe_reference.json` using condition, layout fingerprint, and arm. Require exact equality of relocations, initial N, decisions, retrieved count, reveals, forecast-table SHA-256, and GP model identity. The probe receipt must also record Python/package versions, CPU quota or affinity, worker count, thread-library caps, wall time, and peak memory when available.

Run the probe in Linux bash with a fresh output directory:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WORKERS=4  # Replace with the measured capped worker count.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -B reproducibility/gp_caserta_partition_runner.py \
  --partition cloud --probe --workers "${WORKERS}" \
  --probe-reference reproducibility/references/gp_caserta_crosshost_probe_reference.json \
  --data data/caserta_table2_840.json \
  --output "reproducibility/results/gp_caserta_cloud_probe_${RUN_ID}"
```

Stop and return the mismatch report if any key or value differs. Do not proceed to the cloud partition on a failed probe. The probe is a correctness diagnostic; exclude its rows from the 45,360 formal endpoints.

## Phase 2: cloud partition

After the probe passes, report its elapsed time, actual workers, and a 45,360-endpoint ETA with a stated safety margin. Then run only registered condition indices 15–41 over all 840 Caserta layouts and both arms. Record the effective CPU limit and actual worker count in the partition receipt. Use a new output directory and preserve any incomplete or invalid shard for diagnosis; never overwrite it or mark it complete. Do not wait for the local shard to finish. For example:

```bash
OUT_DIR="reproducibility/results/gp_caserta_cloud_15_41_${RUN_ID}"
PROBE_RECEIPT="reproducibility/results/gp_caserta_cloud_probe_${RUN_ID}/probe_receipt.json"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -B reproducibility/gp_caserta_partition_runner.py \
  --partition cloud --workers "${WORKERS}" \
  --probe-receipt "${PROBE_RECEIPT}" \
  --data data/caserta_table2_840.json --output "${OUT_DIR}"
```

To resume that exact output after inspecting its run manifest, repeat the command with `--resume`, the same `OUT_DIR`, and the same passing `PROBE_RECEIPT`. The runner caps the resumed wall budget at the original persistent deadline. Do not reset that deadline or replace invalid partial files.

The 27 cloud conditions are DP p=.1/.3/.5 with two registered repeats each, overlay p=.3/B=5 with two repeats, all nine regime-mixture profiles with two repeats each, and CLEAN. Verify the exact ordered IDs against the frozen registry before starting. All endpoint rows must be `complete`, legal, and bound to the same formula/model identity. Paired arms must have the same forecast-table hash for each condition/layout key.

## Return artifacts

Return one compressed artifact bundle containing only the completed cloud shard and its integrity evidence:

- one endpoint JSONL and standard run manifest for each of the 27 conditions (or the exact partition-runner equivalent);
- partition/run manifest with condition indices and IDs, expected and observed counts, dataset and model fingerprints, and completion status;
- the successful cross-host probe receipt and comparison result;
- per-condition timing/worker/environment receipts and a `SHA256SUMS.txt` covering every returned file.

The formal endpoint total must be 45,360 with no missing, duplicate, or extra rows. Exclude caches, temporary files, unrelated logs, and partial outputs from a completion bundle. If the run cannot complete, return a clearly labelled partial diagnostic bundle with the original partial files and failure receipt; do not report a completed shard or a performance conclusion. Do not commit endpoints or generated outputs into the source branch.

## Interpretation boundary

This shard is one part of a paired GP-only comparison. Its completion alone is not an accepted result. Final analysis waits for the full local-plus-cloud 41-condition dataset and independent review. The registered GP-only analyzer reports relocations as primary, relocations per container as secondary, SS/TW/DP family effects and an equal-family macro effect, plus condition-level overlay/regime/CLEAN summaries. It uses the registered 10,000-draw PCG64 layout-cluster bootstrap with seed `202610195903`. It does not produce MEAN6 or replace the registered six-solver statistics procedure.
