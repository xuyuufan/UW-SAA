# UWSAA standalone research bundle

This folder contains the final fast public TW-SAA selector route, renamed **UWSAA** throughout its executable code and file names. The original source checkout was copied, not moved or edited. The old source identifiers map as follows: `TWUniformFastSelector` → `UWSAAFastSelector`, `TWSAASelector` → `UWSAASelectorBase`, and `TW_UNIFORM` → `UWSAA`. The frozen scenario seed namespace still contains its original `tw_saa` token to preserve the registered scenario draws. The full shared SAA implementation, including its existing completion scorer, is retained.

## Run one public decision

Use the Python 3.12 environment already installed for the source workspace. `requirements.txt` records the observed environment versions (`numpy 2.4.4`, `torch 2.13.0+cu132`); it is a snapshot, not an installation recipe. Reuse that environment; do not run a package installation for this bundle. From the repository root, run the CPU-only smoke check:

```powershell
py -3.12 -B checks\public_smoke.py
```

The check constructs one detached public frame and calls both UWSAA and the `NATIVE20` comparison baseline once for each registered solver: EM, GP, ERI, EG, Chain, and Minmax. It verifies every returned destination is legal and writes the JSON record to `checks/public_smoke_result.json`. This is a small API and correctness check; it does not train, run a trajectory, or produce performance evidence.

For an application-owned public frame, use the same public data contract:

```python
from EM.scrp.noise_aware_forecast import PublicObservation
from R.scrp.r_order_public import PublicHistoryTracker, attach_history_summary
from R.experiments.uwsaa_entrypoint import load_runtime, load_gp_rule, make_policy

# ranks, occupied, and legal_mask are public NumPy arrays with matching shapes.
observation = PublicObservation(
    rank_tensor=ranks,
    occupied_mask=occupied,
    legal_destination_mask=legal_mask,
    history_hits=(),
    q_prior=5183 / 9920,
    q_t=5183 / 9920,
)
# frame is a frozen object exposing observation, legal_destinations, max_tiers,
# retrieval_count, relocation_count, crane_position, and crane_time.
frame = attach_history_summary(frame, PublicHistoryTracker().summary(frame))
runtime = load_runtime()
gp_rule = load_gp_rule()
policy = make_policy("Minmax", "UWSAA", runtime, gp_rule)
action, trace = policy.select(frame)
```

The frame must contain only public information: rank `1` is the current target, `2..21` are visible upcoming priorities, and `22` marks anonymous tail items. Its legal-destination tuple must match the observation mask. `checks/public_smoke.py` provides a complete frame constructor. The runtime and GP rule loaders resolve paths relative to this folder and do not fall back to the source checkout.

`configs/uwsaa.json` records the method, H20 baseline, solver list, and comparison scope. `Leveling/data/caserta_table2_840.json` contains 840 episodes grouped into 21 parameter groups with 40 instances per group. Stack counts and tier limits vary by parameter group; 21 is not the stack count, and 40 is not the tier limit. The table is a benchmark/provenance input, not a new independent or sealed test set. The source records include episode retrieval sequences; policy calls must receive only a detached public frame and must not receive those private episode fields.

## Formal evaluation boundary

This folder exposes the actual public selector and native H20 comparison policies for one-step use. It does **not** include the official frozen Caserta continuation runner, its full release/receipt tree, or its hash-bound launch protocol. For official experiment reproduction, use the original PLAN-061 runner and registered receipts. The accepted GP prefix/expression is copied from the frozen preparation receipt and identified in `models/gp_accepted_rule.json`. Renaming code changes source hashes, so this bundle's manifest records its own source and final hashes and does not claim the original release hash passes.

Active continuation outputs were excluded while their processes were running; they are not represented as completed results. No archive, broad output directory, or unrelated repository content is included.

## Integrity

`manifest.json` records copied source paths, destination paths, operation type, source SHA-256 where applicable, final SHA-256, and any rename edits. The final verification also records that all source hashes still match their pre-copy values. Recompute target hashes after any local edits. Source files remain in the original checkout.
