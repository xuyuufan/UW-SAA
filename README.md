# Uniform-Width Sample-Average Approximation (UW-SAA)

This folder packages the final fast public selector under the name **Uniform-Width Sample-Average Approximation (UW-SAA)**, with `UWSAA` as its code identifier. The source checkout remains unchanged; this bundle contains byte-identical copies, symbol extracts, and local path/name adaptations. `uwsaa/public.py` keeps the registered scenario seed namespace `plan056_tw_saa_public_scenarios_v1` so the random scenarios retain their original draws. That string is a compatibility constant, not a method or import name.

## Contents

```text
uwsaa/        final fast selector, candidate generation, and local entrypoint
baseline/     EM, GP, ERI, EG, Chain, and Minmax comparison solvers
environment/  leakage-safe scalar environment and detached public frame/history contracts
data/         Caserta Table 2 input (840 instances)
configs/      UWSAA method config and accepted frozen GP rule
checks/       one-frame smoke, source parity evidence, and environment/import check
```

The comparison mapping is `EM → baseline/em.py`, `GP → baseline/gp.py` plus `baseline/gp_expression.py`, `ERI → baseline/eri_eg_public.py`, `EG → baseline/eri_eg_plan059.py`, `Chain → baseline/chain.py`, and `Minmax → baseline/minmax.py`. `NATIVE20` invokes the same solver at fixed horizon 20. Shared offline cores, native actor adapters, public sequence validation, and EG group-transition helpers live alongside these solvers; they contain only solver-independent contracts and the six listed comparison implementations. The generic public history helper was extracted because EG sequence validation requires a decision-fixed public summary.

## Run a public decision

Use the already available Python environment with NumPy. `requirements.txt` records only the runtime dependency; do not install anything for the checks. From this folder:

```powershell
py -3.12 -B checks\public_smoke.py
py -3.12 -B checks\environment_smoke.py
```

`public_smoke.py` builds one legal detached frame and calls UWSAA and Native20 once for all six solvers. It writes `checks/public_smoke_result.json`. This is a CPU correctness smoke, not a trajectory or performance evaluation.

For a live episode, the caller owns the environment and builds a detached frame from the public state. The forecast provider, if used, stays on the evaluator side; pass only its public hit history and `q_t` scalar.

```python
import numpy as np
from environment.noise_aware_forecast import PublicObservation
from environment.online_crp import OnlineCRPEnvironment, load_episode_manifest
from environment.public_frame import make_public_frame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from uwsaa.entrypoint import load_gp_rule, load_runtime, make_policy

episode = load_episode_manifest("data/caserta_table2_840.json")[0]
env = OnlineCRPEnvironment(episode, lookahead_horizon=20)
state = env.state                       # public state only
legal = env.legal_destinations()
ranks = np.zeros((episode.num_stacks, episode.max_tiers), dtype=np.int32)
occupied = np.zeros_like(ranks, dtype=np.bool_)
for stack_id, stack in enumerate(state.stacks):
    for tier, public_rank in enumerate(stack):
        occupied[stack_id, tier] = True
        ranks[stack_id, tier] = public_rank if public_rank else 22  # anonymous tail
legal_mask = np.zeros(episode.num_stacks, dtype=np.bool_)
legal_mask[list(legal)] = True
observation = PublicObservation(
    rank_tensor=ranks, occupied_mask=occupied,
    legal_destination_mask=legal_mask, history_hits=(),
    q_prior=5183 / 9920, q_t=5183 / 9920,
)
frame = make_public_frame(observation, state, legal, max_tiers=episode.max_tiers)
frame = attach_history_summary(frame, PublicHistoryTracker().summary(frame))
policy = make_policy("EG", "UWSAA", load_runtime(), load_gp_rule())
action, trace = policy.select(frame)
# Only the destination crosses back to the environment.
next_state, reward, terminated = env.step(action)
```

`rank_tensor` uses `1` for the current target, `2..21` for visible forecast ranks, `22` for occupied anonymous tail items, and `0` only for empty cells. The legal mask and destination tuple must agree. Do not pass the episode, retrieval sequence, forecast, or environment object to a policy.

## Data and evaluation boundary

`data/caserta_table2_840.json` contains 840 instances in 21 parameter groups with 40 instances in each group; the group key describes stack-count/height geometry, so those counts are not “21 stacks by 40 tiers.” The JSON also contains evaluator-only retrieval sequences. This benchmark was previously used in the project and is not a sealed or out-of-distribution test set.

`configs/gp_accepted_rule.json` contains the accepted frozen GP expression and its receipt identity. The original model/release hash binds the pre-rename source; this bundle uses its own manifest hashes. The official frozen Caserta continuation runner, release receipts, and hash-bound launch protocol are not included. Use the original registered runner for formal evaluation; this package provides the real public method and comparison solver entrypoints without claiming a full pipeline reproduction.

## Integrity and checks

`manifest.json` records every included file's final SHA-256 and the source-file SHA-256 values. It distinguishes byte copies, symbol extraction, path/name edits, and locally generated checks; extracted modules list the exact source symbols. `checks/semantic_parity.json` records the small source-to-clean comparison: 30 horizon proposals, six UWSAA and six Native20 actions/traces, plus EG's second group move and reset on one legal fixture. `checks/verification.json` records static syntax/import checks and the OnlineCRP load-and-step check. No training, benchmark, performance run, or full trajectory was executed.

Run `py -3.12 -B checks\verify_bundle.py` for a lightweight AST/JSON/import audit. The source parity comparison was performed during bundle extraction; its compact result and comparison fields are retained in `checks/semantic_parity.json`.
