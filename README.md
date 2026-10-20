# Uniform-Width Sample-Average Approximation (UW-SAA)

UW-SAA is a public-state decision method for container relocation. This repository exposes the implementation through the API identifier `UWSAA` and includes six solver baselines with `NATIVE20` comparison policies.

For a selected solver, UW-SAA obtains destination sequences at horizons 0, 2, 5, 10, and 20. It merges identical sequences and scores the remaining candidates on eight synthetic completion scenarios: two scenarios for each window width 2, 5, 10, and 20, with equal aggregate weight per width. Each completion uses a fixed Minmax continuation, and the score is the weighted mean relocation count. Candidates share the same scenarios. The policy receives a detached public decision frame rather than the episode or evaluator-only retrieval sequence.

## Repository contents

```text
uwsaa/        UW-SAA selector, candidate generation, and policy entrypoint
baseline/     EM, GP, ERI, EG, Chain, Minmax, and Native20 adapters
environment/  OnlineCRP environment and public observation/frame contracts
data/         Caserta Table 2 input data
configs/      UW-SAA settings and the GP rule used by the entrypoint
checks/       small correctness and interface checks
```

## Install and run

The checks were exercised with Python 3.12 and NumPy 2.4.4. The required Python package is listed in `requirements.txt`.

Create and activate a virtual environment, then install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run these commands from the repository root:

```bash
python -B checks/public_smoke.py
python -B checks/environment_smoke.py
```

The public smoke constructs a synthetic detached frame and makes one decision for each solver under `UWSAA` and `NATIVE20`. The environment smoke checks one scalar environment step and package-local imports. They exercise small interface and correctness cases; they do not run trajectories or measure performance.

## Public API

Build a detached frame from an `environment.noise_aware_forecast.PublicObservation`, an `environment.online_crp.OnlineCRPPublicState`, and legal destinations. Create one policy and history tracker per episode or actor, then reuse them across decisions:

```python
from environment.public_frame import make_public_frame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from uwsaa.entrypoint import load_gp_rule, load_runtime, make_policy

history = PublicHistoryTracker()
policy = make_policy("EG", "UWSAA", load_runtime(), load_gp_rule())

# For each decision, using values from the public environment/forecast interface:
frame = make_public_frame(
    observation, public_state, legal_destinations, max_tiers=max_tiers
)
frame_for_policy = attach_history_summary(frame, history.summary(frame))
destination, trace = policy.select(frame_for_policy)

# After the environment produces the next detached public frame:
history.update(frame, destination, next_frame)
```

The method name can be set to `NATIVE20` to use the selected solver at fixed horizon 20. The complete runnable frame fixture is in [checks/public_smoke.py](checks/public_smoke.py). Keep the episode and future retrieval sequence on the environment/evaluator side; pass the policy only a detached frame.

## Baseline solvers

| Solver | Public comparison implementation |
| --- | --- |
| EM | [baseline/em.py](baseline/em.py) |
| GP | [baseline/gp.py](baseline/gp.py) and [baseline/gp_expression.py](baseline/gp_expression.py) |
| ERI | [baseline/eri_eg_public.py](baseline/eri_eg_public.py) |
| EG | [baseline/eri_eg_plan059.py](baseline/eri_eg_plan059.py) and [baseline/sequence_commit.py](baseline/sequence_commit.py) |
| Chain | [baseline/chain.py](baseline/chain.py) and [baseline/chain_core.py](baseline/chain_core.py) |
| Minmax | [baseline/minmax.py](baseline/minmax.py) and [baseline/minmax_core.py](baseline/minmax_core.py) |

`NATIVE20` policies are exposed through [baseline/native.py](baseline/native.py).

## Data and evaluation scope

The included [Caserta Table 2 file](data/caserta_table2_840.json) contains 840 instances in 21 parameter groups of 40 instances each. Records include retrieval sequences used by the environment/evaluator. A policy call should receive only the corresponding public decision frame.

This repository provides the UW-SAA method, baseline decision interfaces, configuration and data files, and small correctness checks. It does not include a full trajectory benchmark runner or a complete evaluation protocol. The included smoke checks are not benchmark results and do not establish comparative performance or generalization.
