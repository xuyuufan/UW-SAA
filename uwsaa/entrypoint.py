"""One-frame public UWSAA and Native20 entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from uwsaa.fast import UWSAAFastSelector
from baseline.native import SOLVERS, FixedHorizonActor, SingleStepPolicy
from baseline import eri_eg, chain, minmax
from baseline.em import select_em_destination
from baseline.eri_eg_public import select_eri20_destination
from environment.noise_aware_forecast import HORIZONS
from environment.public_frame import truncate_public_frame, public_frame_to_legacy_state

ROOT = Path(__file__).resolve().parents[1]


def load_runtime() -> dict[str, Any]:
    return {
        "truncate_public_frame": truncate_public_frame,
        "public_frame_to_legacy_state": public_frame_to_legacy_state,
        "em_select": select_em_destination,
        "eri_select": select_eri20_destination,
        "eri_eg": eri_eg,
    }


def load_gp_rule() -> dict[str, Any]:
    """Load only a complete, fingerprint-matched GP artifact; pending fails closed."""
    path = ROOT / "configs" / "gp_rule.json"
    if not path.is_file():
        raise RuntimeError(f"GP model configuration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "uwsaa_gp_rule_v1":
        raise ValueError("GP model configuration has an unknown schema")
    status = payload.get("status")
    if status != "trained":
        raise RuntimeError(
            f"GP model is pending (status={status!r}); complete all 30 registered seed runs "
            "and the training-bank selection before enabling GP"
        )

    from baseline.gp import compile_priority, current_fingerprints, select_destination

    expected = current_fingerprints()
    for field, current in expected.items():
        if payload.get(field) != current:
            raise ValueError(f"GP artifact {field} does not match the current source/protocol/data")
    formula = payload.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("trained GP artifact has no formula")
    formula_sha256 = hashlib.sha256(formula.encode("utf-8")).hexdigest()
    if payload.get("formula_sha256") != formula_sha256:
        raise ValueError("trained GP formula SHA-256 does not match its recorded identity")
    identity = payload.get("model_identity")
    required_identity = {
        *expected.keys(), "selected_seed", "training_total_relocations",
        "horizon_training_totals", "seed_receipt_sha256", "selection_basis",
        "validation_status", "formula_sha256", "best_fixed_h",
    }
    if not isinstance(identity, dict) or not required_identity.issubset(identity):
        raise ValueError("trained GP model identity is incomplete")
    for field, current in expected.items():
        if identity[field] != current:
            raise ValueError(f"GP model identity {field} does not match the current artifact inputs")
    if identity["formula_sha256"] != formula_sha256:
        raise ValueError("GP model identity formula SHA-256 does not match the formula")
    if identity["best_fixed_h"] != payload.get("best_fixed_h"):
        raise ValueError("GP model identity best_fixed_h differs from the registered config")
    horizon_totals = identity["horizon_training_totals"]
    if not isinstance(horizon_totals, dict) or set(horizon_totals) != {str(h) for h in HORIZONS}:
        raise ValueError("GP model identity has an incomplete fixed-horizon score grid")
    if int(horizon_totals["20"]) != int(identity["training_total_relocations"]):
        raise ValueError("selected GP training total differs from its H20 score")
    if identity["selection_basis"] != "training_bank_total_relocations":
        raise ValueError("GP rule was not selected by the registered training-bank criterion")
    if identity["validation_status"] != "training_selection_not_independent_validation":
        raise ValueError("GP rule validation status is not the registered training-only status")
    if isinstance(identity["selected_seed"], bool) or int(identity["selected_seed"]) not in range(1, 31):
        raise ValueError("selected GP seed must be in 1..30")
    if payload.get("best_fixed_h") not in HORIZONS:
        raise ValueError("trained GP best_fixed_h is outside the registered horizon grid")
    priority = compile_priority(formula)

    def selector(frame: object, *, horizon: int) -> int:
        return select_destination(priority, frame, horizon=horizon)

    return {
        "selector": selector,
        "formula": formula,
        "formula_sha256": formula_sha256,
        "identity": identity,
        "model_identity": identity,
        "best_fixed_h": int(payload["best_fixed_h"]),
        **expected,
    }


def make_policy(
    solver: str,
    method: str,
    runtime: Mapping[str, Any],
    gp_rule: Mapping[str, Any] | None = None,
):
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if method not in ("UWSAA", "NATIVE20"):
        raise ValueError("method must be UWSAA or NATIVE20")
    effective_gp_rule = gp_rule
    if solver == "GP" and effective_gp_rule is None:
        effective_gp_rule = load_gp_rule()
    if method == "UWSAA":
        return UWSAAFastSelector(solver, runtime, effective_gp_rule)
    if solver in SOLVERS[:4]:
        return FixedHorizonActor(solver, 20, runtime, effective_gp_rule)
    proposer = chain.propose if solver == "Chain" else minmax.propose
    return SingleStepPolicy(solver, proposer)


def select_once(solver, method, frame, *, runtime=None, gp_rule=None):
    return make_policy(
        solver,
        method,
        load_runtime() if runtime is None else runtime,
        gp_rule,
    ).select(frame)
