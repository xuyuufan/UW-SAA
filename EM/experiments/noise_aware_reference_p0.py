"""PLAN-037 P0: freeze existing GP/H20 references and verify a public bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from scrp.attention_data import file_sha256
from scrp.attention_evaluation import load_gp_h20_rules
from scrp.h20_a1_policy import (
    H20_BASE_SHA256,
    load_h20_base_checkpoint,
    tensor_state_sha256,
)
from scrp.noise_aware_forecast import ForecastProvider, PublicObservation, semantic_forecast_seed
from scrp.online_crp import OnlineCRPEpisode, OnlineCRPEnvironment, OnlineCRPPublicState
from scrp.scale_diverse_policy import Online2DPriorityAdapter

from experiments.noise_aware_lookahead import (
    _private_physical_stacks,
    sync_retrieval_events,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = ROOT / "output/noise_aware_lookahead_v1/r01"
H20_PATH = WORKSPACE / "Mijn_RL/checkpoints/h20.best.pt"
GP_SETS = ("C", "C_E", "C_O", "C_E_O")
GP_SET_PRIORITY = {name.replace("_", "+"): index for index, name in enumerate(GP_SETS)}
PUBLIC_UNKNOWN_RANK = 22
TRAIN_CONTRACT_KEYS = (
    "config_sha256",
    "implementation_sha256",
    "train_manifest_sha256",
    "horizon",
    "visible_target_count",
    "population_size",
    "evaluations",
    "max_depth",
    "mutation_probability",
    "fitness_evaluations",
)


@dataclass(frozen=True)
class PublicDecisionFrame:
    """Only public ranks, legality, and physical scalar context reach an actor."""

    observation: PublicObservation
    legal_destinations: tuple[int, ...]
    max_tiers: int
    crane_position: int | None
    crane_time: float
    retrieval_count: int
    relocation_count: int


def make_public_frame(
    observation: PublicObservation,
    public_state: OnlineCRPPublicState,
    legal_destinations: tuple[int, ...],
    *,
    max_tiers: int,
) -> PublicDecisionFrame:
    legal = tuple(int(value) for value in legal_destinations)
    mask_legal = tuple(
        int(value)
        for value in np.flatnonzero(np.asarray(observation.legal_destination_mask))
    )
    if legal != mask_legal or len(set(legal)) != len(legal):
        raise ValueError("public frame legality differs from the observation mask")
    if not legal or max_tiers <= 0:
        raise ValueError("public frame needs a legal action and positive tier count")
    return PublicDecisionFrame(
        observation=observation,
        legal_destinations=legal,
        max_tiers=int(max_tiers),
        crane_position=public_state.crane_position,
        crane_time=float(public_state.crane_time),
        retrieval_count=int(public_state.retrieval_count),
        relocation_count=int(public_state.relocation_count),
    )


def public_frame_to_legacy_state(frame: PublicDecisionFrame) -> OnlineCRPPublicState:
    """Map occupied unknown rank 0 to old tuple 0; omit empty padding."""

    ranks = np.asarray(frame.observation.rank_tensor)
    occupied = np.asarray(frame.observation.occupied_mask, dtype=np.bool_)
    legal = np.asarray(frame.observation.legal_destination_mask, dtype=np.bool_)
    if ranks.ndim != 2 or ranks.shape != occupied.shape:
        raise ValueError("public ranks and occupancy mask have incompatible shapes")
    if ranks.shape[0] != len(legal) or ranks.shape[1] != frame.max_tiers:
        raise ValueError("public observation shape differs from frame dimensions")
    if tuple(int(x) for x in np.flatnonzero(legal)) != frame.legal_destinations:
        raise ValueError("public state cannot be built from a changed legality mask")

    stacks: list[tuple[int, ...]] = []
    visible_ranks: list[int] = []
    for stack_index in range(ranks.shape[0]):
        occupied_row = occupied[stack_index]
        height = int(occupied_row.sum())
        expected_row = np.arange(frame.max_tiers) < height
        if not np.array_equal(occupied_row, expected_row):
            raise ValueError("public occupied cells must be a bottom-filled stack prefix")
        values = []
        for tier in range(height):
            rank = int(ranks[stack_index, tier])
            if rank < 0 or rank > PUBLIC_UNKNOWN_RANK:
                raise ValueError("public rank must be 1..21 or the occupied-unknown sentinel 22")
            values.append(0 if rank == PUBLIC_UNKNOWN_RANK else rank)
            if 0 < rank < PUBLIC_UNKNOWN_RANK:
                visible_ranks.append(rank)
        if np.any(ranks[stack_index, height:] != 0):
            raise ValueError("empty padding must remain zero")
        stacks.append(tuple(values))
    if visible_ranks.count(1) != 1 or len(set(visible_ranks)) != len(visible_ranks):
        raise ValueError("public frame must contain one current target and unique visible ranks")
    if min(visible_ranks) != 1:
        raise ValueError("public ranks must preserve current target rank 1")
    return OnlineCRPPublicState(
        stacks=tuple(stacks),
        current_target_id=1,
        retrieval_count=frame.retrieval_count,
        relocation_count=frame.relocation_count,
        crane_position=frame.crane_position,
        crane_time=frame.crane_time,
        terminated=False,
    )


def h20_action_from_tensor(
    policy: torch.nn.Module,
    priorities: np.ndarray | torch.Tensor,
    legal_destinations: tuple[int, ...],
) -> int:
    tensor = torch.as_tensor(priorities, dtype=torch.float32, device="cpu").unsqueeze(0)
    forbidden = torch.ones((1, tensor.shape[1]), dtype=torch.bool, device="cpu")
    forbidden[0, list(legal_destinations)] = False
    with torch.inference_mode():
        log_probabilities = policy.action_log_probabilities(tensor, forbidden)
        action = int(log_probabilities.argmax(dim=-1).item())
    if action not in legal_destinations:
        raise ValueError(f"frozen H20 selected illegal destination {action}")
    return action


def h20_action(frame: PublicDecisionFrame, policy: torch.nn.Module) -> int:
    legacy = public_frame_to_legacy_state(frame)
    priorities = Online2DPriorityAdapter(frame.max_tiers).build(legacy)
    return h20_action_from_tensor(policy, priorities, frame.legal_destinations)


def gp_action(frame: PublicDecisionFrame, rule: Mapping[str, Any]) -> int:
    legacy = public_frame_to_legacy_state(frame)
    action = int(
        rule["selector"](
            legacy, frame.legal_destinations, max_tiers=frame.max_tiers
        )
    )
    if action not in frame.legal_destinations:
        raise ValueError(f"frozen GP rule selected illegal destination {action}")
    return action


def select_gp_training_winner() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify all 120 frozen candidates, then select by training fitness only."""

    rules = load_gp_h20_rules()
    if len(rules) != 120:
        raise ValueError(f"expected 120 GP candidates, found {len(rules)}")
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for rule in rules:
        run_path = Path(str(rule["run_complete_path"]))
        train_path = run_path.with_name("train_complete.json")
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        train_record = json.loads(train_path.read_text(encoding="utf-8"))
        if run_record["best_training_relocations"] != train_record["best_training_relocations"]:
            raise ValueError(f"GP train/run fitness mismatch: {run_path.parent}")
        if run_record["train_manifest_sha256"] != train_record["train_manifest_sha256"]:
            raise ValueError(f"GP train/run manifest mismatch: {run_path.parent}")
        if int(rule["best_training_relocations"]) != int(train_record["best_training_relocations"]):
            raise ValueError(f"GP loader fitness differs from training receipt: {run_path.parent}")
        rows.append((rule, train_record, run_record))

    contracts = {
        tuple(train.get(key) for key in TRAIN_CONTRACT_KEYS)
        for _rule, train, _run in rows
    }
    if len(contracts) != 1:
        raise ValueError("GP candidate training budgets/config/manifests are not identical")
    manifest_paths = {str(train["train_manifest"]) for _rule, train, _run in rows}
    manifest_hashes = {str(train["train_manifest_sha256"]) for _rule, train, _run in rows}
    if len(manifest_paths) != 1 or len(manifest_hashes) != 1:
        raise ValueError("GP candidates do not share one training manifest")
    manifest_path = Path(next(iter(manifest_paths)))
    if not manifest_path.is_file() or file_sha256(manifest_path) != next(iter(manifest_hashes)):
        raise ValueError("GP training manifest is missing or its hash differs from receipts")

    selected = min(
        (row[0] for row in rows),
        key=lambda rule: (
            int(rule["best_training_relocations"]),
            GP_SET_PRIORITY[str(rule["terminal_set"])],
            int(rule["run"]),
        ),
    )
    train_path = Path(str(selected["run_complete_path"])).with_name("train_complete.json")
    training_record = json.loads(train_path.read_text(encoding="utf-8"))
    receipt = {
        "candidate_count": len(rules),
        "candidate_training_contract_unique_count": len(contracts),
        "selection_rule": "minimum best_training_relocations; ties terminal set C/C_E/C_O/C_E_O then run number",
        "selected_method": selected["method"],
        "selected_terminal_set": selected["terminal_set"],
        "selected_run": int(selected["run"]),
        "selected_seed": int(selected["seed"]),
        "selected_best_training_relocations": int(selected["best_training_relocations"]),
        "selected_best_expression": selected["best_expression"],
        "selected_best_prefix": training_record["best_prefix"],
        "train_manifest_path": str(manifest_path.resolve()),
        "train_manifest_sha256": file_sha256(manifest_path),
        "train_budget_contract": {
            key: training_record.get(key) for key in TRAIN_CONTRACT_KEYS
        },
        "train_receipt_path": str(train_path.resolve()),
        "train_receipt_sha256": file_sha256(train_path),
        "run_receipt_path": str(Path(str(selected["run_complete_path"])).resolve()),
        "run_receipt_sha256": file_sha256(Path(str(selected["run_complete_path"]))),
        "loader_reported_run_receipt_sha256": selected["checkpoint_sha256"],
        "selection_source": "training receipts only; no N3 result or test/select/screen performance read",
    }
    return selected, receipt


def load_frozen_h20() -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    checkpoint, policy, actual_sha = load_h20_base_checkpoint(H20_PATH)
    if actual_sha != H20_BASE_SHA256:
        raise ValueError("H20 checkpoint differs from the fixed PLAN-037 hash")
    policy.to(device="cpu", dtype=torch.float32)
    policy.eval()
    policy.requires_grad_(False)
    state_sha = tensor_state_sha256(policy.state_dict())
    config = checkpoint["online_crp_training_config"]
    if int(config["lookahead_horizon"]) != 20 or float(policy.clip_constant) != 5.0:
        raise ValueError("loaded H20 architecture or score clip differs from frozen checkpoint")
    if policy.training or any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("frozen H20 must be eval with every parameter gradient disabled")
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in policy.parameters()):
        raise ValueError("P0 H20 inference must use CPU FP32")
    receipt = {
        "checkpoint_path": H20_PATH.relative_to(WORKSPACE).as_posix(),
        "checkpoint_sha256": actual_sha,
        "architecture": checkpoint["policy_architecture"],
        "original_training_config": config,
        "lookahead_horizon": int(config["lookahead_horizon"]),
        "clip_constant": float(policy.clip_constant),
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "state_tensor_sha256_before_and_after": state_sha,
        "device": "cpu",
        "dtype": "float32",
        "eval": True,
        "all_requires_grad_false": True,
        "optimizer_or_training": False,
    }
    return checkpoint, policy, receipt


def n24_fixture(
    *, sequence: tuple[int, ...] | None = None, label_map: Mapping[int, int] | None = None
) -> OnlineCRPEpisode:
    base_stacks = tuple(tuple(range(start, start + 4)) for start in range(1, 25, 4))
    true_order = tuple(range(1, 25)) if sequence is None else sequence
    mapping = label_map or {value: value for value in range(1, 25)}
    stacks = tuple(tuple(mapping[value] for value in stack) for stack in base_stacks)
    renamed_sequence = tuple(mapping[value] for value in true_order)
    return OnlineCRPEpisode(
        episode_id="plan037-p0-n24-fixture",
        num_stacks=6,
        max_tiers=5,
        initial_stacks=stacks,
        retrieval_sequence=renamed_sequence,
        parameter_group="P0-N24",
        metadata={"split": "correctness_fixture"},
    )


def initial_public_frame(
    episode: OnlineCRPEpisode,
    *,
    repeat: int = 0,
    forecast_future: tuple[int, ...] | None = None,
) -> PublicDecisionFrame:
    environment = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    count = int(environment.state.retrieval_count)
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=0.0,
        seed=semantic_forecast_seed("plan037-p0-public-fixture", repeat),
        q_prior=0.5,
        max_horizon=20,
        forecast_future=forecast_future,
    )
    sync_retrieval_events(provider, episode.retrieval_sequence, 0, count)
    observation = provider.public_observation(
        _private_physical_stacks(environment),
        environment.legal_destinations(),
        max_tiers=episode.max_tiers,
    )
    return make_public_frame(
        observation,
        environment.state,
        environment.legal_destinations(),
        max_tiers=episode.max_tiers,
    )


def _trace_fixture(
    rule: Mapping[str, Any], policy: torch.nn.Module, episode: OnlineCRPEpisode
) -> dict[str, Any]:
    live = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    exact = OnlineCRPEnvironment(episode, lookahead_horizon=20)
    adapter = Online2DPriorityAdapter(episode.max_tiers)
    fingerprint = hashlib.sha256(
        json.dumps(
            [episode.num_stacks, episode.max_tiers, episode.initial_stacks, episode.retrieval_sequence],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=0.0,
        seed=semantic_forecast_seed(fingerprint, 0),
        q_prior=0.5,
        max_horizon=20,
    )
    start_count = int(live.state.retrieval_count)
    sync_retrieval_events(provider, episode.retrieval_sequence, 0, start_count)
    initial_observation = provider.public_observation(
        _private_physical_stacks(live), live.legal_destinations(), max_tiers=episode.max_tiers
    )
    initial_frame = make_public_frame(
        initial_observation, live.state, live.legal_destinations(), max_tiers=episode.max_tiers
    )
    initial_legacy = public_frame_to_legacy_state(initial_frame)
    unknown_count = int(
        np.count_nonzero(
            initial_observation.occupied_mask
            & (initial_observation.rank_tensor == PUBLIC_UNKNOWN_RANK)
        )
    )
    empty_count = int(np.count_nonzero(~initial_observation.occupied_mask))
    if unknown_count == 0 or empty_count == 0 or episode.num_containers <= 21:
        raise AssertionError("P0 fixture must exercise N>21, occupied unknowns, and empty padding")

    decisions = 0
    compared_gp_terminals = 0
    gp_actions: list[int] = []
    h20_actions: list[int] = []
    fixed_actions: list[int] = []
    while not live.state.terminated:
        legal = live.legal_destinations()
        observation = provider.public_observation(
            _private_physical_stacks(live), legal, max_tiers=episode.max_tiers
        )
        frame = make_public_frame(observation, live.state, legal, max_tiers=episode.max_tiers)
        projected = public_frame_to_legacy_state(frame)
        exact_state = exact.state
        if projected.stacks != exact_state.stacks:
            raise AssertionError(f"P0 exact public ranks differ at decision {decisions}")
        if (
            projected.crane_position != exact_state.crane_position
            or projected.crane_time != exact_state.crane_time
            or projected.retrieval_count != exact_state.retrieval_count
            or projected.relocation_count != exact_state.relocation_count
        ):
            raise AssertionError(f"P0 public physical fields differ at decision {decisions}")
        projected_tensor = adapter.build(projected)
        exact_tensor = adapter.build(exact_state)
        if not np.array_equal(projected_tensor, exact_tensor):
            raise AssertionError(f"P0 H20 tensor differs at decision {decisions}")
        for destination in legal:
            projected_terminals = terminal_features_public(projected, destination, episode.max_tiers)
            exact_terminals = terminal_features_public(exact_state, destination, episode.max_tiers)
            if projected_terminals != exact_terminals:
                raise AssertionError("P0 GP terminal vector differs from original exact H20 state")
            compared_gp_terminals += 1

        selected_gp_action = gp_action(frame, rule)
        exact_gp_action = int(
            rule["selector"](exact_state, legal, max_tiers=episode.max_tiers)
        )
        if selected_gp_action != exact_gp_action:
            raise AssertionError(f"P0 GP actor differs at decision {decisions}")
        selected_h20_action = h20_action(frame, policy)
        exact_h20_action = h20_action_from_tensor(policy, exact_tensor, legal)
        if selected_h20_action != exact_h20_action:
            raise AssertionError(f"P0 H20 greedy action differs at decision {decisions}")
        gp_actions.append(selected_gp_action)
        h20_actions.append(selected_h20_action)

        fixed = legal[0]
        fixed_actions.append(fixed)
        before = int(live.state.retrieval_count)
        live.step(fixed)
        exact.step(fixed)
        after = int(live.state.retrieval_count)
        sync_retrieval_events(
            provider, episode.retrieval_sequence, before, after
        )
        decisions += 1
        if decisions > episode.num_containers * (episode.max_tiers - 1):
            raise RuntimeError("P0 fixed-action trace exceeded the relocation safety bound")

    if not live.state.terminated or not exact.state.terminated:
        raise AssertionError("P0 fixed-action trace did not complete")
    return {
        "fixture": "S6_T5_N24_identity_order",
        "decisions": decisions,
        "complete": True,
        "illegal_actions": 0,
        "occupied_unknown_cells_at_start": unknown_count,
        "public_unknown_rank_sentinel": PUBLIC_UNKNOWN_RANK,
        "empty_padding_cells_at_start": empty_count,
        "compared_exact_public_state_decisions": decisions,
        "compared_gp_terminal_vectors": compared_gp_terminals,
        "gp_action_trace_sha256": hashlib.sha256(bytes(gp_actions)).hexdigest(),
        "h20_greedy_trace_sha256": hashlib.sha256(bytes(h20_actions)).hexdigest(),
        "fixed_action_trace_sha256": hashlib.sha256(bytes(fixed_actions)).hexdigest(),
        "relocations_for_correctness_trace_only": live.relocation_count,
    }


def terminal_features_public(
    state: OnlineCRPPublicState, destination: int, max_tiers: int
) -> tuple[float, ...]:
    from GP.simulator import terminal_features

    return tuple(terminal_features(state, destination, max_tiers=max_tiers))


def asset_preflight(test_stdout_path: Path) -> dict[str, Any]:
    selected_rule, gp_receipt = select_gp_training_winner()
    checkpoint, policy, h20_receipt = load_frozen_h20()
    state_before = tensor_state_sha256(policy.state_dict())
    trace = _trace_fixture(selected_rule, policy, n24_fixture())
    state_after = tensor_state_sha256(policy.state_dict())
    if state_before != state_after:
        raise AssertionError("frozen H20 state changed during inference")
    if not test_stdout_path.is_file():
        raise FileNotFoundError("P0 exactness tests must pass before receipt creation")
    test_text = test_stdout_path.read_text(encoding="utf-8-sig")
    if " passed" not in test_text or " failed" in test_text:
        raise ValueError("P0 targeted test stdout does not record a clean pass")
    command = next(
        (line.removeprefix("COMMAND:").strip() for line in test_text.splitlines() if line.startswith("COMMAND:")),
        None,
    )
    summary = next(
        (line.strip() for line in reversed(test_text.splitlines()) if " passed" in line),
        None,
    )
    if not command or not summary:
        raise ValueError("P0 test stdout lacks the exact command or pass summary")
    source_paths = (
        ROOT / "experiments/noise_aware_reference_p0.py",
        ROOT / "tests/test_noise_aware_reference_p0.py",
        WORKSPACE / "Mijn_RL/scrp/attention_evaluation.py",
        WORKSPACE / "Mijn_RL/scrp/h20_a1_policy.py",
        WORKSPACE / "Mijn_RL/scrp/scale_diverse_policy.py",
        WORKSPACE / "Mijn_RL/scrp/online_crp.py",
        WORKSPACE / "GP/simulator.py",
        WORKSPACE / "GP/expression.py",
    )
    n3_rows = OUTPUT / "N3_SCREEN_ROWS.jsonl"
    record = {
        "phase": "PLAN-037 P0 only",
        "status": "PASS_ASSETS_AND_BRIDGE_CORRECTNESS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scoring": False,
        "P1_reference_scoring": False,
        "P2_bayes_dp": False,
        "selected_gp": gp_receipt,
        "frozen_h20": h20_receipt,
        "exact_p0_fixed_action_trace": trace,
        "h20_state_sha256_before_after_trace": {
            "before": state_before,
            "after": state_after,
            "unchanged": state_before == state_after,
        },
        "targeted_test": {
            "command": command,
            "summary": summary,
            "stdout_path": test_stdout_path.relative_to(ROOT).as_posix(),
            "stdout_sha256": file_sha256(test_stdout_path),
        },
        "n3_screen_identity_only": {
            "rows_path": n3_rows.relative_to(ROOT).as_posix(),
            "rows_sha256": file_sha256(n3_rows),
            "read_for_identity_or_interface": True,
            "scored_or_modified": False,
        },
        "source_sha256": {
            path.relative_to(WORKSPACE if path.is_relative_to(WORKSPACE) else ROOT).as_posix(): file_sha256(path)
            for path in source_paths
        },
        "H20_architecture_config_clip_modified": False,
        "optimizer_gpu_or_training": False,
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-stdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    record = asset_preflight(args.test_stdout.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "N4_P0_ASSET_RECEIPT.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path}")
    receipt_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"P0 COMPLETE gp={record['selected_gp']['selected_method']} "
        f"train_R={record['selected_gp']['selected_best_training_relocations']} "
        f"h20_sha256={record['frozen_h20']['checkpoint_sha256']} "
        f"trace_decisions={record['exact_p0_fixed_action_trace']['decisions']} "
        f"score=false P1=false P2=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
