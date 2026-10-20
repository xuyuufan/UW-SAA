"""Unified public-state evaluation for attention policies and scalar baselines."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .attention_data import file_sha256
from .attention_policy import ATTENTION_POLICY_V1, AttentionPolicyConfig, AttentionRelocationPolicy
from .online_crp import (
    LevelingBaseline,
    OnlineCRPEpisode,
    OnlineCRPEnvironment,
    OnlineCRPPublicState,
)
from .scale_diverse_policy import Online2DPriorityAdapter
from .vectorized_online_crp import VectorizedOnlineCRPEnvironment


def _policy_autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("evaluation precision must be fp32 or bf16")


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scrp/attention_evaluation.py",
        root / "scrp/attention_policy.py",
        root / "scrp/online_crp.py",
        root / "scrp/vectorized_online_crp.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def load_attention_checkpoint(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[AttentionRelocationPolicy, dict[str, object]]:
    """Load only the new full policy architecture; reject V1/adapter weights."""

    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != ATTENTION_POLICY_V1:
        raise ValueError("checkpoint is not a full online_crp_attention_v1 policy")
    config_record = checkpoint.get("architecture_config")
    if not isinstance(config_record, Mapping) or config_record.get("architecture") != ATTENTION_POLICY_V1:
        raise ValueError("checkpoint lacks a matching attention architecture config")
    allowed = {
        key: config_record[key]
        for key in (
            "max_lookahead",
            "embed_dim",
            "intra_stack_layers",
            "inter_stack_layers",
            "num_heads",
            "ffn_dim",
            "clip_constant",
        )
    }
    policy = AttentionRelocationPolicy(AttentionPolicyConfig(**allowed)).to(device)
    policy.load_state_dict(checkpoint["model_state"], strict=True)
    policy.eval()
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "training_method": checkpoint.get("training_method"),
        "network_seed": checkpoint.get("network_seed"),
        "alpha": checkpoint.get("alpha"),
        "global_step": checkpoint.get("global_step"),
        "run_contract_sha256": checkpoint.get("run_contract_sha256"),
    }
    return policy, metadata


def load_legacy_v1_checkpoint(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
):
    """Load the full historical LSTM-attention V1 checkpoint as a reference."""

    from .scale_diverse_policy import SCALE_DIVERSE_POLICY_V1, ScaleDiverseRelocationPolicy

    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("online_crp_training_config", {})
    if checkpoint.get("policy_architecture") != SCALE_DIVERSE_POLICY_V1:
        raise ValueError("checkpoint is not the historical scale-diverse V1 architecture")
    if config.get("policy_architecture") != SCALE_DIVERSE_POLICY_V1:
        raise ValueError("historical checkpoint config architecture does not match")
    policy = ScaleDiverseRelocationPolicy(
        lookahead_horizon=int(config["lookahead_horizon"]),
        embed_dim=int(config["embed_dim"]),
        num_encoder_layers=int(config["num_encoder_layers"]),
        num_heads=int(config["num_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        clip_constant=float(config["clip_constant"]),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()
    return policy, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "training_method": "frozen_v1_historical_reference",
        "network_seed": int(config.get("seed", 20260901)),
        "completed_epoch": checkpoint.get("completed_epoch"),
        "policy_architecture": SCALE_DIVERSE_POLICY_V1,
    }


def load_gp_h20_rules(
    gp_results_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Load all 120 frozen H=20 rules, checking their completion provenance."""

    workspace_root = Path(__file__).resolve().parents[2]
    if gp_results_root is None:
        gp_results_root = workspace_root / "GP/results/h20_four_terminal_sets/runs/H20"
    result_root = Path(gp_results_root).resolve()
    workspace_text = str(workspace_root.resolve())
    if workspace_text not in sys.path:
        sys.path.insert(0, workspace_text)
    from GP.expression import Node
    from GP.simulator import _evaluate_postfix_numba, terminal_features

    rules: list[dict[str, object]] = []
    for terminal_set in ("C", "C_E", "C_O", "C_E_O"):
        for run in range(1, 31):
            run_dir = result_root / terminal_set / f"run_{run:02d}"
            complete_path = run_dir / "run_complete.json"
            train_path = run_dir / "train_complete.json"
            if not complete_path.is_file() or not train_path.is_file():
                raise FileNotFoundError(f"incomplete frozen GP H20 run: {run_dir}")
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            trained = json.loads(train_path.read_text(encoding="utf-8"))
            expected_set = terminal_set.replace("_", "+")
            for record in (complete, trained):
                if (
                    record.get("horizon") != 20
                    or record.get("visible_target_count") != 21
                    or record.get("terminal_set") != expected_set
                    or record.get("run") != run
                    or record.get("protocol") != "durasevic2024_gp_h20_four_terminal_sets_caserta_v1"
                ):
                    raise ValueError(f"GP rule metadata mismatch: {complete_path}")
            if complete.get("best_prefix") != trained.get("best_prefix"):
                raise ValueError(f"GP train/eval rule differs: {run_dir}")
            tree = Node.from_prefix(complete["best_prefix"])
            postfix = tree.postfix()

            def selector(state, legal, *, max_tiers, selected_tokens=postfix):
                priorities = [
                    float(
                        _evaluate_postfix_numba(
                            selected_tokens,
                            np.asarray(
                                terminal_features(state, destination, max_tiers=int(max_tiers)),
                                dtype=np.float64,
                            ),
                        )
                    )
                    for destination in legal
                ]
                return int(min(zip(priorities, legal, strict=True))[1])

            rules.append(
                {
                    "method": f"GP-H20-{terminal_set}-run{run:02d}",
                    "selector": selector,
                    "model_seed": int(complete["seed"]),
                    "checkpoint_sha256": file_sha256(complete_path),
                    "terminal_set": expected_set,
                    "run": run,
                    "seed": int(complete["seed"]),
                    "best_training_relocations": int(complete["best_training_relocations"]),
                    "best_expression": str(complete["best_expression"]),
                    "run_complete_path": str(complete_path),
                    "train_manifest_sha256": complete["train_manifest_sha256"],
                }
            )
    if len(rules) != 120:
        raise AssertionError(f"expected 120 frozen H20 GP rules, found {len(rules)}")
    return rules


def _make_row(
    episode: OnlineCRPEpisode,
    *,
    method: str,
    lookahead_horizon: int,
    relocations: int | None,
    completed: bool,
    failure_reason: str | None,
    illegal_actions: int,
    actions: Sequence[int],
    decision_count: int,
    solve_seconds: float,
    inference_seconds: float,
    hardware: str,
    precision: str,
    model_seed: int | None,
    checkpoint_sha256: str | None,
    protocol_sha256: str,
    code_sha256: str,
) -> dict[str, object]:
    return {
        "method": method,
        "model_seed": model_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256,
        "world_id": episode.episode_id,
        "split": str(episode.metadata.get("split", "external")),
        "canonical_id": episode.canonical_id,
        "label_invariant_fingerprint": _label_fingerprint(episode),
        "domain_id": str(episode.metadata.get("domain_id", episode.parameter_group)),
        "domain_kind": str(episode.metadata.get("domain_kind", "external")),
        "S": episode.num_stacks,
        "T": episode.max_tiers,
        "N": episode.num_containers,
        "fill_rate": episode.fill_rate,
        "n": int(lookahead_horizon),
        "R": relocations if completed else None,
        "R_over_N": relocations / episode.num_containers if completed and relocations is not None else None,
        "completed": bool(completed),
        "failure_reason": failure_reason,
        "illegal_actions": int(illegal_actions),
        "decision_count": int(decision_count),
        "actions": [int(action) for action in actions],
        "solve_seconds": float(solve_seconds),
        "inference_seconds": float(inference_seconds),
        "hardware": hardware,
        "precision": precision,
        "code_sha256": code_sha256,
    }


def _label_fingerprint(episode: OnlineCRPEpisode) -> str:
    from .attention_data import label_invariant_fingerprint

    return label_invariant_fingerprint(episode)


def _evaluate_neural_scalar(
    policy: AttentionRelocationPolicy,
    episode: OnlineCRPEpisode,
    *,
    method: str,
    lookahead_horizon: int,
    device: torch.device,
    max_steps: int,
    precision: str,
    hardware: str,
    model_seed: int | None,
    checkpoint_sha256: str | None,
    protocol_sha256: str,
    code_sha256: str,
) -> dict[str, object]:
    environment = OnlineCRPEnvironment(
        episode, lookahead_horizon=lookahead_horizon, max_steps=max_steps
    )
    adapter = Online2DPriorityAdapter(episode.max_tiers)
    actions: list[int] = []
    inference_seconds = 0.0
    started = time.perf_counter()
    illegal_actions = 0
    failure: str | None = None
    try:
        while not environment.state.terminated:
            state = environment.state
            legal = environment.legal_destinations()
            priorities = torch.as_tensor(adapter.build(state), dtype=torch.float32, device=device)[None]
            forbidden = torch.ones((1, episode.num_stacks), dtype=torch.bool, device=device)
            forbidden[0, list(legal)] = False
            if device.type == "cuda":
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad(), _policy_autocast(device, precision):
                    logp = policy.action_log_probabilities(priorities, forbidden)
                end_event.record()
                end_event.synchronize()
                inference_seconds += start_event.elapsed_time(end_event) / 1000.0
            else:
                before = time.perf_counter()
                with torch.no_grad(), _policy_autocast(device, precision):
                    logp = policy.action_log_probabilities(priorities, forbidden)
                inference_seconds += time.perf_counter() - before
            action = int(logp.argmax(dim=-1).item())
            actions.append(action)
            if action not in legal:
                illegal_actions += 1
                raise ValueError(f"policy selected illegal destination {action}")
            environment.step(action)
    except Exception as error:  # record failed episodes without hiding the cause
        failure = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started
    complete = environment.state.terminated
    return _make_row(
        episode,
        method=method,
        lookahead_horizon=lookahead_horizon,
        relocations=environment.relocation_count,
        completed=complete,
        failure_reason=None if complete else (failure or "not_terminated"),
        illegal_actions=illegal_actions,
        actions=actions,
        decision_count=len(actions),
        solve_seconds=elapsed,
        inference_seconds=inference_seconds,
        hardware=hardware,
        precision=precision,
        model_seed=model_seed,
        checkpoint_sha256=checkpoint_sha256,
        protocol_sha256=protocol_sha256,
        code_sha256=code_sha256,
    )


def _evaluate_neural_group(
    policy: AttentionRelocationPolicy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    method: str,
    lookahead_horizon: int,
    device: torch.device,
    max_steps: int,
    precision: str,
    hardware: str,
    model_seed: int | None,
    checkpoint_sha256: str | None,
    protocol_sha256: str,
    code_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[tuple[torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]],
    list[tuple[float, tuple[int, ...]]],
]:
    env = VectorizedOnlineCRPEnvironment(
        episodes,
        lookahead_horizon=lookahead_horizon,
        device=device,
        max_steps=max_steps,
        validate_actions=True,
    )
    action_lists: list[list[int]] = [[] for _ in episodes]
    decision_counts = [0 for _ in episodes]
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
    cpu_latencies: list[tuple[float, tuple[int, ...]]] = []
    inference_seconds = [0.0 for _ in episodes]
    started = time.perf_counter()
    try:
        with torch.no_grad():
            while bool(env.active.any().item()):
                active_rows = torch.nonzero(env.active, as_tuple=False).flatten()
                priorities = env.priorities[active_rows]
                forbidden = env.forbidden_mask[active_rows]
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with _policy_autocast(device, precision):
                        logp = policy.action_log_probabilities(priorities, forbidden)
                    end_event.record()
                    cuda_events.append((start_event, end_event, tuple(int(row) for row in active_rows.cpu().tolist())))
                else:
                    before = time.perf_counter()
                    with _policy_autocast(device, precision):
                        logp = policy.action_log_probabilities(priorities, forbidden)
                    elapsed = time.perf_counter() - before
                    rows_this_step = tuple(int(row) for row in active_rows.cpu().tolist())
                    cpu_latencies.append((elapsed, rows_this_step))
                    per_action = elapsed / max(len(rows_this_step), 1)
                    for row in rows_this_step:
                        inference_seconds[row] += per_action
                actions = logp.argmax(dim=-1)
                selected_legal = env.legal_destination_mask[active_rows].gather(1, actions[:, None]).squeeze(1)
                if not bool(torch.all(selected_legal).item()):
                    raise ValueError("attention policy selected an illegal destination")
                for row, action in zip(active_rows.cpu().tolist(), actions.cpu().tolist(), strict=True):
                    action_lists[row].append(int(action))
                    decision_counts[row] += 1
                env.step_active_(actions)
    except RuntimeError as error:
        # A safety-limit run is rare, but preserve per-instance failure
        # semantics by replaying the batch through the scalar evaluator.
        if "safety limit" not in str(error).lower() and "max_steps" not in str(error).lower():
            raise
        rows = [
            _evaluate_neural_scalar(
                policy, episode, method=method, lookahead_horizon=lookahead_horizon,
                device=device, max_steps=max_steps, precision=precision, hardware=hardware,
                model_seed=model_seed, checkpoint_sha256=checkpoint_sha256,
                protocol_sha256=protocol_sha256, code_sha256=code_sha256,
            )
            for episode in episodes
        ]
        return rows, [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for start_event, end_event, active_rows in cuda_events:
            elapsed = start_event.elapsed_time(end_event) / 1000.0
            per_action = elapsed / max(len(active_rows), 1)
            for row in active_rows:
                inference_seconds[row] += per_action
    group_seconds = time.perf_counter() - started
    relocations = env.relocation_count.detach().cpu().tolist()
    completed = env.terminated.detach().cpu().tolist()
    rows = []
    per_episode_solve = group_seconds / max(len(episodes), 1)
    for index, episode in enumerate(episodes):
        rows.append(
            _make_row(
                episode,
                method=method,
                lookahead_horizon=lookahead_horizon,
                relocations=int(relocations[index]),
                completed=bool(completed[index]),
                failure_reason=None if completed[index] else "safety_limit_or_nontermination",
                illegal_actions=0,
                actions=action_lists[index],
                decision_count=decision_counts[index],
                solve_seconds=per_episode_solve,
                inference_seconds=inference_seconds[index],
                hardware=hardware,
                precision=precision,
                model_seed=model_seed,
                checkpoint_sha256=checkpoint_sha256,
                protocol_sha256=protocol_sha256,
                code_sha256=code_sha256,
            )
        )
    return rows, cuda_events, cpu_latencies


def _summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    completed = [row for row in rows if row["completed"]]
    by_domain: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in completed:
        domain = str(row["domain_id"])
        kind = str(row["domain_kind"])
        value = float(row["R_over_N"])
        by_domain[domain].append(value)
        by_kind[kind][domain].append(value)
    domain_means = {name: float(np.mean(values)) for name, values in sorted(by_domain.items())}
    kind_means = {
        kind: float(np.mean([np.mean(values) for values in groups.values()]))
        for kind, groups in sorted(by_kind.items())
    }
    normalized = [float(row["R_over_N"]) for row in completed]
    relocations = [int(row["R"]) for row in completed]
    total_n = sum(int(row["N"]) for row in completed)
    return {
        "count": len(rows),
        "completed": len(completed),
        "completed_rate": len(completed) / max(len(rows), 1),
        "failed": len(rows) - len(completed),
        "illegal_actions": sum(int(row["illegal_actions"]) for row in rows),
        "mean_r_over_n": float(np.mean(normalized)) if normalized else None,
        "sum_r_over_sum_n": sum(relocations) / total_n if total_n else None,
        "mean_r": float(np.mean(relocations)) if relocations else None,
        "p95_r_over_n": float(np.percentile(normalized, 95)) if normalized else None,
        "macro_mean_r_over_n": float(np.mean(list(domain_means.values()))) if domain_means else None,
        "small_macro_mean_r_over_n": kind_means.get("table2"),
        "large_macro_mean_r_over_n": kind_means.get("large"),
        "unseen_macro_mean_r_over_n": kind_means.get("unseen_geometry"),
        "per_domain_macro_mean_r_over_n": domain_means,
        "inference_seconds_total": float(sum(float(row["inference_seconds"]) for row in rows)),
        "solve_seconds_total_reported": float(sum(float(row["solve_seconds"]) for row in rows)),
    }


def evaluate_episodes(
    policy: AttentionRelocationPolicy,
    episodes: Sequence[OnlineCRPEpisode],
    *,
    lookahead_horizon: int,
    device: str | torch.device,
    max_steps: int = 100_000,
    precision: str = "fp32",
    method: str = "attention",
    model_seed: int | None = None,
    checkpoint_sha256: str | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, object]:
    if not episodes:
        raise ValueError("evaluation manifest must not be empty")
    torch_device = torch.device(device)
    if precision != "fp32":
        raise ValueError("validation and final evaluation must use FP32")
    policy = policy.to(torch_device)
    previous_mode = policy.training
    policy.eval()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "docs/attention_multiscale/PROTOCOL.md"
    resolved_protocol_sha = protocol_sha256 or file_sha256(protocol_path)
    hardware = torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "CPU"
    code_hash = _code_sha256()
    grouped: dict[tuple[int, int], list[OnlineCRPEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.num_stacks, episode.max_tiers)].append(episode)
    rows: list[dict[str, object]] = []
    total_started = time.perf_counter()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
    cpu_latencies: list[tuple[float, tuple[int, ...]]] = []
    for shape in sorted(grouped):
        batch_rows, batch_events, batch_cpu_latencies = _evaluate_neural_group(
            policy,
            grouped[shape],
            method=method,
            lookahead_horizon=lookahead_horizon,
            device=torch_device,
            max_steps=max_steps,
            precision=precision,
            hardware=hardware,
            model_seed=model_seed,
            checkpoint_sha256=checkpoint_sha256,
            protocol_sha256=resolved_protocol_sha,
            code_sha256=code_hash,
        )
        rows.extend(batch_rows)
        cuda_events.extend(batch_events)
        cpu_latencies.extend(batch_cpu_latencies)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    policy.train(previous_mode)
    elapsed = time.perf_counter() - total_started
    gpu_call_ms = [start.elapsed_time(end) for start, end, _ in cuda_events]
    cpu_call_ms = [seconds * 1000.0 for seconds, _ in cpu_latencies]
    call_ms = gpu_call_ms + cpu_call_ms
    per_action_ms = [
        value / max(len(active_rows), 1)
        for value, (_, _, active_rows) in zip(gpu_call_ms, cuda_events, strict=True)
    ]
    per_action_ms.extend(
        value / max(len(active_rows), 1)
        for value, active_rows in cpu_latencies
    )
    summary = _summarize_rows(rows)
    summary.update(
        {
            "method": method,
            "lookahead_horizon": lookahead_horizon,
            "checkpoint_sha256": checkpoint_sha256,
            "protocol_sha256": resolved_protocol_sha,
            "code_sha256": code_hash,
            "hardware": hardware,
            "precision": precision,
            "evaluator_wall_seconds": elapsed,
            "batch_forward_latency_ms_p50": float(np.percentile(call_ms, 50)) if call_ms else None,
            "batch_forward_latency_ms_p95": float(np.percentile(call_ms, 95)) if call_ms else None,
            "per_action_forward_latency_ms_p50": float(np.percentile(per_action_ms, 50)) if per_action_ms else None,
            "per_action_forward_latency_ms_p95": float(np.percentile(per_action_ms, 95)) if per_action_ms else None,
            "rows": rows,
        }
    )
    return summary


def evaluate_selector(
    episodes: Sequence[OnlineCRPEpisode],
    selector: Callable[..., int],
    *,
    method: str,
    lookahead_horizon: int,
    max_steps: int = 100_000,
    hardware: str = "CPU",
    protocol_sha256: str | None = None,
    model_seed: int | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    """Evaluate a public-state scalar rule such as Leveling or a frozen GP rule."""

    if not episodes:
        raise ValueError("evaluation manifest must not be empty")
    root = Path(__file__).resolve().parents[1]
    protocol_hash = protocol_sha256 or file_sha256(root / "docs/attention_multiscale/PROTOCOL.md")
    code_hash = _code_sha256()
    rows: list[dict[str, object]] = []
    started_all = time.perf_counter()
    selector_needs_capacity = "max_tiers" in inspect.signature(selector).parameters
    for episode in episodes:
        environment = OnlineCRPEnvironment(
            episode, max_steps=max_steps, lookahead_horizon=lookahead_horizon
        )
        actions: list[int] = []
        illegal = 0
        failure = None
        start = time.perf_counter()
        try:
            while not environment.state.terminated:
                state = environment.state
                legal = environment.legal_destinations()
                if selector_needs_capacity:
                    action = int(selector(state, legal, max_tiers=episode.max_tiers))
                else:
                    action = int(selector(state, legal))
                actions.append(action)
                if action not in legal:
                    illegal += 1
                    raise ValueError(f"selector returned illegal destination {action}")
                environment.step(action)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        solve = time.perf_counter() - start
        complete = environment.state.terminated
        rows.append(
            _make_row(
                episode,
                method=method,
                lookahead_horizon=lookahead_horizon,
                relocations=environment.relocation_count,
                completed=complete,
                failure_reason=None if complete else (failure or "not_terminated"),
                illegal_actions=illegal,
                actions=actions,
                decision_count=len(actions),
                solve_seconds=solve,
                inference_seconds=0.0,
                hardware=hardware,
                precision="not_applicable",
                model_seed=model_seed,
                checkpoint_sha256=checkpoint_sha256,
                protocol_sha256=protocol_hash,
                code_sha256=code_hash,
            )
        )
    summary = _summarize_rows(rows)
    summary.update(
        {
            "method": method,
            "lookahead_horizon": lookahead_horizon,
            "protocol_sha256": protocol_hash,
            "code_sha256": code_hash,
            "hardware": hardware,
            "precision": "not_applicable",
            "evaluator_wall_seconds": time.perf_counter() - started_all,
            "rows": rows,
        }
    )
    return summary


def leveling_summary(episodes: Sequence[OnlineCRPEpisode], *, lookahead_horizon: int) -> dict[str, object]:
    return evaluate_selector(
        episodes,
        LevelingBaseline.select_destination,
        method="paper_leveling_L",
        lookahead_horizon=lookahead_horizon,
    )


__all__ = [
    "evaluate_episodes",
    "evaluate_selector",
    "leveling_summary",
    "load_attention_checkpoint",
    "load_gp_h20_rules",
    "load_legacy_v1_checkpoint",
]
