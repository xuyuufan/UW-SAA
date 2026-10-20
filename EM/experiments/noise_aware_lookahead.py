"""PLAN-036 data preparation, N1 contract, N2 phenomenon, and N3 fit/select runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np

from scrp.attention_data import (
    _source_paths,
    episode_to_record,
    file_sha256,
    label_invariant_fingerprint,
)
from scrp.noise_aware_em import (
    candidate_actions,
    select_em_destination,
    select_sr_em,
    selector_features,
)
from scrp.noise_aware_forecast import (
    HORIZONS,
    MAX_HORIZON,
    ForecastProvider,
    PublicObservation,
    semantic_forecast_seed,
)
from scrp.online_crp import LevelingBaseline, OnlineCRPEnvironment, OnlineCRPEpisode


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASE_ROOT = WORKSPACE / "Mijn_RL"
DEFAULT_CONFIG = ROOT / "configs/noise_aware_lookahead_v1.json"
OUTPUT = ROOT / "output/noise_aware_lookahead_v1/r01"
N2_OUTPUT = OUTPUT
SCREEN_METHODS = ("EM0", "EM2", "EM5", "EM10", "EM20", "SR", "R", "S", "Leveling")
FROZEN_N3_MODEL_SHA256 = "2e6515de2f6ff5c2f1d64777de6bbc4d2f874d319fdeb7cd3eaf234b1b501571"
FROZEN_N3_SCREEN_DATA_SHA256 = "91cda7a0e9180015592c74fb942416b1c546f2b33ef625c609f928c8d6563aab"


def _source_path_label(path: Path) -> str:
    root = ROOT if path.is_relative_to(ROOT) else WORKSPACE
    return path.relative_to(root).as_posix()


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "noise_aware_lookahead_v1":
        raise ValueError("unexpected protocol_id")
    scope = config.get("release_scope", {})
    allowed_stages = set(scope.get("allowed_stages", ()))
    if not {"contract", "phenomenon"}.issubset(allowed_stages):
        raise ValueError("config must retain the reviewed contract and phenomenon stages")
    if scope.get("phenomenon_scoring") is not True:
        raise ValueError("the reviewed phenomenon stage must be explicitly enabled")
    screen_released = scope.get("screen_evaluation") is True
    if screen_released:
        if "screen" not in allowed_stages or "n3_screen" not in config:
            raise ValueError("screen evaluation requires the reviewed N3 screen release")
    elif scope.get("screen_evaluation") is not False or "screen" in allowed_stages:
        raise ValueError("screen stage/config release state is inconsistent")
    if scope.get("fitting") is True:
        if "fit" not in allowed_stages or "n3_fit_select" not in config:
            raise ValueError("fitting requires the reviewed N3 fit/select release")
    elif scope.get("fitting") is not False or "fit" in allowed_stages:
        raise ValueError("unexpected fitting-stage release state")
    if scope.get("gp_or_rl_training") is not False or scope.get("bayes_dp") is not False:
        raise ValueError("this release must not enable GP/RL training or Bayes DP")
    if tuple(config["forecast"]["horizons"]) != HORIZONS:
        raise ValueError("forecast horizons differ from the reviewed fixed grid")
    if int(config["forecast"]["max_horizon"]) != MAX_HORIZON:
        raise ValueError("max_horizon differs from the reviewed contract")
    if config["forecast"].get("seed_rule") != (
        "sha256(noise-aware-lookahead-v1|semantic_fingerprint|r); excludes p, h, and method"
    ):
        raise ValueError("forecast seed rule differs from the reviewed N2 contract")
    n2 = config["n2_phenomenon"]
    if int(n2["expected_layout_count"]) != 40:
        raise ValueError("N2 must use the frozen 40-layout D_select")
    if int(n2["bootstrap_replicates"]) != 2000 or int(n2["bootstrap_seed"]) != 2026100904:
        raise ValueError("N2 bootstrap settings differ from the preregistered values")
    if int(n2["wall_clock_limit_minutes"]) != 45:
        raise ValueError("N2 wall-clock limit differs from the reviewed protocol")
    if tuple(float(value) for value in config["forecast"]["p_values"]) != (
        0.0,
        0.1,
        0.3,
        0.5,
    ):
        raise ValueError("N2 p grid differs from the reviewed protocol")
    if tuple(int(value) for value in config["forecast"]["repeats"]) != (0, 1):
        raise ValueError("N2 repeats differ from the reviewed protocol")
    if float(n2["primary_p"]) != 0.5 or (int(n2["primary_h_long"]), int(n2["primary_h_short"])) != (20, 5):
        raise ValueError("N2 primary comparison differs from the preregistered contrast")
    if scope.get("fitting") is True:
        n3 = config["n3_fit_select"]
        if int(n3["sampling_seed"]) != 2026100905:
            raise ValueError("N3 state-sampling seed differs from the reviewed protocol")
        if int(n3["states_per_condition_cap"]) != 25 or int(n3["max_label_continuations"]) != 1000:
            raise ValueError("N3 state/continuation budgets differ from the reviewed protocol")
        if float(n3["ridge_lambda"]) != 1.0 or int(n3["fixed_h_baseline"]) != 20:
            raise ValueError("N3 ridge or fixed-H settings differ from the reviewed protocol")
        if tuple(n3["variants"]) != ("SR", "R", "S"):
            raise ValueError("N3 variants differ from the reviewed fixed feature subsets")
        if n3["screen_evaluation"] is not False:
            raise ValueError("N3 fit/select section must not score D_screen")
    if screen_released:
        screen = config["n3_screen"]
        if screen.get("release_review") != "REVIEW-036-04":
            raise ValueError("N3 screen release review id differs")
        if int(screen["expected_layout_count"]) != 80 or screen["expected_by_geometry"] != {
            "G1": 40,
            "G2": 40,
        }:
            raise ValueError("N3 screen layout budget differs from PLAN-036 v1.4")
        if screen["expected_data_sha256"] != FROZEN_N3_SCREEN_DATA_SHA256:
            raise ValueError("N3 screen data hash differs from the released frozen split")
        if screen["frozen_model_sha256"] != FROZEN_N3_MODEL_SHA256:
            raise ValueError("N3 screen model hash differs from the reviewed frozen model")
        if int(screen["q_prior_hits"]) != 5183 or int(screen["q_prior_events"]) != 9920:
            raise ValueError("N3 screen reliability prior differs from D_fit evidence")
        if int(screen["fixed_h_baseline"]) != 20 or tuple(screen["methods"]) != SCREEN_METHODS:
            raise ValueError("N3 screen methods/fixed-H baseline differ from release")
        if int(screen["bootstrap_replicates"]) != 2000 or int(screen["bootstrap_seed"]) != 2026100907:
            raise ValueError("N3 screen bootstrap differs from the fixed protocol")
        if float(screen["ci_level"]) != 0.9 or int(screen["wall_clock_limit_minutes"]) != 20:
            raise ValueError("N3 screen confidence or wall-clock budget differs from release")
    return config


def _seed(master_seed: int, *parts: object) -> int:
    payload = "/".join((str(int(master_seed)), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _rank_layout_episode(
    *,
    episode_id: str,
    geometry_id: str,
    split: str,
    num_stacks: int,
    max_tiers: int,
    num_containers: int,
    master_seed: int,
    index: int,
    retry: int,
) -> OnlineCRPEpisode:
    if num_containers != num_stacks * (max_tiers - 1):
        raise ValueError("reviewed geometries require exactly T-1 boxes per stack")
    rng = np.random.Generator(
        np.random.PCG64(_seed(master_seed, split, geometry_id, index, retry))
    )
    # Tokens are per-episode opaque labels. Their values are unrelated to rank.
    offset = int(rng.integers(100_000, 1_500_000_000))
    token_by_rank = rng.permutation(np.arange(1, num_containers + 1)) + offset
    rank_stacks = rng.permutation(np.arange(1, num_containers + 1)).reshape(
        num_stacks, max_tiers - 1
    )
    initial_stacks = tuple(
        tuple(int(token_by_rank[int(rank) - 1]) for rank in row)
        for row in rank_stacks
    )
    retrieval_sequence = tuple(int(token) for token in token_by_rank)
    return OnlineCRPEpisode(
        episode_id=episode_id,
        num_stacks=num_stacks,
        max_tiers=max_tiers,
        initial_stacks=initial_stacks,
        retrieval_sequence=retrieval_sequence,
        parameter_group=geometry_id,
        metadata={
            "protocol_id": "noise_aware_lookahead_v1",
            "split": split,
            "geometry_id": geometry_id,
            "master_seed": int(master_seed),
            "sample_index": int(index),
            "retry": int(retry),
            "data_role": "development_only_layout",
        },
    )


def _source_manifest_inventory() -> tuple[set[str], list[dict[str, Any]]]:
    """Reuse the established source list and include all existing local episode manifests."""
    required_sources = list(_source_paths(BASE_ROOT, WORKSPACE))
    paths: dict[Path, str] = {}
    for path, category in required_sources:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"required isolation source is missing: {resolved}")
        paths[resolved] = category

    local_data = BASE_ROOT / "data/online_crp"
    for path in local_data.rglob("*.json"):
        if path.name.endswith(".metadata.json") or path.name == "audit.json":
            continue
        paths.setdefault(path.resolve(), "existing_mijn_rl_manifest")

    excluded: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for path, category in sorted(paths.items(), key=lambda item: str(item[0]).casefold()):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            if category != "existing_mijn_rl_manifest":
                raise ValueError(f"required source is not a non-empty episode list: {path}")
            continue
        first = raw[0]
        if not isinstance(first, dict) or not {
            "initial_stacks",
            "retrieval_sequence",
        }.issubset(first):
            if category != "existing_mijn_rl_manifest":
                raise ValueError(f"required source has no episode records: {path}")
            continue
        episodes = tuple(OnlineCRPEpisode.from_record(row) for row in raw)
        fingerprints = {label_invariant_fingerprint(episode) for episode in episodes}
        excluded.update(fingerprints)
        inventory.append(
            {
                "category": category,
                "path": str(path),
                "sha256": file_sha256(path),
                "episode_count": len(episodes),
                "unique_semantic_fingerprints": len(fingerprints),
            }
        )
    if not inventory:
        raise RuntimeError("no existing episode source manifests were verified")
    return excluded, inventory


def _generate_layout_splits(
    config: dict[str, Any], output: Path
) -> dict[str, Any]:
    excluded, sources = _source_manifest_inventory()
    seen = set(excluded)
    generated_fingerprints: set[str] = set()
    split_specs = config["layout_splits"]
    generated: dict[str, list[OnlineCRPEpisode]] = {}
    split_audit: dict[str, Any] = {}
    for split in ("fit", "select", "screen"):
        rows: list[OnlineCRPEpisode] = []
        per_geometry = int(split_specs["per_geometry"][split])
        master_seed = int(split_specs["master_seeds"][split])
        for geometry in split_specs["geometries"]:
            geometry_id = str(geometry["id"])
            for index in range(per_geometry):
                for retry in range(int(split_specs["max_overlap_retries"])):
                    episode = _rank_layout_episode(
                        episode_id=f"noise-r01-{split}-{geometry_id}-{index:03d}",
                        geometry_id=geometry_id,
                        split=split,
                        num_stacks=int(geometry["num_stacks"]),
                        max_tiers=int(geometry["max_tiers"]),
                        num_containers=int(geometry["num_containers"]),
                        master_seed=master_seed,
                        index=index,
                        retry=retry,
                    )
                    fingerprint = label_invariant_fingerprint(episode)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    generated_fingerprints.add(fingerprint)
                    rows.append(episode)
                    break
                else:
                    raise RuntimeError(
                        f"could not isolate {split}/{geometry_id}/{index} after bounded retries"
                    )
        expected = per_geometry * len(split_specs["geometries"])
        if len(rows) != expected:
            raise AssertionError(f"{split} has {len(rows)} layouts, expected {expected}")
        generated[split] = rows
        data_path = output / "data" / f"{split}.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps([episode_to_record(row) for row in rows], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        split_audit[split] = {
            "count": len(rows),
            "by_geometry": {
                geometry_id: sum(
                    episode.parameter_group == geometry_id for episode in rows
                )
                for geometry_id in (
                    str(item["id"]) for item in split_specs["geometries"]
                )
            },
            "sha256": file_sha256(data_path),
            "unique_semantic_fingerprints": len(
                {label_invariant_fingerprint(row) for row in rows}
            ),
        }
    source_overlap = generated_fingerprints.intersection(excluded)
    if source_overlap:
        raise AssertionError("generated development layouts overlap an existing source")
    if len(generated_fingerprints) != sum(len(rows) for rows in generated.values()):
        raise AssertionError("generated layouts overlap within or across development splits")
    audit = {
        "source_manifests": sources,
        "source_fingerprint_union": len(excluded),
        "generated_splits": split_audit,
        "generated_total": sum(len(rows) for rows in generated.values()),
        "generated_source_overlap": 0,
        "generated_split_overlap": 0,
        "fingerprint": "existing label_invariant_fingerprint: geometry + retrieval-rank layout, stack-order invariant",
        "role": "development layouts only; no performance evaluation or sealed-test claim",
    }
    (output / "DATA_SPLIT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def _tiny_episode(index: int, geometry: dict[str, int]) -> OnlineCRPEpisode:
    seed = 2026100911 + int(index)
    return _rank_layout_episode(
        episode_id=f"noise-r01-contract-{index:02d}",
        geometry_id="tiny",
        split="contract",
        num_stacks=int(geometry["num_stacks"]),
        max_tiers=int(geometry["max_tiers"]),
        num_containers=int(geometry["num_containers"]),
        master_seed=seed,
        index=index,
        retry=0,
    )


def sync_retrieval_events(
    provider: ForecastProvider,
    true_sequence: tuple[int, ...],
    before_count: int,
    after_count: int,
    *,
    event_rows: list[dict[str, Any]] | None = None,
    record_q_t_before: bool = False,
) -> int:
    """Replay every newly revealed target in order, including reset-time cascades."""
    if not 0 <= before_count <= after_count <= len(true_sequence):
        raise ValueError("retrieval count moved outside the hidden sequence")
    updates = 0
    for retrieved_index in range(before_count, after_count):
        next_index = retrieved_index + 1
        if next_index < len(true_sequence):
            q_t_before = provider.history.q_t
            hit = provider.observe_reveal(true_sequence[next_index])
            updates += 1
            if event_rows is not None:
                event = {"event_index": provider.event_count, "hit": bool(hit)}
                if record_q_t_before:
                    event["q_t_before_reveal"] = float(q_t_before)
                event_rows.append(event)
    return updates


def _private_physical_stacks(environment: OnlineCRPEnvironment) -> tuple[tuple[int, ...], ...]:
    # Evaluator-only boundary: raw opaque tokens are used only to construct the
    # rank tensor and are never included in PublicObservation.
    return tuple(tuple(int(item) for item in stack) for stack in environment._stacks)


def _assert_scalar_reference(episode: OnlineCRPEpisode, max_steps: int) -> None:
    original = OnlineCRPEnvironment(
        episode, max_steps=max_steps, lookahead_horizon=0
    )
    forecast_view = OnlineCRPEnvironment(
        episode, max_steps=max_steps, lookahead_horizon=MAX_HORIZON
    )
    steps = 0
    while not original.state.terminated:
        if (
            original._stacks != forecast_view._stacks
            or original.current_target_id != forecast_view.current_target_id
            or original.state.retrieval_count != forecast_view.state.retrieval_count
            or original.relocation_count != forecast_view.relocation_count
            or original.state.terminated != forecast_view.state.terminated
            or original.objective_costs != forecast_view.objective_costs
        ):
            raise AssertionError("lookahead visibility changed scalar physical behavior")
        legal = original.legal_destinations()
        if not legal or legal != forecast_view.legal_destinations():
            raise AssertionError("scalar environments disagree on legal actions")
        action = min(legal)
        original.step(action)
        forecast_view.step(action)
        steps += 1
        if steps > max_steps:
            raise AssertionError("fixed-action scalar contract exceeded its safety bound")
    if not forecast_view.state.terminated:
        raise AssertionError("reference environment did not complete")


def _run_tiny_contract(
    config: dict[str, Any], output: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    limits = config["n1_contract"]
    count = int(limits["tiny_episode_count"])
    geometry = {key: int(value) for key, value in limits["tiny_geometry"].items()}
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    max_horizon = int(config["forecast"]["max_horizon"])
    summaries: list[dict[str, Any]] = []
    totals = {
        "unique_tiny_episodes": 0,
        "completed": 0,
        "legal_actions": 0,
        "decisions": 0,
        "revelation_updates": 0,
        "relocations_without_revelation": 0,
        "candidate_h_length_records": 0,
        "fitted_models": 0,
        "scientific_scores": 0,
    }
    for index in range(count):
        episode = _tiny_episode(index, geometry)
        max_steps = episode.num_containers * (episode.max_tiers - 1)
        _assert_scalar_reference(episode, max_steps)

        p = p_values[index % len(p_values)]
        requested_h = HORIZONS[index % len(HORIZONS)]
        provider = ForecastProvider(
            episode.retrieval_sequence,
            p=p,
            seed=2026101001 + index,
            q_prior=float(config["forecast"]["q_prior_contract_fixture"]),
            max_horizon=max_horizon,
        )
        environment = OnlineCRPEnvironment(
            episode, max_steps=max_steps, lookahead_horizon=0
        )
        retrieval_count = int(environment.state.retrieval_count)
        totals["revelation_updates"] += sync_retrieval_events(
            provider, episode.retrieval_sequence, 0, retrieval_count
        )
        decisions = 0
        legal_actions = 0
        no_reveal_steps = 0
        h_lengths: list[dict[str, int]] = []
        while not environment.state.terminated:
            observation = provider.public_observation(
                _private_physical_stacks(environment),
                environment.legal_destinations(),
                max_tiers=episode.max_tiers,
            )
            before_count = int(environment.state.retrieval_count)
            before_events = provider.event_count
            before_q = observation.q_t
            available = provider.remaining_future_count
            for candidate_h in HORIZONS:
                h_lengths.append(
                    {
                        "candidate_requested_h": candidate_h,
                        "effective_prefix_length": min(candidate_h, available),
                    }
                )
            action = select_em_destination(
                observation, max_tiers=episode.max_tiers, h=requested_h
            )
            legal = environment.legal_destinations()
            if action not in legal:
                raise AssertionError("EM selected a destination outside legal actions")
            environment.step(action)
            legal_actions += 1
            after_count = int(environment.state.retrieval_count)
            updates = sync_retrieval_events(
                provider,
                episode.retrieval_sequence,
                before_count,
                after_count,
            )
            totals["revelation_updates"] += updates
            if after_count == before_count:
                no_reveal_steps += 1
                if provider.event_count != before_events or provider.history.q_t != before_q:
                    raise AssertionError("repeated relocation changed reliability without a reveal")
            decisions += 1
            if decisions > max_steps:
                raise AssertionError("tiny contract episode exceeded its safety bound")
        if not environment.state.terminated:
            raise AssertionError("tiny contract episode did not complete")
        if len(h_lengths) != decisions * len(HORIZONS):
            raise AssertionError("candidate horizon metadata is incomplete")
        totals["unique_tiny_episodes"] += 1
        totals["completed"] += 1
        totals["legal_actions"] += legal_actions
        totals["decisions"] += decisions
        totals["relocations_without_revelation"] += no_reveal_steps
        totals["candidate_h_length_records"] += len(h_lengths)
        summaries.append(
            {
                "episode_id": episode.episode_id,
                "geometry": [episode.num_stacks, episode.max_tiers, episode.num_containers],
                "forecast_p_evaluator_only_contract_condition": p,
                "requested_h_contract_condition": requested_h,
                "decision_count": decisions,
                "revelation_updates": provider.event_count,
                "completed": True,
                "candidate_h_lengths": h_lengths,
                "performance_score_computed": False,
                "model_fitted": False,
            }
        )
    if totals["unique_tiny_episodes"] != 16 or totals["completed"] != 16:
        raise AssertionError("N1 contract release requires exactly 16 complete tiny episodes")
    return summaries, totals


def _load_frozen_select(config: dict[str, Any]) -> tuple[list[OnlineCRPEpisode], Path, str]:
    n2 = config["n2_phenomenon"]
    data_path = ROOT / str(n2["input_layouts"])
    n1_report_path = OUTPUT / "N1_CONTRACT_REPORT.json"
    if not data_path.is_file() or not n1_report_path.is_file():
        raise FileNotFoundError("frozen N1 D_select data or report is missing")
    n1_report = json.loads(n1_report_path.read_text(encoding="utf-8"))
    expected_hash = n1_report["data"]["generated_splits"]["select"]["sha256"]
    actual_hash = file_sha256(data_path)
    if actual_hash != expected_hash:
        raise ValueError("r01/data/select.json differs from the N1-frozen SHA256")
    records = json.loads(data_path.read_text(encoding="utf-8"))
    episodes = [OnlineCRPEpisode.from_record(row) for row in records]
    if len(episodes) != int(n2["expected_layout_count"]):
        raise ValueError(f"frozen D_select has {len(episodes)} layouts, expected 40")
    fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
    if len(set(fingerprints)) != len(episodes):
        raise ValueError("frozen D_select contains duplicate semantic layouts")
    expected_per_geometry = {
        str(item["id"]): int(config["layout_splits"]["per_geometry"]["select"])
        for item in config["layout_splits"]["geometries"]
    }
    actual_per_geometry = {
        geometry_id: sum(
            episode.parameter_group == geometry_id for episode in episodes
        )
        for geometry_id in expected_per_geometry
    }
    if actual_per_geometry != expected_per_geometry:
        raise ValueError(f"frozen D_select geometry counts differ: {actual_per_geometry}")
    if any(episode.metadata.get("split") != "select" for episode in episodes):
        raise ValueError("N2 accepts only the frozen D_select split")
    return episodes, data_path, actual_hash


def _n2_source_receipt(config_path: Path, data_path: Path, data_hash: str) -> dict[str, Any]:
    source_paths = (
        ROOT / "scrp/noise_aware_forecast.py",
        ROOT / "scrp/noise_aware_em.py",
        ROOT / "experiments/noise_aware_lookahead.py",
        BASE_ROOT / "scrp/online_crp.py",
        BASE_ROOT / "scrp/attention_data.py",
        ROOT / "tests/test_noise_aware_lookahead.py",
    )
    sources = {
        _source_path_label(path): file_sha256(path) for path in source_paths
    }
    config_hash = file_sha256(config_path)
    source_bundle = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_sha256": sources,
        "source_bundle_sha256": source_bundle,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "data_path": data_path.relative_to(ROOT).as_posix(),
        "data_sha256": data_hash,
    }


def _forecast_hash(forecast: tuple[int, ...]) -> str:
    payload = json.dumps(list(forecast), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _initial_forecast_metrics(
    true_future: tuple[int, ...], forecast: tuple[int, ...], h: int
) -> tuple[float, float, float | None]:
    if len(true_future) != len(forecast) or set(true_future) != set(forecast):
        raise ValueError("forecast metrics require a legal full-future permutation")
    true_position = {token: index for index, token in enumerate(true_future)}
    predicted_positions = [true_position[token] for token in forecast]
    pair_count = len(true_future) * (len(true_future) - 1) // 2
    inversions = sum(
        predicted_positions[left] > predicted_positions[right]
        for left in range(len(predicted_positions))
        for right in range(left + 1, len(predicted_positions))
    )
    kendall = float(inversions / pair_count) if pair_count else 0.0
    displacement = float(
        np.mean(
            [
                abs(predicted_position - true_position[token])
                for predicted_position, token in enumerate(forecast)
            ]
        )
    ) if forecast else 0.0
    if h == 0:
        membership_error = None
    else:
        effective_h = min(h, len(true_future))
        correct = len(set(true_future[:effective_h]).intersection(forecast[:effective_h]))
        membership_error = float(1.0 - correct / effective_h)
    return kendall, displacement, membership_error


def _run_em_h_rollout(
    episode: OnlineCRPEpisode,
    *,
    fingerprint: str,
    p: float,
    repeat: int,
    h: int,
    seed: int,
    shared_forecast: tuple[int, ...],
    forecast_hash: str,
    noise_metrics: tuple[float, float, float | None],
    q_prior: float,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    decision_count = 0
    illegal_action_count = 0
    environment: OnlineCRPEnvironment | None = None
    error: str | None = None
    started = time.perf_counter()
    try:
        # Each baseline gets a fresh causal history but exactly the same private
        # full forecast; its detached public tensor is truncated to this h.
        provider = ForecastProvider(
            episode.retrieval_sequence,
            p=p,
            seed=seed,
            q_prior=q_prior,
            max_horizon=h,
            forecast_future=shared_forecast,
        )
        environment = OnlineCRPEnvironment(
            episode, max_steps=max_steps, lookahead_horizon=0
        )
        retrieval_count = int(environment.state.retrieval_count)
        sync_retrieval_events(
            provider, episode.retrieval_sequence, 0, retrieval_count
        )
        while not environment.state.terminated:
            legal = environment.legal_destinations()
            observation = provider.public_observation(
                _private_physical_stacks(environment),
                legal,
                max_tiers=episode.max_tiers,
            )
            # The policy boundary is exactly (detached public observation, h).
            action = select_em_destination(
                observation, max_tiers=episode.max_tiers, h=h
            )
            decision_count += 1
            if action not in legal:
                illegal_action_count += 1
                raise ValueError("EM selected an illegal destination")
            before_count = int(environment.state.retrieval_count)
            environment.step(action)
            after_count = int(environment.state.retrieval_count)
            sync_retrieval_events(
                provider,
                episode.retrieval_sequence,
                before_count,
                after_count,
            )
            if decision_count > max_steps:
                raise RuntimeError("rollout exceeded R <= N(T-1) safety bound")
    except Exception as exc:  # preserve the failed condition as a partial row
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    relocation_count = int(environment.relocation_count) if environment is not None else 0
    retrieved_count = int(environment.state.retrieval_count) if environment is not None else 0
    completed = bool(
        environment is not None
        and environment.state.terminated
        and retrieved_count == episode.num_containers
        and error is None
    )
    row: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "fingerprint": fingerprint,
        "geometry_id": episode.parameter_group,
        "num_stacks": episode.num_stacks,
        "max_tiers": episode.max_tiers,
        "N": episode.num_containers,
        "split": "select",
        "p_evaluator_only": float(p),
        "r": int(repeat),
        "h": int(h),
        "initial_forecast_hash": forecast_hash,
        "R": relocation_count,
        "RN": float(relocation_count / episode.num_containers),
        "decision_count": int(decision_count),
        "completed": completed,
        "retrieved_count": retrieved_count,
        "illegal_action_count": int(illegal_action_count),
        "elapsed_seconds_including_em_selection": float(elapsed),
        "evaluator_initial_normalized_kendall": noise_metrics[0],
        "evaluator_initial_mean_abs_rank_displacement": noise_metrics[1],
        "evaluator_prefix_membership_error": noise_metrics[2],
        "source_bundle_sha256": receipt["source_bundle_sha256"],
        "config_sha256": receipt["config_sha256"],
        "data_sha256": receipt["data_sha256"],
    }
    if error is not None:
        row["error"] = error
    return row


def _verify_phenomenon_rows(
    rows: list[dict[str, Any]],
    episodes: list[OnlineCRPEpisode],
    config: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    episode_by_fp = {
        label_invariant_fingerprint(episode): episode for episode in episodes
    }
    expected_keys = {
        (fingerprint, p, repeat, h)
        for fingerprint in episode_by_fp
        for p in p_values
        for repeat in repeats
        for h in horizons
    }
    row_by_key: dict[tuple[str, float, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r"]),
            int(row["h"]),
        )
        if key in row_by_key:
            raise AssertionError(f"duplicate N2 condition key: {key}")
        row_by_key[key] = row
        episode = episode_by_fp.get(key[0])
        if episode is None:
            raise AssertionError("N2 row contains a layout outside frozen D_select")
        if row["source_bundle_sha256"] != receipt["source_bundle_sha256"]:
            raise AssertionError("row source bundle hash differs from N2 receipt")
        if row["config_sha256"] != receipt["config_sha256"]:
            raise AssertionError("row config hash differs from N2 receipt")
        if row["data_sha256"] != receipt["data_sha256"]:
            raise AssertionError("row data hash differs from N2 receipt")
        if row["geometry_id"] != episode.parameter_group or row["N"] != episode.num_containers:
            raise AssertionError("N2 row geometry/layout identity mismatch")
        bound = episode.num_containers * (episode.max_tiers - 1)
        if not row["completed"] or row["retrieved_count"] != episode.num_containers:
            raise AssertionError("N2 contains an incomplete retrieval episode")
        if row["illegal_action_count"] != 0:
            raise AssertionError("N2 contains an illegal relocation action")
        if row["R"] > bound or row["decision_count"] != row["R"]:
            raise AssertionError("N2 relocation count violates the physical safety bound")
        if not np.isclose(row["RN"], row["R"] / episode.num_containers, rtol=0, atol=1e-12):
            raise AssertionError("N2 R/N arithmetic mismatch")
    if set(row_by_key) != expected_keys or len(rows) != 1600:
        raise AssertionError(
            f"N2 keys incomplete: found={len(rows)} unique={len(row_by_key)} expected=1600"
        )

    forecast_hashes_by_group: dict[tuple[str, float, int], set[str]] = {}
    for (fingerprint, p, repeat, _h), row in row_by_key.items():
        forecast_hashes_by_group.setdefault((fingerprint, p, repeat), set()).add(
            str(row["initial_forecast_hash"])
        )
    shared_forecast = all(len(hashes) == 1 for hashes in forecast_hashes_by_group.values())
    if not shared_forecast:
        raise AssertionError("five horizons did not share their initial forecast hash")

    p0_repeats_identical = True
    for fingerprint in episode_by_fp:
        for h in horizons:
            r0 = row_by_key[(fingerprint, 0.0, 0, h)]
            r1 = row_by_key[(fingerprint, 0.0, 1, h)]
            if (
                r0["R"] != r1["R"]
                or r0["initial_forecast_hash"] != r1["initial_forecast_hash"]
                or r0["evaluator_initial_normalized_kendall"] != 0.0
                or r1["evaluator_initial_normalized_kendall"] != 0.0
                or r0["evaluator_initial_mean_abs_rank_displacement"] != 0.0
                or r1["evaluator_initial_mean_abs_rank_displacement"] != 0.0
            ):
                p0_repeats_identical = False
                break
            for row in (r0, r1):
                prefix_error = row["evaluator_prefix_membership_error"]
                if (h == 0 and prefix_error is not None) or (
                    h > 0 and not np.isclose(prefix_error, 0.0, rtol=0, atol=1e-12)
                ):
                    p0_repeats_identical = False
                    break
    if not p0_repeats_identical:
        raise AssertionError("p=0 repeats must have identical accurate forecasts and R")
    return {
        "row_count_1600_unique_keys": len(rows) == 1600 and len(row_by_key) == 1600,
        "all_episodes_completed_and_retrieved": all(
            row["completed"] and row["retrieved_count"] == row["N"] for row in rows
        ),
        "all_actions_legal": all(row["illegal_action_count"] == 0 for row in rows),
        "R_bound_and_RN_arithmetic": all(
            row["R"] <= row["N"] * (row["max_tiers"] - 1)
            and np.isclose(row["RN"], row["R"] / row["N"], rtol=0, atol=1e-12)
            for row in rows
        ),
        "five_horizons_share_forecast_hash": shared_forecast,
        "p0_repeats_have_identical_R_and_exact_forecast": p0_repeats_identical,
        "source_config_data_hashes_match_receipt": True,
    }


def _summarize_phenomenon(
    rows: list[dict[str, Any]],
    episodes: list[OnlineCRPEpisode],
    config: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    checks = _verify_phenomenon_rows(rows, episodes, config, receipt)
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    geometries = tuple(
        str(item["id"]) for item in config["layout_splits"]["geometries"]
    )
    row_by_key = {
        (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r"]),
            int(row["h"]),
        ): row
        for row in rows
    }
    fingerprints_by_geometry = {
        geometry_id: sorted(
            label_invariant_fingerprint(episode)
            for episode in episodes
            if episode.parameter_group == geometry_id
        )
        for geometry_id in geometries
    }
    geometry_summary: list[dict[str, Any]] = []
    geometry_index: dict[tuple[str, float, int], dict[str, Any]] = {}
    for geometry_id in geometries:
        fingerprints = fingerprints_by_geometry[geometry_id]
        for p in p_values:
            for h in horizons:
                per_layout_r: list[float] = []
                per_layout_rn: list[float] = []
                metric_rows: list[dict[str, Any]] = []
                for fingerprint in fingerprints:
                    layout_rows = [
                        row_by_key[(fingerprint, p, repeat, h)] for repeat in repeats
                    ]
                    per_layout_r.append(float(np.mean([row["R"] for row in layout_rows])))
                    per_layout_rn.append(float(np.mean([row["RN"] for row in layout_rows])))
                    metric_rows.extend(layout_rows)
                prefix_values = [
                    row["evaluator_prefix_membership_error"]
                    for row in metric_rows
                    if row["evaluator_prefix_membership_error"] is not None
                ]
                summary = {
                    "geometry_id": geometry_id,
                    "p_evaluator_only": p,
                    "h": h,
                    "layout_count": len(fingerprints),
                    "mean_R": float(np.mean(per_layout_r)),
                    "mean_RN": float(np.mean(per_layout_rn)),
                    "mean_initial_normalized_kendall": float(
                        np.mean(
                            [row["evaluator_initial_normalized_kendall"] for row in metric_rows]
                        )
                    ),
                    "mean_initial_abs_rank_displacement": float(
                        np.mean(
                            [row["evaluator_initial_mean_abs_rank_displacement"] for row in metric_rows]
                        )
                    ),
                    "mean_prefix_membership_error": (
                        float(np.mean(prefix_values)) if prefix_values else None
                    ),
                }
                geometry_summary.append(summary)
                geometry_index[(geometry_id, p, h)] = summary

    macro_summary: list[dict[str, Any]] = []
    macro_index: dict[tuple[float, int], dict[str, Any]] = {}
    for p in p_values:
        for h in horizons:
            by_geometry = [geometry_index[(geometry_id, p, h)] for geometry_id in geometries]
            prefix_values = [
                item["mean_prefix_membership_error"]
                for item in by_geometry
                if item["mean_prefix_membership_error"] is not None
            ]
            summary = {
                "p_evaluator_only": p,
                "h": h,
                "geometry_count": len(geometries),
                "macro_mean_R": float(np.mean([item["mean_R"] for item in by_geometry])),
                "macro_mean_RN": float(np.mean([item["mean_RN"] for item in by_geometry])),
                "macro_initial_normalized_kendall": float(
                    np.mean([item["mean_initial_normalized_kendall"] for item in by_geometry])
                ),
                "macro_initial_abs_rank_displacement": float(
                    np.mean([item["mean_initial_abs_rank_displacement"] for item in by_geometry])
                ),
                "macro_prefix_membership_error": (
                    float(np.mean(prefix_values)) if prefix_values else None
                ),
            }
            macro_summary.append(summary)
            macro_index[(p, h)] = summary

    fixed_h_scores = {
        h: float(np.mean([macro_index[(p, h)]["macro_mean_RN"] for p in p_values]))
        for h in horizons
    }
    best_fixed_h = min(horizons, key=lambda h: (fixed_h_scores[h], h))

    primary_p = float(config["n2_phenomenon"]["primary_p"])
    h_long = int(config["n2_phenomenon"]["primary_h_long"])
    h_short = int(config["n2_phenomenon"]["primary_h_short"])
    differences_by_geometry: dict[str, list[float]] = {}
    h_short_by_geometry: dict[str, list[float]] = {}
    primary_by_geometry: list[dict[str, Any]] = []
    for geometry_id in geometries:
        differences: list[float] = []
        short_values: list[float] = []
        for fingerprint in fingerprints_by_geometry[geometry_id]:
            long_rn = float(
                np.mean(
                    [row_by_key[(fingerprint, primary_p, repeat, h_long)]["RN"] for repeat in repeats]
                )
            )
            short_rn = float(
                np.mean(
                    [row_by_key[(fingerprint, primary_p, repeat, h_short)]["RN"] for repeat in repeats]
                )
            )
            differences.append(long_rn - short_rn)
            short_values.append(short_rn)
        differences_by_geometry[geometry_id] = differences
        h_short_by_geometry[geometry_id] = short_values
        primary_by_geometry.append(
            {
                "geometry_id": geometry_id,
                "layout_count": len(differences),
                "mean_RN_long": float(
                    geometry_index[(geometry_id, primary_p, h_long)]["mean_RN"]
                ),
                "mean_RN_short": float(
                    geometry_index[(geometry_id, primary_p, h_short)]["mean_RN"]
                ),
                "mean_paired_delta_RN_long_minus_short": float(np.mean(differences)),
            }
        )
    macro_delta = float(
        np.mean([np.mean(differences_by_geometry[geometry_id]) for geometry_id in geometries])
    )
    macro_short = float(
        np.mean([np.mean(h_short_by_geometry[geometry_id]) for geometry_id in geometries])
    )
    macro_long = float(
        np.mean(
            [
                geometry_index[(geometry_id, primary_p, h_long)]["mean_RN"]
                for geometry_id in geometries
            ]
        )
    )
    if macro_short <= 0:
        raise AssertionError("primary relative difference denominator EM-5 must be positive")
    relative_delta_percent = 100.0 * macro_delta / macro_short

    n2 = config["n2_phenomenon"]
    bootstrap_rng = np.random.Generator(
        np.random.PCG64(int(n2["bootstrap_seed"]))
    )
    bootstrap_abs: list[float] = []
    bootstrap_relative: list[float] = []
    for _ in range(int(n2["bootstrap_replicates"])):
        sampled_deltas: list[float] = []
        sampled_short: list[float] = []
        for geometry_id in geometries:
            deltas = np.asarray(differences_by_geometry[geometry_id], dtype=np.float64)
            shorts = np.asarray(h_short_by_geometry[geometry_id], dtype=np.float64)
            picked = bootstrap_rng.integers(0, len(deltas), size=len(deltas))
            sampled_deltas.append(float(np.mean(deltas[picked])))
            sampled_short.append(float(np.mean(shorts[picked])))
        boot_delta = float(np.mean(sampled_deltas))
        boot_short = float(np.mean(sampled_short))
        bootstrap_abs.append(boot_delta)
        bootstrap_relative.append(100.0 * boot_delta / boot_short)
    alpha = (1.0 - float(n2["ci_level"])) / 2.0
    abs_ci = np.quantile(bootstrap_abs, [alpha, 1.0 - alpha]).tolist()
    relative_ci = np.quantile(bootstrap_relative, [alpha, 1.0 - alpha]).tolist()
    supported = bool(
        relative_delta_percent >= float(n2["minimum_relative_harm_percent"])
        and relative_ci[0] > 0.0
    )
    primary = {
        "p": primary_p,
        "contrast": f"EM-{h_long} minus EM-{h_short}",
        "h_long": h_long,
        "h_short": h_short,
        "macro_RN_h_long": macro_long,
        "macro_RN_h_short_denominator": macro_short,
        "macro_paired_delta_RN": macro_delta,
        "relative_delta_percent_using_h_short_denominator": relative_delta_percent,
        "paired_by_geometry": primary_by_geometry,
        "bootstrap": {
            "replicates": int(n2["bootstrap_replicates"]),
            "seed": int(n2["bootstrap_seed"]),
            "unit": "layout cluster, resampled within geometry; all repeats/H conditions stay paired",
            "confidence_level": float(n2["ci_level"]),
            "macro_delta_RN_CI": [float(value) for value in abs_ci],
            "relative_delta_percent_CI": [float(value) for value in relative_ci],
        },
        "minimum_relative_harm_percent": float(n2["minimum_relative_harm_percent"]),
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
    }
    secondary_20_minus_0 = []
    for p in p_values:
        per_geometry = []
        for geometry_id in geometries:
            delta = float(
                geometry_index[(geometry_id, p, 20)]["mean_RN"]
                - geometry_index[(geometry_id, p, 0)]["mean_RN"]
            )
            per_geometry.append({"geometry_id": geometry_id, "delta_RN_20_minus_0": delta})
        secondary_20_minus_0.append(
            {
                "p_evaluator_only": p,
                "macro_delta_RN_20_minus_0": float(
                    np.mean([item["delta_RN_20_minus_0"] for item in per_geometry])
                ),
                "by_geometry": per_geometry,
            }
        )
    return {
        "checks": checks,
        "geometry_h_p_summary": geometry_summary,
        "macro_h_p_summary": macro_summary,
        "fixed_h_select_once": {
            "selection_split": "D_select only",
            "aggregation": "equal weight over p and geometry; repeats averaged within layout first",
            "macro_mean_RN_by_h": {str(h): fixed_h_scores[h] for h in horizons},
            "selected_h": int(best_fixed_h),
            "tie_rule": "smaller h",
        },
        "primary_p_0_5_em20_minus_em5": primary,
        "secondary_em20_minus_em0": secondary_20_minus_0,
    }


def run_phenomenon(
    config_path: Path = DEFAULT_CONFIG, output: Path = N2_OUTPUT
) -> dict[str, Any]:
    config = _read_config(config_path)
    if "phenomenon" not in config["release_scope"]["allowed_stages"]:
        raise RuntimeError("N2 phenomenon stage is not released by this config")
    episodes, data_path, data_hash = _load_frozen_select(config)
    receipt = _n2_source_receipt(config_path, data_path, data_hash)
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    n2 = config["n2_phenomenon"]
    q_prior = float(n2["q_prior"])
    output.mkdir(parents=True, exist_ok=True)

    rows_path = output / "N2_ROWS.jsonl"
    report_json_path = output / "N2_REPORT.json"
    report_md_path = output / "N2_REPORT.md"
    receipt_path = output / "N2_CONFIG_RECEIPT.json"
    stdout_path = output / "N2_STDOUT.txt"
    for path in (rows_path, report_json_path, report_md_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"N2 output already exists; refusing to overwrite: {path}")
    if stdout_path.exists() and stdout_path.stat().st_size > 0:
        raise FileExistsError(f"N2 stdout already exists; refusing to overwrite: {stdout_path}")

    config_receipt = {
        "protocol_id": config["protocol_id"],
        "n2_run_id": n2["run_id"],
        "stage": "phenomenon_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_layout_count": len(episodes),
        "input_split": "D_select",
        "p_values": p_values,
        "repeats": repeats,
        "horizons": horizons,
        "q_prior": q_prior,
        "q_prior_used_for_em_decisions": False,
        "forecast_seed_rule": config["forecast"]["seed_rule"],
        "source_config_data_hashes": receipt,
        "n1_select_sha256_expected_from_n1_report": receipt["data_sha256"],
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(config_receipt, ensure_ascii=False, indent=2) + "\n")

    print(
        "N2 phenomenon starting: "
        f"layouts={len(episodes)} p={len(p_values)} r={len(repeats)} "
        f"h={len(horizons)} expected_rows={len(episodes) * len(p_values) * len(repeats) * len(horizons)}"
    )
    run_started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    with rows_path.open("x", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            fingerprint = label_invariant_fingerprint(episode)
            true_future = tuple(episode.retrieval_sequence[1:])
            for p in p_values:
                for repeat in repeats:
                    # The semantic key deliberately excludes p, h, and method.
                    seed = semantic_forecast_seed(fingerprint, repeat)
                    private_source = ForecastProvider(
                        episode.retrieval_sequence,
                        p=p,
                        seed=seed,
                        q_prior=q_prior,
                        max_horizon=MAX_HORIZON,
                    )
                    shared_forecast = tuple(private_source._forecast_future)
                    forecast_hash = _forecast_hash(shared_forecast)
                    for h in horizons:
                        noise_metrics = _initial_forecast_metrics(
                            true_future, shared_forecast, h
                        )
                        row = _run_em_h_rollout(
                            episode,
                            fingerprint=fingerprint,
                            p=p,
                            repeat=repeat,
                            h=h,
                            seed=seed,
                            shared_forecast=shared_forecast,
                            forecast_hash=forecast_hash,
                            noise_metrics=noise_metrics,
                            q_prior=q_prior,
                            receipt=receipt,
                        )
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        handle.flush()
                        rows.append(row)
                        if not row["completed"] or row["illegal_action_count"] != 0:
                            raise RuntimeError(
                                "N2 stopped at first incomplete/illegal condition; "
                                f"partial row preserved for {episode.episode_id}, p={p}, r={repeat}, h={h}: "
                                f"{row.get('error', 'incomplete rollout')}"
                            )
                        elapsed = time.perf_counter() - run_started
                        if len(rows) % 100 == 0:
                            print(f"N2 rows={len(rows)}/1600 elapsed_seconds={elapsed:.1f}")
                        if elapsed > 60.0 * int(n2["wall_clock_limit_minutes"]):
                            raise TimeoutError(
                                "N2 wall-clock cap reached; partial N2_ROWS.jsonl preserved"
                            )

    summary = _summarize_phenomenon(rows, episodes, config, receipt)
    elapsed_total = time.perf_counter() - run_started
    if len(rows) != 1600 or not all(summary["checks"].values()):
        raise AssertionError("N2 completeness/contract checks did not all pass")
    end_utc = datetime.now(timezone.utc).isoformat()
    primary = summary["primary_p_0_5_em20_minus_em5"]
    report = {
        "status": "N2_COMPLETE",
        "interpretation": primary["status"],
        "completed_utc_start": start_utc,
        "completed_utc_end": end_utc,
        "elapsed_seconds_including_em_selection": elapsed_total,
        "protocol_id": config["protocol_id"],
        "n2_run_id": n2["run_id"],
        "scope": {
            "phenomenon_only": True,
            "source_split": "D_select",
            "rows": 1600,
            "fitting": False,
            "screen_evaluation": False,
            "gp_training": False,
            "rl_training": False,
            "bayes_dp": False,
            "calibration_claim": False,
        },
        "configuration_receipt": config_receipt,
        "row_file": rows_path.name,
        "row_count": len(rows),
        "unique_condition_keys": len(
            {
                (
                    row["fingerprint"],
                    row["p_evaluator_only"],
                    row["r"],
                    row["h"],
                )
                for row in rows
            }
        ),
        "conditions": {
            "layout_count": len(episodes),
            "p_values": p_values,
            "repeats": repeats,
            "horizons": horizons,
            "q_prior": q_prior,
            "q_used_for_em_actions": False,
            "full_future_forecast_fixed_per_layout_p_repeat_across_h": True,
        },
        **summary,
    }
    with report_json_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    markdown = [
        "# N2 H×p phenomenon report",
        "",
        f"Status: **{primary['status']}** under the fixed p=0.5 EM-20 minus EM-5 gate; this is development evidence from D_select only.",
        "",
        f"- Complete rows: {len(rows)}/1600; unique layouts: {len(episodes)}; all retrievals complete, all actions legal, R/N arithmetic and the R≤N(T−1) bound verified.",
        f"- Input: frozen r01/data/select.json, SHA-256 `{receipt['data_sha256']}`; no layouts were regenerated or removed.",
        f"- Forecast seed: SHA-256 of semantic layout fingerprint plus r; excludes p, h, and method. Each p/r forecast hash is shared across all five horizons.",
        f"- q prior: {q_prior:.3f}; q was not used for EM actions and is not claimed to be calibrated.",
        f"- Wall time: {elapsed_total:.2f}s including EM action selection; no fitting, screen evaluation, GP/RL, or Bayes DP.",
        "",
        "## Geometry × p × H summary",
        "",
        "Each repeat is averaged within layout first; layouts are averaged within geometry. Macro columns weight G1/G2 equally.",
        "",
        "| Geometry | p | H | layouts | mean R | mean R/N | initial Kendall | mean rank displacement | prefix membership error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["geometry_h_p_summary"]:
        prefix = item["mean_prefix_membership_error"]
        markdown.append(
            "| {geometry_id} | {p_evaluator_only:.1f} | {h} | {layout_count} | {mean_R:.4f} | {mean_RN:.6f} | {mean_initial_normalized_kendall:.6f} | {mean_initial_abs_rank_displacement:.4f} | {prefix} |".format(
                **item, prefix="NA" if prefix is None else f"{prefix:.6f}"
            )
        )
    markdown.extend(
        [
            "",
            "## Equal-geometry macro H×p table",
            "",
            "| p | H | macro mean R | macro mean R/N | Kendall | mean rank displacement | prefix membership error |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["macro_h_p_summary"]:
        prefix = item["macro_prefix_membership_error"]
        markdown.append(
            "| {p_evaluator_only:.1f} | {h} | {macro_mean_R:.4f} | {macro_mean_RN:.6f} | {macro_initial_normalized_kendall:.6f} | {macro_initial_abs_rank_displacement:.4f} | {prefix} |".format(
                **item, prefix="NA" if prefix is None else f"{prefix:.6f}"
            )
        )
    fixed = summary["fixed_h_select_once"]
    markdown.extend(
        [
            "",
            "## Fixed-H selection and preregistered contrast",
            "",
            f"D_select-only equal-p/equal-geometry selection chooses H={fixed['selected_h']} (smaller H wins exact ties). Macro R/N by H: `{json.dumps(fixed['macro_mean_RN_by_h'], sort_keys=True)}`.",
            f"Primary p=0.5 contrast: EM-20 minus EM-5 = {primary['macro_paired_delta_RN']:.6f} R/N, relative to EM-5 = {primary['relative_delta_percent_using_h_short_denominator']:.3f}%.",
            f"90% layout-cluster paired bootstrap CI for relative difference: [{primary['bootstrap']['relative_delta_percent_CI'][0]:.3f}%, {primary['bootstrap']['relative_delta_percent_CI'][1]:.3f}%] ({primary['bootstrap']['replicates']} replicates, seed {primary['bootstrap']['seed']}); absolute ΔR/N CI: [{primary['bootstrap']['macro_delta_RN_CI'][0]:.6f}, {primary['bootstrap']['macro_delta_RN_CI'][1]:.6f}].",
            f"Preregistered gate: relative harm ≥{primary['minimum_relative_harm_percent']:.1f}% and 90% CI lower bound >0. Result: **{primary['status']}**.",
            "",
            "Secondary EM-20 minus EM-0 contrasts are reported for every p in N2_REPORT.json; they do not replace the primary test.",
            "",
            "## Scope and interpretation",
            "",
            "Noise diagnostics are computed evaluator-side from the full initial future permutation: normalized Kendall inversions, mean absolute rank displacement, and `1 - |true top-H ∩ predicted top-H|/H` for nonempty H (H=0 is NA). None is passed to EM. p is a perturbation selection strength, not an asserted error probability. These D_select results are development evidence, not a sealed test or final scientific conclusion.",
            "",
            "The row-level output and exact source/config/data receipt are in N2_ROWS.jsonl and N2_CONFIG_RECEIPT.json; captured command output is N2_STDOUT.txt.",
        ]
    )
    with report_md_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")

    print(
        f"N2 phenomenon complete: rows={len(rows)} layouts={len(episodes)} "
        f"primary_relative_delta_percent={primary['relative_delta_percent_using_h_short_denominator']:.3f} "
        f"ci90=[{primary['bootstrap']['relative_delta_percent_CI'][0]:.3f},"
        f"{primary['bootstrap']['relative_delta_percent_CI'][1]:.3f}] "
        f"status={primary['status']} elapsed_seconds={elapsed_total:.2f}"
    )
    return report


def run_contract(config_path: Path = DEFAULT_CONFIG, output: Path = OUTPUT) -> dict[str, Any]:
    config = _read_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    data_audit = _generate_layout_splits(config, output)
    tiny_rows, totals = _run_tiny_contract(config, output)
    test_stdout_path = output / "N1_TEST_STDOUT.txt"
    test_stdout = (
        test_stdout_path.read_text(encoding="utf-8-sig")
        if test_stdout_path.is_file()
        else ""
    )
    test_summary = next(
        (
            line.strip()
            for line in test_stdout.splitlines()
            if " passed in " in line
            and " failed" not in line
        ),
        None,
    )
    targeted_tests = {
        "command": "py -3.12 -B -m pytest -p no:cacheprovider -q tests/test_noise_aware_lookahead.py",
        "stdout_file": test_stdout_path.name,
        "stdout_sha256": file_sha256(test_stdout_path) if test_stdout_path.is_file() else None,
        "summary": test_summary,
        "status": "PASS" if test_summary else "NOT_VERIFIED",
    }
    sources = [
        ROOT / "scrp/noise_aware_forecast.py",
        ROOT / "scrp/noise_aware_em.py",
        ROOT / "experiments/noise_aware_lookahead.py",
        ROOT / "tests/test_noise_aware_lookahead.py",
        config_path,
        output / "LITERATURE_AUDIT.md",
    ]
    report = {
        "status": "N1_CONTRACT_EVIDENCE_READY_FOR_REVIEW",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": config["protocol_id"],
        "run_id": config["run_id"],
        "scope": {
            "n0_literature_audit": "executed; novelty unresolved",
            "n1_contract_only": True,
            "phenomenon_scoring": False,
            "ridge_fitting": False,
            "counterfactual_delta_r_label_generation": "NOT_IMPLEMENTED",
            "label_policy_isolation_validation": "NOT_IMPLEMENTED",
            "h0_true_delta_r_label_validation": "NOT_IMPLEMENTED",
            "screen_evaluation": False,
            "gp_training": False,
            "rl_training": False,
            "bayes_dp": False,
            "final_test": False,
        },
        "data": data_audit,
        "contract_totals": totals,
        "targeted_tests": targeted_tests,
        "tiny_episode_contract_rows": tiny_rows,
        "checks": {
            "all_16_tiny_episodes_complete": True,
            "all_actions_legal": True,
            "fixed_action_scalar_physics_lookahead_invariant": True,
            "q_unchanged_on_repeated_relocation": True,
            "reset_and_step_revelations_replayed_in_order": True,
            "requested_and_effective_h_logged_outside_policy": True,
            "scientific_scores": 0,
            "fitted_models": 0,
        },
        "sha256": {
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_sha256(path)
            for path in sources
        },
    }
    report_path = output / "N1_CONTRACT_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_md = [
        "# N1 contract report",
        "",
        "Status: contract evidence prepared for root-agent review; this is not a scientific result.",
        "",
        f"- Tiny validation episodes: {totals['completed']}/16 complete.",
        f"- Legal relocation actions: {totals['legal_actions']}; full episode relocation scores were not aggregated or compared.",
        f"- Reliability events: {totals['revelation_updates']}; q remained unchanged on {totals['relocations_without_revelation']} relocation steps without a reveal.",
        f"- Candidate requested/effective H records: {totals['candidate_h_length_records']}; all five fixed candidates were logged as min(candidate_h, remaining_future) by the evaluator.",
        f"- Targeted tests: {targeted_tests['status']} ({targeted_tests['summary'] or 'no passing summary found'}); see {targeted_tests['stdout_file']}.",
        f"- New development layouts: {data_audit['generated_total']} (fit 40, select 40, screen 80), zero semantic overlap with the verified source-manifest union.",
        "- Scientific scoring: 0. Ridge fitting: 0. GP/RL training: 0. Bayes DP: 0.",
        "- NOT_IMPLEMENTED: counterfactual ΔR label generation, label-to-policy isolation, and h=0 true-label validation. These require a later reviewed evaluation-stage contract.",
        "",
        "See N1_CONTRACT_REPORT.json for episode-level contract counts, source-manifest hashes, and code/config hashes. The 16 tiny episodes are correctness fixtures only.",
        "",
        "## Literature boundary",
        "",
        "See LITERATURE_AUDIT.md. The Galle EM source definition was verified from its full author manuscript; nearby RPM, calibration, and online limited-horizon work remains a novelty boundary. No “first” claim is made.",
    ]
    (output / "report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    print(
        "N1 contract: "
        f"tiny_episodes={totals['unique_tiny_episodes']} "
        f"completed={totals['completed']} legal_actions={totals['legal_actions']} "
        f"layouts={data_audit['generated_total']} "
        f"source_overlap={data_audit['generated_source_overlap']} "
        "scientific_scores=0 fitted_models=0"
    )
    return report


def _n3_load_fit_select(config: dict[str, Any]) -> tuple[
    list[OnlineCRPEpisode],
    list[OnlineCRPEpisode],
    dict[str, str],
    dict[tuple[str, float, int], dict[str, Any]],
    dict[str, Path],
]:
    n3 = config["n3_fit_select"]
    n1_report = OUTPUT / "N1_CONTRACT_REPORT.json"
    n2_report = OUTPUT / "N2_REPORT.json"
    n2_rows_path = OUTPUT / "N2_ROWS.jsonl"
    if not n1_report.is_file() or not n2_report.is_file() or not n2_rows_path.is_file():
        raise FileNotFoundError("N3 requires frozen N1 data and completed N2 evidence")
    n1 = json.loads(n1_report.read_text(encoding="utf-8"))
    n2 = json.loads(n2_report.read_text(encoding="utf-8"))
    if n2["fixed_h_select_once"]["selected_h"] != int(n3["fixed_h_baseline"]):
        raise ValueError("N3 fixed-H baseline differs from the completed N2 selection")

    expected_hashes = {
        "fit": n1["data"]["generated_splits"]["fit"]["sha256"],
        "select": n1["data"]["generated_splits"]["select"]["sha256"],
        "screen": n1["data"]["generated_splits"]["screen"]["sha256"],
    }
    paths = {
        "fit": ROOT / str(n3["fit_layouts"]),
        "select": ROOT / str(n3["select_layouts"]),
        "screen": ROOT / str(n3["screen_layouts_hash_only"]),
    }
    actual_hashes = {name: file_sha256(path) for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise ValueError(
            f"N1 frozen data hashes changed: expected={expected_hashes}, actual={actual_hashes}"
        )
    fit_records = json.loads(paths["fit"].read_text(encoding="utf-8"))
    select_records = json.loads(paths["select"].read_text(encoding="utf-8"))
    fit = [OnlineCRPEpisode.from_record(row) for row in fit_records]
    select = [OnlineCRPEpisode.from_record(row) for row in select_records]
    if len(fit) != int(n3["expected_fit_layout_count"]) or len(select) != int(
        n3["expected_select_layout_count"]
    ):
        raise ValueError("N3 fit/select layout counts differ from the frozen protocol")
    for split, episodes in (("fit", fit), ("select", select)):
        if any(episode.metadata.get("split") != split for episode in episodes):
            raise ValueError(f"N3 {split} contains a different split")
        fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(f"N3 {split} contains duplicate semantic layouts")
        by_geometry = {
            str(item["id"]): sum(episode.parameter_group == str(item["id"]) for episode in episodes)
            for item in config["layout_splits"]["geometries"]
        }
        if by_geometry != {"G1": 20, "G2": 20}:
            raise ValueError(f"N3 {split} geometry counts differ: {by_geometry}")
    fit_fingerprints = {label_invariant_fingerprint(episode) for episode in fit}
    select_fingerprints = {label_invariant_fingerprint(episode) for episode in select}
    if fit_fingerprints & select_fingerprints:
        raise ValueError("N3 fit/select semantic layouts overlap")
    if n2["configuration_receipt"]["source_config_data_hashes"]["data_sha256"] != actual_hashes["select"]:
        raise ValueError("completed N2 result is not bound to the frozen D_select input")
    if n2.get("status") != "N2_COMPLETE" or not all(n2.get("checks", {}).values()):
        raise ValueError("N3 requires complete, passing N2 evidence")
    fixed_h_rows: dict[tuple[str, float, int], dict[str, Any]] = {}
    for line in n2_rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["h"]) != int(n3["fixed_h_baseline"]):
            continue
        key = (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r"]),
        )
        if key in fixed_h_rows:
            raise ValueError("N2 contains duplicate fixed-H baseline keys")
        if not row["completed"] or row["illegal_action_count"] != 0:
            raise ValueError("N2 fixed-H baseline contains an incomplete or illegal row")
        fixed_h_rows[key] = row
    expected_fixed_h_keys = {
        (label_invariant_fingerprint(episode), p, repeat)
        for episode in select
        for p in (0.0, 0.1, 0.3, 0.5)
        for repeat in (0, 1)
    }
    if set(fixed_h_rows) != expected_fixed_h_keys:
        raise ValueError("N2 fixed-H cache does not cover frozen D_select conditions")
    for episode in select:
        fingerprint = label_invariant_fingerprint(episode)
        for p in (0.0, 0.1, 0.3, 0.5):
            for repeat in (0, 1):
                seed = semantic_forecast_seed(fingerprint, repeat)
                provider = ForecastProvider(
                    episode.retrieval_sequence,
                    p=p,
                    seed=seed,
                    q_prior=float(config["n2_phenomenon"]["q_prior"]),
                    max_horizon=MAX_HORIZON,
                )
                if _forecast_hash(provider._forecast_future) != fixed_h_rows[
                    (fingerprint, p, repeat)
                ]["initial_forecast_hash"]:
                    raise ValueError("N2 cached H20 forecast differs from semantic N3 seed")
    return fit, select, actual_hashes, fixed_h_rows, paths


def _n3_source_receipt(
    config_path: Path,
    data_hashes: dict[str, str],
    data_paths: dict[str, Path],
) -> dict[str, Any]:
    source_paths = (
        ROOT / "scrp/noise_aware_forecast.py",
        ROOT / "scrp/noise_aware_em.py",
        ROOT / "experiments/noise_aware_lookahead.py",
        BASE_ROOT / "scrp/online_crp.py",
        BASE_ROOT / "scrp/attention_data.py",
        ROOT / "tests/test_noise_aware_lookahead.py",
    )
    source_hashes = {
        _source_path_label(path): file_sha256(path) for path in source_paths
    }
    config_hash = file_sha256(config_path)
    n1_path = OUTPUT / "N1_CONTRACT_REPORT.json"
    n2_path = OUTPUT / "N2_REPORT.json"
    n2_rows_path = OUTPUT / "N2_ROWS.jsonl"
    n2_snapshot_path = OUTPUT / "N2_SOURCE_SNAPSHOT.zip"
    test_stdout_path = OUTPUT / "N3_PREFLIGHT_TEST_STDOUT.txt"
    if not n2_snapshot_path.is_file():
        raise FileNotFoundError("N3 requires the immutable source snapshot used by N2")
    if not test_stdout_path.is_file():
        raise FileNotFoundError("N3 requires the actual targeted test stdout before scoring")
    test_stdout = test_stdout_path.read_text(encoding="utf-8-sig")
    if " passed" not in test_stdout or " failed" in test_stdout:
        raise ValueError("N3 post-review targeted correctness test did not report a clean pass")
    test_command = next(
        (
            line.removeprefix("COMMAND:").strip()
            for line in test_stdout.splitlines()
            if line.startswith("COMMAND:")
        ),
        None,
    )
    test_summary = next(
        (line.strip() for line in reversed(test_stdout.splitlines()) if " passed" in line),
        None,
    )
    if not test_command or not test_summary:
        raise ValueError("N3 targeted test stdout must record its exact command and pass summary")
    provenance = {
        "n1_report_sha256": file_sha256(n1_path),
        "n2_report_sha256": file_sha256(n2_path),
        "n2_rows_sha256": file_sha256(n2_rows_path),
        "n2_source_snapshot_sha256": file_sha256(n2_snapshot_path),
        "n3_preflight_test_stdout_sha256": file_sha256(test_stdout_path),
        "n3_pre_review_full_test_stdout_sha256": (
            file_sha256(OUTPUT / "N3_PREFLIGHT_TEST_STDOUT_PRE_REVIEW.txt")
            if (OUTPUT / "N3_PREFLIGHT_TEST_STDOUT_PRE_REVIEW.txt").is_file()
            else None
        ),
    }
    source_bundle = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_sha256": source_hashes,
        "source_bundle_sha256": source_bundle,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_hash,
        "data_paths": {
            name: path.relative_to(ROOT).as_posix() for name, path in data_paths.items()
        },
        "data_sha256": data_hashes,
        "n1_n2_provenance_sha256": provenance,
        "n3_preflight_test_command": test_command,
        "n3_preflight_test_summary": test_summary,
        "n3_pre_review_test_scope": "historical 12-test full-module pass before the reviewed fit/select row fail-stop and sample-deficit changes; not current-source validation",
    }


def _n3_plan_samples(
    episodes: list[OnlineCRPEpisode],
    p_values: tuple[float, ...],
    repeats: tuple[int, ...],
    config: dict[str, Any],
) -> dict[tuple[str, float, int], dict[str, Any]]:
    seed = int(config["n3_fit_select"]["sampling_seed"])
    cap = int(config["n3_fit_select"]["states_per_condition_cap"])
    rng = np.random.Generator(np.random.PCG64(seed))
    geometry_ids = tuple(str(item["id"]) for item in config["layout_splits"]["geometries"])
    by_geometry = {
        geometry_id: sorted(
            (
                label_invariant_fingerprint(episode),
                episode,
            )
            for episode in episodes
            if episode.parameter_group == geometry_id
        )
        for geometry_id in geometry_ids
    }
    design: dict[tuple[str, float, int], dict[str, Any]] = {}
    condition_index = 0
    stage_targets = (0.15, 0.50, 0.85)
    for p in p_values:
        for repeat in repeats:
            first_geometry_count = 12 if condition_index % 2 == 0 else 13
            counts = {geometry_ids[0]: first_geometry_count, geometry_ids[1]: cap - first_geometry_count}
            for geometry_id in geometry_ids:
                rows = by_geometry[geometry_id]
                if len(rows) < counts[geometry_id]:
                    raise ValueError(f"not enough {geometry_id} layouts for N3 state sampling")
                order = rng.permutation(len(rows))[: counts[geometry_id]].tolist()
                target_order = rng.permutation(len(stage_targets)).tolist()
                for index, selected_index in enumerate(order):
                    fingerprint, _episode = rows[selected_index]
                    target = stage_targets[target_order[index % len(target_order)]]
                    key = (fingerprint, p, repeat)
                    if key in design:
                        raise AssertionError("N3 state sample design contains a duplicate key")
                    design[key] = {
                        "geometry_id": geometry_id,
                        "target_stage_fraction": target,
                        "sampling_seed": seed,
                        "sampling_rule": "uniform layout selection; stage bucket cycled from deterministic seeded order; no labels or future costs used",
                    }
            condition_index += 1
    if len(design) != int(config["n3_fit_select"]["max_sampled_states"]):
        raise AssertionError(f"N3 planned {len(design)} states, expected 200")
    return design


def _n3_observation_record(observation: PublicObservation) -> dict[str, Any]:
    return {
        "rank_tensor": observation.rank_tensor.tolist(),
        "occupied_mask": observation.occupied_mask.astype(int).tolist(),
        "legal_destination_mask": observation.legal_destination_mask.astype(int).tolist(),
        "history_hits": list(observation.history_hits),
        "q_prior": float(observation.q_prior),
        "q_t": float(observation.q_t),
    }


def _n3_rebase_observation(
    observation: PublicObservation, q_prior: float
) -> PublicObservation:
    recent_hits = observation.history_hits[-8:]
    q_t = (2.0 * q_prior + sum(recent_hits)) / (2.0 + len(recent_hits))
    return PublicObservation(
        observation.rank_tensor,
        observation.occupied_mask,
        observation.legal_destination_mask,
        recent_hits,
        q_prior,
        q_t,
    )


def _n3_private_env_snapshot(environment: OnlineCRPEnvironment) -> dict[str, Any]:
    return {
        "stacks": [list(stack) for stack in environment._stacks],
        "locations": {str(item): location for item, location in environment._locations.items()},
        "next_index": int(environment._next_index),
        "relocation_count": int(environment._relocations),
        "crane_position": environment._crane_position,
        "crane_time": float(environment._crane_time),
        "terminated": bool(environment._terminated),
        "max_steps": int(environment.max_steps),
        "lookahead_horizon": int(environment.lookahead_horizon),
        "crane_travel_time": float(environment.crane_travel_time),
        "pick_drop_time": float(environment.pick_drop_time),
    }


def _n3_restore_public_state(
    record: dict[str, Any], q_prior: float
) -> PublicObservation:
    public = record["public_observation_at_capture"]
    raw = PublicObservation(
        np.asarray(public["rank_tensor"], dtype=np.int32),
        np.asarray(public["occupied_mask"], dtype=np.bool_),
        np.asarray(public["legal_destination_mask"], dtype=np.bool_),
        tuple(bool(item) for item in public["history_hits"]),
        float(public["q_prior"]),
        float(public["q_t"]),
    )
    return _n3_rebase_observation(raw, q_prior)


def _n3_replay_provider_at_state(
    environment: OnlineCRPEnvironment,
    episode: OnlineCRPEpisode,
    *,
    p: float,
    seed: int,
    q_prior: float,
    forecast_future: tuple[int, ...],
    expected_history_hits: tuple[bool, ...],
) -> ForecastProvider:
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=p,
        seed=seed,
        q_prior=q_prior,
        max_horizon=MAX_HORIZON,
        forecast_future=forecast_future,
    )
    count = int(environment.state.retrieval_count)
    sync_retrieval_events(provider, episode.retrieval_sequence, 0, count)
    if provider.history.hits != expected_history_hits:
        raise AssertionError("private state history does not match replayed reveal events")
    if provider._current != environment.current_target_id:
        if not environment.state.terminated:
            raise AssertionError("forecast current target differs from the copied environment")
    return provider


def _n3_em0_continuation(
    environment: OnlineCRPEnvironment,
    episode: OnlineCRPEpisode,
    *,
    p: float,
    seed: int,
    q_prior: float,
    forecast_future: tuple[int, ...],
    history_hits: tuple[bool, ...],
    first_destination: int,
) -> int:
    """Force one private evaluator action, then finish with public EM-0."""
    working = copy.deepcopy(environment)
    provider = _n3_replay_provider_at_state(
        working,
        episode,
        p=p,
        seed=seed,
        q_prior=q_prior,
        forecast_future=forecast_future,
        expected_history_hits=history_hits,
    )
    start_relocations = int(working.relocation_count)
    before = int(working.state.retrieval_count)
    if first_destination not in working.legal_destinations():
        raise ValueError("counterfactual first action is not legal at the sampled state")
    working.step(first_destination)
    after = int(working.state.retrieval_count)
    sync_retrieval_events(provider, episode.retrieval_sequence, before, after)
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    decisions = 1
    while not working.state.terminated:
        legal = working.legal_destinations()
        observation = provider.public_observation(
            _private_physical_stacks(working), legal, max_tiers=episode.max_tiers
        )
        action = select_em_destination(observation, max_tiers=episode.max_tiers, h=0)
        if action not in legal:
            raise AssertionError("EM-0 continuation selected an illegal action")
        before = int(working.state.retrieval_count)
        working.step(action)
        after = int(working.state.retrieval_count)
        sync_retrieval_events(provider, episode.retrieval_sequence, before, after)
        decisions += 1
        if decisions > max_steps:
            raise RuntimeError("counterfactual continuation exceeded R<=N(T-1)")
    if not working.state.terminated or int(working.state.retrieval_count) != episode.num_containers:
        raise AssertionError("counterfactual continuation did not retrieve the full episode")
    if working.relocation_count - start_relocations > max_steps:
        raise AssertionError("counterfactual continuation violated the relocation safety bound")
    return int(working.relocation_count - start_relocations)


def _n3_counterfactual_labels(
    environment: OnlineCRPEnvironment,
    episode: OnlineCRPEpisode,
    observation: PublicObservation,
    *,
    fingerprint: str,
    state_id: str,
    p: float,
    repeat: int,
    seed: int,
    q_prior: float,
    forecast_future: tuple[int, ...],
    receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    max_tiers = episode.max_tiers
    candidates = candidate_actions(observation, max_tiers=max_tiers)
    if not candidates or len(candidates) > len(HORIZONS):
        raise AssertionError("N3 sampled state has an invalid EM candidate set")
    baseline = next((candidate for candidate in candidates if candidate.requested_h == 0), None)
    if baseline is None:
        raise AssertionError("N3 candidate deduplication lost the EM-0 anchor")
    leveling_action = LevelingBaseline.select_destination(
        environment.state, environment.legal_destinations()
    )
    if baseline.destination != leveling_action:
        raise AssertionError("same-state EM-0 and Leveling actions differ")
    history_hits = tuple(observation.history_hits)
    r0 = _n3_em0_continuation(
        environment,
        episode,
        p=p,
        seed=seed,
        q_prior=q_prior,
        forecast_future=forecast_future,
        history_hits=history_hits,
        first_destination=baseline.destination,
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rh = (
            r0
            if candidate.requested_h == 0
            else _n3_em0_continuation(
                environment,
                episode,
                p=p,
                seed=seed,
                q_prior=q_prior,
                forecast_future=forecast_future,
                history_hits=history_hits,
                first_destination=candidate.destination,
            )
        )
        delta = int(rh - r0)
        if candidate.requested_h == 0 and delta != 0:
            raise AssertionError("N3 EM-0 true counterfactual label must be exactly zero")
        rows.append(
            {
                "state_id": state_id,
                "fingerprint": fingerprint,
                "geometry_id": episode.parameter_group,
                "p_evaluator_only": float(p),
                "r_evaluator_only": int(repeat),
                "requested_h": int(candidate.requested_h),
                "effective_h": int(
                    min(
                        candidate.requested_h,
                        max(
                            0,
                            len(episode.retrieval_sequence)
                            - int(environment.state.retrieval_count)
                            - 1,
                        ),
                    )
                ),
                "destination_stack_evaluator_output": int(candidate.destination),
                "public_features_8d": [float(value) for value in candidate.features],
                "R_candidate_after_state": int(rh),
                "R_em0_after_state": int(r0),
                "true_delta_R_label_evaluator_only": delta,
                "label_definition": "forced first action then deterministic EM-0 continuation; evaluator-private true suffix",
                "source_bundle_sha256": receipt["source_bundle_sha256"],
                "config_sha256": receipt["config_sha256"],
                "fit_data_sha256": receipt["data_sha256"]["fit"],
            }
        )
    return rows, r0


def _n3_run_rollout(
    episode: OnlineCRPEpisode,
    *,
    fingerprint: str,
    p: float,
    repeat: int,
    seed: int,
    q_prior: float,
    shared_forecast: tuple[int, ...],
    forecast_hash: str,
    receipt: dict[str, Any],
    variant: str | None = None,
    coefficients: tuple[float, ...] | None = None,
    target_stage_fraction: float | None = None,
    record_q_t_before: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    """Run fixed EM-20 or a frozen public selector; optional sample capture uses stage only."""
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    started = time.perf_counter()
    env: OnlineCRPEnvironment | None = None
    event_rows: list[dict[str, Any]] = []
    captured: dict[str, Any] | None = None
    capture_distance = math.inf
    decision_count = 0
    illegal_count = 0
    source_h_counts: dict[str, int] = {}
    effective_h_counts: dict[str, int] = {}
    error: str | None = None
    try:
        env = OnlineCRPEnvironment(episode, max_steps=max_steps, lookahead_horizon=0)
        provider = ForecastProvider(
            episode.retrieval_sequence,
            p=p,
            seed=seed,
            q_prior=q_prior,
            max_horizon=MAX_HORIZON,
            forecast_future=shared_forecast,
        )
        initial_retrieved = int(env.state.retrieval_count)
        sync_retrieval_events(
            provider,
            episode.retrieval_sequence,
            0,
            initial_retrieved,
            event_rows=event_rows,
            record_q_t_before=record_q_t_before,
        )
        while not env.state.terminated:
            legal = env.legal_destinations()
            observation = provider.public_observation(
                _private_physical_stacks(env), legal, max_tiers=episode.max_tiers
            )
            if target_stage_fraction is not None:
                actual_fraction = float(env.state.retrieval_count / episode.num_containers)
                distance = abs(actual_fraction - target_stage_fraction)
                if distance < capture_distance:
                    capture_distance = distance
                    captured = {
                        "episode": episode,
                        "fingerprint": fingerprint,
                        "p": float(p),
                        "repeat": int(repeat),
                        "seed": int(seed),
                        "shared_forecast": tuple(shared_forecast),
                        "forecast_hash": forecast_hash,
                        "target_stage_fraction": float(target_stage_fraction),
                        "actual_stage_fraction": actual_fraction,
                        "decision_ordinal": int(decision_count),
                        "retrieval_count": int(env.state.retrieval_count),
                        "history_hits": tuple(provider.history.hits),
                        "public_observation_at_capture": _n3_observation_record(observation),
                        "environment": copy.deepcopy(env),
                    }
            if variant is None:
                action = select_em_destination(
                    observation, max_tiers=episode.max_tiers, h=MAX_HORIZON
                )
                requested_h = MAX_HORIZON
            else:
                if coefficients is None:
                    raise ValueError("frozen selector coefficients are required")
                decision = select_sr_em(
                    observation,
                    max_tiers=episode.max_tiers,
                    variant=variant,  # type: ignore[arg-type]
                    coefficients=coefficients,
                )
                action = decision.destination
                requested_h = decision.requested_h
            if action not in legal:
                illegal_count += 1
                raise ValueError("N3 policy selected an illegal destination")
            source_h_counts[str(requested_h)] = source_h_counts.get(str(requested_h), 0) + 1
            effective_h = min(int(requested_h), provider.remaining_future_count)
            effective_h_counts[str(effective_h)] = effective_h_counts.get(str(effective_h), 0) + 1
            decision_count += 1
            before = int(env.state.retrieval_count)
            env.step(action)
            after = int(env.state.retrieval_count)
            sync_retrieval_events(
                provider,
                episode.retrieval_sequence,
                before,
                after,
                event_rows=event_rows,
                record_q_t_before=record_q_t_before,
            )
            if decision_count > max_steps:
                raise RuntimeError("N3 rollout exceeded R<=N(T-1)")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    relocations = int(env.relocation_count) if env is not None else 0
    retrieved = int(env.state.retrieval_count) if env is not None else 0
    completed = bool(
        env is not None
        and env.state.terminated
        and retrieved == episode.num_containers
        and error is None
    )
    row: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "fingerprint": fingerprint,
        "geometry_id": episode.parameter_group,
        "N": episode.num_containers,
        "max_tiers": episode.max_tiers,
        "split": "fit" if variant is None else "select",
        "p_evaluator_only": float(p),
        "r_evaluator_only": int(repeat),
        "forecast_hash": forecast_hash,
        "policy": "EM20" if variant is None else variant,
        "q_prior_actor_input": float(q_prior),
        "R": relocations,
        "RN": float(relocations / episode.num_containers),
        "decision_count": int(decision_count),
        "completed": completed,
        "retrieved_count": retrieved,
        "illegal_action_count": int(illegal_count),
        "reveal_event_count": len(event_rows),
        "reveal_hit_count": sum(bool(event["hit"]) for event in event_rows),
        "reveal_hits_evaluator_only": [bool(event["hit"]) for event in event_rows]
        if variant is None
        else None,
        "selected_action_source_h_counts": source_h_counts,
        "selected_action_effective_h_counts": effective_h_counts,
        "elapsed_seconds_including_policy_selection": float(elapsed),
        "source_bundle_sha256": receipt["source_bundle_sha256"],
        "config_sha256": receipt["config_sha256"],
        "fit_data_sha256": receipt["data_sha256"]["fit"],
        "select_data_sha256": receipt["data_sha256"]["select"],
    }
    if error is not None:
        row["error"] = error
    return row, captured, event_rows


def _n3_fit_ridge_models(
    label_rows: list[dict[str, Any]], ridge_lambda: float
) -> dict[str, dict[str, Any]]:
    feature_columns = {
        "SR": tuple(range(8)),
        "R": (0, 1, 2, 5),
        "S": (0, 1, 3, 4),
    }
    if not label_rows or any(len(row["public_features_8d"]) != 8 for row in label_rows):
        raise ValueError("N3 ridge requires all sampled public eight-dimensional label rows")
    y = np.asarray(
        [float(row["true_delta_R_label_evaluator_only"]) for row in label_rows],
        dtype=np.float64,
    )
    fitted: dict[str, dict[str, Any]] = {}
    for variant, columns in feature_columns.items():
        x = np.asarray(
            [
                [float(row["public_features_8d"][index]) for index in columns]
                for row in label_rows
            ],
            dtype=np.float64,
        )
        gram = x.T @ x + float(ridge_lambda) * np.eye(x.shape[1], dtype=np.float64)
        coefficients = np.linalg.solve(gram, x.T @ y)
        residual = y - x @ coefficients
        fitted[variant] = {
            "feature_indices_from_SR8": list(columns),
            "coefficients": [float(value) for value in coefficients],
            "label_row_count": len(label_rows),
            "sampled_state_count": len({row["state_id"] for row in label_rows}),
            "ridge_lambda": float(ridge_lambda),
            "penalty_includes_bias": True,
            "fit_residual_rmse": float(np.sqrt(np.mean(residual**2))),
            "fit_residual_mae": float(np.mean(np.abs(residual))),
            "fit_residual_mean": float(np.mean(residual)),
            "fit_residual_sum_squares": float(np.dot(residual, residual)),
        }
    return fitted


def _n3_verify_fit_trajectories(
    rows: list[dict[str, Any]], episodes: list[OnlineCRPEpisode]
) -> dict[str, bool]:
    expected = {
        (label_invariant_fingerprint(episode), p, repeat)
        for episode in episodes
        for p in (0.0, 0.1, 0.3, 0.5)
        for repeat in (0, 1)
    }
    seen: set[tuple[str, float, int]] = set()
    episode_by_fingerprint = {
        label_invariant_fingerprint(episode): episode for episode in episodes
    }
    for row in rows:
        key = (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r_evaluator_only"]),
        )
        if key in seen:
            raise AssertionError("duplicate N3 D_fit trajectory key")
        seen.add(key)
        episode = episode_by_fingerprint[key[0]]
        if (
            not row["completed"]
            or row["retrieved_count"] != episode.num_containers
            or row["illegal_action_count"] != 0
            or row["decision_count"] != row["R"]
            or row["R"] > episode.num_containers * (episode.max_tiers - 1)
            or row["reveal_event_count"] != episode.num_containers - 1
            or row["reveal_hit_count"] != sum(row["reveal_hits_evaluator_only"])
        ):
            raise AssertionError("N3 D_fit trajectory failed completion/event/score checks")
    if seen != expected:
        raise AssertionError(f"N3 D_fit trajectories incomplete: {len(seen)}/320")
    return {
        "fit_trajectory_keys_320_unique": len(seen) == 320,
        "fit_all_retrieved_and_legal": all(
            row["completed"]
            and row["retrieved_count"] == row["N"]
            and row["illegal_action_count"] == 0
            for row in rows
        ),
        "fit_reveal_event_count_N_minus_1": all(
            row["reveal_event_count"] == row["N"] - 1 for row in rows
        ),
        "fit_relocation_bound_and_arithmetic": all(
            row["R"] <= row["N"] * (row["max_tiers"] - 1)
            and row["decision_count"] == row["R"]
            and np.isclose(row["RN"], row["R"] / row["N"], rtol=0, atol=1e-12)
            for row in rows
        ),
    }


def _n3_verify_select_rows(
    rows: list[dict[str, Any]],
    episodes: list[OnlineCRPEpisode],
    fixed_h_rows: dict[tuple[str, float, int], dict[str, Any]],
    receipt: dict[str, Any],
    model_hash: str,
) -> dict[str, bool]:
    variants = ("SR", "R", "S")
    p_values = (0.0, 0.1, 0.3, 0.5)
    repeats = (0, 1)
    episode_by_fingerprint = {
        label_invariant_fingerprint(episode): episode for episode in episodes
    }
    expected = {
        (fingerprint, p, repeat, variant)
        for fingerprint in episode_by_fingerprint
        for p in p_values
        for repeat in repeats
        for variant in variants
    }
    row_by_key: dict[tuple[str, float, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r_evaluator_only"]),
            str(row["policy"]),
        )
        if key in row_by_key:
            raise AssertionError(f"duplicate N3 D_select key: {key}")
        row_by_key[key] = row
        episode = episode_by_fingerprint.get(key[0])
        if episode is None:
            raise AssertionError("N3 select row contains a non-D_select layout")
        if (
            not row["completed"]
            or row["retrieved_count"] != episode.num_containers
            or row["illegal_action_count"] != 0
            or row["decision_count"] != row["R"]
            or row["R"] > episode.num_containers * (episode.max_tiers - 1)
            or row["reveal_event_count"] != episode.num_containers - 1
            or not np.isclose(row["RN"], row["R"] / episode.num_containers, rtol=0, atol=1e-12)
        ):
            raise AssertionError("N3 select row failed completion/legal/score checks")
        if (
            row["source_bundle_sha256"] != receipt["source_bundle_sha256"]
            or row["config_sha256"] != receipt["config_sha256"]
            or row["fit_data_sha256"] != receipt["data_sha256"]["fit"]
            or row["select_data_sha256"] != receipt["data_sha256"]["select"]
            or row["model_sha256"] != model_hash
        ):
            raise AssertionError("N3 select row hashes differ from its receipt/model")
        baseline = fixed_h_rows[(key[0], key[1], key[2])]
        if row["forecast_hash"] != baseline["initial_forecast_hash"]:
            raise AssertionError("N3 selector forecast differs from paired cached N2 EM20")
    if set(row_by_key) != expected or len(rows) != 960:
        raise AssertionError(f"N3 select keys incomplete: {len(row_by_key)}/960")

    forecast_shared = True
    for fingerprint in episode_by_fingerprint:
        for p in p_values:
            for repeat in repeats:
                hashes = {
                    row_by_key[(fingerprint, p, repeat, variant)]["forecast_hash"]
                    for variant in variants
                }
                if len(hashes) != 1:
                    forecast_shared = False
    if not forecast_shared:
        raise AssertionError("N3 selector variants did not share a forecast per condition")
    return {
        "select_rows_960_unique_keys": len(row_by_key) == 960,
        "select_all_complete_and_legal": all(
            row["completed"]
            and row["retrieved_count"] == row["N"]
            and row["illegal_action_count"] == 0
            for row in rows
        ),
        "select_R_bound_and_RN_arithmetic": all(
            row["R"] <= row["N"] * (row["max_tiers"] - 1)
            and row["decision_count"] == row["R"]
            and np.isclose(row["RN"], row["R"] / row["N"], rtol=0, atol=1e-12)
            for row in rows
        ),
        "three_selectors_share_N2_forecasts_and_H20_cache": forecast_shared,
        "select_source_config_data_model_hashes_match": True,
    }


def _n3_select_performance_summary(
    rows: list[dict[str, Any]],
    episodes: list[OnlineCRPEpisode],
    fixed_h_rows: dict[tuple[str, float, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = ("SR", "R", "S")
    p_values = (0.0, 0.1, 0.3, 0.5)
    repeats = (0, 1)
    geometries = ("G1", "G2")
    row_by_key = {
        (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r_evaluator_only"]),
            str(row["policy"]),
        ): row
        for row in rows
    }
    by_geometry: list[dict[str, Any]] = []
    macro_by_variant_p: list[dict[str, Any]] = []
    macro_values: dict[tuple[str, float], dict[str, float]] = {}
    for variant in variants:
        for p in p_values:
            geom_rows = []
            for geometry_id in geometries:
                layout_values: list[float] = []
                baseline_values: list[float] = []
                for episode in episodes:
                    if episode.parameter_group != geometry_id:
                        continue
                    fingerprint = label_invariant_fingerprint(episode)
                    layout_values.append(
                        float(
                            np.mean(
                                [
                                    row_by_key[(fingerprint, p, repeat, variant)]["RN"]
                                    for repeat in repeats
                                ]
                            )
                        )
                    )
                    baseline_values.append(
                        float(
                            np.mean(
                                [
                                    fixed_h_rows[(fingerprint, p, repeat)]["RN"]
                                    for repeat in repeats
                                ]
                            )
                        )
                    )
                if len(layout_values) != 20:
                    raise AssertionError("N3 macro requires 20 layouts per geometry")
                model_mean = float(np.mean(layout_values))
                baseline_mean = float(np.mean(baseline_values))
                item = {
                    "variant": variant,
                    "geometry_id": geometry_id,
                    "p_evaluator_only": p,
                    "layout_count": len(layout_values),
                    "mean_RN": model_mean,
                    "cached_N2_EM20_mean_RN": baseline_mean,
                    "paired_delta_RN_vs_cached_N2_EM20": model_mean - baseline_mean,
                }
                by_geometry.append(item)
                geom_rows.append(item)
            macro_model = float(np.mean([item["mean_RN"] for item in geom_rows]))
            macro_baseline = float(
                np.mean([item["cached_N2_EM20_mean_RN"] for item in geom_rows])
            )
            macro_delta = macro_model - macro_baseline
            macro_values[(variant, p)] = {
                "model": macro_model,
                "baseline": macro_baseline,
                "delta": macro_delta,
            }
            macro_by_variant_p.append(
                {
                    "variant": variant,
                    "p_evaluator_only": p,
                    "equal_geometry_macro_RN": macro_model,
                    "cached_N2_EM20_equal_geometry_macro_RN": macro_baseline,
                    "paired_delta_RN_vs_cached_N2_EM20": macro_delta,
                    "relative_delta_percent_vs_cached_N2_EM20": (
                        100.0 * macro_delta / macro_baseline if macro_baseline else None
                    ),
                }
            )
    overall = []
    for variant in variants:
        model = float(np.mean([macro_values[(variant, p)]["model"] for p in p_values]))
        baseline = float(np.mean([macro_values[(variant, p)]["baseline"] for p in p_values]))
        overall.append(
            {
                "variant": variant,
                "equal_p_equal_geometry_macro_RN": model,
                "cached_N2_EM20_equal_p_equal_geometry_macro_RN": baseline,
                "paired_delta_RN_vs_cached_N2_EM20": model - baseline,
                "relative_delta_percent_vs_cached_N2_EM20": (
                    100.0 * (model - baseline) / baseline if baseline else None
                ),
                "interpretation": "D_select development check only; not an independent final test",
            }
        )
    return by_geometry, macro_by_variant_p + overall


def _n3_calibration_summary(
    event_rows: list[dict[str, Any]],
    episodes: list[OnlineCRPEpisode],
    q_prior: float,
) -> dict[str, Any]:
    p_values = (0.0, 0.1, 0.3, 0.5)
    repeats = (0, 1)
    episode_by_fingerprint = {
        label_invariant_fingerprint(episode): episode for episode in episodes
    }
    events_by_condition: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in event_rows:
        key = (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r_evaluator_only"]),
        )
        events_by_condition.setdefault(key, []).append(row)
        if not 0.0 <= float(row["q_t_before_reveal"]) <= 1.0:
            raise AssertionError("prequential q_t is outside [0,1]")
    expected_conditions = {
        (fingerprint, p, repeat)
        for fingerprint in episode_by_fingerprint
        for p in p_values
        for repeat in repeats
    }
    if set(events_by_condition) != expected_conditions:
        raise AssertionError("D_select prequential event conditions are incomplete or duplicated")
    condition_scores: dict[tuple[str, float, int], tuple[float, float]] = {}
    for key, rows in events_by_condition.items():
        episode = episode_by_fingerprint[key[0]]
        rows.sort(key=lambda row: int(row["event_index"]))
        expected_events = episode.num_containers - 1
        if len(rows) != expected_events or [int(row["event_index"]) for row in rows] != list(
            range(1, expected_events + 1)
        ):
            raise AssertionError("prequential event sequence does not contain every reveal exactly once")
        if len({row["forecast_hash"] for row in rows}) != 1:
            raise AssertionError("prequential event row forecast hash changed within an episode")
        model_brier = float(
            np.mean(
                [
                    (float(row["q_t_before_reveal"]) - float(bool(row["hit"]))) ** 2
                    for row in rows
                ]
            )
        )
        prior_brier = float(
            np.mean(
                [
                    (q_prior - float(bool(row["hit"]))) ** 2
                    for row in rows
                ]
            )
        )
        condition_scores[key] = (model_brier, prior_brier)

    geometry_p_rows: list[dict[str, Any]] = []
    per_geometry_p: dict[tuple[str, float], tuple[float, float]] = {}
    for geometry_id in ("G1", "G2"):
        for p in p_values:
            layout_model: list[float] = []
            layout_prior: list[float] = []
            for fingerprint, episode in episode_by_fingerprint.items():
                if episode.parameter_group != geometry_id:
                    continue
                repeat_model = [condition_scores[(fingerprint, p, repeat)][0] for repeat in repeats]
                repeat_prior = [condition_scores[(fingerprint, p, repeat)][1] for repeat in repeats]
                layout_model.append(float(np.mean(repeat_model)))
                layout_prior.append(float(np.mean(repeat_prior)))
            q_score = float(np.mean(layout_model))
            prior_score = float(np.mean(layout_prior))
            per_geometry_p[(geometry_id, p)] = (q_score, prior_score)
            geometry_p_rows.append(
                {
                    "geometry_id": geometry_id,
                    "p_evaluator_only": p,
                    "layout_count": len(layout_model),
                    "q_t_prequential_Brier": q_score,
                    "constant_q_prior_Brier": prior_score,
                    "delta_q_minus_prior": q_score - prior_score,
                }
            )
    macro_q = float(np.mean([value[0] for value in per_geometry_p.values()]))
    macro_prior = float(np.mean([value[1] for value in per_geometry_p.values()]))

    bins = []
    for low, high, label in ((0.0, 1.0 / 3.0, "[0,1/3)"), (1.0 / 3.0, 2.0 / 3.0, "[1/3,2/3)"), (2.0 / 3.0, 1.0, "[2/3,1]")):
        selected = [
            row
            for row in event_rows
            if low <= float(row["q_t_before_reveal"]) < high
            or (high == 1.0 and float(row["q_t_before_reveal"]) == 1.0)
        ]
        bins.append(
            {
                "bin": label,
                "event_coverage_count": len(selected),
                "mean_q_t": (
                    float(np.mean([row["q_t_before_reveal"] for row in selected]))
                    if selected
                    else None
                ),
                "observed_hit_rate": (
                    float(np.mean([bool(row["hit"]) for row in selected]))
                    if selected
                    else None
                ),
            }
        )
    return {
        "q_prior_fit_only": float(q_prior),
        "select_event_count_once_across_methods": len(event_rows),
        "condition_event_count": len(expected_conditions),
        "aggregation": "event Brier within condition; average r within layout, layouts within geometry, then equal geometry and p",
        "q_t_prequential_macro_Brier": macro_q,
        "constant_q_prior_macro_Brier": macro_prior,
        "delta_q_t_minus_constant_prior": macro_q - macro_prior,
        "gate_status": "NOT_WORSE_THAN_PRIOR" if macro_q <= macro_prior else "NOT_SUPPORTED",
        "by_geometry_p": geometry_p_rows,
        "fixed_reliability_bins": bins,
        "shared_strategy_event_check": "calibration events recorded once per layout/p/r from SR run; R/S repeats verified identical before deduplication",
    }


def _n3_write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _n3_write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _n3_assert_rollout_complete(row: dict[str, Any], stage: str) -> None:
    valid = (
        bool(row.get("completed"))
        and int(row.get("retrieved_count", -1)) == int(row.get("N", -2))
        and int(row.get("illegal_action_count", -1)) == 0
        and int(row.get("R", -1)) >= 0
        and int(row.get("R", -1)) <= int(row.get("N", 0)) * int(row.get("max_tiers", 0) - 1)
        and int(row.get("decision_count", -1)) == int(row.get("R", -2))
        and "error" not in row
    )
    if not valid:
        raise RuntimeError(
            f"N3 FAIL_STOP: incomplete/illegal {stage} rollout for "
            f"{row.get('fingerprint')} p={row.get('p_evaluator_only')} "
            f"r={row.get('r_evaluator_only')} policy={row.get('policy')}: "
            f"{row.get('error', 'row-level completion check failed')}"
        )


def run_fit_select(config_path: Path, output: Path) -> dict[str, Any]:
    """Run only the released N3 fit/select stage; D_screen is hash-only."""
    config = _read_config(config_path)
    if "fit" not in config["release_scope"]["allowed_stages"]:
        raise RuntimeError("PLAN-036 N3 fit/select is not released by this config")
    fit_episodes, select_episodes, data_hashes, fixed_h_rows, data_paths = (
        _n3_load_fit_select(config)
    )
    output.mkdir(parents=True, exist_ok=True)
    generated_names = (
        "N3_RUN_RECEIPT.json",
        "N3_FIT_ROWS.jsonl",
        "N3_SAMPLE_STATES.jsonl",
        "N3_LABEL_ROWS.jsonl",
        "N3_MODEL.json",
        "N3_SELECT_ROWS.jsonl",
        "N3_Q_CALIBRATION_EVENTS.jsonl",
        "N3_FIT_REPORT.json",
        "N3_FIT_REPORT.md",
    )
    collisions = [name for name in generated_names if (output / name).exists()]
    if collisions:
        raise FileExistsError(
            "N3 outputs already exist; refusing to overwrite: " + ", ".join(collisions)
        )

    receipt = _n3_source_receipt(config_path, data_hashes, data_paths)
    receipt.update(
        {
            "stage": "N3_fit_select_only",
            "stage_status_at_start": "STARTED",
            "job_pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "cli": "py -3.12 -B -m experiments.noise_aware_lookahead --stage fit",
            "scientific_budget": {
                "D_fit_complete_rollouts": 320,
                "D_select_complete_rollouts": 960,
                "sampled_states_max": 200,
                "counterfactual_continuations_max": 1000,
                "D_screen_scored": False,
            },
            "frozen_n1_data_paths": {
                "fit": data_paths["fit"].relative_to(ROOT).as_posix(),
                "select": data_paths["select"].relative_to(ROOT).as_posix(),
                "screen_hash_only": data_paths["screen"].relative_to(ROOT).as_posix(),
            },
        }
    )
    _n3_write_new_json(output / "N3_RUN_RECEIPT.json", receipt)
    print(
        f"N3 fit/select START pid={os.getpid()} layouts_fit={len(fit_episodes)} "
        f"layouts_select={len(select_episodes)} D_screen_scored=false",
        flush=True,
    )

    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    q_prior_initial = float(config["n2_phenomenon"]["q_prior"])
    sample_design = _n3_plan_samples(fit_episodes, p_values, repeats, config)
    captured_states: dict[tuple[str, float, int], dict[str, Any]] = {}
    sample_deficits: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    fit_path = output / "N3_FIT_ROWS.jsonl"
    fit_started = time.perf_counter()
    with fit_path.open("x", encoding="utf-8", newline="\n") as handle:
        condition_count = 0
        for episode in fit_episodes:
            fingerprint = label_invariant_fingerprint(episode)
            for p in p_values:
                for repeat in repeats:
                    condition_count += 1
                    seed = semantic_forecast_seed(fingerprint, repeat)
                    provider = ForecastProvider(
                        episode.retrieval_sequence,
                        p=p,
                        seed=seed,
                        q_prior=q_prior_initial,
                        max_horizon=MAX_HORIZON,
                    )
                    forecast = tuple(provider._forecast_future)
                    forecast_hash = _forecast_hash(forecast)
                    sample_spec = sample_design.get((fingerprint, p, repeat))
                    row, captured, _events = _n3_run_rollout(
                        episode,
                        fingerprint=fingerprint,
                        p=p,
                        repeat=repeat,
                        seed=seed,
                        q_prior=q_prior_initial,
                        shared_forecast=forecast,
                        forecast_hash=forecast_hash,
                        receipt=receipt,
                        target_stage_fraction=(
                            float(sample_spec["target_stage_fraction"])
                            if sample_spec is not None
                            else None
                        ),
                    )
                    row["n3_sample_planned"] = sample_spec is not None
                    if sample_spec is not None:
                        row["n3_sample_target_stage_fraction"] = float(
                            sample_spec["target_stage_fraction"]
                        )
                        if captured is None:
                            if row["decision_count"] == 0 and row["completed"]:
                                reason = (
                                    "episode completed during environment reset; "
                                    "no nonterminal decision state exists"
                                )
                                row["n3_sample_skipped_reason"] = reason
                                sample_deficits.append(
                                    {
                                        "fingerprint": fingerprint,
                                        "geometry_id": episode.parameter_group,
                                        "p_evaluator_only": p,
                                        "r_evaluator_only": repeat,
                                        "target_stage_fraction": float(
                                            sample_spec["target_stage_fraction"]
                                        ),
                                        "reason": reason,
                                    }
                                )
                            else:
                                row["n3_sample_skipped_reason"] = (
                                    "unexpected_missing_capture_with_decisions"
                                )
                    _n3_write_jsonl_row(handle, row)
                    _n3_assert_rollout_complete(row, "D_fit")
                    fit_rows.append(row)
                    if sample_spec is not None:
                        if captured is None:
                            if row["decision_count"] != 0 or not row["completed"]:
                                raise AssertionError(
                                    "planned N3 state capture was missing despite decision states"
                                )
                            continue
                        captured["sample_spec"] = sample_spec
                        captured_states[(fingerprint, p, repeat)] = captured
            if len(fit_rows) % 40 == 0:
                print(
                    f"N3 D_fit progress layouts={len(fit_rows) // 8}/40 "
                    f"rows={len(fit_rows)}/320 samples={len(captured_states)}/200",
                    flush=True,
                )
    fit_checks = _n3_verify_fit_trajectories(fit_rows, fit_episodes)
    if not all(fit_checks.values()):
        raise AssertionError(f"N3 D_fit verification failed: {fit_checks}")
    if len(captured_states) + len(sample_deficits) != len(sample_design):
        raise AssertionError("N3 sample plan has unaccounted missing captures")
    if len(captured_states) > int(config["n3_fit_select"]["max_sampled_states"]):
        raise AssertionError("N3 sampled more states than the reviewed maximum")
    total_fit_hits = sum(int(row["reveal_hit_count"]) for row in fit_rows)
    total_fit_events = sum(int(row["reveal_event_count"]) for row in fit_rows)
    if total_fit_events <= 0:
        raise AssertionError("N3 D_fit produced no actual next-target reveal events")
    q_prior = total_fit_hits / total_fit_events
    fit_elapsed = time.perf_counter() - fit_started
    print(
        f"N3 D_fit COMPLETE rows={len(fit_rows)} hits={total_fit_hits}/{total_fit_events} "
        f"q_prior={q_prior:.8f} elapsed_s={fit_elapsed:.2f}",
        flush=True,
    )

    label_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    label_continuations = 0
    state_counts_by_condition: dict[str, int] = {}
    sample_path = output / "N3_SAMPLE_STATES.jsonl"
    label_path = output / "N3_LABEL_ROWS.jsonl"
    with sample_path.open("x", encoding="utf-8", newline="\n") as sample_handle, label_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as label_handle:
        for (fingerprint, p, repeat), captured in sorted(captured_states.items()):
            episode = captured["episode"]
            state_id = f"{fingerprint}-p{p:.1f}-r{repeat}"
            observation = _n3_restore_public_state(captured, q_prior)
            labels, _r0 = _n3_counterfactual_labels(
                captured["environment"],
                episode,
                observation,
                fingerprint=fingerprint,
                state_id=state_id,
                p=p,
                repeat=repeat,
                seed=int(captured["seed"]),
                q_prior=q_prior,
                forecast_future=tuple(captured["shared_forecast"]),
                receipt=receipt,
            )
            if not labels or len(labels) > len(HORIZONS):
                raise AssertionError("sampled N3 state produced an invalid candidate-label count")
            if any(
                len(row["public_features_8d"]) != 8
                or not np.all(np.isfinite(row["public_features_8d"]))
                for row in labels
            ):
                raise AssertionError("N3 label contains malformed public features")
            if any(
                row["true_delta_R_label_evaluator_only"] != 0
                for row in labels
                if row["requested_h"] == 0
            ):
                raise AssertionError("N3 h=0 true counterfactual label was nonzero")
            condition_key = f"p={p:.1f},r={repeat}"
            state_counts_by_condition[condition_key] = (
                state_counts_by_condition.get(condition_key, 0) + 1
            )
            if state_counts_by_condition[condition_key] > int(
                config["n3_fit_select"]["states_per_condition_cap"]
            ):
                raise AssertionError("N3 state sample exceeded its per-condition cap")
            sample_row = {
                "state_id": state_id,
                "fingerprint": fingerprint,
                "geometry_id": episode.parameter_group,
                "split": "fit",
                "target_stage_fraction": float(
                    captured["sample_spec"]["target_stage_fraction"]
                ),
                "actual_stage_fraction": float(captured["actual_stage_fraction"]),
                "decision_ordinal": int(captured["decision_ordinal"]),
                "retrieval_count": int(captured["retrieval_count"]),
                "sampling_rule": captured["sample_spec"]["sampling_rule"],
                "public_observation_rebased_to_fit_q_prior": _n3_observation_record(
                    observation
                ),
                "evaluator_only_condition": {
                    "p": float(p),
                    "repeat": int(repeat),
                    "semantic_seed": int(captured["seed"]),
                    "forecast_hash": captured["forecast_hash"],
                    "forecast_future_tokens": list(captured["shared_forecast"]),
                    "environment_snapshot": _n3_private_env_snapshot(
                        captured["environment"]
                    ),
                },
                "candidate_count": len(labels),
                "continuations_including_baseline": len(labels),
                "source_bundle_sha256": receipt["source_bundle_sha256"],
                "config_sha256": receipt["config_sha256"],
                "fit_data_sha256": receipt["data_sha256"]["fit"],
            }
            _n3_write_jsonl_row(sample_handle, sample_row)
            sample_rows.append(sample_row)
            for row in labels:
                _n3_write_jsonl_row(label_handle, row)
                label_rows.append(row)
            label_continuations += len(labels)
            if label_continuations > int(
                config["n3_fit_select"]["max_label_continuations"]
            ):
                raise AssertionError("N3 counterfactual continuation budget exceeded")
            if len(sample_rows) % 25 == 0:
                print(
                    f"N3 counterfactual labels states={len(sample_rows)}/200 "
                    f"rows={len(label_rows)} continuations={label_continuations}",
                    flush=True,
                )
    if len(sample_rows) != len(captured_states) or len(
        {row["state_id"] for row in sample_rows}
    ) != len(sample_rows):
        raise AssertionError("N3 sampled-state rows do not match unique captured states")
    if not sample_rows:
        raise RuntimeError("N3 FAIL_STOP: no eligible public state was available for labels")
    if len(label_rows) != label_continuations or label_continuations > 1000:
        raise AssertionError("N3 label-row and continuation budgets disagree")
    if any(
        count > int(config["n3_fit_select"]["states_per_condition_cap"])
        for count in state_counts_by_condition.values()
    ):
        raise AssertionError("N3 state sample exceeded its per-condition cap")

    models = _n3_fit_ridge_models(
        label_rows, float(config["n3_fit_select"]["ridge_lambda"])
    )
    model_document = {
        "model_id": "PLAN-036-N3-ridge-lambda1",
        "role": "development selector fit only; D_screen not evaluated",
        "q_prior_fit_only": float(q_prior),
        "q_prior_estimator": "pooled actual next-target forecast-hit count / actual reveal-event count over the 320 D_fit EM20 trajectories",
        "training_label_definition": "forced candidate first action and deterministic EM-0 continuation minus same-state EM-0 anchor; h=0 label exactly zero",
        "training_row_count": len(label_rows),
        "sampled_state_count": len(sample_rows),
        "counterfactual_continuation_count": label_continuations,
        "ridge_lambda": float(config["n3_fit_select"]["ridge_lambda"]),
        "penalty_includes_bias": True,
        "all_variants_use_same_unweighted_rows": True,
        "models": models,
        "source_bundle_sha256": receipt["source_bundle_sha256"],
        "config_sha256": receipt["config_sha256"],
        "fit_data_sha256": receipt["data_sha256"]["fit"],
        "label_rows_sha256": file_sha256(label_path),
    }
    model_path = output / "N3_MODEL.json"
    _n3_write_new_json(model_path, model_document)
    model_hash = file_sha256(model_path)
    print(
        f"N3 MODEL FROZEN hash={model_hash} labels={len(label_rows)} "
        f"states={len(sample_rows)} continuations={label_continuations}",
        flush=True,
    )

    select_rows: list[dict[str, Any]] = []
    calibration_events: list[dict[str, Any]] = []
    select_started = time.perf_counter()
    select_path = output / "N3_SELECT_ROWS.jsonl"
    events_path = output / "N3_Q_CALIBRATION_EVENTS.jsonl"
    with select_path.open("x", encoding="utf-8", newline="\n") as row_handle, events_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as event_handle:
        for episode in select_episodes:
            fingerprint = label_invariant_fingerprint(episode)
            for p in p_values:
                for repeat in repeats:
                    seed = semantic_forecast_seed(fingerprint, repeat)
                    provider = ForecastProvider(
                        episode.retrieval_sequence,
                        p=p,
                        seed=seed,
                        q_prior=q_prior,
                        max_horizon=MAX_HORIZON,
                    )
                    forecast = tuple(provider._forecast_future)
                    forecast_hash = _forecast_hash(forecast)
                    condition_events: list[dict[str, Any]] | None = None
                    event_signature: tuple[tuple[int, bool, float], ...] | None = None
                    for variant in ("SR", "R", "S"):
                        row, _captured, events = _n3_run_rollout(
                            episode,
                            fingerprint=fingerprint,
                            p=p,
                            repeat=repeat,
                            seed=seed,
                            q_prior=q_prior,
                            shared_forecast=forecast,
                            forecast_hash=forecast_hash,
                            receipt=receipt,
                            variant=variant,
                            coefficients=tuple(models[variant]["coefficients"]),
                            record_q_t_before=True,
                        )
                        row["model_sha256"] = model_hash
                        signature = tuple(
                            (
                                int(event["event_index"]),
                                bool(event["hit"]),
                                float(event["q_t_before_reveal"]),
                            )
                            for event in events
                        )
                        if event_signature is None:
                            event_signature = signature
                            condition_events = events
                        elif signature != event_signature:
                            raise AssertionError(
                                "selector variants changed the causal forecast-hit history"
                            )
                        _n3_write_jsonl_row(row_handle, row)
                        _n3_assert_rollout_complete(row, "D_select")
                        select_rows.append(row)
                        if len(select_rows) % 120 == 0:
                            print(
                                f"N3 D_select progress rows={len(select_rows)}/960 "
                                f"conditions={len(select_rows) // 3}/320",
                                flush=True,
                            )
                    if condition_events is None:
                        raise AssertionError("N3 D_select condition has no calibration events")
                    for event in condition_events:
                        annotated = {
                            **event,
                            "fingerprint": fingerprint,
                            "geometry_id": episode.parameter_group,
                            "p_evaluator_only": float(p),
                            "r_evaluator_only": int(repeat),
                            "forecast_hash": forecast_hash,
                            "q_prior_fit_only": float(q_prior),
                            "source_bundle_sha256": receipt["source_bundle_sha256"],
                            "config_sha256": receipt["config_sha256"],
                            "model_sha256": model_hash,
                        }
                        _n3_write_jsonl_row(event_handle, annotated)
                        calibration_events.append(annotated)

    select_checks = _n3_verify_select_rows(
        select_rows, select_episodes, fixed_h_rows, receipt, model_hash
    )
    if not all(select_checks.values()):
        raise AssertionError(f"N3 D_select verification failed: {select_checks}")
    performance_by_geometry, performance_summary = _n3_select_performance_summary(
        select_rows, select_episodes, fixed_h_rows
    )
    calibration = _n3_calibration_summary(
        calibration_events, select_episodes, q_prior
    )
    select_elapsed = time.perf_counter() - select_started
    report = {
        "status": "N3_FIT_SELECT_COMPLETE_DEVELOPMENT_ONLY",
        "phase": "fit/select only; D_screen score not released or run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_pid": os.getpid(),
        "n2_primary_status_unchanged": "NOT_SUPPORTED",
        "fixed_h_baseline_selected_once_on_n2": int(
            config["n3_fit_select"]["fixed_h_baseline"]
        ),
        "screen_status": "NOT_RUN_RELEASE_PENDING_REVIEW",
        "configuration_receipt": receipt,
        "data_hashes": data_hashes,
        "model_sha256": model_hash,
        "fit": {
            "complete_rollouts": len(fit_rows),
            "expected_rollouts": 320,
            "unique_keys": len(
                {
                    (
                        row["fingerprint"],
                        row["p_evaluator_only"],
                        row["r_evaluator_only"],
                    )
                    for row in fit_rows
                }
            ),
            "actual_hit_events": total_fit_hits,
            "actual_reveal_events": total_fit_events,
            "q_prior_fit_only": q_prior,
            "q_prior_formula": "pooled D_fit EM20 actual forecast hits / pooled D_fit actual target reveals",
            "wall_seconds_including_policy_selection": fit_elapsed,
            "checks": fit_checks,
        },
        "counterfactual_fit": {
            "sampled_states": len(sample_rows),
            "planned_sampled_states_max": len(sample_design),
            "sample_deficit_count": len(sample_deficits),
            "sample_deficits": sample_deficits,
            "states_by_p_repeat_actual": state_counts_by_condition,
            "states_by_geometry": {
                geometry_id: sum(row["geometry_id"] == geometry_id for row in sample_rows)
                for geometry_id in ("G1", "G2")
            },
            "label_rows_and_continuations": len(label_rows),
            "continuation_budget_max": int(
                config["n3_fit_select"]["max_label_continuations"]
            ),
            "h0_true_labels_all_zero": all(
                row["true_delta_R_label_evaluator_only"] == 0
                for row in label_rows
                if row["requested_h"] == 0
            ),
            "model_fits": models,
            "label_actor_boundary": "fit inputs use only public_features_8d and evaluator-provided scalar label; private suffix/p/forecast tokens/env snapshots remain outside candidate and selector functions",
        },
        "select": {
            "complete_rollouts": len(select_rows),
            "expected_rollouts": 960,
            "wall_seconds_including_policy_selection": select_elapsed,
            "checks": select_checks,
            "performance_by_geometry_p": performance_by_geometry,
            "performance_macro": performance_summary,
            "q_calibration": calibration,
        },
        "scientific_limits": [
            "D_select is a development check and was used for no refitting or tuning.",
            "The selector receives only PublicObservation fields; p, true suffix, forecast token order, and evaluator labels are not policy inputs.",
            "These fit/select results do not establish generalization, a final test result, or a Bayes-optimal guarantee.",
            "No D_screen scoring, GP/RL training, Bayes DP, or extra baseline evaluation was run.",
        ],
    }
    _n3_write_new_json(output / "N3_FIT_REPORT.json", report)

    macro_rows = [
        row
        for row in performance_summary
        if "equal_p_equal_geometry_macro_RN" in row
    ]
    macro_p_rows = [
        row
        for row in performance_summary
        if "equal_geometry_macro_RN" in row
    ]
    report_md = [
        "# PLAN-036 N3 Fit/Select Report",
        "",
        f"Status: **{report['status']}**. This is development evidence only; D_screen was not scored.",
        "",
        "N2 primary status remains **NOT_SUPPORTED**. The fixed H=20 comparator stays frozen from N2.",
        "",
        "## Frozen inputs and fit budget",
        "",
        f"- D_fit: {len(fit_rows)}/320 complete EM20 rollouts; actual reveal-hit rate used as q_prior = {q_prior:.6f} ({total_fit_hits}/{total_fit_events}).",
        f"- Counterfactual data: {len(sample_rows)} captured states ({len(sample_deficits)} planned states had no decision point) and {label_continuations} forced-action continuations/label rows; maximum 200 states and 1000 continuations.",
        f"- Three fixed ridge models use the same rows, lambda={config['n3_fit_select']['ridge_lambda']}, with the bias penalized. Model SHA256: `{model_hash}`.",
        f"- D_select: {len(select_rows)}/960 complete closed-loop rollouts; no refitting or tuning on these rows.",
        "",
        "## Selector development checks",
        "",
        "| Variant | Equal-p/equal-geometry mean R/N | Paired delta vs cached H20 | Relative delta |",
        "|---|---:|---:|---:|",
    ]
    for row in macro_rows:
        report_md.append(
            f"| {row['variant']} | {row['equal_p_equal_geometry_macro_RN']:.6f} | {row['paired_delta_RN_vs_cached_N2_EM20']:.6f} | {row['relative_delta_percent_vs_cached_N2_EM20']:.3f}% |"
        )
    report_md.extend(
        [
            "",
            "| Variant | p | Equal-geometry mean R/N | Paired delta vs cached H20 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in macro_p_rows:
        if "p_evaluator_only" not in row:
            continue
        report_md.append(
            f"| {row['variant']} | {row['p_evaluator_only']:.1f} | {row['equal_geometry_macro_RN']:.6f} | {row['paired_delta_RN_vs_cached_N2_EM20']:.6f} |"
        )
    report_md.extend(
        [
            "",
            "## Prequential reliability",
            "",
            f"The learned pooled q_prior is {q_prior:.6f}. D_select q_t prequential Brier = {calibration['q_t_prequential_macro_Brier']:.6f}; constant-prior Brier = {calibration['constant_q_prior_macro_Brier']:.6f}; gate = **{calibration['gate_status']}**.",
            "",
            "| Fixed q_t bin | Event coverage | Mean q_t | Observed hit rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in calibration["fixed_reliability_bins"]:
        mean_q = "—" if row["mean_q_t"] is None else f"{row['mean_q_t']:.4f}"
        hit_rate = "—" if row["observed_hit_rate"] is None else f"{row['observed_hit_rate']:.4f}"
        report_md.append(
            f"| {row['bin']} | {row['event_coverage_count']} | {mean_q} | {hit_rate} |"
        )
    report_md.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "All D_fit/D_select rows completed legally and satisfy the relocation bound. D_select is a development check, not a final or sealed test. The N2 primary conclusion is unchanged; no D_screen scoring, further fitting, GP/RL training, or Bayes DP was run.",
            "",
            "Machine-readable checks, per-layout rows, sample states, evaluator-only labels, model coefficients, calibration events, and hashes are in the adjacent N3-prefixed files.",
            "",
        ]
    )
    with (output / "N3_FIT_REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report_md))
    total_elapsed = fit_elapsed + select_elapsed
    print(
        f"N3 fit/select COMPLETE D_fit={len(fit_rows)}/320 labels={len(label_rows)} "
        f"states={len(sample_rows)} D_select={len(select_rows)}/960 "
        f"q_gate={calibration['gate_status']} total_scoring_seconds={total_elapsed:.2f}",
        flush=True,
    )
    return report


def _n3_assert_screen_model_binding(
    model_document: dict[str, Any], actual_sha256: str, expected_sha256: str, q_prior: float
) -> None:
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen N3 model hash mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    if not math.isclose(
        float(model_document["q_prior_fit_only"]), q_prior, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("N3 model q_prior differs from its frozen D_fit estimate")
    if tuple(model_document["models"]) != ("SR", "R", "S"):
        raise ValueError("frozen N3 model variants differ from the released screen")
    expected_width = {"SR": 8, "R": 4, "S": 4}
    for variant, width in expected_width.items():
        coefficients = model_document["models"][variant]["coefficients"]
        if len(coefficients) != width or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"frozen {variant} coefficients are invalid")


def _n3_load_screen_episodes(
    config: dict[str, Any], output: Path
) -> tuple[list[OnlineCRPEpisode], dict[str, str], dict[str, Path]]:
    screen_config = config["n3_screen"]
    n1_report_path = output / "N1_CONTRACT_REPORT.json"
    n1_report = json.loads(n1_report_path.read_text(encoding="utf-8"))
    data_path = ROOT / str(screen_config["screen_layouts"])
    actual_hash = file_sha256(data_path)
    expected_hash = str(screen_config["expected_data_sha256"])
    n1_hash = str(n1_report["data"]["generated_splits"]["screen"]["sha256"])
    if actual_hash != expected_hash or actual_hash != n1_hash:
        raise ValueError(
            f"frozen D_screen hash mismatch: config={expected_hash}, N1={n1_hash}, actual={actual_hash}"
        )
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    episodes = [OnlineCRPEpisode.from_record(row) for row in rows]
    if len(episodes) != int(screen_config["expected_layout_count"]):
        raise ValueError("frozen D_screen layout count differs from release")
    fingerprints = [label_invariant_fingerprint(episode) for episode in episodes]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("D_screen contains duplicate semantic layouts")
    geometry_counts = {
        geometry_id: sum(episode.parameter_group == geometry_id for episode in episodes)
        for geometry_id in ("G1", "G2")
    }
    if geometry_counts != {"G1": 40, "G2": 40}:
        raise ValueError(f"D_screen geometry counts differ: {geometry_counts}")
    if any(episode.metadata.get("split") != "screen" for episode in episodes):
        raise ValueError("D_screen contains a non-screen episode")

    fit_rows = json.loads((ROOT / config["n3_fit_select"]["fit_layouts"]).read_text(encoding="utf-8"))
    select_rows = json.loads((ROOT / config["n3_fit_select"]["select_layouts"]).read_text(encoding="utf-8"))
    frozen_other = {
        label_invariant_fingerprint(OnlineCRPEpisode.from_record(row))
        for row in fit_rows + select_rows
    }
    if set(fingerprints) & frozen_other:
        raise ValueError("D_screen overlaps a frozen D_fit or D_select layout")
    return episodes, {"screen": actual_hash}, {"screen": data_path, "n1_report": n1_report_path}


def _n3_timed_screen_action(
    method: str,
    observation: PublicObservation,
    environment: OnlineCRPEnvironment,
    legal: tuple[int, ...],
    max_tiers: int,
    models: dict[str, Any],
) -> tuple[int, int | None, float | None, Any | None, float]:
    started = time.perf_counter_ns()
    decision = None
    requested_h: int | None = None
    predicted_delta: float | None = None
    if method.startswith("EM"):
        requested_h = int(method[2:])
        action = select_em_destination(
            observation, max_tiers=max_tiers, h=requested_h
        )
    elif method in ("SR", "R", "S"):
        decision = select_sr_em(
            observation,
            max_tiers=max_tiers,
            variant=method,  # type: ignore[arg-type]
            coefficients=tuple(models[method]["coefficients"]),
        )
        action = decision.destination
        requested_h = decision.requested_h
        predicted_delta = float(decision.predicted_delta)
    elif method == "Leveling":
        action = LevelingBaseline.select_destination(environment.state, legal)
    else:
        raise ValueError(f"unknown N3 screen method {method!r}")
    elapsed_us = (time.perf_counter_ns() - started) / 1000.0
    return int(action), requested_h, predicted_delta, decision, float(elapsed_us)


def _n3_paired_sr_em20_calls(
    observation: PublicObservation,
    environment: OnlineCRPEnvironment,
    legal: tuple[int, ...],
    max_tiers: int,
    models: dict[str, Any],
    decision_ordinal: int,
) -> tuple[Any, int, float, float, str]:
    """Run one timed SR action and one discarded EM20 call on the same public state."""
    if decision_ordinal % 2 == 0:
        order = "SR_then_EM20"
        _sr_action, _h, _delta, decision, sr_us = _n3_timed_screen_action(
            "SR", observation, environment, legal, max_tiers, models
        )
        em20_action, _h20, _d20, _unused, em20_us = _n3_timed_screen_action(
            "EM20", observation, environment, legal, max_tiers, models
        )
    else:
        order = "EM20_then_SR"
        em20_action, _h20, _d20, _unused, em20_us = _n3_timed_screen_action(
            "EM20", observation, environment, legal, max_tiers, models
        )
        _sr_action, _h, _delta, decision, sr_us = _n3_timed_screen_action(
            "SR", observation, environment, legal, max_tiers, models
        )
    if decision is None or em20_action not in legal or decision.destination not in legal:
        raise AssertionError("paired SR/EM20 actor produced an illegal action")
    return decision, int(em20_action), sr_us, em20_us, order


def _n3_screen_warmup(
    episode: OnlineCRPEpisode,
    *,
    p: float,
    repeat: int,
    q_prior: float,
    models: dict[str, Any],
) -> dict[str, Any]:
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    environment = OnlineCRPEnvironment(episode, max_steps=max_steps, lookahead_horizon=0)
    if environment.state.terminated:
        return {"performed": False, "reason": "warmup layout has no decision state"}
    fingerprint = label_invariant_fingerprint(episode)
    seed = semantic_forecast_seed(fingerprint, repeat)
    provider = ForecastProvider(
        episode.retrieval_sequence, p=p, seed=seed, q_prior=q_prior, max_horizon=MAX_HORIZON
    )
    sync_retrieval_events(
        provider, episode.retrieval_sequence, 0, int(environment.state.retrieval_count)
    )
    observation = provider.public_observation(
        _private_physical_stacks(environment),
        environment.legal_destinations(),
        max_tiers=episode.max_tiers,
    )
    legal = environment.legal_destinations()
    for method in SCREEN_METHODS:
        _n3_timed_screen_action(
            method, observation, environment, legal, episode.max_tiers, models
        )
    return {
        "performed": True,
        "layout_fingerprint": fingerprint,
        "p_evaluator_only": float(p),
        "repeat": int(repeat),
        "actor_methods_warmed_once": list(SCREEN_METHODS),
        "environment_steps": 0,
    }


def _n3_run_screen_rollout(
    episode: OnlineCRPEpisode,
    *,
    fingerprint: str,
    p: float,
    repeat: int,
    seed: int,
    q_prior: float,
    forecast: tuple[int, ...],
    forecast_hash: str,
    method: str,
    models: dict[str, Any],
    receipt: dict[str, Any],
    decision_handle: Any | None = None,
) -> tuple[dict[str, Any], tuple[int, ...], tuple[bool, ...], list[float], list[float], list[float]]:
    max_steps = episode.num_containers * (episode.max_tiers - 1)
    environment = OnlineCRPEnvironment(episode, max_steps=max_steps, lookahead_horizon=0)
    provider = ForecastProvider(
        episode.retrieval_sequence,
        p=p,
        seed=seed,
        q_prior=q_prior,
        max_horizon=MAX_HORIZON,
        forecast_future=forecast,
    )
    event_rows: list[dict[str, Any]] = []
    sync_retrieval_events(
        provider,
        episode.retrieval_sequence,
        0,
        int(environment.state.retrieval_count),
        event_rows=event_rows,
    )
    decisions = 0
    actions: list[int] = []
    actor_times: list[float] = []
    paired_sr_times: list[float] = []
    paired_em20_times: list[float] = []
    illegal_count = 0
    while not environment.state.terminated:
        legal = environment.legal_destinations()
        observation = provider.public_observation(
            _private_physical_stacks(environment), legal, max_tiers=episode.max_tiers
        )
        paired_em20_action: int | None = None
        timing_order: str | None = None
        paired_sr_us: float | None = None
        paired_em20_us: float | None = None
        if method == "SR":
            decision, paired_em20_action, paired_sr_us, paired_em20_us, timing_order = (
                _n3_paired_sr_em20_calls(
                    observation,
                    environment,
                    legal,
                    episode.max_tiers,
                    models,
                    decisions,
                )
            )
            action = int(decision.destination)
            requested_h = int(decision.requested_h)
            predicted_delta = float(decision.predicted_delta)
            actor_us = paired_sr_us
            paired_sr_times.append(paired_sr_us)
            paired_em20_times.append(paired_em20_us)
            chosen = next(
                candidate
                for candidate in decision.candidates
                if candidate.destination == decision.destination
                and candidate.requested_h == decision.requested_h
            )
            remaining_future = max(
                0,
                episode.num_containers - int(environment.state.retrieval_count) - 1,
            )
            decision_row = {
                "fingerprint": fingerprint,
                "geometry_id": episode.parameter_group,
                "p_evaluator_only": float(p),
                "r_evaluator_only": int(repeat),
                "decision_ordinal": int(decisions),
                "q_t": float(observation.q_t),
                "remaining_containers_including_current": int(
                    episode.num_containers - int(environment.state.retrieval_count)
                ),
                "remaining_future_count": int(remaining_future),
                "legal_destination_count": len(legal),
                "deduplicated_candidate_count": len(decision.candidates),
                "no_nonzero_candidate_available": len(decision.candidates) == 1,
                "selector_fallback_to_h0": (
                    len(decision.candidates) > 1 and decision.requested_h == 0
                ),
                "action_destination_stack": action,
                "paired_em20_discarded_destination_evaluator_only": paired_em20_action,
                "requested_source_h": requested_h,
                "effective_prefix_h": min(requested_h, remaining_future),
                "selected_public_c": float(chosen.features[3]),
                "selected_public_v": float(chosen.features[4]),
                "predicted_delta_R": predicted_delta,
                "paired_sr_actor_time_us": paired_sr_us,
                "paired_em20_actor_time_us": paired_em20_us,
                "paired_call_order": timing_order,
                "source_bundle_sha256": receipt["source_bundle_sha256"],
                "config_sha256": receipt["config_sha256"],
                "screen_data_sha256": receipt["data_sha256"]["screen"],
                "model_sha256": receipt["model_sha256"],
            }
            if decision_handle is not None:
                _n3_write_jsonl_row(decision_handle, decision_row)
        else:
            action, _requested, predicted_delta, decision, actor_us = _n3_timed_screen_action(
                method, observation, environment, legal, episode.max_tiers, models
            )
            requested_h = -1 if _requested is None else int(_requested)
        if action not in legal:
            illegal_count += 1
            raise ValueError(f"{method} selected illegal destination {action}")
        actor_times.append(float(actor_us))
        actions.append(int(action))
        decisions += 1
        before = int(environment.state.retrieval_count)
        environment.step(action)
        after = int(environment.state.retrieval_count)
        sync_retrieval_events(
            provider,
            episode.retrieval_sequence,
            before,
            after,
            event_rows=event_rows,
        )
        if decisions > max_steps:
            raise RuntimeError("D_screen rollout exceeded R<=N(T-1)")

    actor_array = np.asarray(actor_times, dtype=np.float64)
    elapsed_sum = float(actor_array.sum()) if actor_array.size else 0.0
    row = {
        "fingerprint": fingerprint,
        "geometry_id": episode.parameter_group,
        "N": episode.num_containers,
        "max_tiers": episode.max_tiers,
        "split": "screen",
        "p_evaluator_only": float(p),
        "r_evaluator_only": int(repeat),
        "method": method,
        "forecast_hash": forecast_hash,
        "R": int(environment.relocation_count),
        "RN": float(environment.relocation_count / episode.num_containers),
        "decision_count": int(decisions),
        "completed": bool(environment.state.terminated),
        "retrieved_count": int(environment.state.retrieval_count),
        "illegal_action_count": illegal_count,
        "reveal_event_count": len(event_rows),
        "reveal_hit_history_sha256": hashlib.sha256(
            bytes(int(event["hit"]) for event in event_rows)
        ).hexdigest(),
        "action_sequence_sha256": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "actor_time_total_us": elapsed_sum,
        "actor_time_mean_us": float(actor_array.mean()) if actor_array.size else 0.0,
        "actor_time_p95_us": float(np.percentile(actor_array, 95)) if actor_array.size else 0.0,
        "actor_time_decision_count": int(actor_array.size),
        "paired_actor_event_count": len(paired_sr_times),
        "paired_sr_actor_time_total_us": float(sum(paired_sr_times)),
        "paired_em20_actor_time_total_us": float(sum(paired_em20_times)),
        "source_bundle_sha256": receipt["source_bundle_sha256"],
        "config_sha256": receipt["config_sha256"],
        "screen_data_sha256": receipt["data_sha256"]["screen"],
        "model_sha256": receipt["model_sha256"],
    }
    return (
        row,
        tuple(actions),
        tuple(bool(event["hit"]) for event in event_rows),
        actor_times,
        paired_sr_times,
        paired_em20_times,
    )


def _n3_screen_statistics(
    rows: list[dict[str, Any]], episodes: list[OnlineCRPEpisode], config: dict[str, Any]
) -> dict[str, Any]:
    screen_cfg = config["n3_screen"]
    methods = tuple(screen_cfg["methods"])
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    fingerprints_by_geometry = {
        geometry_id: sorted(
            label_invariant_fingerprint(episode)
            for episode in episodes
            if episode.parameter_group == geometry_id
        )
        for geometry_id in ("G1", "G2")
    }
    row_by_key = {
        (
            str(row["fingerprint"]),
            float(row["p_evaluator_only"]),
            int(row["r_evaluator_only"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    expected = {
        (fingerprint, p, repeat, method)
        for fingerprint in {fp for values in fingerprints_by_geometry.values() for fp in values}
        for p in p_values
        for repeat in repeats
        for method in methods
    }
    if set(row_by_key) != expected or len(rows) != 5760:
        raise AssertionError(f"D_screen key budget mismatch: {len(row_by_key)}/5760")
    layout_values: dict[tuple[str, float, str], np.ndarray] = {}
    for geometry_id, fingerprints in fingerprints_by_geometry.items():
        for p in p_values:
            for method in methods:
                layout_values[(geometry_id, p, method)] = np.asarray(
                    [
                        np.mean(
                            [
                                row_by_key[(fingerprint, p, repeat, method)]["RN"]
                                for repeat in repeats
                            ]
                        )
                        for fingerprint in fingerprints
                    ],
                    dtype=np.float64,
                )
    macro: dict[tuple[str, float], float] = {}
    by_geometry_p = []
    for method in methods:
        for p in p_values:
            geometry_means = []
            for geometry_id in ("G1", "G2"):
                value = float(np.mean(layout_values[(geometry_id, p, method)]))
                geometry_means.append(value)
                by_geometry_p.append(
                    {
                        "method": method,
                        "geometry_id": geometry_id,
                        "p_evaluator_only": p,
                        "layout_count": len(fingerprints_by_geometry[geometry_id]),
                        "macro_RN": value,
                    }
                )
            macro[(method, p)] = float(np.mean(geometry_means))
    overall = {
        method: float(np.mean([macro[(method, p)] for p in p_values]))
        for method in methods
    }

    rng = np.random.Generator(np.random.PCG64(int(screen_cfg["bootstrap_seed"])))
    boot: dict[str, list[float]] = {
        "primary_relative_SR_gain_percent": [],
        "SR_vs_EM0_p03_deterioration_percent": [],
        "SR_vs_EM0_p05_deterioration_percent": [],
        "SR_vs_EM20_p0_deterioration_percent": [],
        "SR_vs_R_relative_gain_percent": [],
    }
    geometry_fp_arrays = {
        geometry_id: np.asarray(fingerprints_by_geometry[geometry_id], dtype=object)
        for geometry_id in ("G1", "G2")
    }
    n_boot = int(screen_cfg["bootstrap_replicates"])
    for _ in range(n_boot):
        geom_macro: dict[tuple[str, str, float], float] = {}
        for geometry_id in ("G1", "G2"):
            fps = geometry_fp_arrays[geometry_id]
            draw = rng.integers(0, len(fps), size=len(fps))
            for method in methods:
                for p in p_values:
                    geom_macro[(geometry_id, method, p)] = float(
                        np.mean(layout_values[(geometry_id, p, method)][draw])
                    )
        boot_macro = {
            (method, p): float(
                np.mean(
                    [geom_macro[(geometry_id, method, p)] for geometry_id in ("G1", "G2")]
                )
            )
            for method in methods
            for p in p_values
        }
        all_method = {
            method: float(np.mean([boot_macro[(method, p)] for p in p_values]))
            for method in methods
        }
        boot["primary_relative_SR_gain_percent"].append(
            100.0 * (all_method["EM20"] - all_method["SR"]) / all_method["EM20"]
        )
        boot["SR_vs_EM0_p03_deterioration_percent"].append(
            100.0 * (boot_macro[("SR", 0.3)] - boot_macro[("EM0", 0.3)])
            / boot_macro[("EM0", 0.3)]
        )
        boot["SR_vs_EM0_p05_deterioration_percent"].append(
            100.0 * (boot_macro[("SR", 0.5)] - boot_macro[("EM0", 0.5)])
            / boot_macro[("EM0", 0.5)]
        )
        boot["SR_vs_EM20_p0_deterioration_percent"].append(
            100.0 * (boot_macro[("SR", 0.0)] - boot_macro[("EM20", 0.0)])
            / boot_macro[("EM20", 0.0)]
        )
        boot["SR_vs_R_relative_gain_percent"].append(
            100.0 * (all_method["R"] - all_method["SR"]) / all_method["R"]
        )
    alpha = 1.0 - float(screen_cfg["ci_level"])
    bootstrap_ci = {
        key: {
            "lower": float(np.quantile(values, alpha / 2.0)),
            "upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "replicates": len(values),
            "seed": int(screen_cfg["bootstrap_seed"]),
            "cluster": "layout fingerprint; resampled within geometry with all p/method/r pairings retained",
        }
        for key, values in boot.items()
    }
    primary_gain = 100.0 * (overall["EM20"] - overall["SR"]) / overall["EM20"]
    p03_deterioration = 100.0 * (macro[("SR", 0.3)] - macro[("EM0", 0.3)]) / macro[("EM0", 0.3)]
    p05_deterioration = 100.0 * (macro[("SR", 0.5)] - macro[("EM0", 0.5)]) / macro[("EM0", 0.5)]
    p0_deterioration = 100.0 * (macro[("SR", 0.0)] - macro[("EM20", 0.0)]) / macro[("EM20", 0.0)]
    sr_vs_r_gain = 100.0 * (overall["R"] - overall["SR"]) / overall["R"]
    return {
        "aggregation": "average r within layout; layout within geometry; equal geometry and p",
        "macro_RN_by_method": overall,
        "macro_RN_by_method_p": [
            {"method": method, "p_evaluator_only": p, "macro_RN": macro[(method, p)]}
            for method in methods
            for p in p_values
        ],
        "by_geometry_p": by_geometry_p,
        "bootstrap_ci": bootstrap_ci,
        "gates": {
            "primary_SR_improvement_percent_vs_EM20": primary_gain,
            "primary_minimum_1_percent": "PASS" if primary_gain >= 1.0 else "NOT_SUPPORTED",
            "primary_90pct_CI_lower_gt_zero": (
                "PASS"
                if bootstrap_ci["primary_relative_SR_gain_percent"]["lower"] > 0.0
                else "NOT_SUPPORTED"
            ),
            "high_noise_SR_deterioration_percent_vs_EM0_p03": p03_deterioration,
            "high_noise_p03_at_most_0_5_percent": "PASS" if p03_deterioration <= 0.5 else "NOT_SUPPORTED",
            "high_noise_SR_deterioration_percent_vs_EM0_p05": p05_deterioration,
            "high_noise_p05_at_most_0_5_percent": "PASS" if p05_deterioration <= 0.5 else "NOT_SUPPORTED",
            "p0_SR_deterioration_percent_vs_EM20": p0_deterioration,
            "p0_at_most_0_5_percent": "PASS" if p0_deterioration <= 0.5 else "NOT_SUPPORTED",
            "SR_relative_improvement_percent_vs_R": sr_vs_r_gain,
            "SR_improvement_at_least_0_5_percent_vs_R": "PASS" if sr_vs_r_gain >= 0.5 else "NOT_SUPPORTED",
        },
        "raw_method_row_count": len(rows),
    }


def _n3_verify_original_snapshot(output: Path) -> dict[str, Any]:
    receipt_path = output / "N3_SNAPSHOT_RECEIPT.json"
    snapshot_path = output / "N3_SOURCE_SNAPSHOT.zip"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    actual_zip_sha = file_sha256(snapshot_path)
    if actual_zip_sha != receipt["snapshot_sha256"]:
        raise ValueError("pre-screen N3 source snapshot hash differs from its receipt")
    if int(receipt["member_count"]) != 7:
        raise ValueError("pre-screen N3 source snapshot must contain the frozen seven members")
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        member_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
        }
    if member_hashes != receipt["members_sha256"]:
        raise ValueError("pre-screen N3 source snapshot member hash mismatch")
    return {
        "snapshot_sha256": actual_zip_sha,
        "receipt_sha256": file_sha256(receipt_path),
        "member_hashes": member_hashes,
    }


def _n3_create_screen_snapshot(
    output: Path, config_path: Path
) -> dict[str, Any]:
    source_paths = (
        ROOT / "scrp/noise_aware_forecast.py",
        ROOT / "scrp/noise_aware_em.py",
        ROOT / "experiments/noise_aware_lookahead.py",
        BASE_ROOT / "scrp/online_crp.py",
        BASE_ROOT / "scrp/attention_data.py",
        ROOT / "tests/test_noise_aware_lookahead.py",
    )
    member_paths = [*source_paths, config_path]
    member_bytes = {
        _source_path_label(path): path.read_bytes() for path in member_paths
    }
    member_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(member_bytes.items())
    }
    snapshot_path = output / "N3_SCREEN_SOURCE_SNAPSHOT.zip"
    with zipfile.ZipFile(snapshot_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(member_bytes.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(snapshot_path, "r") as archive:
        archived_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
        }
    if archived_hashes != member_hashes:
        raise RuntimeError("N3 screen snapshot post-pack source hash mismatch")
    record = {
        "snapshot_path": snapshot_path.relative_to(ROOT).as_posix(),
        "snapshot_sha256": file_sha256(snapshot_path),
        "member_count": len(member_hashes),
        "members_sha256": member_hashes,
        "archive_member_sha256_match": True,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }
    _n3_write_new_json(output / "N3_SCREEN_SNAPSHOT_RECEIPT.json", record)
    return record


def _n3_test_receipt(output: Path) -> tuple[str, str, str]:
    test_path = output / "N3_SCREEN_PREFLIGHT_TEST_STDOUT.txt"
    text = test_path.read_text(encoding="utf-8-sig")
    if " failed" in text or " passed" not in text:
        raise ValueError("N3 screen targeted tests did not record a clean pass")
    command = next(
        (line.removeprefix("COMMAND:").strip() for line in text.splitlines() if line.startswith("COMMAND:")),
        None,
    )
    summary = next(
        (line.strip() for line in reversed(text.splitlines()) if " passed" in line), None
    )
    if not command or not summary:
        raise ValueError("N3 screen targeted test stdout lacks exact command/pass summary")
    return file_sha256(test_path), command, summary


def run_screen(config_path: Path, output: Path) -> dict[str, Any]:
    """Run the one released, frozen-model N3 D_screen evaluation."""
    config = _read_config(config_path)
    if "screen" not in config["release_scope"]["allowed_stages"]:
        raise RuntimeError("N3 D_screen is not released by the active config")
    screen_cfg = config["n3_screen"]
    episodes, data_hashes, data_paths = _n3_load_screen_episodes(config, output)
    model_path = output / "N3_MODEL.json"
    model_sha = file_sha256(model_path)
    model_document = json.loads(model_path.read_text(encoding="utf-8"))
    q_prior = int(screen_cfg["q_prior_hits"]) / int(screen_cfg["q_prior_events"])
    _n3_assert_screen_model_binding(
        model_document,
        model_sha,
        str(screen_cfg["frozen_model_sha256"]),
        q_prior,
    )
    model_variants = model_document["models"]
    old_snapshot = _n3_verify_original_snapshot(output)
    test_sha, test_command, test_summary = _n3_test_receipt(output)
    collisions = [
        name
        for name in (
            "N3_SCREEN_SOURCE_SNAPSHOT.zip",
            "N3_SCREEN_SNAPSHOT_RECEIPT.json",
            "N3_SCREEN_RUN_RECEIPT.json",
            "N3_SCREEN_ROWS.jsonl",
            "N3_SCREEN_DECISIONS.jsonl",
            "N3_SCREEN_REPORT.json",
            "N3_SCREEN_REPORT.md",
        )
        if (output / name).exists()
    ]
    if collisions:
        raise FileExistsError("N3 screen outputs already exist: " + ", ".join(collisions))

    snapshot = _n3_create_screen_snapshot(output, config_path)
    source_sha = {
        _source_path_label(path): file_sha256(path)
        for path in (
            ROOT / "scrp/noise_aware_forecast.py",
            ROOT / "scrp/noise_aware_em.py",
            ROOT / "experiments/noise_aware_lookahead.py",
            BASE_ROOT / "scrp/online_crp.py",
            BASE_ROOT / "scrp/attention_data.py",
            ROOT / "tests/test_noise_aware_lookahead.py",
        )
    }
    source_bundle_sha = hashlib.sha256(
        json.dumps(source_sha, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    n3_run_path = output / "N3_RUN_RECEIPT.json"
    n3_fit_report_path = output / "N3_FIT_REPORT.json"
    fit_path = ROOT / config["n3_fit_select"]["fit_layouts"]
    select_path = ROOT / config["n3_fit_select"]["select_layouts"]
    receipt: dict[str, Any] = {
        "stage": "N3_screen_only",
        "status": "STARTED",
        "job_pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cli": "py -3.12 -B -m experiments.noise_aware_lookahead --config configs/noise_aware_lookahead_v1.json --stage screen",
        "source_sha256": source_sha,
        "source_bundle_sha256": source_bundle_sha,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": file_sha256(config_path),
        "test_stdout_sha256": test_sha,
        "test_command": test_command,
        "test_summary": test_summary,
        "screen_snapshot_sha256": snapshot["snapshot_sha256"],
        "screen_snapshot_receipt_sha256": file_sha256(
            output / "N3_SCREEN_SNAPSHOT_RECEIPT.json"
        ),
        "prior_n3_snapshot": old_snapshot,
        "n3_run_receipt_sha256": file_sha256(n3_run_path),
        "n3_fit_report_sha256": file_sha256(n3_fit_report_path),
        "model_sha256": model_sha,
        "q_prior": {"hits": 5183, "events": 9920, "value": q_prior},
        "data_paths": {
            "screen": data_paths["screen"].relative_to(ROOT).as_posix(),
            "fit": fit_path.relative_to(ROOT).as_posix(),
            "select": select_path.relative_to(ROOT).as_posix(),
        },
        "data_sha256": {
            "screen": data_hashes["screen"],
            "fit": file_sha256(fit_path),
            "select": file_sha256(select_path),
            "n1_report": file_sha256(data_paths["n1_report"]),
        },
        "scientific_budget": {
            "layouts": len(episodes),
            "conditions": 5760,
            "methods": list(SCREEN_METHODS),
            "fit_or_select_rows_changed": False,
            "new_labels_or_refit": False,
            "gp_rl_or_dp": False,
            "screen_time_limit_minutes": int(screen_cfg["wall_clock_limit_minutes"]),
        },
        "stage_status": "RUNNING",
    }
    _n3_write_new_json(output / "N3_SCREEN_RUN_RECEIPT.json", receipt)
    print(
        f"N3 screen START pid={os.getpid()} layouts={len(episodes)} budget=5760 model={model_sha}",
        flush=True,
    )

    warmup = _n3_screen_warmup(
        episodes[0], p=0.0, repeat=0, q_prior=q_prior, models=model_variants
    )
    p_values = tuple(float(value) for value in config["forecast"]["p_values"])
    repeats = tuple(int(value) for value in config["forecast"]["repeats"])
    all_rows: list[dict[str, Any]] = []
    actor_samples: dict[str, list[float]] = {method: [] for method in SCREEN_METHODS}
    paired_sr_samples: list[float] = []
    paired_em20_samples: list[float] = []
    condition_checks: dict[tuple[str, float, int], dict[str, Any]] = {}
    baseline_checks: dict[tuple[str, str], tuple[tuple[int, ...], int]] = {}
    p0_checks: dict[tuple[str, str], tuple[tuple[int, ...], int, str]] = {}
    start = time.perf_counter()
    try:
        with (output / "N3_SCREEN_ROWS.jsonl").open("x", encoding="utf-8", newline="\n") as row_handle, \
             (output / "N3_SCREEN_DECISIONS.jsonl").open("x", encoding="utf-8", newline="\n") as decision_handle:
            for episode_index, episode in enumerate(episodes):
                fingerprint = label_invariant_fingerprint(episode)
                for p in p_values:
                    for repeat in repeats:
                        seed = semantic_forecast_seed(fingerprint, repeat)
                        private_provider = ForecastProvider(
                            episode.retrieval_sequence,
                            p=p,
                            seed=seed,
                            q_prior=q_prior,
                            max_horizon=MAX_HORIZON,
                        )
                        forecast = tuple(private_provider._forecast_future)
                        forecast_hash = _forecast_hash(forecast)
                        condition_key = (fingerprint, p, repeat)
                        condition_records: dict[str, Any] = {}
                        for method in SCREEN_METHODS:
                            (
                                row,
                                actions,
                                hit_history,
                                method_times,
                                paired_sr,
                                paired_em20,
                            ) = _n3_run_screen_rollout(
                                episode,
                                fingerprint=fingerprint,
                                p=p,
                                repeat=repeat,
                                seed=seed,
                                q_prior=q_prior,
                                forecast=forecast,
                                forecast_hash=forecast_hash,
                                method=method,
                                models=model_variants,
                                receipt=receipt,
                                decision_handle=decision_handle if method == "SR" else None,
                            )
                            row["forecast_seed_evaluator_only"] = int(seed)
                            _n3_write_jsonl_row(row_handle, row)
                            _n3_assert_rollout_complete(row, "D_screen")
                            if not math.isclose(
                                float(row["RN"]),
                                int(row["R"]) / int(row["N"]),
                                rel_tol=0.0,
                                abs_tol=1e-15,
                            ):
                                raise RuntimeError("N3 D_screen fail-stop: RN arithmetic mismatch")
                            if int(row["reveal_event_count"]) != len(hit_history):
                                raise RuntimeError("N3 D_screen fail-stop: reveal history count mismatch")
                            actor_samples[method].extend(method_times)
                            paired_sr_samples.extend(paired_sr)
                            paired_em20_samples.extend(paired_em20)
                            condition_records[method] = {
                                "forecast_hash": row["forecast_hash"],
                                "hit_history_sha256": row["reveal_hit_history_sha256"],
                                "actions": actions,
                                "R": int(row["R"]),
                            }
                            all_rows.append(row)
                        forecast_hashes = {
                            value["forecast_hash"] for value in condition_records.values()
                        }
                        hit_hashes = {
                            value["hit_history_sha256"] for value in condition_records.values()
                        }
                        if len(forecast_hashes) != 1 or len(hit_hashes) != 1:
                            raise RuntimeError(
                                "N3 D_screen fail-stop: methods diverged in shared forecast/reveal history"
                            )
                        if condition_records["EM0"]["actions"] != condition_records["Leveling"]["actions"] or condition_records["EM0"]["R"] != condition_records["Leveling"]["R"]:
                            raise RuntimeError("N3 D_screen fail-stop: EM0 and Leveling differ")
                        for method in ("EM0", "Leveling"):
                            key = (fingerprint, method)
                            current = (condition_records[method]["actions"], condition_records[method]["R"])
                            prior = baseline_checks.setdefault(key, current)
                            if prior != current:
                                raise RuntimeError(
                                    f"N3 D_screen fail-stop: {method} changed across p/r"
                                )
                        if p == 0.0:
                            for method in SCREEN_METHODS:
                                key = (fingerprint, method)
                                current = (
                                    condition_records[method]["actions"],
                                    condition_records[method]["R"],
                                    next(iter(forecast_hashes)),
                                )
                                prior = p0_checks.setdefault(key, current)
                                if prior != current:
                                    raise RuntimeError(
                                        f"N3 D_screen fail-stop: p=0 repeat changed for {method}"
                                    )
                        condition_checks[condition_key] = condition_records
                if (episode_index + 1) % 10 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"N3 screen progress layouts={episode_index + 1}/80 rows={len(all_rows)}/5760 elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
                if time.perf_counter() - start > int(screen_cfg["wall_clock_limit_minutes"]) * 60:
                    raise TimeoutError("N3 D_screen exceeded its 20-minute scientific budget")

        if len(all_rows) != 5760:
            raise RuntimeError(f"N3 D_screen stopped with {len(all_rows)}/5760 rows")
        summary = _n3_screen_statistics(all_rows, episodes, config)
        paired_count = len(paired_sr_samples)
        if paired_count != len(paired_em20_samples) or paired_count == 0:
            raise RuntimeError("N3 paired cost sample is empty or mismatched")
        paired_sr_total = float(sum(paired_sr_samples))
        paired_em20_total = float(sum(paired_em20_samples))
        paired_cost = {
            "matched_decision_count": paired_count,
            "mean_cost_ratio_sr_over_em20": paired_sr_total / paired_em20_total,
            "mean_sr_us": float(np.mean(paired_sr_samples)),
            "mean_em20_us": float(np.mean(paired_em20_samples)),
            "p95_sr_us": float(np.percentile(paired_sr_samples, 95)),
            "p95_em20_us": float(np.percentile(paired_em20_samples, 95)),
            "call_order_counts": {
                "SR_then_EM20": sum(
                    int(row["decision_count"] // 2 + row["decision_count"] % 2)
                    for row in all_rows
                    if row["method"] == "SR"
                ),
                "EM20_then_SR": sum(
                    int(row["decision_count"] // 2)
                    for row in all_rows
                    if row["method"] == "SR"
                ),
            },
            "gate_at_most_3x": (
                "PASS"
                if paired_sr_total / paired_em20_total <= 3.0
                else "NOT_SUPPORTED"
            ),
            "timing_scope": "actor-only; candidate generation/features/selection included; environment, forecast rendering and logging excluded",
        }
        own_cost = {
            method: {
                "decision_count": len(actor_samples[method]),
                "mean_us": float(np.mean(actor_samples[method])) if actor_samples[method] else None,
                "p95_us": float(np.percentile(actor_samples[method], 95)) if actor_samples[method] else None,
                "total_us": float(sum(actor_samples[method])),
            }
            for method in SCREEN_METHODS
        }
        summary["paired_actor_cost"] = paired_cost
        summary["actor_cost_on_own_trajectory"] = own_cost
        summary["gates"]["matched_actor_cost_at_most_3x"] = paired_cost["gate_at_most_3x"]
        summary["checks"] = {
            "unique_complete_rollouts": len(all_rows),
            "shared_forecast_and_reveal_history_by_condition": "PASS",
            "EM0_equals_Leveling_actions_and_R": "PASS",
            "EM0_and_Leveling_invariant_across_p_r": "PASS",
            "all_methods_p0_invariant_across_repeat": "PASS",
            "all_rows_complete_legal_and_R_bound": "PASS",
            "decision_rows_written_for_SR": sum(
                int(row["decision_count"]) for row in all_rows if row["method"] == "SR"
            ),
            "q_prior_source": "frozen D_fit hits/events; not re-estimated on D_screen",
            "warmup": warmup,
        }
        summary["evidence_boundary"] = {
            "development_screen_only": True,
            "N2_primary": "NOT_SUPPORTED; unchanged",
            "no_refit_no_new_labels_no_hyperparameter_tuning": True,
            "sealed_final_test": False,
            "N4_or_DP": "NOT_RELEASED_OR_RUN",
        }
        rows_sha = file_sha256(output / "N3_SCREEN_ROWS.jsonl")
        decisions_sha = file_sha256(output / "N3_SCREEN_DECISIONS.jsonl")
        summary["artifact_sha256"] = {
            "N3_SCREEN_ROWS.jsonl": rows_sha,
            "N3_SCREEN_DECISIONS.jsonl": decisions_sha,
            "N3_SCREEN_SOURCE_SNAPSHOT.zip": snapshot["snapshot_sha256"],
            "N3_SCREEN_RUN_RECEIPT_source_bundle": source_bundle_sha,
        }
        _n3_write_new_json(output / "N3_SCREEN_REPORT.json", summary)
        report_lines = [
            "# PLAN-036 N3 D_screen development report",
            "",
            "**Status: complete development screen; N2 primary remains NOT_SUPPORTED.** The frozen N3 model was evaluated once on the existing D_screen split. This is not a sealed final test and does not release N4.",
            "",
            f"Rows: {len(all_rows)}/5760 complete legal rollouts. Layouts: {len(episodes)} (G1=40, G2=40). Model SHA256: `{model_sha}`. D_screen SHA256: `{data_hashes['screen']}`. q_prior: 5183/9920 = {q_prior:.10f}.",
            "",
            "## Macro R/N by method and p",
            "",
            "Aggregation averages r within layout, layouts within geometry, then gives geometries and p equal weight.",
            "",
            "| Method | p=0.0 | p=0.1 | p=0.3 | p=0.5 | Equal-p macro |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        macro_p = {
            (row["method"], float(row["p_evaluator_only"])): float(row["macro_RN"])
            for row in summary["macro_RN_by_method_p"]
        }
        for method in SCREEN_METHODS:
            cells = [macro_p[(method, p)] for p in p_values]
            report_lines.append(
                f"| {method} | " + " | ".join(f"{value:.6f}" for value in cells) + f" | {summary['macro_RN_by_method'][method]:.6f} |"
            )
        report_lines.extend(["", "## Screen gates", "", "| Gate | Result |", "|---|---|"])
        for name, value in summary["gates"].items():
            report_lines.append(f"| {name} | {value} |")
        report_lines.extend(
            [
                "",
                "## Paired actor-only cost",
                "",
                f"On {paired_count} SR public decision states, matched mean SR/EM20 cost ratio = {paired_cost['mean_cost_ratio_sr_over_em20']:.4f}; SR mean/p95 = {paired_cost['mean_sr_us']:.3f}/{paired_cost['p95_sr_us']:.3f} µs; EM20 mean/p95 = {paired_cost['mean_em20_us']:.3f}/{paired_cost['p95_em20_us']:.3f} µs. Cost gate: **{paired_cost['gate_at_most_3x']}**.",
                "",
                "| Method | Decision count | Own-trajectory mean µs | Own-trajectory p95 µs |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in SCREEN_METHODS:
            cost = own_cost[method]
            report_lines.append(
                f"| {method} | {cost['decision_count']} | {cost['mean_us']:.3f} | {cost['p95_us']:.3f} |"
            )
        report_lines.extend(
            [
                "",
                "## Evidence boundary",
                "",
                "The fixed H=20 reference was carried forward from N2 and not reselected. No screen labels, refitting, hyperparameter changes, GP/RL training, Bayes DP, or sealed final-test claims were made. A negative gate is reported as NOT_SUPPORTED without retuning or additional samples.",
                "",
                "Machine-readable row, per-SR-decision diagnostics, source/config snapshot and hashes are in adjacent `N3_SCREEN_*` artifacts. The screen code was frozen before this one scoring run.",
                "",
            ]
        )
        with (output / "N3_SCREEN_REPORT.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(report_lines))
        receipt.update(
            {
                "status": "COMPLETE",
                "stage_status": "COMPLETE",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_elapsed_seconds": time.perf_counter() - start,
                "completed_rows": len(all_rows),
                "completed_decision_rows": summary["checks"]["decision_rows_written_for_SR"],
                "rows_sha256": rows_sha,
                "decisions_sha256": decisions_sha,
                "report_json_sha256": file_sha256(output / "N3_SCREEN_REPORT.json"),
                "report_md_sha256": file_sha256(output / "N3_SCREEN_REPORT.md"),
                "screen_source_snapshot_sha256": snapshot["snapshot_sha256"],
                "result_gates": summary["gates"],
            }
        )
        (output / "N3_SCREEN_RUN_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"N3 screen COMPLETE rows={len(all_rows)}/5760 paired_cost_ratio={paired_cost['mean_cost_ratio_sr_over_em20']:.4f} gates={summary['gates']}",
            flush=True,
        )
        return summary
    except Exception as exc:
        receipt.update(
            {
                "status": "FAIL_STOP",
                "stage_status": "FAIL_STOP",
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "scientific_elapsed_seconds": time.perf_counter() - start,
                "completed_rows_before_stop": len(all_rows),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        (output / "N3_SCREEN_RUN_RECEIPT.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"N3 screen FAIL_STOP rows={len(all_rows)} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage", choices=("contract", "phenomenon", "fit", "screen"), default="contract"
    )
    args = parser.parse_args()
    if args.stage == "contract":
        run_contract(args.config.resolve(), OUTPUT)
    elif args.stage == "phenomenon":
        run_phenomenon(args.config.resolve(), N2_OUTPUT)
    elif args.stage == "fit":
        run_fit_select(args.config.resolve(), OUTPUT)
    else:
        run_screen(args.config.resolve(), OUTPUT)


if __name__ == "__main__":
    main()
