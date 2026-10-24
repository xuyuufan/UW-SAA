"""Compare the public package with a supplied frozen UWSAA source checkout.

This is a bounded correctness check: one detached decision frame per policy
arm and, when supplied, all frozen forecast rows from one archived layout.
It does not run a benchmark or measure performance.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.noise_aware_forecast import ForecastProvider, PublicObservation
from environment.online_crp import OnlineCRPEnvironment, OnlineCRPEpisode, load_episode_manifest
from environment.public_frame import PublicDecisionFrame, make_public_frame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from reproducibility.protocol import CONDITION_SETS, forecast_table
from reproducibility.runner import (
    DEFAULT_DATA, _make_policy, _sync_forecast_provider, load_runtime,
)
from uwsaa.variants import make_variant


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(episode: Any, condition: Mapping[str, Any]) -> object:
    table, _digest = forecast_table(episode, condition)
    provider = ForecastProvider(
        episode.retrieval_sequence, p=0.0, seed=0, q_prior=5183 / 9920,
        max_horizon=20, forecast_future=table[0],
    )
    env = OnlineCRPEnvironment(episode, lookahead_horizon=0)
    if env.state.terminated:
        return None
    _sync_forecast_provider(provider, episode, table, env.state.retrieval_count)
    legal = env.legal_destinations()
    observation = provider.public_observation(env._stacks, legal, max_tiers=episode.max_tiers)
    raw_frame = make_public_frame(observation, env.state, legal, max_tiers=episode.max_tiers)
    history = PublicHistoryTracker()
    return attach_history_summary(raw_frame, history.summary(raw_frame))


def _signature(frame: object) -> tuple[Any, ...]:
    if hasattr(frame, "observation"):
        obs = frame.observation
        ranks, occupied, legal_mask = obs.rank_tensor, obs.occupied_mask, obs.legal_destination_mask
        history_hits, q_prior, q_t = obs.history_hits, obs.q_prior, obs.q_t
        crane_position, crane_time = frame.crane_position, frame.crane_time
    else:
        ranks, occupied, legal_mask = frame.ranks, frame.occupied, frame.legal_mask
        history_hits, q_prior, q_t = frame.history_hits, frame.q_prior, frame.q_t
        crane_position, crane_time = frame.crane_position, frame.crane_time
    return (
        ranks.tolist(), occupied.astype(int).tolist(), tuple(frame.legal_destinations),
        int(frame.max_tiers), int(frame.retrieval_count), int(frame.relocation_count),
        history_hits, float(q_prior), float(q_t), crane_position, float(crane_time),
    )


def _frame_from_snapshot(original: object, snapshot: object) -> object:
    observation = replace(
        original.observation,
        rank_tensor=snapshot.ranks,
        occupied_mask=snapshot.occupied,
        legal_destination_mask=snapshot.legal_mask,
    )
    return replace(
        original, observation=observation,
        legal_destinations=snapshot.legal_destinations,
        relocation_count=snapshot.relocation_count,
        crane_position=snapshot.crane_position, crane_time=snapshot.crane_time,
    )


def _score_map(trace: Mapping[str, Any]) -> dict[tuple[int, ...], float]:
    values = {}
    for row in trace.get("candidate_scores", ()):
        sequence = tuple(int(value) for value in row["sequence"])
        score = next((row[key] for key in (
            "weighted_saa_Rraw", "mean_relocations", "weighted_score",
            "deterministic_minmax_completion_Rraw", "deterministic_mean_relocations",
        ) if key in row), None)
        if score is None:
            raise AssertionError("candidate score row has no registered score field")
        values[sequence] = float(score)
    return values


def _select_and_compare(source_policy: Any, public_policy: Any,
                        source_frame: object, public_frame: object,
                        *, check_group_commit: bool) -> dict[str, Any]:
    if _signature(source_frame) != _signature(public_frame):
        raise AssertionError("source/public native frames differ in actor-visible information")
    source_action, source_trace = source_policy.select(source_frame)
    public_action, public_trace = public_policy.select(public_frame)
    if int(source_action) != int(public_action):
        raise AssertionError(f"action differs: frozen={source_action}, public={public_action}")
    source_scores, public_scores = _score_map(source_trace), _score_map(public_trace)
    if set(source_scores) != set(public_scores):
        raise AssertionError("candidate sequence menus differ")
    for sequence in source_scores:
        if not math.isclose(source_scores[sequence], public_scores[sequence], rel_tol=0, abs_tol=1e-9):
            raise AssertionError(f"candidate score differs for {sequence}")
    for field in ("selected_h", "scenario_count"):
        if field in source_trace and field in public_trace and source_trace[field] != public_trace[field]:
            raise AssertionError(f"selector field {field} differs")

    sequence = tuple(int(value) for value in source_trace.get("committed_sequence", ()))
    cached = 0
    if check_group_commit and len(sequence) > 1:
        source_committer = getattr(source_policy, "_eg_committer", None)
        public_committer = getattr(public_policy, "_eg_committer", None)
        if source_committer is None or public_committer is None:
            raise AssertionError("multi-action EG proposal has no source/public group committer")
        if tuple(source_trace.get("committed_sequence", ())) != tuple(public_trace.get("committed_sequence", ())):
            raise AssertionError("source and public EG committed sequences differ")
        while source_committer.expected_full is not None or public_committer.expected_full is not None:
            if source_committer.expected_full is None or public_committer.expected_full is None:
                raise AssertionError("source/public EG group-cache lifetimes differ")
            source_next_frame = _frame_from_snapshot(source_frame, source_committer.expected_full)
            public_next_frame = _frame_from_snapshot(public_frame, public_committer.expected_full)
            if _signature(source_next_frame) != _signature(public_next_frame):
                raise AssertionError("source/public EG expected next public frame differs")
            source_next, _ = source_policy.select(source_next_frame)
            public_next, _ = public_policy.select(public_next_frame)
            if int(source_next) != int(public_next):
                raise AssertionError("source/public cached EG commit action differs")
            cached += 1
    return {"action": int(source_action), "candidate_sequences": len(source_scores),
            "cached_eg_actions_checked": cached, "committed_sequence_length": len(sequence)}


def _load_source_api(source_root: Path) -> dict[str, Any]:
    if not (source_root / "uw_saa" / "scrp").is_dir():
        raise ValueError("--source-root must contain the frozen uw_saa/scrp package")
    sys.path.insert(0, str(source_root.resolve()))
    from uw_saa.scrp.uw_saa_public import UWSAASelector
    from uw_saa.scrp.uw_saa_followup import FixedHorizonSelector, ScenarioControlSelector
    from uw_saa.scrp.uw_saa_plan063_multik import build_plan063_selector
    from uw_saa.scrp.uw_saa_deterministic_single import UWSAADeterministicSingleSelector
    from uw_saa.scrp.uw_saa_public import _public_state
    from uw_saa.scrp.r_order_public import PublicHistoryTracker as SourceHistoryTracker
    from uw_saa.scrp.r_order_public import attach_history_summary as source_attach_summary

    return {
        "UWSAASelector": UWSAASelector,
        "FixedHorizonSelector": FixedHorizonSelector,
        "ScenarioControlSelector": ScenarioControlSelector,
        "build_plan063_selector": build_plan063_selector,
        "UWSAADeterministicSingleSelector": UWSAADeterministicSingleSelector,
        "public_state": _public_state,
        "history_tracker": SourceHistoryTracker,
        "attach_history_summary": source_attach_summary,
    }


def _source_specific_frame(frame: object, source: Mapping[str, Any]) -> object:
    history = source["history_tracker"]()
    return source["attach_history_summary"](frame, history.summary(frame))


def _source_parity(episode: Any, source_frame: object, public_frame: object, source: Mapping[str, Any],
                   runtime: Mapping[str, Any], gp_rule: Mapping[str, Any],
                   best_h: Mapping[str, Any]) -> dict[str, Any]:
    from baseline.native import SOLVERS

    condition = CONDITION_SETS["main"][-1]
    public_state = source["public_state"](source_frame)
    unknown_tail = len(public_state[-1])
    if unknown_tail <= 0:
        raise AssertionError("selected parity frame does not contain an anonymous forecast tail")

    tested: dict[str, Any] = {}
    nontrivial_by_arm: dict[str, int] = {}
    for solver in (name for name in SOLVERS if name != "GP"):
        source_policy = source["UWSAASelector"](solver, "UW-SAA", runtime, gp_rule)
        public_policy = _make_policy(solver, "UWSAA", runtime, gp_rule)
        tested[f"main/UWSAA/{solver}"] = _select_and_compare(
            source_policy, public_policy, source_frame, public_frame,
            check_group_commit=(solver == "EG")
        )
        source_native = source["FixedHorizonSelector"](solver, 20, runtime, gp_rule)
        public_native = _make_policy(solver, "NATIVE20", runtime, gp_rule)
        tested[f"main/NATIVE20/{solver}"] = _select_and_compare(
            source_native, public_native, source_frame, public_frame, check_group_commit=False
        )

    control_mapping = {
        "UWSAA_NO_WINDOW_SHUFFLE_K8": "NO_WINDOW_SHUFFLE8",
        "UWSAA_SINGLE_WINDOW_B2": "SINGLE_B2_K8",
        "UWSAA_SINGLE_WINDOW_B5": "SINGLE_B5_K8",
        "UWSAA_SINGLE_WINDOW_B10": "SINGLE_B10_K8",
        "UWSAA_SINGLE_WINDOW_B20": "SINGLE_B20_K8",
    }
    control_arms = tuple(control_mapping)
    for solver in ("EM", "EG"):
        for public_arm, source_arm in control_mapping.items():
            source_policy = source["ScenarioControlSelector"](solver, source_arm, runtime, gp_rule)
            public_policy = make_variant(solver, public_arm, runtime, gp_rule)
            tested[f"ablation/{public_arm}/{solver}"] = _select_and_compare(
                source_policy, public_policy, source_frame, public_frame,
                check_group_commit=(solver == "EG")
            )
        horizon = int(best_h["selected_h_by_solver"][solver])
        source_policy = source["FixedHorizonSelector"](solver, horizon, runtime, gp_rule)
        public_policy = _make_policy(solver, "UWSAA_BEST_FIXED_H", runtime, gp_rule)
        tested[f"ablation/UWSAA_BEST_FIXED_H/{solver}"] = _select_and_compare(
            source_policy, public_policy, source_frame, public_frame, check_group_commit=False
        )
        source_policy = source["UWSAADeterministicSingleSelector"](solver, runtime, gp_rule)
        public_policy = make_variant(solver, "UWSAA_DETERMINISTIC_SINGLE", runtime, gp_rule)
        tested[f"p0/UWSAA_DETERMINISTIC_SINGLE/{solver}"] = _select_and_compare(
            source_policy, public_policy, source_frame, public_frame,
            check_group_commit=(solver == "EG")
        )

    multi_map = {
        "UWSAA_K1_ROLLOUT": "P1_K1_ROLLOUT", "UWSAA_K2_ROLLOUT": "P1_K2_ROLLOUT",
        "UWSAA_K4_ROLLOUT": "P1_K4_ROLLOUT", "UWSAA_K16_ROLLOUT": "P1_K16_ROLLOUT",
        "UWSAA_K1_LB1": "P4_K1_LB1", "UWSAA_K8_LB1": "P3_K8_LB1",
    }
    for solver in ("EM", "EG"):
        for public_arm, source_arm in multi_map.items():
            source_policy = source["build_plan063_selector"](source_arm, solver, runtime, gp_rule)
            public_policy = make_variant(solver, public_arm, runtime, gp_rule)
            tested[f"multik/{public_arm}/{solver}"] = _select_and_compare(
                source_policy, public_policy, source_frame, public_frame,
                check_group_commit=(solver == "EG")
            )
    for suite, arms in (
        ("ablation", control_arms),
        ("p0", ("UWSAA_DETERMINISTIC_SINGLE",)),
        ("multik", tuple(multi_map)),
    ):
        for arm in arms:
            counts = [row["candidate_sequences"] for key, row in tested.items()
                      if key.startswith(f"{suite}/{arm}/")]
            if not counts or max(counts) < 2:
                raise AssertionError(f"{suite}/{arm} has no nontrivial source/public candidate-score menu")
            nontrivial_by_arm[f"{suite}/{arm}"] = max(counts)
    eg_cache_actions = sum(row["cached_eg_actions_checked"] for row in tested.values())
    if eg_cache_actions <= 0:
        raise AssertionError("source/public parity did not exercise any cached EG group action")
    return {"policies_checked": len(tested), "unknown_tail_items": unknown_tail,
            "nontrivial_candidate_score_menus_by_arm": nontrivial_by_arm,
            "cached_eg_actions_checked": eg_cache_actions, "source_parity": tested}


def _forecast_parity(trace_path: Path) -> dict[str, Any]:
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 41:
        raise AssertionError(f"frozen main trace must have 41 registered rows, got {len(rows)}")
    first = rows[0]
    sequence = tuple(int(value) for value in first["true_retrieval_sequence_private"])
    stacks = tuple(tuple(int(value) for value in stack) for stack in first["initial_stacks_private"])
    episode = OnlineCRPEpisode(
        episode_id="frozen-layout-parity", num_stacks=len(stacks),
        max_tiers=int(first["max_tiers_private"]), initial_stacks=stacks,
        retrieval_sequence=sequence, parameter_group="source-parity", metadata={},
    )
    matched = set()
    for row in rows:
        source_condition = row["condition_evaluator_only"]
        condition = {
            "condition_id": str(source_condition["condition_id"]),
            "family": str(source_condition["family"]),
            "p": source_condition.get("p"),
            "window_size": source_condition.get("window_size"),
            "repeat": int(source_condition.get("repeat", 0)),
            "profile": source_condition.get("profile"),
        }
        table, _digest = forecast_table(episode, condition)
        frozen = tuple(tuple(int(value) for value in epoch) for epoch in row["forecast_table_private"])
        if table != frozen:
            raise AssertionError(f"forecast table differs for frozen condition {condition['condition_id']}")
        matched.add(condition["condition_id"])
    if len(matched) != 41:
        raise AssertionError("frozen trace does not cover 41 unique conditions")
    return {"forecast_conditions_checked": len(matched),
            "frozen_trace_sha256": _sha256(trace_path),
            "frozen_trace_basename": trace_path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="frozen source checkout containing uw_saa/scrp")
    parser.add_argument("--frozen-trace", type=Path,
                        help="optional 41-row archived private forecast trace for one layout")
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "uwsaa_source_differential_smoke.json")
    args = parser.parse_args()

    source = _load_source_api(args.source_root)
    runtime, gp_rule = load_runtime(), None
    best_h = json.loads((ROOT / "configs" / "best_fixed_h_by_solver.json").read_text(encoding="utf-8"))
    episodes = load_episode_manifest(DEFAULT_DATA)
    condition = CONDITION_SETS["main"][-1]
    from uw_saa.scrp.uw_saa_public import _candidate_sequences

    selected = None
    for episode in episodes:
        frame = _frame(episode, condition)
        if frame is None:
            continue
        source_frame = _source_specific_frame(frame, source)
        if len(source["public_state"](source_frame)[-1]) <= 0:
            continue
        menu = _candidate_sequences(source_frame, "EG", runtime, gp_rule)
        em_menu = _candidate_sequences(source_frame, "EM", runtime, gp_rule)
        if (len(set(menu.values())) > 1 and len(set(em_menu.values())) > 1
                and any(len(sequence) > 1 for sequence in menu.values())):
            selected = episode, source_frame, frame
            break
    if selected is None:
        raise AssertionError("no Caserta frame found with both an anonymous tail and a multi-action EG proposal")
    episode, source_frame, public_frame = selected
    source_files = (
        "uw_saa/scrp/uw_saa_public.py", "uw_saa/scrp/uw_saa_fast.py",
        "uw_saa/scrp/uw_saa_followup.py", "uw_saa/scrp/uw_saa_deterministic_single.py",
        "uw_saa/scrp/uw_saa_plan063_multik.py", "uw_saa/scrp/r_order_actor.py",
        "uw_saa/scrp/r_five_versions_actor.py", "uw_saa/scrp/r_eri_eg_public.py",
        "uw_saa/scrp/universal_r_utility_m1.py",
    )
    source_hashes = {name: _sha256(args.source_root / name) for name in source_files}
    public_files = (
        "checks/source_differential_smoke.py", "reproducibility/runner.py",
        "reproducibility/protocol.py", "uwsaa/variants.py", "uwsaa/fast.py",
    )
    public_hashes = {name: _sha256(ROOT / name) for name in public_files}
    result = {
        "schema": "uwsaa_source_differential_smoke_v1",
        "status": "PASS_NON_GP_PENDING_GP",
        "gp_model_status": "pending; GP excluded from parity comparison",
        "scope": "one detached Caserta frame; non-GP source/public action and score parity; no performance run",
        "episode_id": episode.episode_id,
        "layout_fingerprint": __import__("reproducibility.protocol", fromlist=["label_invariant_fingerprint"]).label_invariant_fingerprint(episode),
        "source_files_sha256": source_hashes,
        "public_files_sha256": public_hashes,
        "source_parity": _source_parity(
            episode, source_frame, public_frame, source, runtime, gp_rule, best_h
        ),
    }
    if args.frozen_trace:
        result["forecast_parity"] = _forecast_parity(args.frozen_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "policies_checked": result["source_parity"]["policies_checked"],
                      "cached_eg_actions_checked": result["source_parity"]["cached_eg_actions_checked"],
                      "forecast_conditions_checked": result.get("forecast_parity", {}).get("forecast_conditions_checked", 0),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
