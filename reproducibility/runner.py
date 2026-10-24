"""Run a registered UWSAA benchmark subset or full trajectory evaluation.

The command defaults to listing the protocol. Use ``--run`` for evaluation or
``--smoke`` for one complete, small trajectory. Output rows are endpoints;
per-decision action traces are a separate local archive artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline import chain, minmax
from baseline.native import FixedHorizonActor, SOLVERS
from environment.noise_aware_forecast import ForecastProvider
from environment.online_crp import OnlineCRPEnvironment, load_episode_manifest
from environment.public_frame import make_public_frame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from reproducibility.protocol import CONDITION_SETS, SUITE_ARMS, forecast_table, label_invariant_fingerprint
from uwsaa.entrypoint import load_gp_rule, load_runtime, make_policy
from uwsaa.variants import make_variant

Q_PRIOR = 5183 / 9920
TRAJECTORY_CAP_SECONDS = 120.0
DEFAULT_DATA = ROOT / "data" / "caserta_table2_840.json"
DEFAULT_OUTPUT = ROOT / "reproducibility" / "results" / "trajectory_endpoints.jsonl"


class FixedHorizonPolicy:
    def __init__(self, solver: str, horizon: int, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any] | None):
        self.solver, self.horizon = solver, int(horizon)
        self.actor = FixedHorizonActor(solver, horizon, runtime, gp_rule) if solver in SOLVERS[:4] else None
        self.proposer = chain.propose if solver == "Chain" else minmax.propose

    def select(self, frame: object):
        if self.actor is not None:
            action, trace = self.actor.select(frame)
        else:
            action, trace = self.proposer(frame, self.horizon)
        return int(action), {**dict(trace), "policy": "UWSAA_BEST_FIXED_H", "requested_h": self.horizon}


def _make_policy(solver: str, arm: str, runtime: Mapping[str, Any], gp_rule: Mapping[str, Any] | None):
    if arm == "UWSAA":
        return make_policy(solver, "UWSAA", runtime, gp_rule)
    if arm == "NATIVE20":
        return make_policy(solver, "NATIVE20", runtime, gp_rule)
    if arm == "UWSAA_BEST_FIXED_H":
        selection = json.loads((ROOT / "configs" / "best_fixed_h_by_solver.json").read_text(encoding="utf-8"))
        if solver == "GP":
            if gp_rule is None or gp_rule.get("best_fixed_h") is None:
                raise RuntimeError("GP best-fixed horizon is pending; select it on the registered training bank")
            horizon = int(gp_rule["best_fixed_h"])
        else:
            horizon = int(selection["selected_h_by_solver"][solver])
        return FixedHorizonPolicy(solver, horizon, runtime, gp_rule)
    return make_variant(solver, arm, runtime, gp_rule)


def _sync_forecast_provider(provider: ForecastProvider, episode: Any, table: tuple[tuple[int, ...], ...],
                            retrieval_count: int) -> None:
    sequence = tuple(int(value) for value in episode.retrieval_sequence)
    event_target = min(int(retrieval_count), len(sequence) - 1)
    while len(provider._retrieved) < event_target:
        new_index = len(provider._retrieved) + 1
        provider.observe_reveal(sequence[new_index])
        current_index = len(provider._retrieved)
        provider._forecast_future = tuple(sequence[1:current_index + 1]) + tuple(table[current_index])


def run_episode(episode: Any, condition: Mapping[str, Any], solver: str, arm: str,
                runtime: Mapping[str, Any], gp_rule: Mapping[str, Any] | None) -> dict[str, Any]:
    started = time.monotonic()
    if solver == "GP" and gp_rule is None:
        raise RuntimeError("GP runner requires a preloaded trained rule; refusing implicit rule loading")
    gp_model = _gp_model_record(gp_rule) if solver == "GP" else None
    fingerprint = ""
    table_hash = ""
    env = None
    provider = None
    decisions = 0
    try:
        fingerprint = label_invariant_fingerprint(episode)
        table, table_hash = forecast_table(episode, condition)
        sequence = tuple(int(value) for value in episode.retrieval_sequence)
        provider = ForecastProvider(
            sequence,
            p=0.0,
            seed=0,
            q_prior=Q_PRIOR,
            max_horizon=20,
            forecast_future=table[0],
        )
        env = OnlineCRPEnvironment(episode, lookahead_horizon=0)
        policy = _make_policy(solver, arm, runtime, gp_rule)
        history = PublicHistoryTracker()
        _sync_forecast_provider(provider, episode, table, env.state.retrieval_count)
        while not env.state.terminated:
            if time.monotonic() - started > TRAJECTORY_CAP_SECONDS:
                raise TimeoutError(f"trajectory exceeded {TRAJECTORY_CAP_SECONDS}s")
            state = env.state
            legal = env.legal_destinations()
            observation = provider.public_observation(env._stacks, legal, max_tiers=episode.max_tiers)
            raw_frame = make_public_frame(observation, state, legal, max_tiers=episode.max_tiers)
            frame = attach_history_summary(raw_frame, history.summary(raw_frame))
            if not frame.legal_destinations:
                raise RuntimeError("nonterminal public frame has no legal destination")
            before_retrieval = int(state.retrieval_count)
            before_relocations = int(env.relocation_count)
            destination, _action_trace = policy.select(frame)
            if time.monotonic() - started > TRAJECTORY_CAP_SECONDS:
                raise TimeoutError(f"trajectory exceeded {TRAJECTORY_CAP_SECONDS}s")
            if int(destination) not in frame.legal_destinations:
                raise ValueError(f"policy returned illegal destination {destination}")
            env.step(int(destination))
            if int(env.relocation_count) != before_relocations + 1:
                raise RuntimeError("relocation conservation failed")
            decisions += 1
            _sync_forecast_provider(provider, episode, table, env.state.retrieval_count)
            expected_events = min(int(env.state.retrieval_count), int(episode.num_containers) - 1)
            if int(provider.event_count) != expected_events:
                raise RuntimeError("forecast provider reveal count differs from retrieved targets")
            if not env.state.terminated:
                next_legal = env.legal_destinations()
                next_observation = provider.public_observation(
                    env._stacks, next_legal, max_tiers=episode.max_tiers
                )
                next_raw_frame = make_public_frame(
                    next_observation, env.state, next_legal, max_tiers=episode.max_tiers
                )
                history.update(frame, int(destination), next_raw_frame)
            if decisions > int(episode.num_containers) * max(1, int(episode.max_tiers) - 1):
                raise RuntimeError("physical step bound exceeded")
        if int(env.state.retrieval_count) != int(episode.num_containers):
            raise RuntimeError("episode terminated before all containers were retrieved")
        if int(provider.event_count) != int(episode.num_containers) - 1:
            raise RuntimeError("forecast provider did not audit every reveal exactly once")
    except Exception as exc:
        return {
            "schema": "uwsaa_trajectory_endpoint_v1",
            "episode_id": episode.episode_id,
            "parameter_group": episode.parameter_group,
            "condition_id": condition["condition_id"],
            "family": condition["family"],
            "repeat": int(condition.get("repeat", 0)),
            "solver": solver,
            "arm": arm,
            "layout_fingerprint": fingerprint,
            "semantic_key": _semantic_key(episode.episode_id, fingerprint, condition, solver, arm),
            "complete": False,
            "decisions": decisions,
            "num_retrieved": None if env is None else int(env.state.retrieval_count),
            "provider_reveal_count": None if provider is None else int(provider.event_count),
            "relocations": 0 if env is None else int(env.relocation_count),
            "num_containers": int(episode.num_containers),
            "relocations_per_container": 0.0 if env is None else float(env.relocation_count / episode.num_containers),
            "forecast_table_sha256": table_hash,
            "gp_model": gp_model,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
    return {
        "schema": "uwsaa_trajectory_endpoint_v1",
        "episode_id": episode.episode_id,
        "parameter_group": episode.parameter_group,
        "condition_id": condition["condition_id"],
        "family": condition["family"],
        "repeat": int(condition.get("repeat", 0)),
        "layout_fingerprint": fingerprint,
        "semantic_key": _semantic_key(episode.episode_id, fingerprint, condition, solver, arm),
        "solver": solver,
        "arm": arm,
        "complete": bool(env.state.terminated),
        "decisions": decisions,
        "num_retrieved": int(env.state.retrieval_count),
        "provider_reveal_count": int(provider.event_count),
        "relocations": int(env.relocation_count),
        "num_containers": int(episode.num_containers),
        "relocations_per_container": float(env.relocation_count / episode.num_containers),
        "forecast_table_sha256": table_hash,
        "gp_model": gp_model,
    }


def _gp_model_record(gp_rule: Mapping[str, Any] | None) -> dict[str, Any]:
    if gp_rule is None:
        raise RuntimeError("GP rule is pending; no model identity is available")
    identity = gp_rule.get("identity", gp_rule.get("model_identity"))
    if not isinstance(identity, Mapping):
        raise ValueError("loaded GP rule lacks model identity")
    fields = (
        "source_manifest_sha256", "protocol_sha256", "train_bank_sha256",
        "implementation_sha256", "seed_receipt_sha256",
    )
    record = {field: gp_rule.get(field, identity.get(field)) for field in fields}
    if any(not isinstance(value, str) or len(value) != 64 for value in record.values()):
        raise ValueError("loaded GP rule has incomplete source/protocol/training SHA bindings")
    record.update({
        "status": "trained",
        "formula_sha256": gp_rule.get("formula_sha256"),
        "best_fixed_h": gp_rule.get("best_fixed_h"),
        "model_identity": dict(identity),
    })
    if not isinstance(record["formula_sha256"], str) or len(record["formula_sha256"]) != 64:
        raise ValueError("loaded GP rule has no verified formula SHA-256")
    return record


def _semantic_key(episode_id: str, fingerprint: str, condition: Mapping[str, Any], solver: str, arm: str) -> str:
    p = condition.get("p")
    width = condition.get("window_size")
    profile = condition.get("profile")
    return (
        f"{episode_id}|{fingerprint}|{condition['family']}|p={'' if p is None else format(float(p), '.1f')}"
        f"|window_size={'' if width is None else int(width)}|profile={profile or ''}"
        f"|repeat={int(condition.get('repeat', 0))}|solver={solver}|arm={arm}"
    )


def _selected(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values) if values else default
    if len(set(result)) != len(result):
        raise ValueError("duplicate selection values are not allowed")
    return result


def _print_registry(suite: str) -> None:
    conditions = CONDITION_SETS[suite]
    payload = {
        "suite": suite,
        "condition_count": len(conditions),
        "conditions": [row["condition_id"] for row in conditions],
        "solvers": list(SOLVERS),
        "arms": list(SUITE_ARMS[suite]),
        "scope": "trajectory-endpoint runner; no performance experiment is executed by --list",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(CONDITION_SETS), default="main")
    parser.add_argument("--list", action="store_true", help="print registered conditions, solvers, and arms")
    parser.add_argument("--run", action="store_true", help="execute the requested trajectory evaluation")
    parser.add_argument("--smoke", action="store_true", help="execute one complete trajectory only")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--condition", action="append", help="registered condition id; repeat to select several")
    parser.add_argument("--solver", action="append", choices=SOLVERS, help="solver; repeat to select several")
    parser.add_argument("--arm", action="append", help="registered policy arm; repeat to select several")
    parser.add_argument("--episode-id", action="append", help="episode identifier; repeat to select several")
    parser.add_argument("--limit", type=int, help="limit the selected dataset rows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.list or not (args.run or args.smoke):
        _print_registry(args.suite)
        return
    if args.run and args.smoke:
        parser.error("choose --run or --smoke, not both")
    conditions = CONDITION_SETS[args.suite]
    arms = _selected(args.arm, SUITE_ARMS[args.suite])
    solvers = _selected(args.solver, SOLVERS)
    if any(arm not in SUITE_ARMS[args.suite] for arm in arms):
        parser.error(f"one or more arms are not registered for suite {args.suite}")
    if args.condition:
        by_id = {row["condition_id"]: row for row in conditions}
        missing = [condition for condition in args.condition if condition not in by_id]
        if missing:
            parser.error(f"unknown condition id(s): {missing}")
        conditions = tuple(by_id[key] for key in args.condition)
    if args.smoke:
        if not args.condition:
            clean = tuple(row for row in conditions if row["family"] == "CLEAN")
            conditions = clean[:1] if clean else conditions[:1]
        if not conditions:
            parser.error("smoke selection produced no registered condition")
        solvers = (args.solver[0] if args.solver else "EM",)
        arms = (args.arm[0] if args.arm else "UWSAA",)
        if args.limit is None:
            args.limit = 1
    episodes = load_episode_manifest(args.data)
    if args.episode_id:
        wanted = set(args.episode_id)
        episodes = tuple(episode for episode in episodes if episode.episode_id in wanted)
        absent = wanted - {episode.episode_id for episode in episodes}
        if absent:
            parser.error(f"unknown episode identifier(s): {sorted(absent)}")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        episodes = episodes[:args.limit]
    if not episodes:
        parser.error("no episodes selected")
    if not conditions:
        parser.error("no conditions selected")

    runtime = load_runtime()
    gp_rule = load_gp_rule() if "GP" in solvers else None
    gp_model = _gp_model_record(gp_rule) if "GP" in solvers else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error(f"output or run manifest already exists; choose new paths: {args.output}")
    rows = 0
    failures = 0
    try:
        stream = args.output.open("x", encoding="utf-8", newline="\n")
    except FileExistsError:
        parser.error(f"output already exists; choose a new path: {args.output}")
    with stream:
        for condition in conditions:
            for solver in solvers:
                for arm in arms:
                    for episode in episodes:
                        result = run_episode(episode, condition, solver, arm, runtime, gp_rule)
                        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                        rows += 1
                        failures += int(not result["complete"])
    try:
        data_file = args.data.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        data_file = "user-provided external dataset"
    data_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    manifest = {
        "schema": "uwsaa_trajectory_run_manifest_v1",
        "status": "failed" if failures else "complete",
        "smoke_only": bool(args.smoke),
        "suite": args.suite,
        "conditions": [dict(row) for row in conditions],
        "solvers": list(solvers),
        "arms": list(arms),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "layout_fingerprint": label_invariant_fingerprint(episode),
                "parameter_group": episode.parameter_group,
                "num_containers": int(episode.num_containers),
            }
            for episode in episodes
        ],
        "dataset_file": data_file,
        "dataset_sha256": data_digest,
        "expected_rows": len(episodes) * len(conditions) * len(solvers) * len(arms),
        "actual_rows": rows,
        "failed_rows": failures,
        "endpoint_file": args.output.name,
        "gp_model": gp_model,
        "pairing_key": "episode_id + label_invariant_fingerprint + condition parameters + solver; policy arms share paired rows",
        "scope": "endpoint records; no action-level traces and no timing or performance report",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "SMOKE_ONLY" if args.smoke else "EXECUTED",
        "suite": args.suite,
        "trajectory_rows": rows,
        "failures": failures,
        "conditions": [row["condition_id"] for row in conditions],
        "solvers": list(solvers),
        "arms": list(arms),
        "episode_count": len(episodes),
        "output": str(args.output),
        "manifest": str(manifest_path),
        "scope": "endpoint records only; not a timing or performance report",
    }, indent=2, ensure_ascii=False))
    if rows == 0:
        raise SystemExit("runner produced no trajectory endpoint rows")
    if rows != manifest["expected_rows"]:
        raise SystemExit("runner row count differs from the selected matrix")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
