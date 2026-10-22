"""Frozen public condition registry and evaluator-side forecast construction."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

WINDOWS = (2, 5, 10, 20)
SOLVERS = ("EM", "GP", "ERI", "EG", "Chain", "Minmax")
PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "C_EQ": {"CLEAN": .25, "time_window_shuffle": .25, "subset_shuffle": .25, "delayed_positions": .25},
    "C_DOM_CLEAN": {"CLEAN": .5, "time_window_shuffle": 1/6, "subset_shuffle": 1/6, "delayed_positions": 1/6},
    "C_DOM_TW": {"CLEAN": 1/6, "time_window_shuffle": .5, "subset_shuffle": 1/6, "delayed_positions": 1/6},
    "C_DOM_SS": {"CLEAN": 1/6, "time_window_shuffle": 1/6, "subset_shuffle": .5, "delayed_positions": 1/6},
    "C_DOM_DP": {"CLEAN": 1/6, "time_window_shuffle": 1/6, "subset_shuffle": 1/6, "delayed_positions": .5},
    "N_EQ": {"CLEAN": 0., "time_window_shuffle": 1/3, "subset_shuffle": 1/3, "delayed_positions": 1/3},
    "N_DOM_TW": {"CLEAN": 0., "time_window_shuffle": .5, "subset_shuffle": .25, "delayed_positions": .25},
    "N_DOM_SS": {"CLEAN": 0., "time_window_shuffle": .25, "subset_shuffle": .5, "delayed_positions": .25},
    "N_DOM_DP": {"CLEAN": 0., "time_window_shuffle": .25, "subset_shuffle": .25, "delayed_positions": .5},
}


def _condition(family: str, *, p: float | None = None, width: int | None = None,
               repeat: int = 0, profile: str | None = None) -> dict[str, Any]:
    bits = [family]
    if p is not None:
        bits.append(f"p={p:.1f}")
    if width is not None:
        bits.append(f"B={width}")
    if profile is not None:
        bits.append(f"profile={profile}")
    if family != "CLEAN":
        bits.append(f"repeat={repeat}")
    return {
        "condition_id": "|".join(bits), "family": family, "p": p,
        "window_size": width, "repeat": int(repeat), "profile": profile,
    }


def main_conditions() -> tuple[dict[str, Any], ...]:
    rows = []
    for p in (.1, .3, .5):
        for repeat in (0, 1):
            rows.append(_condition("subset_shuffle", p=p, repeat=repeat))
    for width in WINDOWS:
        for repeat in (0, 1):
            rows.append(_condition("time_window_shuffle", width=width, repeat=repeat))
    for p in (.1, .3, .5):
        for repeat in (0, 1):
            rows.append(_condition("delayed_positions", p=p, repeat=repeat))
    for repeat in (0, 1):
        rows.append(_condition("overlay_mix", p=.3, width=5, repeat=repeat))
    for profile in PROFILE_WEIGHTS:
        for repeat in (0, 1):
            rows.append(_condition("regime_mix", p=.3, width=5, repeat=repeat, profile=profile))
    rows.append(_condition("CLEAN"))
    if len(rows) != 41 or len({row["condition_id"] for row in rows}) != 41:
        raise AssertionError("registered UWSAA main condition count changed")
    return tuple(rows)


ABLATION_CONDITIONS = tuple(
    _condition(family, p=p, width=width, repeat=repeat)
    for family, p, width, repeat in (
        ("subset_shuffle", .3, None, 0),
        ("subset_shuffle", .3, None, 1),
        ("time_window_shuffle", None, 5, 0),
        ("time_window_shuffle", None, 5, 1),
        ("delayed_positions", .3, None, 0),
        ("delayed_positions", .3, None, 1),
        ("CLEAN", None, None, 0),
    )
)

CONDITION_SETS = {
    "main": main_conditions(),
    "ablation": ABLATION_CONDITIONS,
    "p0": ABLATION_CONDITIONS,
    "multik": ABLATION_CONDITIONS,
}
SUITE_ARMS = {
    "main": ("UWSAA", "NATIVE20"),
    "ablation": ("UWSAA", "NATIVE20", "UWSAA_NO_WINDOW_SHUFFLE_K8", "UWSAA_SINGLE_WINDOW_B2",
                 "UWSAA_SINGLE_WINDOW_B5", "UWSAA_SINGLE_WINDOW_B10", "UWSAA_SINGLE_WINDOW_B20",
                 "UWSAA_BEST_FIXED_H"),
    "p0": ("UWSAA", "UWSAA_DETERMINISTIC_SINGLE"),
    "multik": ("UWSAA", "UWSAA_K1_ROLLOUT", "UWSAA_K2_ROLLOUT",
               "UWSAA_K4_ROLLOUT", "UWSAA_K16_ROLLOUT", "UWSAA_K1_LB1", "UWSAA_K8_LB1"),
}


def label_invariant_fingerprint(episode: Any) -> str:
    rank_by_label = {label: rank for rank, label in enumerate(episode.retrieval_sequence, start=1)}
    rank_stacks = sorted(tuple(rank_by_label[label] for label in stack) for stack in episode.initial_stacks)
    payload = json.dumps({
        "num_stacks": int(episode.num_stacks), "max_tiers": int(episode.max_tiers),
        "rank_stacks": rank_stacks,
    }, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _family_seed(fingerprint: str, family: str, repeat: int) -> int:
    # This historical namespace is retained to reproduce registered forecasts.
    payload = f"history-reliability-r-v1|{fingerprint}|{family}|{int(repeat)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _time_window_seed(fingerprint: str, width: int, repeat: int) -> int:
    payload = {"family": "time_window_shuffle", "fingerprint": fingerprint,
               "protocol_id": "r_noise_diagnostics_v1", "repeat": int(repeat),
               "window_size": int(width)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _family_forecast(truth: Sequence[int], family: str, p: float, seed: int) -> tuple[int, ...]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    values = tuple(int(value) for value in truth)
    if family == "subset_shuffle":
        selected = np.flatnonzero(rng.random(len(values)) < p)
        forecast = list(values)
        if len(selected) > 1:
            permuted = rng.permutation(np.asarray([values[int(i)] for i in selected], dtype=np.int64))
            for index, value in zip(selected, permuted):
                forecast[int(index)] = int(value)
    elif family == "delayed_positions":
        delay = int(math.ceil(len(values) / 4.0))
        mask = rng.random(len(values)) < p
        order = sorted(range(len(values)), key=lambda i: (i + delay * int(mask[i]), i))
        forecast = [values[index] for index in order]
    else:
        raise ValueError(f"unknown forecast family {family!r}")
    result = tuple(forecast)
    if len(result) != len(values) or set(result) != set(values):
        raise ValueError("forecast does not permute the complete true future")
    return result


def _time_window_forecast(truth: Sequence[int], width: int, seed: int) -> tuple[int, ...]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    values = tuple(int(value) for value in truth)
    result = []
    for start in range(0, len(values), int(width)):
        chunk = values[start:start + int(width)]
        result.extend(chunk[int(index)] for index in rng.permutation(len(chunk)))
    return tuple(result)


def _build_regime_plan(fingerprint: str, repeat: int, profile: str, epoch_count: int) -> list[str]:
    # The fixed family-order namespace is historical input to the protocol.
    seed = int.from_bytes(hashlib.sha256(
        f"plan058_regime_order_v1|{fingerprint}|r={repeat}".encode("utf-8")
    ).digest()[:8], "little")
    families = ("CLEAN", "time_window_shuffle", "subset_shuffle", "delayed_positions")
    order = tuple(str(value) for value in np.random.Generator(np.random.PCG64(seed)).permutation(
        np.asarray(families, dtype=object)).tolist())
    weights = PROFILE_WEIGHTS[profile]
    active = tuple(family for family in families if weights[family] > 0)
    active_order = tuple(family for family in order if family in active)
    exact = {family: epoch_count * weights[family] for family in active}
    lengths = {family: int(math.floor(exact[family])) for family in active}
    leftover = epoch_count - sum(lengths.values())
    rank = {family: i for i, family in enumerate(active_order)}
    remainder = sorted(active, key=lambda family: (-(exact[family] - lengths[family]), rank[family]))
    for family in remainder[:leftover]:
        lengths[family] += 1
    assignment = []
    for family in active_order:
        assignment.extend([family] * lengths[family])
    if len(assignment) != epoch_count:
        raise AssertionError("regime schedule does not cover all forecast epochs")
    return assignment


def forecast_table(episode: Any, condition: Mapping[str, Any]) -> tuple[tuple[tuple[int, ...], ...], str]:
    sequence = tuple(int(value) for value in episode.retrieval_sequence)
    truth = sequence[1:]
    fingerprint = label_invariant_fingerprint(episode)
    family = str(condition["family"])
    repeat = int(condition.get("repeat", 0))
    p = float(condition.get("p") or 0.0)
    width = int(condition.get("window_size") or 5)
    if family == "CLEAN":
        full = truth
        table = tuple(tuple(value for value in full if value not in set(sequence[:k + 1]))
                      for k in range(len(sequence)))
    elif family in ("subset_shuffle", "delayed_positions"):
        full = _family_forecast(truth, family, p, _family_seed(fingerprint, family, repeat))
        table = tuple(tuple(value for value in full if value not in set(sequence[:k + 1]))
                      for k in range(len(sequence)))
    elif family == "time_window_shuffle":
        full = _time_window_forecast(truth, width, _time_window_seed(fingerprint, width, repeat))
        table = tuple(tuple(value for value in full if value not in set(sequence[:k + 1]))
                      for k in range(len(sequence)))
    elif family == "overlay_mix":
        tw = _time_window_forecast(truth, width, _time_window_seed(fingerprint, width, repeat))
        full = _family_forecast(tw, "subset_shuffle", p,
                                _family_seed(fingerprint, "subset_shuffle", repeat))
        table = tuple(tuple(value for value in full if value not in set(sequence[:k + 1]))
                      for k in range(len(sequence)))
    elif family == "regime_mix":
        profile = str(condition["profile"])
        channels = {
            "CLEAN": truth,
            "time_window_shuffle": _time_window_forecast(truth, width, _time_window_seed(fingerprint, width, repeat)),
            "subset_shuffle": _family_forecast(truth, "subset_shuffle", p, _family_seed(fingerprint, "subset_shuffle", repeat)),
            "delayed_positions": _family_forecast(truth, "delayed_positions", p, _family_seed(fingerprint, "delayed_positions", repeat)),
        }
        assignment = _build_regime_plan(fingerprint, repeat, profile, len(sequence) - 1)
        rows = []
        for k, chosen_family in enumerate(assignment):
            remaining = set(sequence[k + 1:])
            row = tuple(value for value in channels[chosen_family] if value in remaining)
            if len(row) != len(remaining) or set(row) != remaining:
                raise ValueError("regime forecast does not cover current remaining future")
            rows.append(row)
        rows.append(())
        table = tuple(rows)
    else:
        raise ValueError(f"unknown condition family {family!r}")
    if len(table) != len(sequence) or table[-1] != ():
        raise AssertionError("forecast table must contain one row per retrieval count")
    canonical = json.dumps([list(row) for row in table], separators=(",", ":"), ensure_ascii=True).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()
    return table, digest


__all__ = [
    "ABLATION_CONDITIONS", "CONDITION_SETS", "PROFILE_WEIGHTS", "SOLVERS", "SUITE_ARMS",
    "WINDOWS", "forecast_table", "label_invariant_fingerprint", "main_conditions",
]
