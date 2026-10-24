from __future__ import annotations

import numpy as np
import pytest

from reproducibility.statistics import _bootstrap_indices as registered_bootstrap_indices
from reproducibility.gp_arm_paired_analysis import (
    ARMS,
    DEFAULT_DATA,
    actual_episode_index,
    bootstrap_draws,
    macro_effect,
    paired_indices,
    validate_endpoint_row,
    validate_pairs,
)


MODEL = {"formula_sha256": "f" * 64}


def _effect_row(geometry, fingerprint, condition, arm, value):
    return {
        "_geometry": geometry,
        "_fingerprint": fingerprint,
        "condition_id": condition,
        "arm": arm,
        "relocations": value,
    }


def test_macro_order_and_reduction_sign_use_equal_geometry_then_equal_families():
    rows = []
    # Geometry 1 has one layout and geometry 2 has three, so this distinguishes
    # equal geometry weighting from a pooled-layout mean.
    for geo, fps, baseline, candidates in (
        ("g1", ["a"], (100, 300), ((90, 330),)),
        ("g2", ["b", "c", "d"], (1000, 1000), ((900, 800),) * 3),
    ):
        for fp, candidate in zip(fps, candidates, strict=True):
            for condition, base, cand in zip(("ss1", "ss2"), baseline, candidate, strict=True):
                rows.append(_effect_row(geo, fp, condition, "NATIVE20", base))
                rows.append(_effect_row(geo, fp, condition, "UWSAA", cand))
            rows.extend((
                _effect_row(geo, fp, "tw1", "NATIVE20", 50),
                _effect_row(geo, fp, "tw1", "UWSAA", 25),
            ))

    cells = {}
    layouts = {"g1": ["a"], "g2": ["b", "c", "d"]}
    for row in rows:
        key = (row["_geometry"], row["_fingerprint"], row["condition_id"], row["arm"])
        cells[key] = row["relocations"]

    ss_base, ss_candidate, ss_reduction = macro_effect(
        cells, layouts, ("g1", "g2"), ("ss1", "ss2"), "relocations"
    )
    assert ss_base == pytest.approx(600.0)
    assert ss_candidate == pytest.approx(530.0)
    assert ss_reduction == pytest.approx(7 / 60)  # positive means UWSAA reduced relocations

    _, _, tw_reduction = macro_effect(cells, layouts, ("g1", "g2"), ("tw1",), "relocations")
    assert tw_reduction == pytest.approx(0.5)
    # The registered primary averages family-relative effects equally, not
    # conditions or raw costs across SS and TW.
    assert (ss_reduction + tw_reduction) / 2 == pytest.approx(37 / 120)


def test_bootstrap_reuses_identical_within_geometry_indices_for_both_arms():
    geometries = ("g1", "g2")
    layouts = {"g1": ["a", "b"], "g2": ["c", "d"]}
    cells = {}
    base_values = {"a": 10, "b": 100, "c": 20, "d": 200}
    for geo, fps in layouts.items():
        for fp in fps:
            for condition in ("ss1", "ss2"):
                base = base_values[fp] + (5 if condition == "ss2" else 0)
                for arm, value in (("NATIVE20", base), ("UWSAA", base - 2)):
                    cells[(geo, fp, condition, arm)] = value

    indices = paired_indices(layouts, geometries, draws=8, seed=202610195903)
    replay = paired_indices(layouts, geometries, draws=8, seed=202610195903)
    for geometry in geometries:
        np.testing.assert_array_equal(indices[geometry], replay[geometry])

    base_draws, candidate_draws = bootstrap_draws(cells, layouts, geometries, ("ss1", "ss2"), indices)
    expected_base_geos, expected_candidate_geos = [], []
    for geometry in geometries:
        fps = layouts[geometry]
        sampled = [fps[int(i)] for i in indices[geometry][0]]
        expected_base_geos.append(np.mean([
            np.mean([cells[(geometry, fp, c, "NATIVE20")] for c in ("ss1", "ss2")])
            for fp in sampled
        ]))
        expected_candidate_geos.append(np.mean([
            np.mean([cells[(geometry, fp, c, "UWSAA")] for c in ("ss1", "ss2")])
            for fp in sampled
        ]))
    assert base_draws[0] == pytest.approx(np.mean(expected_base_geos))
    assert candidate_draws[0] == pytest.approx(np.mean(expected_candidate_geos))


def _paired_rows():
    fingerprint = "a" * 64
    episode = {"e1": {"episode_id": "e1", "layout_fingerprint": fingerprint,
                      "geometry": "g1", "num_containers": 4}}
    common = {
        "condition_id": "CLEAN", "episode_id": "e1", "layout_fingerprint": fingerprint,
        "forecast_table_sha256": "b" * 64, "num_containers": 4,
        "parameter_group": "g1", "family": "CLEAN", "repeat": 0, "gp_model": MODEL,
    }
    rows = [{**common, "arm": arm} for arm in ARMS]
    return rows, episode


def test_pair_validator_rejects_missing_arm_mixed_model_duplicate_and_forecast_mismatch():
    rows, episodes = _paired_rows()
    validate_pairs(rows, {"CLEAN"}, episodes, MODEL)

    with pytest.raises(ValueError, match="incomplete endpoint matrix"):
        validate_pairs(rows[:1], {"CLEAN"}, episodes, MODEL)

    mixed = [dict(row) for row in rows]
    mixed[1]["gp_model"] = {"formula_sha256": "0" * 64}
    with pytest.raises(ValueError, match="mixes GP model identities"):
        validate_pairs(mixed, {"CLEAN"}, episodes, MODEL)

    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate endpoint pairing key"):
        validate_pairs(duplicate, {"CLEAN"}, episodes, MODEL)

    mismatched_forecast = [dict(row) for row in rows]
    mismatched_forecast[1]["forecast_table_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="disagree on forecast_table_sha256"):
        validate_pairs(mismatched_forecast, {"CLEAN"}, episodes, MODEL)

def test_bootstrap_indices_match_registered_pcq64_int32_sequence():
    layouts = {"g1": ["a", "b"], "g2": ["c", "d", "e"]}
    rows = [
        {"geometry": geometry, "layout_fingerprint": fingerprint}
        for geometry, fingerprints in layouts.items()
        for fingerprint in fingerprints
    ]
    ours = paired_indices(layouts, ("g1", "g2"), draws=8, seed=202610195903)
    registered = registered_bootstrap_indices(rows, 202610195903, 8)
    for geometry in layouts:
        np.testing.assert_array_equal(ours[geometry], registered[geometry])
        assert ours[geometry].dtype == np.int32


def test_manifest_episodes_are_reconstructed_from_current_caserta_bytes():
    actual = actual_episode_index(DEFAULT_DATA)
    assert len(actual) == 840
    assert len({row["layout_fingerprint"] for row in actual.values()}) == 840
    assert len({row["geometry"] for row in actual.values()}) == 21


def test_complete_endpoint_rejects_failure_fields_and_inconsistent_conservation():
    episode = {"episode_id": "e1", "layout_fingerprint": "a" * 64,
               "geometry": "g1", "num_containers": 4}
    condition = {"condition_id": "CLEAN", "family": "CLEAN", "repeat": 0}
    row = {
        "complete": True, "gp_model": MODEL, "layout_fingerprint": "a" * 64,
        "parameter_group": "g1", "num_containers": 4, "family": "CLEAN", "repeat": 0,
        "forecast_table_sha256": "b" * 64, "relocations": 2,
        "relocations_per_container": 0.5, "decisions": 2,
        "num_retrieved": 4, "provider_reveal_count": 3,
    }
    validate_endpoint_row(row, episode, condition, MODEL)

    with pytest.raises(ValueError, match="failure metadata"):
        validate_endpoint_row({**row, "failure_type": None}, episode, condition, MODEL)
    with pytest.raises(ValueError, match="decision/retrieval/reveal conservation"):
        validate_endpoint_row({**row, "decisions": 1}, episode, condition, MODEL)
    with pytest.raises(ValueError, match="decision/retrieval/reveal conservation"):
        validate_endpoint_row({**row, "num_retrieved": 3}, episode, condition, MODEL)
    with pytest.raises(ValueError, match="decision/retrieval/reveal conservation"):
        validate_endpoint_row({**row, "provider_reveal_count": 2}, episode, condition, MODEL)
