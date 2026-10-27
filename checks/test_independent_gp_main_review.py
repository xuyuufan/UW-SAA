"""Hand-calculated checks of the independent review's numerical primitives."""
from collections import Counter

import numpy as np
import pytest

from checks.independent_gp_main_review import (
    condition_design, fingerprint, linear_percentile, multiplicities,
)


def test_duplicate_layout_draws_become_paired_frequency_weights():
    indices = np.array([[0,0,2,3],[1,1,1,1]],dtype=np.int32)
    frequencies = multiplicities(indices,4)
    np.testing.assert_array_equal(frequencies,[[2,0,1,1],[0,4,0,0]])
    paired_costs = np.array([[10,8],[20,16],[30,24],[40,32]],dtype=float)
    # First draw is (10 + 10 + 30 + 40)/4 and (8 + 8 + 24 + 32)/4.
    np.testing.assert_allclose(frequencies @ paired_costs/4,[[22.5,18],[20,16]])


def test_percentiles_use_explicit_linear_interpolation():
    data = np.array([8.0,1.0,4.0,2.0])
    assert linear_percentile(data,.025) == pytest.approx(1.075)
    assert linear_percentile(data,.975) == pytest.approx(7.7)
    assert linear_percentile(np.array([0.0,10.0,20.0]),.05/6) == pytest.approx(1/6)
    np.testing.assert_allclose(linear_percentile(np.column_stack([data,data*2]),.5),[3,6])


def test_fingerprint_ignores_stack_order_and_label_names_but_preserves_stack_ordering():
    first = {"num_stacks":2,"max_tiers":3,"initial_stacks":[[1,3],[2]],"retrieval_sequence":[1,2,3]}
    relabeled = {"num_stacks":2,"max_tiers":3,"initial_stacks":[[20],[10,30]],"retrieval_sequence":[10,20,30]}
    changed = {**first,"initial_stacks":[[3,1],[2]]}
    assert fingerprint(first) == fingerprint(relabeled)
    assert fingerprint(first) != fingerprint(changed)
    with pytest.raises(ValueError,match="retrieval permutation"):
        fingerprint({**first,"initial_stacks":[[1,1],[2]]})


def test_independent_condition_design_has_registered_family_and_repeat_coverage():
    rows = condition_design()
    assert Counter(row["family"] for row in rows) == {
        "subset_shuffle":6,"time_window_shuffle":8,"delayed_positions":6,
        "overlay_mix":2,"regime_mix":18,"CLEAN":1,
    }
    assert len({row["condition_id"] for row in rows}) == 41
    assert all(row["repeat"] == 0 for row in rows if row["family"] == "CLEAN")
    for family in ("subset_shuffle","time_window_shuffle","delayed_positions"):
        assert {row["repeat"] for row in rows if row["family"] == family} == {0,1}
