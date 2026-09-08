"""Averaging rules and the privacy invariant of the update payload."""

from __future__ import annotations

import pytest

from fedxgb import aggregator, weights
from fedxgb.bank_node import BankUpdate


def _update(bank_id: str, rows: int, trees: int = 1) -> BankUpdate:
    stub_tree = {
        "left_children": [-1],
        "split_conditions": [1.0],
        "base_weights": [1.0],
        "id": 0,
    }
    return BankUpdate(
        bank_id=bank_id,
        round_index=1,
        trees=[dict(stub_tree) for _ in range(trees)],
        n_local_rows=rows,
        n_local_positives=rows // 100,
        local_train_auc=0.9,
    )


def test_uniform_averaging_gives_every_bank_an_equal_share():
    updates = [_update("a", 100), _update("b", 900)]

    assert aggregator.averaging_factors(updates, weighted_by_volume=False) == [0.5, 0.5]


def test_volume_weighting_follows_local_row_counts():
    updates = [_update("a", 100), _update("b", 900)]

    factors = aggregator.averaging_factors(updates, weighted_by_volume=True)

    assert factors == pytest.approx([0.1, 0.9])


def test_averaging_factors_always_sum_to_one():
    updates = [_update("a", 1), _update("b", 2), _update("c", 3)]

    for weighted in (True, False):
        assert sum(aggregator.averaging_factors(updates, weighted)) == pytest.approx(1.0)


def test_empty_round_yields_no_factors():
    assert aggregator.averaging_factors([], weighted_by_volume=True) == []


def test_zero_row_participants_fall_back_to_uniform_weights():
    updates = [_update("a", 0), _update("b", 0)]

    assert aggregator.averaging_factors(updates, weighted_by_volume=True) == [0.5, 0.5]


def test_aggregate_round_appends_every_participants_trees():
    global_model = {
        "learner": {
            "gradient_booster": {
                "model": {
                    "trees": [],
                    "tree_info": [],
                    "iteration_indptr": [0],
                    "gbtree_model_param": {"num_trees": "0"},
                }
            }
        }
    }
    updates = [_update("a", 10, trees=2), _update("b", 10, trees=2)]

    merged = aggregator.aggregate_round(global_model, updates)

    assert weights.tree_count(merged) == 4


def test_aggregate_round_scales_leaves_by_the_bank_share():
    global_model = {
        "learner": {
            "gradient_booster": {
                "model": {
                    "trees": [],
                    "tree_info": [],
                    "iteration_indptr": [0],
                    "gbtree_model_param": {"num_trees": "0"},
                }
            }
        }
    }
    merged = aggregator.aggregate_round(global_model, [_update("a", 10), _update("b", 10)])

    leaves = [t["split_conditions"][0] for t in merged["learner"]["gradient_booster"]["model"]["trees"]]
    assert leaves == pytest.approx([0.5, 0.5])


def test_update_payload_carries_no_row_level_data():
    """A BankUpdate must expose counts and trees - never transactions."""
    update = _update("a", 10)

    shared = set(vars(update))
    assert shared == {
        "bank_id",
        "round_index",
        "trees",
        "n_local_rows",
        "n_local_positives",
        "local_train_auc",
        "payload_kb",
    }


def test_round_report_mentions_zero_rows_shared():
    report = aggregator.round_report(1, [_update("a", 10)])

    assert "0 rows shared" in report
