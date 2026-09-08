"""Tree-weight surgery must be exact - the whole federation rests on it."""

from __future__ import annotations

import math

import numpy as np
import pytest
import xgboost as xgb

from fedxgb import weights

PARAMS = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "base_score": 0.5}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _booster(rng: np.random.Generator, rounds: int, rule) -> xgb.Booster:
    features = rng.normal(size=(400, 4))
    labels = rule(features).astype(int)
    return xgb.train(PARAMS, xgb.DMatrix(features, label=labels), num_boost_round=rounds)


def test_extract_trees_returns_only_the_new_ones(rng):
    # Arrange
    booster = _booster(rng, 5, lambda x: x[:, 0] > 0)
    model = weights.booster_to_dict(booster)

    # Act
    delta = weights.extract_trees(model, start=3)

    # Assert
    assert len(delta) == 2
    assert weights.tree_count(model) == 5


def test_extract_trees_rejects_negative_start(rng):
    model = weights.booster_to_dict(_booster(rng, 2, lambda x: x[:, 0] > 0))
    with pytest.raises(ValueError):
        weights.extract_trees(model, start=-1)


def test_scale_leaf_weights_leaves_split_thresholds_untouched(rng):
    # Arrange
    model = weights.booster_to_dict(_booster(rng, 1, lambda x: x[:, 0] > 0))
    tree = model["learner"]["gradient_booster"]["model"]["trees"][0]
    internal = [i for i, c in enumerate(tree["left_children"]) if c != -1]

    # Act
    scaled = weights.scale_leaf_weights(tree, 0.5)

    # Assert
    for node in internal:
        assert scaled["split_conditions"][node] == tree["split_conditions"][node]
    leaves = [i for i, c in enumerate(tree["left_children"]) if c == -1]
    for node in leaves:
        assert scaled["split_conditions"][node] == pytest.approx(
            tree["split_conditions"][node] * 0.5
        )


def test_scale_leaf_weights_does_not_mutate_input(rng):
    model = weights.booster_to_dict(_booster(rng, 1, lambda x: x[:, 0] > 0))
    tree = model["learner"]["gradient_booster"]["model"]["trees"][0]
    before = list(tree["split_conditions"])

    weights.scale_leaf_weights(tree, 0.25)

    assert tree["split_conditions"] == before


def test_merged_model_equals_the_average_of_its_contributors(rng):
    """The core correctness claim: appending scaled trees averages margins."""
    # Arrange
    left = _booster(rng, 3, lambda x: x[:, 0] + x[:, 1] > 0)
    right = _booster(rng, 3, lambda x: x[:, 0] - x[:, 2] > 0)
    probe = xgb.DMatrix(rng.normal(size=(300, 4)))
    base_margin = math.log(0.5 / 0.5)

    # Act
    merged = weights.booster_to_dict(left)
    merged = weights.with_trees(merged, [])
    for booster in (left, right):
        trees = weights.extract_trees(weights.booster_to_dict(booster), start=0)
        merged = weights.append_trees(
            merged, [weights.scale_leaf_weights(t, 0.5) for t in trees]
        )
    actual = weights.dict_to_booster(merged).predict(probe, output_margin=True)

    # Assert
    expected = base_margin + 0.5 * (
        left.predict(probe, output_margin=True) - base_margin
    ) + 0.5 * (right.predict(probe, output_margin=True) - base_margin)
    assert np.abs(actual - expected).max() < 1e-5


def test_with_trees_renumbers_ids_and_iteration_index(rng):
    model = weights.booster_to_dict(_booster(rng, 4, lambda x: x[:, 0] > 0))
    trees = weights.extract_trees(model, start=0)

    rebuilt = weights.with_trees(model, trees + trees)
    section = rebuilt["learner"]["gradient_booster"]["model"]

    assert [t["id"] for t in section["trees"]] == list(range(8))
    assert section["gbtree_model_param"]["num_trees"] == "8"
    assert section["iteration_indptr"] == list(range(9))


def test_empty_like_produces_a_loadable_zero_tree_model(rng):
    model = weights.booster_to_dict(_booster(rng, 3, lambda x: x[:, 0] > 0))

    empty = weights.empty_like(model)

    assert weights.tree_count(weights.booster_to_dict(weights.dict_to_booster(empty))) == 0


def test_payload_size_is_reported_in_kilobytes(rng):
    trees = weights.extract_trees(weights.booster_to_dict(_booster(rng, 2, lambda x: x[:, 0] > 0)), 0)
    assert weights.payload_size_kb(trees) > 0
