"""End-to-end: does the federation actually do what the README claims?"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xgboost as xgb

from fedxgb import config, data_gen, server, weights
from fedxgb.bank_node import BankNode, roc_auc

_TINY_ROWS = 12_000
_TINY_PARAMS = {**config.XGB_PARAMS, "max_depth": 3, "nthread": 2}


@pytest.fixture(scope="module")
def ledgers(tmp_path_factory) -> dict[str, str]:
    """Small ledgers with the real generator, so tests exercise real data."""
    out = tmp_path_factory.mktemp("ledgers")
    profiles = [
        config.BankProfile(bank_id, config.BANK_TYPOLOGIES[bank_id], _TINY_ROWS, seed)
        for seed, bank_id in enumerate(config.BANK_IDS, start=1)
    ]
    paths = {p.bank_id: data_gen.generate_bank_file(p, str(out)) for p in profiles}
    return {**paths, "holdout": data_gen.generate_holdout_file(str(out), rows=_TINY_ROWS)}


@pytest.fixture(scope="module")
def nodes(ledgers) -> list[BankNode]:
    loaded = [BankNode(b, ledgers[b], _TINY_PARAMS) for b in config.BANK_IDS]
    for node in loaded:
        node.load_local_data()
    return loaded


@pytest.fixture(scope="module")
def holdout(ledgers) -> server.HoldoutSet:
    return server.HoldoutSet(ledgers["holdout"])


def test_bootstrap_global_model_starts_with_no_trees():
    assert weights.tree_count(server.bootstrap_global_model(_TINY_PARAMS)) == 0


def test_a_node_refuses_to_train_before_its_data_is_loaded(ledgers):
    node = BankNode("unloaded", ledgers["northwind_bank"], _TINY_PARAMS)

    with pytest.raises(RuntimeError, match="load_local_data"):
        node.train_round(server.bootstrap_global_model(_TINY_PARAMS), round_index=1)


def test_a_node_returns_only_the_trees_it_grew(nodes):
    """The delta, not the whole global model - this is the privacy boundary."""
    # Arrange: a global model that already holds trees from an earlier round
    global_model = server.bootstrap_global_model(_TINY_PARAMS)
    first = nodes[0].train_round(global_model, round_index=1, num_trees=3)
    global_model = weights.append_trees(global_model, first.trees)

    # Act
    second = nodes[0].train_round(global_model, round_index=2, num_trees=2)

    # Assert
    assert len(first.trees) == 3
    assert len(second.trees) == 2
    assert weights.tree_count(global_model) == 3


def test_a_federation_of_zero_rounds_does_nothing(nodes, holdout):
    assert server.run_federation(nodes, holdout, num_rounds=0, verbose=False) == []


def test_each_round_adds_one_contribution_per_bank(nodes, holdout):
    history = server.run_federation(nodes, holdout, num_rounds=3, verbose=False)

    expected = len(nodes) * config.TREES_PER_BANK_PER_ROUND
    assert [r.total_trees for r in history] == [expected, expected * 2, expected * 3]


def test_holdout_auc_of_an_empty_model_is_a_coin_flip(holdout):
    assert holdout.auc(server.bootstrap_global_model(_TINY_PARAMS)) == 0.5


def test_the_federated_model_beats_every_bank_training_alone(nodes, holdout):
    """The demo's central claim, asserted rather than asserted-in-prose."""
    # Arrange
    rounds = 6
    tree_budget = rounds * config.TREES_PER_BANK_PER_ROUND * len(nodes)

    # Act
    federated = server.run_federation(nodes, holdout, num_rounds=rounds, verbose=False)[-1]
    solo_aucs = [holdout.auc(server.train_local_only(node, tree_budget)) for node in nodes]

    # Assert
    assert federated.holdout_auc > max(solo_aucs)


def test_the_federated_model_detects_every_typology(nodes, holdout):
    """A solo bank is blind to typologies it has never seen; the federation is not."""
    federated = server.run_federation(nodes, holdout, num_rounds=6, verbose=False)[-1]

    recalls = holdout.detection(federated.global_model)

    assert len(recalls) == len(config.BANK_TYPOLOGIES)
    assert all(r.detected > 0 for r in recalls)


def test_volume_weighted_aggregation_also_produces_a_usable_model(nodes, holdout):
    history = server.run_federation(
        nodes, holdout, num_rounds=3, weighted_by_volume=True, verbose=False
    )

    assert history[-1].holdout_auc > 0.6


def test_pooled_reference_is_an_upper_bound_on_the_federation(nodes, holdout):
    """Federation should approach, not beat, what raw data pooling achieves."""
    rounds = 6
    tree_budget = rounds * config.TREES_PER_BANK_PER_ROUND * len(nodes)

    federated = server.run_federation(nodes, holdout, num_rounds=rounds, verbose=False)[-1]
    pooled_auc = holdout.auc(server.train_pooled_reference(nodes, tree_budget))

    assert federated.holdout_auc == pytest.approx(pooled_auc, abs=0.08)


def test_the_saved_global_model_loads_as_a_standard_xgboost_booster(nodes, holdout, tmp_path):
    # Arrange
    final = server.run_federation(nodes, holdout, num_rounds=2, verbose=False)[-1]

    # Act
    path = server.save_global_model(final.global_model, str(tmp_path))
    with open(path, encoding="utf-8") as handle:
        reloaded = xgb.Booster(model_file=bytearray(handle.read(), "utf-8"))

    # Assert
    assert reloaded.num_features() == len(config.FEATURE_COLUMNS)
    np.testing.assert_allclose(
        reloaded.predict(holdout.dmatrix), holdout.scores(final.global_model)
    )


def test_generated_ledgers_carry_only_their_own_banks_typology(ledgers):
    import polars as pl

    for bank_id in config.BANK_IDS:
        frame = pl.read_parquet(ledgers[bank_id]).filter(pl.col(config.LABEL_COLUMN) == 1)
        assert set(frame[config.TYPOLOGY_COLUMN].unique()) == {config.BANK_TYPOLOGIES[bank_id]}


def test_the_holdout_contains_every_typology(ledgers):
    import polars as pl

    frame = pl.read_parquet(ledgers["holdout"]).filter(pl.col(config.LABEL_COLUMN) == 1)

    assert set(frame[config.TYPOLOGY_COLUMN].unique()) == set(config.BANK_TYPOLOGIES.values())


def test_roc_auc_is_one_for_a_perfect_ranking():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_roc_auc_is_half_when_every_score_ties():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_is_undefined_for_a_single_class():
    assert np.isnan(roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])))
