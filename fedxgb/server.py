"""Round orchestration for the federated transaction-monitoring programme."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import xgboost as xgb

from fedxgb import aggregator, config, evaluation, features, weights
from fedxgb.bank_node import BankNode, roc_auc

_SCHEMA_PROBE_ROWS = 2


def bootstrap_global_model(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create the empty round-zero global model from the agreed schema only."""
    probe = np.vstack(
        [np.zeros(len(config.FEATURE_COLUMNS)), np.ones(len(config.FEATURE_COLUMNS))]
    )
    matrix = xgb.DMatrix(
        probe, label=np.arange(_SCHEMA_PROBE_ROWS), feature_names=list(config.FEATURE_COLUMNS)
    )
    seed = xgb.train(dict(params or config.XGB_PARAMS), matrix, num_boost_round=0)
    return weights.empty_like(weights.booster_to_dict(seed))


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    global_model: dict[str, Any]
    holdout_auc: float
    total_trees: int
    payload_kb: float


class HoldoutSet:
    """The central service's evaluation data - all typologies, no bank owns it."""

    def __init__(self, path: str):
        frame = features.collect_streaming(
            features.build_feature_plan(path, extra_columns=(config.TYPOLOGY_COLUMN,))
        )
        self.labels = frame[config.LABEL_COLUMN].to_numpy()
        self.typologies = frame[config.TYPOLOGY_COLUMN].to_numpy()
        self.dmatrix = xgb.DMatrix(
            frame.select(config.FEATURE_COLUMNS).to_numpy(),
            label=self.labels,
            feature_names=list(config.FEATURE_COLUMNS),
        )

    def scores(self, model: dict[str, Any]) -> np.ndarray:
        return weights.dict_to_booster(model).predict(self.dmatrix)

    def auc(self, model: dict[str, Any]) -> float:
        if weights.tree_count(model) == 0:
            return 0.5
        return roc_auc(self.labels, self.scores(model))

    def detection(self, model: dict[str, Any]) -> list[evaluation.TypologyRecall]:
        """Per-typology detection rate at the agreed alert budget."""
        return evaluation.detection_by_typology(
            self.scores(model), self.labels, self.typologies
        )


def run_federation(
    nodes: Sequence[BankNode],
    holdout: HoldoutSet,
    num_rounds: int = config.NUM_ROUNDS,
    weighted_by_volume: bool = False,
    verbose: bool = True,
) -> list[RoundResult]:
    """Broadcast, train locally, collect weights, aggregate. Repeat."""
    global_model = bootstrap_global_model(nodes[0].params if nodes else None)
    history: list[RoundResult] = []

    for round_index in range(1, num_rounds + 1):
        updates = [node.train_round(global_model, round_index) for node in nodes]
        global_model = aggregator.aggregate_round(global_model, updates, weighted_by_volume)
        result = RoundResult(
            round_index=round_index,
            global_model=global_model,
            holdout_auc=holdout.auc(global_model),
            total_trees=weights.tree_count(global_model),
            payload_kb=sum(u.payload_kb for u in updates),
        )
        history.append(result)
        if verbose:
            print(aggregator.round_report(round_index, updates))
            print(f"    global: {result.total_trees} trees, holdout AUC {result.holdout_auc:.4f}\n")

    return history


def train_local_only(node: BankNode, num_trees: int) -> dict[str, Any]:
    """Control model: what this bank could build entirely on its own."""
    seed = bootstrap_global_model(node.params)
    update = node.train_round(seed, round_index=0, num_trees=num_trees)
    return weights.append_trees(seed, update.trees)


def train_pooled_reference(nodes: Sequence[BankNode], num_trees: int) -> dict[str, Any]:
    """Upper bound: the model you get if every bank pools its raw ledgers.

    This is the thing the federation exists to avoid doing. It is trained here
    purely as a reference point.
    """
    matrices, labels = zip(*(node.local_arrays for node in nodes))
    pooled = xgb.DMatrix(
        np.vstack(matrices),
        label=np.concatenate(labels),
        feature_names=list(config.FEATURE_COLUMNS),
    )
    seed = bootstrap_global_model(nodes[0].params)
    booster = xgb.train(nodes[0].params, pooled, num_boost_round=num_trees)
    return weights.append_trees(seed, weights.extract_trees(weights.booster_to_dict(booster), 0))


def save_global_model(model: dict[str, Any], out_dir: str = config.ARTIFACT_DIR) -> str:
    """Persist the aggregated model so downstream scoring can consume it."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "global_model.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model, handle)
    return path
