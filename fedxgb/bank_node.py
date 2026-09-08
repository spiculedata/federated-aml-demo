"""A participating bank.

Loads its own ledger through the Polars streaming engine, continues boosting
from the global model it was handed, and returns *only* the trees it grew.
Raw transactions never leave this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xgboost as xgb

from fedxgb import config, features, weights


@dataclass(frozen=True)
class BankUpdate:
    """The complete payload a bank sends back to the central service."""

    bank_id: str
    round_index: int
    trees: list[dict[str, Any]]
    n_local_rows: int
    n_local_positives: int
    local_train_auc: float
    payload_kb: float = field(default=0.0)

    def summary(self) -> str:
        return (
            f"{self.bank_id:>16}  round {self.round_index:>2}  "
            f"trees={len(self.trees):>2}  payload={self.payload_kb:6.1f} KB  "
            f"local_auc={self.local_train_auc:.4f}"
        )


class BankNode:
    """One institution's private training node."""

    def __init__(self, bank_id: str, ledger_path: str, params: dict[str, Any] | None = None):
        self.bank_id = bank_id
        self.ledger_path = ledger_path
        self.params = dict(params or config.XGB_PARAMS)
        self._dmatrix: xgb.DMatrix | None = None
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None
        self._n_rows = 0
        self._n_positives = 0

    def load_local_data(self) -> None:
        """Stream the private ledger into an in-memory training matrix."""
        frame = features.collect_streaming(features.build_feature_plan(self.ledger_path))
        labels = frame[config.LABEL_COLUMN].to_numpy()
        matrix = frame.select(config.FEATURE_COLUMNS).to_numpy()
        self._dmatrix = xgb.DMatrix(
            matrix, label=labels, feature_names=list(config.FEATURE_COLUMNS)
        )
        self._arrays = (matrix, labels)
        self._n_rows = len(labels)
        self._n_positives = int(labels.sum())

    @property
    def local_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels. Used only by the illegal-pooling reference run."""
        if self._arrays is None:
            raise RuntimeError(f"{self.bank_id}: call load_local_data() first")
        return self._arrays

    @property
    def dmatrix(self) -> xgb.DMatrix:
        if self._dmatrix is None:
            raise RuntimeError(f"{self.bank_id}: call load_local_data() first")
        return self._dmatrix

    def train_round(
        self,
        global_model: dict[str, Any],
        round_index: int,
        num_trees: int = config.TREES_PER_BANK_PER_ROUND,
    ) -> BankUpdate:
        """Boost ``num_trees`` further on local data and return just those trees."""
        seed_booster = weights.dict_to_booster(global_model)
        trees_before = weights.tree_count(global_model)

        local = xgb.train(
            self.params,
            self.dmatrix,
            num_boost_round=num_trees,
            xgb_model=seed_booster,
            verbose_eval=False,
        )
        local_model = weights.booster_to_dict(local)
        new_trees = weights.extract_trees(local_model, start=trees_before)

        return BankUpdate(
            bank_id=self.bank_id,
            round_index=round_index,
            trees=new_trees,
            n_local_rows=self._n_rows,
            n_local_positives=self._n_positives,
            local_train_auc=self._auc(local),
            payload_kb=weights.payload_size_kb(new_trees),
        )

    def _auc(self, booster: xgb.Booster) -> float:
        scores = booster.predict(self.dmatrix)
        labels = self.dmatrix.get_label()
        return roc_auc(labels, scores)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC. Kept local so the demo has no scikit-learn dependency."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    ranks = _average_ranks(scores)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """Competition-free ranks, with tied scores sharing their mean rank."""
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    for end in range(1, len(ordered) + 1):
        if end == len(ordered) or ordered[end] != ordered[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return ranks
