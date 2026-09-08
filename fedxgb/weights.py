"""The wire format: XGBoost tree weights, sliced and spliced as plain JSON.

Nothing in this module touches a transaction. It only ever handles split
thresholds and leaf weights - the payload that is allowed to leave a bank.
All functions are pure and return new structures.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import xgboost as xgb

_LEAF_SENTINEL = -1


def booster_to_dict(booster: xgb.Booster) -> dict[str, Any]:
    """Serialise a booster into its native JSON representation."""
    return json.loads(booster.save_raw(raw_format="json").decode("utf-8"))


def dict_to_booster(model: dict[str, Any]) -> xgb.Booster:
    """Rehydrate a booster from a JSON model dictionary."""
    return xgb.Booster(model_file=bytearray(json.dumps(model), "utf-8"))


def _model_section(model: dict[str, Any]) -> dict[str, Any]:
    return model["learner"]["gradient_booster"]["model"]


def tree_count(model: dict[str, Any]) -> int:
    """Number of boosted trees currently held in the model."""
    return len(_model_section(model)["trees"])


def extract_trees(model: dict[str, Any], start: int) -> list[dict[str, Any]]:
    """Return a deep copy of the trees added at or after ``start``.

    This is how a bank isolates *its contribution* from the global model it
    was seeded with, so only the delta is transmitted.
    """
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    return copy.deepcopy(_model_section(model)["trees"][start:])


def scale_leaf_weights(tree: dict[str, Any], factor: float) -> dict[str, Any]:
    """Return a copy of ``tree`` with every leaf weight multiplied by ``factor``.

    Split thresholds live in the same ``split_conditions`` array as leaf values,
    so only nodes without children may be touched.
    """
    scaled = copy.deepcopy(tree)
    for node, left_child in enumerate(scaled["left_children"]):
        if left_child == _LEAF_SENTINEL:
            scaled["split_conditions"][node] *= factor
            scaled["base_weights"][node] *= factor
    return scaled


def with_trees(model: dict[str, Any], trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``model`` whose tree ensemble is exactly ``trees``.

    Tree ids, ``tree_info`` and the iteration index are all rewritten so the
    result is a structurally valid XGBoost model.
    """
    rebuilt = copy.deepcopy(model)
    section = _model_section(rebuilt)
    section["trees"] = [{**tree, "id": index} for index, tree in enumerate(trees)]
    section["tree_info"] = [0] * len(trees)
    section["iteration_indptr"] = list(range(len(trees) + 1))
    section["gbtree_model_param"]["num_trees"] = str(len(trees))
    return rebuilt


def append_trees(model: dict[str, Any], trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``model`` with ``trees`` appended to the ensemble."""
    return with_trees(model, _model_section(model)["trees"] + trees)


def empty_like(model: dict[str, Any]) -> dict[str, Any]:
    """A structurally identical model carrying no trees - the round-zero global."""
    return with_trees(model, [])


def payload_size_kb(trees: list[dict[str, Any]]) -> float:
    """Size of a tree payload on the wire, for the 'what did we share?' report."""
    return len(json.dumps(trees).encode("utf-8")) / 1024
