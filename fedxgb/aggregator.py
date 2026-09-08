"""The central service.

Receives tree payloads from every participant, averages their contributions
and folds them into one global model. It never sees a transaction, an account
number or a bank's row-level data.
"""

from __future__ import annotations

from typing import Any, Sequence

from fedxgb.bank_node import BankUpdate
from fedxgb import weights


def averaging_factors(updates: Sequence[BankUpdate], weighted_by_volume: bool) -> list[float]:
    """Per-bank scaling factors applied to leaf weights before merging.

    Uniform averaging treats each institution equally; volume weighting gives
    a larger bank proportionally more influence, mirroring FedAvg.
    """
    if not updates:
        return []
    if not weighted_by_volume:
        return [1.0 / len(updates)] * len(updates)

    total_rows = sum(u.n_local_rows for u in updates)
    if total_rows == 0:
        return [1.0 / len(updates)] * len(updates)
    return [u.n_local_rows / total_rows for u in updates]


def aggregate_round(
    global_model: dict[str, Any],
    updates: Sequence[BankUpdate],
    weighted_by_volume: bool = False,
) -> dict[str, Any]:
    """Fold one round of bank updates into a new global model.

    Each bank boosted from the same starting point, so its trees encode a
    candidate update. Scaling every leaf by the bank's share and appending all
    of them is equivalent to averaging those updates in margin space.
    """
    factors = averaging_factors(updates, weighted_by_volume)
    merged = [
        weights.scale_leaf_weights(tree, factor)
        for update, factor in zip(updates, factors)
        for tree in update.trees
    ]
    return weights.append_trees(global_model, merged)


def round_report(round_index: int, updates: Sequence[BankUpdate]) -> str:
    """One line per participant, plus what the round cost on the wire."""
    lines = [f"  round {round_index:>2}"]
    lines += [f"    {u.summary()}" for u in updates]
    lines.append(
        f"    -> merged {sum(len(u.trees) for u in updates)} trees, "
        f"{sum(u.payload_kb for u in updates):.1f} KB transmitted, 0 rows shared"
    )
    return "\n".join(lines)
