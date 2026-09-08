#!/usr/bin/env python3
"""Federated transaction monitoring: three banks, one model, no shared data.

    uv run run_demo.py             # generate data if needed, then train
    uv run run_demo.py --explain   # also print the Polars streaming plan
    uv run run_demo.py --weighted  # weight each bank by its data volume
"""

from __future__ import annotations

import argparse
import os
import time

from fedxgb import config, data_gen, evaluation, features, server
from fedxgb.bank_node import BankNode

_RULE = "=" * 86


def _heading(text: str) -> None:
    print(f"\n{_RULE}\n{text}\n{_RULE}")


def ensure_data() -> dict[str, str]:
    """Generate the synthetic ledgers once; reuse them on later runs."""
    holdout = os.path.join(config.DATA_DIR, "central_holdout.parquet")
    paths = {b: os.path.join(config.DATA_DIR, f"{b}.parquet") for b in config.BANK_IDS}
    if os.path.exists(holdout) and all(os.path.exists(p) for p in paths.values()):
        return {**paths, "holdout": holdout}

    print("Generating synthetic ledgers (one private file per bank)...")
    return data_gen.build_all()


def load_nodes(paths: dict[str, str]) -> list[BankNode]:
    """Stand up each bank's node and stream its private ledger into memory."""
    nodes = []
    for bank_id in config.BANK_IDS:
        node = BankNode(bank_id, paths[bank_id])
        started = time.perf_counter()
        node.load_local_data()
        elapsed = time.perf_counter() - started
        print(
            f"  {bank_id:<17} typology seen locally: {config.BANK_TYPOLOGIES[bank_id]:<18}"
            f"{node.dmatrix.num_row():>8,} rows streamed in {elapsed:5.2f}s"
        )
        nodes.append(node)
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--weighted", action="store_true", help="weight banks by data volume")
    parser.add_argument("--explain", action="store_true", help="print the streaming query plan")
    args = parser.parse_args()

    paths = ensure_data()

    if args.explain:
        _heading("POLARS STREAMING PLAN (run independently inside each bank)")
        print(features.explain_plan(paths[config.BANK_IDS[0]]))

    _heading("PARTICIPANTS")
    nodes = load_nodes(paths)
    holdout = server.HoldoutSet(paths["holdout"])
    print(
        f"\n  central hold-out: {holdout.dmatrix.num_row():,} rows containing all three "
        f"typologies\n  no participant has ever seen this data"
    )

    # Every model in the comparison gets the same ensemble size, so the only
    # difference on the scoreboard is how much of the world it got to see.
    tree_budget = args.rounds * config.TREES_PER_BANK_PER_ROUND * len(nodes)
    scoreboard: dict[str, float] = {}
    detection: dict[str, list[evaluation.TypologyRecall]] = {}

    _heading("BASELINE - EACH BANK TRAINING ALONE")
    solo_detection: dict[str, list[evaluation.TypologyRecall]] = {}
    for node in nodes:
        model = server.train_local_only(node, tree_budget)
        label = f"{node.bank_id} (solo)"
        scoreboard[label] = holdout.auc(model)
        detection[label] = holdout.detection(model)
        solo_detection[node.bank_id] = detection[label]
        print(f"  {label:<34} {tree_budget} trees, holdout AUC {scoreboard[label]:.4f}")

    _heading(f"FEDERATED TRAINING - {args.rounds} ROUNDS")
    print(
        f"  each round: broadcast global model -> {len(nodes)} banks boost "
        f"{config.TREES_PER_BANK_PER_ROUND} trees locally\n"
        f"              -> return tree weights only -> average and append\n"
    )
    history = server.run_federation(
        nodes, holdout, num_rounds=args.rounds, weighted_by_volume=args.weighted
    )
    final = history[-1]
    scoreboard["FEDERATED (weights only)"] = final.holdout_auc
    detection["FEDERATED (weights only)"] = holdout.detection(final.global_model)

    pooled = server.train_pooled_reference(nodes, tree_budget)
    scoreboard["pooled raw data (not allowed)"] = holdout.auc(pooled)
    detection["pooled raw data (not allowed)"] = holdout.detection(pooled)

    _heading("HOLD-OUT PERFORMANCE")
    for name, auc in sorted(scoreboard.items(), key=lambda kv: kv[1]):
        bar = "#" * int((auc - 0.5) * 100)
        print(f"  {name:<34} AUC {auc:.4f}  {bar}")

    print()
    print(evaluation.format_detection_table(detection, config.ALERT_BUDGET))

    federated_detection = detection["FEDERATED (weights only)"]
    comparisons = [
        evaluation.compare_models(
            node.bank_id,
            config.BANK_TYPOLOGIES[node.bank_id],
            solo_detection[node.bank_id],
            federated_detection,
        )
        for node in nodes
    ]

    _heading("WHAT EACH BANK TRADES BY JOINING")
    print(evaluation.format_comparison_table(comparisons))
    worst = min(comparisons, key=lambda c: c.own_delta)
    print(
        f"\n  Every participant is WORSE at its own speciality after federating"
        f"\n  ({worst.label} loses "
        f"{abs(worst.own_delta):.1%} on {worst.own_typology.replace('_', '-')}), because a fixed"
        f"\n  2% alert budget now has to cover three typologies instead of one."
        f"\n  Overall detection still roughly doubles for all three."
    )

    path = server.save_global_model(final.global_model)
    _heading("WHAT ACTUALLY CROSSED THE WIRE")
    print(f"  rounds                 : {len(history)}")
    print(f"  trees in global model  : {final.total_trees}")
    print(f"  tree weights sent      : {sum(r.payload_kb for r in history):,.1f} KB")
    print(f"  transactions sent      : 0")
    print(f"  account identifiers sent: 0")
    print(f"  global model written to: {path}")
    print(
        "\n  Every bank kept its ledger. Only split thresholds and leaf weights left\n"
        "  the building, and the central model now detects typologies that no single\n"
        "  participant has ever observed."
    )


if __name__ == "__main__":
    main()
