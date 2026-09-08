#!/usr/bin/env python3
"""The federated demo, staged for a screen recording.

    uv run present.py                # full pace, for a talk
    uv run present.py --pace 0.3     # quick rehearsal
    uv run present.py --rounds 4     # shorter federation

Same code path and same numbers as run_demo.py - only the presentation
differs. Nothing here computes a result.
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console

from fedxgb import config, evaluation, presenter, server
from fedxgb.bank_node import BankNode
from fedxgb.presenter import STYLE_FEDERATED, STYLE_SOLO, Presentation, ScoreRow
from run_demo import ensure_data

DISPLAY_NAMES = {
    "northwind_bank": "Northwind Bank",
    "caledonia_trust": "Caledonia Trust",
    "meridian_pcb": "Meridian PCB",
}


class LiveRoundObserver:
    """Renders each federation round as it happens."""

    def __init__(self, show: Presentation, num_rounds: int):
        self.show = show
        self.num_rounds = num_rounds
        self.round_index = 0
        self.payloads: dict[str, float] = {}

    def on_round_start(self, round_index: int, num_rounds: int) -> None:
        self.round_index = round_index
        self.payloads = {}

    def on_bank_update(self, update) -> None:
        self.payloads[update.bank_id] = update.payload_kb
        self._draw(trees=0)
        time.sleep(0.18 * self.show.pace)

    def on_round_end(self, result) -> None:
        self._draw(trees=result.total_trees)
        time.sleep(0.35 * self.show.pace)

    def _draw(self, trees: int) -> None:
        rows = [
            (DISPLAY_NAMES[bank_id], self.payloads[bank_id])
            for bank_id in config.BANK_IDS
            if bank_id in self.payloads
        ]
        self.show.federating(self.round_index, self.num_rounds, rows, trees)


def overall_detection(holdout: server.HoldoutSet, model: dict) -> float:
    """Share of all criminal cases caught within the alert budget."""
    return evaluation.overall(holdout.detection(model)).rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--pace", type=float, default=1.0, help="1.0 = talk pace")
    args = parser.parse_args()

    console = Console()
    show = Presentation(console, pace=args.pace)

    paths = ensure_data()

    show.title()
    show.participants(
        [(DISPLAY_NAMES[b], config.BANK_TYPOLOGIES[b]) for b in config.BANK_IDS],
        rows_each=config.ROWS_PER_BANK,
    )

    nodes: list[BankNode] = []
    loaded: list[tuple[str, int, float]] = []
    for bank_id in config.BANK_IDS:
        node = BankNode(bank_id, paths[bank_id])
        started = time.perf_counter()
        node.load_local_data()
        loaded.append((DISPLAY_NAMES[bank_id], node.dmatrix.num_row(), time.perf_counter() - started))
        nodes.append(node)
    show.streaming(loaded)

    holdout = server.HoldoutSet(paths["holdout"])
    tree_budget = args.rounds * config.TREES_PER_BANK_PER_ROUND * len(nodes)

    solo = [
        ScoreRow(
            DISPLAY_NAMES[node.bank_id],
            overall_detection(holdout, server.train_local_only(node, tree_budget)),
            STYLE_SOLO,
        )
        for node in nodes
    ]
    show.solo_results(solo)

    observer = LiveRoundObserver(show, args.rounds)
    history = server.run_federation(
        nodes, holdout, num_rounds=args.rounds, verbose=False, observer=observer, parallel=True
    )
    final = history[-1]

    show.reveal(
        solo,
        ScoreRow("FEDERATED", overall_detection(holdout, final.global_model), STYLE_FEDERATED),
    )
    show.wire_summary(sum(r.payload_kb for r in history))


if __name__ == "__main__":
    main()
