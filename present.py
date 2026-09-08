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


def trade_off_rows(
    nodes: list[BankNode],
    solo_detection: dict[str, list[evaluation.TypologyRecall]],
    federated_detection: list[evaluation.TypologyRecall],
) -> list[tuple[str, float, float, float, float]]:
    """Per bank: what it gives up on its speciality, what it gains elsewhere."""
    rows = []
    for node in nodes:
        comparison = evaluation.compare_models(
            DISPLAY_NAMES[node.bank_id],
            config.BANK_TYPOLOGIES[node.bank_id],
            solo_detection[node.bank_id],
            federated_detection,
        )
        rows.append(
            (
                comparison.label,
                comparison.own_alone,
                comparison.own_federated,
                comparison.unseen_alone,
                comparison.unseen_federated,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument("--pace", type=float, default=1.0, help="1.0 = talk pace")
    parser.add_argument(
        "--hold",
        type=float,
        default=presenter.BEAT_STUDY,
        help="seconds to hold the trade-off comparison (default 12)",
    )
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

    solo: list[ScoreRow] = []
    solo_detection: dict[str, list[evaluation.TypologyRecall]] = {}
    for node in nodes:
        model = server.train_local_only(node, tree_budget)
        solo_detection[node.bank_id] = holdout.detection(model)
        solo.append(
            ScoreRow(
                DISPLAY_NAMES[node.bank_id],
                evaluation.overall(solo_detection[node.bank_id]).rate,
                STYLE_SOLO,
            )
        )
    show.solo_results(solo)

    observer = LiveRoundObserver(show, args.rounds)
    history = server.run_federation(
        nodes, holdout, num_rounds=args.rounds, verbose=False, observer=observer, parallel=True
    )
    final = history[-1]

    federated_detection = holdout.detection(final.global_model)
    show.reveal(
        solo,
        ScoreRow("FEDERATED", evaluation.overall(federated_detection).rate, STYLE_FEDERATED),
    )
    show.trade_off(trade_off_rows(nodes, solo_detection, federated_detection), hold=args.hold)
    show.wire_summary(sum(r.payload_kb for r in history))


if __name__ == "__main__":
    main()
