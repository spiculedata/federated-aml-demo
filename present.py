#!/usr/bin/env python3
"""The federated demo, staged for a screen recording.

    uv run present.py                # full pace, for a talk
    uv run present.py --step         # advance every screen by hand
    uv run present.py --pace 0.3     # quick rehearsal
    uv run present.py --no-tour      # skip the code walkthrough

Same code path and same numbers as run_demo.py - only the presentation
differs. Nothing here computes a result.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rich.console import Console

from fedxgb import code_tour, config, evaluation, presenter, server
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


def deployment_rows(
    nodes: list[BankNode],
    local_scores: dict[str, "np.ndarray"],
    federated_scores: "np.ndarray",
    holdout: server.HoldoutSet,
) -> list[tuple[str, float, float, float, float, float, float]]:
    """Per bank: its own alert queue, then that queue plus the federated one."""
    rows = []
    for node in nodes:
        c = evaluation.compare_deployment(
            DISPLAY_NAMES[node.bank_id],
            config.BANK_TYPOLOGIES[node.bank_id],
            local_scores[node.bank_id],
            federated_scores,
            holdout.labels,
            holdout.typologies,
        )
        rows.append(
            (
                c.label,
                c.own_alone,
                c.own_combined,
                c.unseen_alone,
                c.unseen_combined,
                c.overall_alone,
                c.overall_combined,
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
    parser.add_argument(
        "--code-hold",
        type=float,
        default=presenter.BEAT_CODE,
        help="seconds to hold each code slide (default 15)",
    )
    parser.add_argument(
        "--no-tour", action="store_true", help="skip the code walkthrough"
    )
    parser.add_argument(
        "--step", action="store_true", help="advance every screen on Enter instead of a timer"
    )
    args = parser.parse_args()

    console = Console()
    show = Presentation(console, pace=args.pace, step=args.step)

    paths = ensure_data()

    show.title()
    if not args.no_tour:
        if code_tour.sources_available():
            for snippet in code_tour.snippets():
                show.code(snippet, hold=args.code_hold)
        else:
            console.print(
                "[yellow]Skipping the code tour: Python sources are not on disk "
                "(compiled install).[/yellow]"
            )
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
    local_scores: dict[str, np.ndarray] = {}
    for node in nodes:
        model = server.train_local_only(node, tree_budget)
        local_scores[node.bank_id] = holdout.scores(model)
        solo.append(
            ScoreRow(
                DISPLAY_NAMES[node.bank_id],
                evaluation.overall(holdout.detection(model)).rate,
                STYLE_SOLO,
            )
        )
    show.solo_results(solo)

    observer = LiveRoundObserver(show, args.rounds)
    history = server.run_federation(
        nodes, holdout, num_rounds=args.rounds, verbose=False, observer=observer, parallel=True
    )
    final = history[-1]

    federated_scores = holdout.scores(final.global_model)
    show.reveal(
        solo,
        ScoreRow(
            "FEDERATED",
            evaluation.overall(holdout.detection(final.global_model)).rate,
            STYLE_FEDERATED,
        ),
    )
    show.trade_off(
        deployment_rows(nodes, local_scores, federated_scores, holdout), hold=args.hold
    )
    show.wire_summary(sum(r.payload_kb for r in history))


if __name__ == "__main__":
    main()
