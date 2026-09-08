"""Presentation layer for a live screen recording.

Designed for a projector, not a desk: one idea per screen, large bars, very
few numbers visible at once. Everything here is display only - it renders
results the federation has already produced and never computes any of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.text import Text

BAR_WIDTH = 30
LABEL_WIDTH = 18
OWN_COL_WIDTH = 20
UNSEEN_COL_WIDTH = 24

FILLED_BLOCK = "█"
EMPTY_BLOCK = "░"

STYLE_HEADING = "bold bright_white"
STYLE_BANK = "bright_cyan"
STYLE_SOLO = "yellow"
STYLE_FEDERATED = "bold bright_green"
STYLE_ALERT = "bright_red"
STYLE_QUIET = "grey62"

# Beats, multiplied by --pace. Tuned so a narrator can finish a sentence.
BEAT_SHORT = 1.2
BEAT_LONG = 2.5
# The trade-off screen is the one people read rather than glance at, so it
# gets long enough to actually take the numbers in and talk over them.
BEAT_STUDY = 12.0
ANIMATION_SECONDS = 0.9
ANIMATION_FRAMES = 24


def render_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """A fixed-width block bar. Clamped, so a bad fraction cannot corrupt it."""
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    return FILLED_BLOCK * filled + EMPTY_BLOCK * (width - filled)


def bar_row(label: str, fraction: float, style: str, show_value: bool = True) -> Text:
    """One labelled bar: name, bar, percentage."""
    row = Text("  ")
    row.append(f"{label:<{LABEL_WIDTH}}", style=STYLE_BANK if style != STYLE_FEDERATED else style)
    row.append(render_bar(fraction), style=style)
    if show_value:
        row.append(f"  {fraction:>5.1%}", style=style)
    return row


@dataclass(frozen=True)
class ScoreRow:
    """A single labelled result destined for a bar chart."""

    label: str
    fraction: float
    style: str = STYLE_SOLO


class Presentation:
    """Drives the screens. Each method owns the whole screen and then pauses."""

    def __init__(self, console: Console | None = None, pace: float = 1.0):
        self.console = console or Console()
        self.pace = pace

    # --- plumbing ---------------------------------------------------------

    def pause(self, beats: float = BEAT_SHORT) -> None:
        time.sleep(beats * self.pace)

    def _screen(self, *renderables: object, top_padding: int = 3) -> None:
        """Clear and draw one screen, padded down from the top edge."""
        self.console.clear()
        self.console.print(Group(*([Text("")] * top_padding), *renderables))

    def _heading(self, text: str) -> Text:
        return Text(f"  {text}", style=STYLE_HEADING)

    def _note(self, text: str) -> Text:
        return Text(f"  {text}", style=STYLE_QUIET)

    # --- screens ----------------------------------------------------------

    def title(self) -> None:
        self._screen(
            Align.left(Text("  FEDERATED TRANSACTION MONITORING", style=STYLE_HEADING)),
            Text(""),
            self._note("Three banks. One criminal network. No shared data."),
            top_padding=6,
        )
        self.pause(BEAT_LONG)

    def participants(self, banks: list[tuple[str, str]], rows_each: int) -> None:
        """banks: (display name, the typology this bank has actually seen)."""
        lines: list[Text] = []
        for name, typology in banks:
            row = Text("  ")
            row.append(f"{name:<{LABEL_WIDTH}}", style=STYLE_BANK)
            row.append("sees  ", style=STYLE_QUIET)
            row.append(typology.replace("_", "-").upper(), style=STYLE_HEADING)
            lines.append(row)

        self._screen(
            self._heading("THREE BANKS"),
            Text(""),
            *lines,
            Text(""),
            self._note(f"{rows_each:,} transactions each. Each blind to the other two."),
        )
        self.pause(BEAT_LONG)

    def streaming(self, loaded: list[tuple[str, int, float]]) -> None:
        """Show each bank streaming its own ledger, one line at a time."""
        lines: list[Text] = []
        for name, rows, seconds in loaded:
            row = Text("  ")
            row.append(f"{name:<{LABEL_WIDTH}}", style=STYLE_BANK)
            row.append(f"{rows:>9,} rows", style=STYLE_HEADING)
            row.append(f"   {seconds:>5.2f}s", style=STYLE_QUIET)
            lines.append(row)
            self._screen(
                self._heading("EACH BANK STREAMS ITS OWN LEDGER"),
                Text(""),
                *lines,
                Text(""),
                self._note("Polars streaming engine, inside the bank."),
            )
            self.pause(0.5)
        self.pause(BEAT_SHORT)

    def _animated_bars(self, heading: str, rows: list[ScoreRow], note: str) -> None:
        """Grow every bar from zero to its value, then hold."""
        for frame in range(ANIMATION_FRAMES + 1):
            progress = frame / ANIMATION_FRAMES
            self._screen(
                self._heading(heading),
                Text(""),
                *[bar_row(r.label, r.fraction * progress, r.style) for r in rows],
                Text(""),
                self._note(note),
            )
            time.sleep(ANIMATION_SECONDS / ANIMATION_FRAMES)

    def solo_results(self, rows: list[ScoreRow]) -> None:
        self._animated_bars(
            "ALONE, EACH BANK CATCHES ONE THING",
            rows,
            "share of all criminal cases caught, within a 2% alert budget",
        )
        self.pause(BEAT_LONG)

    def federating(self, round_index: int, num_rounds: int, payloads: list[tuple[str, float]],
                   trees: int) -> None:
        """Live frame for one federation round."""
        header = Text("  FEDERATING", style=STYLE_HEADING)
        header.append(f"{'round ' + str(round_index) + ' of ' + str(num_rounds):>44}",
                      style=STYLE_QUIET)

        lines: list[Text] = []
        for name, kilobytes in payloads:
            row = Text("  ")
            row.append(f"{name:<{LABEL_WIDTH}}", style=STYLE_BANK)
            row.append("↑ ", style=STYLE_FEDERATED)
            row.append(f"{kilobytes:>4.1f} KB of tree weights", style=STYLE_HEADING)
            lines.append(row)

        totals = Text("  ")
        totals.append(f"{'global model':<{LABEL_WIDTH}}", style=STYLE_QUIET)
        totals.append(f"{trees} trees", style=STYLE_HEADING)

        shared = Text("  ")
        shared.append(f"{'transactions sent':<{LABEL_WIDTH}}", style=STYLE_QUIET)
        shared.append("0", style=STYLE_FEDERATED)

        self._screen(header, Text(""), *lines, Text(""), totals, shared)

    def reveal(self, solo: list[ScoreRow], federated: ScoreRow) -> None:
        """The money shot: the federated bar grows past every solo bar."""
        static = [bar_row(r.label, r.fraction, r.style) for r in solo]
        for frame in range(ANIMATION_FRAMES + 1):
            progress = frame / ANIMATION_FRAMES
            self._screen(
                self._heading("TOGETHER"),
                Text(""),
                *static,
                Text(""),
                bar_row(federated.label, federated.fraction * progress, STYLE_FEDERATED),
            )
            time.sleep(ANIMATION_SECONDS / ANIMATION_FRAMES)
        self.pause(BEAT_LONG)

    def trade_off(
        self,
        rows: list[tuple[str, float, float, float, float]],
        hold: float = BEAT_STUDY,
    ) -> None:
        """The honest screen: what each bank gives up, and what it gains.

        rows: (name, own_alone, own_federated, unseen_alone, unseen_federated)
        ``hold`` is in beats, so it still scales with --pace for rehearsals.
        """
        header = Text("  ")
        header.append(" " * LABEL_WIDTH)
        header.append(f"{'its own typology':^{OWN_COL_WIDTH}}", style=STYLE_QUIET)
        header.append(f"{'the ones it never saw':^{UNSEEN_COL_WIDTH}}", style=STYLE_QUIET)

        lines: list[Text] = []
        for name, own_before, own_after, unseen_before, unseen_after in rows:
            own = f"{own_before:.0%} → {own_after:.0%}"
            unseen = f"{unseen_before:.0%} → {unseen_after:.0%}"
            row = Text("  ")
            row.append(f"{name:<{LABEL_WIDTH}}", style=STYLE_BANK)
            row.append(f"{own:^{OWN_COL_WIDTH}}", style=STYLE_ALERT)
            row.append(f"{unseen:^{UNSEEN_COL_WIDTH}}", style=STYLE_FEDERATED)
            lines.append(row)

        self._screen(
            self._heading("WHAT EACH BANK TRADES"),
            Text(""),
            header,
            Text(""),
            *lines,
            Text(""),
            self._note("worse at its speciality. far better at everything else."),
        )
        self.pause(hold)

    def wire_summary(self, weights_kb: float) -> None:
        rows = [
            ("tree weights", f"{weights_kb:,.0f} KB", STYLE_HEADING),
            ("transactions", "0", STYLE_FEDERATED),
            ("account identifiers", "0", STYLE_FEDERATED),
            ("amounts, dates, names", "0", STYLE_FEDERATED),
        ]
        lines = []
        for label, value, style in rows:
            row = Text("  ")
            row.append(f"{label:<24}", style=STYLE_QUIET)
            row.append(f"{value:>10}", style=style)
            lines.append(row)

        self._screen(self._heading("WHAT CROSSED THE WIRE"), Text(""), *lines)
        self.pause(BEAT_LONG)
