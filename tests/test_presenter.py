"""Presentation rendering. Pure string work - no federation involved."""

from __future__ import annotations

import pytest

from fedxgb import presenter


@pytest.mark.parametrize(
    "fraction,expected_filled",
    [(0.0, 0), (0.5, 5), (1.0, 10), (-3.0, 0), (2.0, 10)],
)
def test_bar_is_clamped_to_its_width(fraction, expected_filled):
    bar = presenter.render_bar(fraction, width=10)

    assert len(bar) == 10
    assert bar.count(presenter.FILLED_BLOCK) == expected_filled


def test_bar_row_shows_the_label_and_percentage():
    row = presenter.bar_row("Northwind Bank", 0.428, presenter.STYLE_SOLO)

    assert "Northwind Bank" in row.plain
    assert "42.8%" in row.plain


def test_bar_row_can_omit_the_percentage():
    row = presenter.bar_row("Northwind Bank", 0.428, presenter.STYLE_SOLO, show_value=False)

    assert "%" not in row.plain


def test_every_bar_renders_to_the_same_width():
    widths = {len(presenter.render_bar(f / 20)) for f in range(21)}

    assert widths == {presenter.BAR_WIDTH}


def test_trade_off_holds_long_enough_to_read(monkeypatch):
    """The comparison screen must not revert to a glance-length pause."""
    import io

    from rich.console import Console

    held: list[float] = []
    show = presenter.Presentation(Console(file=io.StringIO()), pace=1.0)
    monkeypatch.setattr(show, "pause", held.append)

    show.trade_off([("Bank", 0.5, 0.5, 0.0, 0.4, 0.2, 0.45)])

    assert held == [presenter.BEAT_STUDY]
    assert presenter.BEAT_STUDY >= 10.0


def test_trade_off_hold_is_overridable_per_take(monkeypatch):
    import io

    from rich.console import Console

    held: list[float] = []
    show = presenter.Presentation(Console(file=io.StringIO()), pace=1.0)
    monkeypatch.setattr(show, "pause", held.append)

    show.trade_off([("Bank", 0.5, 0.5, 0.0, 0.4, 0.2, 0.45)], hold=15.0)

    assert held == [15.0]


def test_step_mode_waits_for_the_speaker_instead_of_a_timer():
    """On code slides the speaker needs control, not a stopwatch."""
    import io

    from rich.console import Console

    class StubConsole(Console):
        def __init__(self):
            super().__init__(file=io.StringIO())
            self.prompts = 0

        def input(self, *args, **kwargs):
            self.prompts += 1
            return ""

    console = StubConsole()
    show = presenter.Presentation(console, pace=1.0, step=True)

    show.pause(beats=999.0)

    assert console.prompts == 1
