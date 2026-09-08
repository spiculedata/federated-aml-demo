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
