"""The code tour must always match the code that actually ships."""

from __future__ import annotations

import pathlib

import pytest

from fedxgb import code_tour


@pytest.mark.parametrize("stop", code_tour.TOUR, ids=lambda s: s.symbol)
def test_every_tour_stop_still_resolves(stop):
    """Renaming or deleting a symbol must break the build, not the recording."""
    snippet = code_tour.extract(stop)

    assert snippet.code.strip()
    assert stop.symbol in snippet.code


@pytest.mark.parametrize("stop", code_tour.TOUR, ids=lambda s: s.symbol)
def test_every_slide_fits_a_projector(stop):
    snippet = code_tour.extract(stop)

    assert snippet.widest_line <= code_tour.MAX_CODE_WIDTH


@pytest.mark.parametrize("stop", code_tour.TOUR, ids=lambda s: s.symbol)
def test_start_line_points_at_the_real_definition(stop):
    snippet = code_tour.extract(stop)
    file_lines = pathlib.Path(stop.path).read_text().splitlines()

    assert file_lines[snippet.start_line - 1] == snippet.code.splitlines()[0]


def test_docstrings_are_stripped_by_default():
    stop = code_tour.TourStop("fedxgb/weights.py", "extract_trees", "h", "n")

    assert "Return a deep copy" not in code_tour.extract(stop).code


def test_docstrings_can_be_kept():
    stop = code_tour.TourStop("fedxgb/weights.py", "extract_trees", "h", "n")

    assert "Return a deep copy" in code_tour.extract(stop, strip_docstring=False).code


def test_decorators_are_included():
    """BankUpdate is a frozen dataclass - the decorator is part of the point."""
    stop = code_tour.TourStop("fedxgb/bank_node.py", "BankUpdate", "h", "n")

    assert code_tour.extract(stop).code.startswith("@dataclass")


def test_an_unknown_symbol_fails_loudly():
    stop = code_tour.TourStop("fedxgb/weights.py", "no_such_function", "h", "n")

    with pytest.raises(LookupError, match="no_such_function"):
        code_tour.extract(stop)


def test_snippets_returns_every_stop():
    assert len(code_tour.snippets()) == len(code_tour.TOUR)
