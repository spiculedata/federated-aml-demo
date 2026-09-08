"""The code walkthrough shown before the demo runs.

Snippets are pulled out of the real source files at runtime by symbol name,
never copied into this module. If a function is renamed or deleted, the tour
raises instead of showing a slide that no longer matches the code - and a test
walks every stop to catch that before a recording does.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Widest line a code slide may contain, so it survives projector zoom.
MAX_CODE_WIDTH = 78


@dataclass(frozen=True)
class TourStop:
    """One code slide: where it comes from, and what to say about it."""

    path: str
    symbol: str
    headline: str
    note: str


@dataclass(frozen=True)
class Snippet:
    """Extracted source, ready to render."""

    stop: TourStop
    code: str
    start_line: int

    @property
    def widest_line(self) -> int:
        return max((len(line) for line in self.code.splitlines()), default=0)


def _definition(tree: ast.Module, symbol: str) -> ast.FunctionDef | ast.ClassDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            return node
    raise LookupError(f"{symbol!r} is not a top-level definition")


def _docstring_end(node: ast.FunctionDef | ast.ClassDef) -> int | None:
    """Line number the leading docstring ends on, if there is one."""
    if not node.body:
        return None
    first = node.body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    return first.end_lineno if is_docstring else None


def extract(stop: TourStop, strip_docstring: bool = True) -> Snippet:
    """Pull one symbol's source out of its file.

    The docstring is dropped by default: the speaker narrates the slide, so
    on-screen prose competes with them and costs vertical space.
    """
    source = Path(stop.path).read_text(encoding="utf-8")
    lines = source.splitlines()
    node = _definition(ast.parse(source), stop.symbol)

    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    docstring_end = _docstring_end(node) if strip_docstring else None

    if docstring_end is None:
        body = lines[start - 1 : node.end_lineno]
    else:
        signature = lines[start - 1 : node.body[0].lineno - 1]
        remainder = lines[docstring_end : node.end_lineno]
        body = signature + remainder

    while body and not body[-1].strip():
        body.pop()
    return Snippet(stop=stop, code="\n".join(body), start_line=start)


TOUR: tuple[TourStop, ...] = (
    TourStop(
        path="fedxgb/features.py",
        symbol="build_feature_plan",
        headline="EACH BANK PREPARES ITS OWN DATA",
        note="A lazy plan. Nothing is read until it is collected, and then it streams.",
    ),
    TourStop(
        path="fedxgb/bank_node.py",
        symbol="BankUpdate",
        headline="EVERYTHING A BANK IS ALLOWED TO SEND",
        note="The whole payload. No field here could carry a transaction.",
    ),
    TourStop(
        path="fedxgb/weights.py",
        symbol="scale_leaf_weights",
        headline="THE ONE LINE THAT MUST NOT BE WRONG",
        note="Split thresholds share this array with leaf values. Scale one and the tree lies.",
    ),
    TourStop(
        path="fedxgb/aggregator.py",
        symbol="aggregate_round",
        headline="AVERAGING, WITHOUT THE DATA",
        note="Scale each bank's leaves by its share, append. Equivalent to averaging the updates.",
    ),
)


def snippets() -> list[Snippet]:
    """Every tour stop, extracted from the current source."""
    return [extract(stop) for stop in TOUR]
