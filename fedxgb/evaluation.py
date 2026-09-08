"""Scoring a candidate model the way a transaction monitoring team would.

AUC is the headline, but an investigations unit has a fixed alert capacity.
What matters operationally is: at the alerts we can actually review, which
typologies do we catch?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedxgb import config


_ALL_TYPOLOGIES = "ALL CASES"


@dataclass(frozen=True)
class TypologyRecall:
    typology: str
    n_cases: int
    detected: int

    @property
    def rate(self) -> float:
        return self.detected / self.n_cases if self.n_cases else float("nan")


def alert_threshold(scores: np.ndarray, budget: float = config.ALERT_BUDGET) -> float:
    """The score cut-off that produces exactly ``budget`` share of alerts."""
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    return float(np.quantile(scores, 1.0 - budget))


def detection_by_typology(
    scores: np.ndarray,
    labels: np.ndarray,
    typologies: np.ndarray,
    budget: float = config.ALERT_BUDGET,
) -> list[TypologyRecall]:
    """Share of each typology's true cases that land inside the alert budget."""
    alerted = scores >= alert_threshold(scores, budget)
    results = []
    for typology in sorted(set(typologies[labels == 1])):
        mask = (labels == 1) & (typologies == typology)
        results.append(
            TypologyRecall(
                typology=typology,
                n_cases=int(mask.sum()),
                detected=int((mask & alerted).sum()),
            )
        )
    return results


def overall(recalls: list[TypologyRecall]) -> TypologyRecall:
    """Pooled detection rate across every typology - the headline number."""
    return TypologyRecall(
        typology=_ALL_TYPOLOGIES,
        n_cases=sum(r.n_cases for r in recalls),
        detected=sum(r.detected for r in recalls),
    )


@dataclass(frozen=True)
class ModelComparison:
    """One bank, measured against the federated model on the same hold-out.

    Split three ways because the headline hides the trade-off: a bank gives up
    accuracy on the typology it specialises in, and gains on the ones it has
    never seen.
    """

    label: str
    own_typology: str
    own_alone: float
    own_federated: float
    unseen_alone: float
    unseen_federated: float
    overall_alone: float
    overall_federated: float

    @property
    def own_delta(self) -> float:
        return self.own_federated - self.own_alone

    @property
    def unseen_delta(self) -> float:
        return self.unseen_federated - self.unseen_alone

    @property
    def overall_delta(self) -> float:
        return self.overall_federated - self.overall_alone


def _pooled(recalls: list[TypologyRecall], typologies: set[str]) -> float:
    """Detection rate across a chosen subset of typologies."""
    chosen = [r for r in recalls if r.typology in typologies]
    return overall(chosen).rate if chosen else float("nan")


def compare_models(
    label: str,
    own_typology: str,
    alone: list[TypologyRecall],
    federated: list[TypologyRecall],
) -> ModelComparison:
    """Contrast one bank's solo model with the federated model it helped build."""
    every = {r.typology for r in alone} | {r.typology for r in federated}
    unseen = every - {own_typology}
    return ModelComparison(
        label=label,
        own_typology=own_typology,
        own_alone=_pooled(alone, {own_typology}),
        own_federated=_pooled(federated, {own_typology}),
        unseen_alone=_pooled(alone, unseen),
        unseen_federated=_pooled(federated, unseen),
        overall_alone=overall(alone).rate,
        overall_federated=overall(federated).rate,
    )


def _arrow(before: float, after: float) -> str:
    return f"{before:>5.1%} -> {after:>5.1%}"


def format_comparison_table(comparisons: list[ModelComparison]) -> str:
    """The trade-off each participant actually makes by joining."""
    width = max((len(c.label) for c in comparisons), default=0) + 2
    lines = [
        f"  {'':<{width}}{'its own typology':>22}{'the ones it never saw':>26}{'overall':>20}",
        "  " + "-" * (width + 68),
    ]
    for c in comparisons:
        lines.append(
            f"  {c.label:<{width}}"
            f"{_arrow(c.own_alone, c.own_federated):>22}"
            f"{_arrow(c.unseen_alone, c.unseen_federated):>26}"
            f"{_arrow(c.overall_alone, c.overall_federated):>20}"
        )
    return "\n".join(lines)


def format_detection_table(rows: dict[str, list[TypologyRecall]], budget: float) -> str:
    """Model-by-typology detection rates as an aligned text table."""
    typologies = sorted({r.typology for rs in rows.values() for r in rs}) + [_ALL_TYPOLOGIES]
    name_width = max((len(n) for n in rows), default=0) + 2
    header = f"  {'model':<{name_width}}" + "".join(f"{t:>20}" for t in typologies)
    lines = [
        f"  detection rate within a {budget:.0%} alert budget\n",
        header,
        "  " + "-" * (name_width + 20 * len(typologies)),
    ]
    for name, recalls in rows.items():
        by_typology = {r.typology: r for r in recalls} | {_ALL_TYPOLOGIES: overall(recalls)}
        cells = "".join(
            f"{by_typology[t].rate:>20.1%}" if t in by_typology else f"{'-':>20}"
            for t in typologies
        )
        lines.append(f"  {name:<{name_width}}{cells}")
    return "\n".join(lines)
