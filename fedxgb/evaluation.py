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


def alert_mask(scores: np.ndarray, budget: float = config.ALERT_BUDGET) -> np.ndarray:
    """Which transactions an investigations team would actually see."""
    return scores >= alert_threshold(scores, budget)


def alert_share(alerted: np.ndarray) -> float:
    """Fraction of all traffic sitting in the alert queue - the review cost."""
    return float(alerted.mean())


def detection_from_alerts(
    alerted: np.ndarray, labels: np.ndarray, typologies: np.ndarray
) -> list[TypologyRecall]:
    """Per-typology detection for an already-chosen set of alerts."""
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


def detection_by_typology(
    scores: np.ndarray,
    labels: np.ndarray,
    typologies: np.ndarray,
    budget: float = config.ALERT_BUDGET,
) -> list[TypologyRecall]:
    """Share of each typology's true cases that land inside the alert budget."""
    return detection_from_alerts(alert_mask(scores, budget), labels, typologies)


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


@dataclass(frozen=True)
class DeploymentComparison:
    """One bank running its own model, then adding the federated queue alongside.

    This is how a federation is actually deployed: nobody switches off a model
    that works. The bank reviews both queues, so it keeps every case it already
    caught and gains the ones it could never have seen.
    """

    label: str
    own_typology: str
    alone: list[TypologyRecall]
    combined: list[TypologyRecall]
    alert_share_alone: float
    alert_share_combined: float

    def _rate(self, recalls: list[TypologyRecall], typologies: set[str]) -> float:
        return _pooled(recalls, typologies)

    @property
    def _unseen(self) -> set[str]:
        return {r.typology for r in self.alone} - {self.own_typology}

    @property
    def own_alone(self) -> float:
        return self._rate(self.alone, {self.own_typology})

    @property
    def own_combined(self) -> float:
        return self._rate(self.combined, {self.own_typology})

    @property
    def unseen_alone(self) -> float:
        return self._rate(self.alone, self._unseen)

    @property
    def unseen_combined(self) -> float:
        return self._rate(self.combined, self._unseen)

    @property
    def overall_alone(self) -> float:
        return overall(self.alone).rate

    @property
    def overall_combined(self) -> float:
        return overall(self.combined).rate

    @property
    def has_regression(self) -> bool:
        """True if any typology got worse. Adding a queue can never cause this."""
        before = {r.typology: r.rate for r in self.alone}
        return any(r.rate < before[r.typology] - 1e-9 for r in self.combined)


def compare_deployment(
    label: str,
    own_typology: str,
    local_scores: np.ndarray,
    federated_scores: np.ndarray,
    labels: np.ndarray,
    typologies: np.ndarray,
    budget: float = config.ALERT_BUDGET,
) -> DeploymentComparison:
    """Score a bank's own alert queue against that queue plus the federated one."""
    local_alerts = alert_mask(local_scores, budget)
    combined_alerts = local_alerts | alert_mask(federated_scores, budget)
    return DeploymentComparison(
        label=label,
        own_typology=own_typology,
        alone=detection_from_alerts(local_alerts, labels, typologies),
        combined=detection_from_alerts(combined_alerts, labels, typologies),
        alert_share_alone=alert_share(local_alerts),
        alert_share_combined=alert_share(combined_alerts),
    )


def format_deployment_table(rows: list[DeploymentComparison]) -> str:
    """What each bank gains by adding the federated queue, and what it costs."""
    width = max((len(r.label) for r in rows), default=0) + 2
    lines = [
        f"  {'':<{width}}{'its own typology':>22}{'the ones it never saw':>26}{'all cases':>20}",
        "  " + "-" * (width + 68),
    ]
    for r in rows:
        lines.append(
            f"  {r.label:<{width}}"
            f"{_arrow(r.own_alone, r.own_combined):>22}"
            f"{_arrow(r.unseen_alone, r.unseen_combined):>26}"
            f"{_arrow(r.overall_alone, r.overall_combined):>20}"
        )
    cost = ", ".join(
        f"{r.label} {r.alert_share_alone:.1%}->{r.alert_share_combined:.1%}" for r in rows
    )
    lines.append(f"\n  alert volume reviewed: {cost}")
    return "\n".join(lines)


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
