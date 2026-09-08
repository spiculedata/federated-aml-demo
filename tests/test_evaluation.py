"""Alert-budget scoring must be exact - it is the demo's headline number."""

from __future__ import annotations

import numpy as np
import pytest

from fedxgb import evaluation


def test_alert_threshold_selects_the_requested_share():
    scores = np.linspace(0, 1, 1000)

    threshold = evaluation.alert_threshold(scores, budget=0.1)

    assert (scores >= threshold).mean() == pytest.approx(0.1, abs=0.002)


def test_alert_threshold_rejects_a_budget_outside_zero_to_one():
    with pytest.raises(ValueError):
        evaluation.alert_threshold(np.array([0.1, 0.2]), budget=0.0)


def test_detection_counts_only_true_cases_inside_the_budget():
    # Arrange: 10 rows, the top 2 scores are one real case and one false positive
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    typologies = np.array(["mule", "legitimate"] + ["mule"] + ["legitimate"] * 7)

    # Act
    result = evaluation.detection_by_typology(scores, labels, typologies, budget=0.2)

    # Assert
    assert len(result) == 1
    assert result[0].n_cases == 2
    assert result[0].detected == 1
    assert result[0].rate == pytest.approx(0.5)


def test_each_typology_is_reported_separately():
    scores = np.array([0.9, 0.85, 0.1, 0.1])
    labels = np.array([1, 1, 1, 0])
    typologies = np.array(["a", "b", "a", "legitimate"])

    result = evaluation.detection_by_typology(scores, labels, typologies, budget=0.5)

    assert [r.typology for r in result] == ["a", "b"]
    assert {r.typology: r.detected for r in result} == {"a": 1, "b": 1}


def test_table_lists_one_row_per_model_and_one_column_per_typology():
    rows = {
        "solo": [evaluation.TypologyRecall("a", 10, 8), evaluation.TypologyRecall("b", 10, 1)],
        "federated": [evaluation.TypologyRecall("a", 10, 9), evaluation.TypologyRecall("b", 10, 9)],
    }

    table = evaluation.format_detection_table(rows, budget=0.01)

    assert "solo" in table and "federated" in table
    assert "80.0%" in table and "90.0%" in table


def test_overall_pools_cases_across_typologies():
    recalls = [
        evaluation.TypologyRecall("a", 100, 40),
        evaluation.TypologyRecall("b", 100, 60),
    ]

    total = evaluation.overall(recalls)

    assert total.n_cases == 200
    assert total.detected == 100
    assert total.rate == pytest.approx(0.5)


def test_overall_of_no_typologies_is_not_a_number():
    assert evaluation.overall([]).rate != evaluation.overall([]).rate  # nan


def test_table_includes_an_all_cases_column():
    rows = {"solo": [evaluation.TypologyRecall("a", 10, 5)]}

    assert "ALL CASES" in evaluation.format_detection_table(rows, budget=0.02)


def _recalls(**by_typology) -> list[evaluation.TypologyRecall]:
    return [
        evaluation.TypologyRecall(name, cases, detected)
        for name, (cases, detected) in by_typology.items()
    ]


def test_comparison_splits_the_banks_own_typology_from_the_rest():
    alone = _recalls(cross_border=(100, 85), structuring=(100, 1), card_not_present=(100, 2))
    federated = _recalls(cross_border=(100, 49), structuring=(100, 38), card_not_present=(100, 42))

    result = evaluation.compare_models("Meridian", "cross_border", alone, federated)

    assert result.own_alone == pytest.approx(0.85)
    assert result.own_federated == pytest.approx(0.49)
    assert result.unseen_alone == pytest.approx(0.015)
    assert result.unseen_federated == pytest.approx(0.40)


def test_comparison_surfaces_a_regression_on_the_specialist_typology():
    """Joining costs a bank accuracy on the thing it was already good at."""
    alone = _recalls(cross_border=(100, 85), structuring=(100, 0))
    federated = _recalls(cross_border=(100, 49), structuring=(100, 38))

    result = evaluation.compare_models("Meridian", "cross_border", alone, federated)

    assert result.own_delta < 0
    assert result.unseen_delta > 0
    assert result.overall_delta > 0


def test_overall_delta_is_the_headline_and_stays_positive():
    alone = _recalls(a=(100, 60), b=(100, 0), c=(100, 0))
    federated = _recalls(a=(100, 40), b=(100, 40), c=(100, 40))

    result = evaluation.compare_models("Bank", "a", alone, federated)

    assert result.overall_alone == pytest.approx(0.2)
    assert result.overall_federated == pytest.approx(0.4)


def test_comparison_table_shows_before_and_after_for_each_bank():
    alone = _recalls(a=(100, 60), b=(100, 0))
    federated = _recalls(a=(100, 40), b=(100, 40))
    rows = [evaluation.compare_models("Bank One", "a", alone, federated)]

    table = evaluation.format_comparison_table(rows)

    assert "Bank One" in table
    assert "60.0% -> 40.0%" in table
    assert "its own typology" in table
