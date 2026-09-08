"""The streaming feature pipeline must be schema-stable and leak-free."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from fedxgb import config, features


@pytest.fixture
def ledger() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "account_id": ["ACC-1", "ACC-1", "ACC-2", "ACC-2"],
            "event_date": [
                date(2025, 1, 1),
                date(2025, 1, 1),
                date(2025, 1, 1),
                date(2025, 1, 2),
            ],
            "hour_of_day": [2, 14, 12, 23],
            "amount": [100.0, 9_500.0, 50_000.0, 200.0],
            "channel": ["online", "branch", "online", "card_present"],
            "counterparty_country": ["GB", "GB", "PA", "GB"],
            "account_age_days": [900, 900, 20, 20],
            "is_new_counterparty": [0, 0, 1, 0],
            config.LABEL_COLUMN: [0, 1, 1, 0],
        }
    ).with_columns(pl.col("event_date").cast(pl.Date))


def test_plan_emits_exactly_the_declared_feature_columns(ledger):
    result = features.collect_streaming(features.build_feature_plan(ledger))

    assert result.columns == [*config.FEATURE_COLUMNS, config.LABEL_COLUMN]


def test_daily_velocity_counts_transactions_per_account_per_day(ledger):
    result = features.collect_streaming(
        features.build_feature_plan(ledger)
    ).sort("log_amount")

    # ACC-1 transacted twice on the same day; ACC-2 once per day.
    assert sorted(result["daily_txn_count"].to_list()) == [1, 1, 2, 2]


def test_just_under_threshold_flags_structuring_amounts(ledger):
    result = features.collect_streaming(features.build_feature_plan(ledger))

    flagged = result.filter(pl.col("is_just_under_threshold") == 1)
    assert len(flagged) == 1
    assert flagged["amount_to_threshold_ratio"][0] == pytest.approx(0.95)


def test_night_flag_wraps_around_midnight(ledger):
    result = features.collect_streaming(features.build_feature_plan(ledger))

    assert result["is_night"].sum() == 2  # 02:00 and 23:00


def test_high_risk_country_is_distinct_from_cross_border(ledger):
    result = features.collect_streaming(features.build_feature_plan(ledger))

    assert result["is_high_risk_country"].sum() == 1
    assert result["is_cross_border"].sum() == 1


def test_build_feature_plan_is_lazy_and_does_no_work_until_collected(ledger):
    plan = features.build_feature_plan(ledger)

    assert isinstance(plan, pl.LazyFrame)


def test_amount_vs_account_mean_is_finite_for_tiny_balances(ledger):
    result = features.collect_streaming(features.build_feature_plan(ledger))

    assert result["amount_vs_account_mean"].is_finite().all()
