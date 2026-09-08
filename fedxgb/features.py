"""Polars streaming feature pipeline.

Every bank runs this over its own ledger. It is written entirely as a
LazyFrame plan and executed with Polars' streaming engine, so a participant
with a 50 GB ledger and 8 GB of RAM runs exactly the same code as the demo.
"""

from __future__ import annotations

import polars as pl

from fedxgb import config

_ACCOUNT_KEY = "account_id"
_DAY_KEY = "event_date"


def _daily_account_aggregates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Velocity signals: how busy was this account on this day?"""
    return lf.group_by(_ACCOUNT_KEY, _DAY_KEY).agg(
        pl.len().alias("daily_txn_count"),
        pl.col("amount").sum().alias("daily_amount_total"),
    )


def _account_baselines(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Behavioural baseline: what does normal look like for this account?"""
    return lf.group_by(_ACCOUNT_KEY).agg(
        pl.col("amount").mean().alias("account_mean_amount"),
    )


def _row_level_expressions() -> list[pl.Expr]:
    """Cheap per-transaction derivations, computed inside the stream."""
    amount = pl.col("amount")
    hour = pl.col("hour_of_day")
    return [
        amount.log1p().alias("log_amount"),
        (amount / config.REPORTING_THRESHOLD).alias("amount_to_threshold_ratio"),
        (amount.is_between(config.REPORTING_THRESHOLD * 0.8, config.REPORTING_THRESHOLD))
        .cast(pl.Int8)
        .alias("is_just_under_threshold"),
        ((hour >= config.NIGHT_START_HOUR) | (hour <= config.NIGHT_END_HOUR))
        .cast(pl.Int8)
        .alias("is_night"),
        pl.col("counterparty_country")
        .is_in(config.HIGH_RISK_COUNTRIES)
        .cast(pl.Int8)
        .alias("is_high_risk_country"),
        (pl.col("counterparty_country") != pl.lit("GB")).cast(pl.Int8).alias("is_cross_border"),
        (pl.col("channel") == pl.lit("online")).cast(pl.Int8).alias("channel_is_online"),
        (pl.col("channel") == pl.lit("card_present")).cast(pl.Int8).alias("channel_is_card_present"),
    ]


def build_feature_plan(
    source: str | pl.LazyFrame, extra_columns: tuple[str, ...] = ()
) -> pl.LazyFrame:
    """Return the lazy plan turning a raw ledger into the model's feature matrix.

    ``extra_columns`` are carried through for reporting only; they are never
    part of ``config.FEATURE_COLUMNS`` and so never reach the model.
    Pure and side-effect free: nothing is read until the plan is collected.
    """
    lf = pl.scan_parquet(source) if isinstance(source, str) else source
    daily = _daily_account_aggregates(lf)
    baselines = _account_baselines(lf)

    return (
        lf.with_columns(_row_level_expressions())
        .join(daily, on=[_ACCOUNT_KEY, _DAY_KEY], how="left")
        .join(baselines, on=_ACCOUNT_KEY, how="left")
        .with_columns(
            (pl.col("amount") / pl.col("account_mean_amount").clip(lower_bound=1.0))
            .alias("amount_vs_account_mean")
        )
        .select(*config.FEATURE_COLUMNS, config.LABEL_COLUMN, *extra_columns)
    )


def collect_streaming(plan: pl.LazyFrame) -> pl.DataFrame:
    """Execute a feature plan through Polars' out-of-core streaming engine."""
    return plan.collect(engine="streaming")


def explain_plan(source: str) -> str:
    """Human-readable physical plan, for showing what streaming actually does."""
    return build_feature_plan(source).explain(engine="streaming")
