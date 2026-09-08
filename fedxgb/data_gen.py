"""Synthetic transaction generator.

Produces one Parquet file per bank plus a central hold-out file. Every bank
sees the same benign traffic distribution but a *different* fraud typology,
which is what makes the federation worth joining.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import polars as pl

from fedxgb import config

_DAYS_IN_WINDOW = 30
_BENIGN_ACCOUNT_POOL = 40_000
_FRAUD_ACCOUNT_POOL = 400
_BASE_FRAUD_RATE = 0.002

# Real criminals are inconsistent and real customers are odd. Only this share
# of a typology's signature markers actually shows up on a given fraud row...
_SIGNATURE_STRENGTH = 0.55
# ...and this share of ordinary customers coincidentally look like a typology.
_LOOKALIKE_RATE = 0.05
_LOOKALIKE_STRENGTH = 0.45


def _benign_rows(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Ordinary retail and commercial traffic."""
    return {
        "account_id": rng.integers(0, _BENIGN_ACCOUNT_POOL, n),
        "day": rng.integers(0, _DAYS_IN_WINDOW, n),
        "hour_of_day": rng.integers(6, 23, n),
        "amount": np.round(rng.lognormal(4.2, 1.3, n), 2),
        "channel": rng.choice(config.CHANNELS, n, p=[0.1, 0.15, 0.3, 0.25, 0.2]),
        "counterparty_country": rng.choice(
            config.ALL_COUNTRIES, n, p=[0.55, 0.12, 0.1, 0.08, 0.07, 0.02, 0.02, 0.02, 0.02]
        ),
        "account_age_days": rng.integers(30, 4000, n),
        "is_new_counterparty": (rng.random(n) < 0.15).astype(np.int8),
    }


def _structuring_rows(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Deposits deliberately kept just below the reporting threshold."""
    return {
        "account_id": rng.integers(0, _FRAUD_ACCOUNT_POOL, n),
        "day": rng.integers(0, _DAYS_IN_WINDOW, n),
        "hour_of_day": rng.integers(9, 18, n),
        "amount": np.round(rng.uniform(8_400, 9_960, n), 2),
        "channel": rng.choice(("branch", "atm"), n),
        "counterparty_country": rng.choice(("GB", "US"), n, p=[0.9, 0.1]),
        "account_age_days": rng.integers(200, 3000, n),
        "is_new_counterparty": (rng.random(n) < 0.1).astype(np.int8),
    }


def _card_not_present_rows(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Overnight online spend to counterparties never seen before."""
    return {
        "account_id": rng.integers(0, _BENIGN_ACCOUNT_POOL, n),
        "day": rng.integers(0, _DAYS_IN_WINDOW, n),
        "hour_of_day": rng.choice(np.array([23, 0, 1, 2, 3, 4]), n),
        "amount": np.round(rng.uniform(120, 2_400, n), 2),
        "channel": np.full(n, "online"),
        "counterparty_country": rng.choice(("GB", "US", "IE"), n),
        "account_age_days": rng.integers(60, 4000, n),
        "is_new_counterparty": np.ones(n, dtype=np.int8),
    }


def _cross_border_rows(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Large outbound payments from young accounts to high-risk jurisdictions."""
    return {
        "account_id": rng.integers(0, _BENIGN_ACCOUNT_POOL, n),
        "day": rng.integers(0, _DAYS_IN_WINDOW, n),
        "hour_of_day": rng.integers(8, 20, n),
        "amount": np.round(rng.uniform(15_000, 90_000, n), 2),
        "channel": rng.choice(("online", "branch"), n),
        "counterparty_country": rng.choice(config.HIGH_RISK_COUNTRIES, n),
        "account_age_days": rng.integers(5, 90, n),
        "is_new_counterparty": (rng.random(n) < 0.8).astype(np.int8),
    }


def _blend(
    rng: np.random.Generator,
    signature: dict[str, np.ndarray],
    benign: dict[str, np.ndarray],
    strength: float,
) -> dict[str, np.ndarray]:
    """Apply each signature marker to only a random subset of rows.

    Without this the typologies are trivially separable and every model - solo
    or federated - scores a perfect AUC, which proves nothing.
    """
    return {
        column: np.where(rng.random(len(values)) < strength, values, benign[column])
        for column, values in signature.items()
    }


_TYPOLOGY_GENERATORS = {
    "structuring": _structuring_rows,
    "card_not_present": _card_not_present_rows,
    "cross_border": _cross_border_rows,
}


def _to_frame(parts: list[tuple[dict[str, np.ndarray], int, str]], prefix: str) -> pl.DataFrame:
    """Concatenate labelled blocks into a single shuffled transaction frame.

    ``typology`` is reporting metadata only - it is never fed to the model.
    """
    blocks = [
        pl.DataFrame(cols).with_columns(
            pl.lit(label, dtype=pl.Int8).alias(config.LABEL_COLUMN),
            pl.lit(typology, dtype=pl.String).alias("typology"),
        )
        for cols, label, typology in parts
    ]
    frame = pl.concat(blocks, how="vertical")
    return (
        frame.sample(fraction=1.0, shuffle=True, seed=7)
        .with_row_index("row_no")
        .with_columns(
            (pl.lit(prefix) + pl.col("row_no").cast(pl.String)).alias("transaction_id"),
            (pl.lit("ACC-") + pl.col("account_id").cast(pl.String)).alias("account_id"),
            pl.date(2025, 1, 1).dt.offset_by(pl.format("{}d", pl.col("day"))).alias("event_date"),
        )
        .drop("row_no", "day")
    )


def _typology_block(
    rng: np.random.Generator, typology: str, n: int, strength: float
) -> dict[str, np.ndarray]:
    """A block of rows carrying a partially-expressed typology signature."""
    return _blend(rng, _TYPOLOGY_GENERATORS[typology](rng, n), _benign_rows(rng, n), strength)


def _lookalike_blocks(
    rng: np.random.Generator, n_benign: int, typologies: Sequence[str]
) -> list[tuple[dict[str, np.ndarray], int, str]]:
    """Legitimate customers who happen to trip the same markers as criminals."""
    per_typology = int(n_benign * _LOOKALIKE_RATE / len(typologies))
    return [
        (_typology_block(rng, typology, per_typology, _LOOKALIKE_STRENGTH), 0, config.NO_TYPOLOGY)
        for typology in typologies
    ]


def generate_bank_file(profile: config.BankProfile, out_dir: str) -> str:
    """Write one bank's private transaction ledger. Never leaves the bank."""
    rng = np.random.default_rng(profile.seed)
    n_fraud = int(profile.rows * _BASE_FRAUD_RATE)
    n_benign = profile.rows - n_fraud
    parts: list[tuple[dict[str, np.ndarray], int, str]] = [
        (_benign_rows(rng, n_benign), 0, config.NO_TYPOLOGY),
        (_typology_block(rng, profile.typology, n_fraud, _SIGNATURE_STRENGTH), 1, profile.typology),
    ]
    # Every bank sees confusing-but-innocent traffic of all three shapes.
    parts += _lookalike_blocks(rng, n_benign, tuple(_TYPOLOGY_GENERATORS))
    path = os.path.join(out_dir, f"{profile.bank_id}.parquet")
    _to_frame(parts, f"{profile.bank_id.upper()}-").write_parquet(path)
    return path


def generate_holdout_file(out_dir: str, rows: int = config.HOLDOUT_ROWS) -> str:
    """A regulator-style evaluation set containing *all* known typologies."""
    rng = np.random.default_rng(9_999)
    per_typology = int(rows * _BASE_FRAUD_RATE)
    n_benign = rows - per_typology * len(_TYPOLOGY_GENERATORS)
    parts: list[tuple[dict[str, np.ndarray], int, str]] = [
        (_benign_rows(rng, n_benign), 0, config.NO_TYPOLOGY)
    ]
    parts += [
        (_typology_block(rng, typology, per_typology, _SIGNATURE_STRENGTH), 1, typology)
        for typology in _TYPOLOGY_GENERATORS
    ]
    parts += _lookalike_blocks(rng, n_benign, tuple(_TYPOLOGY_GENERATORS))
    path = os.path.join(out_dir, "central_holdout.parquet")
    _to_frame(parts, "HOLDOUT-").write_parquet(path)
    return path


def build_all(out_dir: str = config.DATA_DIR) -> dict[str, str]:
    """Materialise every bank ledger plus the central hold-out set."""
    os.makedirs(out_dir, exist_ok=True)
    profiles = [
        config.BankProfile(bank_id, config.BANK_TYPOLOGIES[bank_id], config.ROWS_PER_BANK, seed)
        for seed, bank_id in enumerate(config.BANK_IDS, start=1)
    ]
    paths = {p.bank_id: generate_bank_file(p, out_dir) for p in profiles}
    return {**paths, "holdout": generate_holdout_file(out_dir)}


if __name__ == "__main__":
    for name, p in build_all().items():
        print(f"{name:>18}  {p}")
