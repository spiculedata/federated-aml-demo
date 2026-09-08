"""Static configuration for the federated transaction-monitoring demo."""

from __future__ import annotations

from dataclasses import dataclass

# --- Federation topology -----------------------------------------------------

BANK_IDS: tuple[str, ...] = ("northwind_bank", "caledonia_trust", "meridian_pcb")

# Each participant sees a different slice of the criminal landscape. This is the
# whole point of the demo: alone, nobody has enough signal.
BANK_TYPOLOGIES: dict[str, str] = {
    "northwind_bank": "structuring",
    "caledonia_trust": "card_not_present",
    "meridian_pcb": "cross_border",
}

ROWS_PER_BANK = 400_000
HOLDOUT_ROWS = 60_000

# --- Federated training schedule ---------------------------------------------

# Gradient-boosted ranking converges in a handful of rounds on a problem this
# shaped; the rounds are here to exercise the protocol, not because the model
# needs dozens of them.
NUM_ROUNDS = 8
TREES_PER_BANK_PER_ROUND = 2

XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "eta": 0.12,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "base_score": 0.5,
    "tree_method": "hist",
    "nthread": 4,
    "scale_pos_weight": 20.0,
}

# --- Domain constants --------------------------------------------------------

REPORTING_THRESHOLD = 10_000.0
NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 5

HIGH_RISK_COUNTRIES: tuple[str, ...] = ("PA", "CY", "AE", "SC")
ALL_COUNTRIES: tuple[str, ...] = ("GB", "US", "DE", "FR", "IE", *HIGH_RISK_COUNTRIES)
CHANNELS: tuple[str, ...] = ("branch", "atm", "online", "mobile", "card_present")

FEATURE_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "amount_to_threshold_ratio",
    "is_just_under_threshold",
    "hour_of_day",
    "is_night",
    "is_high_risk_country",
    "is_cross_border",
    "channel_is_online",
    "channel_is_card_present",
    "account_age_days",
    "is_new_counterparty",
    "daily_txn_count",
    "daily_amount_total",
    "amount_vs_account_mean",
)
LABEL_COLUMN = "is_suspicious"
TYPOLOGY_COLUMN = "typology"
NO_TYPOLOGY = "legitimate"

# Transaction monitoring teams work to a fixed alert capacity, so detection
# rate at a realistic review budget matters more than raw AUC.
ALERT_BUDGET = 0.02

# --- Paths -------------------------------------------------------------------

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"


@dataclass(frozen=True)
class BankProfile:
    """Immutable description of one participating institution."""

    bank_id: str
    typology: str
    rows: int
    seed: int
