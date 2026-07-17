"""
Data cleaning pipeline for uploaded bank/card statements.

Design: every step is a pure function (DataFrame in -> DataFrame out) plus a
short log message. `clean_transactions` chains them and returns both the
cleaned DataFrame and the ordered list of log messages, so the dashboard can
show an "upload summary" of exactly what was done to the user's data.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import REQUIRED_COLUMNS

# Common currency symbols stripped when coercing amount to numeric.
_CURRENCY_CHARS = str.maketrans("", "", "$€£,")

# A conservative set of fixed conversion rates to USD, used only if a
# `currency` column is present. In production this would call a live FX API;
# hardcoded here so the pipeline is deterministic and works fully offline.
_FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.73, "INR": 0.012}


class SchemaError(ValueError):
    """Raised when an uploaded file is missing required columns."""


def validate_schema(df: pd.DataFrame) -> None:
    cols_lower = {c.lower().strip() for c in df.columns}
    missing = REQUIRED_COLUMNS - cols_lower
    if missing:
        raise SchemaError(
            f"Uploaded file is missing required column(s): {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}."
        )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/trim column names and map common aliases to canonical names."""
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    aliases = {
        "transaction_date": "date", "posted_date": "date", "txn_date": "date",
        "memo": "description", "narrative": "description", "details": "description",
        "value": "amount", "debit_credit": "amount",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the date column to pandas datetime, dropping unparsable rows."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    return df


def coerce_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure amount is numeric, stripping currency symbols/commas if present."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        df["amount"] = (
            df["amount"].astype(str).str.translate(_CURRENCY_CHARS).str.strip()
        )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def convert_currency(df: pd.DataFrame) -> pd.DataFrame:
    """Convert amount to USD if a `currency` column is present."""
    df = df.copy()
    if "currency" in df.columns:
        rates = df["currency"].str.upper().map(_FX_TO_USD).fillna(1.0)
        df["amount"] = df["amount"] * rates
        df["currency"] = "USD"
    return df


def drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with unparsable dates/amounts, empty descriptions, or zero amount."""
    before = len(df)
    df = df.dropna(subset=["date", "amount"])
    df = df[df["description"].astype(str).str.strip() != ""]
    df = df[df["amount"] != 0]
    return df, before - len(df)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any remaining gaps: blank descriptions -> 'Unknown', category -> 'Miscellaneous'."""
    df = df.copy()
    df["description"] = df["description"].fillna("Unknown").astype(str).str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].fillna("Miscellaneous")
    return df


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Flag/remove near-duplicate transactions: same description, amount within
    1 cent, and date within 1 day of each other (accidental double charges,
    authorization holds that post twice, etc.).
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["_amount_round"] = df["amount"].round(2)
    df["is_duplicate_flagged"] = False

    seen = {}
    for idx, row in df.iterrows():
        key = (row["description"], row["_amount_round"])
        if key in seen:
            prev_date = seen[key]
            if abs((row["date"] - prev_date).days) <= 1:
                df.at[idx, "is_duplicate_flagged"] = True
        seen[key] = row["date"]

    before = len(df)
    df = df[~df["is_duplicate_flagged"]].drop(columns=["_amount_round", "is_duplicate_flagged"])
    return df.reset_index(drop=True), before - len(df)


def clean_transactions(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run the full cleaning pipeline, returning (cleaned_df, log_messages)."""
    log: list[str] = []
    n_start = len(raw_df)
    log.append(f"Loaded {n_start} raw transactions.")

    df = standardize_columns(raw_df)
    validate_schema(df)
    log.append("Validated required columns: date, description, amount.")

    df = parse_dates(df)
    df = coerce_amount(df)
    df = convert_currency(df)
    if "currency" in [c.lower() for c in raw_df.columns]:
        log.append("Converted multi-currency amounts to USD.")

    df, n_invalid = drop_invalid_rows(df)
    log.append(f"Removed {n_invalid} invalid rows (bad date/amount or empty description).")

    df = fill_missing(df)

    df, n_dupes = deduplicate(df)
    log.append(f"Removed {n_dupes} near-duplicate transactions (same merchant/amount within 1 day).")

    n_final = len(df)
    log.append(f"Cleaning complete: {n_start} -> {n_final} transactions ({n_start - n_final} removed).")
    return df, log
