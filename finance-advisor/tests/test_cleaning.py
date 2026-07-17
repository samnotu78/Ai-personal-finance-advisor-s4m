import pandas as pd
import pytest

from src.cleaning.clean import clean_transactions, SchemaError


def _raw_df():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "not-a-date", "2024-01-03"],
        "Description": ["STARBUCKS #04521", "STARBUCKS #04521", "TARGET 00021453", "BAD ROW", ""],
        "Amount": ["$5.25", "5.25", "-45.00", "10.00", "12.00"],
    })


def test_missing_required_columns_raises():
    bad_df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(SchemaError):
        clean_transactions(bad_df)


def test_clean_transactions_removes_invalid_and_duplicates():
    cleaned, log = clean_transactions(_raw_df())
    # 1 invalid date row + 1 empty-description row removed, 1 near-duplicate removed
    assert len(cleaned) == 2
    assert cleaned["amount"].dtype.kind in "fi"
    assert any("duplicate" in line.lower() for line in log)
    assert any("invalid" in line.lower() for line in log)


def test_currency_symbols_stripped():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "description": ["TEST MERCHANT"],
        "amount": ["$1,234.50"],
    })
    cleaned, _ = clean_transactions(df)
    assert cleaned["amount"].iloc[0] == pytest.approx(1234.50)


def test_column_aliases_recognized():
    df = pd.DataFrame({
        "Transaction Date": ["2024-01-01"],
        "Memo": ["TEST MERCHANT"],
        "Value": [15.0],
    })
    cleaned, _ = clean_transactions(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["description"] == "TEST MERCHANT"
