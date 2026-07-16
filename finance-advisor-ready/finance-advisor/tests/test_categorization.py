import pandas as pd

from src.categorization.categorize import rule_based_category, categorize_transactions


def test_rule_based_category_known_merchant():
    assert rule_based_category("STARBUCKS #04521") == "Food"
    assert rule_based_category("sq *starbucks") == "Food"


def test_rule_based_category_keyword_fallback():
    assert rule_based_category("RENT - PROPERTY MGMT CO") == "Bills"
    assert rule_based_category("ACME UNIVERSITY TUITION") == "Education"


def test_rule_based_category_unknown_returns_none():
    assert rule_based_category("XZQW RANDOM MERCHANT 99912") is None


def test_categorize_transactions_end_to_end():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "description": ["STARBUCKS #04521", "TARGET 00021453", "ACME CORP PAYROLL DIRECT DEP", "UNKNOWN MERCHANT XJ22"],
        "amount": [-5.25, -45.0, 3200.0, -30.0],
    })
    result = categorize_transactions(df)
    assert result.loc[0, "category"] == "Food"
    assert result.loc[1, "category"] == "Shopping"
    assert result.loc[2, "category"] == "Salary"
    assert "category" in result.columns and result["category"].notna().all()
