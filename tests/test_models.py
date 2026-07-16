import numpy as np
import pandas as pd

from src.models.forecast import monthly_summary, forecast_expenses, category_trend
from src.models.anomaly import detect_anomalies, evaluate_anomaly_detection, overspending_categories
from src.config import SYNTHETIC_DATA_PATH
from src.cleaning.clean import clean_transactions
from src.categorization.categorize import categorize_transactions


def _load_sample():
    raw = pd.read_csv(SYNTHETIC_DATA_PATH)
    cleaned, _ = clean_transactions(raw)
    return categorize_transactions(cleaned)


def test_monthly_summary_columns():
    df = _load_sample()
    summary = monthly_summary(df)
    assert {"month", "income", "expenses", "net", "savings_rate"}.issubset(summary.columns)
    assert (summary["expenses"] >= 0).all()


def test_forecast_expenses_returns_positive_estimates():
    df = _load_sample()
    forecast = forecast_expenses(df)
    assert forecast.predicted_next_month_expense >= 0
    assert forecast.predicted_annual_expense == forecast.predicted_next_month_expense * 12
    assert forecast.n_months_used > 0


def test_category_trend_only_expenses():
    df = _load_sample()
    trend = category_trend(df, "Food")
    assert (trend["amount"] >= 0).all()


def test_detect_anomalies_flags_reasonable_fraction():
    df = _load_sample()
    result = detect_anomalies(df)
    assert "predicted_anomaly" in result.columns
    flagged_fraction = result["predicted_anomaly"].mean()
    assert 0 <= flagged_fraction <= 0.15


def test_evaluate_anomaly_detection_against_ground_truth():
    df = _load_sample()
    result = detect_anomalies(df)
    metrics = evaluate_anomaly_detection(result)
    assert metrics is not None
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1


def test_overspending_categories_schema():
    df = _load_sample()
    flagged = overspending_categories(df)
    assert list(flagged.columns) == ["category", "latest_month_spend", "baseline_avg_spend", "pct_change"]
