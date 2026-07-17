import pandas as pd

from src.config import SYNTHETIC_DATA_PATH
from src.cleaning.clean import clean_transactions
from src.categorization.categorize import categorize_transactions
from src.models.forecast import forecast_expenses
from src.insights.insights import generate_all_insights, largest_category_insight, savings_rate_insight


def _load_sample():
    raw = pd.read_csv(SYNTHETIC_DATA_PATH)
    cleaned, _ = clean_transactions(raw)
    return categorize_transactions(cleaned)


def test_largest_category_insight_mentions_a_real_category():
    df = _load_sample()
    insights = largest_category_insight(df)
    assert len(insights) == 1
    assert "largest expense category" in insights[0]


def test_savings_rate_insight_returns_sentence():
    df = _load_sample()
    insights = savings_rate_insight(df)
    assert len(insights) <= 1
    if insights:
        assert "savings rate" in insights[0]


def test_generate_all_insights_returns_strings():
    df = _load_sample()
    forecast = forecast_expenses(df)
    insights = generate_all_insights(df, forecast)
    assert isinstance(insights, list)
    assert all(isinstance(i, str) and len(i) > 0 for i in insights)
