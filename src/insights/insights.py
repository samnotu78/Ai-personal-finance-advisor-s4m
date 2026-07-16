"""
Turns numeric analysis (monthly summaries, category trends, forecasts,
overspending flags) into short, human-readable insight strings for the
dashboard's "AI Financial Insights" panel.

Every function returns a list[str] of complete sentences so the dashboard
can just concatenate/bullet them; no function raises on missing data --
they return an empty list instead, so the panel gracefully shows fewer
insights for short histories rather than erroring.
"""
from __future__ import annotations

import pandas as pd

from src.models.forecast import monthly_summary, category_trend, ForecastResult
from src.models.anomaly import overspending_categories


def category_mom_change(df: pd.DataFrame, min_months: int = 2, top_n: int = 3) -> list[str]:
    """Return sentences for the `top_n` categories with the largest MoM % swing."""
    candidates = []
    for category in df.loc[df["amount"] < 0, "category"].unique():
        trend = category_trend(df, category)
        if len(trend) < min_months:
            continue
        prev, latest = trend["amount"].iloc[-2], trend["amount"].iloc[-1]
        if prev == 0:
            continue
        pct = (latest - prev) / prev * 100
        if abs(pct) >= 15:
            candidates.append((category, pct))

    candidates.sort(key=lambda c: abs(c[1]), reverse=True)
    insights = []
    for category, pct in candidates[:top_n]:
        direction = "increased" if pct > 0 else "decreased"
        insights.append(f"Your {category.lower()} expenses {direction} by {abs(pct):.0f}% last month.")
    return insights


def largest_category_insight(df: pd.DataFrame) -> list[str]:
    expenses = df[df["amount"] < 0]
    if expenses.empty:
        return []
    totals = expenses.groupby("category")["amount"].sum().abs().sort_values(ascending=False)
    top_category = totals.index[0]
    share = totals.iloc[0] / totals.sum() * 100
    return [f"{top_category} is your largest expense category, making up {share:.0f}% of total spending."]


def budget_projection_insight(forecast: ForecastResult) -> list[str]:
    insights = []
    if forecast.n_months_used >= 2:
        insights.append(
            f"Based on recent trends, next month's expenses are projected at "
            f"${forecast.predicted_next_month_expense:,.0f}."
        )
        if forecast.predicted_next_month_savings < 0:
            insights.append(
                "You are on track to spend more than you earn next month "
                "if current trends continue."
            )
    return insights


def overspending_insight(df: pd.DataFrame) -> list[str]:
    flagged = overspending_categories(df)
    insights = []
    for _, row in flagged.iterrows():
        insights.append(
            f"{row['category']} spending is up {row['pct_change']*100:.0f}% "
            f"over your recent average (${row['latest_month_spend']:,.0f} vs "
            f"${row['baseline_avg_spend']:,.0f})."
        )
    return insights


def subscription_insight(df: pd.DataFrame, min_recurring_merchants: int = 3) -> list[str]:
    """Flag if the user has several small, frequent recurring charges (subscriptions)."""
    expenses = df[df["amount"] < 0].copy()
    counts = expenses.groupby("description")["amount"].agg(["count", "mean"])
    recurring = counts[(counts["count"] >= 3) & (counts["mean"].abs() < 60)]
    if len(recurring) >= min_recurring_merchants:
        monthly_cost = recurring["mean"].abs().sum()
        return [
            f"You have {len(recurring)} recurring small charges (subscriptions/memberships) "
            f"totaling roughly ${monthly_cost:,.0f}/month -- reviewing these could increase savings."
        ]
    return []


def savings_rate_insight(df: pd.DataFrame) -> list[str]:
    summary = monthly_summary(df)
    valid = summary.dropna(subset=["savings_rate"])
    if valid.empty:
        return []
    avg_rate = valid["savings_rate"].tail(3).mean() * 100
    if avg_rate < 0:
        return ["Your average savings rate over the last 3 months is negative -- expenses are exceeding income."]
    elif avg_rate < 10:
        return [f"Your average savings rate is {avg_rate:.0f}%, below the commonly recommended 20% target."]
    else:
        return [f"Your average savings rate is a healthy {avg_rate:.0f}% over the last 3 months."]


def generate_all_insights(df: pd.DataFrame, forecast: ForecastResult) -> list[str]:
    insights = []
    insights += largest_category_insight(df)
    insights += category_mom_change(df)
    insights += overspending_insight(df)
    insights += subscription_insight(df)
    insights += savings_rate_insight(df)
    insights += budget_projection_insight(forecast)
    return insights
