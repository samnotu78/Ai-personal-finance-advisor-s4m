"""
Forecasting models built on monthly aggregates of cleaned+categorized
transactions.

Kept intentionally simple (linear regression over a rolling window) rather
than reaching for XGBoost/Prophet: with 12-24 months of a single account's
history there isn't enough data to responsibly justify a heavier model, and
a transparent trend line is easier for an end user to trust and audit than
a black box. The interface (`fit_expense_forecast` -> object with
`.predict_next_month()`) is designed so a heavier model could be swapped in
without touching the dashboard code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.config import FORECAST_LOOKBACK_MONTHS


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per calendar month: income, expenses, net, savings_rate."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    income = df[df["amount"] > 0].groupby("month")["amount"].sum()
    expenses = df[df["amount"] < 0].groupby("month")["amount"].sum().abs()
    summary = pd.DataFrame({"income": income, "expenses": expenses}).fillna(0.0)
    summary["net"] = summary["income"] - summary["expenses"]
    summary["savings_rate"] = np.where(
        summary["income"] > 0, summary["net"] / summary["income"], np.nan
    )
    summary = summary.reset_index()
    summary["month"] = summary["month"].astype(str)
    return summary


@dataclass
class ForecastResult:
    predicted_next_month_expense: float
    predicted_annual_expense: float
    predicted_next_month_savings: float
    mae: float | None
    mape: float | None
    n_months_used: int


def forecast_expenses(df: pd.DataFrame, lookback: int = FORECAST_LOOKBACK_MONTHS) -> ForecastResult:
    """
    Fit a linear trend over the last `lookback` months of expenses to predict
    next month, then extrapolate to a naive annual estimate. Also backtests
    the trend on held-out recent months (if enough history) to report MAE/MAPE.
    """
    summary = monthly_summary(df)
    summary = summary[(summary["income"] > 0) | (summary["expenses"] > 0)]  # drop empty months
    recent = summary.tail(lookback).reset_index(drop=True)
    n = len(recent)

    if n < 2:
        avg_expense = recent["expenses"].mean() if n else 0.0
        return ForecastResult(avg_expense, avg_expense * 12, 0.0, None, None, n)

    X = np.arange(n).reshape(-1, 1)
    y = recent["expenses"].values

    mae = mape = None
    if n >= 4:
        # Backtest: fit on all but last point, evaluate on last point, roll forward.
        errors, pct_errors = [], []
        for cut in range(3, n):
            model_bt = LinearRegression().fit(X[:cut], y[:cut])
            pred = model_bt.predict(X[cut].reshape(1, -1))[0]
            errors.append(abs(pred - y[cut]))
            if y[cut] != 0:
                pct_errors.append(abs(pred - y[cut]) / y[cut])
        if errors:
            mae = float(np.mean(errors))
            mape = float(np.mean(pct_errors)) if pct_errors else None

    model = LinearRegression().fit(X, y)
    next_month_expense = round(max(0.0, float(model.predict([[n]])[0])), 2)
    annual_expense = round(next_month_expense * 12, 2)

    avg_income = recent["income"].mean()
    next_month_savings = float(avg_income - next_month_expense)

    return ForecastResult(
        predicted_next_month_expense=next_month_expense,
        predicted_annual_expense=annual_expense,
        predicted_next_month_savings=round(next_month_savings, 2),
        mae=round(mae, 2) if mae is not None else None,
        mape=round(mape, 4) if mape is not None else None,
        n_months_used=n,
    )


def category_trend(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Month-by-month total spend for a single category (for trend charts)."""
    cat_df = df[(df["category"] == category) & (df["amount"] < 0)].copy()
    cat_df["month"] = cat_df["date"].dt.to_period("M")
    trend = cat_df.groupby("month")["amount"].sum().abs().reset_index()
    trend["month"] = trend["month"].astype(str)
    return trend
