"""
Unusual-transaction / overspending-trend detection.

Uses an IsolationForest over a small, interpretable feature set (amount,
z-score of amount within its own category, day-of-week) so flagged
transactions can be explained ("this was 8x your usual grocery spend")
rather than just labeled "anomaly" with no reason given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

from src.config import ANOMALY_CONTAMINATION, RANDOM_SEED


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()
    expenses["day_of_week"] = expenses["date"].dt.dayofweek

    cat_stats = expenses.groupby("category")["abs_amount"].agg(["mean", "std"]).fillna(0)
    expenses = expenses.join(cat_stats, on="category", rsuffix="_cat")
    expenses["cat_zscore"] = np.where(
        expenses["std"] > 0,
        (expenses["abs_amount"] - expenses["mean"]) / expenses["std"],
        0.0,
    )
    return expenses


def detect_anomalies(df: pd.DataFrame, contamination: float = ANOMALY_CONTAMINATION) -> pd.DataFrame:
    """
    Return the expense rows of `df` with an added `predicted_anomaly` bool
    column. Non-expense rows (income/investment) are excluded from scoring.
    """
    expenses = _build_features(df)
    if len(expenses) < 20:
        expenses["predicted_anomaly"] = False
        return expenses

    features = expenses[["abs_amount", "cat_zscore", "day_of_week"]]
    model = IsolationForest(
        contamination=contamination, random_state=RANDOM_SEED, n_estimators=200
    )
    labels = model.fit_predict(features)  # -1 = anomaly, 1 = normal
    expenses["predicted_anomaly"] = labels == -1
    return expenses


def evaluate_anomaly_detection(expenses_with_predictions: pd.DataFrame) -> dict | None:
    """
    If ground-truth `is_anomaly` labels are present (synthetic data only),
    compute precision/recall against the model's predictions.
    """
    if "is_anomaly" not in expenses_with_predictions.columns:
        return None
    y_true = expenses_with_predictions["is_anomaly"].astype(bool)
    y_pred = expenses_with_predictions["predicted_anomaly"].astype(bool)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "n_flagged": int(y_pred.sum()),
        "n_true_anomalies": int(y_true.sum()),
    }


def overspending_categories(df: pd.DataFrame, threshold_pct: float = 0.15) -> pd.DataFrame:
    """
    Compare each category's most recent month of spend to its trailing
    3-month average; flag categories running > threshold_pct over trend.
    """
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df[df["amount"] < 0]
        .groupby(["category", "month"])["amount"].sum().abs().reset_index()
    )
    results = []
    for category, g in monthly.groupby("category"):
        g = g.sort_values("month")
        if len(g) < 2:
            continue
        latest = g["amount"].iloc[-1]
        baseline = g["amount"].iloc[:-1].tail(3).mean()
        if baseline > 0:
            pct_change = (latest - baseline) / baseline
            if pct_change > threshold_pct:
                results.append({
                    "category": category,
                    "latest_month_spend": round(latest, 2),
                    "baseline_avg_spend": round(baseline, 2),
                    "pct_change": round(pct_change, 3),
                })
    return pd.DataFrame(results).sort_values("pct_change", ascending=False) if results else pd.DataFrame(
        columns=["category", "latest_month_spend", "baseline_avg_spend", "pct_change"]
    )
