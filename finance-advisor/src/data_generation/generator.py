"""
Synthetic bank/card transaction generator.

Simulates ~2 years of a single account's activity with:
  - Biweekly salary deposits (with occasional bonus months)
  - Fixed recurring subscriptions/bills (same day each month, small jitter)
  - Seasonally-weighted discretionary spending across categories
  - Inconsistent merchant naming (multiple raw-string variants per merchant,
    the way real statements show POS prefixes / store numbers / abbreviations)
  - Injected near-duplicate transactions (accidental double charges / holds)
    to give the cleaning module something real to deduplicate, and injected
    injected anomalies to give the anomaly-detection model something to catch.

Ground truth columns (category, is_duplicate, is_anomaly) are included so
they can be used to evaluate the ML pipeline. In a real product these
wouldn't exist in raw bank data -- clearly documented in the README as
synthetic-only fields.
"""
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import string

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RANDOM_SEED, SYNTHETIC_DATA_PATH
from data_generation.merchants import MERCHANT_CATALOG, RECURRING_SUBSCRIPTIONS, SEASONAL_MULTIPLIERS


def _fake_swift() -> str:
    """Lightweight stand-in for Faker's swift() - avoids adding a dependency
    for a single cosmetic field."""
    letters = "".join(random.choices(string.ascii_uppercase, k=6))
    return f"{letters}{random.choice(['XXX', 'US1', 'GB2', 'DE3'])}"


def _seasonal_weight(category: str, month: int) -> float:
    return SEASONAL_MULTIPLIERS.get(category, {}).get(month, 1.0)


def _pick_category_weights(month: int) -> dict:
    """Discretionary categories only (excludes Salary/Investment/Bills, which
    are handled as recurring/scheduled items)."""
    discretionary = ["Food", "Shopping", "Travel", "Entertainment", "Healthcare", "Education", "Miscellaneous"]
    base_weights = {
        "Food": 0.32, "Shopping": 0.22, "Travel": 0.06, "Entertainment": 0.14,
        "Healthcare": 0.08, "Education": 0.03, "Miscellaneous": 0.15,
    }
    weights = {c: base_weights[c] * _seasonal_weight(c, month) for c in discretionary}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def _random_merchant(category: str):
    profile = random.choice(MERCHANT_CATALOG[category])
    description = random.choice(profile["variants"])
    lo, hi = profile["amount_range"]
    amount = round(random.uniform(lo, hi), 2)
    return description, amount


def generate_salary_transactions(start: date, end: date, base_salary: float = 3200.0) -> list:
    """Biweekly salary deposits, with a ~1/12 chance per pay period of a bonus."""
    txns = []
    current = start
    # anchor to a Friday
    while current.weekday() != 4:
        current += timedelta(days=1)
    while current <= end:
        amount = base_salary + np.random.normal(0, 40)
        if random.random() < 0.08:
            amount += random.uniform(300, 1500)  # occasional bonus
        txns.append({
            "date": current,
            "description": "ACME CORP PAYROLL DIRECT DEP" if random.random() > 0.5 else "ACME CORP DIRECT DEPOSIT",
            "amount": round(amount, 2),
            "category": "Salary",
            "is_duplicate": False,
            "is_anomaly": False,
        })
        current += timedelta(days=14)
    return txns


def generate_recurring_transactions(start: date, end: date) -> list:
    """Fixed monthly subscriptions/bills, same day each month with small jitter."""
    txns = []
    current = date(start.year, start.month, 1)
    while current <= end:
        for sub in RECURRING_SUBSCRIPTIONS:
            day = min(sub["day"], 28)
            jitter = random.randint(-1, 1)
            try:
                txn_date = date(current.year, current.month, day) + timedelta(days=jitter)
            except ValueError:
                continue
            if txn_date < start or txn_date > end:
                continue
            amount = round(sub["amount"] * np.random.normal(1.0, 0.01), 2)
            txns.append({
                "date": txn_date,
                "description": sub["variant"],
                "amount": amount,
                "category": sub["category"],
                "is_duplicate": False,
                "is_anomaly": False,
            })
        # advance one month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return txns


def generate_investment_transactions(start: date, end: date) -> list:
    """Monthly automated investment contribution."""
    txns = []
    current = date(start.year, start.month, 5)
    while current <= end:
        if current >= start:
            amount = round(random.uniform(200, 600), 2)
            merchant = random.choice(MERCHANT_CATALOG["Investment"])
            description = random.choice(merchant["variants"])
            txns.append({
                "date": current,
                "description": description,
                "amount": amount,
                "category": "Investment",
                "is_duplicate": False,
                "is_anomaly": False,
            })
        if current.month == 12:
            current = date(current.year + 1, 1, 5)
        else:
            current = date(current.year, current.month + 1, 5)
    return txns


def generate_discretionary_transactions(start: date, end: date, avg_per_day: float = 1.3) -> list:
    """Day-by-day discretionary spending, seasonally weighted by category."""
    txns = []
    current = start
    while current <= end:
        n_today = np.random.poisson(avg_per_day)
        weights = _pick_category_weights(current.month)
        categories = list(weights.keys())
        probs = list(weights.values())
        for _ in range(n_today):
            category = np.random.choice(categories, p=probs)
            description, amount = _random_merchant(category)
            txns.append({
                "date": current,
                "description": description,
                "amount": amount,
                "category": category,
                "is_duplicate": False,
                "is_anomaly": False,
            })
        current += timedelta(days=1)
    return txns


def inject_duplicates(txns: list, rate: float = 0.015) -> list:
    """Randomly duplicate ~rate fraction of expense transactions within 0-1
    days, simulating accidental double-charges/authorization holds."""
    eligible = [t for t in txns if t["category"] not in ("Salary", "Investment")]
    n_dupes = int(len(eligible) * rate)
    new_txns = []
    for original in random.sample(eligible, min(n_dupes, len(eligible))):
        dupe = original.copy()
        dupe["date"] = original["date"] + timedelta(days=random.choice([0, 0, 1]))
        # near-duplicate: sometimes off by a cent due to tip/rounding
        dupe["amount"] = round(original["amount"] + random.choice([0.0, 0.0, 0.01, -0.01]), 2)
        dupe["is_duplicate"] = True
        new_txns.append(dupe)
    return txns + new_txns


def inject_anomalies(txns: list, n_anomalies: int = 25) -> list:
    """Inject a handful of genuinely unusual transactions: large one-off
    purchases, odd-hour-pattern-breaking amounts, category/amount mismatches.
    Used as ground truth for evaluating the anomaly detector."""
    expense_idxs = [i for i, t in enumerate(txns) if t["category"] not in ("Salary", "Investment")]
    chosen = random.sample(expense_idxs, min(n_anomalies, len(expense_idxs)))
    for i in chosen:
        t = txns[i]
        spike_type = random.choice(["amount_spike", "rare_merchant"])
        if spike_type == "amount_spike":
            t["amount"] = round(t["amount"] * random.uniform(6, 15), 2)
        else:
            t["description"] = f"WIRE TRANSFER INTL {_fake_swift()}"
            t["amount"] = round(random.uniform(800, 5000), 2)
            t["category"] = "Miscellaneous"
        t["is_anomaly"] = True
    return txns


def generate_dataset(
    start: date = date(2024, 1, 1),
    end: date = date(2025, 12, 31),
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    txns = []
    txns += generate_salary_transactions(start, end)
    txns += generate_recurring_transactions(start, end)
    txns += generate_investment_transactions(start, end)
    txns += generate_discretionary_transactions(start, end)

    txns = inject_duplicates(txns)
    txns = inject_anomalies(txns)

    df = pd.DataFrame(txns)
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TXN{100000 + i}" for i in range(len(df))])

    # sign convention: income positive, expenses negative
    income_categories = {"Salary"}
    df["amount"] = df.apply(
        lambda r: abs(r["amount"]) if r["category"] in income_categories else -abs(r["amount"]),
        axis=1,
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"Generated {len(df)} transactions -> {SYNTHETIC_DATA_PATH}")
    print(df["category"].value_counts())
    print(f"Duplicates injected: {df['is_duplicate'].sum()}")
    print(f"Anomalies injected: {df['is_anomaly'].sum()}")
