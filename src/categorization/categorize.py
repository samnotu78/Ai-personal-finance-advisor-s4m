"""
Smart expense categorization.

Two layers, in order:
  1. Rule-based: exact/substring match against the known merchant catalog
     (src/data_generation/merchants.py) plus a keyword table for common
     terms that aren't specific merchants (e.g. "rent", "tuition").
  2. AI fallback: a TF-IDF + Logistic Regression classifier, trained on
     whatever the rule-based layer confidently labeled, is used to guess a
     category for anything the rules didn't catch. This is the "extendable
     AI approach" -- as more labeled data accumulates the classifier keeps
     improving without anyone touching the rule table.

Salary/Investment are intentionally NOT part of the keyword fallback net --
sign of the amount (positive = income) is used to steer ambiguous credits
away from expense categories.
"""
from __future__ import annotations

import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_SEED
from src.data_generation.merchants import MERCHANT_CATALOG, RECURRING_SUBSCRIPTIONS

_MODEL_PATH = MODELS_DIR / "categorizer.joblib"

# Extra keyword rules for things that aren't in the merchant catalog
# (generic descriptions real statements often contain).
_KEYWORD_RULES = [
    (r"\brent\b|property mgmt|leasing", "Bills"),
    (r"electric|gas co|water (co|utility)|power & light", "Utilities"),
    (r"tuition|university|college|student loan", "Education"),
    (r"payroll|direct dep|salary", "Salary"),
    (r"401k|brokerage|vanguard|fidelity|schwab|invest", "Investment"),
    (r"pharmacy|clinic|dental|medical|urgent care|cvs|walgreens", "Healthcare"),
]


def _build_merchant_lookup() -> dict:
    """Map every known raw description variant (lowercased) -> category."""
    lookup = {}
    for category, profiles in MERCHANT_CATALOG.items():
        for profile in profiles:
            for variant in profile["variants"]:
                lookup[variant.lower()] = category
    for sub in RECURRING_SUBSCRIPTIONS:
        lookup[sub["variant"].lower()] = sub["category"]
    return lookup


_MERCHANT_LOOKUP = _build_merchant_lookup()


def rule_based_category(description: str) -> str | None:
    """Return a category if `description` matches a known merchant/keyword, else None."""
    desc_lower = description.lower().strip()

    if desc_lower in _MERCHANT_LOOKUP:
        return _MERCHANT_LOOKUP[desc_lower]

    for variant, category in _MERCHANT_LOOKUP.items():
        if variant in desc_lower:
            return category

    for pattern, category in _KEYWORD_RULES:
        if re.search(pattern, desc_lower):
            return category

    return None


def train_fallback_classifier(df: pd.DataFrame) -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression classifier on rows the rule-based
    layer could confidently label, so it can generalize to unseen merchant
    strings (new POS prefixes, store numbers, etc.) at inference time.
    """
    labeled = df.dropna(subset=["rule_category"])
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    pipeline.fit(labeled["description"], labeled["rule_category"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, _MODEL_PATH)
    return pipeline


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `category` column (and `category_source` for transparency: "rule"
    vs "model") to a cleaned transactions DataFrame.
    """
    df = df.copy()
    df["rule_category"] = df["description"].apply(rule_based_category)

    # Steer sign-ambiguous credits: any positive amount not caught by rules
    # is far more likely Salary/Investment than an expense category.
    unmatched_credit = df["rule_category"].isna() & (df["amount"] > 0)
    df.loc[unmatched_credit, "rule_category"] = "Salary"

    n_unmatched = df["rule_category"].isna().sum()
    if n_unmatched > 0 and df["rule_category"].notna().sum() >= 5:
        model = train_fallback_classifier(df)
        mask = df["rule_category"].isna()
        df.loc[mask, "model_category"] = model.predict(df.loc[mask, "description"])
    else:
        df["model_category"] = None

    df["category"] = df["rule_category"].fillna(df["model_category"]).fillna("Miscellaneous")
    df["category_source"] = df["rule_category"].notna().map({True: "rule", False: "model"})
    return df.drop(columns=["rule_category", "model_category"])
