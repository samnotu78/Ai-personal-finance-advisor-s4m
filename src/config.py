"""
Central configuration for the AI Personal Finance Advisor.

Every module imports paths and constants from here instead of hardcoding
strings, so the project can be repointed (e.g. to a different data file,
or a Postgres DB later) by editing exactly one file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_transactions.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_transactions.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Food", "Shopping", "Travel", "Entertainment", "Healthcare",
    "Utilities", "Education", "Salary", "Investment", "Bills",
    "Miscellaneous",
]

INCOME_CATEGORIES = {"Salary"}
INVESTMENT_CATEGORIES = {"Investment"}
EXPENSE_CATEGORIES = [c for c in CATEGORIES if c not in INCOME_CATEGORIES]

# ---------------------------------------------------------------------------
# Required raw columns for an uploaded statement to be accepted
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {"date", "description", "amount"}

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
ANOMALY_CONTAMINATION = 0.02  # expected fraction of outlier transactions
FORECAST_LOOKBACK_MONTHS = 6  # months of history used to forecast next month
