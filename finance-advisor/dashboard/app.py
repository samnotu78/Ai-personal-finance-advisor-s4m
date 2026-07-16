"""
AI Personal Finance Advisor -- Streamlit dashboard.

Run with:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import SYNTHETIC_DATA_PATH
from src.cleaning.clean import clean_transactions, SchemaError
from src.categorization.categorize import categorize_transactions
from src.models.forecast import monthly_summary, forecast_expenses, category_trend
from src.models.anomaly import detect_anomalies, evaluate_anomaly_detection
from src.insights.insights import generate_all_insights

st.set_page_config(page_title="AI Personal Finance Advisor", page_icon="💰", layout="wide")


@st.cache_data(show_spinner=False)
def run_pipeline(raw_df: pd.DataFrame):
    cleaned, log = clean_transactions(raw_df)
    categorized = categorize_transactions(cleaned)
    return categorized, log


def load_data():
    st.sidebar.header("1. Upload your statement")
    uploaded = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    use_sample = st.sidebar.checkbox("Use sample synthetic data instead", value=uploaded is None)

    if use_sample or uploaded is None:
        raw_df = pd.read_csv(SYNTHETIC_DATA_PATH)
        st.sidebar.caption(f"Using bundled sample data ({len(raw_df)} rows).")
    else:
        if uploaded.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)
    return raw_df


def kpi_cards(summary_df: pd.DataFrame):
    total_income = summary_df["income"].sum()
    total_expenses = summary_df["expenses"].sum()
    net = total_income - total_expenses
    avg_savings_rate = summary_df["savings_rate"].dropna().mean() * 100 if not summary_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income", f"${total_income:,.0f}")
    c2.metric("Total Expenses", f"${total_expenses:,.0f}")
    c3.metric("Net Cash Flow", f"${net:,.0f}", delta=f"{net:,.0f}")
    c4.metric("Avg Savings Rate", f"{avg_savings_rate:.1f}%")


def main():
    st.title("💰 AI Personal Finance Advisor")
    st.caption("Upload a bank/card statement to get an instant financial analysis.")

    raw_df = load_data()

    try:
        df, cleaning_log = run_pipeline(raw_df)
    except SchemaError as e:
        st.error(str(e))
        st.stop()

    with st.expander("📋 Data cleaning summary", expanded=False):
        for line in cleaning_log:
            st.write("•", line)

    # --- Filters -----------------------------------------------------------
    st.sidebar.header("2. Filters")
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    categories = st.sidebar.multiselect("Categories", sorted(df["category"].unique()), default=sorted(df["category"].unique()))

    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[df["category"].isin(categories)]

    if df.empty:
        st.warning("No transactions match the selected filters.")
        st.stop()

    # --- KPIs ----------------------------------------------------------------
    summary = monthly_summary(df)
    kpi_cards(summary)

    st.divider()

    # --- Trend charts --------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Income vs Expenses")
        fig = px.bar(summary, x="month", y=["income", "expenses"], barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Net Cash Flow Trend")
        fig2 = px.line(summary, x="month", y="net", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    # --- Category analysis -----------------------------------------------------
    st.subheader("Category-wise Spending")
    expenses = df[df["amount"] < 0].copy()
    cat_totals = expenses.groupby("category")["amount"].sum().abs().sort_values(ascending=False).reset_index()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig3 = px.pie(cat_totals, names="category", values="amount", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.dataframe(cat_totals.rename(columns={"amount": "total_spent"}), use_container_width=True, hide_index=True)

    # --- Top merchants & largest transactions ---------------------------------
    st.subheader("Top Merchants & Largest Transactions")
    m1, m2 = st.columns(2)
    with m1:
        top_merchants = (
            expenses.groupby("description")["amount"].sum().abs().sort_values(ascending=False).head(10).reset_index()
        )
        st.write("**Top merchants by total spend**")
        st.dataframe(top_merchants, use_container_width=True, hide_index=True)
    with m2:
        largest = expenses.assign(abs_amount=expenses["amount"].abs()).sort_values("abs_amount", ascending=False).head(10)
        st.write("**Largest individual transactions**")
        st.dataframe(largest[["date", "description", "category", "amount"]], use_container_width=True, hide_index=True)

    st.divider()

    # --- Predictions -----------------------------------------------------------
    st.subheader("📈 Predictions")
    forecast = forecast_expenses(df)
    p1, p2, p3 = st.columns(3)
    p1.metric("Next Month Expense (predicted)", f"${forecast.predicted_next_month_expense:,.0f}")
    p2.metric("Projected Annual Expense", f"${forecast.predicted_annual_expense:,.0f}")
    p3.metric("Next Month Savings (predicted)", f"${forecast.predicted_next_month_savings:,.0f}")
    if forecast.mae is not None:
        st.caption(f"Model backtest: MAE ${forecast.mae:,.0f}" + (f", MAPE {forecast.mape*100:.1f}%" if forecast.mape else ""))
    else:
        st.caption("Not enough monthly history yet for a backtested error estimate.")

    # --- Anomaly detection -------------------------------------------------------
    st.subheader("🚨 Unusual Transactions")
    anomalies = detect_anomalies(df)
    flagged = anomalies[anomalies["predicted_anomaly"]]
    st.write(f"{len(flagged)} unusual transaction(s) flagged out of {len(anomalies)} expenses analyzed.")
    if not flagged.empty:
        st.dataframe(
            flagged[["date", "description", "category", "amount"]].sort_values("amount"),
            use_container_width=True, hide_index=True,
        )
    eval_metrics = evaluate_anomaly_detection(anomalies)
    if eval_metrics:
        st.caption(
            f"Evaluated against synthetic ground truth -- precision: {eval_metrics['precision']}, "
            f"recall: {eval_metrics['recall']} ({eval_metrics['n_true_anomalies']} true anomalies in this period)."
        )

    st.divider()

    # --- AI Insights -----------------------------------------------------------
    st.subheader("🤖 AI Financial Insights")
    for insight in generate_all_insights(df, forecast):
        st.info(insight)

    st.divider()

    # --- Downloadable reports ----------------------------------------------------
    st.subheader("⬇️ Download Reports")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download cleaned & categorized data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_transactions.csv",
        mime="text/csv",
    )
    monthly_report = summary.to_csv(index=False).encode("utf-8")
    d2.download_button(
        "Download monthly summary report (CSV)",
        data=monthly_report,
        file_name="monthly_summary.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
