import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = Path("data/olist.db")


def load_transaction_data(engine) -> pd.DataFrame:
    """Extracts customer-level order transactions for CLV modeling."""
    query = """
    SELECT 
        c.customer_unique_id,
        DATE(o.order_purchase_timestamp) AS order_date,
        SUM(p.payment_value) AS total_payment
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, o.order_id, DATE(o.order_purchase_timestamp)
    """
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def train_clv_pipeline() -> None:
    """Fits BG/NBD and Gamma-Gamma models to predict 90-day Customer Lifetime Value."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    logging.info("Loading transaction history from SQLite...")
    df_tx = load_transaction_data(engine)

    max_date = df_tx["order_date"].max()
    logging.info(f"Building RFM-T matrix (Observation end date: {max_date})...")

    rfmt = summary_data_from_transaction_data(
        df_tx,
        customer_id_col="customer_unique_id",
        datetime_col="order_date",
        monetary_value_col="total_payment",
        observation_period_end=max_date,
        freq="D",
    )

    # Filter customers with positive monetary value
    rfmt = rfmt[rfmt["monetary_value"] >= 0].copy()

    # 1. Fit BG/NBD Model for expected purchase frequency
    logging.info("Fitting Beta-Geometric/NBD (BG/NBD) model...")
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfmt["frequency"], rfmt["recency"], rfmt["T"])

    rfmt["predicted_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90, rfmt["frequency"], rfmt["recency"], rfmt["T"]
    )
    rfmt["prob_alive"] = bgf.conditional_probability_alive(
        rfmt["frequency"], rfmt["recency"], rfmt["T"]
    )

    # 2. Fit Gamma-Gamma Model on repeat buyers for expected monetary value
    repeat_buyers = rfmt[(rfmt["frequency"] > 0) & (rfmt["monetary_value"] > 0)].copy()
    logging.info(f"Fitting Gamma-Gamma model on {len(repeat_buyers):,} repeat buyers...")

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])

    # Compute 90-day CLV (3 months, 1% monthly discount rate)
    repeat_buyers["expected_avg_order_value"] = ggf.conditional_expected_average_profit(
        repeat_buyers["frequency"], repeat_buyers["monetary_value"]
    )
    repeat_buyers["clv_90d"] = ggf.customer_lifetime_value(
        bgf,
        repeat_buyers["frequency"],
        repeat_buyers["recency"],
        repeat_buyers["T"],
        repeat_buyers["monetary_value"],
        time=3,
        discount_rate=0.01,
        freq="D",
    )

    # Merge predictions back and persist to SQLite
    output_df = repeat_buyers.reset_index().sort_values(by="clv_90d", ascending=False)
    output_df = output_df.round(4)
    output_df.to_sql("ml_clv_predictions", engine, if_exists="replace", index=False)

    top_10_summary = output_df[
        [
            "customer_unique_id",
            "frequency",
            "monetary_value",
            "prob_alive",
            "expected_avg_order_value",
            "clv_90d",
        ]
    ].head(10)

    logging.info(
        f"CLV predictions saved to 'ml_clv_predictions' ({len(output_df):,} repeat customers).\n"
        + "--- TOP 10 CUSTOMERS BY 90-DAY PREDICTED CLV ---\n"
        + top_10_summary.to_string(index=False)
    )


if __name__ == "__main__":
    train_clv_pipeline()