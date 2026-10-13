import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = Path("data/olist.db")
SQL_DIR = Path("sql")


def run_sql_file(engine, filename: str) -> pd.DataFrame:
    """Executes an external SQL script and returns the result as a DataFrame."""
    sql_path = SQL_DIR / filename
    query = sql_path.read_text(encoding="utf-8")
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


def execute_analytics_pipeline() -> None:
    """Runs RFM, Cohort Retention, and Logistics KPI queries and persists analytical tables."""
    engine = create_engine(f"sqlite:///{DB_PATH}")

    # 1. RFM Segmentation
    logging.info("Running RFM segmentation analysis...")
    df_rfm = run_sql_file(engine, "rfm_analysis.sql")
    df_rfm.to_sql("analytics_rfm", engine, if_exists="replace", index=False)

    segment_summary = (
        df_rfm.groupby("customer_segment")
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .round(2)
        .sort_values(by="total_revenue", ascending=False)
    )
    logging.info("\n--- RFM SEGMENT SUMMARY ---\n" + segment_summary.to_string())

    # 2. Cohort Retention Analysis
    logging.info("Running Cohort Retention analysis...")
    df_cohort = run_sql_file(engine, "cohort_retention.sql")
    df_cohort.to_sql("analytics_cohort", engine, if_exists="replace", index=False)
    logging.info(f"Cohort retention table created ({len(df_cohort):,} rows).")

    # 3. Logistics & Delivery Delay KPIs
    logging.info("Running Logistics and Delivery Delay KPI analysis...")
    df_logistics = run_sql_file(engine, "logistics_kpis.sql")
    df_logistics.to_sql("analytics_logistics_kpi", engine, if_exists="replace", index=False)
    logging.info(
        "\n--- DELIVERY DELAY VS. CUSTOMER SATISFACTION ---\n"
        + df_logistics.to_string(index=False)
    )


if __name__ == "__main__":
    execute_analytics_pipeline()