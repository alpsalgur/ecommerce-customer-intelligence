import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_DATA_DIR = Path("data/raw")
DB_PATH = Path("data/olist.db")

# Map raw CSV filenames to clean relational SQL table names
TABLE_MAPPING = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def build_database(raw_dir: Path = RAW_DATA_DIR, db_path: Path = DB_PATH) -> None:
    """Reads raw CSV files, parses timestamps, and loads tables into a SQLite database."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    engine = create_engine(f"sqlite:///{db_path}")

    for csv_file, table_name in TABLE_MAPPING.items():
        file_path = raw_dir / csv_file
        if not file_path.exists():
            logging.warning(f"Missing file skipped: {csv_file}")
            continue

        logging.info(f"Loading: {csv_file} -> table '{table_name}'...")
        parse_dates = DATE_COLUMNS.get(table_name, False)
        df = pd.read_csv(file_path, parse_dates=parse_dates)

        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logging.info(f"Completed: '{table_name}' ({len(df):,} rows)")

    verify_database(engine)


def verify_database(engine) -> None:
    """Verifies database tables and logs the order status distribution."""
    query = text("""
        SELECT 
            order_status, 
            COUNT(*) as total_orders 
        FROM orders 
        GROUP BY order_status 
        ORDER BY total_orders DESC;
    """)
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
        logging.info(
            "Database verification query succeeded. Order distribution:\n"
            + result.to_string(index=False)
        )


if __name__ == "__main__":
    build_database()