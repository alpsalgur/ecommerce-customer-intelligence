import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = Path("data/olist.db")

FEATURE_QUERY = """
WITH order_agg AS (
    SELECT 
        oi.order_id,
        COUNT(oi.order_item_id) AS item_count,
        SUM(oi.price) AS total_price,
        SUM(oi.freight_value) AS total_freight,
        AVG(p.product_weight_g) AS avg_weight_g,
        AVG(p.product_length_cm * p.product_height_cm * p.product_width_cm) AS avg_volume_cm3,
        AVG(p.product_photos_qty) AS avg_photos_qty,
        MAX(CASE WHEN s.seller_state != c.customer_state THEN 1 ELSE 0 END) AS is_interstate
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN sellers s ON oi.seller_id = s.seller_id
    WHERE o.order_status = 'delivered'
    GROUP BY oi.order_id
),
payment_agg AS (
    SELECT 
        order_id,
        MAX(payment_installments) AS max_installments,
        SUM(payment_value) AS payment_total
    FROM order_payments
    GROUP BY order_id
),
review_agg AS (
    SELECT 
        order_id,
        MIN(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
)
SELECT 
    o.order_id,
    c.customer_unique_id,
    oa.item_count,
    oa.total_price,
    oa.total_freight,
    ROUND(oa.total_freight / NULLIF(oa.total_price, 0), 4) AS freight_ratio,
    oa.avg_weight_g,
    oa.avg_volume_cm3,
    oa.avg_photos_qty,
    oa.is_interstate,
    pa.max_installments,
    ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp), 2) AS actual_shipping_days,
    ROUND(julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp), 2) AS estimated_shipping_days,
    ROUND(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date), 2) AS delivery_delay_days,
    CASE WHEN ra.review_score <= 2 THEN 1 ELSE 0 END AS is_high_risk_detractor
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_agg oa ON o.order_id = oa.order_id
JOIN payment_agg pa ON o.order_id = pa.order_id
JOIN review_agg ra ON o.order_id = ra.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL;
"""

FEATURE_COLS = [
    "item_count",
    "total_price",
    "total_freight",
    "freight_ratio",
    "avg_weight_g",
    "avg_volume_cm3",
    "avg_photos_qty",
    "is_interstate",
    "max_installments",
    "actual_shipping_days",
    "estimated_shipping_days",
    "delivery_delay_days",
]


def train_risk_model() -> None:
    """Trains an XGBoost classifier to predict customer dissatisfaction/churn risk and computes SHAP values."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    logging.info("Extracting engineered features from SQLite...")
    with engine.connect() as conn:
        df = pd.read_sql_query(FEATURE_QUERY, conn)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS]
    y = df["is_high_risk_detractor"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count

    logging.info(
        f"Training XGBoost model on {len(X_train):,} orders "
        f"(Positive risk ratio: {y.mean():.2%}, scale_pos_weight: {scale_weight:.2f})..."
    )

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probs), 4),
        "precision": round(precision_score(y_test, y_preds), 4),
        "recall": round(recall_score(y_test, y_preds), 4),
        "f1_score": round(f1_score(y_test, y_preds), 4),
    }
    logging.info(f"Model Evaluation Metrics (Test Set): {metrics}")

    # Persist evaluation metrics
    pd.DataFrame([metrics]).to_sql("ml_model_metrics", engine, if_exists="replace", index=False)

    # Compute exact TreeSHAP values natively on a 5,000-order test sample
    logging.info("Computing TreeSHAP feature contributions for model explainability...")
    X_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
    dmatrix_sample = xgb.DMatrix(X_sample, feature_names=FEATURE_COLS)
    shap_contribs = model.get_booster().predict(dmatrix_sample, pred_contribs=True)

    # Exclude the last column (bias term) and compute mean absolute SHAP impact
    mean_abs_shap = np.abs(shap_contribs[:, :-1]).mean(axis=0)
    shap_df = (
        pd.DataFrame({
            "feature_name": FEATURE_COLS,
            "mean_abs_shap_value": np.round(mean_abs_shap, 4),
        })
        .sort_values(by="mean_abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_sql("ml_shap_importance", engine, if_exists="replace", index=False)
    logging.info("\n--- SHAP FEATURE IMPORTANCE (GLOBAL EXPLAINABILITY) ---\n" + shap_df.to_string(index=False))

    # Score all orders and persist top high-risk records for dashboard inspection
    df["predicted_risk_prob"] = np.round(model.predict_proba(X)[:, 1], 4)
    scored_orders = df[
        [
            "order_id",
            "customer_unique_id",
            "total_price",
            "freight_ratio",
            "actual_shipping_days",
            "delivery_delay_days",
            "is_high_risk_detractor",
            "predicted_risk_prob",
        ]
    ].sort_values(by="predicted_risk_prob", ascending=False)

    scored_orders.to_sql("ml_order_risk_scores", engine, if_exists="replace", index=False)
    logging.info(f"Saved {len(scored_orders):,} scored orders to 'ml_order_risk_scores'.")


if __name__ == "__main__":
    train_risk_model()