from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

DB_PATH = Path("data/olist.db")


@pytest.fixture(scope="module")
def db_engine():
    if not DB_PATH.exists():
        pytest.skip("Database file data/olist.db not found.")
    return create_engine(f"sqlite:///{DB_PATH}")


def test_required_tables_exist(db_engine):
    """Verifies that all raw, analytical, and ML tables are created in SQLite."""
    inspector = inspect(db_engine)
    existing_tables = set(inspector.get_table_names())
    expected_tables = {
        "orders",
        "customers",
        "analytics_rfm",
        "analytics_cohort",
        "analytics_logistics_kpi",
        "ml_clv_predictions",
        "ml_model_metrics",
        "ml_shap_importance",
        "ml_order_risk_scores",
    }
    assert expected_tables.issubset(existing_tables), (
        f"Missing tables: {expected_tables - existing_tables}"
    )


def test_rfm_score_boundaries(db_engine):
    """Ensures RFM scores are strictly within valid bounds and monetary values are positive."""
    df_rfm = pd.read_sql_table("analytics_rfm", db_engine)
    assert not df_rfm.empty, "RFM table should not be empty."
    assert df_rfm["r_score"].between(1, 5).all(), "R-score out of [1, 5] bounds."
    assert df_rfm["f_score"].between(1, 5).all(), "F-score out of [1, 5] bounds."
    assert df_rfm["m_score"].between(1, 5).all(), "M-score out of [1, 5] bounds."
    assert (df_rfm["monetary"] >= 0).all(), "Negative monetary values detected."


def test_clv_predictions_validity(db_engine):
    """Validates BG/NBD and Gamma-Gamma model outputs."""
    df_clv = pd.read_sql_table("ml_clv_predictions", db_engine)
    assert len(df_clv) > 1000, "Expected at least 1,000 repeat buyers in CLV table."
    assert df_clv["prob_alive"].between(0.0, 1.0).all(), "Invalid probability bounds."
    assert (df_clv["clv_90d"] >= 0).all(), "CLV predictions must be non-negative."


def test_xgboost_model_quality_gate(db_engine):
    """Enforces a minimum ROC-AUC quality threshold for the customer risk classifier."""
    df_metrics = pd.read_sql_table("ml_model_metrics", db_engine)
    roc_auc = df_metrics["roc_auc"].iloc[0]
    recall = df_metrics["recall"].iloc[0]
    assert roc_auc >= 0.70, f"Model ROC-AUC ({roc_auc}) fell below the 0.70 quality gate."
    assert recall >= 0.45, f"Model Recall ({recall}) fell below the 0.45 quality gate."


def test_shap_importance_completeness(db_engine):
    """Verifies that TreeSHAP values were computed for all 12 engineered features."""
    df_shap = pd.read_sql_table("ml_shap_importance", db_engine)
    assert len(df_shap) == 12, "Expected exactly 12 engineered features in SHAP table."
    assert (df_shap["mean_abs_shap_value"] >= 0).all(), "Mean absolute SHAP must be >= 0."