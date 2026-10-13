from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(
    page_title="E-Commerce Customer & Revenue Intelligence",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path("data/olist.db")


@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    """Loads a table from the SQLite database with caching."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as conn:
        return pd.read_sql_table(table_name, conn)


def main() -> None:
    st.title("📊 E-Commerce Customer & Revenue Intelligence Platform")
    st.markdown(
        "**End-to-End Business Analytics & Machine Learning Pipeline** | "
        "*RFM Segmentation • Cohort Retention • Logistics KPIs • BG/NBD CLV • XGBoost + TreeSHAP*"
    )

    if not DB_PATH.exists():
        st.error("Database `data/olist.db` not found. Please run the ETL and model scripts first.")
        return

    # Load precomputed analytical and ML tables
    df_rfm = load_table("analytics_rfm")
    df_cohort = load_table("analytics_cohort")
    df_logistics = load_table("analytics_logistics_kpi")
    df_clv = load_table("ml_clv_predictions")
    df_metrics = load_table("ml_model_metrics")
    df_shap = load_table("ml_shap_importance")
    df_risk = load_table("ml_order_risk_scores")

    # --- TOP-LEVEL EXECUTIVE KPIs ---
    total_revenue = df_rfm["monetary"].sum()
    total_customers = len(df_rfm)
    at_risk_revenue = df_rfm.loc[
        df_rfm["customer_segment"] == "At Risk (High Value)", "monetary"
    ].sum()
    roc_auc = df_metrics["roc_auc"].iloc[0]
    recall_score_val = df_metrics["recall"].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Delivered Revenue", f"R$ {total_revenue:,.0f}")
    col2.metric("Unique Customers", f"{total_customers:,}")
    col3.metric("High-Value Revenue at Risk", f"R$ {at_risk_revenue:,.0f}")
    col4.metric("XGBoost Risk Model AUC", f"{roc_auc:.4f}")
    col5.metric("Detractor Recall Rate", f"{recall_score_val:.1%}")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "👥 RFM & Cohort Retention",
        "🚚 Logistics & Satisfaction Cliff",
        "🤖 Predictive CLV, SHAP & ROI Simulator",
    ])

    # ================= TAB 1: RFM & COHORT RETENTION =================
    with tab1:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Revenue & Customer Distribution by RFM Segment")
            rfm_summary = (
                df_rfm.groupby("customer_segment", as_index=False)
                .agg(
                    customers=("customer_unique_id", "count"),
                    revenue=("monetary", "sum"),
                    avg_monetary=("monetary", "mean"),
                )
                .sort_values(by="revenue", ascending=True)
            )
            fig_rfm = px.bar(
                rfm_summary,
                x="revenue",
                y="customer_segment",
                orientation="h",
                text_auto=".3s",
                color="avg_monetary",
                color_continuous_scale="Blues",
                labels={
                    "revenue": "Total Revenue (R$)",
                    "customer_segment": "RFM Segment",
                    "avg_monetary": "Avg Spend (R$)",
                },
            )
            fig_rfm.update_layout(height=420)
            st.plotly_chart(fig_rfm, use_container_width=True)

        with c2:
            st.subheader("Segment Breakdown Table")
            st.dataframe(
                rfm_summary.sort_values(by="revenue", ascending=False).style.format({
                    "customers": "{:,}",
                    "revenue": "R$ {:,.2f}",
                    "avg_monetary": "R$ {:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Monthly Customer Cohort Retention Heatmap (%)")
        cohort_pivot = df_cohort.pivot(
            index="cohort_month", columns="period_number", values="retention_rate_pct"
        )
        # Filter out Period 0 (100%) to highlight repeat retention variations clearly
        cohort_repeat = cohort_pivot.iloc[:, 1:]
        fig_cohort = px.imshow(
            cohort_repeat,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            labels={
                "x": "Months Since First Purchase",
                "y": "Cohort Month",
                "color": "Retention (%)",
            },
        )
        fig_cohort.update_layout(height=500)
        st.plotly_chart(fig_cohort, use_container_width=True)

    # ================= TAB 2: LOGISTICS & SATISFACTION CLIFF =================
    with tab2:
        st.subheader("Impact of Delivery Delays on Customer Review Scores & Order Value")
        st.markdown(
            "**Key Insight:** Customer satisfaction remains resilient up to a 3-day delay (**3.77★**), "
            "but experiences a sharp cliff at **4–7 days late (2.32★)** and **>7 days late (1.73★)**."
        )

        fig_log = go.Figure()
        fig_log.add_trace(
            go.Bar(
                x=df_logistics["delivery_status_bucket"],
                y=df_logistics["total_orders"],
                name="Total Orders",
                marker_color="#4C78A8",
                yaxis="y1",
            )
        )
        fig_log.add_trace(
            go.Scatter(
                x=df_logistics["delivery_status_bucket"],
                y=df_logistics["avg_review_score"],
                name="Avg Review Score (1-5★)",
                mode="lines+markers+text",
                text=df_logistics["avg_review_score"],
                textposition="top center",
                marker=dict(size=10, color="#E45756"),
                line=dict(width=3, color="#E45756"),
                yaxis="y2",
            )
        )
        fig_log.update_layout(
            xaxis_title="Delivery Timing Bucket",
            yaxis=dict(title="Order Volume", side="left"),
            yaxis2=dict(
                title="Average Review Score (★)",
                side="right",
                overlaying="y",
                range=[1, 5],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
        )
        st.plotly_chart(fig_log, use_container_width=True)

        st.dataframe(
            df_logistics.style.format({
                "total_orders": "{:,}",
                "avg_shipping_days": "{:.1f} days",
                "avg_review_score": "{:.2f} ★",
                "avg_order_value": "R$ {:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ================= TAB 3: PREDICTIVE ML, SHAP & ROI SIMULATOR =================
    with tab3:
        m1, m2 = st.columns([1, 1])

        with m1:
            st.subheader("Global Model Explainability (TreeSHAP Impact)")
            fig_shap = px.bar(
                df_shap.sort_values(by="mean_abs_shap_value", ascending=True),
                x="mean_abs_shap_value",
                y="feature_name",
                orientation="h",
                text_auto=".4f",
                color="mean_abs_shap_value",
                color_continuous_scale="Tealgrn",
                labels={
                    "mean_abs_shap_value": "Mean |SHAP Value| (Log-Odds Impact)",
                    "feature_name": "Feature",
                },
            )
            fig_shap.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)

        with m2:
            st.subheader("💰 Proactive Retention Campaign ROI Simulator")
            st.markdown(
                "Simulate the financial impact of triggering automated apology vouchers "
                "for orders flagged by the **XGBoost Detractor Risk Model**."
            )

            risk_threshold = st.slider(
                "Model Probability Threshold for Intervention",
                min_value=0.40,
                max_value=0.90,
                value=0.65,
                step=0.05,
            )
            voucher_cost = st.slider(
                "Apology Voucher Cost per Flagged Order (R$)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=5.0,
            )
            recovery_rate = st.slider(
                "Estimated Customer Retention Conversion Rate (%)",
                min_value=5,
                max_value=40,
                value=20,
                step=5,
            ) / 100.0

            flagged_orders = df_risk[df_risk["predicted_risk_prob"] >= risk_threshold]
            num_flagged = len(flagged_orders)
            true_detractors_caught = flagged_orders["is_high_risk_detractor"].sum()

            campaign_cost = num_flagged * voucher_cost
            avg_repeat_value = df_clv["expected_avg_order_value"].median()
            gross_saved_revenue = true_detractors_caught * recovery_rate * avg_repeat_value
            net_roi_value = gross_saved_revenue - campaign_cost

            r1, r2, r3 = st.columns(3)
            r1.metric("Flagged High-Risk Orders", f"{num_flagged:,}")
            r2.metric("Campaign Cost", f"R$ {campaign_cost:,.0f}")
            r3.metric(
                "Estimated Net Profit Impact",
                f"R$ {net_roi_value:,.0f}",
                delta=f"{(net_roi_value / max(campaign_cost, 1)) * 100:.1f}% ROI",
            )

        st.divider()
        b1, b2 = st.columns([1, 1])

        with b1:
            st.subheader("Top Predicted 90-Day CLV VIP Customers (BG/NBD + Gamma-Gamma)")
            st.dataframe(
                df_clv[
                    [
                        "customer_unique_id",
                        "frequency",
                        "monetary_value",
                        "prob_alive",
                        "expected_avg_order_value",
                        "clv_90d",
                    ]
                ]
                .head(15)
                .style.format({
                    "frequency": "{:.0f}",
                    "monetary_value": "R$ {:,.2f}",
                    "prob_alive": "{:.1%}",
                    "expected_avg_order_value": "R$ {:,.2f}",
                    "clv_90d": "R$ {:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with b2:
            st.subheader("Top Flagged High-Risk Orders (XGBoost Queue)")
            st.dataframe(
                df_risk.head(15).style.format({
                    "total_price": "R$ {:,.2f}",
                    "freight_ratio": "{:.1%}",
                    "actual_shipping_days": "{:.1f} d",
                    "delivery_delay_days": "{:+.1f} d",
                    "predicted_risk_prob": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()