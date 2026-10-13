# 📊 E-Commerce Customer & Revenue Intelligence Platform

An end-to-end **Business Analytics & Machine Learning Pipeline** built on the Olist Brazilian E-Commerce relational dataset (**99,441 orders**, **9 relational tables**, **R$ 15.4M+ revenue**).

This project bridges the gap between **Business Data Analytics** (SQL Window Functions, Cohort Retention, RFM Segmentation, Logistics KPIs) and **Predictive Data Science** (Probabilistic CLV via BG/NBD & Gamma-Gamma, Cost-Sensitive XGBoost Detractor Modeling, TreeSHAP Explainability, and an Interactive Financial ROI Simulator).

---

## 🎯 Executive Summary & Key Business Insights

### 1. Revenue Concentration & Customer Retention Bottleneck (RFM Analysis)
Analysis across **93,357 unique delivered customers** revealed severe retention friction and high revenue exposure:

| Customer Segment | Customer Count | Avg Recency (Days) | Avg Frequency | Avg Monetary (R$) | Total Revenue (R$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High-Spend Newcomers** | 14,445 | 90.61 | 1.00 | 302.15 | 4,364,515.85 |
| **At Risk (High Value)** | 13,739 | 393.15 | 1.00 | 307.88 | 4,229,926.62 |
| **Core Customers** | 18,007 | 165.91 | 1.00 | 174.08 | 3,134,724.26 |
| **Standard Customers** | 29,052 | 197.55 | 1.00 | 68.05 | 1,976,866.64 |
| **Hibernating / Lost** | 15,495 | 394.55 | 1.01 | 55.81 | 864,711.55 |
| **Loyal Customers** | 1,629 | 295.90 | 2.08 | 296.70 | 483,329.11 |
| **Champions** | 990 | 87.83 | 2.18 | 372.11 | 368,387.74 |

* **High-Value Churn Exposure:** **13,739 customers** fall into the **At Risk (High Value)** segment, representing **R$ 4.23M** in historical revenue with an average inactivity of **393 days**.
* **One-Time Buyer Dominance:** Outside of **Champions** and **Loyal Customers**, over **96%** of customers never make a second purchase, underscoring the need for automated post-purchase lifecycle campaigns.

---

### 2. The "4-Day Delivery Delay" Satisfaction Cliff (Logistics KPIs)
SQL delivery timing analysis across **95,823 reviewed orders** uncovered a sharp operational threshold where customer satisfaction collapses:

| Delivery Status Bucket | Total Orders | Avg Shipping Days | Avg Review Score (1–5★) | Avg Order Value (R$) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Very Early (>7 days early)** | 70,929 | 10.2 | 4.32 ★ | 160.75 |
| **2. On Time (0–7 days early)** | 17,234 | 13.6 | 4.20 ★ | 148.84 |
| **3. Slight Delay (1–3 days late)** | 2,636 | 21.1 | 3.77 ★ | 157.13 |
| **4. Moderate Delay (4–7 days late)** | 1,773 | 26.7 | 2.32 ★ | 173.21 |
| **5. Severe Delay (>7 days late)** | 3,251 | 42.3 | 1.73 ★ | 181.45 |

* **Actionable Insight:** Customers tolerate minor delays up to 3 days (**3.77★**), but review scores plummet to **2.32★** once a shipment is **4 to 7 days late**. Furthermore, severely delayed orders carry the highest average order value (**R$ 181.45**), putting premium buyers at the highest risk of churn.

---

### 3. Predictive Detractor Modeling & TreeSHAP Explainability
An **XGBoost Classifier** was trained on **76,658 orders** (`scale_pos_weight = 6.78` to handle the `12.85%` positive detractor class imbalance) using strictly operational and logistics features to flag dissatisfied customers (*Detractors: Review Score ≤ 2*) before they churn.

* **Test Set Performance:**
  * **ROC-AUC:** `0.7537`
  * **Recall:** `53.94%` (Successfully captures over half of dissatisfied customers purely from pre-review logistics metadata)
  * **Precision:** `37.55%` | **F1-Score:** `0.4427`

* **Global Feature Attribution (Mean |TreeSHAP| Impact on Log-Odds):**
  1. `delivery_delay_days` (**0.4114**): Primary driver of customer dissatisfaction.
  2. `actual_shipping_days` (**0.2764**): Prolonged transit times degrade customer perception even when delivered before the estimated deadline.
  3. `item_count` (**0.2716**): Multi-item baskets experience significantly higher detractor risk due to split shipments across multiple marketplace sellers and partial fulfillment issues.
  4. `avg_volume_cm3` (**0.0952**) & `total_freight` (**0.0862**): Bulky items with high shipping fees trigger stricter customer expectations.

---

### 4. Probabilistic Customer Lifetime Value (BG/NBD + Gamma-Gamma)
Implemented **Beta-Geometric/NBD** and **Gamma-Gamma** probabilistic models across **2,015 repeat buyers** to forecast 90-day purchase probabilities (`prob_alive`), expected average order value, and 90-day Customer Lifetime Value (`clv_90d`) for VIP retention targeting.

---

## 🛠️ Tech Stack & Engineering Architecture

* **Database & ETL:** Python, SQLAlchemy, SQLite (`9` normalized source tables + `6` analytical and ML materialized tables)
* **Advanced SQL Analytics:** Common Table Expressions (`WITH`), Window Functions (`NTILE`, `FIRST_VALUE`, `PARTITION BY`), Julian Date Arithmetic
* **Machine Learning & Statistical Modeling:** `XGBoost`, `lifetimes` (BG/NBD & Gamma-Gamma), `scikit-learn`, `NumPy`, `Pandas`
* **Explainable AI (XAI):** Native `TreeSHAP` feature attribution
* **Interactive BI & ROI Application:** `Streamlit`, `Plotly` (3-tab executive dashboard + interactive retention campaign ROI simulator)
* **Quality Assurance & DevOps:** `pytest` automated data/model quality gates, `Docker` & `docker-compose`

---

## 📂 Project Structure

```text
ecommerce-customer-intelligence/
├── data/
│   └── raw/                        # Raw Olist CSV files (ignored in Git)
├── sql/
│   ├── rfm_analysis.sql            # CTE & NTILE window queries for customer segmentation
│   ├── cohort_retention.sql        # Monthly cohort retention matrix queries
│   └── logistics_kpis.sql          # Shipping delay buckets vs. review score analysis
├── src/
│   ├── data_loader.py              # Automated CSV-to-SQLite relational ETL pipeline
│   ├── analytics.py                # SQL execution runner & analytical table persistence
│   └── models/
│       ├── clv_model.py            # BG/NBD & Gamma-Gamma 90-day CLV forecasting
│       └── risk_model.py           # XGBoost classifier & TreeSHAP attribution pipeline
├── dashboard/
│   └── app.py                      # Interactive Streamlit Executive BI & ROI Simulator
├── tests/
│   └── test_pipeline.py            # Pytest suite for schema, RFM bounds, and AUC gates
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start & Reproducibility

### 1. Clone the Repository & Set Up Virtual Environment
```bash
git clone https://github.com/alpsalgur/ecommerce-customer-intelligence.git
cd ecommerce-customer-intelligence

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Dataset & Run End-to-End Pipeline
1. Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from Kaggle and extract the 9 `.csv` files into `data/raw/`.
2. Execute the ETL, SQL Analytics, and Machine Learning pipelines sequentially:

```bash
# 1. Build the relational SQLite database from raw CSVs
python src/data_loader.py

# 2. Run SQL analytical queries (RFM, Cohort Retention, Logistics KPIs)
python src/analytics.py

# 3. Train Probabilistic CLV (BG/NBD + Gamma-Gamma) and XGBoost Risk models
python src/models/clv_model.py
python src/models/risk_model.py

# 4. Run automated data quality and model performance tests
pytest -v
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Or run containerized via **Docker**:
```bash
docker compose up --build
```
Access the application at `http://localhost:8501`.

---

## 👤 Author

**Alp Salgür**  
B.Sc. in Electrical Engineering & Computer Science — Czech Technical University in Prague (ČVUT)  
* **GitHub:** [github.com/alpsalgur](https://github.com/alpsalgur)
* **LinkedIn:** [linkedin.com/in/AlpSalgür](https://linkedin.com/in/AlpSalgür)