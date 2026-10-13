WITH delivered_orders AS (
    SELECT 
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp,
        SUM(p.payment_value) AS order_value
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, o.order_id, o.order_purchase_timestamp
),
reference_date AS (
    SELECT MAX(order_purchase_timestamp) AS max_date FROM delivered_orders
),
rfm_raw AS (
    SELECT 
        d.customer_unique_id,
        CAST(julianday(r.max_date) - julianday(MAX(d.order_purchase_timestamp)) AS INTEGER) AS recency_days,
        COUNT(DISTINCT d.order_id) AS frequency,
        ROUND(SUM(d.order_value), 2) AS monetary
    FROM delivered_orders d
    CROSS JOIN reference_date r
    GROUP BY d.customer_unique_id
),
rfm_scores AS (
    SELECT 
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        CASE 
            WHEN frequency = 1 THEN 1
            WHEN frequency = 2 THEN 3
            WHEN frequency = 3 THEN 4
            ELSE 5 
        END AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_raw
)
SELECT 
    *,
    (r_score + f_score + m_score) AS rfm_total_score,
    CASE 
        WHEN r_score >= 4 AND f_score >= 3 AND m_score >= 4 THEN 'Champions'
        WHEN f_score >= 3 AND m_score >= 3 THEN 'Loyal Customers'
        WHEN r_score >= 4 AND f_score = 1 AND m_score >= 4 THEN 'High-Spend Newcomers'
        WHEN r_score >= 3 AND m_score >= 3 THEN 'Core Customers'
        WHEN r_score <= 2 AND m_score >= 4 THEN 'At Risk (High Value)'
        WHEN r_score <= 2 AND m_score <= 2 THEN 'Hibernating / Lost'
        ELSE 'Standard Customers'
    END AS customer_segment
FROM rfm_scores;