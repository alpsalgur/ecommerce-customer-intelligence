WITH customer_orders AS (
    SELECT 
        c.customer_unique_id,
        strftime('%Y-%m-01', o.order_purchase_timestamp) AS order_month
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, strftime('%Y-%m-01', o.order_purchase_timestamp)
),
cohorts AS (
    SELECT 
        customer_unique_id,
        MIN(order_month) AS cohort_month
    FROM customer_orders
    GROUP BY customer_unique_id
),
cohort_activities AS (
    SELECT 
        co.cohort_month,
        (CAST(strftime('%Y', o.order_month) AS INTEGER) - CAST(strftime('%Y', co.cohort_month) AS INTEGER)) * 12 +
        (CAST(strftime('%m', o.order_month) AS INTEGER) - CAST(strftime('%m', co.cohort_month) AS INTEGER)) AS period_number,
        COUNT(DISTINCT co.customer_unique_id) AS active_customers
    FROM customer_orders o
    JOIN cohorts co ON o.customer_unique_id = co.customer_unique_id
    WHERE co.cohort_month >= '2017-01-01'
    GROUP BY co.cohort_month, period_number
)
SELECT 
    cohort_month,
    period_number,
    active_customers,
    FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY period_number) AS cohort_size,
    ROUND(CAST(active_customers AS FLOAT) * 100.0 / 
          FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY period_number), 2) AS retention_rate_pct
FROM cohort_activities
WHERE period_number BETWEEN 0 AND 12
ORDER BY cohort_month, period_number;