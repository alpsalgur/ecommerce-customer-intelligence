WITH delivery_metrics AS (
    SELECT 
        o.order_id,
        julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) AS actual_shipping_days,
        julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS delay_days,
        r.review_score,
        p.payment_value
    FROM orders o
    JOIN order_reviews r ON o.order_id = r.order_id
    JOIN (
        SELECT order_id, SUM(payment_value) AS payment_value 
        FROM order_payments 
        GROUP BY order_id
    ) p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
)
SELECT 
    CASE 
        WHEN delay_days <= -7 THEN '1. Very Early (>7 days early)'
        WHEN delay_days <= 0 THEN '2. On Time (0-7 days early)'
        WHEN delay_days <= 3 THEN '3. Slight Delay (1-3 days late)'
        WHEN delay_days <= 7 THEN '4. Moderate Delay (4-7 days late)'
        ELSE '5. Severe Delay (>7 days late)'
    END AS delivery_status_bucket,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(AVG(actual_shipping_days), 1) AS avg_shipping_days,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(AVG(payment_value), 2) AS avg_order_value
FROM delivery_metrics
GROUP BY delivery_status_bucket
ORDER BY delivery_status_bucket;