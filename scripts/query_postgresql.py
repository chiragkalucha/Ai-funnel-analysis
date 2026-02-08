"""
PostgreSQL Query Examples
Advanced SQL queries on your funnel data
"""

from sqlalchemy import create_engine
import pandas as pd

# Connect to PostgreSQL
engine = create_engine('postgresql://postgres:yonex.Chirag1@localhost:5432/funnel_analysis')

print("="*60)
print("🐘 POSTGRESQL QUERY EXAMPLES")
print("="*60)

# ============================================
# QUERY 1: Use the View we created
# ============================================

print("\n📊 QUERY 1: User Conversion Summary")
print("-" * 40)

query = """
SELECT 
    device_type,
    COUNT(*) as total_users,
    SUM(has_converted) as converters,
    ROUND(100.0 * SUM(has_converted) / COUNT(*), 2) as conversion_rate
FROM user_conversions
GROUP BY device_type
ORDER BY conversion_rate DESC
"""

df = pd.read_sql(query, engine)
print(df)

# ============================================
# QUERY 2: Funnel Drop-off Analysis
# ============================================

print("\n📊 QUERY 2: Funnel Drop-off Rates")
print("-" * 40)

query = """
WITH funnel_data AS (
    SELECT * FROM funnel_metrics
),
funnel_with_lag AS (
    SELECT 
        event_type,
        unique_users,
        LAG(unique_users) OVER (ORDER BY 
            CASE event_type
                WHEN 'homepage_visit' THEN 1
                WHEN 'product_view' THEN 2
                WHEN 'add_to_cart' THEN 3
                WHEN 'checkout_start' THEN 4
                WHEN 'payment_info' THEN 5
                WHEN 'purchase_complete' THEN 6
            END
        ) as previous_stage_users
    FROM funnel_data
)
SELECT 
    event_type,
    unique_users,
    previous_stage_users,
    CASE 
        WHEN previous_stage_users IS NOT NULL 
        THEN ROUND(100.0 * (previous_stage_users - unique_users) / previous_stage_users, 2)
        ELSE 0 
    END as drop_off_rate
FROM funnel_with_lag
ORDER BY 
    CASE event_type
        WHEN 'homepage_visit' THEN 1
        WHEN 'product_view' THEN 2
        WHEN 'add_to_cart' THEN 3
        WHEN 'checkout_start' THEN 4
        WHEN 'payment_info' THEN 5
        WHEN 'purchase_complete' THEN 6
    END
"""

df = pd.read_sql(query, engine)
print(df)

# ============================================
# QUERY 3: Cohort Analysis by Signup Month
# ============================================

print("\n📊 QUERY 3: Monthly Cohort Conversion")
print("-" * 40)

query = """
SELECT 
    TO_CHAR(u.signup_date, 'YYYY-MM') as signup_month,
    COUNT(DISTINCT u.user_id) as total_users,
    COUNT(DISTINCT CASE 
        WHEN e.event_type = 'purchase_complete' 
        THEN e.user_id 
    END) as converters,
    ROUND(100.0 * COUNT(DISTINCT CASE 
        WHEN e.event_type = 'purchase_complete' 
        THEN e.user_id 
    END) / COUNT(DISTINCT u.user_id), 2) as conversion_rate
FROM users u
LEFT JOIN events e ON u.user_id = e.user_id
GROUP BY TO_CHAR(u.signup_date, 'YYYY-MM')
ORDER BY signup_month
"""

df = pd.read_sql(query, engine)
print(df)

# ============================================
# QUERY 4: Session Analysis
# ============================================

print("\n📊 QUERY 4: Average Sessions: Converters vs Non-converters")
print("-" * 40)

query = """
SELECT 
    uc.has_converted,
    CASE 
        WHEN uc.has_converted = 1 THEN 'Converted'
        ELSE 'Did Not Convert'
    END as user_type,
    COUNT(DISTINCT uc.user_id) as user_count,
    ROUND(AVG(session_count), 2) as avg_sessions,
    ROUND(AVG(total_events), 2) as avg_events
FROM user_conversions uc
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(DISTINCT session_id) as session_count,
        COUNT(*) as total_events
    FROM events
    GROUP BY user_id
) e ON uc.user_id = e.user_id
GROUP BY uc.has_converted
"""

df = pd.read_sql(query, engine)
print(df)

# ============================================
# QUERY 5: Time-to-Convert Analysis
# ============================================

print("\n📊 QUERY 5: Time from Signup to First Purchase")
print("-" * 40)

query = """
SELECT 
    u.user_id,
    u.signup_date,
    MIN(e.timestamp) as first_purchase,
    EXTRACT(EPOCH FROM (MIN(e.timestamp) - u.signup_date))/3600 as hours_to_convert
FROM users u
JOIN events e ON u.user_id = e.user_id
WHERE e.event_type = 'purchase_complete'
GROUP BY u.user_id, u.signup_date
ORDER BY hours_to_convert
LIMIT 10
"""

df = pd.read_sql(query, engine)
print(df)

print("\n" + "="*60)
print("✅ All PostgreSQL queries completed!")
print("="*60)