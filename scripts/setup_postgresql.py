"""
PostgreSQL Database Setup
Creates database, tables, and loads data from CSV
"""

import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

print("🐘 PostgreSQL Database Setup Starting...")
print("="*60)

# ============================================
# 1. CREATE DATABASE
# ============================================

print("\n📦 Step 1: Creating database...")

# First, connect to the default 'postgres' database to create our database
try:
    # Connect to default postgres database
    conn = psycopg2.connect(
        host='localhost',
        user='postgres',
        password='yonex.Chirag1',
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Drop database if exists (for clean start)
    cursor.execute("DROP DATABASE IF EXISTS funnel_analysis")
    print("   🗑️  Dropped existing database (if any)")
    
    # Create new database
    cursor.execute("CREATE DATABASE funnel_analysis")
    print("   ✅ Created database: funnel_analysis")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Make sure PostgreSQL is running and password is correct!")
    exit(1)

# ============================================
# 2. CONNECT TO OUR NEW DATABASE
# ============================================

print("\n🔌 Step 2: Connecting to funnel_analysis database...")

# Create SQLAlchemy engine for our new database
engine = create_engine('postgresql://postgres:yonex.Chirag1@localhost:5432/funnel_analysis')

print("   ✅ Connected successfully!")

# ============================================
# 3. CREATE TABLES WITH PROPER SCHEMA
# ============================================

print("\n📋 Step 3: Creating tables with schema...")

with engine.connect() as conn:
    
    # Create USERS table
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(20) PRIMARY KEY,
        signup_date TIMESTAMP NOT NULL,
        country VARCHAR(50),
        device_type VARCHAR(20),
        acquisition_channel VARCHAR(50),
        age_group VARCHAR(20)
    )
    """
    conn.execute(text(create_users_table))
    print("   ✅ Created 'users' table")
    
    # Create EVENTS table
    create_events_table = """
    CREATE TABLE IF NOT EXISTS events (
        event_id VARCHAR(20) PRIMARY KEY,
        user_id VARCHAR(20) REFERENCES users(user_id),
        session_id VARCHAR(50),
        timestamp TIMESTAMP NOT NULL,
        event_type VARCHAR(50),
        device_type VARCHAR(20),
        page_url VARCHAR(200),
        duration_seconds INTEGER,
        country VARCHAR(50),
        hour_of_day INTEGER,
        day_of_week INTEGER,
        is_weekend INTEGER,
        clicks_count INTEGER,
        scroll_depth_percent INTEGER
    )
    """
    conn.execute(text(create_events_table))
    print("   ✅ Created 'events' table")
    
    conn.commit()

# ============================================
# 4. LOAD DATA FROM CSV
# ============================================

print("\n📊 Step 4: Loading data from CSV files...")

# Read CSV files
users_df = pd.read_csv('data/users.csv')
events_df = pd.read_csv('data/events.csv')

print(f"   📁 Loaded {len(users_df):,} users from CSV")
print(f"   📁 Loaded {len(events_df):,} events from CSV")

# Convert date columns to datetime
users_df['signup_date'] = pd.to_datetime(users_df['signup_date'])
events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

# ============================================
# 5. INSERT DATA INTO DATABASE
# ============================================

print("\n💾 Step 5: Inserting data into PostgreSQL...")
print("   ⏳ This may take 1-2 minutes for 50K users...")

# Insert users (faster method using to_sql)
users_df.to_sql('users', engine, if_exists='append', index=False, method='multi', chunksize=1000)
print(f"   ✅ Inserted {len(users_df):,} users")

# Insert events (in chunks for better performance)
print("   ⏳ Inserting events (this is the big one)...")
events_df.to_sql('events', engine, if_exists='append', index=False, method='multi', chunksize=5000)
print(f"   ✅ Inserted {len(events_df):,} events")

# ============================================
# 6. CREATE INDEXES FOR PERFORMANCE
# ============================================

print("\n⚡ Step 6: Creating indexes for fast queries...")

with engine.connect() as conn:
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
        "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_device ON events(device_type)",
        "CREATE INDEX IF NOT EXISTS idx_users_device ON users(device_type)",
        "CREATE INDEX IF NOT EXISTS idx_users_channel ON users(acquisition_channel)",
    ]
    
    for idx_query in indexes:
        conn.execute(text(idx_query))
        index_name = idx_query.split("INDEX IF NOT EXISTS ")[1].split(" ON")[0]
        print(f"   ✅ Created index: {index_name}")
    
    conn.commit()

# ============================================
# 7. CREATE ANALYTICS VIEWS
# ============================================

print("\n👁️  Step 7: Creating analytics views...")

with engine.connect() as conn:
    
    # View 1: User conversion status
    conversion_view = """
    CREATE OR REPLACE VIEW user_conversions AS
    SELECT 
        u.user_id,
        u.device_type,
        u.country,
        u.acquisition_channel,
        u.signup_date,
        CASE 
            WHEN EXISTS (
                SELECT 1 FROM events e 
                WHERE e.user_id = u.user_id 
                AND e.event_type = 'purchase_complete'
            ) THEN 1 
            ELSE 0 
        END as has_converted
    FROM users u
    """
    conn.execute(text(conversion_view))
    print("   ✅ Created view: user_conversions")
    
    # View 2: Funnel metrics
    funnel_view = """
    CREATE OR REPLACE VIEW funnel_metrics AS
    SELECT 
        event_type,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(*) as total_events,
        AVG(duration_seconds) as avg_duration
    FROM events
    GROUP BY event_type
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
    conn.execute(text(funnel_view))
    print("   ✅ Created view: funnel_metrics")
    
    conn.commit()

# ============================================
# 8. VERIFY DATA
# ============================================

print("\n🔍 Step 8: Verifying data...")

with engine.connect() as conn:
    
    # Count users
    result = conn.execute(text("SELECT COUNT(*) FROM users"))
    user_count = result.fetchone()[0]
    print(f"   ✅ Users in database: {user_count:,}")
    
    # Count events
    result = conn.execute(text("SELECT COUNT(*) FROM events"))
    event_count = result.fetchone()[0]
    print(f"   ✅ Events in database: {event_count:,}")
    
    # Count converters
    result = conn.execute(text("SELECT COUNT(*) FROM user_conversions WHERE has_converted = 1"))
    converter_count = result.fetchone()[0]
    print(f"   ✅ Users who converted: {converter_count:,}")
    
    # Show funnel
    print("\n   📊 Funnel Overview:")
    result = conn.execute(text("SELECT * FROM funnel_metrics"))
    for row in result:
        print(f"      {row[0]:20s}: {row[1]:,} users")

# ============================================
# 9. DATABASE INFO
# ============================================

print("\n" + "="*60)
print("🎉 POSTGRESQL SETUP COMPLETE!")
print("="*60)

print("\n📊 Database Details:")
print(f"   Host: localhost")
print(f"   Port: 5432")
print(f"   Database: funnel_analysis")
print(f"   Username: postgres")
print(f"   Password: postgres123")

print("\n📋 Created Tables:")
print("   1. users - User profile data")
print("   2. events - User behavior events")

print("\n👁️  Created Views:")
print("   1. user_conversions - User conversion status")
print("   2. funnel_metrics - Funnel stage metrics")

print("\n⚡ Created Indexes:")
print("   7 indexes for optimized query performance")

print("\n🔗 Connection String:")
print("   postgresql://postgres:postgres123@localhost:5432/funnel_analysis")

print("\n✅ You can now:")
print("   - Query with SQL")
print("   - Connect via pgAdmin")
print("   - Run analytics")
print("   - Build ML models")

print("\n" + "="*60)