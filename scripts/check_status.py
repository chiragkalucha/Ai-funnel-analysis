"""
Project Status Check
Verify everything is working
"""

from sqlalchemy import create_engine, text
import pandas as pd

# UPDATE THIS WITH YOUR PASSWORD
PASSWORD = 'postgres123'  # Change if needed

engine = create_engine(f'postgresql://postgres:yonex.Chirag1@localhost:5432/funnel_analysis')

print("="*60)
print("📊 AI FUNNEL ANALYSIS PROJECT - STATUS CHECK")
print("="*60)

with engine.connect() as conn:
    
    # Check 1: Tables
    print("\n✅ TABLES:")
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """))
    for row in result:
        print(f"   - {row[0]}")
    
    # Check 2: Views
    print("\n✅ VIEWS:")
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_type = 'VIEW'
    """))
    for row in result:
        print(f"   - {row[0]}")
    
    # Check 3: Record counts
    print("\n✅ DATA COUNTS:")
    
    result = conn.execute(text("SELECT COUNT(*) FROM users"))
    user_count = result.fetchone()[0]
    print(f"   Users: {user_count:,}")
    
    result = conn.execute(text("SELECT COUNT(*) FROM events"))
    event_count = result.fetchone()[0]
    print(f"   Events: {event_count:,}")
    
    result = conn.execute(text("SELECT COUNT(DISTINCT session_id) FROM events"))
    session_count = result.fetchone()[0]
    print(f"   Sessions: {session_count:,}")
    
    # Check 4: Conversion metrics
    print("\n✅ KEY METRICS:")
    
    result = conn.execute(text("""
        SELECT COUNT(*) FROM user_conversions WHERE has_converted = 1
    """))
    converters = result.fetchone()[0]
    conversion_rate = (converters / user_count) * 100
    print(f"   Converters: {converters:,} ({conversion_rate:.2f}%)")
    
    # Check 5: Funnel breakdown
    print("\n✅ FUNNEL STAGES:")
    result = conn.execute(text("SELECT * FROM funnel_metrics"))
    for row in result:
        print(f"   {row[0]:20s}: {row[1]:,} users")
    
    # Check 6: Top performing segment
    print("\n✅ BEST PERFORMING DEVICE:")
    result = conn.execute(text("""
        SELECT 
            device_type,
            ROUND(100.0 * SUM(has_converted) / COUNT(*), 2) as conversion_rate
        FROM user_conversions
        GROUP BY device_type
        ORDER BY conversion_rate DESC
        LIMIT 1
    """))
    row = result.fetchone()
    print(f"   {row[0]}: {row[1]}% conversion rate")

print("\n" + "="*60)
print("🎉 ALL SYSTEMS OPERATIONAL!")
print("="*60)
print("\n📈 READY FOR: Machine Learning & Analytics")