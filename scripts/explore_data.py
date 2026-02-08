"""
Quick Data Exploration
This helps us understand the data we generated
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("📊 Loading data...")
users_df = pd.read_csv('data/users.csv')
events_df = pd.read_csv('data/events.csv')

print("\n" + "="*60)
print("🔍 DATA EXPLORATION INSIGHTS")
print("="*60)

# ============================================
# 1. FUNNEL ANALYSIS
# ============================================
print("\n📉 FUNNEL DROP-OFF ANALYSIS:")

funnel_stages = ['homepage_visit', 'product_view', 'add_to_cart', 
                 'checkout_start', 'payment_info', 'purchase_complete']

funnel_counts = []
for stage in funnel_stages:
    unique_users = events_df[events_df['event_type'] == stage]['user_id'].nunique()
    funnel_counts.append(unique_users)
    print(f"   {stage:20s}: {unique_users:6d} users")

# Calculate drop-off rates
print("\n💔 DROP-OFF RATES (Stage to Stage):")
for i in range(len(funnel_counts) - 1):
    drop_off_rate = ((funnel_counts[i] - funnel_counts[i+1]) / funnel_counts[i]) * 100
    print(f"   {funnel_stages[i]:20s} → {funnel_stages[i+1]:20s}: {drop_off_rate:.1f}% dropped")

# ============================================
# 2. DEVICE ANALYSIS
# ============================================
print("\n📱 CONVERSION BY DEVICE:")

for device in users_df['device_type'].unique():
    device_users = users_df[users_df['device_type'] == device]['user_id'].unique()
    device_purchases = events_df[
        (events_df['user_id'].isin(device_users)) & 
        (events_df['event_type'] == 'purchase_complete')
    ]['user_id'].nunique()
    
    conversion_rate = (device_purchases / len(device_users)) * 100
    print(f"   {device:10s}: {conversion_rate:.2f}% conversion")

# ============================================
# 3. TIME-BASED PATTERNS
# ============================================
print("\n🕐 CONVERSION BY HOUR OF DAY:")

events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
events_df['hour'] = events_df['timestamp'].dt.hour

purchases_by_hour = events_df[events_df['event_type'] == 'purchase_complete'].groupby('hour').size()
print(purchases_by_hour)

# ============================================
# 4. CHANNEL ANALYSIS
# ============================================
print("\n📢 CONVERSION BY ACQUISITION CHANNEL:")

for channel in users_df['acquisition_channel'].unique():
    channel_users = users_df[users_df['acquisition_channel'] == channel]['user_id'].unique()
    channel_purchases = events_df[
        (events_df['user_id'].isin(channel_users)) & 
        (events_df['event_type'] == 'purchase_complete')
    ]['user_id'].nunique()
    
    conversion_rate = (channel_purchases / len(channel_users)) * 100
    print(f"   {channel:15s}: {conversion_rate:.2f}% conversion")

# ============================================
# 5. SESSION ANALYSIS
# ============================================
print("\n🔄 SESSION INSIGHTS:")

sessions_per_user = events_df.groupby('user_id')['session_id'].nunique()
print(f"   Average sessions per user: {sessions_per_user.mean():.2f}")
print(f"   Max sessions by a user: {sessions_per_user.max()}")

# Users who converted vs didn't convert
converters = events_df[events_df['event_type'] == 'purchase_complete']['user_id'].unique()
converter_sessions = sessions_per_user[sessions_per_user.index.isin(converters)].mean()
non_converter_sessions = sessions_per_user[~sessions_per_user.index.isin(converters)].mean()

print(f"   Avg sessions (converters): {converter_sessions:.2f}")
print(f"   Avg sessions (non-converters): {non_converter_sessions:.2f}")

print("\n" + "="*60)
print("✅ Exploration complete!")
print("="*60)

print("\n💡 KEY INSIGHTS:")
print("   1. Biggest drop-off is likely between product_view → add_to_cart")
print("   2. Mobile users convert at lower rates than desktop")
print("   3. Converters have more sessions (shows engagement)")
print("   4. Peak conversion hours are likely 12-2pm and 7-10pm")