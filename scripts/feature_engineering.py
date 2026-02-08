"""
Feature Engineering for Churn Prediction (OPTIMIZED)
Transform raw user/event data into ML-ready features
"""

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import numpy as np

print("🔧 Starting Feature Engineering (Optimized)...")
print("="*60)

# UPDATE WITH YOUR PASSWORD
PASSWORD = 'yonex.Chirag1'

# Connect to database
engine = create_engine(f'postgresql://postgres:{PASSWORD}@localhost:5432/funnel_analysis')

print("\n📊 Step 1: Loading data from database...")

# Load users
users_df = pd.read_sql("SELECT * FROM users", engine)
print(f"✅ Loaded {len(users_df):,} users")

# Load events
events_df = pd.read_sql("SELECT * FROM events", engine)
print(f"✅ Loaded {len(events_df):,} events")

# ============================================
# FEATURE ENGINEERING STARTS HERE
# ============================================

print("\n🎯 Step 2: Creating aggregation features...")

# Group events by user
user_features = events_df.groupby('user_id').agg({
    'event_id': 'count',              # Total events per user
    'session_id': 'nunique',          # Number of unique sessions
    'duration_seconds': 'sum',        # Total time spent
    'clicks_count': 'sum',            # Total clicks
    'scroll_depth_percent': 'mean'    # Average scroll depth
}).reset_index()

# Rename columns for clarity
user_features.columns = [
    'user_id',
    'total_events',
    'session_count',
    'total_duration_seconds',
    'total_clicks',
    'avg_scroll_depth'
]

print(f"✅ Created {len(user_features.columns)-1} aggregation features")

# ============================================
print("\n🎯 Step 3: Creating funnel stage features...")

# Create binary flags for each funnel stage
funnel_stages = [
    'homepage_visit',
    'product_view', 
    'add_to_cart',
    'checkout_start',
    'payment_info',
    'purchase_complete'
]

for stage in funnel_stages:
    # Check if user reached this stage
    stage_users = events_df[events_df['event_type'] == stage]['user_id'].unique()
    user_features[f'reached_{stage}'] = user_features['user_id'].isin(stage_users).astype(int)

print(f"✅ Created {len(funnel_stages)} funnel stage features")

# ============================================
print("\n🎯 Step 4: Creating behavioral features (OPTIMIZED)...")

# OPTIMIZED VERSION - Calculate furthest stage for ALL users at once
stage_order = {stage: i for i, stage in enumerate(funnel_stages)}

# Map event types to their stage numbers
events_df['stage_number'] = events_df['event_type'].map(stage_order).fillna(0)

# Get max stage per user (FAST!)
furthest_stages = events_df.groupby('user_id')['stage_number'].max().reset_index()
furthest_stages.columns = ['user_id', 'furthest_stage_reached']

# Merge with user_features
user_features = user_features.merge(furthest_stages, on='user_id', how='left')
user_features['furthest_stage_reached'] = user_features['furthest_stage_reached'].fillna(0)

print(f"   ✅ Calculated furthest stage reached")

# Average time per session
user_features['avg_duration_per_session'] = (
    user_features['total_duration_seconds'] / user_features['session_count']
)

# Events per session
user_features['avg_events_per_session'] = (
    user_features['total_events'] / user_features['session_count']
)

print(f"✅ Created 3 behavioral features")

# ============================================
print("\n🎯 Step 5: Creating temporal features...")

# Get first and last event for each user
user_temporal = events_df.groupby('user_id').agg({
    'timestamp': ['min', 'max']
}).reset_index()

user_temporal.columns = ['user_id', 'first_event_time', 'last_event_time']

# Convert to datetime
user_temporal['first_event_time'] = pd.to_datetime(user_temporal['first_event_time'])
user_temporal['last_event_time'] = pd.to_datetime(user_temporal['last_event_time'])

# Calculate time span
user_temporal['activity_span_hours'] = (
    (user_temporal['last_event_time'] - user_temporal['first_event_time'])
    .dt.total_seconds() / 3600
)

# Merge with main features
user_features = user_features.merge(user_temporal[['user_id', 'activity_span_hours']], on='user_id')

# Get most common hour of activity
user_hour_mode = events_df.groupby('user_id')['hour_of_day'].agg(
    lambda x: x.mode()[0] if len(x.mode()) > 0 else 12
).reset_index()
user_hour_mode.columns = ['user_id', 'most_active_hour']

user_features = user_features.merge(user_hour_mode, on='user_id')

print(f"✅ Created 2 temporal features")

# ============================================
print("\n🎯 Step 6: Merging with user demographic data...")

# Merge with users table
final_features = users_df.merge(user_features, on='user_id', how='left')

# Fill NaN values (users with no events)
final_features = final_features.fillna(0)

print(f"✅ Merged user demographics")

# ============================================
print("\n🎯 Step 7: Creating target variable (LABEL)...")

# Target: Did user convert? (1 = Yes, 0 = No)
final_features['has_converted'] = final_features['reached_purchase_complete']

# Alternative target: Did user drop off early?
final_features['churned'] = (final_features['furthest_stage_reached'] < 3).astype(int)

print(f"✅ Created target variables")

# ============================================
print("\n🎯 Step 8: Encoding categorical features...")

# One-hot encode device_type
device_dummies = pd.get_dummies(final_features['device_type'], prefix='device')
final_features = pd.concat([final_features, device_dummies], axis=1)

# One-hot encode country
country_dummies = pd.get_dummies(final_features['country'], prefix='country')
final_features = pd.concat([final_features, country_dummies], axis=1)

# One-hot encode acquisition_channel
channel_dummies = pd.get_dummies(final_features['acquisition_channel'], prefix='channel')
final_features = pd.concat([final_features, channel_dummies], axis=1)

# One-hot encode age_group
age_dummies = pd.get_dummies(final_features['age_group'], prefix='age')
final_features = pd.concat([final_features, age_dummies], axis=1)

print(f"✅ One-hot encoded categorical features")

# ============================================
print("\n💾 Step 9: Saving engineered features...")

# Save to CSV
final_features.to_csv('data/features_engineered.csv', index=False)
print(f"✅ Saved to: data/features_engineered.csv")

# Also save to database
final_features.to_sql('features_ml', engine, if_exists='replace', index=False)
print(f"✅ Saved to database table: features_ml")

# ============================================
print("\n📊 Step 10: Feature Summary...")

print(f"\n📈 FEATURE ENGINEERING COMPLETE!")
print("="*60)
print(f"Total users: {len(final_features):,}")
print(f"Total features: {len(final_features.columns)}")
print(f"Converters: {final_features['has_converted'].sum():,} ({final_features['has_converted'].mean()*100:.2f}%)")
print(f"Churned users: {final_features['churned'].sum():,} ({final_features['churned'].mean()*100:.2f}%)")

print(f"\n🎯 Feature Categories:")
print(f"   Aggregation features: 6")
print(f"   Funnel stage features: {len(funnel_stages)}")
print(f"   Behavioral features: 3")
print(f"   Temporal features: 2")
print(f"   Categorical features: {len(device_dummies.columns) + len(country_dummies.columns) + len(channel_dummies.columns) + len(age_dummies.columns)}")

print(f"\n👀 Sample of engineered features:")
print(final_features[['user_id', 'total_events', 'session_count', 'furthest_stage_reached', 'has_converted']].head(10))

print(f"\n✅ Features ready for machine learning!")
print("="*60)