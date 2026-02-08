"""
Data Generation Script for Funnel Analysis
This creates realistic user behavior data for our AI project
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker for realistic fake data
fake = Faker()
np.random.seed(42)  # For reproducibility
random.seed(42)

print("🚀 Starting data generation...")

# ============================================
# CONFIGURATION
# ============================================
NUM_USERS = 50000  # Number of users to generate
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 31)

# ============================================
# 1. GENERATE USERS TABLE
# ============================================
print("📊 Generating users...")

users_data = {
    'user_id': [f'USR_{str(i).zfill(6)}' for i in range(1, NUM_USERS + 1)],
    'signup_date': [fake.date_time_between(start_date=START_DATE, end_date=END_DATE) for _ in range(NUM_USERS)],
    'country': np.random.choice(['USA', 'India', 'UK', 'Canada', 'Germany', 'France', 'Australia'], NUM_USERS, 
                                p=[0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], NUM_USERS, 
                                   p=[0.55, 0.35, 0.10]),
    'acquisition_channel': np.random.choice(['Organic Search', 'Paid Ads', 'Social Media', 'Email', 'Direct'], 
                                           NUM_USERS, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], NUM_USERS,
                                 p=[0.20, 0.35, 0.25, 0.15, 0.05])
}

users_df = pd.DataFrame(users_data)
print(f"✅ Generated {len(users_df)} users")

# ============================================
# 2. GENERATE USER EVENTS (FUNNEL STAGES)
# ============================================
print("🎯 Generating funnel events...")

# Define funnel stages
FUNNEL_STAGES = [
    'homepage_visit',      # Stage 1: Everyone starts here
    'product_view',        # Stage 2: View a product
    'add_to_cart',         # Stage 3: Add product to cart
    'checkout_start',      # Stage 4: Start checkout process
    'payment_info',        # Stage 5: Enter payment details
    'purchase_complete'    # Stage 6: Complete purchase
]

# Conversion rates for each stage (realistic drop-off)
CONVERSION_RATES = {
    'homepage_visit': 1.0,       # 100% visit homepage
    'product_view': 0.70,        # 70% view products
    'add_to_cart': 0.35,         # 35% of product viewers add to cart
    'checkout_start': 0.60,      # 60% of cart users start checkout
    'payment_info': 0.75,        # 75% enter payment info
    'purchase_complete': 0.85    # 85% complete purchase
}

# Different conversion rates by device (Mobile converts less)
DEVICE_MULTIPLIERS = {
    'Mobile': 0.85,
    'Desktop': 1.1,
    'Tablet': 0.95
}

events_list = []

for idx, user in users_df.iterrows():
    user_id = user['user_id']
    device = user['device_type']
    signup_date = user['signup_date']
    
    # Each user can have 1-5 sessions
    num_sessions = random.randint(1, 5)
    
    for session_num in range(num_sessions):
        session_id = f"{user_id}_S{session_num + 1}"
        session_start = signup_date + timedelta(days=random.randint(0, 30), 
                                                hours=random.randint(0, 23))
        
        current_stage_index = 0
        
        # Simulate user going through funnel stages
        for stage in FUNNEL_STAGES:
            # Calculate conversion probability with device multiplier
            base_conversion = CONVERSION_RATES[stage]
            device_multiplier = DEVICE_MULTIPLIERS[device]
            conversion_prob = min(base_conversion * device_multiplier, 1.0)
            
            # Decide if user progresses to this stage
            if random.random() < conversion_prob:
                event_time = session_start + timedelta(minutes=random.randint(1, 30))
                
                # Create event
                event = {
                    'event_id': f"EVT_{len(events_list) + 1}",
                    'user_id': user_id,
                    'session_id': session_id,
                    'timestamp': event_time,
                    'event_type': stage,
                    'device_type': device,
                    'page_url': f"/shop/{stage}",
                    'duration_seconds': random.randint(10, 300),
                    'country': user['country']
                }
                events_list.append(event)
                current_stage_index += 1
            else:
                # User dropped off at this stage
                break

events_df = pd.DataFrame(events_list)
print(f"✅ Generated {len(events_df)} events across {len(events_df['session_id'].unique())} sessions")

# ============================================
# 3. ADD ADVANCED FEATURES
# ============================================
print("⚙️ Adding behavioral features...")

# Add some realistic behavioral patterns
events_df['hour_of_day'] = events_df['timestamp'].dt.hour
events_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
events_df['is_weekend'] = events_df['day_of_week'].isin([5, 6]).astype(int)

# Add click counts (random between 1-20)
events_df['clicks_count'] = np.random.randint(1, 20, len(events_df))

# Add scroll depth percentage
events_df['scroll_depth_percent'] = np.random.randint(20, 100, len(events_df))

print("✅ Features added")

# ============================================
# 4. SAVE TO CSV FILES
# ============================================
print("💾 Saving data to CSV files...")

users_df.to_csv('data/users.csv', index=False)
events_df.to_csv('data/events.csv', index=False)

print("✅ Files saved!")

# ============================================
# 5. DISPLAY SUMMARY STATISTICS
# ============================================
print("\n" + "="*60)
print("📈 DATA GENERATION COMPLETE!")
print("="*60)

print(f"\n👥 USERS TABLE:")
print(f"   Total Users: {len(users_df)}")
print(f"   Countries: {users_df['country'].nunique()}")
print(f"   Device Types: {users_df['device_type'].value_counts().to_dict()}")

print(f"\n🎯 EVENTS TABLE:")
print(f"   Total Events: {len(events_df)}")
print(f"   Total Sessions: {events_df['session_id'].nunique()}")
print(f"   Date Range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")

print(f"\n📊 FUNNEL CONVERSION RATES:")
for stage in FUNNEL_STAGES:
    count = len(events_df[events_df['event_type'] == stage])
    percentage = (count / len(users_df)) * 100
    print(f"   {stage:20s}: {count:6d} users ({percentage:5.1f}%)")

# Calculate overall conversion rate
total_purchases = len(events_df[events_df['event_type'] == 'purchase_complete'])
overall_conversion = (total_purchases / len(users_df)) * 100
print(f"\n🎉 OVERALL CONVERSION RATE: {overall_conversion:.2f}%")

print("\n" + "="*60)
print("📁 Files created:")
print("   - data/users.csv")
print("   - data/events.csv")
print("="*60)

# ============================================
# 6. PREVIEW THE DATA
# ============================================
print("\n👀 PREVIEW - First 5 users:")
print(users_df.head())

print("\n👀 PREVIEW - First 10 events:")
print(events_df.head(10))

print("\n✅ ALL DONE! Your data is ready for analysis! 🚀")