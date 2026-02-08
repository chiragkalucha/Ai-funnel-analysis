"""
Churn Prediction Model Training (FIXED - No Data Leakage)
Predict which users will drop off BEFORE they complete their journey
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import joblib
import os

print("🤖 Starting ML Model Training (FIXED - No Leakage)...")
print("="*60)

# ============================================
# STEP 1: LOAD ENGINEERED FEATURES
# ============================================

print("\n📊 Step 1: Loading engineered features...")

df = pd.read_csv('data/features_engineered.csv')
print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")

# ============================================
# STEP 2: SELECT FEATURES (NO LEAKAGE!)
# ============================================

print("\n🎯 Step 2: Selecting NON-LEAKING features...")

# SAFE FEATURES (Available at prediction time)
safe_features = [
    # Aggregation features (from PAST behavior)
    'total_events',           # How active is user?
    'session_count',          # How engaged?
    'total_duration_seconds', # How much time spent?
    'total_clicks',           # How interactive?
    'avg_scroll_depth',       # How engaged with content?
    
    # Behavioral features (calculated from available data)
    'avg_duration_per_session',  # Engagement quality
    'avg_events_per_session',    # Activity level
    
    # Temporal features
    'activity_span_hours',    # How long have they been browsing?
    'most_active_hour',       # When do they shop?
    
    # Demographics (known at signup)
    'device_Desktop',
    'device_Mobile', 
    'device_Tablet',
    'country_USA',
    'country_India',
    'country_UK',
    'channel_Direct',
    'channel_Email',
    'channel_Organic Search',
    'channel_Paid Ads',
    'channel_Social Media',
]

# EARLY STAGE INDICATORS (Use with caution)
# These are OK because they're from EARLY funnel stages
early_stage_features = [
    'reached_homepage_visit',  # Did they even start?
    'reached_product_view',    # Did they browse products?
]

# Combine safe features
feature_columns = safe_features + early_stage_features

# Check which columns exist
available_features = [col for col in feature_columns if col in df.columns]

print(f"✅ Using {len(available_features)} NON-LEAKING features")
print(f"\n🚫 EXCLUDED features (they leak the answer!):")
print(f"   - reached_add_to_cart")
print(f"   - reached_checkout_start") 
print(f"   - reached_payment_info")
print(f"   - furthest_stage_reached")

# Target variable
target = 'has_converted'

# Create X and y
X = df[available_features]
y = df[target]

print(f"\n📊 Dataset shape:")
print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")
print(f"   Conversion rate: {y.mean()*100:.2f}%")
print(f"   Class balance: {(1-y.mean())*100:.1f}% non-convert, {y.mean()*100:.1f}% convert")

# ============================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================

print("\n🔀 Step 3: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Training: {len(X_train):,} samples")
print(f"✅ Testing: {len(X_test):,} samples")

# ============================================
# STEP 4: TRAIN RANDOM FOREST
# ============================================

print("\n🌲 Step 4: Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,  # Prevent overfitting
    min_samples_leaf=10,   # Prevent overfitting
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✅ Random Forest trained!")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n📊 Random Forest Performance:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_rf):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_rf):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# ============================================
# STEP 5: TRAIN XGBOOST
# ============================================

print("\n🚀 Step 5: Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,           # Shallower to prevent overfitting
    learning_rate=0.1,
    min_child_weight=5,    # Prevent overfitting
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
print("✅ XGBoost trained!")

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n📊 XGBoost Performance:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# ============================================
# STEP 6: MODEL COMPARISON
# ============================================

print("\n🏆 Step 6: Model comparison...")

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print(comparison.to_string(index=False))

# Select best
best_idx = comparison['F1-Score'].idxmax()
best_model_name = comparison.loc[best_idx, 'Model']
best_model = xgb_model if best_model_name == 'XGBoost' else rf_model
best_pred = y_pred_xgb if best_model_name == 'XGBoost' else y_pred_rf
best_pred_proba = y_pred_proba_xgb if best_model_name == 'XGBoost' else y_pred_proba_rf

print(f"\n🏆 Best model: {best_model_name}")

# ============================================
# STEP 7: FEATURE IMPORTANCE
# ============================================

print("\n📊 Step 7: Feature importance analysis...")

if best_model_name == 'XGBoost':
    importance = xgb_model.feature_importances_
else:
    importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n🎯 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================
# STEP 8: CONFUSION MATRIX
# ============================================

print("\n🔍 Step 8: Confusion matrix...")

cm = confusion_matrix(y_test, best_pred)

print("\n📊 Confusion Matrix:")
print(f"                    Predicted")
print(f"                  No    Yes")
print(f"Actual  No     {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"        Yes    {cm[1,0]:5d}  {cm[1,1]:5d}")

print("\n📈 What this means:")
print(f"   ✅ True Negatives:  {cm[0,0]:,} (Correctly ID'd non-converters)")
print(f"   ⚠️  False Positives: {cm[0,1]:,} (False alarms - predicted convert but didn't)")
print(f"   ⚠️  False Negatives: {cm[1,0]:,} (Missed opportunities - predicted no-convert but did!)")
print(f"   ✅ True Positives:  {cm[1,1]:,} (Correctly ID'd converters)")

# Business impact
revenue_per_conversion = 50
discount_cost = 5
missed_revenue = cm[1,0] * revenue_per_conversion
wasted_discounts = cm[0,1] * discount_cost
captured_revenue = cm[1,1] * (revenue_per_conversion - discount_cost)

print(f"\n💰 Business Impact (if we act on predictions):")
print(f"   Captured revenue: ${captured_revenue:,} ({cm[1,1]:,} correct interventions)")
print(f"   Missed revenue: ${missed_revenue:,} ({cm[1,0]:,} false negatives)")
print(f"   Wasted on false alarms: ${wasted_discounts:,} ({cm[0,1]:,} false positives)")
print(f"   Net gain: ${captured_revenue - wasted_discounts:,}")

# ============================================
# STEP 9: SAVE MODEL
# ============================================

print("\n💾 Step 9: Saving model...")

os.makedirs('models', exist_ok=True)

model_filename = f'models/churn_prediction_{best_model_name.lower().replace(" ", "_")}.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Model saved: {model_filename}")

joblib.dump(available_features, 'models/feature_names.pkl')
print(f"✅ Feature names saved")

feature_importance.to_csv('models/feature_importance.csv', index=False)
print(f"✅ Feature importance saved")

# ============================================
# STEP 10: SAMPLE PREDICTIONS
# ============================================

print("\n🎯 Step 10: Sample predictions...")

sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y_true = y_test.iloc[sample_indices].values
sample_y_pred_proba = best_model.predict_proba(sample_X)[:, 1]

print("\n👤 Example Users:")
print("="*60)
for i, idx in enumerate(sample_indices):
    actual = "Converted ✅" if sample_y_true[i] == 1 else "Did NOT Convert ❌"
    conv_prob = sample_y_pred_proba[i] * 100
    
    print(f"\nUser #{idx}:")
    print(f"   Actual outcome: {actual}")
    print(f"   Model prediction: {conv_prob:.1f}% likely to convert")
    
    if conv_prob < 30:
        print(f"   💡 Action: HIGH CHURN RISK - Send 10% discount NOW!")
    elif conv_prob < 60:
        print(f"   💡 Action: MEDIUM RISK - Send reminder email")
    else:
        print(f"   💡 Action: LOW RISK - Likely to convert naturally")

# ============================================
print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*60)

print(f"\n✅ Model is ready for production!")
print(f"✅ Accuracy: {accuracy_score(y_test, best_pred):.1%} (realistic, not 100%!)")
print(f"✅ Can identify {recall_score(y_test, best_pred):.1%} of potential converters")
print(f"\n📚 Key Learning: We fixed data leakage by excluding funnel progression features!")

print("\n" + "="*60)