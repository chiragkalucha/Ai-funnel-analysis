"""
Model Performance Visualizations
Create charts for model evaluation and business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("📊 Creating Visualizations...")
print("="*60)

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# ============================================
# LOAD DATA & MODEL
# ============================================

print("\n📂 Loading data and model...")

# Load features
df = pd.read_csv('data/features_engineered.csv')

# Load model and feature names
model = joblib.load('models/churn_prediction_random_forest.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare data
X = df[feature_names]
y = df['has_converted']

# Split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("✅ Data and model loaded")

# ============================================
# VISUALIZATION 1: CONFUSION MATRIX HEATMAP
# ============================================

print("\n📊 Creating Confusion Matrix heatmap...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Did Not Convert', 'Converted'],
            yticklabels=['Did Not Convert', 'Converted'],
            cbar_kws={'label': 'Count'})

plt.title('Confusion Matrix - Churn Prediction Model', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/1_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 1_confusion_matrix.png")

# ============================================
# VISUALIZATION 2: ROC CURVE
# ============================================

print("\n📊 Creating ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve - Model Performance', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/2_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 2_roc_curve.png")

# ============================================
# VISUALIZATION 3: FEATURE IMPORTANCE
# ============================================

print("\n📊 Creating Feature Importance chart...")

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['Importance'], 
         color='steelblue', edgecolor='black')
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 15 Most Important Features', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 3_feature_importance.png")

# ============================================
# VISUALIZATION 4: PREDICTION DISTRIBUTION
# ============================================

print("\n📊 Creating Prediction Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Converters
converters_proba = y_pred_proba[y_test == 1]
ax1.hist(converters_proba, bins=30, color='green', alpha=0.7, edgecolor='black')
ax1.axvline(converters_proba.mean(), color='darkgreen', linestyle='--', 
            linewidth=2, label=f'Mean: {converters_proba.mean():.2f}')
ax1.set_xlabel('Predicted Conversion Probability', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Distribution for ACTUAL CONVERTERS', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Non-converters
non_converters_proba = y_pred_proba[y_test == 0]
ax2.hist(non_converters_proba, bins=30, color='red', alpha=0.7, edgecolor='black')
ax2.axvline(non_converters_proba.mean(), color='darkred', linestyle='--', 
            linewidth=2, label=f'Mean: {non_converters_proba.mean():.2f}')
ax2.set_xlabel('Predicted Conversion Probability', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Distribution for ACTUAL NON-CONVERTERS', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/4_prediction_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 4_prediction_distribution.png")

# ============================================
# VISUALIZATION 5: FUNNEL CONVERSION RATES
# ============================================

print("\n📊 Creating Funnel Conversion chart...")

# Load events data
events_df = pd.read_sql("SELECT * FROM events", 
                        f"postgresql://postgres:yonex.Chirag1@localhost:5432/funnel_analysis")

funnel_stages = [
    'homepage_visit',
    'product_view',
    'add_to_cart',
    'checkout_start',
    'payment_info',
    'purchase_complete'
]

funnel_counts = []
for stage in funnel_stages:
    count = events_df[events_df['event_type'] == stage]['user_id'].nunique()
    funnel_counts.append(count)

# Create funnel visualization
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
y_pos = np.arange(len(funnel_stages))

bars = ax.barh(y_pos, funnel_counts, color=colors, edgecolor='black', linewidth=1.5)

# Add labels
for i, (bar, count) in enumerate(zip(bars, funnel_counts)):
    percentage = (count / funnel_counts[0]) * 100
    drop_off = 0 if i == 0 else ((funnel_counts[i-1] - count) / funnel_counts[i-1]) * 100
    
    ax.text(count + 500, bar.get_y() + bar.get_height()/2, 
            f'{count:,} users ({percentage:.1f}%)', 
            va='center', fontsize=11, fontweight='bold')
    
    if i > 0:
        ax.text(10, bar.get_y() + bar.get_height()/2 - 0.3, 
                f'Drop: {drop_off:.1f}%', 
                va='center', fontsize=9, color='red', style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace('_', ' ').title() for s in funnel_stages])
ax.set_xlabel('Number of Users', fontsize=12)
ax.set_title('Conversion Funnel Analysis', fontsize=16, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/5_funnel_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 5_funnel_analysis.png")

# ============================================
# VISUALIZATION 6: CONVERSION BY DEVICE
# ============================================

print("\n📊 Creating Conversion by Device chart...")

device_conversion = df.groupby('device_type').agg({
    'user_id': 'count',
    'has_converted': 'sum'
}).reset_index()

device_conversion['conversion_rate'] = (
    device_conversion['has_converted'] / device_conversion['user_id'] * 100
)
device_conversion.columns = ['Device', 'Total Users', 'Converters', 'Conversion Rate']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
colors_device = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(device_conversion['Device'], 
               device_conversion['Conversion Rate'],
               color=colors_device, edgecolor='black', linewidth=1.5)

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Conversion Rate (%)', fontsize=12)
ax1.set_title('Conversion Rate by Device Type', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Pie chart
ax2.pie(device_conversion['Total Users'], 
        labels=device_conversion['Device'],
        autopct='%1.1f%%',
        colors=colors_device,
        startangle=90,
        explode=(0.05, 0.05, 0.05),
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('User Distribution by Device', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/6_conversion_by_device.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 6_conversion_by_device.png")

# ============================================
# VISUALIZATION 7: CONVERSION BY CHANNEL
# ============================================

print("\n📊 Creating Conversion by Channel chart...")

channel_conversion = df.groupby('acquisition_channel').agg({
    'user_id': 'count',
    'has_converted': 'sum'
}).reset_index()

channel_conversion['conversion_rate'] = (
    channel_conversion['has_converted'] / channel_conversion['user_id'] * 100
)
channel_conversion = channel_conversion.sort_values('conversion_rate', ascending=False)
channel_conversion.columns = ['Channel', 'Total Users', 'Converters', 'Conversion Rate']

plt.figure(figsize=(12, 6))
bars = plt.barh(channel_conversion['Channel'], 
                channel_conversion['Conversion Rate'],
                color='coral', edgecolor='black', linewidth=1.5)

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
             f'{width:.1f}%',
             va='center', fontsize=11, fontweight='bold')

plt.xlabel('Conversion Rate (%)', fontsize=12)
plt.title('Conversion Rate by Acquisition Channel', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/7_conversion_by_channel.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 7_conversion_by_channel.png")

# ============================================
# VISUALIZATION 8: SESSION vs CONVERSION
# ============================================

print("\n📊 Creating Session Count vs Conversion chart...")

session_groups = pd.cut(df['session_count'], bins=[0, 1, 2, 3, 5, 10, 100], 
                        labels=['1', '2', '3', '4-5', '6-10', '10+'])

session_conversion = df.groupby(session_groups).agg({
    'user_id': 'count',
    'has_converted': 'sum'
}).reset_index()

session_conversion['conversion_rate'] = (
    session_conversion['has_converted'] / session_conversion['user_id'] * 100
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Conversion rate by sessions
ax1.plot(range(len(session_conversion)), 
         session_conversion['conversion_rate'],
         marker='o', markersize=10, linewidth=3, color='purple')

for i, row in session_conversion.iterrows():
    ax1.text(i, row['conversion_rate'] + 1, 
             f"{row['conversion_rate']:.1f}%",
             ha='center', fontsize=10, fontweight='bold')

ax1.set_xticks(range(len(session_conversion)))
ax1.set_xticklabels(session_conversion['session_count'])
ax1.set_xlabel('Number of Sessions', fontsize=12)
ax1.set_ylabel('Conversion Rate (%)', fontsize=12)
ax1.set_title('Conversion Rate by Session Count', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

# User distribution
ax2.bar(range(len(session_conversion)), 
        session_conversion['user_id'],
        color='teal', edgecolor='black', linewidth=1.5)

ax2.set_xticks(range(len(session_conversion)))
ax2.set_xticklabels(session_conversion['session_count'])
ax2.set_xlabel('Number of Sessions', fontsize=12)
ax2.set_ylabel('Number of Users', fontsize=12)
ax2.set_title('User Distribution by Sessions', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/8_session_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 8_session_analysis.png")

# ============================================
# VISUALIZATION 9: BUSINESS IMPACT
# ============================================

print("\n📊 Creating Business Impact chart...")

# Calculate scenarios
revenue_per_conversion = 50
discount_cost = 5
total_test_users = len(y_test)

# Scenario 1: No ML
baseline_conversions = y_test.sum()
baseline_revenue = baseline_conversions * revenue_per_conversion

# Scenario 2: With ML
tp = cm[1, 1]  # True positives
fp = cm[0, 1]  # False positives
fn = cm[1, 0]  # False negatives

ml_revenue = tp * (revenue_per_conversion - discount_cost)
ml_cost = fp * discount_cost
ml_missed = fn * revenue_per_conversion

scenarios = pd.DataFrame({
    'Scenario': ['Without ML\n(Baseline)', 'With ML\n(Our Model)'],
    'Revenue': [baseline_revenue, ml_revenue],
    'Cost': [0, ml_cost],
    'Net': [baseline_revenue, ml_revenue - ml_cost]
})

fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(len(scenarios))
width = 0.25

bars1 = ax.bar(x - width, scenarios['Revenue'], width, 
               label='Revenue', color='green', edgecolor='black')
bars2 = ax.bar(x, scenarios['Cost'], width, 
               label='Cost', color='red', edgecolor='black')
bars3 = ax.bar(x + width, scenarios['Net'], width, 
               label='Net Profit', color='blue', edgecolor='black')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Amount ($)', fontsize=12)
ax.set_title('Business Impact: ML vs Baseline', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios['Scenario'], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add improvement annotation
improvement = ((ml_revenue - ml_cost) / baseline_revenue - 1) * 100
ax.text(1, max(scenarios['Net']) * 1.1, 
        f'Improvement: +{improvement:.1f}%',
        ha='center', fontsize=13, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('visualizations/9_business_impact.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 9_business_impact.png")

# ============================================
# CREATE SUMMARY REPORT
# ============================================

print("\n📋 Creating summary metrics file...")

summary = {
    'Model Performance': {
        'Accuracy': f"{(cm[0,0] + cm[1,1]) / cm.sum():.2%}",
        'Precision': f"{cm[1,1] / (cm[1,1] + cm[0,1]):.2%}",
        'Recall': f"{cm[1,1] / (cm[1,1] + cm[1,0]):.2%}",
        'ROC-AUC': f"{roc_auc:.3f}"
    },
    'Confusion Matrix': {
        'True Negatives': int(cm[0,0]),
        'False Positives': int(cm[0,1]),
        'False Negatives': int(cm[1,0]),
        'True Positives': int(cm[1,1])
    },
    'Business Impact': {
        'Baseline Revenue': f"${baseline_revenue:,}",
        'ML Revenue': f"${ml_revenue:,}",
        'ML Cost': f"${ml_cost:,}",
        'Net Profit': f"${ml_revenue - ml_cost:,}",
        'Improvement': f"{improvement:.1f}%"
    }
}

with open('visualizations/summary_metrics.txt', 'w') as f:
    for section, metrics in summary.items():
        f.write(f"\n{'='*50}\n")
        f.write(f"{section}\n")
        f.write(f"{'='*50}\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value}\n")

print("✅ Saved: summary_metrics.txt")

print("\n" + "="*60)
print("🎉 ALL VISUALIZATIONS CREATED!")
print("="*60)
print(f"\n📁 Location: visualizations/ folder")
print(f"📊 Total charts: 9")
print(f"\n✅ Charts created:")
for i in range(1, 10):
    print(f"   {i}. visualization_{i}.png")

print("\n💡 Use these images in your Power BI dashboard!")
print("="*60)