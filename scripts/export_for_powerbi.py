"""
Export data for Power BI dashboard
"""

import pandas as pd
from sqlalchemy import create_engine

PASSWORD = 'yonex.Chirag1'
engine = create_engine(f'postgresql://postgres:{PASSWORD}@localhost:5432/funnel_analysis')

print("📤 Exporting data for Power BI...")

# 1. Users with features
users_df = pd.read_csv('data/features_engineered.csv')
users_df.to_csv('data/powerbi_users.csv', index=False)
print("✅ Exported: powerbi_users.csv")

# 2. Events data
events_df = pd.read_sql("SELECT * FROM events", engine)
events_df.to_csv('data/powerbi_events.csv', index=False)
print("✅ Exported: powerbi_events.csv")

# 3. Funnel metrics
funnel_df = pd.read_sql("SELECT * FROM funnel_metrics", engine)
funnel_df.to_csv('data/powerbi_funnel.csv', index=False)
print("✅ Exported: powerbi_funnel.csv")

# 4. Model predictions
model_predictions = pd.read_csv('data/features_engineered.csv')[['user_id', 'has_converted']]
import joblib
model = joblib.load('models/churn_prediction_random_forest.pkl')
features = joblib.load('models/feature_names.pkl')

X = users_df[features]
predictions = model.predict_proba(X)[:, 1]

model_predictions['churn_probability'] = 1 - predictions
model_predictions['risk_category'] = pd.cut(
    model_predictions['churn_probability'],
    bins=[0, 0.3, 0.7, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

model_predictions.to_csv('data/powerbi_predictions.csv', index=False)
print("✅ Exported: powerbi_predictions.csv")

print("\n✅ All data exported! Ready for Power BI")