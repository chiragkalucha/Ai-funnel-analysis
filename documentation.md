# 🎯 AI-Powered Funnel Analysis - Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Models](#machine-learning-models)
5. [API Documentation](#api-documentation)
6. [Database Schema](#database-schema)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

### Business Problem
E-commerce platforms lose 76% of users before conversion. This project predicts which users will churn and provides actionable recommendations to prevent revenue loss.

### Solution
End-to-end ML system that:
- Analyzes user behavior in real-time
- Predicts churn probability with 95.36% accuracy
- Provides personalized intervention recommendations
- Delivers predictions via REST API in <50ms

### Impact Metrics
- **Accuracy**: 95.36%
- **Potential Revenue Saved**: $83,355 per 10,000 users
- **Response Time**: <50ms per prediction
- **False Negative Rate**: 8.27% (170 out of 2,055 converters missed)

---

## 2. System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL Database                                        │
│  ├─ users table (50,000 records)                           │
│  ├─ events table (303,754 records)                         │
│  └─ Indexed on user_id, event_type, timestamp              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering Pipeline                               │
│  ├─ Aggregation features (6)                               │
│  ├─ Behavioral features (3)                                │
│  ├─ Temporal features (2)                                  │
│  └─ Categorical encoding (12)                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                               │
├─────────────────────────────────────────────────────────────┤
│  Random Forest Classifier                                   │
│  ├─ 100 trees, max_depth=10                                │
│  ├─ Accuracy: 95.36%                                       │
│  └─ Saved as .pkl for inference                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   SERVING LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  FastAPI REST API                                           │
│  ├─ /predict - Single user prediction                      │
│  ├─ /predict/batch - Batch predictions                     │
│  ├─ /health - Health check                                 │
│  └─ /docs - Interactive documentation                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  Power BI Dashboard                                         │
│  ├─ Executive Summary                                      │
│  ├─ ML Performance Metrics                                │
│  └─ Business Insights                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Data Generation

**Script**: `scripts/generate_data.py`

**Purpose**: Creates realistic synthetic e-commerce data

**Output**:
- `data/users.csv` - User demographics and attributes
- `data/events.csv` - User behavior events

**Key Parameters**:
```python
NUM_USERS = 50000
NUM_EVENTS = 303754
CONVERSION_RATE = 0.2397 (23.97%)
```

**Funnel Stages**:
1. Homepage Visit (100%)
2. Product View (90%)
3. Add to Cart (30%)
4. Checkout Start (18%)
5. Payment Info (13%)
6. Purchase Complete (24%)

### 3.2 Database Setup

**Script**: `scripts/setup_postgresql.py`

**Schema Design**:

**Users Table**:
```sql
CREATE TABLE users (
    user_id VARCHAR(20) PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    country VARCHAR(50),
    device_type VARCHAR(20),
    acquisition_channel VARCHAR(50),
    age_group VARCHAR(20)
);
```

**Events Table**:
```sql
CREATE TABLE events (
    event_id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(20) REFERENCES users(user_id),
    session_id VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50),
    -- additional fields...
);
```

**Indexes Created**:
- `idx_events_user_id` - Fast user lookups
- `idx_events_type` - Filter by event type
- `idx_events_timestamp` - Time-based queries
- `idx_events_session` - Session analysis

### 3.3 Feature Engineering

**Script**: `scripts/feature_engineering.py`

**Features Created** (23 total):

**Aggregation Features** (6):
- `total_events` - Count of all user events
- `session_count` - Number of unique sessions
- `total_duration_seconds` - Time spent on site
- `total_clicks` - Total click count
- `avg_scroll_depth` - Average scroll percentage
- `activity_span_hours` - Time between first and last event

**Behavioral Features** (3):
- `avg_duration_per_session` - Engagement quality
- `avg_events_per_session` - Activity level
- `most_active_hour` - Peak usage time

**Temporal Features** (2):
- `activity_span_hours` - Browsing duration
- `most_active_hour` - Time of day preference

**Categorical Features** (12):
- Device: Desktop, Mobile, Tablet (one-hot encoded)
- Country: USA, India, UK (one-hot encoded)
- Channel: Direct, Email, Organic Search, Paid Ads, Social Media (one-hot encoded)

**Target Variable**:
- `has_converted` - Binary (0 = churned, 1 = converted)

---

## 4. Machine Learning Models

### 4.1 Model Selection

**Candidates Evaluated**:
1. Random Forest Classifier
2. XGBoost Classifier

**Winner**: Random Forest (marginal performance advantage)

### 4.2 Training Process

**Script**: `scripts/train_model.py`

**Data Split**:
- Training: 80% (40,000 users)
- Testing: 20% (10,000 users)
- Stratified split to maintain class balance

**Hyperparameters** (Random Forest):
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42
}
```

### 4.3 Model Performance

**Metrics**:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | 95.36% | Overall correctness |
| Precision | 86.51% | Of predicted converters, 86.51% actually converted |
| Recall | 91.73% | Of actual converters, caught 91.73% |
| F1-Score | 89.04% | Harmonic mean of precision/recall |
| ROC-AUC | 98.88% | Excellent class separation |

**Confusion Matrix**:
```
                 Predicted
                No      Yes
Actual  No    7,651    294   (FP: False alarms)
        Yes     170  1,885   (FN: Missed opportunities)
```

**Business Impact**:
- True Positives (1,885): Correctly identified converters → Save with targeted offers
- False Positives (294): Wasted discount offers → $1,470 cost
- False Negatives (170): Missed converters → $8,500 lost revenue
- True Negatives (7,651): Correctly identified non-converters → No action needed

**Net Gain**: $83,355 from ML-driven interventions

### 4.4 Feature Importance

**Top 10 Features**:
1. `avg_events_per_session` (30.06%) - Engagement quality
2. `total_events` (20.45%) - Overall activity
3. `avg_duration_per_session` (16.82%) - Time investment
4. `total_duration_seconds` (13.10%) - Total time spent
5. `total_clicks` (8.17%) - Interaction level
6. `activity_span_hours` (2.91%) - Browsing duration
7. `session_count` (2.91%) - Visit frequency
8. `device_Desktop` (1.94%) - Device preference
9. `device_Mobile` (1.79%) - Mobile usage
10. `avg_scroll_depth` (0.74%) - Content engagement

**Key Insight**: Behavioral features (avg_events_per_session, avg_duration_per_session) are more predictive than demographics.

---

## 5. API Documentation

### 5.1 Endpoints

#### GET /
**Description**: Root endpoint, API information

**Response**:
```json
{
  "message": "Churn Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "predict_batch": "/predict/batch",
    "docs": "/docs"
  }
}
```

#### GET /health
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 23,
  "timestamp": "2025-02-07T14:30:22.123456"
}
```

#### POST /predict
**Description**: Predict churn for single user

**Request Body**:
```json
{
  "total_events": 5,
  "session_count": 2,
  "total_duration_seconds": 300,
  "total_clicks": 15,
  "avg_scroll_depth": 65.5,
  "avg_duration_per_session": 150,
  "avg_events_per_session": 2.5,
  "activity_span_hours": 24,
  "most_active_hour": 14,
  "device_Desktop": 0,
  "device_Mobile": 1,
  "device_Tablet": 0,
  "country_USA": 1,
  "country_India": 0,
  "country_UK": 0,
  "channel_Direct": 0,
  "channel_Email": 0,
  "channel_Organic_Search": 1,
  "channel_Paid_Ads": 0,
  "channel_Social_Media": 0,
  "reached_homepage_visit": 1,
  "reached_product_view": 1
}
```

**Response**:
```json
{
  "user_id": "USER_20250207143022",
  "churn_probability": 0.7612,
  "conversion_probability": 0.2388,
  "risk_category": "High Risk",
  "recommended_action": "URGENT: Show 10% discount popup immediately!",
  "confidence": "High",
  "timestamp": "2025-02-07T14:30:22.123456"
}
```

**Risk Categories**:
- **Low Risk** (churn < 30%): No intervention needed
- **Medium Risk** (30% ≤ churn < 70%): Send reminder email
- **High Risk** (churn ≥ 70%): Show discount popup immediately

### 5.2 API Usage Examples

**Python**:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "total_events": 5,
    "session_count": 2,
    # ... other features
}

response = requests.post(url, json=data)
result = response.json()

print(f"Churn Risk: {result['churn_probability']*100:.1f}%")
print(f"Action: {result['recommended_action']}")
```

**cURL**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"total_events": 5, "session_count": 2, ...}'
```

**JavaScript**:
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    total_events: 5,
    session_count: 2,
    // ... other features
  })
})
.then(res => res.json())
.then(data => {
  console.log(`Churn Risk: ${data.churn_probability * 100}%`);
  console.log(`Action: ${data.recommended_action}`);
});
```

---

## 6. Database Schema

### 6.1 Entity Relationship Diagram
```
┌─────────────────────────────┐
│         USERS               │
├─────────────────────────────┤
│ PK  user_id                 │
│     signup_date             │
│     country                 │
│     device_type             │
│     acquisition_channel     │
│     age_group               │
└─────────────────────────────┘
            │
            │ 1:N
            │
            ▼
┌─────────────────────────────┐
│         EVENTS              │
├─────────────────────────────┤
│ PK  event_id                │
│ FK  user_id                 │
│     session_id              │
│     timestamp               │
│     event_type              │
│     device_type             │
│     page_url                │
│     duration_seconds        │
│     country                 │
│     hour_of_day             │
│     day_of_week             │
│     is_weekend              │
│     clicks_count            │
│     scroll_depth_percent    │
└─────────────────────────────┘
```

### 6.2 Sample Queries

**Get user conversion status**:
```sql
SELECT 
    u.user_id,
    u.device_type,
    u.acquisition_channel,
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM events e 
            WHERE e.user_id = u.user_id 
            AND e.event_type = 'purchase_complete'
        ) THEN 1 
        ELSE 0 
    END as has_converted
FROM users u;
```

**Funnel drop-off analysis**:
```sql
SELECT 
    event_type,
    COUNT(DISTINCT user_id) as users,
    COUNT(DISTINCT user_id) * 100.0 / 
        (SELECT COUNT(DISTINCT user_id) FROM events WHERE event_type = 'homepage_visit') 
        as percentage
FROM events
WHERE event_type IN (
    'homepage_visit', 'product_view', 'add_to_cart', 
    'checkout_start', 'payment_info', 'purchase_complete'
)
GROUP BY event_type
ORDER BY percentage DESC;
```

**Conversion rate by device**:
```sql
SELECT 
    device_type,
    COUNT(DISTINCT user_id) as total_users,
    SUM(CASE WHEN has_converted = 1 THEN 1 ELSE 0 END) as converters,
    AVG(has_converted) * 100 as conversion_rate
FROM features_ml
GROUP BY device_type
ORDER BY conversion_rate DESC;
```

---

## 7. Deployment Guide

### 7.1 Local Development Setup

**Prerequisites**:
- Python 3.10+
- PostgreSQL 16+
- 8GB RAM minimum

**Installation Steps**:
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai_funnel_analysis.git
cd ai_funnel_analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up PostgreSQL database
# Update password in scripts/setup_postgresql.py
python scripts/setup_postgresql.py

# 5. Generate data
python scripts/generate_data.py

# 6. Engineer features
python scripts/feature_engineering.py

# 7. Train model
python scripts/train_model.py

# 8. Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 9. Access API documentation
# Open browser: http://localhost:8000/docs
```

### 7.2 Docker Deployment

**Build and run**:
```bash
docker-compose up -d
```

**Check status**:
```bash
docker-compose ps
```

**View logs**:
```bash
docker-compose logs -f api
```

**Stop services**:
```bash
docker-compose down
```

### 7.3 Production Deployment

**Recommended Platforms**:
1. **AWS ECS** - Container orchestration
2. **Google Cloud Run** - Serverless containers
3. **Azure Container Instances** - Simple container deployment
4. **Heroku** - PaaS deployment

**Production Checklist**:
- [ ] Environment variables in `.env` file
- [ ] Database connection pooling
- [ ] API rate limiting
- [ ] HTTPS/SSL certificates
- [ ] Logging and monitoring
- [ ] Health check endpoints
- [ ] Backup and recovery plan
- [ ] CI/CD pipeline

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: "Password authentication failed"
**Cause**: Incorrect PostgreSQL password

**Solution**:
```python
# Update password in scripts
PASSWORD = 'your_actual_password'
```

#### Issue: "Model file not found"
**Cause**: Model not trained yet

**Solution**:
```bash
python scripts/train_model.py
```

#### Issue: "Port 8000 already in use"
**Cause**: Another process using port

**Solution**:
```bash
# Find process
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <process_id> /F

# Or change port
uvicorn api.main:app --port 8001
```

#### Issue: "Feature count mismatch"
**Cause**: Model trained with different features

**Solution**:
```bash
# Retrain model with current features
python scripts/train_model.py
```

### 8.2 Performance Optimization

**Database**:
- Add indexes on frequently queried columns
- Use connection pooling
- Implement query caching

**API**:
- Enable Gzip compression
- Implement response caching
- Use async endpoints for I/O operations

**Model**:
- Consider model quantization
- Batch predictions where possible
- Cache feature transformations

---

## 9. Future Enhancements

### Short-term (1-2 weeks)
- [ ] A/B testing framework
- [ ] Real-time model monitoring
- [ ] Automated model retraining
- [ ] Additional visualization dashboards

### Medium-term (1-2 months)
- [ ] Multi-model ensemble
- [ ] Deep learning models (LSTM for sequence prediction)
- [ ] Feature store implementation
- [ ] Kubernetes deployment

### Long-term (3-6 months)
- [ ] Real-time streaming predictions
- [ ] AutoML for hyperparameter tuning
- [ ] Explainable AI (SHAP values)
- [ ] Multi-region deployment

---

## 10. References

**Technologies Used**:
- FastAPI: https://fastapi.tiangolo.com/
- scikit-learn: https://scikit-learn.org/
- PostgreSQL: https://www.postgresql.org/
- Docker: https://www.docker.com/

**Related Research**:
- Gradient Boosting: Friedman, J. H. (2001)
- Random Forests: Breiman, L. (2001)
- Churn Prediction: Verbeke et al. (2012)

---

**Document Version**: 1.0.0  
**Last Updated**: February 7, 2025  
**Author**: [Your Name]  
**Contact**: [Your Email]