"""
Churn Prediction API
FastAPI server for real-time predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================
# INITIALIZE API
# ============================================

app = FastAPI(
    title="Churn Prediction API",
    description="Predict user churn probability for funnel analysis",
    version="1.0.0"
)

# Enable CORS (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD MODEL AT STARTUP
# ============================================

print("🚀 Loading ML model...")

# Get the model path (go up one directory from api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_prediction_random_forest.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"✅ Model loaded: {len(feature_names)} features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = None

# ============================================
# DATA MODELS (Request/Response Schemas)
# ============================================

class UserFeatures(BaseModel):
    """
    Input schema for prediction
    These are the features the model needs
    """
    total_events: int = Field(..., ge=0, description="Total number of events")
    session_count: int = Field(..., ge=1, description="Number of sessions")
    total_duration_seconds: float = Field(..., ge=0, description="Total time spent in seconds")
    total_clicks: int = Field(..., ge=0, description="Total clicks")
    avg_scroll_depth: float = Field(..., ge=0, le=100, description="Average scroll depth percentage")
    avg_duration_per_session: float = Field(..., ge=0, description="Average duration per session")
    avg_events_per_session: float = Field(..., ge=0, description="Average events per session")
    activity_span_hours: float = Field(..., ge=0, description="Hours between first and last activity")
    most_active_hour: int = Field(..., ge=0, le=23, description="Most active hour (0-23)")
    
    # Device (one-hot encoded)
    device_Desktop: int = Field(0, ge=0, le=1, description="1 if Desktop, else 0")
    device_Mobile: int = Field(0, ge=0, le=1, description="1 if Mobile, else 0")
    device_Tablet: int = Field(0, ge=0, le=1, description="1 if Tablet, else 0")
    
    # Country (one-hot encoded)
    country_USA: int = Field(0, ge=0, le=1, description="1 if USA, else 0")
    country_India: int = Field(0, ge=0, le=1, description="1 if India, else 0")
    country_UK: int = Field(0, ge=0, le=1, description="1 if UK, else 0")
    
    # Channel (one-hot encoded)
    channel_Direct: int = Field(0, ge=0, le=1, description="1 if Direct, else 0")
    channel_Email: int = Field(0, ge=0, le=1, description="1 if Email, else 0")
    channel_Organic_Search: int = Field(0, ge=0, le=1, description="1 if Organic Search, else 0")
    channel_Paid_Ads: int = Field(0, ge=0, le=1, description="1 if Paid Ads, else 0")
    channel_Social_Media: int = Field(0, ge=0, le=1, description="1 if Social Media, else 0")
    
    # Early funnel stages (safe to use)
    reached_homepage_visit: int = Field(1, ge=0, le=1, description="1 if reached homepage, else 0")
    reached_product_view: int = Field(0, ge=0, le=1, description="1 if viewed products, else 0")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Output schema for prediction
    """
    user_id: str = None
    churn_probability: float
    conversion_probability: float
    risk_category: str
    recommended_action: str
    confidence: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """
    For predicting multiple users at once
    """
    users: List[UserFeatures]


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    model_loaded: bool
    features_count: int
    timestamp: str

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        features_count=len(feature_names) if feature_names else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(features: UserFeatures):
    """
    Predict churn probability for a single user
    
    Returns:
    - churn_probability: Probability user will NOT convert (0-1)
    - conversion_probability: Probability user WILL convert (0-1)
    - risk_category: Low/Medium/High risk
    - recommended_action: What to do
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = features.dict()
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in feature_names:
            # Handle feature name mapping (underscores vs spaces)
            mapped_name = feature_name.replace(' ', '_')
            if mapped_name in input_dict:
                feature_vector.append(input_dict[mapped_name])
            elif feature_name in input_dict:
                feature_vector.append(input_dict[feature_name])
            else:
                # Default to 0 if feature not provided
                feature_vector.append(0)
        
        # Reshape for single prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Get prediction probability
        prediction_proba = model.predict_proba(X)[0]
        conversion_prob = float(prediction_proba[1])  # Probability of class 1 (converted)
        churn_prob = float(1 - conversion_prob)  # Probability of churn
        
        # Determine risk category
        if churn_prob < 0.3:
            risk_category = "Low Risk"
            recommended_action = "No intervention needed - user likely to convert naturally"
            confidence = "High"
        elif churn_prob < 0.7:
            risk_category = "Medium Risk"
            recommended_action = "Send reminder email or show social proof"
            confidence = "Medium"
        else:
            risk_category = "High Risk"
            recommended_action = "URGENT: Show 10% discount popup immediately!"
            confidence = "High"
        
        # Generate user_id if not provided
        user_id = f"USER_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return PredictionResponse(
            user_id=user_id,
            churn_probability=round(churn_prob, 4),
            conversion_probability=round(conversion_prob, 4),
            risk_category=risk_category,
            recommended_action=recommended_action,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple users at once
    Useful for batch processing
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for user_features in request.users:
            # Reuse single prediction logic
            prediction = await predict_churn(user_features)
            predictions.append(prediction.dict())
        
        return {
            "count": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "features": feature_names,
        "model_params": model.get_params() if hasattr(model, 'get_params') else {}
    }

# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Run when API starts
    """
    print("="*60)
    print("🚀 Churn Prediction API Started!")
    print("="*60)
    print(f"📊 Model loaded: {model is not None}")
    print(f"🎯 Features: {len(feature_names) if feature_names else 0}")
    print(f"📍 Docs available at: http://localhost:8000/docs")
    print("="*60)

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)