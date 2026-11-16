
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "random_forest_pipeline.joblib"

# Business decision threshold
CHURN_THRESHOLD = 0.30

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description=(
        "Independent academic JPMorganChase-inspired case study. "
        "Uses synthetic Kaggle data, not actual JPMorganChase data."
    ),
    version="1.0.0",
)

# Load the trained preprocessing + Random Forest pipeline
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=0)
    Geography: str
    Gender: str
    Age: float = Field(..., ge=0)
    Tenure: int = Field(..., ge=0)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=0)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0)


class PredictionOutput(BaseModel):
    churn_probability: float
    predicted_churn: int
    risk_level: str
    threshold: float
    recommended_action: str


@app.get("/")
def root():
    return {
        "message": "Bank Customer Churn Prediction API",
        "documentation": "/docs",
        "model": "Random Forest",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "model_name": "Random Forest",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train Random Forest first.",
        )

    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        probability = float(model.predict_proba(customer_df)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check model compatibility.",
        ) from exc

    predicted_churn = int(probability >= CHURN_THRESHOLD)

    if probability >= 0.70:
        risk_level = "High"
        action = "Prioritize for personalized retention outreach"
    elif probability >= CHURN_THRESHOLD:
        risk_level = "Medium"
        action = "Consider targeted retention communication"
    else:
        risk_level = "Low"
        action = "Continue standard customer engagement"

    return PredictionOutput(
        churn_probability=round(probability, 4),
        predicted_churn=predicted_churn,
        risk_level=risk_level,
        threshold=CHURN_THRESHOLD,
        recommended_action=action,
    )
