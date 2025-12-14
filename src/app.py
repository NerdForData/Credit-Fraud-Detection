# src/api/main.py
"""
FastAPI inference service for credit-risk-ml-system.

Loads:
- models/pipeline.joblib
- models/threshold.json

Exposes:
POST /predict

Run locally:
    uvicorn src.api.main:app --reload

Then open:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

# -----------------------
# Load model + threshold
# -----------------------

PIPELINE_PATH = "models/pipeline.joblib"
THRESHOLD_PATH = "models/threshold.json"

pipeline = joblib.load(PIPELINE_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(json.load(f)["threshold"])


# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict loan default probability using a trained ML pipeline.",
    version="1.0"
)


# -----------------------
# Request schema
# -----------------------

class Applicant(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# -----------------------
# Prediction endpoint
# -----------------------

@app.post("/predict")
def predict_default(applicant: Applicant):

    # Convert the applicant object into a DataFrame row
    import pandas as pd
    data = pd.DataFrame([applicant.dict()])

    # Get model probability of default (positive class)
    proba = pipeline.predict_proba(data)[0][1]

    # Use the saved threshold to convert into binary prediction
    prediction = int(proba >= THRESHOLD)

    return {
        "default_probability": float(proba),
        "prediction": prediction,
        "threshold_used": THRESHOLD
    }

