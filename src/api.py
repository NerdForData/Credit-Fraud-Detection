# src/api.py
"""
Production-Ready FastAPI Application for Credit Risk Prediction System.

This REST API serves the trained credit risk model through multiple endpoints,
providing both single and batch prediction capabilities along with health
monitoring and model information endpoints. The API also serves an interactive
web interface for user-friendly predictions.

Architecture:
    - FastAPI framework for high-performance async request handling
    - Pydantic models for request/response validation and documentation
    - Static file serving for interactive web UI
    - Global model loading at startup for efficient inference
    - Comprehensive error handling and logging

Core Endpoints:
    1. GET  /                  - Serve interactive web interface
    2. POST /predict           - Single loan application prediction
    3. POST /predict/batch     - Batch predictions (up to 1000 applications)
    4. GET  /health            - API health check and model status
    5. GET  /model/info        - Model metadata and configuration

API Features:
    - Request validation with Pydantic (type checking, range validation)
    - Automatic OpenAPI documentation at /docs (Swagger UI)
    - ReDoc alternative documentation at /redoc
    - Structured logging for monitoring and debugging
    - Risk categorization (Low/Medium/High) based on probability
    - Confidence scoring for prediction reliability
    - Binary flag mapping (Yes/No → 1/0) for categorical features
    - Consistent timestamp tracking for audit trails

Production Deployment:
    # Development mode (auto-reload on code changes)
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
    
    # Production mode (Docker container)
    docker-compose up -d
    
    # Direct production mode
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

Accessing the API:
    - Web UI: http://localhost:8000/
    - API Docs: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - Health Check: http://localhost:8000/health

Example API Usage:
    # Python client example
    >>> import requests
    >>> payload = {
    ...     "Age": 35, "Income": 75000, "LoanAmount": 25000,
    ...     "CreditScore": 720, "MonthsEmployed": 48,
    ...     "NumCreditLines": 5, "InterestRate": 5.5,
    ...     "LoanTerm": 60, "DTIRatio": 0.35,
    ...     "Education": "Bachelor's", "EmploymentType": "Full-time",
    ...     "MaritalStatus": "Married", "HasMortgage": "Yes",
    ...     "HasDependents": "Yes", "LoanPurpose": "Home",
    ...     "HasCoSigner": "No"
    ... }
    >>> response = requests.post("http://localhost:8000/predict", json=payload)
    >>> print(response.json())
    {
        "default_probability": 0.2543,
        "prediction": "No Default",
        "risk_level": "Low",
        "threshold": 0.6028,
        "confidence": 0.4914,
        "timestamp": "2024-01-15T10:30:45.123456"
    }

Prerequisites:
    - Trained model: models/pipeline.joblib
    - Optimized threshold: models/threshold.json
    - Static files: static/ directory with web UI

Dependencies:
    - fastapi: Web framework
    - uvicorn: ASGI server
    - pydantic: Data validation
    - joblib: Model loading
    - pandas: Data manipulation
    - numpy: Numerical operations

Notes:
    - Model and threshold loaded once at startup for performance
    - Binary columns (Yes/No) automatically converted to 1/0
    - Feature order must match training data
    - Maximum batch size: 1000 applications
    - All predictions include timestamp for audit tracking
"""

# Standard library imports
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Third-party imports
import joblib
import pandas as pd
import numpy as np

# FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Pydantic for data validation
from pydantic import BaseModel, Field, validator

# Local imports
from src.data_prep import map_binary_flags, get_feature_lists

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Configure structured logging for production monitoring and debugging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# ============================================================================
# These models provide automatic validation, type checking, and API documentation.
# FastAPI uses these for:
# - Request payload validation with descriptive error messages
# - OpenAPI schema generation (Swagger/ReDoc docs)
# - Response serialization with type safety

class LoanApplication(BaseModel):
    """
    Schema for a single loan application input.
    
    This model defines all required features for making a prediction,
    with validation rules to ensure data quality and realistic values.
    
    Validation Rules:
        - Age: 18-100 years (legal age to maximum reasonable age)
        - Income: Non-negative (annual income in dollars)
        - LoanAmount: Non-negative (requested loan in dollars)
        - CreditScore: 300-850 (FICO score range)
        - MonthsEmployed: Non-negative (employment stability indicator)
        - NumCreditLines: Non-negative (credit history depth)
        - InterestRate: 0-100% (annual percentage rate)
        - LoanTerm: Positive (loan duration in months)
        - DTIRatio: Non-negative (debt-to-income ratio)
        - Categorical fields: Free-form strings (validated by model)
    
    Attributes:
        Age: Applicant's age in years
        Income: Annual income in dollars
        LoanAmount: Requested loan amount in dollars
        CreditScore: FICO credit score (300-850)
        MonthsEmployed: Duration at current employment in months
        NumCreditLines: Number of active credit lines
        InterestRate: Loan interest rate as percentage
        LoanTerm: Loan repayment term in months
        DTIRatio: Debt-to-Income ratio (total debt / gross income)
        Education: Education level (e.g., "Bachelor's", "High School")
        EmploymentType: Employment category (e.g., "Full-time", "Part-time")
        MaritalStatus: Marital status (e.g., "Single", "Married")
        HasMortgage: Mortgage flag ("Yes" or "No")
        HasDependents: Dependents flag ("Yes" or "No")
        LoanPurpose: Purpose of loan (e.g., "Home", "Auto", "Education")
        HasCoSigner: Co-signer flag ("Yes" or "No")
    
    Example:
        >>> app = LoanApplication(
        ...     Age=35, Income=75000, LoanAmount=25000, CreditScore=720,
        ...     MonthsEmployed=48, NumCreditLines=5, InterestRate=5.5,
        ...     LoanTerm=60, DTIRatio=0.35, Education="Bachelor's",
        ...     EmploymentType="Full-time", MaritalStatus="Married",
        ...     HasMortgage="Yes", HasDependents="Yes",
        ...     LoanPurpose="Home", HasCoSigner="No"
        ... )
    """
    # Numeric features with range validation
    Age: int = Field(..., ge=18, le=100, description="Applicant's age (18-100)")
    Income: float = Field(..., ge=0, description="Annual income in dollars")
    LoanAmount: float = Field(..., ge=0, description="Requested loan amount in dollars")
    CreditScore: int = Field(..., ge=300, le=850, description="FICO credit score (300-850)")
    MonthsEmployed: int = Field(..., ge=0, description="Months at current employment")
    NumCreditLines: int = Field(..., ge=0, description="Number of credit lines")
    InterestRate: float = Field(..., ge=0, le=100, description="Interest rate percentage")
    LoanTerm: int = Field(..., ge=1, description="Loan term in months")
    DTIRatio: float = Field(..., ge=0, description="Debt-to-Income ratio")
    
    # Categorical features (validated by model preprocessing)
    Education: str = Field(..., description="Education level (e.g., Bachelor's, High School, Master's)")
    EmploymentType: str = Field(..., description="Employment type (e.g., Full-time, Part-time, Self-employed)")
    MaritalStatus: str = Field(..., description="Marital status (e.g., Single, Married, Divorced)")
    
    # Binary features (Yes/No converted to 1/0 during preprocessing)
    HasMortgage: str = Field(..., description="Has mortgage (Yes/No)")
    HasDependents: str = Field(..., description="Has dependents (Yes/No)")
    HasCoSigner: str = Field(..., description="Has co-signer (Yes/No)")
    
    # Purpose feature
    LoanPurpose: str = Field(..., description="Loan purpose (e.g., Home, Auto, Education, Business)")

    class Config:
        """Pydantic configuration with example data for API documentation."""
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Income": 75000.0,
                "LoanAmount": 25000.0,
                "CreditScore": 720,
                "MonthsEmployed": 48,
                "NumCreditLines": 5,
                "InterestRate": 5.5,
                "LoanTerm": 60,
                "DTIRatio": 0.35,
                "Education": "Bachelor's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Married",
                "HasMortgage": "Yes",
                "HasDependents": "Yes",
                "LoanPurpose": "Home",
                "HasCoSigner": "No"
            }
        }


class BatchLoanApplications(BaseModel):
    """
    Schema for batch prediction requests.
    
    Allows processing multiple loan applications in a single API call
    for efficiency. Maximum batch size is limited to prevent resource
    exhaustion and ensure reasonable response times.
    
    Attributes:
        applications: List of LoanApplication objects (1-1000 items)
    
    Validation:
        - Minimum 1 application required
        - Maximum 1000 applications per batch
        - Each application validated individually
    
    Example:
        >>> batch = BatchLoanApplications(
        ...     applications=[app1, app2, app3]
        ... )
    """
    applications: List[LoanApplication] = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="List of loan applications to process (max 1000)"
    )


class PredictionResponse(BaseModel):
    """
    Schema for single prediction response.
    
    Provides comprehensive prediction information including probability,
    binary classification, risk categorization, and metadata for audit trails.
    
    Attributes:
        default_probability: Probability of loan default (0.0 to 1.0)
            Values closer to 1.0 indicate higher default risk
        prediction: Human-readable binary classification
            "Default" or "No Default"
        risk_level: Categorical risk assessment
            "Low" (prob < 0.3), "Medium" (0.3-0.6), "High" (> 0.6)
        threshold: Classification threshold used for decision
            From models/threshold.json, optimized during training
        confidence: Prediction confidence score (0.0 to 1.0)
            Distance from decision boundary; higher is more confident
        timestamp: ISO format timestamp of when prediction was made
            For audit tracking and temporal analysis
    
    Example:
        >>> response = PredictionResponse(
        ...     default_probability=0.2543,
        ...     prediction="No Default",
        ...     risk_level="Low",
        ...     threshold=0.6028,
        ...     confidence=0.6914,
        ...     timestamp="2024-01-15T10:30:45.123456"
        ... )
    """
    default_probability: float = Field(..., description="Probability of default (0-1)")
    prediction: str = Field(..., description="Binary prediction: 'Default' or 'No Default'")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High")
    threshold: float = Field(..., description="Classification threshold used")
    confidence: float = Field(..., description="Confidence score (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction response.
    
    Returns predictions for all applications in the batch along with
    metadata about the batch processing.
    
    Attributes:
        predictions: List of individual PredictionResponse objects
            One per input application, in same order
        total_processed: Total number of applications processed
            Should match len(predictions)
        timestamp: Batch processing timestamp
            When the batch request was completed
    
    Example:
        >>> response = BatchPredictionResponse(
        ...     predictions=[pred1, pred2, pred3],
        ...     total_processed=3,
        ...     timestamp="2024-01-15T10:30:45.123456"
        ... )
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Number of applications processed")
    timestamp: str = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """
    Schema for health check response.
    
    Provides API status and model loading information for monitoring
    and readiness probes in production environments.
    
    Attributes:
        status: Overall API status ("healthy" or "unhealthy")
        model_loaded: Whether pipeline.joblib loaded successfully
        threshold_loaded: Whether threshold.json loaded successfully
        model_path: Path to pipeline file
        threshold_path: Path to threshold file
        timestamp: Health check timestamp
    
    Example:
        >>> response = HealthResponse(
        ...     status="healthy",
        ...     model_loaded=True,
        ...     threshold_loaded=True,
        ...     model_path="models/pipeline.joblib",
        ...     threshold_path="models/threshold.json",
        ...     timestamp="2024-01-15T10:30:45.123456"
        ... )
    """
    status: str = Field(..., description="API status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Model loaded successfully")
    threshold_loaded: bool = Field(..., description="Threshold loaded successfully")
    model_path: str = Field(..., description="Path to model file")
    threshold_path: str = Field(..., description="Path to threshold file")
    timestamp: str = Field(..., description="Health check timestamp")


class ModelInfoResponse(BaseModel):
    """
    Schema for model information response.
    
    Provides metadata about the loaded model for transparency and debugging.
    
    Attributes:
        model_type: Type of model (e.g., "XGBoost Pipeline")
        features: List of expected feature names in order
        threshold: Current classification threshold
        model_file_size_mb: Size of pipeline.joblib in MB
        last_modified: When model file was last modified
    
    Example:
        >>> response = ModelInfoResponse(
        ...     model_type="XGBoost Pipeline",
        ...     features=["Age", "Income", ...],
        ...     threshold=0.6028,
        ...     model_file_size_mb=2.45,
        ...     last_modified="2024-01-15T08:00:00"
        ... )
    """
    model_type: str = Field(..., description="Type of ML model")
    features: List[str] = Field(..., description="Expected feature names")
    threshold: float = Field(..., description="Classification threshold")
    model_file_size_mb: float = Field(..., description="Model file size in MB")
    last_modified: str = Field(..., description="Model last modified timestamp")


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================
# Create FastAPI instance with metadata for automatic API documentation

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Machine learning API for predicting loan default probability using XGBoost",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI documentation
    redoc_url="/redoc"     # ReDoc alternative documentation
)


# ============================================================================
# STATIC FILES CONFIGURATION
# ============================================================================
# Serve interactive web UI from static/ directory
# Accessible at http://localhost:8000/static/

app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
# These are loaded once at startup and reused for all predictions
# This avoids loading the model on every request, improving performance

MODEL = None            # Trained pipeline (preprocessor + XGBoost)
THRESHOLD = None        # Optimized classification threshold
FEATURE_COLUMNS = None  # Expected feature order
MODEL_PATH = "models/pipeline.joblib"
THRESHOLD_PATH = "models/threshold.json"


# ============================================================================
# STARTUP AND SHUTDOWN EVENT HANDLERS
# ============================================================================
# These functions run when the API starts up and shuts down

@app.on_event("startup")
async def load_model():
    """
    Load model and threshold on application startup.
    
    This function runs once when the FastAPI application starts,
    loading the trained model and threshold into memory for fast
    inference. If loading fails, the application will not start.
    
    Loading Strategy:
        - Model loaded once at startup (not per request)
        - Reduces latency: ~1ms per prediction vs ~100ms if loading each time
        - Memory trade-off: Model stays in RAM but predictions are fast
    
    Side Effects:
        - Sets global variables: MODEL, THRESHOLD, FEATURE_COLUMNS
        - Logs loading progress and errors
        - Raises exception if model or threshold not found
    
    Global Variables Modified:
        MODEL: Loaded pipeline (ColumnTransformer + XGBClassifier)
        THRESHOLD: Optimized classification threshold (float)
        FEATURE_COLUMNS: List of expected feature names in order
    
    Raises:
        FileNotFoundError: If model or threshold file not found
        Exception: If loading fails for any other reason
    
    Notes:
        - Run src/train.py first to generate required files
        - Model and threshold must exist before starting API
        - Check logs for detailed loading information
    """
    global MODEL, THRESHOLD, FEATURE_COLUMNS
    
    try:
        logger.info("="*60)
        logger.info("Starting API initialization...")
        logger.info("="*60)
        
        # ====================================================================
        # STEP 1: LOAD PIPELINE
        # ====================================================================
        # Pipeline contains: preprocessor (scaling, encoding) + XGBoost model
        
        logger.info(f"Loading pipeline from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found: {MODEL_PATH}"
            logger.error(error_msg)
            logger.error("Please run 'python -m src.train' first to generate the model")
            raise FileNotFoundError(error_msg)
        
        # Load pipeline using joblib
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
        
        # ====================================================================
        # STEP 2: LOAD THRESHOLD
        # ====================================================================
        # Threshold determines decision boundary (default vs no default)
        
        logger.info(f"Loading threshold from: {THRESHOLD_PATH}")
        
        if not os.path.exists(THRESHOLD_PATH):
            error_msg = f"Threshold file not found: {THRESHOLD_PATH}"
            logger.error(error_msg)
            logger.error("Please run 'python -m src.train' first to generate the threshold")
            raise FileNotFoundError(error_msg)
        
        # Load threshold JSON
        with open(THRESHOLD_PATH, "r") as f:
            threshold_data = json.load(f)
            THRESHOLD = threshold_data.get("threshold", 0.5)
        
        logger.info(f"✓ Threshold loaded: {THRESHOLD:.4f}")
        
        # ====================================================================
        # STEP 3: DEFINE FEATURE COLUMNS
        # ====================================================================
        # These must match the order expected by the trained model
        # Same order as in src/data_prep.py FEATURE_COLUMNS list
        
        FEATURE_COLUMNS = [
            # Numeric features
            "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
            # Categorical features
            "Education", "EmploymentType", "MaritalStatus",
            # Binary features (will be converted from Yes/No to 1/0)
            "HasMortgage", "HasDependents", "HasCoSigner",
            # Purpose feature
            "LoanPurpose"
        ]
        
        logger.info(f"✓ Feature columns defined: {len(FEATURE_COLUMNS)} features")
        logger.info("="*60)
        logger.info("API READY!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Error during startup: {str(e)}")
        logger.error("API startup failed - application will not start")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup operations on application shutdown.
    
    This function runs when the FastAPI application is shutting down,
    allowing for graceful cleanup of resources.
    
    Side Effects:
        - Logs shutdown event
        - Can be extended to close database connections, etc.
    
    Notes:
        - In current implementation, model stays in memory until process ends
        - No explicit cleanup needed for joblib-loaded models
    """
    logger.info("="*60)
    logger.info("Shutting down API...")
    logger.info("="*60)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
# These utility functions support prediction logic and response formatting

def get_risk_level(probability: float) -> str:
    """
    Categorize default risk based on predicted probability.
    
    This function converts continuous probability into categorical
    risk level for easier interpretation and decision-making.
    
    Risk Categories:
        - Low Risk: probability < 0.3 (< 30% chance of default)
        - Medium Risk: 0.3 <= probability < 0.6 (30-60% chance)
        - High Risk: probability >= 0.6 (>= 60% chance)
    
    Args:
        probability (float): Predicted default probability (0.0 to 1.0)
    
    Returns:
        str: Risk category ("Low", "Medium", or "High")
    
    Example:
        >>> get_risk_level(0.25)
        'Low'
        >>> get_risk_level(0.45)
        'Medium'
        >>> get_risk_level(0.75)
        'High'
    
    Notes:
        - Thresholds (0.3, 0.6) can be adjusted based on business requirements
        - Consider making thresholds configurable for different use cases
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def calculate_confidence(probability: float) -> float:
    """
    Calculate prediction confidence score.
    
    Confidence is based on distance from decision boundary (0.5).
    Predictions closer to 0.0 or 1.0 are more confident than those
    near 0.5 (the boundary between classes).
    
    Confidence Calculation:
        confidence = |probability - 0.5| * 2
        
    This scales distance from 0.5 to range [0, 1]:
        - probability = 0.0 or 1.0 → confidence = 1.0 (very confident)
        - probability = 0.5 → confidence = 0.0 (uncertain)
        - probability = 0.25 or 0.75 → confidence = 0.5 (moderate)
    
    Args:
        probability (float): Predicted default probability (0.0 to 1.0)
    
    Returns:
        float: Confidence score (0.0 to 1.0)
            Higher values indicate more confident predictions
    
    Example:
        >>> calculate_confidence(0.9)  # Very confident
        0.8
        >>> calculate_confidence(0.5)  # Uncertain
        0.0
        >>> calculate_confidence(0.1)  # Very confident
        0.8
    
    Notes:
        - Independent of actual classification threshold
        - Symmetric: same confidence for 0.1 and 0.9
        - Can be used to flag uncertain predictions for manual review
    """
    return abs(probability - 0.5) * 2


def predict_single(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate prediction for a single loan application.
    
    This function orchestrates the complete prediction workflow:
    1. Convert input dictionary to DataFrame
    2. Map binary flags (Yes/No → 1/0)
    3. Ensure correct feature order
    4. Get model prediction probability
    5. Apply threshold for binary classification
    6. Calculate risk level and confidence
    7. Package results with metadata
    
    Preprocessing Steps:
        - Binary columns (HasMortgage, HasDependents, HasCoSigner) converted to 1/0
        - Features reordered to match training data
        - Pipeline handles scaling and encoding automatically
    
    Args:
        data (Dict[str, Any]): Loan application data as dictionary
            Must contain all required features from LoanApplication schema
    
    Returns:
        Dict[str, Any]: Prediction result with keys:
            - default_probability: Probability of default (0-1)
            - prediction: "Default" or "No Default"
            - risk_level: "Low", "Medium", or "High"
            - threshold: Classification threshold used
            - confidence: Confidence score (0-1)
            - timestamp: ISO format timestamp
    
    Raises:
        Exception: If prediction fails (model error, missing features, etc.)
    
    Example:
        >>> data = {
        ...     "Age": 35, "Income": 75000, "LoanAmount": 25000,
        ...     "CreditScore": 720, "MonthsEmployed": 48,
        ...     "NumCreditLines": 5, "InterestRate": 5.5,
        ...     "LoanTerm": 60, "DTIRatio": 0.35,
        ...     "Education": "Bachelor's", "EmploymentType": "Full-time",
        ...     "MaritalStatus": "Married", "HasMortgage": "Yes",
        ...     "HasDependents": "Yes", "LoanPurpose": "Home",
        ...     "HasCoSigner": "No"
        ... }
        >>> result = predict_single(data)
        >>> print(result["prediction"])
        'No Default'
    
    Notes:
        - Uses global MODEL and THRESHOLD variables
        - Binary flag conversion is critical (was a bug fix)
        - Feature order must match training data
    """
    try:
        # ====================================================================
        # STEP 1: CONVERT TO DATAFRAME
        # ====================================================================
        # Model expects DataFrame input, not raw dictionary
        df = pd.DataFrame([data])
        
        # ====================================================================
        # STEP 2: HANDLE BINARY COLUMNS
        # ====================================================================
        # Convert Yes/No strings to 1/0 integers
        # This was added as a bug fix - API was receiving string "Yes"/"No"
        # but model expects numeric 1/0
        binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
        df = map_binary_flags(df, binary_cols)
        
        # ====================================================================
        # STEP 3: ENSURE CORRECT COLUMN ORDER
        # ====================================================================
        # Model expects features in specific order matching training data
        df = df[FEATURE_COLUMNS]
        
        # ====================================================================
        # STEP 4: GET PREDICTION PROBABILITY
        # ====================================================================
        # Pipeline applies preprocessing then XGBoost prediction
        # predict_proba returns shape (n_samples, 2): [prob_class_0, prob_class_1]
        # We take [0, 1] to get probability of positive class (default)
        probability = float(MODEL.predict_proba(df)[0, 1])
        
        # ====================================================================
        # STEP 5: MAKE BINARY PREDICTION
        # ====================================================================
        # Apply threshold to convert probability to binary decision
        # If probability >= threshold, predict Default, else No Default
        prediction = "Default" if probability >= THRESHOLD else "No Default"
        
        # ====================================================================
        # STEP 6: CALCULATE ADDITIONAL METRICS
        # ====================================================================
        # Get risk category and confidence score for richer response
        risk_level = get_risk_level(probability)
        confidence = calculate_confidence(probability)
        
        # ====================================================================
        # STEP 7: PACKAGE RESPONSE
        # ====================================================================
        return {
            "default_probability": round(probability, 4),  # Round to 4 decimal places
            "prediction": prediction,
            "risk_level": risk_level,
            "threshold": THRESHOLD,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()  # ISO 8601 format timestamp
        }
        
    except Exception as e:
        # Log error and re-raise with context
        logger.error(f"Prediction error: {str(e)}")
        raise
        
        # Get risk level and confidence
        risk_level = get_risk_level(probability)
        confidence = calculate_confidence(probability)
        
        return {
            "default_probability": round(probability, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "threshold": THRESHOLD,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# -------------------------
# API Endpoints
# -------------------------

@app.get("/", tags=["General"])
async def root():
    """Serve the main web application"""
    return FileResponse("static/index.html")


@app.get("/api", tags=["General"])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL is not None and THRESHOLD is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "threshold_loaded": THRESHOLD is not None,
        "model_path": MODEL_PATH,
        "threshold_path": THRESHOLD_PATH,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert to MB
        last_modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        
        return {
            "model_type": "XGBoost Pipeline with Preprocessing",
            "features": FEATURE_COLUMNS,
            "threshold": THRESHOLD,
            "model_file_size_mb": round(model_size, 2),
            "last_modified": last_modified.isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication):
    """
    Predict default probability for a single loan application.
    
    Returns:
    - default_probability: Probability of default (0-1)
    - prediction: Binary prediction (Default/No Default)
    - risk_level: Risk category (Low/Medium/High)
    - confidence: Prediction confidence score
    """
    if MODEL is None or THRESHOLD is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server health."
        )
    
    try:
        # Convert Pydantic model to dict
        data = application.model_dump()
        
        # Make prediction
        result = predict_single(data)
        
        logger.info(f"Prediction made: {result['prediction']} (prob: {result['default_probability']})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchLoanApplications):
    """
    Predict default probability for multiple loan applications.
    
    Maximum 1000 applications per request.
    """
    if MODEL is None or THRESHOLD is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server health."
        )
    
    try:
        predictions = []
        
        for app in batch.applications:
            data = app.model_dump()
            result = predict_single(data)
            predictions.append(result)
        
        logger.info(f"Batch prediction completed: {len(predictions)} applications processed")
        
        return {
            "predictions": predictions,
            "total_processed": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict/batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# -------------------------
# Error Handlers
# -------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
