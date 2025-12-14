# API Usage Guide

## Credit Risk Prediction API Documentation

### Quick Start

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Start the API Server
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Access Interactive Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API and model are loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold_loaded": true,
  "model_path": "models/pipeline.joblib",
  "threshold_path": "models/threshold.json",
  "timestamp": "2025-12-14T10:30:00"
}
```

### 2. Model Information
**GET** `/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "model_type": "XGBoost Pipeline with Preprocessing",
  "features": ["Age", "Income", "LoanAmount", ...],
  "threshold": 0.6027,
  "model_file_size_mb": 2.45,
  "last_modified": "2025-12-14T09:15:00"
}
```

### 3. Single Prediction
**POST** `/predict`

Predict default probability for a single loan application.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "default_probability": 0.2543,
  "prediction": "No Default",
  "risk_level": "Low",
  "threshold": 0.6027,
  "confidence": 0.6954,
  "timestamp": "2025-12-14T10:30:00"
}
```

### 4. Batch Prediction
**POST** `/predict/batch`

Predict for multiple applications (up to 1000 per request).

**Request Body:**
```json
{
  "applications": [
    {
      "Age": 35,
      "Income": 75000.0,
      ...
    },
    {
      "Age": 28,
      "Income": 45000.0,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "default_probability": 0.2543,
      "prediction": "No Default",
      "risk_level": "Low",
      "threshold": 0.6027,
      "confidence": 0.6954,
      "timestamp": "2025-12-14T10:30:00"
    },
    ...
  ],
  "total_processed": 2,
  "timestamp": "2025-12-14T10:30:00"
}
```

---

## Python Client Examples

### Example 1: Single Prediction
```python
import requests

url = "http://localhost:8000/predict"
data = {
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

response = requests.post(url, json=data)
print(response.json())
```

### Example 2: Batch Prediction
```python
import requests

url = "http://localhost:8000/predict/batch"
data = {
    "applications": [
        {...},  # Application 1
        {...}   # Application 2
    ]
}

response = requests.post(url, json=data)
result = response.json()
print(f"Processed {result['total_processed']} applications")
for i, pred in enumerate(result['predictions']):
    print(f"Application {i+1}: {pred['prediction']} (prob: {pred['default_probability']})")
```

### Example 3: Health Check
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

---

## cURL Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 75000.0,
    "LoanAmount": 25000.0,
    "CreditScore": 720,
    "MonthsEmployed": 48,
    "NumCreditLines": 5,
    "InterestRate": 5.5,
    "LoanTerm": 60,
    "DTIRatio": 0.35,
    "Education": "Bachelor'"'"'s",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Home",
    "HasCoSigner": "No"
  }'
```

---

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will test:
- ✅ Health check endpoint
- ✅ Model info endpoint
- ✅ Single prediction
- ✅ Batch prediction
- ✅ High-risk application scenario

---

## Field Descriptions

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| Age | int | 18-100 | Applicant's age |
| Income | float | ≥0 | Annual income in dollars |
| LoanAmount | float | ≥0 | Requested loan amount |
| CreditScore | int | 300-850 | Credit score |
| MonthsEmployed | int | ≥0 | Months at current employment |
| NumCreditLines | int | ≥0 | Number of credit lines |
| InterestRate | float | 0-100 | Interest rate percentage |
| LoanTerm | int | ≥1 | Loan term in months |
| DTIRatio | float | ≥0 | Debt-to-Income ratio |
| Education | string | - | Education level |
| EmploymentType | string | - | Employment type |
| MaritalStatus | string | - | Marital status |
| HasMortgage | string | Yes/No | Has mortgage |
| HasDependents | string | Yes/No | Has dependents |
| LoanPurpose | string | - | Loan purpose |
| HasCoSigner | string | Yes/No | Has co-signer |

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| default_probability | float | Probability of default (0-1) |
| prediction | string | "Default" or "No Default" |
| risk_level | string | "Low", "Medium", or "High" |
| threshold | float | Classification threshold used |
| confidence | float | Prediction confidence (0-1) |
| timestamp | string | ISO format timestamp |

---

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK` - Success
- `422 Unprocessable Entity` - Invalid input data
- `500 Internal Server Error` - Server/model error
- `503 Service Unavailable` - Model not loaded

**Example Error Response:**
```json
{
  "detail": "Prediction failed: Invalid input format"
}
```

---

## Production Deployment Tips

1. **Use a production ASGI server:**
   ```bash
   gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Enable CORS if needed:**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(CORSMiddleware, allow_origins=["*"])
   ```

3. **Add authentication/rate limiting**

4. **Monitor with logging and metrics**

5. **Deploy with Docker** (see Dockerfile)

---

## Next Steps

- ✅ API is ready for integration
- 🔄 Add authentication (JWT tokens)
- 🔄 Implement rate limiting
- 🔄 Add request logging to database
- 🔄 Set up monitoring dashboards
- 🔄 Deploy with Docker/Kubernetes
