# Credit Risk Prediction System 

An end-to-end machine learning system for predicting loan default probability with a production-ready REST API.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

This project implements a complete ML pipeline for credit risk assessment:
- **Data preprocessing** with robust feature engineering
- **XGBoost classifier** with optimized threshold tuning
- **Stratified K-Fold cross-validation** for reliable metrics
- **SHAP interpretability** for model explainability
- **Production-ready FastAPI** for real-time predictions
- **Interactive Web Interface** for easy user interaction
- **Docker containerization** for seamless deployment
- **Comprehensive evaluation** with ROC, PR curves, and confusion matrices

## 🚀 Features

✅ **Complete ML Pipeline**
- Automated preprocessing (scaling, encoding, imputation)
- Stratified K-fold cross-validation
- Threshold optimization for F1 score
- Model persistence with joblib

✅ **Interactive Web Application**
- Beautiful, responsive UI
- Real-time predictions
- Risk visualization with progress bars
- Sample data for quick testing
- Mobile-friendly design

✅ **REST API** 
- FastAPI with automatic OpenAPI documentation
- Single and batch prediction endpoints
- Health check and monitoring
- Input validation with Pydantic
- Error handling and logging

✅ **Docker Deployment**
- Containerized application
- Docker Compose for easy deployment
- Health checks and auto-restart
- Production-ready configuration

✅ **Model Interpretability**
- SHAP summary plots
- Feature importance analysis
- Risk categorization (Low/Medium/High)

✅ **Comprehensive Metrics**
- ROC-AUC, PR-AUC
- Precision, Recall, F1-score
- Confusion matrix visualization
- Custom threshold tuning

## 📁 Project Structure

```
Credit-Fraud-Detection/
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # Data preprocessing utilities
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Model evaluation & metrics
│   └── api.py              # FastAPI application
├── static/                 # Web UI files
│   ├── index.html          # Interactive web interface
│   ├── css/
│   │   └── styles.css      # Styling
│   └── js/
│       └── app.js          # Frontend logic
├── data/
│   └── raw/
│       └── Loan_default.csv
├── models/
│   ├── pipeline.joblib     # Trained model
│   └── threshold.json      # Optimized threshold
├── artifacts/
│   ├── evaluation_metrics.json
│   ├── evaluation_summary.json
│   ├── metrics.csv
│   └── plots/              # ROC, PR, SHAP plots
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker orchestration
├── start_api.bat           # Windows startup script
├── start_api.sh            # Linux/Mac startup script
├── test_api.py             # API test suite
├── API_USAGE.md            # API documentation
├── DOCKER_GUIDE.md         # Docker deployment guide
├── CODE_DOCUMENTATION.md   # Code documentation guide
├── WEB_UI_SUMMARY.md       # Web UI documentation
├── PROJECT_STRUCTURE.md    # Detailed project structure
├── IMPLEMENTATION_SUMMARY.md # Implementation details
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/NerdForData/Credit-Fraud-Detection.git
cd Credit-Fraud-Detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🌟 Quick Start - Web Application

The easiest way to use the system is through the web interface:

### Start the Application

**Option 1: Using the start script (Windows)**
```bash
start_api.bat
```

**Option 2: Using uvicorn directly**
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Option 3: Using Docker**
```bash
docker-compose up
```

### Access the Application

Once started, open your browser and navigate to:

- **🌐 Web Interface**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **📖 ReDoc**: http://localhost:8000/redoc
- **💚 Health Check**: http://localhost:8000/health

### Using the Web Interface

1. **Fill in the loan application form** with applicant details
2. **Click "Fill Sample Data"** to see an example (optional)
3. **Click "Predict Risk"** to get the prediction
4. **View Results** including:
   - Default probability
   - Risk level (Low/Medium/High)
   - Confidence score
   - Detailed interpretation

## 📊 Training the Model

Train the XGBoost model with cross-validation:

```bash
python -m src.train --data-path data/raw/Loan_default.csv
```

**Output:**
- Trained model: `models/pipeline.joblib`
- Optimized threshold: `models/threshold.json`
- Training metrics: `artifacts/metrics.csv`

**Training Features:**
- 5-fold stratified cross-validation
- Threshold tuning for optimal F1 score
- Handles class imbalance with scale_pos_weight
- Saves best model and threshold

## 📈 Model Evaluation

Evaluate model performance on test data:

```bash
python -m src.evaluate --data-path data/raw/Loan_default.csv
```

**Generated Artifacts:**
- `artifacts/evaluation_metrics.json` - Detailed metrics
- `artifacts/plots/roc_curve.png` - ROC curve
- `artifacts/plots/pr_curve.png` - Precision-Recall curve
- `artifacts/plots/confusion_matrix.png` - Confusion matrix
- `artifacts/plots/shap_summary.png` - SHAP feature importance
- `artifacts/predictions_sample.csv` - Sample predictions

## 🌐 API Usage

### Start the API Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

### Quick Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

result = response.json()
print(f"Default Probability: {result['default_probability']}")
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
```

**See [API_USAGE.md](API_USAGE.md) for complete documentation.**

## 🐳 Docker Deployment

### Quick Start with Docker

**1. Build and run with Docker Compose:**
```bash
docker-compose up --build
```

**2. Access the application:**
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs

**3. Stop the container:**
```bash
docker-compose down
```

### Manual Docker Commands

**Build the image:**
```bash
docker build -t credit-risk-api .
```

**Run the container:**
```bash
docker run -d -p 8000:8000 --name credit-risk-api credit-risk-api
```

**View logs:**
```bash
docker logs -f credit-risk-api
```

**See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed deployment instructions.**

## 🧪 Testing

Run the comprehensive API test suite:

```bash
python test_api.py
```

Tests include:
- Health check validation
- Single prediction
- Batch prediction
- High-risk scenario testing
- Error handling

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.817 |
| PR-AUC | 0.409 |
| Precision | 0.331 |
| Recall | 0.595 |
| F1-Score | 0.425 |
| Threshold | 0.603 |

**Note:** Metrics optimized for F1 score with threshold tuning.

## 🔍 Model Features

**Numeric Features:**
- Age, Income, LoanAmount
- CreditScore, MonthsEmployed
- NumCreditLines, InterestRate
- LoanTerm, DTIRatio

**Categorical Features:**
- Education, EmploymentType
- MaritalStatus, HasMortgage
- HasDependents, LoanPurpose
- HasCoSigner

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- MLflow experiment tracking
- Model drift detection
- A/B testing framework
- Authentication & authorization
- Database integration for predictions
- CI/CD pipeline (GitHub Actions)
- Advanced feature engineering
- Model versioning and registry

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- XGBoost for the gradient boosting framework
- FastAPI for the modern web framework
- SHAP for model interpretability
- scikit-learn for preprocessing utilities
---
