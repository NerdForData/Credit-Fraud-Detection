@echo off
REM Quick Start Script for Credit Risk Prediction API

echo ================================
echo Credit Risk Prediction API
echo ================================
echo.

echo Starting API server...
echo API will be available at: http://localhost:8000
echo Documentation at: http://localhost:8000/docs
echo.

venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
