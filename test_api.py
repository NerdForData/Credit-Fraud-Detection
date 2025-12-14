# test_api.py
"""
Test script for Credit Risk Prediction API

Usage:
    python test_api.py
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_info():
    """Test the model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n=== Testing Single Prediction ===")
    
    # Sample loan application
    loan_data = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=loan_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n=== Testing Batch Prediction ===")
    
    # Sample batch of applications
    batch_data = {
        "applications": [
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
            },
            {
                "Age": 28,
                "Income": 45000.0,
                "LoanAmount": 15000.0,
                "CreditScore": 650,
                "MonthsEmployed": 24,
                "NumCreditLines": 3,
                "InterestRate": 7.5,
                "LoanTerm": 48,
                "DTIRatio": 0.45,
                "Education": "High School",
                "EmploymentType": "Part-time",
                "MaritalStatus": "Single",
                "HasMortgage": "No",
                "HasDependents": "No",
                "LoanPurpose": "Auto",
                "HasCoSigner": "Yes"
            },
            {
                "Age": 45,
                "Income": 120000.0,
                "LoanAmount": 50000.0,
                "CreditScore": 780,
                "MonthsEmployed": 120,
                "NumCreditLines": 8,
                "InterestRate": 4.5,
                "LoanTerm": 84,
                "DTIRatio": 0.25,
                "Education": "Master's",
                "EmploymentType": "Full-time",
                "MaritalStatus": "Married",
                "HasMortgage": "Yes",
                "HasDependents": "Yes",
                "LoanPurpose": "Business",
                "HasCoSigner": "No"
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Total Processed: {result.get('total_processed')}")
    print(f"First Prediction: {json.dumps(result['predictions'][0], indent=2)}")
    return response.status_code == 200


def test_high_risk_application():
    """Test a high-risk loan application"""
    print("\n=== Testing High-Risk Application ===")
    
    # High-risk profile
    high_risk_data = {
        "Age": 22,
        "Income": 25000.0,
        "LoanAmount": 40000.0,
        "CreditScore": 580,
        "MonthsEmployed": 6,
        "NumCreditLines": 2,
        "InterestRate": 12.5,
        "LoanTerm": 36,
        "DTIRatio": 0.65,
        "Education": "High School",
        "EmploymentType": "Part-time",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "No",
        "LoanPurpose": "Other",
        "HasCoSigner": "No"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=high_risk_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def run_all_tests():
    """Run all API tests"""
    print("=" * 60)
    print("Credit Risk Prediction API - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("High-Risk Application", test_high_risk_application)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the API is running with: uvicorn src.api:app --reload")
