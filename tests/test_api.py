
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

CUSTOMER = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 5,
    "Balance": 125000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 85000,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_success():
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["churn_probability"] <= 1
    assert result["predicted_churn"] in (0, 1)
    assert result["threshold"] == 0.30
    assert result["risk_level"] in ("Low", "Medium", "High")


def test_predict_missing_field():
    customer = CUSTOMER.copy()
    del customer["CreditScore"]
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_predict_invalid_age():
    customer = CUSTOMER.copy()
    customer["Age"] = -5
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_predict_invalid_credit_card():
    customer = CUSTOMER.copy()
    customer["HasCrCard"] = 3
    response = client.post("/predict", json=customer)
    assert response.status_code == 422
