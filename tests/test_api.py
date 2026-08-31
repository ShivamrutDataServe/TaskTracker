"""
API-level tests, exercising the actual FastAPI app the way a real client
would. Requires model.pkl and metrics.json to already exist
(train_model.py must have been run first) - the CI workflow guarantees
this ordering.
"""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure_months": 12,
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "contract_type": 0,
    "support_calls": 2,
    "is_senior_citizen": 0,
    "has_tech_support": 0,
}


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_version_endpoint_reports_model_metrics():
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert "app_version" in body
    assert body["model_metrics"] is not None
    assert "f1_score" in body["model_metrics"]
    assert "roc_auc" in body["model_metrics"]


def test_predict_endpoint_returns_probability_and_label():
    response = client.post("/predict/", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)


def test_predict_endpoint_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "monthly_charges"}
    response = client.post("/predict/", json=incomplete)
    assert response.status_code == 422


def test_predict_endpoint_rejects_negative_charges():
    bad = {**VALID_CUSTOMER, "monthly_charges": -10.0}
    response = client.post("/predict/", json=bad)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_contract_type():
    bad = {**VALID_CUSTOMER, "contract_type": 5}
    response = client.post("/predict/", json=bad)
    assert response.status_code == 422


def test_predict_form_endpoint_renders_prediction():
    response = client.post(
        "/predict_form",
        data={
            "tenure_months": "12",
            "monthly_charges": "70.0",
            "total_charges": "840.0",
            "contract_type": "0",
            "support_calls": "2",
            "is_senior_citizen": "0",
            "has_tech_support": "0",
        },
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "churn probability" in response.text
