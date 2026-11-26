"""
tests/test_api.py

Unit and integration tests for FastAPI application using TestClient.
Tests:
- GET /health returns {"status": "ok"}
- POST /predict returns valid predictions
- POST /predict rejects other_atms_1km > 25 with HTTP 422 (Client Rule 1)
- POST /predict rejects distance_to_market_km < 0 with HTTP 422 (Client Rule 2)
- Missing required fields return HTTP 422
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Valid prediction request payload with the 8 required fields."""
    return {
        "nearby_shops": 55,
        "days_to_salary": 14,
        "footfall_index": 62,
        "distance_to_market_km": 2.8,
        "avg_ticket_rs": 2400,
        "other_atms_1km": 8,
        "atm_type": "standalone",
        "area": "Residential",
    }


def test_health_check_returns_ok(client):
    """GET /health must return HTTP 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_200(client, valid_payload):
    """POST /predict with valid payload returns HTTP 200 and positive float prediction."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert data["prediction"] >= 0.0


def test_predict_rejects_other_atms_above_25(client, valid_payload):
    """
    Client Rule 1:
    'other_atms_1km above 25 is impossible for us. Reject it.'
    Must return HTTP 422.
    """
    bad_payload = valid_payload.copy()
    bad_payload["other_atms_1km"] = 26
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_distance(client, valid_payload):
    """
    Client Rule 2:
    'distance_to_market_km below 0 is always a typing mistake. Reject it.'
    Must return HTTP 422.
    """
    bad_payload = valid_payload.copy()
    bad_payload["distance_to_market_km"] = -1.5
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_boundary_values_accepted(client, valid_payload):
    """Boundary conditions: distance_to_market_km=0.0 and other_atms_1km=25 must be accepted."""
    boundary_payload = valid_payload.copy()
    boundary_payload["distance_to_market_km"] = 0.0
    boundary_payload["other_atms_1km"] = 25
    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_missing_required_field(client, valid_payload):
    """Request missing any required field must be rejected with HTTP 422."""
    incomplete_payload = valid_payload.copy()
    del incomplete_payload["footfall_index"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
