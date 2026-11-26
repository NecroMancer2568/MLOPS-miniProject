"""FastAPI service endpoint tests using TestClient."""

from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "record_id": "R_TEST_01",
    "date": "2026-09-15",
    "nearby_shops": 60,
    "days_to_salary": 5,
    "footfall_index": 70,
    "distance_to_market_km": 1.8,
    "avg_ticket_rs": 2500,
    "other_atms_1km": 12,
    "atm_type": "standalone",
    "area": "Residential",
}


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_predict_valid_request():
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["record_id"] == "R_TEST_01"
    assert isinstance(body["predicted_withdrawal_thousand_rs"], float)
    assert body["predicted_withdrawal_thousand_rs"] >= 0.0


def test_predict_invalid_inputs_return_422():
    with TestClient(app) as client:
        too_many_atms = {**VALID_PAYLOAD, "other_atms_1km": 26}
        assert client.post("/predict", json=too_many_atms).status_code == 422

        neg_dist = {**VALID_PAYLOAD, "distance_to_market_km": -0.5}
        assert client.post("/predict", json=neg_dist).status_code == 422

        bad_date = {**VALID_PAYLOAD, "date": "2026/09/15"}
        assert client.post("/predict", json=bad_date).status_code == 422

        bad_cat = {**VALID_PAYLOAD, "area": "Suburb"}
        assert client.post("/predict", json=bad_cat).status_code == 422


def test_predict_batch_and_metrics():
    batch_payload = {
        "records": [
            VALID_PAYLOAD,
            {**VALID_PAYLOAD, "record_id": "R_TEST_02", "other_atms_1km": 21, "area": "Highway"},
        ]
    }
    with TestClient(app) as client:
        resp = client.post("/predict/batch", json=batch_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2
        assert all(p["predicted_withdrawal_thousand_rs"] >= 0.0 for p in body["predictions"])

        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        assert "mp076_api_requests_total" in metrics_resp.text
