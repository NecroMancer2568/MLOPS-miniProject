"""
app/main.py

FastAPI Prediction Service for MP076 (ATM Cash Demand - Nagpur).
Endpoints:
- GET /health: Returns {"status": "ok"}
- POST /predict: Predicts cash withdrawal for the next day (in thousand rupees).
Enforces client validation rules via Pydantic schemas.
Logs one JSON line per prediction to stdout.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd

from app.logger import service_logger
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

MODEL_PATH = Path("models/model.joblib")
_model: Optional[Any] = None


def load_model(path: Path = MODEL_PATH):
    """Load the trained model pipeline artifact."""
    if not path.exists():
        service_logger.warning(f"Model file not found at {path}. Model must be trained.")
        return None
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading model artifact at startup."""
    global _model
    _model = load_model()
    if _model is not None:
        service_logger.info("Champion model loaded successfully at startup.")
    else:
        service_logger.warning("Service started without a loaded model artifact.")
    yield


app = FastAPI(
    title="MP076 ATM Cash Demand Prediction Service",
    description="Nagpur Cooperative Bank - ATM cash withdrawal prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["Health"],
)
async def health_check():
    """
    Returns service health status.
    Must return {"status": "ok"}.
    """
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict ATM cash demand",
    tags=["Prediction"],
)
async def predict_demand(payload: PredictionRequest):
    """
    Predict next-day cash withdrawal (in thousand rupees).
    Validates input schema and client constraints:
    1. other_atms_1km <= 25 (status 422 if invalid)
    2. distance_to_market_km >= 0 (status 422 if invalid)
    """
    global _model
    if _model is None:
        # Fallback dynamic load if not loaded during lifespan
        _model = load_model()
        if _model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available. Please ensure models/model.joblib exists.",
            )

    start_time = time.perf_counter()

    # Convert request payload to DataFrame matching training feature columns
    features = [
        "nearby_shops",
        "days_to_salary",
        "footfall_index",
        "distance_to_market_km",
        "avg_ticket_rs",
        "other_atms_1km",
        "atm_type",
        "area",
    ]
    input_data = {feat: [getattr(payload, feat)] for feat in features}
    input_df = pd.DataFrame(input_data)

    # Generate prediction
    try:
        raw_pred = float(_model.predict(input_df)[0])
    except Exception as exc:
        service_logger.error(f"Inference error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )

    # Cash withdrawal demand is non-negative
    prediction_value = max(0.0, round(raw_pred, 4))
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Structured JSON log to stdout
    log_record = {
        "event": "prediction",
        "input": payload.model_dump(),
        "prediction_thousand_rs": prediction_value,
        "latency_ms": latency_ms,
    }
    service_logger.info(
        f"Prediction generated: {prediction_value} thousand Rs",
        extra={"prediction_data": log_record},
    )

    return PredictionResponse(prediction=prediction_value)
