"""FastAPI inference service for MP076 ATM Cash Demand Prediction."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import prepare_inference_dataframe

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MP076API")

DEFAULT_MODEL_PATH = Path(os.getenv("MP076_MODEL_PATH", "artifacts/best_model.joblib"))

REQUEST_COUNT = Counter(
    "mp076_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "mp076_prediction_latency_seconds",
    "Model inference latency in seconds",
    ["endpoint"],
)
PREDICTED_AMOUNT = Histogram(
    "mp076_predicted_withdrawal_thousand_rs",
    "Predicted ATM cash demand distribution (thousand Rs)",
    buckets=(0, 50, 150, 300, 450, 600, 750, 900, 1200),
)

_MODEL_STATE: Dict[str, Any] = {"model": None, "model_path": str(DEFAULT_MODEL_PATH)}


def log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


def get_model() -> Any:
    if _MODEL_STATE["model"] is None:
        model_path = Path(os.getenv("MP076_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at '{model_path.as_posix()}'.",
            )
        _MODEL_STATE["model"] = joblib.load(model_path)
        _MODEL_STATE["model_path"] = model_path.as_posix()
        log_event("model_loaded", model_path=model_path.as_posix())
    return _MODEL_STATE["model"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_path = Path(os.getenv("MP076_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if model_path.exists():
        _MODEL_STATE["model"] = joblib.load(model_path)
        _MODEL_STATE["model_path"] = model_path.as_posix()
        log_event("startup_model_loaded", model_path=model_path.as_posix())
    else:
        log_event("startup_model_missing", model_path=model_path.as_posix())
    yield


app = FastAPI(
    title="MP076 ATM Cash Demand Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class ATMFeatureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: Optional[str] = Field(default=None, description="Optional record identifier")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    nearby_shops: int = Field(..., ge=0, le=150)
    days_to_salary: int = Field(..., ge=0, le=31)
    footfall_index: int = Field(..., ge=0, le=100)
    distance_to_market_km: float = Field(..., ge=0.0, le=50.0)
    avg_ticket_rs: float = Field(..., ge=500.0, le=6000.0)
    other_atms_1km: int = Field(..., ge=0, le=25)
    atm_type: Literal["standalone", "bank_branch", "mall"]
    area: Literal["Residential", "Highway", "Market"]

    @field_validator("atm_type", "area", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must be a valid YYYY-MM-DD date") from exc
        return value

    @field_validator("distance_to_market_km", "avg_ticket_rs")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Value must be finite")
        return value


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[ATMFeatureInput] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    record_id: Optional[str] = None
    predicted_withdrawal_thousand_rs: float


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


@app.get("/health")
def health() -> Dict[str, Any]:
    model_path = Path(os.getenv("MP076_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    loaded = _MODEL_STATE["model"] is not None or model_path.exists()
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_path": model_path.as_posix(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ATMFeatureInput) -> PredictionResponse:
    start = time.perf_counter()
    model = get_model()
    X = prepare_inference_dataframe([payload.model_dump()])
    pred = float(model.predict(X)[0])
    if not math.isfinite(pred):
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail="Model produced non-finite prediction.")
    pred = round(max(0.0, pred), 2)

    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.labels(endpoint="/predict").observe(elapsed)
    PREDICTED_AMOUNT.observe(pred)
    REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
    log_event("predict", record_id=payload.record_id, prediction=pred, latency_s=round(elapsed, 6))
    return PredictionResponse(record_id=payload.record_id, predicted_withdrawal_thousand_rs=pred)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    start = time.perf_counter()
    model = get_model()
    records = [item.model_dump() for item in payload.records]
    X = prepare_inference_dataframe(records)
    raw_preds = model.predict(X)

    results: List[PredictionResponse] = []
    for item, val in zip(payload.records, raw_preds):
        pred = float(val)
        if not math.isfinite(pred):
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="500").inc()
            raise HTTPException(status_code=500, detail="Model produced non-finite prediction.")
        pred = round(max(0.0, pred), 2)
        PREDICTED_AMOUNT.observe(pred)
        results.append(
            PredictionResponse(record_id=item.record_id, predicted_withdrawal_thousand_rs=pred)
        )

    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(elapsed)
    REQUEST_COUNT.labels(endpoint="/predict/batch", status="200").inc()
    log_event("predict_batch", count=len(results), latency_s=round(elapsed, 6))
    return BatchPredictionResponse(count=len(results), predictions=results)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
