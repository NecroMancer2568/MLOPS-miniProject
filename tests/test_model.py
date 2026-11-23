"""
tests/test_model.py

Unit tests for model loading, prediction output validity, and schema handling.
Tests run completely on in-memory synthetic data fixtures without depending on external CSV files.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


@pytest.fixture
def model_path() -> Path:
    """Path to the exported champion model artifact."""
    return Path("models/model.joblib")


@pytest.fixture
def sample_feature_row() -> pd.DataFrame:
    """Synthetic single-row DataFrame matching the 8 required prediction features."""
    return pd.DataFrame(
        [
            {
                "nearby_shops": 55,
                "days_to_salary": 14,
                "footfall_index": 62,
                "distance_to_market_km": 2.8,
                "avg_ticket_rs": 2400,
                "other_atms_1km": 8,
                "atm_type": "standalone",
                "area": "Residential",
            }
        ]
    )


@pytest.fixture
def sample_feature_batch() -> pd.DataFrame:
    """Synthetic multi-row DataFrame matching the 8 required prediction features."""
    return pd.DataFrame(
        [
            {
                "nearby_shops": 55,
                "days_to_salary": 14,
                "footfall_index": 62,
                "distance_to_market_km": 2.8,
                "avg_ticket_rs": 2400,
                "other_atms_1km": 8,
                "atm_type": "standalone",
                "area": "Residential",
            },
            {
                "nearby_shops": 80,
                "days_to_salary": 1,
                "footfall_index": 85,
                "distance_to_market_km": 0.5,
                "avg_ticket_rs": 3500,
                "other_atms_1km": 3,
                "atm_type": "bank_branch",
                "area": "Market",
            },
            {
                "nearby_shops": 20,
                "days_to_salary": 28,
                "footfall_index": 30,
                "distance_to_market_km": 4.9,
                "avg_ticket_rs": 1800,
                "other_atms_1km": 15,
                "atm_type": "mall",
                "area": "Highway",
            },
        ]
    )


def test_model_artifact_exists_and_loads(model_path):
    """Verify that models/model.joblib exists and can be deserialized into a scikit-learn Pipeline."""
    assert model_path.exists(), f"Model file not found at {model_path}. Run training first."
    model = joblib.load(model_path)
    assert isinstance(model, Pipeline), "Loaded model must be an instance of sklearn.pipeline.Pipeline"


def test_single_row_prediction(model_path, sample_feature_row):
    """Verify inference on a single row returns valid float within reasonable boundaries."""
    model = joblib.load(model_path)
    preds = model.predict(sample_feature_row)

    assert len(preds) == 1
    val = float(preds[0])
    assert not np.isnan(val), "Prediction must not be NaN"
    assert val >= 0.0, f"Cash withdrawal prediction should be non-negative, got {val}"
    assert val <= 2500.0, f"Cash withdrawal prediction unexpectedly large: {val}"


def test_batch_prediction(model_path, sample_feature_batch):
    """Verify batch inference returns predictions matching input row count."""
    model = joblib.load(model_path)
    preds = model.predict(sample_feature_batch)

    assert len(preds) == len(sample_feature_batch)
    assert all(p >= 0.0 for p in preds)


def test_model_handles_unseen_categories(model_path):
    """Verify pipeline handles unseen/novel categorical values gracefully without throwing errors."""
    unseen_row = pd.DataFrame(
        [
            {
                "nearby_shops": 40,
                "days_to_salary": 10,
                "footfall_index": 50,
                "distance_to_market_km": 2.0,
                "avg_ticket_rs": 2000,
                "other_atms_1km": 5,
                "atm_type": "novel_kiosk_type",
                "area": "Industrial",
            }
        ]
    )
    model = joblib.load(model_path)
    preds = model.predict(unseen_row)
    assert len(preds) == 1
    assert not np.isnan(preds[0])
