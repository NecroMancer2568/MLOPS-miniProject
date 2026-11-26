"""Shared feature definitions, preprocessing builders, and non-negative wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES: List[str] = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
]

CATEGORICAL_FEATURES: List[str] = ["atm_type", "area"]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "withdrawal_thousand_rs"

ALLOWED_ATM_TYPES = ["bank_branch", "mall", "standalone"]
ALLOWED_AREAS = ["Highway", "Market", "Residential"]


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """Wraps a sklearn regressor/pipeline to enforce finite, non-negative predictions."""

    def __init__(self, estimator: BaseEstimator, clip_non_negative: bool = True):
        self.estimator = estimator
        self.clip_non_negative = clip_non_negative

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None):
        self.estimator_ = clone(self.estimator)
        if sample_weight is not None:
            try:
                self.estimator_.fit(X, y, model__sample_weight=sample_weight)
            except (TypeError, ValueError):
                self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self.estimator_.predict(X), dtype=np.float64)
        if self.clip_non_negative:
            preds = np.clip(preds, 0.0, None)
        return preds


def build_preprocessor() -> ColumnTransformer:
    """Create a leak-free ColumnTransformer for numeric and categorical ATM features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[ALLOWED_ATM_TYPES, ALLOWED_AREAS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def extract_features_from_cleaned_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select ordered model feature columns from a cleaned dataset."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return df[FEATURE_COLUMNS].copy()


def prepare_inference_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw API records (with date string) into model-ready feature DataFrame."""
    df = pd.DataFrame(records)
    df["atm_type"] = df["atm_type"].astype(str).str.strip()
    df["area"] = df["area"].astype(str).str.strip()
    date_dt = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    df["day_of_week"] = date_dt.dt.dayofweek.astype(np.int64)
    df["day_of_month"] = date_dt.dt.day.astype(np.int64)
    df["month"] = date_dt.dt.month.astype(np.int64)
    df["is_weekend"] = (date_dt.dt.dayofweek >= 5).astype(np.int64)
    return extract_features_from_cleaned_df(df)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
