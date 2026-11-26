"""Tests for feature engineering, model training, non-negativity, and holdout predictions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, NonNegativeRegressor, build_preprocessor, prepare_inference_dataframe
from src.predict import generate_holdout_predictions
from src.train import build_candidate_estimators, chronological_cv_evaluate


def test_prepare_inference_dataframe_extracts_exact_columns():
    records = [
        {
            "record_id": "R999",
            "date": "2026-09-12",
            "nearby_shops": 45,
            "days_to_salary": 5,
            "footfall_index": 72,
            "distance_to_market_km": 1.4,
            "avg_ticket_rs": 2400.0,
            "other_atms_1km": 8,
            "atm_type": " mall ",
            "area": " Market ",
        }
    ]
    df_feat = prepare_inference_dataframe(records)
    assert list(df_feat.columns) == FEATURE_COLUMNS
    assert df_feat.loc[0, "is_weekend"] == 1
    assert df_feat.loc[0, "month"] == 9
    assert df_feat.loc[0, "atm_type"] == "mall"
    assert df_feat.loc[0, "area"] == "Market"


def test_non_negative_regressor_clips_negative_predictions():
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "nearby_shops": rng.integers(5, 100, n),
            "days_to_salary": rng.integers(0, 30, n),
            "footfall_index": rng.integers(10, 100, n),
            "distance_to_market_km": rng.uniform(0.1, 6.0, n),
            "avg_ticket_rs": rng.integers(500, 5000, n),
            "other_atms_1km": rng.integers(0, 15, n),
            "day_of_week": rng.integers(0, 7, n),
            "day_of_month": rng.integers(1, 29, n),
            "month": rng.integers(1, 7, n),
            "is_weekend": rng.integers(0, 2, n),
            "atm_type": rng.choice(["standalone", "bank_branch", "mall"], n),
            "area": rng.choice(["Residential", "Highway", "Market"], n),
        }
    )
    y = pd.Series(np.maximum(0.0, 500.0 - 45.0 * df["other_atms_1km"] + rng.normal(0, 10, n)))

    candidates = build_candidate_estimators(random_state=42)
    assert len(candidates) >= 3

    cv_raw, cv_clip = chronological_cv_evaluate(candidates["ridge"], df, y, n_splits=3, clip_non_negative=True)
    assert "mae" in cv_raw and "mae" in cv_clip

    extreme_test = df.iloc[:5].copy()
    extreme_test["other_atms_1km"] = 25
    model = NonNegativeRegressor(candidates["ridge"], clip_non_negative=True).fit(df, y)
    preds = model.predict(extreme_test)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)


def test_saved_model_and_holdout_predictions_exist_and_valid(tmp_path):
    model_path = Path("artifacts/best_model.joblib")
    pred_path = Path("predictions/holdout_predictions.csv")
    holdout_raw = Path("data/holdout_features.csv")

    assert model_path.exists(), "Saved model artifact missing"
    assert pred_path.exists(), "Holdout predictions CSV missing"

    model = joblib.load(model_path)
    df_raw = pd.read_csv(holdout_raw)
    df_pred = pd.read_csv(pred_path)

    assert len(df_pred) == len(df_raw) == 250
    assert df_pred["record_id"].equals(df_raw["record_id"])
    assert "withdrawal_thousand_rs" not in df_raw.columns
    assert np.all(np.isfinite(df_pred["predicted_withdrawal_thousand_rs"]))
    assert (df_pred["predicted_withdrawal_thousand_rs"] >= 0.0).all()

    # Reproduce holdout predictions into tmp_path when holdout_clean exists
    holdout_clean = Path("data/processed/holdout_clean.csv")
    if holdout_clean.exists():
        repro_path = tmp_path / "repro_holdout.csv"
        repro_df = generate_holdout_predictions(
            model_path=str(model_path),
            holdout_path=str(holdout_clean),
            output_path=str(repro_path),
        )
        assert repro_df.equals(df_pred)
    else:
        # Still verify model can predict on engineered holdout features
        X_ho = prepare_inference_dataframe(df_raw.to_dict(orient="records"))
        preds = model.predict(X_ho)
        assert len(preds) == 250 and np.all(preds >= 0.0)
