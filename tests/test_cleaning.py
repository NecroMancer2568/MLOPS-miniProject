"""Unit tests for data cleaning, validation, and leak-free imputation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.dataset_cleaning.clean_and_preprocess import (
    add_engineered_features,
    clean_avg_ticket,
    clean_categorical_columns,
    clean_target_outliers,
    impute_nearby_shops,
    run_pipeline,
    validate_dataset,
)


def _sample_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"R000{i}" for i in range(n)],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"][:n],
            "nearby_shops": [10, -999, 30, 50][:n],
            "days_to_salary": [5, 10, 15, 20][:n],
            "footfall_index": [40, 50, 60, 70][:n],
            "distance_to_market_km": [1.0, 2.0, 3.0, 4.0][:n],
            "avg_ticket_rs": [2000.0, 2.5, 1500.0, 3000.0][:n],
            "other_atms_1km": [3, 5, 7, 9][:n],
            "atm_type": [" mall ", "mall", "bank_branch", "standalone"][:n],
            "area": [" Market", "Market", "Highway", "Residential"][:n],
            "withdrawal_thousand_rs": [400.0, 5000.0, 0.0, 350.0][:n],
        }
    )


def test_clean_avg_ticket_and_nan_guard():
    df = _sample_df()
    cleaned, count = clean_avg_ticket(df, "test")
    assert count == 1
    assert cleaned.loc[1, "avg_ticket_rs"] == 2500
    assert pd.api.types.is_integer_dtype(cleaned["avg_ticket_rs"])

    df_nan = df.copy()
    df_nan.loc[0, "avg_ticket_rs"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        clean_avg_ticket(df_nan, "test")


def test_clean_target_outliers():
    df = _sample_df()
    cleaned, count = clean_target_outliers(df, "test")
    assert count == 1
    assert cleaned.loc[1, "withdrawal_thousand_rs"] == 500.0
    assert cleaned.loc[2, "withdrawal_thousand_rs"] == 0.0


def test_impute_nearby_shops_strips_whitespace_before_grouping_and_tracks_counts():
    df_train = _sample_df(4)
    df_later = _sample_df(2).drop(index=0).reset_index(drop=True)
    df_later.loc[0, "record_id"] = "R20001"
    df_later.loc[0, "nearby_shops"] = -999

    cleaned_train, [cleaned_later], meta, counts = impute_nearby_shops(
        df_train, [(df_later, "later")], strategy="group_median"
    )
    assert counts["train"] == 1
    assert counts["later"] == 1
    assert cleaned_train.loc[1, "nearby_shops"] == 10
    assert cleaned_later.loc[0, "nearby_shops"] == 10
    assert "Market|mall" in meta["group_medians"]


def test_validate_dataset_catches_invalid_month_and_duplicate_ids():
    df = clean_categorical_columns(_sample_df(2))
    df["nearby_shops"] = [10, 20]
    df["avg_ticket_rs"] = [2000, 2500]
    df["withdrawal_thousand_rs"] = [400.0, 500.0]
    df = add_engineered_features(df, "test")
    validate_dataset(df, "test", is_holdout=False)

    bad_month = df.copy()
    bad_month.loc[0, "month"] = 13
    with pytest.raises(AssertionError, match="month"):
        validate_dataset(bad_month, "test", is_holdout=False)

    dup_id = df.copy()
    dup_id.loc[1, "record_id"] = dup_id.loc[0, "record_id"]
    with pytest.raises(AssertionError, match="Duplicate record_id"):
        validate_dataset(dup_id, "test", is_holdout=False)


def test_run_pipeline_on_real_data_if_present(tmp_path):
    data_dir = Path("data")
    if not (data_dir / "train.csv").exists() or not (data_dir / "later.csv").exists():
        pytest.skip("Raw train.csv/later.csv not present in CI environment.")

    summary = run_pipeline(input_dir="data", output_dir=str(tmp_path))
    assert summary["corrections"]["train"]["nearby_shops_imputed"] == 45
    assert summary["corrections"]["train"]["avg_ticket_scaled"] == 196
    assert summary["corrections"]["train"]["target_outliers_fixed"] == 9
    assert summary["corrections"]["later"]["zero_withdrawals_retained"] == 27
    assert "\\" not in summary["output_directory"]
    assert (tmp_path / "train_clean.csv").read_bytes().count(b"\r\n") == 0
