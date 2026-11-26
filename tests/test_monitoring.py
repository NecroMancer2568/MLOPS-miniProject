"""Tests for data drift calculations and monitoring output."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift_monitor import compare_distributions, compute_categorical_drift, compute_psi


def test_compute_psi_handles_zero_bins_and_constant_arrays():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    same = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    shifted = np.array([10.0, 11.0, 12.0, 13.0, 14.0] * 10)
    constant = np.array([3.0] * 30)

    assert compute_psi(ref, same, bins=5) < 0.05
    assert compute_psi(ref, shifted, bins=5) > 0.25
    assert compute_psi(constant, shifted, bins=5) == 0.0
    assert compute_psi(np.array([np.nan, 1.0, 2.0]), np.array([1.0, 2.0]), bins=3) >= 0.0


def test_categorical_drift_and_compare_distributions():
    ref = pd.DataFrame(
        {
            "nearby_shops": [20, 40, 60, 80] * 10,
            "days_to_salary": [5, 10, 15, 20] * 10,
            "footfall_index": [30, 50, 70, 90] * 10,
            "distance_to_market_km": [1.0, 2.0, 3.0, 4.0] * 10,
            "avg_ticket_rs": [1000, 2000, 3000, 4000] * 10,
            "other_atms_1km": [2, 4, 6, 8] * 10,
            "day_of_week": [0, 1, 2, 3] * 10,
            "day_of_month": [1, 10, 20, 28] * 10,
            "month": [1, 2, 3, 4] * 10,
            "is_weekend": [0, 0, 0, 1] * 10,
            "atm_type": ["mall", "standalone", "bank_branch", "mall"] * 10,
            "area": ["Market", "Highway", "Residential", "Market"] * 10,
        }
    )
    cur = ref.copy()
    cur["other_atms_1km"] = 20

    cat_res = compute_categorical_drift(ref["area"], cur["area"])
    assert cat_res["psi"] == 0.0

    report = compare_distributions(ref, cur)
    assert report["numeric_features"]["other_atms_1km"]["psi_status"] == "ALERT"
    assert "other_atms_1km" in report["flagged_features"]


def test_generated_drift_report_exists_and_valid():
    report_path = Path("monitoring/reports/drift_report.json")
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "train_vs_later" in data
    assert "train_vs_holdout" in data
    assert data["retraining_trigger"]["auto_deploy"] is False
    assert data["retraining_trigger"]["status"] == "RETRAIN_RECOMMENDED"
