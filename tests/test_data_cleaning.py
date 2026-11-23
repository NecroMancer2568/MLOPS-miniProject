"""
tests/test_data_cleaning.py

Unit tests for data cleaning logic for MP076 (ATM Cash Demand - Nagpur).
Tests are completely self-contained with synthetic test fixtures (zero dependency on external CSVs).
"""

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    clean_avg_ticket,
    clean_data,
    clean_nearby_shops,
    clean_target_typos,
)


@pytest.fixture
def synthetic_raw_data() -> pd.DataFrame:
    """Fixture providing a synthetic raw dataframe with the three known data issues."""
    return pd.DataFrame(
        {
            "record_id": ["R001", "R002", "R003", "R004", "R005"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "nearby_shops": [50, -999, 60, -999, 70],  # Contains sentinel values
            "days_to_salary": [5, 10, 15, 20, 25],
            "footfall_index": [40, 50, 60, 70, 80],
            "distance_to_market_km": [2.5, 3.0, 1.2, 4.5, 0.8],
            "avg_ticket_rs": [2500.0, 2.74, 3100.0, 1.50, 2800.0],  # Unit mixup (2.74, 1.50)
            "other_atms_1km": [5, 8, 10, 12, 6],
            "atm_type": ["standalone", "bank_branch", "mall", "standalone", "bank_branch"],
            "area": ["Highway", "Market", "Residential", "Market", "Highway"],
            "withdrawal_thousand_rs": [450.0, 5200.0, 600.0, 380.0, 7090.0],  # Typos (5200, 7090)
        }
    )


def test_clean_nearby_shops_replaces_sentinel():
    """Verify sentinel value -999 is replaced by the median of valid values."""
    df = pd.DataFrame({"nearby_shops": [10, -999, 30, -999, 50]})
    cleaned_df, imputed_val = clean_nearby_shops(df, sentinel=-999)

    assert (cleaned_df["nearby_shops"] >= 0).all()
    assert -999 not in cleaned_df["nearby_shops"].values
    assert imputed_val == 30.0
    assert list(cleaned_df["nearby_shops"]) == [10, 30, 30, 30, 50]


def test_clean_avg_ticket_scales_small_values():
    """Verify values < 100 are multiplied by 1000 and converted to integer."""
    df = pd.DataFrame({"avg_ticket_rs": [2500.0, 2.74, 1.50, 4200.0]})
    cleaned_df = clean_avg_ticket(df, threshold=100.0, factor=1000.0)

    assert (cleaned_df["avg_ticket_rs"] >= 500).all()
    assert pd.api.types.is_integer_dtype(cleaned_df["avg_ticket_rs"])
    assert list(cleaned_df["avg_ticket_rs"]) == [2500, 2740, 1500, 4200]


def test_clean_target_typos_divides_by_ten():
    """Verify target typos (> 1000) are divided by 10 and rounded to 1 decimal place."""
    df = pd.DataFrame({"withdrawal_thousand_rs": [450.0, 5200.0, 600.0, 7090.0]})
    cleaned_df = clean_target_typos(df, threshold=1000.0, divisor=10.0)

    assert (cleaned_df["withdrawal_thousand_rs"] <= 1000.0).all()
    assert list(cleaned_df["withdrawal_thousand_rs"]) == [450.0, 520.0, 600.0, 709.0]


def test_clean_data_end_to_end(synthetic_raw_data):
    """Verify full clean_data function correctly resolves all issues simultaneously."""
    cleaned_df, meta = clean_data(synthetic_raw_data, is_train=True)

    # Invariants
    assert (cleaned_df["nearby_shops"] >= 0).all()
    assert -999 not in cleaned_df["nearby_shops"].values
    assert (cleaned_df["avg_ticket_rs"] >= 500).all()
    assert (cleaned_df["withdrawal_thousand_rs"] <= 1000.0).all()
    assert len(cleaned_df) == len(synthetic_raw_data)
    assert meta["imputed_nearby_shops"] > 0
