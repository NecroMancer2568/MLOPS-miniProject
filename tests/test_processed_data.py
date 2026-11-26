import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    train = pd.DataFrame({
        "record_id": ["T001", "T002"],
        "nearby_shops": [5, 8],
        "withdrawal_thousand_rs": [120, 150],
    })

    later = pd.DataFrame({
        "record_id": ["L001", "L002"],
        "nearby_shops": [4, 7],
        "withdrawal_thousand_rs": [110, 160],
    })

    holdout = pd.DataFrame({
        "record_id": ["H001", "H002"],
        "nearby_shops": [6, 9],
    })

    return train, later, holdout


def test_target_present_only_in_labeled_data(sample_data):
    train, later, holdout = sample_data
    target = "withdrawal_thousand_rs"

    assert target in train.columns
    assert target in later.columns
    assert target not in holdout.columns


def test_record_ids_are_unique(sample_data):
    train, later, holdout = sample_data

    for df in (train, later, holdout):
        assert df["record_id"].is_unique


def test_expected_columns_exist(sample_data):
    train, later, holdout = sample_data

    assert "nearby_shops" in train.columns
    assert "nearby_shops" in later.columns
    assert "nearby_shops" in holdout.columns


def test_labeled_targets_are_numeric(sample_data):
    train, later, _ = sample_data

    assert pd.api.types.is_numeric_dtype(train["withdrawal_thousand_rs"])
    assert pd.api.types.is_numeric_dtype(later["withdrawal_thousand_rs"])