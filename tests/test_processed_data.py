from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def test_processed_files_exist():
    for filename in ("train_clean.csv", "later_clean.csv", "holdout_clean.csv"):
        assert (DATA_DIR / filename).exists()


def test_expected_row_counts():
    assert len(pd.read_csv(DATA_DIR / "train_clean.csv")) == 1000
    assert len(pd.read_csv(DATA_DIR / "later_clean.csv")) == 300
    assert len(pd.read_csv(DATA_DIR / "holdout_clean.csv")) == 250


def test_target_present_only_in_labeled_data():
    target = "withdrawal_thousand_rs"

    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    later = pd.read_csv(DATA_DIR / "later_clean.csv")
    holdout = pd.read_csv(DATA_DIR / "holdout_clean.csv")

    assert target in train.columns
    assert target in later.columns
    assert target not in holdout.columns


def test_record_ids_are_unique():
    for filename in ("train_clean.csv", "later_clean.csv", "holdout_clean.csv"):
        df = pd.read_csv(DATA_DIR / filename)
        assert df["record_id"].is_unique
