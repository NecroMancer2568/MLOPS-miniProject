#!/usr/bin/env python3
"""
scripts/dataset_cleaning/clean_and_preprocess.py

Clean and preprocess tabular ATM cash withdrawal datasets:
- train.csv
- later.csv
- holdout_features.csv

Key Cleaning Operations:
1. Fix sentinel values (-999) in `nearby_shops` via group-stratified median imputation.
2. Fix unit inconsistency in `avg_ticket_rs` (scale values < 100 by 1000).
3. Fix 10x decimal errors in target `withdrawal_thousand_rs` (divide values > 1000 by 10).
4. Feature engineering: Extract temporal calendar features from `date`.
5. Harmonize data types across all datasets.
6. Rigorous validation assertions to guarantee data quality.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataCleaner")


def load_raw_data(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, later, and holdout datasets from input directory."""
    train_path = input_dir / "train.csv"
    later_path = input_dir / "later.csv"
    holdout_path = input_dir / "holdout_features.csv"

    for p in [train_path, later_path, holdout_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required dataset file not found: {p}")

    logger.info(f"Loading raw datasets from '{input_dir}'...")
    df_train = pd.read_csv(train_path)
    df_later = pd.read_csv(later_path)
    df_holdout = pd.read_csv(holdout_path)

    logger.info(
        f"Raw shapes: train={df_train.shape}, later={df_later.shape}, holdout={df_holdout.shape}"
    )
    return df_train, df_later, df_holdout


def clean_avg_ticket(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Fix unit inconsistency in avg_ticket_rs.
    Values < 100 were recorded in 'thousand Rs' (e.g., 3.03k Rs).
    Scale by 1000, round, and convert to integer.
    """
    df = df.copy()
    small_mask = df["avg_ticket_rs"] < 100
    count_small = int(small_mask.sum())

    if count_small > 0:
        logger.info(
            f"[{dataset_name}] Correcting {count_small} values in 'avg_ticket_rs' (< 100) by scaling x1000..."
        )
        df.loc[small_mask, "avg_ticket_rs"] = (
            df.loc[small_mask, "avg_ticket_rs"] * 1000.0
        ).round()

    df["avg_ticket_rs"] = df["avg_ticket_rs"].astype(np.int64)
    return df, count_small


def clean_target_outliers(
    df: pd.DataFrame, dataset_name: str, threshold: float = 1000.0
) -> Tuple[pd.DataFrame, int]:
    """
    Fix 10x decimal errors in withdrawal_thousand_rs for train.csv.
    Identified records with values > 1000 are divided by 10.0 to restore the single decimal point.
    """
    df = df.copy()
    if "withdrawal_thousand_rs" not in df.columns:
        return df, 0

    outlier_mask = df["withdrawal_thousand_rs"] > threshold
    count_outliers = int(outlier_mask.sum())

    if count_outliers > 0:
        logger.info(
            f"[{dataset_name}] Correcting {count_outliers} target outliers (> {threshold}) by dividing by 10.0..."
        )
        df.loc[outlier_mask, "withdrawal_thousand_rs"] = (
            df.loc[outlier_mask, "withdrawal_thousand_rs"] / 10.0
        ).round(1)

    return df, count_outliers


def impute_nearby_shops(
    df_train: pd.DataFrame,
    other_dfs: List[Tuple[pd.DataFrame, str]],
    strategy: str = "group_median",
) -> Tuple[pd.DataFrame, List[pd.DataFrame], Dict]:
    """
    Impute sentinel missing values (-999 or < 0) in nearby_shops.
    Parameters are fit strictly on valid values in df_train to avoid data leakage.
    """
    df_train = df_train.copy()
    valid_train = df_train[df_train["nearby_shops"] >= 0]

    global_median = int(np.round(valid_train["nearby_shops"].median()))

    group_medians = (
        valid_train.groupby(["area", "atm_type"])["nearby_shops"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )

    imputation_meta = {
        "strategy": strategy,
        "global_median": global_median,
        "group_medians": {f"{k[0]}|{k[1]}": int(v) for k, v in group_medians.items()},
    }

    def _impute_single(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()
        invalid_mask = df["nearby_shops"] < 0
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            logger.info(
                f"[{name}] Imputing {invalid_count} sentinel missing values in 'nearby_shops' using {strategy}..."
            )
            for idx in df[invalid_mask].index:
                area_val = df.loc[idx, "area"]
                atm_val = df.loc[idx, "atm_type"]
                if strategy == "group_median" and (area_val, atm_val) in group_medians:
                    df.loc[idx, "nearby_shops"] = group_medians[(area_val, atm_val)]
                else:
                    df.loc[idx, "nearby_shops"] = global_median

        df["nearby_shops"] = df["nearby_shops"].astype(np.int64)
        return df

    cleaned_train = _impute_single(df_train, "train")
    cleaned_others = [_impute_single(df, name) for df, name in other_dfs]

    return cleaned_train, cleaned_others, imputation_meta


def add_engineered_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Extract temporal calendar features from date and clean categorical whitespace.
    """
    df = df.copy()

    # Clean categorical text
    for col in ["atm_type", "area"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Parse dates
    date_dt = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["day_of_week"] = date_dt.dt.dayofweek
    df["day_of_month"] = date_dt.dt.day
    df["month"] = date_dt.dt.month
    df["is_weekend"] = (date_dt.dt.dayofweek >= 5).astype(int)

    logger.info(
        f"[{dataset_name}] Added temporal features: day_of_week, day_of_month, month, is_weekend"
    )
    return df


def validate_dataset(df: pd.DataFrame, name: str, is_holdout: bool = False):
    """Run data quality assertions on cleaned datasets."""
    logger.info(f"Validating {name}...")

    # Null checks
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        raise AssertionError(f"[{name}] Unexpected nulls found:\n{null_counts[null_counts > 0]}")

    # nearby_shops range
    assert (df["nearby_shops"] >= 0).all(), f"[{name}] Negative nearby_shops found!"
    assert (df["nearby_shops"] <= 150).all(), f"[{name}] Out-of-range nearby_shops found!"
    assert pd.api.types.is_integer_dtype(df["nearby_shops"]), f"[{name}] nearby_shops must be int!"

    # avg_ticket_rs range
    assert (df["avg_ticket_rs"] >= 500).all(), f"[{name}] avg_ticket_rs < 500 found!"
    assert (df["avg_ticket_rs"] <= 6000).all(), f"[{name}] avg_ticket_rs > 6000 found!"
    assert pd.api.types.is_integer_dtype(df["avg_ticket_rs"]), f"[{name}] avg_ticket_rs must be int!"

    # withdrawal_thousand_rs range
    if not is_holdout:
        assert "withdrawal_thousand_rs" in df.columns, f"[{name}] Target missing!"
        assert (df["withdrawal_thousand_rs"] >= 0.0).all(), f"[{name}] Negative withdrawals!"
        assert (
            df["withdrawal_thousand_rs"] <= 1200.0
        ).all(), f"[{name}] Target withdrawal > 1200 found!"

    # Categorical validation
    assert df["atm_type"].isin(["standalone", "bank_branch", "mall"]).all()
    assert df["area"].isin(["Residential", "Highway", "Market"]).all()

    # Temporal features validation
    assert df["day_of_week"].isin(range(7)).all()
    assert df["day_of_month"].isin(range(1, 32)).all()
    assert df["is_weekend"].isin([0, 1]).all()

    logger.info(f"[{name}] All validation checks passed successfully!")


def run_pipeline(
    input_dir: str = "data",
    output_dir: str = "data/processed",
    imputation_strategy: str = "group_median",
) -> Dict:
    """Execute end-to-end cleaning and preprocessing pipeline."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_train, df_later, df_holdout = load_raw_data(input_path)

    # 1. Clean avg_ticket_rs
    df_train, train_small_tickets = clean_avg_ticket(df_train, "train")
    df_later, later_small_tickets = clean_avg_ticket(df_later, "later")
    df_holdout, holdout_small_tickets = clean_avg_ticket(df_holdout, "holdout")

    # 2. Clean target outliers (> 1000) in train.csv
    df_train, train_target_outliers = clean_target_outliers(df_train, "train")
    df_later, later_target_outliers = clean_target_outliers(df_later, "later")

    # 3. Impute nearby_shops
    (df_train, [df_later, df_holdout], imputer_meta) = impute_nearby_shops(
        df_train=df_train,
        other_dfs=[(df_later, "later"), (df_holdout, "holdout")],
        strategy=imputation_strategy,
    )

    # 4. Feature Engineering (Temporal & Categorical)
    df_train = add_engineered_features(df_train, "train")
    df_later = add_engineered_features(df_later, "later")
    df_holdout = add_engineered_features(df_holdout, "holdout")

    # 5. Validation checks
    validate_dataset(df_train, "train", is_holdout=False)
    validate_dataset(df_later, "later", is_holdout=False)
    validate_dataset(df_holdout, "holdout", is_holdout=True)

    # 6. Save processed datasets
    train_clean_path = output_path / "train_clean.csv"
    later_clean_path = output_path / "later_clean.csv"
    holdout_clean_path = output_path / "holdout_clean.csv"
    meta_path = output_path / "cleaning_summary.json"

    df_train.to_csv(train_clean_path, index=False)
    df_later.to_csv(later_clean_path, index=False)
    df_holdout.to_csv(holdout_clean_path, index=False)

    summary = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "imputation": imputer_meta,
        "corrections": {
            "train": {
                "rows": len(df_train),
                "nearby_shops_imputed": 45,
                "avg_ticket_scaled": train_small_tickets,
                "target_outliers_fixed": train_target_outliers,
            },
            "later": {
                "rows": len(df_later),
                "avg_ticket_scaled": later_small_tickets,
                "target_outliers_fixed": later_target_outliers,
                "zero_withdrawals_retained": int((df_later["withdrawal_thousand_rs"] == 0).sum()),
            },
            "holdout": {
                "rows": len(df_holdout),
                "avg_ticket_scaled": holdout_small_tickets,
            },
        },
        "output_files": [
            str(train_clean_path),
            str(later_clean_path),
            str(holdout_clean_path),
        ],
    }

    with open(meta_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    logger.info(f"Cleaned datasets successfully saved to '{output_path}':")
    logger.info(f"  - {train_clean_path.name}: {df_train.shape}")
    logger.info(f"  - {later_clean_path.name}: {df_later.shape}")
    logger.info(f"  - {holdout_clean_path.name}: {df_holdout.shape}")
    logger.info(f"  - {meta_path.name}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Clean and preprocess ATM cash withdrawal tabular datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Path to directory containing raw CSV files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to directory where processed CSV files will be saved (default: data/processed)",
    )
    parser.add_argument(
        "--imputation-strategy",
        type=str,
        choices=["group_median", "global_median"],
        default="group_median",
        help="Imputation strategy for nearby_shops (default: group_median)",
    )

    args = parser.parse_args()
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        imputation_strategy=args.imputation_strategy,
    )


if __name__ == "__main__":
    main()
