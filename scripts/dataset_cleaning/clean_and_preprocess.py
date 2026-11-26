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


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from categorical text columns before grouping or validation."""
    df = df.copy()
    for col in ["atm_type", "area"]:
        if col in df.columns:
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains null values.")
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_avg_ticket(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Fix unit inconsistency in avg_ticket_rs.
    Values < 100 were recorded in 'thousand Rs' (e.g., 3.03k Rs).
    Scale by 1000, round, and convert to integer.
    """
    df = df.copy()
    if df["avg_ticket_rs"].isnull().any():
        raise ValueError(f"[{dataset_name}] 'avg_ticket_rs' contains NaN values before integer conversion.")

    small_mask = df["avg_ticket_rs"] < 100
    count_small = int(small_mask.sum())

    if count_small > 0:
        logger.info(
            f"[{dataset_name}] Correcting {count_small} values in 'avg_ticket_rs' (< 100) by scaling x1000..."
        )
        df.loc[small_mask, "avg_ticket_rs"] = (
            df.loc[small_mask, "avg_ticket_rs"] * 1000.0
        ).round()

    df["avg_ticket_rs"] = df["avg_ticket_rs"].round().astype(np.int64)
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

    if df["withdrawal_thousand_rs"].isnull().any():
        raise ValueError(f"[{dataset_name}] 'withdrawal_thousand_rs' contains NaN values.")

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
) -> Tuple[pd.DataFrame, List[pd.DataFrame], Dict, Dict[str, int]]:
    """
    Impute sentinel missing values (-999 or < 0) in nearby_shops.
    Parameters are fit strictly on valid values in df_train to avoid data leakage.
    """
    df_train = clean_categorical_columns(df_train)
    if df_train["nearby_shops"].isnull().any():
        raise ValueError("[train] 'nearby_shops' contains NaN values before imputation.")

    valid_train = df_train[df_train["nearby_shops"] >= 0]
    if valid_train.empty:
        raise ValueError("[train] No valid non-negative 'nearby_shops' values to fit imputer.")

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

    def _impute_single(df: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, int]:
        df = clean_categorical_columns(df)
        if df["nearby_shops"].isnull().any():
            raise ValueError(f"[{name}] 'nearby_shops' contains NaN values before integer conversion.")

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

        df["nearby_shops"] = df["nearby_shops"].round().astype(np.int64)
        return df, invalid_count

    cleaned_train, train_imputed = _impute_single(df_train, "train")
    cleaned_others: List[pd.DataFrame] = []
    imputed_counts: Dict[str, int] = {"train": train_imputed}

    for df, name in other_dfs:
        cleaned_df, count = _impute_single(df, name)
        cleaned_others.append(cleaned_df)
        imputed_counts[name] = count

    return cleaned_train, cleaned_others, imputation_meta, imputed_counts


def add_engineered_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Extract temporal calendar features from date and clean categorical whitespace.
    """
    df = clean_categorical_columns(df)

    if df["date"].isnull().any():
        raise ValueError(f"[{dataset_name}] 'date' column contains null values.")

    # Parse dates strictly
    date_dt = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    df["day_of_week"] = date_dt.dt.dayofweek.astype(np.int64)
    df["day_of_month"] = date_dt.dt.day.astype(np.int64)
    df["month"] = date_dt.dt.month.astype(np.int64)
    df["is_weekend"] = (date_dt.dt.dayofweek >= 5).astype(np.int64)

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

    # record_id uniqueness and non-empty check
    assert "record_id" in df.columns, f"[{name}] record_id column missing!"
    assert df["record_id"].astype(str).str.len().gt(0).all(), f"[{name}] Empty record_id found!"
    assert df["record_id"].is_unique, f"[{name}] Duplicate record_id values found!"

    # Valid date parsing
    pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")

    # nearby_shops range
    assert (df["nearby_shops"] >= 0).all(), f"[{name}] Negative nearby_shops found!"
    assert (df["nearby_shops"] <= 150).all(), f"[{name}] Out-of-range nearby_shops found!"
    assert pd.api.types.is_integer_dtype(df["nearby_shops"]), f"[{name}] nearby_shops must be int!"

    # Numerical feature ranges
    assert (df["days_to_salary"] >= 0).all() and (df["days_to_salary"] <= 31).all(), (
        f"[{name}] Out-of-range days_to_salary found!"
    )
    assert (df["footfall_index"] >= 0).all() and (df["footfall_index"] <= 100).all(), (
        f"[{name}] Out-of-range footfall_index found!"
    )
    assert (df["distance_to_market_km"] >= 0.0).all() and (df["distance_to_market_km"] <= 50.0).all(), (
        f"[{name}] Out-of-range distance_to_market_km found!"
    )
    assert (df["other_atms_1km"] >= 0).all() and (df["other_atms_1km"] <= 25).all(), (
        f"[{name}] Out-of-range other_atms_1km found!"
    )

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
    assert df["atm_type"].isin(["standalone", "bank_branch", "mall"]).all(), (
        f"[{name}] Invalid atm_type category found!"
    )
    assert df["area"].isin(["Residential", "Highway", "Market"]).all(), (
        f"[{name}] Invalid area category found!"
    )

    # Temporal features validation
    assert df["day_of_week"].isin(range(7)).all(), f"[{name}] Invalid day_of_week found!"
    assert df["day_of_month"].isin(range(1, 32)).all(), f"[{name}] Invalid day_of_month found!"
    assert df["month"].isin(range(1, 13)).all(), f"[{name}] Invalid month found!"
    assert df["is_weekend"].isin([0, 1]).all(), f"[{name}] Invalid is_weekend found!"

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

    # 0. Clean categorical whitespace before any group-based operations
    df_train = clean_categorical_columns(df_train)
    df_later = clean_categorical_columns(df_later)
    df_holdout = clean_categorical_columns(df_holdout)

    # 1. Clean avg_ticket_rs
    df_train, train_small_tickets = clean_avg_ticket(df_train, "train")
    df_later, later_small_tickets = clean_avg_ticket(df_later, "later")
    df_holdout, holdout_small_tickets = clean_avg_ticket(df_holdout, "holdout")

    # 2. Clean target outliers (> 1000) in train.csv / later.csv
    df_train, train_target_outliers = clean_target_outliers(df_train, "train")
    df_later, later_target_outliers = clean_target_outliers(df_later, "later")

    # 3. Impute nearby_shops (fit strictly on train)
    (df_train, [df_later, df_holdout], imputer_meta, imputed_counts) = impute_nearby_shops(
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

    # 6. Save processed datasets with consistent LF line endings across OSes
    train_clean_path = output_path / "train_clean.csv"
    later_clean_path = output_path / "later_clean.csv"
    holdout_clean_path = output_path / "holdout_clean.csv"
    meta_path = output_path / "cleaning_summary.json"

    df_train.to_csv(train_clean_path, index=False, lineterminator="\n")
    df_later.to_csv(later_clean_path, index=False, lineterminator="\n")
    df_holdout.to_csv(holdout_clean_path, index=False, lineterminator="\n")

    summary = {
        "input_directory": input_path.as_posix(),
        "output_directory": output_path.as_posix(),
        "imputation": imputer_meta,
        "corrections": {
            "train": {
                "rows": len(df_train),
                "nearby_shops_imputed": imputed_counts["train"],
                "avg_ticket_scaled": train_small_tickets,
                "target_outliers_fixed": train_target_outliers,
                "zero_withdrawals_retained": int((df_train["withdrawal_thousand_rs"] == 0).sum()),
            },
            "later": {
                "rows": len(df_later),
                "nearby_shops_imputed": imputed_counts["later"],
                "avg_ticket_scaled": later_small_tickets,
                "target_outliers_fixed": later_target_outliers,
                "zero_withdrawals_retained": int((df_later["withdrawal_thousand_rs"] == 0).sum()),
            },
            "holdout": {
                "rows": len(df_holdout),
                "nearby_shops_imputed": imputed_counts["holdout"],
                "avg_ticket_scaled": holdout_small_tickets,
            },
        },
        "output_files": [
            train_clean_path.as_posix(),
            later_clean_path.as_posix(),
            holdout_clean_path.as_posix(),
        ],
    }

    with open(meta_path, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")

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
