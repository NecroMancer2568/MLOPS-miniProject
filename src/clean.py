"""
src/clean.py

Data cleaning and preprocessing routines for MP076 (ATM Cash Demand - Nagpur).
Handles sentinel values (-999), unit mixup (Market x1000), and target typos (>1000 /10).
Provides scikit-learn ColumnTransformer for standard tabular preprocessing.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_nearby_shops(
    df: pd.DataFrame,
    sentinel: int = -999,
    imputed_value: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Clean sentinel value (-999) in nearby_shops.
    If imputed_value is not provided, computes median of valid entries (>= 0).
    """
    df = df.copy()
    if "nearby_shops" not in df.columns:
        return df, imputed_value or 0.0

    valid_mask = df["nearby_shops"] >= 0
    if imputed_value is None:
        if valid_mask.any():
            imputed_value = float(np.round(df.loc[valid_mask, "nearby_shops"].median()))
        else:
            imputed_value = 0.0

    invalid_mask = df["nearby_shops"] == sentinel
    if invalid_mask.any():
        df.loc[invalid_mask, "nearby_shops"] = int(imputed_value)

    df["nearby_shops"] = df["nearby_shops"].astype(np.int64)
    return df, imputed_value


def clean_avg_ticket(
    df: pd.DataFrame,
    threshold: float = 100.0,
    factor: float = 1000.0,
) -> pd.DataFrame:
    """
    Clean unit mixup in avg_ticket_rs.
    Rows with avg_ticket_rs < threshold were recorded in thousand rupees; scale by factor.
    """
    df = df.copy()
    if "avg_ticket_rs" not in df.columns:
        return df

    small_mask = df["avg_ticket_rs"] < threshold
    if small_mask.any():
        df.loc[small_mask, "avg_ticket_rs"] = (
            df.loc[small_mask, "avg_ticket_rs"] * factor
        ).round()

    df["avg_ticket_rs"] = df["avg_ticket_rs"].astype(np.int64)
    return df


def clean_target_typos(
    df: pd.DataFrame,
    target_col: str = "withdrawal_thousand_rs",
    threshold: float = 1000.0,
    divisor: float = 10.0,
) -> pd.DataFrame:
    """
    Clean target typos (extra zero / 10x error).
    Values > threshold are divided by divisor.
    """
    df = df.copy()
    if target_col not in df.columns:
        return df

    typo_mask = df[target_col] > threshold
    if typo_mask.any():
        df.loc[typo_mask, target_col] = (df.loc[typo_mask, target_col] / divisor).round(1)

    return df


def clean_data(
    df: pd.DataFrame,
    is_train: bool = True,
    imputed_nearby_shops: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end cleaning for training or evaluation datasets.
    Returns cleaned DataFrame and metadata dictionary with fitted parameters.
    """
    df = df.copy()

    # 1. Clean nearby_shops
    df, fitted_impute = clean_nearby_shops(
        df, sentinel=-999, imputed_value=imputed_nearby_shops
    )

    # 2. Clean avg_ticket_rs
    df = clean_avg_ticket(df, threshold=100.0, factor=1000.0)

    # 3. Clean target typos if target is present
    if is_train and "withdrawal_thousand_rs" in df.columns:
        df = clean_target_typos(df, target_col="withdrawal_thousand_rs", threshold=1000.0, divisor=10.0)

    metadata = {
        "imputed_nearby_shops": fitted_impute,
    }
    return df, metadata


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for standard tabular preprocessing.
    Numeric: StandardScaler.
    Categorical: OneHotEncoder (handle_unknown='ignore').
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
