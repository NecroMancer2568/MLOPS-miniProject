"""Generate holdout predictions from the saved trained model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features import extract_features_from_cleaned_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MP076Predictor")


def generate_holdout_predictions(
    model_path: str = "artifacts/best_model.joblib",
    holdout_path: str = "data/processed/holdout_clean.csv",
    output_path: str = "predictions/holdout_predictions.csv",
) -> pd.DataFrame:
    model_file = Path(model_path)
    holdout_file = Path(holdout_path)
    out_file = Path(output_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Trained model not found at {model_file}")
    if not holdout_file.exists():
        raise FileNotFoundError(f"Cleaned holdout dataset not found at {holdout_file}")

    model = joblib.load(model_file)
    df_holdout = pd.read_csv(holdout_file)
    if "withdrawal_thousand_rs" in df_holdout.columns:
        raise ValueError("Holdout file must not contain target labels.")

    X_holdout = extract_features_from_cleaned_df(df_holdout)
    preds = np.asarray(model.predict(X_holdout), dtype=np.float64)

    if not np.all(np.isfinite(preds)):
        raise ValueError("Non-finite values found in holdout predictions.")
    preds = np.clip(preds, 0.0, None)

    out_df = pd.DataFrame(
        {
            "record_id": df_holdout["record_id"],
            "date": df_holdout["date"],
            "predicted_withdrawal_thousand_rs": np.round(preds, 2),
        }
    )

    if len(out_df) != len(df_holdout):
        raise AssertionError("Row count mismatch between holdout input and predictions.")
    if not out_df["record_id"].equals(df_holdout["record_id"]):
        raise AssertionError("record_id order mismatch in predictions.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False, lineterminator="\n")
    logger.info(f"Saved {len(out_df)} holdout predictions to '{out_file.as_posix()}'.")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Generate MP076 holdout predictions.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as fp:
        params = yaml.safe_load(fp)
    paths = params["paths"]
    generate_holdout_predictions(
        model_path=paths["model_path"],
        holdout_path=f"{paths['processed_dir']}/holdout_clean.csv",
        output_path=paths["holdout_predictions_path"],
    )


if __name__ == "__main__":
    main()
