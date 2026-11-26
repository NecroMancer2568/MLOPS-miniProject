"""Training, chronological validation, drift-adaptation comparison, and MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    NonNegativeRegressor,
    build_preprocessor,
    compute_sha256,
    extract_features_from_cleaned_df,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MP076Trainer")


def load_params(params_path: Path = Path("params.yaml")) -> Dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def build_candidate_estimators(random_state: int) -> Dict[str, Pipeline]:
    """Return baseline and candidate regression pipelines."""
    return {
        "dummy_median": Pipeline(
            [("preprocessor", build_preprocessor()), ("model", DummyRegressor(strategy="median"))]
        ),
        "ridge": Pipeline(
            [("preprocessor", build_preprocessor()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=random_state, n_jobs=1
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.05,
                        max_depth=6,
                        min_samples_leaf=15,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def chronological_cv_evaluate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    clip_non_negative: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run TimeSeriesSplit CV returning unclipped and clipped aggregate metrics."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    unclipped_scores: List[Dict[str, float]] = []
    clipped_scores: List[Dict[str, float]] = []

    for train_idx, val_idx in tscv.split(X):
        model = NonNegativeRegressor(pipeline, clip_non_negative=False)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        raw_preds = model.predict(X.iloc[val_idx])
        clipped_preds = np.clip(raw_preds, 0.0, None) if clip_non_negative else raw_preds

        unclipped_scores.append(regression_metrics(y.iloc[val_idx].to_numpy(), raw_preds))
        clipped_scores.append(regression_metrics(y.iloc[val_idx].to_numpy(), clipped_preds))

    def _avg(items: List[Dict[str, float]]) -> Dict[str, float]:
        return {k: round(float(np.mean([d[k] for d in items])), 4) for k in items[0]}

    return _avg(unclipped_scores), _avg(clipped_scores)


def run_training(params_path: str = "params.yaml") -> Dict[str, Any]:
    params = load_params(Path(params_path))
    paths = params["paths"]
    train_cfg = params["training"]

    processed_dir = Path(paths["processed_dir"])
    artifacts_dir = Path(paths["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train_clean.csv"
    later_path = processed_dir / "later_clean.csv"

    df_train = pd.read_csv(train_path).sort_values("date").reset_index(drop=True)
    df_later = pd.read_csv(later_path).sort_values("date").reset_index(drop=True)

    dataset_hashes = {
        "train_clean_sha256": compute_sha256(train_path),
        "later_clean_sha256": compute_sha256(later_path),
    }

    X_train = extract_features_from_cleaned_df(df_train)
    y_train = df_train[TARGET_COLUMN].astype(np.float64)
    X_later = extract_features_from_cleaned_df(df_later)
    y_later = df_later[TARGET_COLUMN].astype(np.float64)

    random_state = int(train_cfg["random_state"])
    cv_splits = int(train_cfg["cv_splits"])
    clip_nn = bool(train_cfg["clip_non_negative"])
    later_weight = float(train_cfg["later_adaptation_weight"])

    mlflow.set_tracking_uri(train_cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(train_cfg["mlflow_experiment"])

    candidates = build_candidate_estimators(random_state=random_state)
    train_only_results: Dict[str, Any] = {}

    logger.info("Phase 1: Evaluating train-only models (chronological CV + out-of-time later evaluation)...")
    for name, pipeline in candidates.items():
        cv_unclipped, cv_clipped = chronological_cv_evaluate(
            pipeline, X_train, y_train, n_splits=cv_splits, clip_non_negative=clip_nn
        )

        raw_model = NonNegativeRegressor(pipeline, clip_non_negative=False).fit(X_train, y_train)
        raw_later_preds = raw_model.predict(X_later)
        clipped_later_preds = np.clip(raw_later_preds, 0.0, None) if clip_nn else raw_later_preds

        later_unclipped = regression_metrics(y_later.to_numpy(), raw_later_preds)
        later_clipped = regression_metrics(y_later.to_numpy(), clipped_later_preds)
        negative_raw_count = int((raw_later_preds < 0.0).sum())

        train_only_results[name] = {
            "cv_unclipped": cv_unclipped,
            "cv_clipped": cv_clipped,
            "later_unclipped": later_unclipped,
            "later_clipped": later_clipped,
            "later_negative_raw_predictions": negative_raw_count,
        }

        with mlflow.start_run(run_name=f"train_only_{name}"):
            mlflow.log_params(
                {
                    "strategy": "train_only",
                    "model_name": name,
                    "random_state": random_state,
                    "cv_splits": cv_splits,
                    "clip_non_negative": clip_nn,
                    "n_features": len(FEATURE_COLUMNS),
                    **dataset_hashes,
                }
            )
            mlflow.log_metrics(
                {
                    "cv_mae": cv_clipped["mae"],
                    "cv_rmse": cv_clipped["rmse"],
                    "cv_r2": cv_clipped["r2"],
                    "later_mae_unclipped": later_unclipped["mae"],
                    "later_mae": later_clipped["mae"],
                    "later_rmse": later_clipped["rmse"],
                    "later_r2": later_clipped["r2"],
                    "later_negative_raw_count": negative_raw_count,
                }
            )

    # Phase 2: Chronological adaptation check on later data
    # Hold out final 30% of later_clean chronologically; compare train-only vs train+early_later
    split_idx = int(len(df_later) * 0.7)
    X_later_early, y_later_early = X_later.iloc[:split_idx], y_later.iloc[:split_idx]
    X_later_recent, y_later_recent = X_later.iloc[split_idx:], y_later.iloc[split_idx:]

    X_adapted = pd.concat([X_train, X_later_early], ignore_index=True)
    y_adapted = pd.concat([y_train, y_later_early], ignore_index=True)
    weights_adapted = np.concatenate(
        [np.ones(len(X_train), dtype=np.float64), np.full(len(X_later_early), later_weight, dtype=np.float64)]
    )

    logger.info("Phase 2: Comparing train-only vs drift-adapted models on recent chronological later slice...")
    adaptation_results: Dict[str, Any] = {}
    best_strategy_key = ""
    best_recent_mae = float("inf")

    for name, pipeline in candidates.items():
        if name == "dummy_median":
            continue

        # Train-only evaluated on recent slice
        m_train_only = NonNegativeRegressor(pipeline, clip_non_negative=clip_nn).fit(X_train, y_train)
        metrics_to_recent = regression_metrics(y_later_recent.to_numpy(), m_train_only.predict(X_later_recent))

        # Adapted (train + early_later with recency weighting) evaluated on recent slice
        m_adapted = NonNegativeRegressor(pipeline, clip_non_negative=clip_nn).fit(
            X_adapted, y_adapted, sample_weight=weights_adapted
        )
        metrics_ad_recent = regression_metrics(y_later_recent.to_numpy(), m_adapted.predict(X_later_recent))

        adaptation_results[name] = {
            "train_only_on_later_recent": metrics_to_recent,
            "adapted_on_later_recent": metrics_ad_recent,
        }

        with mlflow.start_run(run_name=f"adapted_{name}"):
            mlflow.log_params(
                {
                    "strategy": "train_plus_later_weighted",
                    "model_name": name,
                    "later_adaptation_weight": later_weight,
                    "later_early_rows": len(X_later_early),
                    "later_recent_rows": len(X_later_recent),
                    **dataset_hashes,
                }
            )
            mlflow.log_metrics(
                {
                    "recent_later_mae_train_only": metrics_to_recent["mae"],
                    "recent_later_mae_adapted": metrics_ad_recent["mae"],
                    "recent_later_rmse_adapted": metrics_ad_recent["rmse"],
                    "recent_later_r2_adapted": metrics_ad_recent["r2"],
                }
            )

        for strat_label, m_dict in [("train_only", metrics_to_recent), ("adapted", metrics_ad_recent)]:
            if m_dict["mae"] < best_recent_mae:
                best_recent_mae = m_dict["mae"]
                best_strategy_key = f"{strat_label}:{name}"

    best_strategy, best_model_name = best_strategy_key.split(":")
    logger.info(f"Selected best model/strategy: {best_model_name} ({best_strategy}) with recent-later MAE={best_recent_mae}")

    # Refit final selected model using approved data for its strategy
    selected_pipeline = build_candidate_estimators(random_state=random_state)[best_model_name]
    if best_strategy == "adapted":
        X_final = pd.concat([X_train, X_later], ignore_index=True)
        y_final = pd.concat([y_train, y_later], ignore_index=True)
        w_final = np.concatenate(
            [np.ones(len(X_train), dtype=np.float64), np.full(len(X_later), later_weight, dtype=np.float64)]
        )
        final_model = NonNegativeRegressor(selected_pipeline, clip_non_negative=clip_nn).fit(
            X_final, y_final, sample_weight=w_final
        )
    else:
        final_model = NonNegativeRegressor(selected_pipeline, clip_non_negative=clip_nn).fit(X_train, y_train)

    model_path = Path(paths["model_path"])
    joblib.dump(final_model, model_path)

    # Log final production candidate run + attempt local MLflow model registration
    registry_status = {"attempted": True, "registered": False, "details": ""}
    with mlflow.start_run(run_name=f"final_{best_strategy}_{best_model_name}") as run:
        mlflow.log_params(
            {
                "selected_strategy": best_strategy,
                "selected_model": best_model_name,
                "clip_non_negative": clip_nn,
                "later_adaptation_weight": later_weight if best_strategy == "adapted" else 0.0,
                **dataset_hashes,
            }
        )
        mlflow.log_metrics({"selected_recent_later_mae": best_recent_mae})
        mlflow.log_artifact(str(model_path), artifact_path="joblib_model")
        try:
            mlflow.sklearn.log_model(
                final_model,
                name="model",
                serialization_format="cloudpickle",
            )
            model_uri = f"runs:/{run.info.run_id}/model"
            reg = mlflow.register_model(model_uri=model_uri, name=train_cfg["registered_model_name"])
            registry_status = {
                "attempted": True,
                "registered": True,
                "name": train_cfg["registered_model_name"],
                "version": str(reg.version),
                "run_id": run.info.run_id,
            }
        except Exception as exc:
            registry_status = {
                "attempted": True,
                "registered": False,
                "details": str(exc),
                "run_id": run.info.run_id,
            }

    best_train_only_name = min(
        train_only_results, key=lambda k: train_only_results[k]["later_clipped"]["mae"]
    )
    summary = {
        "primary_metric": "MAE",
        "dataset_hashes": dataset_hashes,
        "train_only_comparison": train_only_results,
        "best_train_only_on_later": {
            "model": best_train_only_name,
            **train_only_results[best_train_only_name]["later_clipped"],
        },
        "adaptation_comparison_on_recent_later": adaptation_results,
        "selected_final_model": {
            "model_name": best_model_name,
            "strategy": best_strategy,
            "recent_later_mae": best_recent_mae,
            "model_artifact": model_path.as_posix(),
        },
        "mlflow": {
            "experiment": train_cfg["mlflow_experiment"],
            "tracking_uri": train_cfg["mlflow_tracking_uri"],
            "registry": registry_status,
        },
    }

    metrics_path = Path(paths["metrics_path"])
    with open(metrics_path, "w", encoding="utf-8", newline="\n") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")

    logger.info(f"Saved model to '{model_path}' and metrics to '{metrics_path}'.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and compare MP076 regression models.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    run_training(params_path=args.params)


if __name__ == "__main__":
    main()
