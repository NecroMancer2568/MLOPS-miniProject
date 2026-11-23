"""
src/train.py

Model training, cross-validation, and MLflow tracking for MP076.
Logs 3+ candidate models to SQLite MLflow database (sqlite:///mlflow.db).
Logs data_sha256, metrics containing 'mae', registers version 1, sets @champion alias,
exports models/model.joblib, and updates findings.yaml.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import yaml

from src.clean import build_preprocessor, clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelTrainer")


def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """Load settings from params.yaml using safe_load."""
    with open(config_path, "r") as fp:
        return yaml.safe_load(fp)


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    with open(file_path, "rb") as fp:
        return hashlib.sha256(fp.read()).hexdigest()


def get_candidate_models(random_state: int = 42) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Define candidate regression models and their key parameters."""
    return [
        (
            "Ridge",
            Ridge(alpha=1.0),
            {"alpha": 1.0},
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=random_state
            ),
            {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 3},
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=random_state
            ),
            {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1},
        ),
    ]


def update_findings_tracking(
    findings_path: Path,
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: int = 1,
) -> None:
    """Update tracking block in findings.yaml preserving formatting."""
    with open(findings_path, "r") as fp:
        content = fp.read()

    lines = content.splitlines()
    new_lines = []
    in_tracking = False

    for line in lines:
        if line.startswith("tracking:"):
            in_tracking = True
            new_lines.append(line)
            continue
        elif in_tracking and line.startswith("monitoring:"):
            in_tracking = False
            new_lines.append(line)
            continue

        if in_tracking:
            if line.strip().startswith("experiment_name:"):
                new_lines.append(f'  experiment_name: "{experiment_name}"')
            elif line.strip().startswith("best_run_id:"):
                new_lines.append(f'  best_run_id: "{best_run_id}"        # the MLflow run id of your v1 model')
            elif line.strip().startswith("registered_model_name:"):
                new_lines.append(f'  registered_model_name: "{registered_model_name}"')
            elif line.strip().startswith("champion_version:"):
                new_lines.append(f"  champion_version: {champion_version}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(findings_path, "w") as fp:
        fp.write("\n".join(new_lines) + "\n")


def train_and_track() -> Dict[str, Any]:
    """Main training routine with MLflow tracking and champion model export."""
    config = load_config()

    raw_train_path = config["data"]["raw_train_path"]
    data_sha256 = compute_file_sha256(raw_train_path)
    logger.info(f"Raw train SHA256: {data_sha256}")

    # Load and clean training data
    df_raw = pd.read_csv(raw_train_path)
    df_clean, clean_meta = clean_data(df_raw, is_train=True)

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    all_features = numeric_features + categorical_features
    target_col = config["features"]["target"]

    X = df_clean[all_features]
    y = df_clean[target_col]

    tracking_uri = config["training"]["tracking_uri"]
    experiment_name = config["training"]["experiment_name"]
    registered_model_name = config["training"]["registered_model_name"]
    model_export_path = Path(config["training"]["model_export_path"])
    model_export_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    candidates = get_candidate_models(config["training"]["random_state"])
    cv = KFold(n_splits=config["training"]["cv_folds"], shuffle=True, random_state=config["training"]["random_state"])

    best_cv_mae = float("inf")
    best_run_id = None
    best_pipeline = None
    best_model_name = None

    for model_name, regressor, hyperparams in candidates:
        logger.info(f"Evaluating {model_name}...")
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

        # Cross-validation MAE
        scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae_mean = float(scores.mean())
        cv_mae_std = float(scores.std())

        # Fit on full data
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        train_mae = float(mean_absolute_error(y, y_pred))

        logger.info(f"[{model_name}] CV MAE: {cv_mae_mean:.4f} (+/- {cv_mae_std:.4f}), Train MAE: {train_mae:.4f}")

        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            run_id = run.info.run_id
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_name", model_name)
            for p_k, p_v in hyperparams.items():
                mlflow.log_param(p_k, p_v)

            # Log metrics containing 'mae'
            mlflow.log_metric("mae", cv_mae_mean)
            mlflow.log_metric("cv_mae", cv_mae_mean)
            mlflow.log_metric("cv_mae_std", cv_mae_std)
            mlflow.log_metric("train_mae", train_mae)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            if cv_mae_mean < best_cv_mae:
                best_cv_mae = cv_mae_mean
                best_run_id = run_id
                best_pipeline = pipeline
                best_model_name = model_name

    logger.info(f"Best model: {best_model_name} (Run ID: {best_run_id}, CV MAE: {best_cv_mae:.4f})")

    # Register best model as v1
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri, registered_model_name)
    version = str(reg_model.version)

    # Set alias 'champion'
    client = MlflowClient()
    client.set_registered_model_alias(registered_model_name, "champion", version)
    logger.info(f"Registered model '{registered_model_name}' version {version} with alias '@champion'")

    # Export best model artifact to models/model.joblib
    joblib.dump(best_pipeline, model_export_path)
    logger.info(f"Exported champion model to {model_export_path}")

    # Update findings.yaml in both locations
    for path in [Path("findings.yaml"), Path("reports/findings.yaml")]:
        if path.exists():
            update_findings_tracking(
                findings_path=path,
                experiment_name=experiment_name,
                best_run_id=best_run_id,
                registered_model_name=registered_model_name,
                champion_version=int(version),
            )
            logger.info(f"Updated tracking section in {path}")

    return {
        "best_model_name": best_model_name,
        "best_run_id": best_run_id,
        "best_cv_mae": best_cv_mae,
        "registered_model_name": registered_model_name,
        "version": version,
    }


if __name__ == "__main__":
    train_and_track()
