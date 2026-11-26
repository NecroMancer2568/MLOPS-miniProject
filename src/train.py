from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(ROOT / "data/processed/train_clean.csv")

target = config["training"]["target"]
excluded = ["record_id", "date", target]
X = df.drop(columns=excluded)
y = df[target]

categorical_features = ["atm_type", "area"]
numeric_features = [column for column in X.columns if column not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"],
)

mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Three runs with different model settings, as required for comparison.
model_configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": None},
]

for model_config in model_configs:
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    **model_config,
                    random_state=config["model"]["random_state"],
                ),
            ),
        ]
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_params(model_config)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(
            f"Run: {model_config} | "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}"
        )