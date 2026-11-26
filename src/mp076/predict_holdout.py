"""Export predictions and the serving pipeline from the current champion."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import FEATURES, ID, input_csv, load_config
from .train import configure_tracking


def predict_holdout(model, holdout):
    if holdout.empty or holdout[ID].isna().any() or not holdout[ID].is_unique:
        raise ValueError("Holdout must contain a unique record_id for every row.")
    predictions = model.predict(holdout[FEATURES])
    if predictions.shape != (len(holdout),) or not np.isfinite(predictions).all():
        raise ValueError("Expected one finite prediction per holdout row.")
    return pd.DataFrame({ID: holdout[ID], "prediction": predictions})


def update_findings(path, tracking, v1, champion):
    template = path if path.exists() else Path("findings.yaml")
    findings = yaml.safe_load(template.read_text(encoding="utf-8"))
    findings["tracking"].update(
        experiment_name=tracking["experiment_name"],
        best_run_id=v1.run_id,
        registered_model_name=tracking["registered_model_name"],
        champion_version=int(champion.version),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")


def main():
    import mlflow.sklearn

    config = load_config()
    paths = config["paths"]
    tracking = config["tracking"]
    client = configure_tracking(tracking)
    name = tracking["registered_model_name"]
    v1 = client.get_model_version(name, "1")
    champion = client.get_model_version_by_alias(name, "champion")
    # Resolve the alias once so all three outputs use the very same version.
    model = mlflow.sklearn.load_model(f"models:/{name}/{champion.version}")
    holdout = pd.read_csv(input_csv(paths["holdout"]), dtype={ID: str})
    result = predict_holdout(model, holdout)

    predictions_path = Path(paths["predictions"])
    service_path = Path(paths["service_model"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    service_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(predictions_path, index=False)
    joblib.dump(model, service_path)
    update_findings(Path(paths["findings"]), tracking, v1, champion)
    print(f"Champion version {champion.version}: wrote {len(result)} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
