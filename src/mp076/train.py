"""Compare v1 settings and register the best model in the local MLflow registry."""

import argparse
import hashlib

import pandas as pd

from .data import DATE, FEATURES, TARGET, clean_training, input_csv, load_config
from .model import build_model, evaluate_candidates


def configure_tracking(tracking):
    import mlflow

    mlflow.set_tracking_uri(tracking["uri"])
    mlflow.set_registry_uri(tracking["uri"])
    return mlflow.MlflowClient()


def register_v1(config, scores, fit, check, frame, data_sha256):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    tracking = config["tracking"]
    training = config["training"]
    client = configure_tracking(tracking)
    model_name = tracking["registered_model_name"]
    versions = client.search_model_versions(f"name = '{model_name}'")
    if versions:
        v1 = client.get_model_version(model_name, "1")
        if client.get_run(v1.run_id).data.params.get("data_sha256") != data_sha256:
            raise RuntimeError("Version 1 already exists for a different training CSV.")
    else:
        experiment = client.get_experiment_by_name(tracking["experiment_name"])
        experiment_id = (
            experiment.experiment_id if experiment else client.create_experiment(
                tracking["experiment_name"], artifact_location=tracking["artifact_location"]
            )
        )
        best = min(scores, key=scores.get)
        for name, score in scores.items():
            settings = training["candidates"][name]
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"v1-{name}"):
                mlflow.log_params({
                    **settings,
                    "estimator": name,
                    "data_sha256": data_sha256,
                    "validation_start_date": check[DATE].min().date().isoformat(),
                    "fit_rows": len(fit),
                    "validation_rows": len(check),
                    "random_state": training["random_state"],
                })
                mlflow.log_metric("mae_validation", score)
                if name == best:
                    model = build_model(name, settings, training["random_state"])
                    model.fit(frame[FEATURES], frame[TARGET])
                    example = frame[FEATURES].dropna().head(5)
                    winner = mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        serialization_format="cloudpickle",
                        signature=infer_signature(example, model.predict(example)),
                        input_example=example,
                    )
        v1 = mlflow.register_model(winner.model_uri, model_name)

    if "champion" not in client.get_registered_model(model_name).aliases:
        client.set_registered_model_alias(model_name, "champion", v1.version)
    return v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Compare candidates without writing to MLflow.")
    args = parser.parse_args()
    config = load_config()
    source = input_csv(config["paths"]["train"])
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = clean_training(pd.read_csv(source))
    scores, fit, check = evaluate_candidates(frame, config["training"])
    print(f"Training data SHA-256 {fingerprint}")
    print(f"{len(fit)} fit rows and {len(check)} validation rows")
    for name, score in scores.items():
        print(f"{name}: validation MAE {score:.3f} thousand rupees")
    if not args.validate_only:
        version = register_v1(config, scores, fit, check, frame, fingerprint)
        print(f"Registered v1 run {version.run_id}, version {version.version}")


if __name__ == "__main__":
    main()
