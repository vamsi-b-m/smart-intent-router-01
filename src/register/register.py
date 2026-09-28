import os

from pathlib import Path
import json

import mlflow
from mlflow import MlflowClient


import os

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

MODEL_NAME = "smart-intent-router"

RUN_ID_PATH = Path("models/mlflow_run_id.txt")
METRICS_PATH = Path("models/metrics.json")
REGISTERED_VERSION_PATH = Path("models/registered_model_version.txt")

MIN_MACRO_F1 = 0.85

CANDIDATE_ALIAS = "candidate"

def check_quality_gate():
    """Check whether the model passed the required quality threshold."""

    if not METRICS_PATH.exists():
        raise RuntimeError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    macro_f1 = metrics["macro_f1"]

    print(f"Test Macro F1:      {macro_f1:.4f}")
    print(f"Minimum Macro F1:   {MIN_MACRO_F1:.4f}")

    if macro_f1 < MIN_MACRO_F1:
        print(
            f"QUALITY GATE FAILED: "
            f"Macro F1 {macro_f1:.4f} "
            f"is below minimum {MIN_MACRO_F1:.4f}"
        )

        raise SystemExit(1)

    print(
        f"QUALITY GATE PASSED: "
        f"Macro F1 {macro_f1:.4f} "
        f">= minimum {MIN_MACRO_F1:.4f}"
    )


def get_run_id():
    """Read the MLflow run ID produced by the training stage."""

    if not RUN_ID_PATH.exists():
        raise RuntimeError(
            f"MLflow run ID file not found: {RUN_ID_PATH}"
        )

    run_id = RUN_ID_PATH.read_text().strip()

    if not run_id:
        raise RuntimeError(
            "MLflow run ID file is empty."
        )

    return run_id


def register_model(run_id):
    """Register the logged MLflow model in the Model Registry."""

    client = MlflowClient()

    print(f"Finding logged model for run: {run_id}")

    experiment = client.get_experiment_by_name("smart-intent-router")

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment 'smart-intent-router' was not found."
        )

    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
    )

    if not logged_models:
        raise RuntimeError(
            f"No logged model found for MLflow run: {run_id}"
        )

    if len(logged_models) > 1:
        raise RuntimeError(
            f"Expected exactly one logged model for run {run_id}, "
            f"but found {len(logged_models)}."
        )

    logged_model = logged_models[0]

    model_id = logged_model.model_id

    print(f"Logged model ID: {model_id}")

    model_uri = f"models:/{model_id}"

    print(f"Registering model from: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = registered_model.version

    client.set_registered_model_alias(
        MODEL_NAME,
        CANDIDATE_ALIAS,
        version
    )

    print(
        f"Model version {version} assigned to alias "
        f"'{CANDIDATE_ALIAS}'"
    )
    
    REGISTERED_VERSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGISTERED_VERSION_PATH.write_text(str(version))

    print("\nModel registered successfully:")
    print(f"Model name:    {registered_model.name}")
    print(f"Model version: {version}")

def main():
    print("Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("\nChecking model quality gate...")
    check_quality_gate()

    print("\nReading MLflow run ID...")
    run_id = get_run_id()

    print(f"MLflow Run ID: {run_id}")

    register_model(run_id)


if __name__ == "__main__":
    main()
