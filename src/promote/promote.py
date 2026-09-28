import os

import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

MODEL_NAME = "smart-intent-router"
SOURCE_ALIAS = "candidate"
TARGET_ALIAS = "champion"


def main():
    print("Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    print(
        f"Finding model assigned to alias "
        f"'{SOURCE_ALIAS}'..."
    )

    candidate = client.get_model_version_by_alias(
        MODEL_NAME,
        SOURCE_ALIAS,
    )

    print(f"Candidate version: {candidate.version}")

    client.set_registered_model_alias(
        MODEL_NAME,
        TARGET_ALIAS,
        candidate.version,
    )

    print(
        f"Model version {candidate.version} promoted "
        f"from '{SOURCE_ALIAS}' to '{TARGET_ALIAS}'."
    )


if __name__ == "__main__":
    main()
