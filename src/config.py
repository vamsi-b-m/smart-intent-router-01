import os

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/intent_classifier.joblib",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "smart-intent-router",
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

APP_NAME = os.getenv(
    "APP_NAME",
    "smart-intent-router",
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "registry",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "local",
)
