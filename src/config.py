import os
from pathlib import Path


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "models/intent_classifier.joblib",
    )
)

APP_NAME = os.getenv(
    "APP_NAME",
    "smart-intent-router",
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "intent-classifier",
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1.0.0",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "local",
)
