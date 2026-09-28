import os

from pathlib import Path

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")
MODEL_PATH = Path("models/intent_classifier.joblib")
RUN_ID_PATH = Path("models/mlflow_run_id.txt")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001",
)

def load_data():
    """Load training and validation datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    return train_df, val_df


def build_model():
    """Build the text classification pipeline."""

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )

    return model


def main():
    print("Loading data...")
    train_df, val_df = load_data()

    X_train = train_df["text"]
    y_train = train_df["intent"]
    X_val = val_df["text"]
    y_val = val_df["intent"]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {y_train.nunique()}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("smart-intent-router")

    with mlflow.start_run():
        print("\nBuilding model...")
        model = build_model()

        mlflow.log_params(
            {
                "ngram_range": "(1, 2)",
                "max_features": 50000,
                "max_iter": 1000,
            }
        )

        print("Training model...")
        model.fit(X_train, y_train)
        print("Training completed.")

        print("\nEvaluating model...")
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
        )

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro F1:  {macro_f1:.4f}")

        mlflow.log_metrics(
            {
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
            }
        )

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        joblib.dump(model, MODEL_PATH)

        print(f"\nModel saved to: {MODEL_PATH}")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="intent-classifier",
        )

        print("Model logged to MLflow.")

        run_id = mlflow.active_run().info.run_id

        RUN_ID_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        RUN_ID_PATH.write_text(run_id)
        print(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    main()
