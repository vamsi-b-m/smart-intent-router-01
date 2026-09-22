from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


MODEL_PATH = Path("models/intent_classifier.joblib")
TEST_PATH = Path("data/processed/test.csv")
METRICS_PATH = Path("models/metrics.json")

MIN_MACRO_F1 = 0.85


def load_model():
    return joblib.load(MODEL_PATH)


def load_test_data():
    return pd.read_csv(TEST_PATH)


def evaluate(model, test_df):
    X_test = test_df["text"]
    y_test = test_df["intent"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    report = classification_report(
        y_test,
        predictions,
    )

    return accuracy, macro_f1, report


def main():
    print("Loading model...")
    model = load_model()

    print("Loading test data...")
    test_df = load_test_data()

    print(f"Test samples: {len(test_df)}")

    print("\nEvaluating model...")

    accuracy, macro_f1, report = evaluate(
        model,
        test_df,
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "test_samples": len(test_df),
        "minimum_macro_f1": MIN_MACRO_F1,
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {METRICS_PATH}")

    # Quality gate
    print("\nRunning model quality gate...")

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


if __name__ == "__main__":
    main()