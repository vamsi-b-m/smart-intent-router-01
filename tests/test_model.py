import pandas as pd

from src.train.train import build_model
from src.evaluate.evaluate import evaluate


def test_build_model():
    model = build_model()

    assert model is not None
    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_can_train_and_predict():
    train_df = pd.DataFrame(
        {
            "text": [
                "I want to transfer money",
                "send money to another account",
                "what is my account balance",
                "show me my balance",
            ],
            "intent": [
                "transfer",
                "transfer",
                "balance",
                "balance",
            ],
        }
    )

    model = build_model()
    model.fit(train_df["text"], train_df["intent"])

    predictions = model.predict(train_df["text"])

    assert len(predictions) == len(train_df)
    assert set(predictions).issubset({"transfer", "balance"})


def test_evaluate_returns_valid_metrics():
    train_df = pd.DataFrame(
        {
            "text": [
                "I want to transfer money",
                "send money to another account",
                "what is my account balance",
                "show me my balance",
            ],
            "intent": [
                "transfer",
                "transfer",
                "balance",
                "balance",
            ],
        }
    )

    model = build_model()
    model.fit(train_df["text"], train_df["intent"])

    accuracy, macro_f1, report = evaluate(model, train_df)

    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert report
    