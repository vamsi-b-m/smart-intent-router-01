from pathlib import Path
import json
import pandas as pd


RAW_DATA_PATH = Path("data/raw/oos-eval/data/data_full.json")
PROCESSED_DIR = Path("data/processed")


def load_data():
    """Load the complete CLINC150 dataset."""
    with open(RAW_DATA_PATH, "r") as f:
        return json.load(f)


def convert_to_dataframe(records):
    """Convert CLINC150 records into a DataFrame."""
    return pd.DataFrame(
        records,
        columns=["text", "intent"],
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print("Available splits:")
    print(data.keys())

    train_df = convert_to_dataframe(data["train"])
    val_df = convert_to_dataframe(data["val"])
    test_df = convert_to_dataframe(data["test"])

    print("\nDataset sizes:")
    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nColumns:")
    print(train_df.columns.tolist())

    print("\nSample training data:")
    print(train_df.head())

    print("\nNumber of unique intents:")
    print(train_df["intent"].nunique())

    print("\nExample intents:")
    print(train_df["intent"].unique()[:10])

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
    )

    print("\nProcessed datasets created successfully.")


if __name__ == "__main__":
    main()
