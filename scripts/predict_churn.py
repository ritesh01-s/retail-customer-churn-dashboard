
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam
import os


# --------------------------------------------------
# 1. Project paths and database connection
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_dataset.csv"
)

MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "churn_predictions.csv"
)

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL was not found in the project .env file."
    )


def get_risk_level(probability):
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    return "Low"


def main():
    # --------------------------------------------------
    # 2. Load model and customer features
    # --------------------------------------------------

    print("Loading trained model...")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    print(f"Model: {bundle['model_name']}")

    df = pd.read_csv(DATA_FILE)

    for column in [
        "first_purchase_date",
        "last_purchase_date",
    ]:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

    cutoff_date = pd.Timestamp(bundle["cutoff_date"])

    df["customer_tenure_days"] = (
        cutoff_date
        - df["first_purchase_date"].dt.normalize()
    ).dt.days

    # Use exactly the same feature columns as training.
    X = df[feature_columns].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)

    # --------------------------------------------------
    # 3. Predict churn probabilities
    # --------------------------------------------------

    print("Generating churn probabilities...")

    probabilities = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({
        "customer_id": df["customer_id"].astype(int),
        "churn_probability": probabilities.round(4),
    })

    predictions["risk_level"] = (
        predictions["churn_probability"]
        .apply(get_risk_level)
    )

    # Save a CSV copy for inspection and backup.
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    predictions.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------
    # 4. Save predictions to the existing MySQL table
    # --------------------------------------------------

    print("Connecting to MySQL...")

    engine = create_engine(DATABASE_URL)

    customer_ids = predictions["customer_id"].tolist()

    delete_statement = text("""
        DELETE FROM churn_predictions
        WHERE customer_id IN :customer_ids
    """).bindparams(
        bindparam("customer_ids", expanding=True)
    )

    insert_statement = text("""
        INSERT INTO churn_predictions (
            customer_id,
            churn_probability,
            risk_level,
            predicted_at
        )
        VALUES (
            :customer_id,
            :churn_probability,
            :risk_level,
            CURRENT_TIMESTAMP
        )
    """)

    records = predictions.to_dict(orient="records")

    # Replaces previous predictions for this cohort only.
    # Both operations commit together or roll back together.
    with engine.begin() as connection:
        connection.execute(
            delete_statement,
            {"customer_ids": customer_ids},
        )

        connection.execute(insert_statement, records)

    engine.dispose()

    # --------------------------------------------------
    # 5. Display prediction summary
    # --------------------------------------------------

    risk_counts = (
        predictions["risk_level"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )

    print("\nPREDICTIONS COMPLETED")
    print("-" * 40)
    print(f"Customers scored: {len(predictions):,}")
    print(f"High risk: {risk_counts['High']:,}")
    print(f"Medium risk: {risk_counts['Medium']:,}")
    print(f"Low risk: {risk_counts['Low']:,}")
    print(
        "Average churn probability: "
        f"{predictions['churn_probability'].mean():.2%}"
    )
    print(f"CSV saved to: {OUTPUT_FILE}")
    print("MySQL table updated: churn_predictions")

    print("\nSample predictions:")
    print(
        predictions.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
