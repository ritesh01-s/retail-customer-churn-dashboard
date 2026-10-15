
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Project paths and database configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from your .env file.")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def main():
    print("Loading customer transaction data...")

    query = text("""
        SELECT
            customer_id,
            invoice_no,
            transaction_date,
            amount
        FROM transactions
        WHERE customer_id IS NOT NULL
          AND transaction_date IS NOT NULL
          AND amount IS NOT NULL
    """)

    with engine.connect() as conn:
        transactions = pd.read_sql(query, conn)

    if transactions.empty:
        raise ValueError("No valid customer transactions were found.")

    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], errors="coerce"
    )
    transactions["amount"] = pd.to_numeric(
        transactions["amount"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=["customer_id", "transaction_date", "amount"]
    )

    # Use the latest date in the dataset as the reference date.
    reference_date = transactions["transaction_date"].max().normalize() + pd.Timedelta(days=1)

    # Calculate RFM features for each customer.
    rfm = transactions.groupby("customer_id").agg(
        last_purchase=("transaction_date", "max"),
        monetary=("amount", "sum"),
        frequency=("invoice_no", "nunique"),
    ).reset_index()

    # If an invoice number is missing, use the transaction count as a fallback.
    missing_invoice = transactions["invoice_no"].isna()

    if missing_invoice.any():
        fallback_counts = (
            transactions.loc[missing_invoice]
            .groupby("customer_id")
            .size()
        )

        fallback_counts = fallback_counts.reindex(
            rfm["customer_id"], fill_value=0
        ).to_numpy()

        rfm["frequency"] = (
            rfm["frequency"].fillna(0).to_numpy() + fallback_counts
        )

    rfm["recency"] = (
        reference_date - rfm["last_purchase"].dt.normalize()
    ).dt.days

    rfm["frequency"] = rfm["frequency"].fillna(0)
    rfm["monetary"] = rfm["monetary"].fillna(0)

    # K-Means requires finite numeric features.
    features = rfm[["recency", "frequency", "monetary"]].copy()

    if len(features) < 4:
        raise ValueError(
            "At least four customers are required for four clusters."
        )

    # Log transformation reduces the effect of highly skewed spending.
    features["frequency"] = features["frequency"].clip(lower=0)
    features["monetary"] = features["monetary"].clip(lower=0)

    features["frequency"] = __import__("numpy").log1p(
        features["frequency"]
    )
    features["monetary"] = __import__("numpy").log1p(
        features["monetary"]
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Start with four segments; later we can compare other cluster counts.
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    rfm["cluster"] = model.fit_predict(scaled_features)

    # Silhouette score is a diagnostic, not proof that four clusters are best.
    if rfm["cluster"].nunique() > 1 and len(rfm) > rfm["cluster"].nunique():
        score = silhouette_score(
            scaled_features,
            rfm["cluster"],
            sample_size=min(2000, len(rfm)),
            random_state=42,
        )
    else:
        score = None

    # Save customer assignments.
    output_path = DATA_DIR / "customer_segments.csv"
    rfm.to_csv(output_path, index=False)

    # Save the fitted model and scaler for reproducibility.
    model_path = MODEL_DIR / "customer_segmentation_model.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": ["recency", "frequency", "monetary"],
            "reference_date": str(reference_date.date()),
        },
        model_path,
    )

    # Summarize the actual characteristics of each cluster.
    profile = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"),
        average_recency=("recency", "mean"),
        average_frequency=("frequency", "mean"),
        average_spending=("monetary", "mean"),
        total_spending=("monetary", "sum"),
    ).round(2)

    report_path = DATA_DIR / "customer_segment_summary.csv"
    profile.to_csv(report_path)

    print("\nCustomer segmentation completed.")
    print(f"Customers segmented: {len(rfm):,}")
    print(f"Reference date: {reference_date.date()}")
    print(f"Silhouette score: {score if score is not None else 'Unavailable'}")
    print("\nCluster summary:")
    print(profile)
    print(f"\nAssignments saved to: {output_path}")
    print(f"Model saved to: {model_path}")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        engine.dispose()
