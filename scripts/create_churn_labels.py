
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define paths and dates
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "online_retail_cleaned.csv"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_dataset.csv"
)

CUTOFF_DATE = pd.Timestamp("2011-09-10 23:59:59")
FOLLOWUP_END = pd.Timestamp("2011-12-09 23:59:59")


def main():
    # --------------------------------------------------
    # 2. Load the transaction data and customer features
    # --------------------------------------------------

    print("Loading transaction data and customer features...")

    transactions = pd.read_csv(RAW_FILE)
    features = pd.read_csv(FEATURES_FILE)

    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )

    transactions["CustomerID"] = pd.to_numeric(
        transactions["CustomerID"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=["CustomerID", "InvoiceDate"]
    ).copy()

    transactions["CustomerID"] = (
        transactions["CustomerID"].astype("int64")
    )

    features["customer_id"] = pd.to_numeric(
        features["customer_id"], errors="coerce"
    )

    features = features.dropna(
        subset=["customer_id"]
    ).copy()

    features["customer_id"] = (
        features["customer_id"].astype("int64")
    )

    # --------------------------------------------------
    # 3. Select eligible customers
    # --------------------------------------------------

    eligible = features[
        features["active_in_last_90_days"] == 1
    ].copy()

    print(f"Eligible customers: {len(eligible):,}")

    # --------------------------------------------------
    # 4. Find customers with a future purchase
    # --------------------------------------------------

    print("Checking purchases during the follow-up period...")

    followup = transactions[
        (transactions["InvoiceDate"] > CUTOFF_DATE)
        & (transactions["InvoiceDate"] <= FOLLOWUP_END)
    ].copy()

    # A customer is retained if they have at least one
    # valid purchase in the follow-up period.
    future_customer_ids = set(
        followup["CustomerID"].unique()
    )

    eligible["churn_label"] = (
        ~eligible["customer_id"].isin(future_customer_ids)
    ).astype(int)

    # --------------------------------------------------
    # 5. Save the model-ready customer dataset
    # --------------------------------------------------

    eligible.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------
    # 6. Display label statistics
    # --------------------------------------------------

    churned = int(
        (eligible["churn_label"] == 1).sum()
    )

    retained = int(
        (eligible["churn_label"] == 0).sum()
    )

    total = len(eligible)

    churn_rate = (
        churned / total * 100 if total else 0
    )

    print("\nCHURN LABEL CREATION COMPLETED")
    print("-" * 40)
    print(f"Eligible customers: {total:,}")
    print(f"Churned customers (label 1): {churned:,}")
    print(f"Retained customers (label 0): {retained:,}")
    print(f"Observed churn rate: {churn_rate:.2f}%")
    print(f"Follow-up starts after: {CUTOFF_DATE}")
    print(f"Follow-up ends: {FOLLOWUP_END}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample labeled customers:")
    print(
        eligible[
            [
                "customer_id",
                "total_spending",
                "total_transactions",
                "days_since_last_purchase",
                "active_in_last_90_days",
                "churn_label",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nNote: The source dataset's final recorded "
        "transaction is on December 9, 2011. Verify "
        "that the final-day coverage is sufficient "
        "before interpreting labels as a full 90-day "
        "follow-up."
    )


if __name__ == "__main__":
    main()
