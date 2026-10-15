
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project paths and observation cutoff
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "online_retail_cleaned.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

# Use historical data only up to this cutoff.
CUTOFF_DATE = pd.Timestamp("2011-09-10 23:59:59")

# Recent activity window used later to define the
# population eligible for churn prediction.
RECENT_WINDOW_START = CUTOFF_DATE - pd.Timedelta(days=90)


def main():
    # --------------------------------------------------
    # 2. Load the cleaned transaction dataset
    # --------------------------------------------------

    print("Reading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "CustomerID",
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    # Remove unusable records and ensure only valid,
    # completed purchases are included.
    df = df.dropna(
        subset=[
            "CustomerID",
            "InvoiceNo",
            "InvoiceDate",
            "Quantity",
            "UnitPrice",
        ]
    ).copy()

    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["InvoiceNo"].astype(str).str.startswith("C"))
    ].copy()

    # Only use transactions available by the cutoff.
    df = df[df["InvoiceDate"] <= CUTOFF_DATE].copy()

    if df.empty:
        raise ValueError("No valid transactions found.")

    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    print(f"Transactions through cutoff: {len(df):,}")

    # --------------------------------------------------
    # 3. Calculate transaction-level customer features
    # --------------------------------------------------

    print("Calculating customer features...")

    customer_features = (
        df.groupby("CustomerID")
        .agg(
            total_spending=("Amount", "sum"),
            total_transactions=("InvoiceNo", "nunique"),
            total_items=("Quantity", "sum"),
            last_purchase_date=("InvoiceDate", "max"),
            first_purchase_date=("InvoiceDate", "min"),
            active_days=(
                "InvoiceDate",
                lambda dates: dates.dt.date.nunique(),
            ),
            unique_products=("StockCode", "nunique"),
        )
        .reset_index()
        .rename(columns={"CustomerID": "customer_id"})
    )

    # --------------------------------------------------
    # 4. Calculate average order value
    # --------------------------------------------------

    invoice_totals = (
        df.groupby(["CustomerID", "InvoiceNo"])["Amount"]
        .sum()
        .reset_index(name="order_value")
    )

    average_orders = (
        invoice_totals.groupby("CustomerID")["order_value"]
        .mean()
        .rename("average_order_value")
    )

    customer_features = customer_features.merge(
        average_orders,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------
    # 5. Calculate recency and purchase frequency
    # --------------------------------------------------

    customer_features["days_since_last_purchase"] = (
        CUTOFF_DATE.normalize()
        - customer_features["last_purchase_date"].dt.normalize()
    ).dt.days

    # Dataset observation starts on December 1, 2010.
    observation_start = pd.Timestamp("2010-12-01")

    observation_days = (
        CUTOFF_DATE.normalize() - observation_start
    ).days + 1

    customer_features["purchase_frequency_per_30_days"] = (
        customer_features["total_transactions"]
        / observation_days
        * 30
    )

    # --------------------------------------------------
    # 6. Mark customers active in the recent 90-day window
    # --------------------------------------------------

    recent_customers = set(
        df.loc[
            df["InvoiceDate"] > RECENT_WINDOW_START,
            "CustomerID",
        ].unique()
    )

    customer_features["active_in_last_90_days"] = (
        customer_features["customer_id"]
        .isin(recent_customers)
        .astype(int)
    )

    # --------------------------------------------------
    # 7. Clean, arrange and save the output
    # --------------------------------------------------

    customer_features["total_spending"] = (
        customer_features["total_spending"].round(2)
    )

    customer_features["average_order_value"] = (
        customer_features["average_order_value"].round(2)
    )

    customer_features["purchase_frequency_per_30_days"] = (
        customer_features[
            "purchase_frequency_per_30_days"
        ].round(4)
    )

    customer_features = customer_features.sort_values(
        "customer_id"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    customer_features.to_csv(
        OUTPUT_FILE, index=False
    )

    # --------------------------------------------------
    # 8. Display summary
    # --------------------------------------------------

    eligible_count = int(
        customer_features["active_in_last_90_days"].sum()
    )

    print("\nFEATURE ENGINEERING COMPLETED")
    print("-" * 40)
    print(
        f"Customers with historical features: "
        f"{len(customer_features):,}"
    )
    print(
        f"Customers active in recent 90-day window: "
        f"{eligible_count:,}"
    )
    print(
        f"Customers not active in recent window: "
        f"{len(customer_features) - eligible_count:,}"
    )
    print(f"Observation cutoff: {CUTOFF_DATE.date()}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample customer features:")
    print(customer_features.head().to_string(index=False))


if __name__ == "__main__":
    main()
