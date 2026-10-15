import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

# --------------------------------------------------
# 1. Load project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from your .env file.")

# Change this path if your CSV is stored elsewhere.
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "online_retail_cleaned.csv"

BATCH_SIZE = 5000

# --------------------------------------------------
# 2. Connect to MySQL
# --------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

# --------------------------------------------------
# 3. Validate the CSV and database tables
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned CSV not found: {CSV_PATH}\n"
        "Update CSV_PATH to match your project folder."
    )

inspector = inspect(engine)

required_tables = {
    "customers",
    "transactions",
}

existing_tables = set(inspector.get_table_names())

if not required_tables.issubset(existing_tables):
    raise RuntimeError(
        "The customers or transactions table is missing. "
        "Create the tables before importing data."
    )

# Verify required columns exist.
customer_columns = {
    col["name"] for col in inspector.get_columns("customers")
}
transaction_columns = {
    col["name"] for col in inspector.get_columns("transactions")
}

if not {"customer_id", "join_date"}.issubset(customer_columns):
    raise RuntimeError(
        "The customers table must contain customer_id and join_date."
    )

if not {
    "customer_id",
    "transaction_date",
    "amount",
    "product_category",
}.issubset(transaction_columns):
    raise RuntimeError(
        "The transactions table does not match the expected project fields."
    )

# --------------------------------------------------
# 4. Read and validate the cleaned dataset
# --------------------------------------------------

print("Reading cleaned dataset...")

df = pd.read_csv(CSV_PATH)

required_csv_columns = {
    "CustomerID",
    "InvoiceDate",
    "Quantity",
    "UnitPrice",
    "Description",
    "Amount",
}

missing_columns = required_csv_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {sorted(missing_columns)}"
    )

df["CustomerID"] = pd.to_numeric(
    df["CustomerID"], errors="coerce"
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

df["Amount"] = pd.to_numeric(
    df["Amount"], errors="coerce"
)

df = df.dropna(
    subset=[
        "CustomerID",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "Amount",
    ]
)

# Customer IDs in this dataset are integer identifiers.
df = df[
    (df["CustomerID"] > 0)
    & (df["CustomerID"] % 1 == 0)
].copy()

df["CustomerID"] = df["CustomerID"].astype("int64")

if df.empty:
    raise ValueError("No valid records remain after validation.")

print(f"Valid transaction rows: {len(df):,}")

# --------------------------------------------------
# 5. Prepare one record per customer
# --------------------------------------------------

customers_df = (
    df.groupby("CustomerID", as_index=False)
    .agg(join_date=("InvoiceDate", "min"))
    .rename(columns={"CustomerID": "customer_id"})
)

customers_df["join_date"] = (
    customers_df["join_date"].dt.date
)

print(f"Unique customers: {len(customers_df):,}")

# --------------------------------------------------
# 6. Prepare transaction records
# --------------------------------------------------

transactions_df = pd.DataFrame({
    "customer_id": df["CustomerID"],
    "transaction_date": df["InvoiceDate"],
    "amount": df["Amount"].round(2),
    "product_category": df["Description"].fillna(
        "Unknown product"
    ).astype(str).str.slice(0, 255),

    # Required by your MySQL table
    "quantity": df["Quantity"].astype(int),
    "unit_price": df["UnitPrice"].round(2),
})

# --------------------------------------------------
# 7. Import in batches
# --------------------------------------------------

customer_insert = text("""
    INSERT INTO customers (
        customer_id,
        join_date
    )
    VALUES (
        :customer_id,
        :join_date
    )
""")

transaction_insert = text("""
    INSERT INTO transactions (
        customer_id,
        transaction_date,
        amount,
        product_category,
        quantity,
        unit_price
    )
    VALUES (
        :customer_id,
        :transaction_date,
        :amount,
        :product_category,
        :quantity,
        :unit_price
    )
""")

with engine.begin() as connection:

    # Avoid duplicate imports if the script is run again.
    customer_count = connection.execute(
        text("SELECT COUNT(*) FROM customers")
    ).scalar_one()

    transaction_count = connection.execute(
        text("SELECT COUNT(*) FROM transactions")
    ).scalar_one()

    if customer_count or transaction_count:
        raise RuntimeError(
            "Import stopped: customers or transactions already "
            "contain records. Verify the existing data before retrying."
        )

    # Insert customers first to satisfy the foreign key.
    print("Importing customers...")

    for start in range(0, len(customers_df), BATCH_SIZE):
        batch = customers_df.iloc[start:start + BATCH_SIZE]

        records = batch.to_dict(orient="records")
        connection.execute(customer_insert, records)

        print(
            f"Customers imported: "
            f"{min(start + len(batch), len(customers_df)):,}"
            f"/{len(customers_df):,}"
        )

    # Then insert transactions.
    print("Importing transactions...")

    for start in range(0, len(transactions_df), BATCH_SIZE):
        batch = transactions_df.iloc[start:start + BATCH_SIZE]

        records = batch.to_dict(orient="records")
        connection.execute(transaction_insert, records)

        print(
            f"Transactions imported: "
            f"{min(start + len(batch), len(transactions_df)):,}"
            f"/{len(transactions_df):,}"
        )

print("\nImport completed successfully.")

engine.dispose()