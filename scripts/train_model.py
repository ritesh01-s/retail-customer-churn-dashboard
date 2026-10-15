
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_FILE = MODEL_DIR / "churn_model.joblib"
REPORT_FILE = REPORT_DIR / "model_evaluation.txt"

CUTOFF_DATE = pd.Timestamp("2011-09-10")


def main():
    # --------------------------------------------------
    # 2. Load the labeled customer dataset
    # --------------------------------------------------

    print("Loading customer churn dataset...")

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("The customer dataset is empty.")

    if "churn_label" not in df.columns:
        raise ValueError("Missing churn_label column.")

    if df["churn_label"].nunique() != 2:
        raise ValueError(
            "Training requires both churn and retained labels."
        )

    print(f"Dataset rows: {len(df):,}")

    # --------------------------------------------------
    # 3. Prepare machine-learning features
    # --------------------------------------------------

    print("Preparing features...")

    # Convert purchase dates to datetime.
    for column in [
        "first_purchase_date",
        "last_purchase_date",
    ]:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

    # Customer tenure as of the prediction cutoff.
    df["customer_tenure_days"] = (
        CUTOFF_DATE
        - df["first_purchase_date"].dt.normalize()
    ).dt.days

    feature_columns = [
        "total_spending",
        "total_transactions",
        "total_items",
        "active_days",
        "unique_products",
        "average_order_value",
        "days_since_last_purchase",
        "purchase_frequency_per_30_days",
        "customer_tenure_days",
    ]

    # Do not use customer_id as a predictive feature.
    # Do not use raw dates or churn labels as features.
    # active_in_last_90_days is always 1 for this cohort.
    X = df[feature_columns].copy()
    y = df["churn_label"].astype(int)

    X = X.replace([float("inf"), float("-inf")], pd.NA)

    # Handle missing feature values safely.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)

    if X.isnull().any().any():
        raise ValueError("Missing values remain in features.")

    print(f"Number of features: {len(feature_columns)}")
    print(f"Churn cases: {(y == 1).sum():,}")
    print(f"Retained cases: {(y == 0).sum():,}")

    # --------------------------------------------------
    # 4. Split training and testing data
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    # --------------------------------------------------
    # 5. Define the two candidate models
    # --------------------------------------------------

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # --------------------------------------------------
    # 6. Train and evaluate each model
    # --------------------------------------------------

    results = {}
    report_lines = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Retained", "Churned"],
            zero_division=0,
        )

        results[name] = {
            "model": model,
            "roc_auc": auc,
            "accuracy": accuracy,
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        print("Confusion matrix (rows=true, columns=predicted):")
        print(matrix)
        print("\nClassification report:")
        print(report)

        report_lines.extend([
            f"\n{'=' * 55}",
            f"MODEL: {name}",
            f"Accuracy: {accuracy:.4f}",
            f"ROC-AUC: {auc:.4f}",
            "Confusion matrix (rows=true, columns=predicted):",
            str(matrix),
            "Classification report:",
            report,
        ])

    # --------------------------------------------------
    # 7. Select and save the best model
    # --------------------------------------------------

    # ROC-AUC measures how well the model ranks churn
    # cases above retained cases across thresholds.
    best_name = max(
        results,
        key=lambda model_name: results[model_name]["roc_auc"],
    )

    best_model = results[best_name]["model"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_name,
        "cutoff_date": str(CUTOFF_DATE.date()),
        "label_definition": (
            "No valid purchase during the follow-up period"
        ),
    }

    joblib.dump(model_bundle, MODEL_FILE)

    report_lines.extend([
        "\n",
        f"Selected model: {best_name}",
        f"Selected model ROC-AUC: "
        f"{results[best_name]['roc_auc']:.4f}",
        "Selection criterion: highest test ROC-AUC",
        "Note: Labels are provisional due to dataset "
        "follow-up coverage limitations.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nMODEL TRAINING COMPLETED")
    print("-" * 40)
    print(f"Selected model: {best_name}")
    print(
        "Test ROC-AUC: "
        f"{results[best_name]['roc_auc']:.4f}"
    )
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved evaluation report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
