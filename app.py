
import os
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing from your .env file."
    )

app = Flask(
    __name__,

)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

if not app.config["SECRET_KEY"]:
    raise RuntimeError(
        "SECRET_KEY is missing. Check your project's .env file."
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))

        return view_function(*args, **kwargs)

    return wrapped_view


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            flash("Please fill in all fields.", "danger")
            return render_template("register.html")

        if len(username) > 100:
            flash("Username must be 100 characters or fewer.", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        with engine.begin() as connection:
            existing_user = connection.execute(
                text("""
                    SELECT user_id
                    FROM users
                    WHERE username = :username
                """),
                {"username": username},
            ).first()

            if existing_user:
                flash("That username is already registered.", "danger")
                return render_template("register.html")

            password_hash = generate_password_hash(password)

            connection.execute(
                text("""
                    INSERT INTO users (username, password_hash, role)
                    VALUES (:username, :password_hash, :role)
                """),
                {
                    "username": username,
                    "password_hash": password_hash,
                    "role": "analyst",
                },
            )

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with engine.connect() as connection:
            user = connection.execute(
                text("""
                    SELECT user_id, username, password_hash, role
                    FROM users
                    WHERE username = :username
                """),
                {"username": username},
            ).mappings().first()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            session.clear()
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["role"] = user["role"] or "analyst"

            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")

@app.route("/profile")
@login_required
def profile():
    with engine.connect() as connection:
        user = connection.execute(
            text("""
                SELECT user_id, username, role
                FROM users
                WHERE user_id = :user_id
            """),
            {"user_id": session["user_id"]},
        ).mappings().first()

    if user is None:
        session.clear()
        flash("Please log in again.", "warning")
        return redirect(url_for("login"))

    return render_template("profile.html", user=user)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/")
@login_required
def dashboard():
    with engine.connect() as connection:

        # Overall business metrics
        summary = connection.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM customers)
                    AS total_customers,
                (SELECT COUNT(*) FROM transactions)
                    AS total_transactions,
                (SELECT COALESCE(SUM(amount), 0)
                    FROM transactions)
                    AS total_revenue,
                (SELECT COUNT(*) FROM churn_predictions)
                    AS customers_scored
        """)).mappings().one()

        # Customer risk distribution
        risk_rows = connection.execute(text("""
            SELECT
                risk_level,
                COUNT(*) AS customer_count
            FROM churn_predictions
            GROUP BY risk_level
        """)).mappings().all()

        risk_counts = {
            "High": 0,
            "Medium": 0,
            "Low": 0,
        }

        for row in risk_rows:
            if row["risk_level"] in risk_counts:
                risk_counts[row["risk_level"]] = (
                    row["customer_count"]
                )

        # Highest-risk customers
        top_customers = connection.execute(text("""
            SELECT
                c.customer_id,
                COALESCE(
                    NULLIF(c.name, ''),
                    CONCAT('Customer #', c.customer_id)
                ) AS customer_name,
                c.city,
                cp.churn_probability,
                cp.risk_level
            FROM churn_predictions cp
            JOIN customers c
                ON c.customer_id = cp.customer_id
            ORDER BY cp.churn_probability DESC
            LIMIT 10
        """)).mappings().all()

        # Monthly sales trend
        monthly_rows = connection.execute(text("""
            SELECT
                YEAR(transaction_date) AS sales_year,
                MONTH(transaction_date) AS sales_month,
                SUM(amount) AS monthly_revenue
            FROM transactions
            GROUP BY
                YEAR(transaction_date),
                MONTH(transaction_date)
            ORDER BY sales_year, sales_month
        """)).mappings().all()

    monthly_labels = [
        f"{row['sales_year']}-{row['sales_month']:02d}"
        for row in monthly_rows
    ]

    monthly_revenue = [
        round(float(row["monthly_revenue"] or 0), 2)
        for row in monthly_rows
    ]

    average_probability = 0

    if summary["customers_scored"]:
        with engine.connect() as connection:
            average_probability = connection.execute(
                text("""
                    SELECT AVG(churn_probability)
                    FROM churn_predictions
                """)
            ).scalar_one() or 0

    return render_template(
        "dashboard.html",
        summary=summary,
        risk_counts=risk_counts,
        top_customers=top_customers,
        monthly_labels=monthly_labels,
        monthly_revenue=monthly_revenue,
        average_probability=float(average_probability),
    )

@app.route("/customers")
@login_required
def customers():
    search = request.args.get("search", "").strip()
    risk = request.args.get("risk", "").strip()

    query = """
        SELECT
            c.customer_id,
            c.name,
            c.age,
            c.city,
            c.country,
            c.join_date,
            COALESCE(cp.churn_probability, 0) AS churn_probability,
            COALESCE(cp.risk_level, 'Not scored') AS risk_level
        FROM customers c
        LEFT JOIN churn_predictions cp
            ON cp.customer_id = c.customer_id
        WHERE (
            :search = ''
            OR CAST(c.customer_id AS CHAR) LIKE :pattern
            OR COALESCE(c.name, '') LIKE :pattern
        )
        AND (:risk = '' OR cp.risk_level = :risk)
        ORDER BY c.customer_id
        LIMIT 200
    """

    with engine.connect() as conn:
        result = conn.execute(
            text(query),
            {
                "search": search,
                "pattern": f"%{search}%",
                "risk": risk,
            },
        )
        customer_list = result.mappings().all()

    return render_template(
        "customers.html",
        customers=customer_list,
        search=search,
        selected_risk=risk,
    )


@app.route("/customers/<int:customer_id>")
def customer_profile(customer_id):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    c.customer_id,
                    c.name,
                    c.age,
                    c.city,
                    c.country,
                    c.join_date,
                    c.segment,
                    cp.churn_probability,
                    cp.risk_level,
                    cp.predicted_at
                FROM customers c
                LEFT JOIN churn_predictions cp
                    ON cp.customer_id = c.customer_id
                WHERE c.customer_id = :customer_id
            """),
            {"customer_id": customer_id},
        )
        customer = result.mappings().first()

        if customer is None:
            from flask import abort
            abort(404)

        transactions = conn.execute(
            text("""
                SELECT
                    transaction_id,
                    transaction_date,
                    product_category,
                    quantity,
                    unit_price,
                    amount
                FROM transactions
                WHERE customer_id = :customer_id
                ORDER BY transaction_date DESC
                LIMIT 20
            """),
            {"customer_id": customer_id},
        ).mappings().all()

        metrics = conn.execute(
            text("""
                SELECT
                    COUNT(*) AS transaction_count,
                    COALESCE(SUM(amount), 0) AS total_spending,
                    MAX(transaction_date) AS last_purchase
                FROM transactions
                WHERE customer_id = :customer_id
            """),
            {"customer_id": customer_id},
        ).mappings().first()

    return render_template(
        "customer_profile.html",
        customer=customer,
        transactions=transactions,
        metrics=metrics,
    )


@app.route("/customers/<int:customer_id>/edit", methods=["POST"])
def edit_customer(customer_id):
    name = request.form.get("name", "").strip()
    age_text = request.form.get("age", "").strip()
    city = request.form.get("city", "").strip()

    if not name:
        flash("Please enter the customer's name.", "error")
        return redirect(
            url_for("customer_profile", customer_id=customer_id)
        )

    try:
        age = int(age_text) if age_text else None
        if age is not None and not 1 <= age <= 120:
            raise ValueError
    except ValueError:
        flash("Enter a valid age between 1 and 120, or leave it blank.", "error")
        return redirect(
            url_for("customer_profile", customer_id=customer_id)
        )

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                UPDATE customers
                SET name = :name,
                    age = :age,
                    city = :city
                WHERE customer_id = :customer_id
            """),
            {
                "name": name,
                "age": age,
                "city": city or None,
                "customer_id": customer_id,
            },
        )

        if result.rowcount == 0:
            from flask import abort
            abort(404)

    flash("Customer details updated successfully.", "success")
    return redirect(
        url_for("customer_profile", customer_id=customer_id)
    )

@app.route("/analytics")
@login_required
def analytics():
    with engine.connect() as conn:

        # 1. Overall business and customer metrics
        summary = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM customers)
                    AS total_customers,
                (SELECT COUNT(*) FROM transactions)
                    AS total_transactions,
                (SELECT COALESCE(SUM(amount), 0)
                 FROM transactions) AS total_revenue,
                (SELECT COUNT(DISTINCT customer_id)
                 FROM transactions) AS purchasing_customers,
                (SELECT COUNT(DISTINCT invoice_no)
                 FROM transactions
                 WHERE invoice_no IS NOT NULL) AS total_orders
        """)).mappings().one()

        # 2. Average order value
        order_stats = conn.execute(text("""
            SELECT AVG(order_total) AS average_order_value
            FROM (
                SELECT invoice_no, SUM(amount) AS order_total
                FROM transactions
                WHERE invoice_no IS NOT NULL
                GROUP BY invoice_no
            ) AS orders
        """)).mappings().one()

        # 3. Monthly revenue trend
        monthly_rows = conn.execute(text("""
            SELECT
                YEAR(transaction_date) AS sales_year,
                MONTH(transaction_date) AS sales_month,
                SUM(amount) AS revenue
            FROM transactions
            GROUP BY
                YEAR(transaction_date),
                MONTH(transaction_date)
            ORDER BY sales_year, sales_month
        """)).mappings().all()

        # 4. Top 10 customers by spending
        top_customers = conn.execute(text("""
            SELECT
                c.customer_id,
                COUNT(t.transaction_id) AS transaction_count,
                COUNT(DISTINCT t.invoice_no) AS order_count,
                COALESCE(SUM(t.amount), 0) AS total_spending,
                MAX(t.transaction_date) AS last_purchase
            FROM customers c
            JOIN transactions t
                ON t.customer_id = c.customer_id
            GROUP BY c.customer_id
            ORDER BY total_spending DESC
            LIMIT 10
        """)).mappings().all()

        # 5. Revenue by country
        country_rows = conn.execute(text("""
            SELECT
                COALESCE(
                    NULLIF(TRIM(c.country), ''),
                    'Not available'
                ) AS country_name,
                SUM(t.amount) AS revenue
            FROM customers c
            JOIN transactions t
                ON t.customer_id = c.customer_id
            GROUP BY
                COALESCE(
                    NULLIF(TRIM(c.country), ''),
                    'Not available'
                )
            ORDER BY revenue DESC
            LIMIT 10
        """)).mappings().all()

        # 6. Purchase recency relative to the latest dataset date
        activity_rows = conn.execute(text("""
            SELECT
                CASE
                    WHEN days_inactive <= 30 THEN '0-30 days'
                    WHEN days_inactive <= 90 THEN '31-90 days'
                    WHEN days_inactive <= 180 THEN '91-180 days'
                    ELSE '181+ days'
                END AS activity_group,
                COUNT(*) AS customer_count
            FROM (
                SELECT
                    customer_id,
                    DATEDIFF(
                        (SELECT MAX(transaction_date)
                         FROM transactions),
                        MAX(transaction_date)
                    ) AS days_inactive
                FROM transactions
                WHERE customer_id IS NOT NULL
                GROUP BY customer_id
            ) AS customer_activity
            GROUP BY activity_group
        """)).mappings().all()

    monthly_labels = [
        f"{row['sales_year']}-{row['sales_month']:02d}"
        for row in monthly_rows
    ]

    monthly_revenue = [
        round(float(row["revenue"] or 0), 2)
        for row in monthly_rows
    ]

    country_labels = [
        row["country_name"] for row in country_rows
    ]

    country_revenue = [
        round(float(row["revenue"] or 0), 2)
        for row in country_rows
    ]

    activity_labels = [
        "0-30 days",
        "31-90 days",
        "91-180 days",
        "181+ days",
    ]

    activity_map = {
        row["activity_group"]: int(row["customer_count"])
        for row in activity_rows
    }

    activity_counts = [
        activity_map.get(label, 0)
        for label in activity_labels
    ]

    return render_template(
        "analytics/analytics.html",
        summary=summary,
        average_order_value=float(
            order_stats["average_order_value"] or 0
        ),
        monthly_labels=monthly_labels,
        monthly_revenue=monthly_revenue,
        top_customers=top_customers,
        country_labels=country_labels,
        country_revenue=country_revenue,
        activity_labels=activity_labels,
        activity_counts=activity_counts,
    )


@app.route("/segmentation")
@login_required
def segmentation():
    import pandas as pd

    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"

    segments_file = processed_dir / "customer_segments.csv"
    summary_file = processed_dir / "customer_segment_summary.csv"

    if not segments_file.exists() or not summary_file.exists():
        return (
            "Segmentation files are missing. "
            "Run scripts/customer_segmentation.py first.",
            500,
        )

    # Read the saved segmentation results.
    customers_df = pd.read_csv(segments_file)
    summary_df = pd.read_csv(summary_file)

    # Friendly names for the four K-Means clusters.
    segment_names = {
        0: "Regular Customers",
        1: "Low-Spending Customers",
        2: "High-Value Customers",
        3: "Inactive / At-Risk Customers",
    }

    # Normalize cluster IDs so they work whether CSV stores
    # them as integers or numeric strings.
    customers_df["cluster"] = pd.to_numeric(
        customers_df["cluster"], errors="coerce"
    )
    summary_df["cluster"] = pd.to_numeric(
        summary_df["cluster"], errors="coerce"
    )

    customers_df = customers_df.dropna(subset=["cluster"]).copy()
    summary_df = summary_df.dropna(subset=["cluster"]).copy()

    customers_df["cluster"] = customers_df["cluster"].astype(int)
    summary_df["cluster"] = summary_df["cluster"].astype(int)

    summary_df["name"] = summary_df["cluster"].map(segment_names)
    summary_df["name"] = summary_df["name"].fillna(
        "Other Segment"
    )

    # Build the four summary cards.
    segment_cards = []

    for _, row in summary_df.sort_values("cluster").iterrows():
        segment_cards.append({
            "cluster": int(row["cluster"]),
            "name": row["name"],
            "customers": int(row["customers"]),
            "average_recency": float(row["average_recency"]),
            "average_spending": float(row["average_spending"]),
            "average_frequency": float(row["average_frequency"]),
            "total_spending": float(row["total_spending"]),
        })

    # Prepare chart data from the same summary.
    chart_labels = [item["name"] for item in segment_cards]
    chart_customers = [
        item["customers"] for item in segment_cards
    ]
    chart_revenue = [
        round(item["total_spending"], 2)
        for item in segment_cards
    ]

    # Prepare the customer table.
    # Show the most recently active customers first.
    customers_df["segment_name"] = customers_df["cluster"].map(
        segment_names
    ).fillna("Other Segment")

    required_columns = [
        "customer_id",
        "cluster",
        "segment_name",
        "recency",
        "frequency",
        "monetary",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in customers_df.columns
    ]

    if missing_columns:
        return (
            "The segmentation CSV is missing required columns: "
            + ", ".join(missing_columns),
            500,
        )

    customers_df = customers_df.sort_values(
        ["recency", "monetary"],
        ascending=[True, False],
    )

    segmented_customers = customers_df[
        required_columns
    ].where(pd.notna(customers_df[required_columns]), None)

    segmented_customers = segmented_customers.to_dict(
        orient="records"
    )

    return render_template(
        "segmentation.html",
        segment_cards=segment_cards,
        chart_labels=chart_labels,
        chart_customers=chart_customers,
        chart_revenue=chart_revenue,
        segmented_customers=segmented_customers,
    )


@app.route("/retention")
@login_required
def retention():
    import pandas as pd

    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"
    segments_file = processed_dir / "customer_segments.csv"

    if not segments_file.exists():
        return (
            "Customer segmentation file is missing. "
            "Run scripts/customer_segmentation.py first.",
            500,
        )

    # Load customer segment assignments.
    segments_df = pd.read_csv(segments_file)

    required_columns = {
        "customer_id", "cluster", "recency",
        "frequency", "monetary"
    }

    if not required_columns.issubset(segments_df.columns):
        return "The customer segmentation CSV has unexpected columns.", 500

    segments_df["customer_id"] = pd.to_numeric(
        segments_df["customer_id"], errors="coerce"
    )
    segments_df["cluster"] = pd.to_numeric(
        segments_df["cluster"], errors="coerce"
    )
    segments_df = segments_df.dropna(
        subset=["customer_id", "cluster"]
    ).copy()

    segments_df["customer_id"] = segments_df["customer_id"].astype(int)
    segments_df["cluster"] = segments_df["cluster"].astype(int)

    segment_names = {
        0: "Regular Customers",
        1: "Low-Spending Customers",
        2: "High-Value Customers",
        3: "Inactive / At-Risk Customers",
    }

    # Fetch the latest saved churn prediction for each customer.
    with engine.connect() as conn:
        prediction_rows = conn.execute(text("""
            SELECT cp.customer_id,
                   cp.churn_probability,
                   cp.risk_level
            FROM churn_predictions cp
            INNER JOIN (
                SELECT customer_id, MAX(predicted_at) AS latest_prediction
                FROM churn_predictions
                GROUP BY customer_id
            ) latest
              ON latest.customer_id = cp.customer_id
             AND latest.latest_prediction = cp.predicted_at
        """)).mappings().all()

    predictions_df = pd.DataFrame(
        [dict(row) for row in prediction_rows],
        columns=["customer_id", "churn_probability", "risk_level"],
    )

    if not predictions_df.empty:
        predictions_df["customer_id"] = pd.to_numeric(
            predictions_df["customer_id"], errors="coerce"
        )
        predictions_df = predictions_df.dropna(
            subset=["customer_id"]
        ).copy()
        predictions_df["customer_id"] = (
            predictions_df["customer_id"].astype(int)
        )

        # If duplicate latest timestamps exist, keep one prediction per customer.
        predictions_df = predictions_df.drop_duplicates(
            subset=["customer_id"], keep="last"
        )

    # Combine segment assignments and churn predictions.
    merged_df = segments_df.merge(
        predictions_df,
        on="customer_id",
        how="left",
    )

    merged_df["segment_name"] = (
        merged_df["cluster"].map(segment_names)
        .fillna("Other Segment")
    )

    merged_df["risk_level"] = (
        merged_df["risk_level"]
        .fillna("Not scored")
        .astype(str)
    )

    merged_df["churn_probability"] = pd.to_numeric(
        merged_df["churn_probability"], errors="coerce"
    )

    # Generate transparent, rule-based recommendations.
    def recommend_action(row):
        risk = row["risk_level"].strip().lower()
        segment = row["segment_name"]

        if risk == "high":
            if segment == "High-Value Customers":
                return "Priority personal outreach and tailored win-back offer"
            return "Prioritize reactivation outreach"

        if risk == "medium":
            return "Send a targeted offer and monitor engagement"

        if risk == "low":
            if segment == "High-Value Customers":
                return "Maintain loyalty benefits and personalized service"
            return "Encourage repeat purchases with relevant offers"

        if segment == "Inactive / At-Risk Customers":
            return "Review purchase history and consider a win-back campaign"

        if segment == "High-Value Customers":
            return "Maintain VIP benefits and personalized recommendations"

        if segment == "Regular Customers":
            return "Encourage repeat purchases and loyalty participation"

        return "Test relevant product recommendations and promotions"

    merged_df["recommendation"] = merged_df.apply(
        recommend_action, axis=1
    )

    # Build segment summary cards.
    segment_cards = []

    for cluster, group in merged_df.groupby("cluster"):
        cluster = int(cluster)

        segment_cards.append({
            "cluster": cluster,
            "name": segment_names.get(cluster, "Other Segment"),
            "customers": int(len(group)),
            "recommendation": {
                0: "Encourage repeat purchases and loyalty participation.",
                1: "Test targeted promotions and product recommendations.",
                2: "Prioritize loyalty benefits and personalized retention.",
                3: "Consider reactivation campaigns for long-inactive customers.",
            }.get(cluster, "Review customer behavior and tailor outreach."),
        })

    segment_cards.sort(key=lambda item: item["cluster"])

    # Prioritize high-risk customers, then medium-risk customers.
    risk_priority = {"high": 0, "medium": 1, "low": 2}

    merged_df["_risk_priority"] = (
        merged_df["risk_level"].str.lower()
        .map(risk_priority)
        .fillna(3)
    )

    merged_df = merged_df.sort_values(
        ["_risk_priority", "churn_probability"],
        ascending=[True, False],
        na_position="last",
    )

    retention_customers = []

    for _, row in merged_df.iterrows():
        probability = row["churn_probability"]

        retention_customers.append({
            "customer_id": int(row["customer_id"]),
            "segment_name": row["segment_name"],
            "risk_level": row["risk_level"],
            "churn_probability": (
                float(probability)
                if pd.notna(probability) else None
            ),
            "recommendation": row["recommendation"],
        })

    return render_template(
        "retention.html",
        segment_cards=segment_cards,
        retention_customers=retention_customers,
    )


@app.route("/churn-analytics")
@login_required
def churn_analytics():
    import re

    project_root = Path(__file__).resolve().parent
    report_file = project_root / "reports" / "model_evaluation.txt"

    # Get the latest prediction for each customer.
    with engine.connect() as conn:
        prediction_rows = conn.execute(text("""
            SELECT
                cp.customer_id,
                cp.churn_probability,
                cp.risk_level,
                cp.predicted_at
            FROM churn_predictions cp
            INNER JOIN (
                SELECT
                    customer_id,
                    MAX(predicted_at) AS latest_prediction
                FROM churn_predictions
                GROUP BY customer_id
            ) latest
                ON latest.customer_id = cp.customer_id
                AND latest.latest_prediction = cp.predicted_at
            ORDER BY cp.churn_probability DESC
        """)).mappings().all()

    predictions = [dict(row) for row in prediction_rows]

    # Summarize the available predictions.
    total_scored = len(predictions)
    high_risk_count = sum(
        1 for row in predictions
        if str(row["risk_level"] or "").strip().lower() == "high"
    )

    probabilities = [
        float(row["churn_probability"])
        for row in predictions
        if row["churn_probability"] is not None
    ]

    average_probability = (
        sum(probabilities) / len(probabilities)
        if probabilities else 0.0
    )

    # Keep chart categories consistent, including empty categories.
    risk_labels = ["High", "Medium", "Low"]
    risk_counts = [
        sum(
            1 for row in predictions
            if str(row["risk_level"] or "").strip().lower() == label.lower()
        )
        for label in risk_labels
    ]

    # Prepare the customer table.
    top_risk_customers = []

    for row in predictions[:100]:
        probability = row["churn_probability"]
        predicted_at = row["predicted_at"]

        top_risk_customers.append({
            "customer_id": row["customer_id"],
            "risk_level": row["risk_level"] or "Not scored",
            "churn_probability": (
                float(probability) if probability is not None else 0.0
            ),
            "predicted_at": (
                predicted_at.strftime("%Y-%m-%d %H:%M")
                if hasattr(predicted_at, "strftime")
                else str(predicted_at or "—")
            ),
        })

    # Read the existing model evaluation report.
    evaluation_text = ""
    roc_auc = None

    if report_file.exists():
        evaluation_text = report_file.read_text(encoding="utf-8")

        match = re.search(
            r"ROC[- ]AUC\s*[:=]\s*([0-9]*\.?[0-9]+)",
            evaluation_text,
            re.IGNORECASE,
        )

        if match:
            roc_auc = float(match.group(1))

    return render_template(
        "churn_analytics.html",
        total_scored=total_scored,
        high_risk_count=high_risk_count,
        average_probability=average_probability,
        risk_labels=risk_labels,
        risk_counts=risk_counts,
        top_risk_customers=top_risk_customers,
        evaluation_text=evaluation_text,
        roc_auc=roc_auc,
    )


if __name__ == "__main__":
    app.run(debug=True)
