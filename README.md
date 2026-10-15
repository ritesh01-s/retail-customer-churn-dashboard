
# Retail Customer Churn Prediction & Retention Dashboard

<p align="center">
  <strong>Turning Retail Customer Data into Actionable Insights</strong>
</p>

<p align="center">
  An end-to-end data analytics and machine learning project for
  customer churn prediction, customer segmentation, and retention analysis.
</p>

---

## Overview

The **Retail Customer Churn Prediction & Retention Dashboard** is a
data-driven web application designed to help analyze customer purchasing
behavior, identify customers who may stop purchasing, and support
data-informed retention strategies.

The project combines data preprocessing, exploratory data analysis,
machine learning, customer segmentation, and interactive visualizations
within a Flask-based dashboard connected to a MySQL database.

It transforms historical retail transaction data into customer-level
features, estimates churn probabilities, categorizes customers by risk,
and provides analytical insights to support business decision-making.

## Key Features

- **Customer Analytics:** Explore customer profiles, purchasing activity,
  and transaction behavior.
- **Churn Prediction:** Estimate customer churn probability using
  supervised machine learning.
- **Risk Classification:** Categorize scored customers into High, Medium,
  and Low churn-risk groups.
- **Customer Segmentation:** Apply K-Means clustering to identify groups
  of customers with similar purchasing patterns.
- **Interactive Dashboard:** Visualize key performance indicators,
  revenue trends, and churn-risk distributions.
- **Retention Analytics:** Support the identification of customers who
  may benefit from targeted retention activities.
- **Customer Management:** View customer information and related
  analytical details.
- **Authentication:** Provide registration, login, password hashing,
  and session-based access control.
- **Database Integration:** Store and retrieve application data using MySQL.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend Framework | Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | MySQL |
| Database Connectivity | SQLAlchemy, PyMySQL |
| Frontend | HTML5, CSS3, JavaScript |
| Data Visualization | Chart.js |
| Model Persistence | Joblib |
| Version Control | Git and GitHub |

## Machine Learning Approach

### 1. Data Preprocessing

The retail transaction dataset is cleaned and prepared for analysis.
The preprocessing workflow includes handling invalid transaction values,
preparing transaction dates, and deriving customer-level behavioral
features.

### 2. Feature Engineering

Customer-level features include:

- Total spending
- Total transactions
- Total items purchased
- Average order value
- Purchase frequency
- Number of unique products
- Active days
- Days since last purchase
- Recent purchasing activity

These features help represent customer purchasing behavior for churn
prediction and segmentation.

### 3. Churn Labeling

Customers are labeled using historical purchase activity and a defined
observation and follow-up period. The resulting dataset is used to train
and evaluate supervised learning models.

Churn labels depend on the selected observation window and business
definition; they should not automatically be interpreted as confirmed
customer cancellations.

### 4. Model Training and Evaluation

Two classification models were evaluated:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 63.42% | 64.74% |
| ROC-AUC | 0.7607 | 0.7196 |
| Churn Precision | 0.45 | 0.46 |
| Churn Recall | 0.82 | 0.69 |
| Churn F1-score | 0.58 | 0.55 |

**Selected model: Logistic Regression**

Logistic Regression was selected based on its higher ROC-AUC score and
higher recall for the churn class in the recorded evaluation results.

The results indicate that the model identifies a substantial proportion
of labeled churn cases, although its precision shows that some customers
predicted as churn risks are not labeled as churners.

These metrics describe the recorded test evaluation and are not a
guarantee of future business performance.

### 5. Customer Risk Scoring

The recorded prediction output contains 1,896 scored customers:

| Risk Category | Customers |
|---|---:|
| High | 313 |
| Medium | 933 |
| Low | 650 |
| **Total** | **1,896** |

The average predicted churn probability in this output is approximately
46.30%.

Risk categories are intended to help prioritize customer analysis and
retention efforts. They do not establish that an individual customer
will definitely churn.

### 6. Customer Segmentation

K-Means clustering is used to group customers according to selected
behavioral features, including Recency, Frequency, and Monetary value.

The project generates four customer clusters for further analysis.
These segments can help explore differences in purchasing activity and
support the development of segment-specific engagement strategies.

## Dashboard Capabilities

The web application includes analytical areas for:

- Dashboard overview and key performance indicators
- Customer records and customer details
- Transaction analytics
- Churn analytics and risk distribution
- Customer segmentation
- Retention analysis
- User profile and authentication

Interactive charts help present analytical results in a form that is
easier to interpret than raw transaction data.

## Project Structure

The following is a representative structure. Some filenames may differ
depending on the current version of the project.

```text
retail-churn-dashboard/
├── app.py
├── config.py
├── models.py
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── import_data.py
│   ├── feature_engineering.py
│   ├── create_churn_labels.py
│   ├── train_model.py
│   └── predict_churn.py
│
├── models/
│   └── trained model artifacts
│
├── templates/
│   ├── dashboard.html
│   ├── login.html
│   ├── register.html
│   ├── customers.html
│   ├── analytics.html
│   ├── segmentation.html
│   └── retention.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── instance/
│   └── local database files, if applicable
│
└── README.md
```

> This tree is illustrative. Update it to match the files actually
> committed to your GitHub repository.

## Installation and Setup

### Prerequisites

Install the following before running the application:

- Python compatible with the project's dependencies
- MySQL Server
- Git
- A code editor such as Visual Studio Code

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/retail-customer-churn-dashboard.git
cd retail-customer-churn-dashboard
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your organization's security
policy or use the appropriate activation method for your terminal.

### 3. Install Dependencies

Once `requirements.txt` contains the project's actual dependencies, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not leave `requirements.txt` empty. Include the packages and compatible
versions required by the application and its processing scripts.

### 4. Configure Environment Variables

Create a local `.env` file in the project root. Use the variable names
expected by the application configuration.

For example:

```dotenv
SECRET_KEY=replace_with_a_secure_random_secret
DATABASE_URL=mysql+pymysql://DB_USER:DB_PASSWORD@localhost/DB_NAME
```

Replace the example values with your local configuration.

**Security:** Never commit `.env` or publish real passwords, secret keys,
database credentials, or private data. Commit `.env.example` with
placeholder values only.

### 5. Create the MySQL Database

Create a database in MySQL using the database name configured in
`DATABASE_URL`.

For example:

```sql
CREATE DATABASE retail_churn_db;
```

Configure the database schema and tables according to the application's
SQLAlchemy models and initialization workflow before running the app.

### 6. Prepare the Data

Place the authorized source dataset in the location expected by the
project's import script.

Run the data-processing scripts in their intended order. Depending on
the available scripts, the workflow includes:

1. Import and clean retail transaction data.
2. Generate customer-level features.
3. Create churn labels.
4. Train and evaluate the classification models.
5. Generate customer churn probabilities.
6. Perform customer segmentation.
7. Load the required results into the application database.

Check each script's arguments and configuration before execution.
Do not assume that the scripts will download the dataset automatically.

### 7. Run the Application

Use the entry point and configuration defined by your project. If
`app.py` supports direct execution, start it with:

```bash
python app.py
```

Open the local URL printed by Flask in your terminal.

For local development, use the development server only. Configure a
production-ready server and appropriate security settings before
deploying the application publicly.

## Security Considerations

- Keep `.env` and other secret-bearing files out of version control.
- Use password hashing rather than storing plaintext passwords.
- Protect routes that display or modify customer information.
- Validate user input and enforce authorization checks.
- Avoid publishing identifiable customer information.
- Do not use Flask debug mode in production.
- Apply appropriate database permissions and secure credentials.
- Review the repository for large files and sensitive data before pushing.

## Limitations

- Churn predictions depend on historical transaction patterns and the
  chosen churn definition.
- Model performance may change with different datasets or time periods.
- False-positive predictions can lead to unnecessary retention efforts.
- Customer segments require business interpretation and validation.
- Historical retail data may not represent current customer behavior.
- Evaluation metrics alone do not prove that retention campaigns will
  increase revenue.

## Future Enhancements

- Improve model evaluation using time-based validation.
- Tune classification thresholds according to business costs.
- Compare additional machine-learning algorithms.
- Add model monitoring and retraining workflows.
- Develop personalized retention recommendations.
- Add campaign tracking and retention outcome analysis.
- Improve dashboard filtering and export capabilities.
- Deploy the application to a secure cloud environment.
- Add automated testing and continuous integration.
- Document reproducible database and data preparation steps.

## Intended Use

This project is intended for academic learning, retail analytics,
machine-learning experimentation, and demonstration of a data-driven
customer retention workflow.

Predictions should support human decision-making rather than serve as
the sole basis for customer-related actions.

## Author

**Ritesh**

<p align="center">
  <strong>Retail Analytics • Machine Learning • Customer Retention</strong>
</p>
