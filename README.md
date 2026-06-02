## Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model for Bati Bank's buy-now-pay-later service, using eCommerce transaction data from the Xente platform.

### Project Structure

credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # excluded from git
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed data for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training & MLflow tracking
│   ├── predict.py                 # Inference
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

### Credit Scoring Business Understanding

#### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires financial institutions to hold capital reserves proportional to the credit risk they carry, and critically, to **justify and document** how that risk is measured. This creates a direct regulatory requirement for model interpretability, not just accuracy.

In practice, this means:

- **Regulators and internal audit teams must be able to understand why a model assigns a given risk score** to a borrower. A black-box model that produces accurate predictions but cannot explain its decisions is non-compliant under Basel II's Pillar 2 (Supervisory Review Process), regardless of its predictive performance.
- **Model documentation must cover the full lifecycle**: data sources, feature selection rationale, target variable definition, training methodology, validation results, and known limitations. Any proxy variable (such as the RFM-based label used in this project) must be explicitly justified and its assumptions stated.
- **Model monitoring and stability reporting** are expected on an ongoing basis. This means tracking input feature distributions (Population Stability Index) and output score distributions over time to detect model drift, a requirement that influences how the pipeline is designed from the start.

Practically, these constraints favor models like **Logistic Regression with Weight of Evidence (WoE)-encoded features**, which produce a linear, auditable score that maps directly to a probability of default. Each coefficient can be explained to a non-technical risk officer in plain language. High-performance ensemble models, while potentially more accurate, require additional interpretability tooling (e.g., SHAP values) to meet the same standard.

The implication for this project is that interpretability is a **first-class modeling requirement**, not a post-hoc nicety.


#### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The Xente transaction dataset contains no column indicating whether a customer ever defaulted on a loan. This is common with alternative data sources. The data was not collected for credit purposes, so it lacks the outcome variable (repayment vs. default) that supervised credit models require.

A **proxy variable** solves this by using observable behavioral signals to infer credit risk. In this project, RFM (Recency, Frequency, Monetary) metrics are calculated per customer and used to cluster them. Customers who are disengaged,  infrequent, low-spending, and long-absent from the platform are labeled as high-risk (`is_high_risk = 1`), on the assumption that behavioral disengagement correlates with financial stress or unreliability.

However, proxy-based prediction introduces several business risks that must be acknowledged:

- **Construct validity risk**: The proxy (disengagement) may not actually correlate with loan default. A customer could be inactive on this eCommerce platform for reasons entirely unrelated to creditworthiness (e.g., they switched platforms, moved abroad, or simply don't need the product anymore). If the proxy is wrong, the model penalizes creditworthy customers.
- **Label noise risk**: Because the target is derived, not observed, every model trained on it inherits the noise and assumptions baked into the clustering step. Errors in proxy construction propagate into every downstream artifact.
- **Fairness and discrimination risk**: Behavioral proxies can inadvertently encode demographic patterns. If certain customer segments (by geography, age, or income) naturally show lower transaction frequency for structural reasons, the proxy may disproportionately classify them as high-risk, raising ethical and potentially legal concerns.
- **Regulatory defensibility risk**: Basel II expects models to be validated against actual default outcomes. A proxy-based model must be clearly labeled as a **temporary or provisional tool**, with a plan to retrain against real default labels once loan repayment data becomes available.

These risks must be explicitly stated in the final report and model documentation. The proxy is a reasonable starting point given data constraints. But it is a modeling assumption, not ground truth.


#### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost / LightGBM) |
|---|---|---|
| **Interpretability** | High: linear coefficients map directly to score contributions | Low: requires SHAP or LIME for post-hoc explanation |
| **Regulatory compliance** | Easier to document and defend under Basel II Pillar 2 | Requires additional interpretability layer to meet the same standard |
| **Predictive performance** | Moderate: assumes linear relationships between WoE features and log-odds | High: captures non-linear interactions and feature dependencies |
| **Handling imbalanced data** | Sensitive: requires explicit resampling or class weighting | More robust natively, though still benefits from class weighting |
| **Implementation complexity** | Low: well-understood, fast to train and validate | Higher: more hyperparameters, longer training, more complex pipeline |
| **Auditability** | Full: every score can be decomposed into feature contributions analytically | Partial: requires tooling (e.g., SHAP) to decompose predictions |
| **Stability** | High: WoE binning is robust to outliers and monotonic transformations | Moderate: prone to overfitting without careful tuning and cross-validation |

In a regulated financial context like Bati Bank, the preferred approach is typically to **start with Logistic Regression + WoE** as the production model (for compliance and auditability), and use a Gradient Boosting model as a **challenger model** to benchmark performance. If the challenger significantly outperforms and interpretability requirements can be met via SHAP documentation, the case for deploying it can be made to the risk committee with appropriate justification.

This project trains both types, tracks all experiments in MLflow, and selects the best model with explicit documentation of the trade-off decision.


### Exploratory Data Analysis Summary

The EDA notebook (`notebooks/eda.ipynb`) explores the Xente transaction dataset across 7 sections:

#### Key Findings

1. **Severe Class Imbalance** Only ~0.2% of transactions are fraudulent (193/95,662).
   F1 and Precision-Recall AUC are the correct evaluation metrics, not accuracy.

2. **Amount is Bidirectional; Value is Always Positive** `Amount` contains both
   credits (negative) and debits (positive). The sign of Amount is a meaningful
   feature. `Value` is always the absolute amount.

3. **Extreme Outliers Require Log Transformation** Both `Amount` and `Value` are
   highly right-skewed with heavy tails reaching tens of millions UGX.
   Feature engineering must apply `log1p` transformation and/or robust scaling.

4. **Strong Temporal Patterns** Activity peaks at 16:00–17:00 and is near-zero
   between midnight and 4 AM. `transaction_hour` and `transaction_day_of_week`
   carry predictive signal and will be extracted in Task 3.

5. **ProductCategory Drives Fraud Rate Unevenly** `transport` has an 8% fraud rate;
   `financial_services` 0.35%; `airtime` near zero. WoE encoding per category
   will capture this risk gradient better than one-hot encoding.

Zero missing values found across all 16 columns.
`CountryCode` is zero-variance (constant = 256) will be dropped in Task 3.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Maki4444444/credit_risk_model__.git
cd credit_risk_model__

# 2. Create a virtual environment (Python 3.11 required)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the Xente dataset in data/raw/
# Download from: https://www.kaggle.com/datasets/atwine/xente-challenge
# Expected file: data/raw/data.csv

> **Note:** `data/` is excluded from version control. You must download
> the dataset manually and place it in `data/raw/` before running any notebooks.

### Notebooks

| Notebook | Description | nbviewer |
|---|---|---|
| `notebooks/eda.ipynb` | Task 2 — Exploratory Data Analysis | [View](https://nbviewer.org/github/Maki4444444/credit_risk_model__/blob/main/notebooks/eda.ipynb) |


Tasks
Task	Description	Branch	Status
1	Set up project structure, README, and Basel II business understanding	task-1	Completed
2	Perform EDA: distributions, correlations, missing values, outliers	task-2	Completed
3	Build feature engineering pipeline: aggregations, datetime, encoding, scaling, WoE	task-3	Completed
4	Engineer proxy target: RFM, K‑means clustering, create is_high_risk label	task-4	Completed
5	Train multiple models, hyperparameter tuning, MLflow tracking & registration	task-5	Completed
6	Deploy FastAPI app, containerize with Docker, add CI/CD with GitHub Actions	task-6	Completed
