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


### Setup

```bash
# Clone the repo
git clone https://github.com/.../credit_risk_model__.git
cd credit-risk-model

# Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## Tasks

| Task | Description | Branch | Status |
|---|---|---|---|
| 1 | Business Understanding + Repo Setup | `task-1` | Complete |
| 2 | Exploratory Data Analysis | `task-2` | To Be Completed | 
| 3 | Feature Engineering Pipeline | `task-3` | To Be Completed |
| 4 | Proxy Target Variable Engineering | `task-4` | To Be Completed |
| 5 | Model Training & MLflow Tracking | `task-5` | To Be Completed |
| 6 | API Deployment & CI/CD | `task-6` | To Be Completed |

TBC: To be completed