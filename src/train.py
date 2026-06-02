"""
train.py
--------
Model training and experiment tracking for the Bati Bank Credit Risk
Scoring project.

Trains multiple classification models on the processed dataset with
is_high_risk as the target variable, logs all experiments to MLflow,
and registers the best model in the MLflow Model Registry.

Models trained:
    - Logistic Regression  (primary production candidate — Basel II compliant)
    - Decision Tree        (interpretable baseline)
    - Random Forest        (ensemble challenger)
    - XGBoost              (gradient boosting challenger)

Usage:
    python src/train.py
"""

import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = "data/processed/data_with_target.csv"
TARGET_COL = "is_high_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT = "credit_risk_scoring"
REGISTERED_MODEL_NAME = "credit_risk_best_model"

# Columns to exclude from features
DROP_FROM_FEATURES = [TARGET_COL, "FraudResult", "CustomerId"]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Preparation
# ════════════════════════════════════════════════════════════════════════════

def load_model_data(filepath: str) -> tuple:
    """
    Load the processed dataset and split into features and target.

    Parameters
    ----------
    filepath : str
        Path to the processed CSV with is_high_risk column.

    Returns
    -------
    tuple : (X, y)
        X — feature DataFrame
        y — binary target Series

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If TARGET_COL is not in the dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model data not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded model data: %d rows × %d columns", *df.shape)

    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found. "
            "Run Task 4 proxy target engineering first."
        )

    drop_cols = [c for c in DROP_FROM_FEATURES if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    logger.info(
        "Features: %d | Target: %s | Positive rate: %.1f%%",
        X.shape[1], TARGET_COL, y.mean() * 100,
    )
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train: %d | Test: %d | Train positive rate: %.1f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a fitted model and return all required metrics.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model Configs
# ════════════════════════════════════════════════════════════════════════════

def get_model_configs() -> list:
    """
    Return list of (name, model, param_grid, search_strategy) tuples.

    search_strategy: 'grid' or 'random'
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    configs = [
        (
            "LogisticRegression",
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
            "grid",
            cv,
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            "grid",
            cv,
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
            "random",
            cv,
        ),
        (
            "XGBoost",
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            },
            "random",
            cv,
        ),
    ]
    return configs


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Training with MLflow
# ════════════════════════════════════════════════════════════════════════════

def train_and_track(
    name: str,
    model,
    param_grid: dict,
    search_strategy: str,
    cv,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple:
    """
    Run hyperparameter search, evaluate, and log everything to MLflow.

    Parameters
    ----------
    name : str
        Model name for MLflow run.
    model : sklearn estimator
    param_grid : dict
    search_strategy : str — 'grid' or 'random'
    cv : cross-validation splitter
    X_train, X_test, y_train, y_test : split data

    Returns
    -------
    tuple : (best_estimator, metrics, run_id)
    """
    logger.info("Training %s ...", name)

    if search_strategy == "grid":
        searcher = GridSearchCV(
            model, param_grid, cv=cv,
            scoring="f1", n_jobs=-1, refit=True,
        )
    else:
        searcher = RandomizedSearchCV(
            model, param_grid, cv=cv,
            n_iter=20, scoring="f1",
            n_jobs=-1, refit=True,
            random_state=RANDOM_STATE,
        )

    with mlflow.start_run(run_name=name) as run:
        searcher.fit(X_train, y_train)
        best_model = searcher.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        # Log parameters
        mlflow.log_params(searcher.best_params_)
        mlflow.log_param("model_name", name)
        mlflow.log_param("search_strategy", search_strategy)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=None,
        )

        run_id = run.info.run_id

    logger.info(
        "%s — F1: %.4f | ROC-AUC: %.4f | Best params: %s",
        name, metrics["f1"], metrics["roc_auc"], searcher.best_params_,
    )
    return best_model, metrics, run_id


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Register Best Model
# ════════════════════════════════════════════════════════════════════════════

def register_best_model(results: list) -> None:
    """
    Identify the best model by ROC-AUC and register it in the
    MLflow Model Registry.

    Parameters
    ----------
    results : list of (name, model, metrics, run_id) tuples
    """
    best = max(results, key=lambda x: x[2]["roc_auc"])
    best_name, best_model, best_metrics, best_run_id = best

    logger.info(
        "Best model: %s | ROC-AUC: %.4f | F1: %.4f",
        best_name, best_metrics["roc_auc"], best_metrics["f1"],
    )

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    logger.info(
        "Model registered as '%s' version %s",
        REGISTERED_MODEL_NAME, mv.version,
    )

    # Print final comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 70)
    for name, _, metrics, _ in sorted(results, key=lambda x: x[2]["roc_auc"], reverse=True):
        print(
            f"{name:<22} {metrics['accuracy']:>9.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['roc_auc']:>9.4f}"
        )
    print("=" * 70)
    print(f"\n✅ Best model: {best_name} registered as '{REGISTERED_MODEL_NAME}' v{mv.version}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Main Entry Point
# ════════════════════════════════════════════════════════════════════════════

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("MLflow experiment: '%s'", MLFLOW_EXPERIMENT)

    # Load and split data
    X, y = load_model_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train all models
    results = []
    for name, model, param_grid, strategy, cv in get_model_configs():
        best_model, metrics, run_id = train_and_track(
            name, model, param_grid, strategy, cv,
            X_train, X_test, y_train, y_test,
        )
        results.append((name, best_model, metrics, run_id))

    # Register the best model
    register_best_model(results)


if __name__ == "__main__":
    main()
    