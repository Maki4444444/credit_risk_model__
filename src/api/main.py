"""
main.py
-------
FastAPI application for the Bati Bank Credit Risk Scoring API.

Loads the best registered model from the MLflow Model Registry and
exposes a /predict endpoint that returns a binary risk label and
probability score for a given customer's feature vector.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health   — Health check and model version info
    POST /predict  — Credit risk prediction
    GET  /docs     — Auto-generated Swagger UI
"""

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.pydantic_models import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME", "credit_risk_best_model"
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "mlruns"
)
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")

# ── Feature column order must match training ──────────────────────────────────

FEATURE_COLUMNS = [
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "total_value",
    "avg_value",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "PricingStrategy",
]

# ── App initialisation ────────────────────────────────────────────────────────

app = FastAPI(
    title="Bati Bank Credit Risk Scoring API",
    description=(
        "Predicts the probability that a customer is high-risk "
        "for a buy-now-pay-later loan, using a model trained on "
        "Xente eCommerce transaction data."
    ),
    version="1.0.0",
)

# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_model_version = "unknown"


def load_model():
    """
    Load the registered model from the MLflow Model Registry.
    Called once at startup.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded from the registry.
    """
    global _model, _model_version

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=["None", "Staging", "Production"]
        )

        if not versions:
            raise RuntimeError(
                f"No versions found for model '{REGISTERED_MODEL_NAME}'. "
                "Run src/train.py first to train and register a model."
            )

        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
        _model = mlflow.sklearn.load_model(model_uri)
        _model_version = latest.version

        logger.info(
            "Model loaded: '%s' version %s from '%s'",
            REGISTERED_MODEL_NAME, _model_version, MLFLOW_TRACKING_URI,
        )

    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(f"Model loading failed: {exc}") from exc


@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts."""
    try:
        load_model()
    except RuntimeError as exc:
        logger.warning(
            "Model not loaded at startup: %s. "
            "/predict will return 503 until the model is available.",
            exc,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Check API health and confirm the model is loaded.

    Returns
    -------
    HealthResponse with status, model name, and version.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run src/train.py to train and register "
                "a model, then restart the API."
            ),
        )
    return HealthResponse(
        status="ok",
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(_model_version),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict credit risk for a customer.

    Accepts a feature vector and returns:
    - is_high_risk: binary label (1 = high risk, 0 = low risk)
    - risk_probability: probability of being high risk (0.0 to 1.0)
    - model_version: version of the model used

    Raises
    ------
    HTTPException 503
        If the model is not loaded.
    HTTPException 422
        If the request body is invalid (handled by Pydantic).
    HTTPException 500
        If prediction fails unexpectedly.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. See /health for details.",
        )

    try:
        # Build feature dict from request — only include training features
        feature_dict = {
            "total_transaction_amount": request.total_transaction_amount,
            "avg_transaction_amount": request.avg_transaction_amount,
            "transaction_count": request.transaction_count,
            "std_transaction_amount": request.std_transaction_amount,
            "total_value": request.total_value,
            "avg_value": request.avg_value,
            "transaction_hour": request.transaction_hour,
            "transaction_day_of_week": request.transaction_day_of_week,
            "transaction_day": request.transaction_day,
            "transaction_month": request.transaction_month,
            "transaction_year": request.transaction_year,
            "PricingStrategy": request.PricingStrategy,
        }

        # Build DataFrame in correct column order
        X = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

        # Predict
        prediction = int(_model.predict(X)[0])

        if hasattr(_model, "predict_proba"):
            probability = float(_model.predict_proba(X)[0][1])
        else:
            probability = float(prediction)

        logger.info(
            "Prediction — customer: %s | is_high_risk: %d | probability: %.4f",
            request.customer_id, prediction, probability,
        )

        return PredictResponse(
            customer_id=request.customer_id,
            is_high_risk=prediction,
            risk_probability=round(probability, 4),
            model_version=str(_model_version),
        )

    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.get("/", tags=["Root"])
def root():
    """Redirect information to /docs."""
    return JSONResponse(
        content={
            "message": "Bati Bank Credit Risk Scoring API",
            "docs": "/docs",
            "health": "/health",
            "predict": "POST /predict",
        }
    )