"""
pydantic_models.py
------------------
Request and response schemas for the Bati Bank Credit Risk Scoring API.

Used by:
    - src/api/main.py
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """
    Input features required to score a customer's credit risk.

    All numerical features are expected to be in the same scale as
    the training data (i.e., already processed by the feature pipeline).
    """

    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer identifier for tracking purposes.",
        example="CustomerId_4406",
    )

    # Aggregate features
    total_transaction_amount: float = Field(
        ..., description="Sum of all transaction amounts for the customer.",
        example=15000.0,
    )
    avg_transaction_amount: float = Field(
        ..., description="Mean transaction amount for the customer.",
        example=3000.0,
    )
    transaction_count: int = Field(
        ..., description="Total number of transactions.",
        example=5,
    )
    std_transaction_amount: float = Field(
        ..., description="Standard deviation of transaction amounts.",
        example=1200.0,
    )
    total_value: float = Field(
        ..., description="Sum of absolute transaction values.",
        example=15000.0,
    )
    avg_value: float = Field(
        ..., description="Mean absolute transaction value.",
        example=3000.0,
    )

    # Temporal features
    transaction_hour: float = Field(
        ..., description="Mean hour of day of transactions (0-23).",
        example=14.5,
    )
    transaction_day_of_week: float = Field(
        ..., description="Mean day of week (0=Monday, 6=Sunday).",
        example=2.3,
    )
    transaction_day: float = Field(
        ..., description="Mean day of month.",
        example=15.0,
    )
    transaction_month: float = Field(
        ..., description="Mean month of year.",
        example=11.0,
    )
    transaction_year: float = Field(
        ..., description="Mean year of transactions.",
        example=2018.0,
    )

    # Other features
    PricingStrategy: float = Field(
        ..., description="Mean pricing strategy tier.",
        example=2.0,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CustomerId_4406",
                "total_transaction_amount": 15000.0,
                "avg_transaction_amount": 3000.0,
                "transaction_count": 5,
                "std_transaction_amount": 1200.0,
                "total_value": 15000.0,
                "avg_value": 3000.0,
                "transaction_hour": 14.5,
                "transaction_day_of_week": 2.3,
                "transaction_day": 15.0,
                "transaction_month": 11.0,
                "transaction_year": 2018.0,
                "PricingStrategy": 2.0,
            }
        }
    }


class PredictResponse(BaseModel):
    """
    Credit risk prediction output returned by the /predict endpoint.
    """

    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier echoed from the request.",
    )
    is_high_risk: int = Field(
        ...,
        description="Binary risk label: 1 = high risk, 0 = low risk.",
        example=1,
    )
    risk_probability: float = Field(
        ...,
        description="Probability of being high risk (0.0 to 1.0).",
        example=0.82,
    )
    model_version: str = Field(
        ...,
        description="Version of the registered MLflow model used.",
        example="1",
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(..., example="ok")
    model_name: str = Field(..., example="credit_risk_best_model")
    model_version: str = Field(..., example="1")
    