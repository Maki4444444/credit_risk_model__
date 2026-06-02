"""
test_data_processing.py
-----------------------
Unit tests for src/data_processing.py helper functions.

Run with:
    pytest tests/test_data_processing.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing import (
    cap_outliers,
    compute_rfm,
    assign_high_risk_label,
    engineer_features,
    get_data_overview,
    get_missing_value_report,
    get_outlier_report,
    build_model_dataset,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transactions():
    """Minimal transaction DataFrame matching Xente schema."""
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "BatchId": ["B1", "B1", "B2", "B2", "B3", "B3"],
        "AccountId": ["A1", "A1", "A2", "A2", "A3", "A3"],
        "SubscriptionId": ["S1", "S1", "S2", "S2", "S3", "S3"],
        "CustomerId": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "CurrencyCode": ["UGX"] * 6,
        "CountryCode": [256] * 6,
        "ProviderId": ["ProviderId_4"] * 6,
        "ProductId": ["ProductId_1"] * 6,
        "ProductCategory": ["airtime", "airtime", "financial_services",
                             "financial_services", "airtime", "utility_bill"],
        "ChannelId": ["ChannelId_3"] * 6,
        "Amount": [1000, 2000, 500, -200, 3000, 1500],
        "Value": [1000, 2000, 500, 200, 3000, 1500],
        "TransactionStartTime": pd.to_datetime([
            "2018-11-15 08:00:00",
            "2018-12-01 14:00:00",
            "2018-11-20 10:00:00",
            "2019-01-05 16:00:00",
            "2018-11-10 09:00:00",
            "2019-02-01 11:00:00",
        ]),
        "PricingStrategy": [2, 2, 2, 2, 2, 2],
        "FraudResult": [0, 0, 0, 1, 0, 0],
    })


@pytest.fixture
def sample_rfm():
    """Minimal RFM DataFrame for clustering tests."""
    return pd.DataFrame({
        "CustomerId": [f"C{i}" for i in range(20)],
        "recency":    [10, 90, 5, 120, 30, 200, 15, 80, 45, 160,
                       20, 100, 8, 140, 25, 180, 12, 95, 50, 170],
        "frequency":  [50, 3, 80, 2, 30, 1, 60, 5, 25, 2,
                       45, 4, 70, 2, 35, 1, 55, 6, 20, 3],
        "monetary":   [500000, 5000, 900000, 3000, 200000, 1000,
                       700000, 8000, 150000, 2000, 400000, 6000,
                       800000, 2500, 250000, 500, 600000, 7000,
                       120000, 4000],
    })


# ── Tests: cap_outliers ───────────────────────────────────────────────────────

class TestCapOutliers:

    def test_caps_values_at_percentiles(self):
        df = pd.DataFrame({"amount": list(range(1, 101))})
        result = cap_outliers(df, cols=["amount"], lower_pct=0.05, upper_pct=0.95)
        assert result["amount"].min() >= df["amount"].quantile(0.05)
        assert result["amount"].max() <= df["amount"].quantile(0.95)

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"amount": [1, 2, 1000]})
        original_max = df["amount"].max()
        cap_outliers(df, cols=["amount"])
        assert df["amount"].max() == original_max  # input unchanged

    def test_raises_type_error_for_non_dataframe(self):
        with pytest.raises(TypeError):
            cap_outliers([1, 2, 3], cols=["amount"])

    def test_raises_key_error_for_missing_column(self):
        df = pd.DataFrame({"amount": [1, 2, 3]})
        with pytest.raises(KeyError):
            cap_outliers(df, cols=["nonexistent"])

    def test_raises_value_error_for_non_numeric(self):
        df = pd.DataFrame({"category": ["a", "b", "c"]})
        with pytest.raises(ValueError):
            cap_outliers(df, cols=["category"])


# ── Tests: engineer_features ─────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_returns_one_row_per_customer(self, sample_transactions):
        result = engineer_features(sample_transactions)
        assert len(result) == sample_transactions["CustomerId"].nunique()

    def test_expected_aggregate_columns_present(self, sample_transactions):
        result = engineer_features(sample_transactions)
        expected = [
            "total_transaction_amount", "avg_transaction_amount",
            "transaction_count", "std_transaction_amount",
            "total_value", "avg_value",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_expected_temporal_columns_present(self, sample_transactions):
        result = engineer_features(sample_transactions)
        temporal = [
            "transaction_hour", "transaction_day_of_week",
            "transaction_day", "transaction_month", "transaction_year",
        ]
        for col in temporal:
            assert col in result.columns, f"Missing temporal column: {col}"

    def test_std_is_zero_for_single_transaction_customer(self, sample_transactions):
        single = sample_transactions[sample_transactions["CustomerId"] == "C1"].iloc[:1].copy()
        result = engineer_features(single)
        assert result["std_transaction_amount"].iloc[0] == 0.0

    def test_raises_type_error_for_non_dataframe(self):
        with pytest.raises(TypeError):
            engineer_features("not a dataframe")

    def test_raises_key_error_for_missing_timestamp(self, sample_transactions):
        df = sample_transactions.drop(columns=["TransactionStartTime"])
        with pytest.raises(KeyError):
            engineer_features(df)


# ── Tests: compute_rfm ───────────────────────────────────────────────────────

class TestComputeRFM:

    def test_returns_one_row_per_customer(self, sample_transactions):
        result = compute_rfm(sample_transactions)
        assert len(result) == sample_transactions["CustomerId"].nunique()

    def test_rfm_columns_present(self, sample_transactions):
        result = compute_rfm(sample_transactions)
        for col in ["recency", "frequency", "monetary"]:
            assert col in result.columns

    def test_recency_is_non_negative(self, sample_transactions):
        result = compute_rfm(sample_transactions)
        assert (result["recency"] >= 0).all()

    def test_frequency_matches_transaction_count(self, sample_transactions):
        result = compute_rfm(sample_transactions)
        expected_counts = sample_transactions.groupby("CustomerId").size()
        for _, row in result.iterrows():
            assert row["frequency"] == expected_counts[row["CustomerId"]]

    def test_snapshot_date_override(self, sample_transactions):
        snapshot = pd.Timestamp("2019-03-01")
        result = compute_rfm(sample_transactions, snapshot_date=snapshot)
        assert result["recency"].min() >= 0

    def test_raises_key_error_for_missing_columns(self, sample_transactions):
        with pytest.raises(KeyError):
            compute_rfm(sample_transactions.drop(columns=["Value"]),
                        amount_col="Value")


# ── Tests: assign_high_risk_label ────────────────────────────────────────────

class TestAssignHighRiskLabel:

    def test_is_high_risk_column_created(self, sample_rfm):
        result = assign_high_risk_label(sample_rfm)
        assert "is_high_risk" in result.columns

    def test_binary_label_only_zero_and_one(self, sample_rfm):
        result = assign_high_risk_label(sample_rfm)
        assert set(result["is_high_risk"].unique()).issubset({0, 1})

    def test_cluster_column_created(self, sample_rfm):
        result = assign_high_risk_label(sample_rfm)
        assert "cluster" in result.columns

    def test_n_clusters_equals_three(self, sample_rfm):
        result = assign_high_risk_label(sample_rfm, n_clusters=3)
        assert result["cluster"].nunique() == 3

    def test_reproducibility_with_same_random_state(self, sample_rfm):
        r1 = assign_high_risk_label(sample_rfm.copy(), random_state=42)
        r2 = assign_high_risk_label(sample_rfm.copy(), random_state=42)
        pd.testing.assert_series_equal(r1["is_high_risk"], r2["is_high_risk"])

    def test_raises_value_error_for_n_clusters_less_than_2(self, sample_rfm):
        with pytest.raises(ValueError):
            assign_high_risk_label(sample_rfm, n_clusters=1)

    def test_raises_key_error_for_missing_rfm_columns(self):
        df = pd.DataFrame({"CustomerId": ["C1"], "recency": [10]})
        with pytest.raises(KeyError):
            assign_high_risk_label(df)


# ── Tests: get_data_overview ─────────────────────────────────────────────────

class TestGetDataOverview:

    def test_returns_one_row_per_column(self, sample_transactions):
        result = get_data_overview(sample_transactions)
        assert len(result) == len(sample_transactions.columns)

    def test_expected_columns_in_output(self, sample_transactions):
        result = get_data_overview(sample_transactions)
        for col in ["column", "dtype", "null_count", "null_pct", "unique_count"]:
            assert col in result.columns

    def test_null_count_is_zero_for_complete_data(self, sample_transactions):
        result = get_data_overview(sample_transactions)
        assert result["null_count"].sum() == 0

    def test_raises_type_error_for_non_dataframe(self):
        with pytest.raises(TypeError):
            get_data_overview([1, 2, 3])


# ── Tests: get_outlier_report ─────────────────────────────────────────────────

class TestGetOutlierReport:

    def test_returns_expected_columns(self):
        df = pd.DataFrame({"amount": list(range(100))})
        result = get_outlier_report(df, ["amount"])
        for col in ["outlier_count", "outlier_pct", "lower_fence", "upper_fence"]:
            assert col in result.columns

    def test_detects_known_outliers(self):
        df = pd.DataFrame({"amount": [1] * 98 + [9999, -9999]})
        result = get_outlier_report(df, ["amount"])
        assert result.loc["amount", "outlier_count"] >= 2

    def test_raises_key_error_for_missing_column(self):
        df = pd.DataFrame({"amount": [1, 2, 3]})
        with pytest.raises(KeyError):
            get_outlier_report(df, ["nonexistent"])

    def test_raises_value_error_for_non_numeric(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        with pytest.raises(ValueError):
            get_outlier_report(df, ["cat"])


# ── Tests: build_model_dataset ───────────────────────────────────────────────

class TestBuildModelDataset:

    def test_is_high_risk_column_present_in_output(self):
        processed = pd.DataFrame({
            "CustomerId": ["C1", "C2", "C3"],
            "feature_a": [1.0, 2.0, 3.0],
        })
        rfm = pd.DataFrame({
            "CustomerId": ["C1", "C2", "C3"],
            "is_high_risk": [0, 1, 0],
        })
        result = build_model_dataset(processed, rfm)
        assert "is_high_risk" in result.columns

    def test_row_count_preserved_after_merge(self):
        processed = pd.DataFrame({
            "CustomerId": ["C1", "C2", "C3"],
            "feature_a": [1.0, 2.0, 3.0],
        })
        rfm = pd.DataFrame({
            "CustomerId": ["C1", "C2", "C3"],
            "is_high_risk": [0, 1, 0],
        })
        result = build_model_dataset(processed, rfm)
        assert len(result) == 3

    def test_raises_key_error_if_is_high_risk_missing(self):
        processed = pd.DataFrame({"CustomerId": ["C1"], "f": [1.0]})
        rfm = pd.DataFrame({"CustomerId": ["C1"], "recency": [10]})
        with pytest.raises(KeyError):
            build_model_dataset(processed, rfm)