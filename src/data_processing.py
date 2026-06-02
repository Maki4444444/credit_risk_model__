"""
data_processing.py
------------------
Feature engineering pipeline for the Bati Bank Credit Risk Scoring project.

Transforms raw Xente transaction-level data into a model-ready DataFrame by:
    1. Loading and validating raw data
    2. Capping outliers
    3. Engineering aggregate and temporal features
    4. Encoding categorical variables (one-hot or WoE)
    5. Normalizing/standardizing numerical features
    6. Computing Weight of Evidence (WoE) and Information Value (IV)

Deliverable (Task 3): A single fitted sklearn Pipeline object that produces
a model-ready DataFrame from raw input, exposed via fit_transform_pipeline().

Used by:
    - notebooks/eda.ipynb          (Task 2 — EDA utility functions)
    - src/train.py                 (Task 5)
    - src/predict.py               (Task 5)
    - src/api/main.py              (Task 6)
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ── Column constants (importable by notebooks) ────────────────────────────────

# Columns to drop: zero-variance and pure identifier columns
COLUMNS_TO_DROP = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CountryCode",
    "CurrencyCode",
    "ProductId",
]

# Categorical columns to encode
CATEGORICAL_COLS = ["ProviderId", "ProductCategory", "ChannelId"]

# WoE columns (high-signal categoricals transformed to log-odds)
WOE_COLUMNS = ["ProductCategory", "ChannelId", "ProviderId"]

# Numerical columns to scale (after aggregation)
NUMERICAL_COLS = [
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


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading
# ════════════════════════════════════════════════════════════════════════════

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Xente transaction CSV into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with TransactionStartTime parsed as datetime.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is empty or cannot be parsed as CSV.
    """
    if not os.path.exists(filepath):
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_csv(filepath, parse_dates=["TransactionStartTime"])
        logger.info(
            "Loaded %d records with %d columns.",
            df.shape[0], df.shape[1],
        )
    except Exception as exc:
        logger.error("Failed to read CSV: %s", exc)
        raise ValueError(
            f"Could not parse CSV at '{filepath}': {exc}"
        ) from exc

    if df.empty:
        raise ValueError(f"File loaded but is empty: {filepath}")

    return df


# Keep load_data as an alias for backward compatibility
load_data = load_raw_data


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load a previously processed CSV produced by fit_transform_pipeline().

    Parameters
    ----------
    filepath : str
        Path to the processed CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    if not os.path.exists(filepath):
        logger.error("Processed file not found: %s", filepath)
        raise FileNotFoundError(f"Processed file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(
        "Loaded processed data from: %s — %d records with %d columns.",
        filepath, df.shape[0], df.shape[1],
    )
    if df.empty:
        raise ValueError(f"File loaded but is empty: {filepath}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Outlier Capping
# ════════════════════════════════════════════════════════════════════════════

def cap_outliers(
    df: pd.DataFrame,
    cols: list,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Cap outliers in the specified columns at given percentile thresholds.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str
        Numerical columns to cap.
    lower_pct : float
        Lower percentile threshold (default 0.01).
    upper_pct : float
        Upper percentile threshold (default 0.99).

    Returns
    -------
    pd.DataFrame
        DataFrame with capped values. Does NOT modify the input in place.

    Raises
    ------
    TypeError
        If df is not a DataFrame or cols is not a list.
    KeyError
        If any column in cols does not exist in df.
    ValueError
        If any specified column is non-numerical.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    if not isinstance(cols, list):
        raise TypeError(f"'cols' must be a list, got {type(cols)}")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    non_numeric = [
        c for c in cols if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numerical columns cannot be capped: {non_numeric}"
        )

    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower=lo, upper=hi)
        logger.info(
            "Capped '%s' at [%.4f, %.4f].", col, lo, hi
        )
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Engineering
# ════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the raw DataFrame:
      - Extract temporal features from TransactionStartTime
      - Aggregate transaction-level records to customer-level features
      - Drop zero-variance and identifier columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw Xente transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Customer-level engineered feature DataFrame.

    Raises
    ------
    TypeError
        If input is not a DataFrame.
    KeyError
        If required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    df = df.copy()

    # 1. Temporal features
    ts_col = "TransactionStartTime"
    if ts_col not in df.columns:
        raise KeyError(f"Required column '{ts_col}' not found.")

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception as exc:
            raise ValueError(
                f"Cannot parse '{ts_col}' as datetime: {exc}"
            ) from exc

    df["transaction_hour"] = df[ts_col].dt.hour
    df["transaction_day_of_week"] = df[ts_col].dt.dayofweek
    df["transaction_day"] = df[ts_col].dt.day
    df["transaction_month"] = df[ts_col].dt.month
    df["transaction_year"] = df[ts_col].dt.year

    # 2. Aggregate to customer level
    required = ["CustomerId", "Amount", "Value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for aggregation: {missing}")

    agg = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std"),
            total_value=("Value", "sum"),
            avg_value=("Value", "mean"),
            transaction_hour=("transaction_hour", "mean"),
            transaction_day_of_week=("transaction_day_of_week", "mean"),
            transaction_day=("transaction_day", "mean"),
            transaction_month=("transaction_month", "mean"),
            transaction_year=("transaction_year", "mean"),
            # Keep most-frequent categorical value per customer
            ProviderId=("ProviderId", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            ProductCategory=("ProductCategory", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            ChannelId=("ChannelId", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            PricingStrategy=("PricingStrategy", "mean"),
            FraudResult=("FraudResult", "max"),
        )
        .reset_index()
    )

    # Fill NaN std (customers with a single transaction)
    agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)

    new_cols = [
        "total_transaction_amount", "avg_transaction_amount",
        "transaction_count", "std_transaction_amount",
        "total_value", "avg_value",
        "transaction_hour", "transaction_day_of_week",
        "transaction_day", "transaction_month", "transaction_year",
    ]
    logger.info(
        "Feature engineering complete. New columns: %s",
        ", ".join(new_cols),
    )
    return agg


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WoE / IV
# ════════════════════════════════════════════════════════════════════════════

def compute_all_iv(
    df: pd.DataFrame,
    target_col: str = "FraudResult",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute Information Value (IV) for all numerical features in df
    using equal-frequency binning.

    IV interpretation:
        < 0.02  : Useless
        0.02-0.1: Weak
        0.1-0.3 : Medium
        > 0.3   : Strong
        > 0.5   : Very strong (check for data leakage)

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Binary target column name (default: 'FraudResult').
    n_bins : int
        Number of bins for equal-frequency binning.

    Returns
    -------
    pd.DataFrame
        Columns: feature, iv, predictive_power — sorted by IV descending.

    Raises
    ------
    KeyError
        If target_col is not in df.
    ValueError
        If target_col is not binary.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    if set(df[target_col].dropna().unique()) - {0, 1}:
        raise ValueError(
            f"Target '{target_col}' must be binary (0/1)."
        )

    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    total_events = df[target_col].sum()
    total_non_events = len(df) - total_events

    if total_events == 0 or total_non_events == 0:
        raise ValueError(
            "Target column has only one class — IV cannot be computed."
        )

    rows = []
    for col in num_cols:
        try:
            bins = pd.qcut(df[col], q=n_bins, duplicates="drop")
        except Exception:
            continue

        grouped = df.groupby(bins, observed=False)[target_col].agg(
            ["sum", "count"]
        )
        grouped.columns = ["events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]
        grouped["dist_events"] = (grouped["events"] + 0.5) / total_events
        grouped["dist_non_events"] = (grouped["non_events"] + 0.5) / total_non_events
        grouped["woe"] = np.log(
            grouped["dist_non_events"] / grouped["dist_events"]
        )
        iv = ((grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]).sum()

        if iv < 0.02:
            power = "Useless"
        elif iv < 0.1:
            power = "Weak"
        elif iv < 0.3:
            power = "Medium"
        elif iv < 0.5:
            power = "Strong"
        else:
            power = "Very Strong"

        rows.append({"feature": col, "iv": round(iv, 6), "predictive_power": power})

    iv_df = (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    return iv_df


def encode_woe(
    df: pd.DataFrame,
    columns: list,
    target_col: str = "FraudResult",
) -> tuple:
    """
    Apply Weight of Evidence (WoE) encoding to categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Categorical columns to encode.
    target_col : str
        Binary target column name.

    Returns
    -------
    tuple : (encoded_df, woe_maps)
        encoded_df  — DataFrame with original columns replaced by {col}_woe
        woe_maps    — dict mapping column name -> {category: woe_value}

    Raises
    ------
    KeyError
        If any column or target is missing.
    ValueError
        If target is not binary.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    total_events = df[target_col].sum()
    total_non_events = len(df) - total_events

    if total_events == 0 or total_non_events == 0:
        raise ValueError(
            "Target column has only one class — WoE cannot be computed."
        )

    df = df.copy()
    woe_maps = {}

    for col in columns:
        grouped = df.groupby(col)[target_col].agg(["sum", "count"])
        grouped.columns = ["events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]
        grouped["dist_events"] = (grouped["events"] + 0.5) / total_events
        grouped["dist_non_events"] = (grouped["non_events"] + 0.5) / total_non_events
        grouped["woe"] = np.log(
            grouped["dist_non_events"] / grouped["dist_events"]
        )

        woe_map = grouped["woe"].to_dict()
        woe_maps[col] = woe_map
        df[f"{col}_woe"] = df[col].map(woe_map).fillna(0)
        df = df.drop(columns=[col])
        logger.info("WoE encoded '%s' -> '%s_woe'.", col, col)

    return df, woe_maps


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Custom sklearn Transformers
# ════════════════════════════════════════════════════════════════════════════

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from a datetime column."""

    def __init__(self, timestamp_col: str = "TransactionStartTime"):
        self.timestamp_col = timestamp_col

    def fit(self, X: pd.DataFrame, y=None):
        if self.timestamp_col not in X.columns:
            raise KeyError(f"Column '{self.timestamp_col}' not found.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X[self.timestamp_col]):
            X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])
        X["transaction_hour"] = X[self.timestamp_col].dt.hour
        X["transaction_day_of_week"] = X[self.timestamp_col].dt.dayofweek
        X["transaction_day"] = X[self.timestamp_col].dt.day
        X["transaction_month"] = X[self.timestamp_col].dt.month
        X["transaction_year"] = X[self.timestamp_col].dt.year
        X = X.drop(columns=[self.timestamp_col])
        return X


class AggregateFeatureBuilder(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level records to customer-level features."""

    def __init__(
        self,
        customer_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
    ):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.value_col = value_col

    def fit(self, X: pd.DataFrame, y=None):
        required = [self.customer_col, self.amount_col, self.value_col]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        agg = (
            X.groupby(self.customer_col)
            .agg(
                total_transaction_amount=(self.amount_col, "sum"),
                avg_transaction_amount=(self.amount_col, "mean"),
                transaction_count=(self.amount_col, "count"),
                std_transaction_amount=(self.amount_col, "std"),
                total_value=(self.value_col, "sum"),
                avg_value=(self.value_col, "mean"),
                transaction_hour=("transaction_hour", "mean"),
                transaction_day_of_week=("transaction_day_of_week", "mean"),
                transaction_day=("transaction_day", "mean"),
                transaction_month=("transaction_month", "mean"),
                transaction_year=("transaction_year", "mean"),
                ProviderId=("ProviderId", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
                ProductCategory=("ProductCategory", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
                ChannelId=("ChannelId", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
                PricingStrategy=("PricingStrategy", "mean"),
                FraudResult=("FraudResult", "max"),
            )
            .reset_index()
        )
        agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)
        return agg


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot or label encode categorical columns."""

    def __init__(self, columns: list, strategy: str = "onehot"):
        self.columns = columns
        self.strategy = strategy
        self._label_encoders = {}

    def fit(self, X: pd.DataFrame, y=None):
        if self.strategy not in ("onehot", "label"):
            raise ValueError(f"strategy must be 'onehot' or 'label', got '{self.strategy}'")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        if self.strategy == "label":
            for col in self.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self._label_encoders[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.strategy == "onehot":
            X = pd.get_dummies(X, columns=self.columns, drop_first=True)
        elif self.strategy == "label":
            for col in self.columns:
                le = self._label_encoders[col]
                X[col] = X[col].astype(str).map(
                    lambda val, le=le: (
                        le.transform([val])[0] if val in le.classes_ else -1
                    )
                )
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop specified columns (silently skips missing ones)."""

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.columns if c in X.columns]
        return X.drop(columns=cols_to_drop)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Replace categorical columns with their WoE values."""

    def __init__(self, columns: list, target_col: str = "FraudResult"):
        self.columns = columns
        self.target_col = target_col
        self._woe_maps = {}
        self.iv_summary_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        total_events = X[self.target_col].sum()
        total_non_events = len(X) - total_events
        for col in self.columns:
            grouped = X.groupby(col)[self.target_col].agg(["sum", "count"])
            grouped.columns = ["events", "total"]
            grouped["non_events"] = grouped["total"] - grouped["events"]
            grouped["dist_events"] = (grouped["events"] + 0.5) / total_events
            grouped["dist_non_events"] = (grouped["non_events"] + 0.5) / total_non_events
            grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
            grouped["iv"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]
            self._woe_maps[col] = grouped["woe"].to_dict()
            self.iv_summary_[col] = grouped["iv"].sum()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[f"{col}_woe"] = X[col].map(self._woe_maps.get(col, {})).fillna(0)
            X = X.drop(columns=[col])
        return X

    def get_iv_summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(self.iv_summary_, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
        )


class NumericalScaler(BaseEstimator, TransformerMixin):
    """Scale numerical columns using StandardScaler or MinMaxScaler."""

    def __init__(self, columns: list, strategy: str = "standard"):
        self.columns = columns
        self.strategy = strategy
        self._scaler = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in self.columns if c in X.columns]
        self._scaler = StandardScaler() if self.strategy == "standard" else MinMaxScaler()
        self._scaler.fit(X[cols])
        self._fitted_cols = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._fitted_cols] = self._scaler.transform(X[self._fitted_cols])
        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Impute missing values with median/mean/most_frequent."""

    def __init__(self, columns: list, strategy: str = "median"):
        self.columns = columns
        self.strategy = strategy
        self._imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in self.columns if c in X.columns]
        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X[cols])
        self._fitted_cols = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self._fitted_cols] = self._imputer.transform(X[self._fitted_cols])
        return X

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Pipeline Builder
# ════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    target_col: str = "FraudResult",
    encoding_strategy: str = "onehot",
    scaling_strategy: str = "standard",
    use_woe: bool = False,
) -> Pipeline:
    """
    Build and return the full feature engineering Pipeline.

    Steps:
        1. temporal   — extract datetime features, drop timestamp column
        2. drop_cols  — remove zero-variance and identifier columns
        3. aggregate  — aggregate transactions to customer level
        4. impute     — median imputation for post-aggregation nulls
        5. encode/woe — categorical encoding (one-hot/label or WoE)
        6. scale      — numerical scaling

    Parameters
    ----------
    target_col : str
    encoding_strategy : str — 'onehot' or 'label'
    scaling_strategy  : str — 'standard' or 'minmax'
    use_woe : bool — if True, use WoE instead of categorical encoding

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = [
        ("temporal", TemporalFeatureExtractor("TransactionStartTime")),
        ("drop_cols", DropColumns(columns=COLUMNS_TO_DROP)),
        ("aggregate", AggregateFeatureBuilder("CustomerId", "Amount", "Value")),
        ("impute", MissingValueImputer(["std_transaction_amount"], strategy="median")),
    ]

    if use_woe:
        steps.append(("woe", WoETransformer(WOE_COLUMNS, target_col)))
    else:
        steps.append(("encode", CategoricalEncoder(CATEGORICAL_COLS, encoding_strategy)))

    steps.append(("scale", NumericalScaler(NUMERICAL_COLS, scaling_strategy)))

    return Pipeline(steps=steps)


def fit_transform_pipeline(
    df: pd.DataFrame,
    target_col: str = "FraudResult",
    encoding_strategy: str = "onehot",
    scaling_strategy: str = "standard",
    use_woe: bool = False,
) -> tuple:
    """
    Build, fit, and apply the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Xente transaction DataFrame.
    target_col : str
    encoding_strategy : str
    scaling_strategy  : str
    use_woe : bool

    Returns
    -------
    tuple : (pipeline, df_ready)
        pipeline  — fitted sklearn Pipeline
        df_ready  — transformed, model-ready DataFrame
    """
    pipeline = build_pipeline(target_col, encoding_strategy, scaling_strategy, use_woe)
    df_ready = pipeline.fit_transform(df)
    logger.info("Pipeline fit complete. Output shape: %s", df_ready.shape)
    return pipeline, df_ready


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EDA Utility Functions (used by notebooks/eda.ipynb)
# ════════════════════════════════════════════════════════════════════════════

def get_data_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, null count, null %, unique count, sample value."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    rows = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2),
            "unique_count": df[col].nunique(),
            "sample_value": df[col].dropna().iloc[0] if df[col].notna().any() else None,
        })
    return pd.DataFrame(rows)


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted DataFrame of missing value counts and percentages."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    return (
        pd.DataFrame({
            "null_count": df.isnull().sum(),
            "null_pct": (df.isnull().mean() * 100).round(2),
        })
        .sort_values("null_count", ascending=False)
    )


def get_outlier_report(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """IQR-based outlier counts and fence values for specified columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")
    if not isinstance(columns, list):
        raise TypeError(f"'columns' must be a list, got {type(columns)}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numerical columns: {non_numeric}")
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "column": col,
            "outlier_count": n_out,
            "outlier_pct": round(n_out / len(df) * 100, 2),
            "lower_fence": round(lower, 2),
            "upper_fence": round(upper, 2),
        })
    return pd.DataFrame(rows).set_index("column")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Proxy Target Variable Engineering (Task 4)
# ════════════════════════════════════════════════════════════════════════════

from sklearn.cluster import KMeans


def compute_rfm(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp = None,
    customer_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    amount_col: str = "Value",
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) metrics per customer
    from raw transaction-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Xente transaction DataFrame.
    snapshot_date : pd.Timestamp, optional
        Reference date for recency calculation.
        Defaults to one day after the latest transaction in the dataset.
    customer_col : str
        Column identifying the customer (default: 'CustomerId').
    date_col : str
        Datetime column for recency (default: 'TransactionStartTime').
    amount_col : str
        Monetary value column (default: 'Value').

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        CustomerId, recency, frequency, monetary

    Raises
    ------
    TypeError
        If df is not a DataFrame.
    KeyError
        If required columns are missing.
    ValueError
        If date_col cannot be parsed as datetime.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    required = [customer_col, date_col, amount_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for RFM: {missing}")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as exc:
            raise ValueError(
                f"Cannot parse '{date_col}' as datetime: {exc}"
            ) from exc

    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
        logger.info("Snapshot date set to: %s", snapshot_date)

    rfm = (
        df.groupby(customer_col)
        .agg(
            recency=(date_col, lambda x: (snapshot_date - x.max()).days),
            frequency=(date_col, "count"),
            monetary=(amount_col, "sum"),
        )
        .reset_index()
    )

    logger.info(
        "RFM computed for %d customers. Snapshot date: %s",
        len(rfm), snapshot_date,
    )
    return rfm


def assign_high_risk_label(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster customers on scaled RFM features using K-Means and label
    the highest-risk cluster as is_high_risk = 1.

    The high-risk cluster is identified as the one with the lowest
    frequency and lowest monetary value — indicating disengaged customers
    who are least active and generate the least value, consistent with
    the behavioral finance logic that RFM disengagement precedes default.

    Parameters
    ----------
    rfm : pd.DataFrame
        Output of compute_rfm() with columns: CustomerId, recency,
        frequency, monetary.
    n_clusters : int
        Number of K-Means clusters (default: 3).
    random_state : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    pd.DataFrame
        rfm DataFrame with two new columns:
        - cluster     : integer cluster label (0, 1, or 2)
        - is_high_risk: binary label (1 = high risk, 0 = low risk)

    Raises
    ------
    TypeError
        If rfm is not a DataFrame.
    KeyError
        If required RFM columns are missing.
    ValueError
        If n_clusters is less than 2.
    """
    if not isinstance(rfm, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(rfm)}")

    required = ["recency", "frequency", "monetary"]
    missing = [c for c in required if c not in rfm.columns]
    if missing:
        raise KeyError(f"Missing RFM columns: {missing}")

    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    rfm = rfm.copy()

    # Scale RFM before clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify the high-risk cluster: lowest frequency + lowest monetary
    cluster_summary = (
        rfm.groupby("cluster")[["frequency", "monetary"]]
        .mean()
    )
    # Score each cluster: lower frequency and monetary = higher risk
    cluster_summary["risk_score"] = (
        cluster_summary["frequency"].rank(ascending=True)
        + cluster_summary["monetary"].rank(ascending=True)
    )
    high_risk_cluster = cluster_summary["risk_score"].idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    logger.info(
        "High-risk cluster identified: cluster %d "
        "(mean freq=%.2f, mean monetary=%.2f). "
        "High-risk customers: %d / %d (%.1f%%)",
        high_risk_cluster,
        cluster_summary.loc[high_risk_cluster, "frequency"],
        cluster_summary.loc[high_risk_cluster, "monetary"],
        rfm["is_high_risk"].sum(),
        len(rfm),
        rfm["is_high_risk"].mean() * 100,
    )
    return rfm


def build_model_dataset(
    processed_df: pd.DataFrame,
    rfm: pd.DataFrame,
    customer_col: str = "CustomerId",
) -> pd.DataFrame:
    """
    Merge the is_high_risk label from the RFM DataFrame into the
    processed feature DataFrame produced by fit_transform_pipeline().

    Parameters
    ----------
    processed_df : pd.DataFrame
        Output of fit_transform_pipeline() — customer-level feature matrix.
    rfm : pd.DataFrame
        Output of assign_high_risk_label() — contains is_high_risk column.
    customer_col : str
        Column to join on (default: 'CustomerId').

    Returns
    -------
    pd.DataFrame
        Merged DataFrame ready for model training, with is_high_risk
        as the target column.

    Raises
    ------
    KeyError
        If customer_col or is_high_risk column is missing.
    ValueError
        If the merge produces no rows.
    """
    if customer_col not in processed_df.columns:
        raise KeyError(
            f"'{customer_col}' not found in processed DataFrame."
        )
    if customer_col not in rfm.columns:
        raise KeyError(
            f"'{customer_col}' not found in RFM DataFrame."
        )
    if "is_high_risk" not in rfm.columns:
        raise KeyError(
            "'is_high_risk' column not found in RFM DataFrame. "
            "Run assign_high_risk_label() first."
        )

    label_df = rfm[[customer_col, "is_high_risk"]]
    merged = processed_df.merge(label_df, on=customer_col, how="inner")

    if merged.empty:
        raise ValueError(
            "Merge produced an empty DataFrame. "
            "Check that customer IDs match between processed_df and rfm."
        )

    logger.info(
        "Model dataset built: %d customers × %d features "
        "(high-risk rate: %.1f%%)",
        merged.shape[0],
        merged.shape[1],
        merged["is_high_risk"].mean() * 100,
    )
    return merged