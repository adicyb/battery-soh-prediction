"""
feature_engineering.py

Builds the feature matrix X and target vector y (SoH) used by the ML model.

Only features that actually exist in the processed dataframe are used --
nothing is fabricated. Rolling/derived features are computed only from
columns that are present, so this works identically whether the underlying
data source is real NASA measurements or the synthetic fallback (both
produce the same column schema).
"""

import numpy as np
import pandas as pd

# Columns we will use as ML features IF they exist in the dataframe.
CANDIDATE_FEATURE_COLUMNS = [
    "cycle",
    "voltage_mean", "voltage_min", "voltage_max",
    "current_mean", "current_min",
    "temperature_mean", "temperature_max",
]

TARGET_COLUMN = "soh"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features computed only from available columns."""
    df = df.copy()

    if "voltage_max" in df.columns and "voltage_min" in df.columns:
        df["voltage_range"] = df["voltage_max"] - df["voltage_min"]

    if "temperature_max" in df.columns and "temperature_mean" in df.columns:
        df["temperature_rise"] = df["temperature_max"] - df["temperature_mean"]

    return df


def get_feature_columns(df: pd.DataFrame):
    """Return the list of feature columns that actually exist in df."""
    base = [c for c in CANDIDATE_FEATURE_COLUMNS if c in df.columns]
    extra = [c for c in ["voltage_range", "temperature_rise"] if c in df.columns]
    return base + extra


def build_features(df: pd.DataFrame):
    """
    Returns:
        X (DataFrame): feature matrix
        y (Series): target SoH values
        feature_cols (list): names of columns used
    """
    df = add_derived_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y, feature_cols, df


if __name__ == "__main__":
    from data_loader import load_battery_data
    from preprocessing import preprocess

    raw_df, source, message = load_battery_data()
    processed = preprocess(raw_df, source, raw_df["battery_id"].iloc[0], save=False)
    X, y, feature_cols, full_df = build_features(processed)
    print("Feature columns used:", feature_cols)
    print(X.head())
    print("\nTarget (SoH) preview:")
    print(y.head())
