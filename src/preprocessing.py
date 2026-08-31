"""
preprocessing.py

Cleans the raw per-cycle battery dataframe and computes State of Health (SoH).

SoH definition used (standard definition, as required by the project):

    SoH(n) = ( Capacity(n) / Capacity_initial ) * 100

Capacity_initial is defined as the capacity measured at the EARLIEST valid
cycle in the dataset (cycle 1, or the first cycle with a valid capacity
reading), which is the standard reference-capacity convention used across
NASA-PCoE-based battery-SoH literature. This is a reference-capacity
definition, not the manufacturer's nameplate rated capacity, because the
manufacturer's original as-new capacity is not separately provided in the
NASA dataset.
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and sort by cycle number."""
    df = df.copy()
    df = df.dropna(subset=["capacity_ah"])
    df = df[df["capacity_ah"] > 0]
    df = df.sort_values("cycle").reset_index(drop=True)
    # Re-number cycles contiguously starting at 1 in case any were dropped
    df["cycle"] = np.arange(1, len(df) + 1)
    return df


def compute_soh(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SoH (%) using the earliest valid capacity as reference."""
    df = df.copy()
    initial_capacity = df["capacity_ah"].iloc[0]
    df["initial_capacity_ah"] = initial_capacity
    df["soh"] = (df["capacity_ah"] / initial_capacity) * 100.0
    df["soh"] = df["soh"].clip(upper=100.0)  # SoH cannot exceed 100% of its own reference
    return df


def preprocess(df: pd.DataFrame, data_source: str, battery_id: str,
               save: bool = True, filename: str = "processed_battery_data.csv") -> pd.DataFrame:
    """Full preprocessing pipeline: clean -> compute SoH -> save CSV."""
    df = clean_data(df)
    df = compute_soh(df)
    df["battery_id"] = battery_id
    df["data_source"] = data_source

    # Reorder columns for readability
    preferred_order = [
        "cycle", "capacity_ah", "initial_capacity_ah", "soh",
        "voltage_mean", "voltage_min", "voltage_max",
        "current_mean", "current_min",
        "temperature_mean", "temperature_max",
        "battery_id", "data_source",
    ]
    cols = [c for c in preferred_order if c in df.columns] + \
           [c for c in df.columns if c not in preferred_order]
    df = df[cols]

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, filename)
        df.to_csv(out_path, index=False)
        print(f"[preprocessing] Saved processed data -> {out_path}")

    return df


if __name__ == "__main__":
    from data_loader import load_battery_data
    raw_df, source, message = load_battery_data()
    print(message)
    processed = preprocess(raw_df, source, raw_df["battery_id"].iloc[0])
    print(processed.head())
    print(f"\nInitial capacity reference: {processed['initial_capacity_ah'].iloc[0]:.4f} Ah")
    print(f"Final SoH at last measured cycle ({processed['cycle'].iloc[-1]}): "
          f"{processed['soh'].iloc[-1]:.2f}%")
