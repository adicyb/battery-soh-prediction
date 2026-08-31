"""
preprocessing.py

Cleans standardized battery data and computes State of Health (SoH).

Supported target definitions
----------------------------
1. Capacity available:
       SoH(n) = Capacity(n) / Capacity_initial * 100

2. SoH already supplied:
       The supplied SoH values are preserved and normalized to
       percentage units where possible.

The earliest valid capacity is used as the reference capacity when
capacity-based SoH must be calculated.

Important:
This module preserves the original experimental cycle numbers.
It does NOT silently renumber cycles after removing invalid rows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the standardized per-cycle battery dataframe.

    Required:
        cycle

    Target:
        capacity_ah OR soh

    Behaviour:
        - preserve original cycle numbers
        - remove invalid cycle values
        - remove rows with neither capacity nor SoH
        - remove non-positive capacities
        - sort chronologically
        - remove duplicate cycle numbers
    """
    df = df.copy()

    if "cycle" not in df.columns:
        raise ValueError(
            "Preprocessing requires a 'cycle' column."
        )

    if "capacity_ah" not in df.columns and "soh" not in df.columns:
        raise ValueError(
            "Preprocessing requires either 'capacity_ah' or 'soh'."
        )

    # ---------------------------------------------------------------
    # Cycle cleaning
    # ---------------------------------------------------------------

    df["cycle"] = pd.to_numeric(
        df["cycle"],
        errors="coerce",
    )

    df = df.dropna(subset=["cycle"])

    if df.empty:
        raise ValueError(
            "No valid numeric cycle values remain after cleaning."
        )

    # Preserve the experimental cycle numbering.
    # We do NOT renumber cycles here.
    df["cycle"] = df["cycle"].astype(int)

    df = (
        df.sort_values("cycle")
        .drop_duplicates(
            subset=["cycle"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Capacity cleaning
    # ---------------------------------------------------------------

    if "capacity_ah" in df.columns:
        df["capacity_ah"] = pd.to_numeric(
            df["capacity_ah"],
            errors="coerce",
        )

        # Invalid/non-positive capacity cannot be used to calculate SoH.
        if "soh" not in df.columns:
            df = df.dropna(subset=["capacity_ah"])
            df = df[df["capacity_ah"] > 0]
        else:
            # If direct SoH is available, retain rows even if capacity is
            # missing because SoH can serve as the target.
            df.loc[
                df["capacity_ah"] <= 0,
                "capacity_ah",
            ] = np.nan

    # ---------------------------------------------------------------
    # Direct SoH cleaning
    # ---------------------------------------------------------------

    if "soh" in df.columns:
        df["soh"] = pd.to_numeric(
            df["soh"],
            errors="coerce",
        )

        # Ignore obviously invalid percentage values.
        df.loc[
            (df["soh"] < 0) | (df["soh"] > 100),
            "soh",
        ] = np.nan

    # ---------------------------------------------------------------
    # Final target check
    # ---------------------------------------------------------------

    has_valid_capacity = (
        "capacity_ah" in df.columns
        and df["capacity_ah"].notna().any()
    )

    has_valid_soh = (
        "soh" in df.columns
        and df["soh"].notna().any()
    )

    if not has_valid_capacity and not has_valid_soh:
        raise ValueError(
            "No valid battery capacity or SoH values remain after cleaning."
        )

    # If both columns exist, a row only needs one valid target.
    if has_valid_capacity and has_valid_soh:
        df = df[
            df["capacity_ah"].notna()
            | df["soh"].notna()
        ].copy()
    elif has_valid_capacity:
        df = df[df["capacity_ah"].notna()].copy()
    else:
        df = df[df["soh"].notna()].copy()

    if len(df) < 3:
        raise ValueError(
            "At least 3 valid cycles are required after preprocessing."
        )

    return df.reset_index(drop=True)


def compute_soh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute or preserve State of Health.

    If SoH is already provided:
        preserve it.

    Otherwise, calculate:
        SoH = Capacity / Initial Capacity * 100

    The initial capacity is the capacity at the earliest cycle with
    a valid capacity measurement.
    """
    df = df.copy()

    # ---------------------------------------------------------------
    # Case 1: SoH already supplied
    # ---------------------------------------------------------------

    if "soh" in df.columns and df["soh"].notna().any():

        df["soh"] = pd.to_numeric(
            df["soh"],
            errors="coerce",
        )

        # Normalize fractional SoH if a dataset supplies values such as
        # 0.98 rather than 98%.
        valid_soh = df["soh"].dropna()

        if not valid_soh.empty and valid_soh.max() <= 1.5:
            df["soh"] = df["soh"] * 100.0

        df["soh"] = df["soh"].clip(
            lower=0.0,
            upper=100.0,
        )

        # Add initial capacity if available for reporting.
        if "capacity_ah" in df.columns:
            valid_capacity = df["capacity_ah"].dropna()

            if not valid_capacity.empty:
                initial_capacity = float(
                    valid_capacity.iloc[0]
                )
                df["initial_capacity_ah"] = initial_capacity

        return df

    # ---------------------------------------------------------------
    # Case 2: Calculate SoH from capacity
    # ---------------------------------------------------------------

    if "capacity_ah" not in df.columns:
        raise ValueError(
            "Cannot calculate SoH because the dataset contains neither "
            "a valid 'soh' column nor 'capacity_ah'."
        )

    valid_capacity = df["capacity_ah"].dropna()

    if valid_capacity.empty:
        raise ValueError(
            "Cannot calculate SoH because no valid capacity values exist."
        )

    initial_capacity = float(
        valid_capacity.iloc[0]
    )

    if initial_capacity <= 0:
        raise ValueError(
            f"Initial capacity must be positive; got "
            f"{initial_capacity} Ah."
        )

    df["initial_capacity_ah"] = initial_capacity

    df["soh"] = (
        df["capacity_ah"]
        / initial_capacity
        * 100.0
    )

    # SoH should not exceed 100% when using the earliest valid capacity
    # as the reference.
    df["soh"] = df["soh"].clip(
        lower=0.0,
        upper=100.0,
    )

    return df


def preprocess(
    df: pd.DataFrame,
    data_source: str,
    battery_id: str,
    save: bool = True,
    filename: str = "processed_battery_data.csv",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:

        clean → compute/preserve SoH → metadata → save
    """
    df = clean_data(df)

    df = compute_soh(df)

    df["battery_id"] = battery_id
    df["data_source"] = data_source

    # ---------------------------------------------------------------
    # Reorder columns for readability.
    # ---------------------------------------------------------------

    preferred_order = [
        "cycle",
        "capacity_ah",
        "initial_capacity_ah",
        "soh",
        "voltage_mean",
        "voltage_min",
        "voltage_max",
        "current_mean",
        "current_min",
        "temperature_mean",
        "temperature_max",
        "battery_id",
        "data_source",
    ]

    cols = (
        [column for column in preferred_order if column in df.columns]
        + [
            column
            for column in df.columns
            if column not in preferred_order
        ]
    )

    df = df[cols]

    if save:
        os.makedirs(
            PROCESSED_DIR,
            exist_ok=True,
        )

        out_path = os.path.join(
            PROCESSED_DIR,
            filename,
        )

        df.to_csv(
            out_path,
            index=False,
        )

        print(
            f"[preprocessing] Saved processed data -> {out_path}"
        )

    return df


if __name__ == "__main__":
    from data_loader import load_battery_data

    raw_df, source, message = load_battery_data()

    print(message)

    processed = preprocess(
        raw_df,
        source,
        str(raw_df["battery_id"].iloc[0]),
    )

    print(processed.head())

    if "initial_capacity_ah" in processed.columns:
        print(
            "\nInitial capacity reference: "
            f"{processed['initial_capacity_ah'].iloc[0]:.4f} Ah"
        )

    print(
        f"\nFinal SoH at last measured cycle "
        f"({processed['cycle'].iloc[-1]}): "
        f"{processed['soh'].iloc[-1]:.2f}%"
    )