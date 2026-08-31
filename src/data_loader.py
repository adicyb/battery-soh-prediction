"""
data_loader.py

Battery dataset loading and standardization for the SoH prediction project.

Supported inputs
----------------
1. NASA PCoE Li-ion Battery Aging .mat files
   - B0005
   - B0006
   - B0007
   - B0018

2. Generic CSV battery datasets

Canonical internal columns
--------------------------
The rest of the pipeline receives a standardized dataframe containing,
where available:

    cycle
    capacity_ah
    voltage_mean
    voltage_min
    voltage_max
    current_mean
    current_min
    temperature_mean
    temperature_max
    battery_id
    data_source

If a generic CSV contains a direct SoH column, it is also preserved as
"soh". If it contains capacity but not SoH, SoH is calculated later by the
preprocessing stage.

The generic CSV loader accepts common column-name aliases, for example:

    cycle / Cycle / Cycle_Number
    voltage / Voltage / Voltage_V
    current / Current / Current_A
    temperature / Temperature / Temp_C
    capacity / Capacity / Capacity_Ah
    soh / SoH / State_of_Health

The loader does NOT silently assume that arbitrary variables represent
battery measurements. If the required information is missing, it raises
a clear error.

NASA parsing remains compatible with the original B0005 implementation.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed"
)

NASA_BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_mat_files() -> dict[str, str]:
    """
    Look for supported NASA .mat files in data/raw/ and its subdirectories.
    """
    found: dict[str, str] = {}

    for battery_id in NASA_BATTERY_IDS:
        patterns = [
            os.path.join(RAW_DIR, f"{battery_id}.mat"),
            os.path.join(RAW_DIR, "**", f"{battery_id}.mat"),
        ]

        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found[battery_id] = matches[0]
                break

    return found


# ---------------------------------------------------------------------------
# NASA .mat loader
# ---------------------------------------------------------------------------

def _load_nasa_mat(battery_id: str, filepath: str) -> pd.DataFrame:
    """
    Parse one NASA PCoE .mat battery file into a per-discharge-cycle
    dataframe.

    Expected NASA structure:

        B0005.cycle[i]

    Each cycle contains:
        type
        ambient_temperature
        time
        data

    Discharge cycle data typically contains:
        Voltage_measured
        Current_measured
        Temperature_measured
        Capacity

    Returns the same standardized dataframe schema used by the CSV loader.
    """
    import scipy.io as sio

    mat = sio.loadmat(filepath, simplify_cells=True)

    if battery_id not in mat:
        raise ValueError(
            f"Expected MATLAB variable '{battery_id}' was not found "
            f"in {filepath}."
        )

    battery_struct = mat[battery_id]

    if "cycle" not in battery_struct:
        raise ValueError(
            f"MAT file {filepath} does not contain a 'cycle' field."
        )

    cycles = battery_struct["cycle"]

    rows = []
    discharge_index = 0

    for cycle_record in cycles:
        if not isinstance(cycle_record, dict):
            continue

        if cycle_record.get("type") != "discharge":
            continue

        discharge_index += 1

        data = cycle_record.get("data", {})
        if not isinstance(data, dict):
            continue

        capacity = data.get("Capacity")

        if capacity is None:
            continue

        try:
            capacity = float(np.ravel(capacity)[0])
        except (TypeError, ValueError, IndexError):
            continue

        voltage = np.ravel(
            data.get("Voltage_measured", np.array([np.nan]))
        ).astype(float)

        current = np.ravel(
            data.get("Current_measured", np.array([np.nan]))
        ).astype(float)

        temperature = np.ravel(
            data.get("Temperature_measured", np.array([np.nan]))
        ).astype(float)

        rows.append(
            {
                "cycle": discharge_index,
                "capacity_ah": capacity,
                "voltage_mean": _safe_nanmean(voltage),
                "voltage_min": _safe_nanmin(voltage),
                "voltage_max": _safe_nanmax(voltage),
                "current_mean": _safe_nanmean(current),
                "current_min": _safe_nanmin(current),
                "temperature_mean": _safe_nanmean(temperature),
                "temperature_max": _safe_nanmax(temperature),
            }
        )

    if not rows:
        raise ValueError(
            f"No valid discharge-cycle records with capacity were found "
            f"in {filepath}."
        )

    df = pd.DataFrame(rows)

    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    df["capacity_ah"] = pd.to_numeric(
        df["capacity_ah"], errors="coerce"
    )

    df["battery_id"] = battery_id
    df["data_source"] = "NASA_PCoE_MEASURED"

    return _clean_cycle_dataframe(df)


# ---------------------------------------------------------------------------
# Generic CSV support
# ---------------------------------------------------------------------------

# Canonical column -> accepted aliases.
COLUMN_ALIASES = {
    "cycle": [
        "cycle",
        "cycles",
        "cycle_number",
        "cycle_no",
        "cyclenumber",
        "cycleindex",
    ],
    "voltage": [
        "voltage",
        "voltage_v",
        "voltage_v_",
        "voltage_measured",
        "v",
    ],
    "current": [
        "current",
        "current_a",
        "current_a_",
        "current_measured",
        "i",
    ],
    "temperature": [
        "temperature",
        "temperature_c",
        "temp",
        "temp_c",
        "temperature_measured",
        "t",
    ],
    "capacity": [
        "capacity",
        "capacity_ah",
        "capacity_a_h",
        "discharge_capacity",
        "discharge_capacity_ah",
        "capacity_measured",
    ],
    "soh": [
        "soh",
        "state_of_health",
        "stateofhealth",
        "state_of_health_percent",
        "state_of_health_percentage",
        "health",
    ],
    "battery_id": [
        "battery_id",
        "battery",
        "cell_id",
        "cell",
    ],
}


def _normalize_column_name(name: str) -> str:
    """
    Convert a raw column name to a simple comparison form.

    Examples:
        'Voltage_V' -> 'voltage_v'
        'State of Health (%)' -> 'state_of_health'
    """
    text = str(name).strip().lower()

    replacements = {
        "%": " percent ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "/": " ",
        "-": "_",
        " ": "_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def _build_column_mapping(columns) -> dict[str, str]:
    """
    Find canonical column names from common aliases.

    Returns:
        canonical_name -> original_dataframe_column
    """
    normalized_to_original = {
        _normalize_column_name(column): column
        for column in columns
    }

    mapping: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        normalized_aliases = {
            _normalize_column_name(alias)
            for alias in aliases
        }

        for alias in normalized_aliases:
            if alias in normalized_to_original:
                mapping[canonical] = normalized_to_original[alias]
                break

    return mapping


def _validate_generic_csv(
    df: pd.DataFrame,
    mapping: dict[str, str],
    filepath: str,
) -> None:
    """
    Validate that a generic CSV contains enough information for SoH
    analysis.

    Required:
        cycle
        capacity OR soh

    Operating variables such as voltage/current/temperature are strongly
    preferred but are not required for a minimal capacity-based SoH model.
    """
    if "cycle" not in mapping:
        raise ValueError(
            f"\nCSV validation failed for:\n{filepath}\n\n"
            "Required column not found: cycle\n\n"
            "Examples accepted:\n"
            "  cycle\n"
            "  Cycle\n"
            "  Cycle_Number\n"
            "  cycle_no\n"
        )

    if "capacity" not in mapping and "soh" not in mapping:
        raise ValueError(
            f"\nCSV validation failed for:\n{filepath}\n\n"
            "The dataset must contain either:\n"
            "  - battery capacity, OR\n"
            "  - State of Health (SoH)\n\n"
            "Examples accepted:\n"
            "  capacity\n"
            "  Capacity_Ah\n"
            "  discharge_capacity\n"
            "  soh\n"
            "  State_of_Health\n"
        )


def _load_generic_csv(
    filepath: str,
    battery_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a generic battery CSV and convert it into the project's canonical
    per-cycle schema.

    Supported CSV styles:

    A) One row per cycle with summary columns:

        cycle, voltage, current, temperature, capacity

    B) One row per cycle with already-aggregated measurements:

        cycle, voltage_mean, current_mean, temperature_mean, capacity

    C) One row per cycle containing SoH directly:

        cycle, soh

    The loader also accepts common alternative column names.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    try:
        raw = pd.read_csv(filepath)
    except Exception as exc:
        raise ValueError(
            f"Could not read CSV file '{filepath}': {exc}"
        ) from exc

    if raw.empty:
        raise ValueError(f"CSV file '{filepath}' is empty.")

    mapping = _build_column_mapping(raw.columns)
    _validate_generic_csv(raw, mapping, filepath)

    # ------------------------------------------------------------------
    # Rename recognized columns into canonical names.
    # ------------------------------------------------------------------

    working = raw.copy()

    rename_map = {
        original: canonical
        for canonical, original in mapping.items()
    }

    working = working.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    working["cycle"] = pd.to_numeric(
        working["cycle"], errors="coerce"
    )

    # ------------------------------------------------------------------
    # Battery ID
    # ------------------------------------------------------------------

    if "battery_id" in working.columns:
        working["battery_id"] = (
            working["battery_id"]
            .fillna(battery_id or "CSV_BATTERY")
            .astype(str)
        )
    else:
        working["battery_id"] = battery_id or "CSV_BATTERY"

    # ------------------------------------------------------------------
    # SoH
    # ------------------------------------------------------------------

    if "soh" in working.columns:
        working["soh"] = pd.to_numeric(
            working["soh"], errors="coerce"
        )

    if "capacity" in working.columns:
        working["capacity_ah"] = pd.to_numeric(
            working["capacity"], errors="coerce"
        )

    # ------------------------------------------------------------------
    # Operating variables
    # ------------------------------------------------------------------

    if "voltage" in working.columns:
        working["voltage"] = pd.to_numeric(
            working["voltage"], errors="coerce"
        )

    if "current" in working.columns:
        working["current"] = pd.to_numeric(
            working["current"], errors="coerce"
        )

    if "temperature" in working.columns:
        working["temperature"] = pd.to_numeric(
            working["temperature"], errors="coerce"
        )

    # ------------------------------------------------------------------
    # If the CSV contains already-aggregated feature columns, preserve
    # them where possible.
    # ------------------------------------------------------------------

    existing_aliases = _build_column_mapping_from_feature_names(
        working.columns
    )

    for target, source in existing_aliases.items():
        if target not in working.columns and source in working.columns:
            working[target] = pd.to_numeric(
                working[source], errors="coerce"
            )

    # ------------------------------------------------------------------
    # Build cycle-level summary features.
    # ------------------------------------------------------------------

    result = _aggregate_csv_to_cycle_level(working)

    # SoH may already be directly supplied. Preserve it if available.
    if "soh" in working.columns:
        soh_by_cycle = (
            working.groupby("cycle", as_index=False)["soh"]
            .mean()
            .rename(columns={"soh": "soh"})
        )

        result = result.drop(columns=["soh"], errors="ignore")
        result = result.merge(
            soh_by_cycle,
            on="cycle",
            how="left",
        )

    result["battery_id"] = (
        result["battery_id"]
        if "battery_id" in result.columns
        else battery_id or "CSV_BATTERY"
    )

    result["data_source"] = "GENERIC_CSV"

    result = _clean_cycle_dataframe(result)

    if len(result) < 3:
        raise ValueError(
            f"CSV '{filepath}' produced only {len(result)} valid cycles. "
            "At least 3 valid cycles are recommended for the prediction "
            "pipeline."
        )

    return result


def _aggregate_csv_to_cycle_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a generic CSV into one row per cycle.

    If the CSV is already one row per cycle, this effectively preserves
    those values.

    If there are multiple measurements per cycle, numerical measurements
    are summarized using mean/min/max where appropriate.
    """
    if "cycle" not in df.columns:
        raise ValueError("Internal error: canonical 'cycle' is missing.")

    df = df.dropna(subset=["cycle"]).copy()

    if df.empty:
        raise ValueError(
            "No valid numeric cycle values remain after cleaning."
        )

    # Numeric operating-variable summaries.
    grouped = (
        df.groupby("cycle", as_index=False)
        .agg(
            capacity_ah=(
                "capacity_ah",
                "mean"
                if "capacity_ah" in df.columns
                else "first",
            )
            if "capacity_ah" in df.columns
            else ("cycle", "first"),
        )
    )

    # The above conditional aggregation is awkward when a column is absent,
    # so replace with a clearer construction below.

    grouped = df.groupby("cycle", as_index=False)

    result = grouped.size().rename(columns={"size": "_rows"})

    if "capacity_ah" in df.columns:
        capacity = (
            grouped["capacity_ah"]
            .mean()
            .rename(columns={"capacity_ah": "capacity_ah"})
        )
        result = result.drop(columns="_rows").merge(
            capacity, on="cycle", how="left"
        )

    if "voltage" in df.columns:
        voltage_stats = grouped["voltage"].agg(
            voltage_mean="mean",
            voltage_min="min",
            voltage_max="max",
        )
        result = result.drop(columns="_rows", errors="ignore").merge(
            voltage_stats, on="cycle", how="left"
        )

    if "current" in df.columns:
        current_stats = grouped["current"].agg(
            current_mean="mean",
            current_min="min",
        )
        result = result.drop(columns="_rows", errors="ignore").merge(
            current_stats, on="cycle", how="left"
        )

    if "temperature" in df.columns:
        temperature_stats = grouped["temperature"].agg(
            temperature_mean="mean",
            temperature_max="max",
        )
        result = result.drop(columns="_rows", errors="ignore").merge(
            temperature_stats, on="cycle", how="left"
        )

    # Preserve optional SoH if supplied.
    if "soh" in df.columns:
        soh = (
            grouped["soh"]
            .mean()
            .rename(columns={"soh": "soh"})
        )
        result = result.drop(columns="_rows", errors="ignore").merge(
            soh, on="cycle", how="left"
        )

    # Preserve battery ID if supplied.
    if "battery_id" in df.columns:
        battery = (
            df.groupby("cycle")["battery_id"]
            .first()
            .reset_index()
        )
        result = result.drop(columns="_rows", errors="ignore").merge(
            battery, on="cycle", how="left"
        )

    result = result.drop(columns="_rows", errors="ignore")

    return result


def _build_column_mapping_from_feature_names(
    columns,
) -> dict[str, str]:
    """
    Recognize optional already-aggregated feature columns in CSV input.

    Examples:
        voltage_mean
        voltage_min
        voltage_max
        current_mean
        current_min
        temperature_mean
        temperature_max
    """
    normalized_to_original = {
        _normalize_column_name(column): column
        for column in columns
    }

    aliases = {
        "voltage_mean": [
            "voltage_mean",
            "mean_voltage",
        ],
        "voltage_min": [
            "voltage_min",
            "minimum_voltage",
            "min_voltage",
        ],
        "voltage_max": [
            "voltage_max",
            "maximum_voltage",
            "max_voltage",
        ],
        "current_mean": [
            "current_mean",
            "mean_current",
        ],
        "current_min": [
            "current_min",
            "minimum_current",
            "min_current",
        ],
        "temperature_mean": [
            "temperature_mean",
            "mean_temperature",
            "temp_mean",
        ],
        "temperature_max": [
            "temperature_max",
            "maximum_temperature",
            "max_temperature",
            "temp_max",
        ],
    }

    mapping = {}

    for canonical, candidates in aliases.items():
        for candidate in candidates:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_to_original:
                mapping[canonical] = normalized_to_original[normalized]
                break

    return mapping


# ---------------------------------------------------------------------------
# Data cleaning helpers
# ---------------------------------------------------------------------------

def _safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


def _safe_nanmin(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmin(values))


def _safe_nanmax(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


def _clean_cycle_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply common cleaning rules while preserving the standardized schema.
    """
    df = df.copy()

    # Normalize cycle type.
    df["cycle"] = pd.to_numeric(
        df["cycle"], errors="coerce"
    )

    df = df.dropna(subset=["cycle"])

    # Integer cycle IDs are more useful downstream.
    df["cycle"] = df["cycle"].astype(int)

    # Remove duplicate cycle IDs, keeping the first standardized record.
    df = (
        df.sort_values("cycle")
        .drop_duplicates(subset=["cycle"], keep="first")
        .reset_index(drop=True)
    )

    # Numeric conversion for known numeric fields.
    numeric_columns = [
        "capacity_ah",
        "soh",
        "voltage_mean",
        "voltage_min",
        "voltage_max",
        "current_mean",
        "current_min",
        "temperature_mean",
        "temperature_max",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    # Remove completely empty feature columns.
    for column in numeric_columns:
        if column in df.columns and df[column].isna().all():
            df = df.drop(columns=[column])

    return df


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _generate_synthetic_fallback(
    n_cycles: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a clearly labeled synthetic Li-ion degradation dataset.

    This is NOT real measured data.

    It exists only as a fallback so the complete pipeline remains runnable
    when no real dataset is available.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_cycles + 1)

    C0 = 2.0
    a = 0.00006
    b = 0.25
    tau = 900.0

    capacity = C0 * (
        1 - a * n - b * (1 - np.exp(-n / tau))
    )

    noise = rng.normal(0, 0.004, size=n_cycles)

    capacity = np.clip(
        capacity + noise,
        0.05 * C0,
        C0 * 1.02,
    )

    voltage_mean = (
        3.6
        - 0.15 * (1 - capacity / C0)
        + rng.normal(0, 0.01, n_cycles)
    )

    voltage_min = (
        voltage_mean
        - 0.5
        - rng.normal(0, 0.02, n_cycles)
    )

    voltage_max = (
        voltage_mean
        + 0.55
        + rng.normal(0, 0.02, n_cycles)
    )

    current_mean = -2.0 + rng.normal(
        0, 0.03, n_cycles
    )

    current_min = current_mean - 0.2

    temperature_mean = (
        24
        + 6 * (1 - capacity / C0)
        + rng.normal(0, 0.3, n_cycles)
    )

    temperature_max = (
        temperature_mean
        + 8
        + rng.normal(0, 0.4, n_cycles)
    )

    df = pd.DataFrame(
        {
            "cycle": n,
            "capacity_ah": capacity,
            "voltage_mean": voltage_mean,
            "voltage_min": voltage_min,
            "voltage_max": voltage_max,
            "current_mean": current_mean,
            "current_min": current_min,
            "temperature_mean": temperature_mean,
            "temperature_max": temperature_max,
        }
    )

    df["battery_id"] = "SYNTHETIC_DEMO"
    df["data_source"] = "SYNTHETIC_FALLBACK_NOT_REAL"

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_battery_data(
    preferred_battery_id: str = "B0005",
    input_path: Optional[str] = None,
    allow_synthetic_fallback: bool = True,
):
    """
    Main data-loading entry point.

    Parameters
    ----------
    preferred_battery_id:
        NASA battery ID to use when multiple NASA .mat files exist.

    input_path:
        Optional explicit dataset path.

        Supported:
            .mat
            .csv

        If omitted, NASA .mat files are searched for automatically in
        data/raw/.

    allow_synthetic_fallback:
        If True, generate the clearly labeled synthetic fallback when
        no real dataset can be loaded.

    Returns
    -------
    tuple:
        (dataframe, data_source_label, message)

    Supported source labels:
        NASA_PCoE_MEASURED
        GENERIC_CSV
        SYNTHETIC_FALLBACK_NOT_REAL
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ================================================================
    # Explicit input path
    # ================================================================

    if input_path is not None:
        input_path = os.path.abspath(input_path)

        if not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"Input dataset not found:\n{input_path}"
            )

        extension = os.path.splitext(input_path)[1].lower()

        if extension == ".csv":
            df = _load_generic_csv(input_path)

            message = (
                f"Loaded GENERIC CSV battery dataset from {input_path} "
                f"({df['cycle'].nunique()} cycles)."
            )

            return df, "GENERIC_CSV", message

        if extension == ".mat":
            filename = os.path.basename(input_path)
            stem = os.path.splitext(filename)[0].upper()

            battery_id = None

            for supported_id in NASA_BATTERY_IDS:
                if supported_id in stem:
                    battery_id = supported_id
                    break

            if battery_id is None:
                raise ValueError(
                    f"Could not determine a supported NASA battery ID "
                    f"from filename '{filename}'.\n"
                    f"Expected one of: {', '.join(NASA_BATTERY_IDS)}"
                )

            df = _load_nasa_mat(
                battery_id,
                input_path,
            )

            message = (
                f"Loaded REAL measured NASA PCoE data for battery "
                f"{battery_id} ({len(df)} discharge cycles) "
                f"from {input_path}"
            )

            return df, "NASA_PCoE_MEASURED", message

        raise ValueError(
            f"Unsupported dataset format: '{extension}'\n"
            "Supported formats are .mat and .csv."
        )

    # ================================================================
    # Automatic NASA search
    # ================================================================

    found = _find_mat_files()

    if found:
        if preferred_battery_id in found:
            battery_id = preferred_battery_id
        else:
            battery_id = sorted(found.keys())[0]

        filepath = found[battery_id]

        try:
            df = _load_nasa_mat(
                battery_id,
                filepath,
            )

            message = (
                f"Loaded REAL measured NASA PCoE data for battery "
                f"{battery_id} ({len(df)} discharge cycles) "
                f"from {filepath}"
            )

            return df, "NASA_PCoE_MEASURED", message

        except Exception as exc:
            print(
                f"[WARNING] Found {filepath}, but failed to parse it: "
                f"{exc}"
            )

            if not allow_synthetic_fallback:
                raise

            print(
                "[WARNING] Falling back to clearly labeled synthetic "
                "demo data."
            )

    # ================================================================
    # Synthetic fallback
    # ================================================================

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            "\nNo real battery dataset was found.\n\n"
            "Provide one of:\n"
            "  - NASA B0005/B0006/B0007/B0018 .mat files\n"
            "  - a compatible CSV dataset\n"
        )

    message = (
        "\n"
        "==================================================================\n"
        "  NO REAL BATTERY DATA FOUND -- USING SYNTHETIC FALLBACK DATA\n"
        "==================================================================\n"
        "This synthetic dataset is NOT experimental NASA data.\n"
        "It exists only to verify that the software pipeline runs.\n\n"
        "For real data, provide either:\n"
        "  - a NASA .mat file such as B0005.mat, or\n"
        "  - a compatible CSV dataset.\n"
        "==================================================================\n"
    )

    df = _generate_synthetic_fallback()

    return (
        df,
        "SYNTHETIC_FALLBACK_NOT_REAL",
        message,
    )


if __name__ == "__main__":
    dataframe, source, message = load_battery_data()

    print(message)
    print("\nFirst 5 rows:")
    print(dataframe.head())

    print(f"\nTotal cycles loaded: {len(dataframe)}")
    print(f"Data source: {source}")