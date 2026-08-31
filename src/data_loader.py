"""
data_loader.py

Loads battery cycling data for the SoH prediction project.

PRIMARY SOURCE (preferred):
    NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset
    Batteries B0005, B0006, B0007, B0018 (18650 cells).
    Source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
    Also mirrored at: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

    These .mat files contain repeated charge/discharge/impedance cycles at
    24C ambient temperature until the battery reached ~30% capacity fade
    (i.e. end-of-life relative to rated capacity). Each cycle record
    includes time-series voltage, current, and temperature during
    discharge, and the discharge capacity for that cycle.

FALLBACK (only if the real .mat files are not present in data/raw/):
    A clearly-labeled SYNTHETIC degradation dataset is generated instead,
    using a standard empirical Li-ion capacity-fade model (combined linear +
    exponential knee degradation with realistic measurement noise). This is
    NOT NASA data and is labeled "SYNTHETIC" everywhere it is used -- in the
    CSV, in the plots, and in the console output. This exists purely so the
    pipeline is runnable before the real dataset is downloaded and placed in
    data/raw/. Replace it with real data as soon as possible.

This module never mixes the two silently -- the data source is recorded in
every processed file so downstream steps and the report can state clearly
whether results are based on real measured data or synthetic fallback data.
"""

import os
import glob
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

NASA_BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]


def _find_mat_files():
    """Look for NASA .mat files in data/raw/."""
    found = {}
    for bid in NASA_BATTERY_IDS:
        matches = glob.glob(os.path.join(RAW_DIR, f"{bid}.mat")) + \
                  glob.glob(os.path.join(RAW_DIR, "**", f"{bid}.mat"), recursive=True)
        if matches:
            found[bid] = matches[0]
    return found


def _load_nasa_mat(battery_id, filepath):
    """
    Parse a NASA PCoE .mat file into a per-cycle dataframe.

    The NASA .mat structure (as distributed by PCoE) is a MATLAB struct
    array named after the battery id, e.g. B0005.cycle[i], where each
    cycle has fields:
        type      - 'charge', 'discharge', or 'impedance'
        ambient_temperature
        time
        data      - struct with Voltage_measured, Current_measured,
                    Temperature_measured, Current_charge, Voltage_charge,
                    Capacity (discharge cycles only), etc.

    We extract, for every DISCHARGE cycle:
        cycle index, discharge capacity (Ah),
        mean/max measured voltage, mean/max measured current,
        mean/max measured temperature.
    """
    import scipy.io as sio

    mat = sio.loadmat(filepath, simplify_cells=True)
    battery_struct = mat[battery_id]
    cycles = battery_struct["cycle"]

    rows = []
    discharge_index = 0
    for c in cycles:
        if c.get("type") != "discharge":
            continue
        discharge_index += 1
        d = c.get("data", {})

        capacity = d.get("Capacity", None)
        # Capacity is stored as a scalar (Ah) for discharge cycles
        if capacity is None:
            continue
        capacity = float(np.ravel(capacity)[0])

        voltage = np.ravel(d.get("Voltage_measured", np.array([np.nan])))
        current = np.ravel(d.get("Current_measured", np.array([np.nan])))
        temperature = np.ravel(d.get("Temperature_measured", np.array([np.nan])))

        rows.append({
            "cycle": discharge_index,
            "capacity_ah": capacity,
            "voltage_mean": float(np.nanmean(voltage)),
            "voltage_min": float(np.nanmin(voltage)),
            "voltage_max": float(np.nanmax(voltage)),
            "current_mean": float(np.nanmean(current)),
            "current_min": float(np.nanmin(current)),
            "temperature_mean": float(np.nanmean(temperature)),
            "temperature_max": float(np.nanmax(temperature)),
        })

    df = pd.DataFrame(rows)
    df["battery_id"] = battery_id
    df["data_source"] = "NASA_PCoE_MEASURED"
    return df


def _generate_synthetic_fallback(n_cycles=1200, seed=42):
    """
    Generate a CLEARLY LABELED synthetic Li-ion degradation dataset.

    This is NOT real measured data. It exists only so the full pipeline
    (preprocessing -> ML -> evaluation -> plots) can be demonstrated and
    verified before the real NASA .mat files are placed in data/raw/.

    Degradation model used (standard empirical form for Li-ion capacity
    fade under cyclic aging):
        capacity(n) = C0 * (1 - a*n - b*(1 - exp(-n/tau))) + noise

    This produces the well-known two-stage behaviour: a slow near-linear
    fade phase followed by an accelerating "knee" as the cell approaches
    end of life -- consistent with widely reported Li-ion aging behaviour,
    but the specific numbers here are illustrative, not measured.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_cycles + 1)

    C0 = 2.0  # nominal rated capacity in Ah (typical for an 18650 cell)
    a = 0.00006      # slow linear fade rate
    b = 0.25         # magnitude of accelerated "knee" fade
    tau = 900.0      # knee onset characteristic cycle count

    capacity = C0 * (1 - a * n - b * (1 - np.exp(-n / tau)))
    noise = rng.normal(0, 0.004, size=n_cycles)
    capacity = capacity + noise
    capacity = np.clip(capacity, 0.05 * C0, C0 * 1.02)

    voltage_mean = 3.6 - 0.15 * (1 - capacity / C0) + rng.normal(0, 0.01, n_cycles)
    voltage_min = voltage_mean - 0.5 - rng.normal(0, 0.02, n_cycles)
    voltage_max = voltage_mean + 0.55 + rng.normal(0, 0.02, n_cycles)
    current_mean = -2.0 + rng.normal(0, 0.03, n_cycles)
    current_min = current_mean - 0.2
    temperature_mean = 24 + 6 * (1 - capacity / C0) + rng.normal(0, 0.3, n_cycles)
    temperature_max = temperature_mean + 8 + rng.normal(0, 0.4, n_cycles)

    df = pd.DataFrame({
        "cycle": n,
        "capacity_ah": capacity,
        "voltage_mean": voltage_mean,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "current_mean": current_mean,
        "current_min": current_min,
        "temperature_mean": temperature_mean,
        "temperature_max": temperature_max,
    })
    df["battery_id"] = "SYNTHETIC_DEMO"
    df["data_source"] = "SYNTHETIC_FALLBACK_NOT_REAL"
    return df


def load_battery_data(preferred_battery_id="B0005"):
    """
    Main entry point. Returns (dataframe, data_source_label, message).

    data_source_label is one of:
        "NASA_PCoE_MEASURED"      -- real measured NASA data was found and used
        "SYNTHETIC_FALLBACK_NOT_REAL" -- no .mat file found, synthetic data used instead
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    found = _find_mat_files()

    if found:
        if preferred_battery_id in found:
            bid = preferred_battery_id
        else:
            bid = sorted(found.keys())[0]
        filepath = found[bid]
        try:
            df = _load_nasa_mat(bid, filepath)
            msg = (f"Loaded REAL measured NASA PCoE data for battery {bid} "
                   f"({len(df)} discharge cycles) from {filepath}")
            return df, "NASA_PCoE_MEASURED", msg
        except Exception as e:
            print(f"[WARNING] Found {filepath} but failed to parse it ({e}).")
            print("[WARNING] Falling back to synthetic demo data instead.")

    msg = (
        "\n"
        "==================================================================\n"
        "  NO REAL NASA BATTERY DATA FOUND -- USING SYNTHETIC FALLBACK DATA\n"
        "==================================================================\n"
        "To use REAL measured data, download one or more of these NASA .mat\n"
        "files:\n"
        "    B0005.mat, B0006.mat, B0007.mat, B0018.mat\n"
        "from:\n"
        "    https://www.nasa.gov/intelligent-systems-division/discovery-and-\n"
        "    systems-health/pcoe/pcoe-data-set-repository/\n"
        "    (dataset #5: 'Battery Data Set')\n"
        "  or the mirror:\n"
        "    https://data.nasa.gov/dataset/li-ion-battery-aging-datasets\n"
        "\n"
        "Then place the .mat file(s) directly into:\n"
        f"    {os.path.abspath(RAW_DIR)}\n"
        "\n"
        "and re-run this program. It will automatically detect and use them.\n"
        "\n"
        "Until then, this run uses a CLEARLY LABELED synthetic degradation\n"
        "dataset (data_source = SYNTHETIC_FALLBACK_NOT_REAL) so you can verify\n"
        "the full pipeline works end-to-end.\n"
        "==================================================================\n"
    )
    df = _generate_synthetic_fallback()
    return df, "SYNTHETIC_FALLBACK_NOT_REAL", msg


if __name__ == "__main__":
    df, source, message = load_battery_data()
    print(message)
    print(df.head())
    print(f"\nTotal cycles loaded: {len(df)}")
    print(f"Data source: {source}")
