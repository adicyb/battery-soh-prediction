"""
model.py

Machine learning model for SoH prediction.

Model choice: RandomForestRegressor (scikit-learn).
Reasoning (for viva): SoH degradation is a smooth, mostly monotonic tabular
regression problem with a modest number of engineered features and a modest
number of samples. Random Forest:
  - handles non-linear degradation curves (linear-fade + knee) well,
  - needs no feature scaling,
  - is robust to noise in individual cycle measurements,
  - is fast to train (seconds, not minutes) and easy to explain in a viva,
  - does not require the large sample sizes deep learning would need.
This matches the project's constraint to avoid unnecessary deep learning.

Two things this module provides:
  1. train_test_split_by_cycle(): implements the "train on early cycles,
     test on later cycles" experiment described in the project notes
     (default: train on the first ~1500 cycles' worth of proportion of data,
     test on the rest -- adapted automatically if the dataset has fewer
     cycles, with the adaptation clearly printed/logged).
  2. forecast_to_cycle(): extends predictions out to a target cycle count
     (default 3000) using the trained model plus a fitted degradation trend
     for the extrapolated feature values. This is clearly a FORECAST, not
     measured data, and is labeled as such everywhere downstream.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def train_test_split_by_cycle(df: pd.DataFrame, train_up_to_cycle: int = 1500):
    """
    Splits the processed dataframe into an early-life training set and a
    later-life testing set, based on cycle number (NOT randomly shuffled --
    this is a genuine forecasting-style split, appropriate for time-ordered
    degradation data, and matches the project's "feed early cycles, predict
    later cycles" requirement).

    If the dataset has fewer than `train_up_to_cycle` cycles total, the
    split is adapted to use the first 70% of available cycles for training
    and the remaining 30% for testing, and this adaptation is reported.

    IMPORTANT METHODOLOGY NOTE (see also forecast_to_cycle docstring):
    Because this is a genuine extrapolation split (the model never sees
    cycle numbers, or the sensor-value ranges that come with them, beyond
    the training cutoff), a feature-based ML regressor such as Random
    Forest is NOT expected to extrapolate accurately on cycle-indexed
    features here -- this is a well-known and important limitation of
    tree-based models, and is exactly why Step 7 (3000-cycle forecasting)
    uses a physically-motivated curve fit instead of the Random Forest.
    This split is still used honestly to report the RF's real,
    unmodified MAE/R2 on genuinely unseen future cycles.

    Returns: (train_df, test_df, actual_split_cycle, was_adapted: bool)
    """
    max_cycle = df["cycle"].max()

    if max_cycle > train_up_to_cycle:
        split_cycle = train_up_to_cycle
        was_adapted = False
    else:
        split_cycle = int(max_cycle * 0.7)
        was_adapted = True

    train_df = df[df["cycle"] <= split_cycle].reset_index(drop=True)
    test_df = df[df["cycle"] > split_cycle].reset_index(drop=True)

    return train_df, test_df, split_cycle, was_adapted


def train_test_split_interpolation(df: pd.DataFrame, test_size: float = 0.25,
                                    random_state: int = 42):
    """
    A SECOND, complementary evaluation: a randomly shuffled (interpolation)
    train/test split, which is the setting Random Forest is actually well
    suited to (predicting SoH for cycles scattered throughout the measured
    range, rather than purely extrapolating into the future). Reported
    alongside the cycle-ordered split so the report/viva can honestly
    compare "interpolation accuracy" vs "future-extrapolation accuracy".

    Returns: (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    train_df = train_df.sort_values("cycle").reset_index(drop=True)
    test_df = test_df.sort_values("cycle").reset_index(drop=True)
    return train_df, test_df


def train_model(X_train, y_train, random_state: int = 42) -> RandomForestRegressor:
    """Train the Random Forest SoH regression model."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _bounded_degradation_curve(n, floor, tau, power):
    """Bounded empirical degradation curve for long-range SoH forecasting.

    SoH(n) = floor + (100 - floor) * exp(-(n / tau)^power)

    The model is deliberately bounded and monotonically decreasing for
    positive parameters.  It is an empirical curve fit, not a first-principles
    electrochemical model.  The fitted floor is the long-run asymptote of the
    curve; it is *not* a claim about the true physical life of the cell.
    """
    n = np.asarray(n, dtype=float)
    return floor + (100.0 - floor) * np.exp(-np.power(np.maximum(n, 0.0) / tau, power))


def forecast_to_cycle(df: pd.DataFrame, model, feature_cols: list,
                      target_cycle: int = 3000) -> pd.DataFrame:
    """Forecast SoH beyond the last measured cycle out to ``target_cycle``.

    The NASA B0005 experiment contains far fewer than 3000 measured discharge
    cycles.  A Random Forest is therefore *not* used for the long-range
    extrapolation because tree ensembles do not extrapolate reliably outside
    their training feature range.

    Instead, an empirical bounded stretched-exponential curve is fitted to
    the measured SoH-vs-cycle history.  This guarantees a non-negative,
    monotonic, bounded forecast and avoids the previous implementation's
    collapse to an artificial 0% SoH at cycle 3000.

    IMPORTANT: the 169..3000 region for B0005 is a model-based extrapolation,
    not experimental measurement.  Because 3000 is far outside the observed
    range, the forecast should be treated as a scenario/trend illustration,
    not a validated lifetime estimate.
    """
    del model, feature_cols  # retained in the function signature for compatibility

    last_cycle = int(df["cycle"].max())
    if target_cycle <= last_cycle:
        return pd.DataFrame(columns=["cycle", "soh_forecast"])

    cycles_measured = df["cycle"].to_numpy(dtype=float)
    soh_measured = df["soh"].to_numpy(dtype=float)

    # Enforce the project convention that the first measured SoH is 100%.
    # The floor is constrained below the minimum measured SoH so the fitted
    # curve cannot jump upward or create an unphysical asymptote above data.
    floor_upper = float(np.min(soh_measured) - 1e-6)
    floor_upper = min(floor_upper, 95.0)
    floor_upper = max(floor_upper, 1.0)

    p0 = [max(1.0, min(60.0, floor_upper * 0.8)),
          max(10.0, cycles_measured.max() / 2.0),
          1.0]

    lower_bounds = [0.0, 1.0, 0.20]
    upper_bounds = [floor_upper, 100000.0, 2.50]

    try:
        popt, _ = curve_fit(
            _bounded_degradation_curve,
            cycles_measured,
            soh_measured,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000,
        )
    except (RuntimeError, ValueError):
        # Deterministic fallback: fit a bounded exponential with power=1.
        def exponential_fallback(x, floor, tau):
            return _bounded_degradation_curve(x, floor, tau, 1.0)

        p0_fallback = [p0[0], p0[1]]
        popt2, _ = curve_fit(
            exponential_fallback,
            cycles_measured,
            soh_measured,
            p0=p0_fallback,
            bounds=(lower_bounds[:2], upper_bounds[:2]),
            maxfev=50000,
        )
        floor_fit, tau_fit = popt2
        power_fit = 1.0
    else:
        floor_fit, tau_fit, power_fit = popt

    future_cycles = np.arange(last_cycle + 1, target_cycle + 1, dtype=int)
    soh_forecast = _bounded_degradation_curve(
        future_cycles, floor_fit, tau_fit, power_fit
    )

    # Anchor forecast exactly to the last measured value and preserve
    # monotonic non-increase from the measured endpoint onward.
    last_soh = float(soh_measured[-1])
    soh_forecast = np.minimum.accumulate(
        np.concatenate([[last_soh], soh_forecast])
    )[1:]
    soh_forecast = np.clip(soh_forecast, 0.0, 100.0)

    return pd.DataFrame({
        "cycle": future_cycles,
        "soh_forecast": soh_forecast,
        "forecast_floor": float(floor_fit),
        "forecast_tau": float(tau_fit),
        "forecast_power": float(power_fit),
    })


if __name__ == "__main__":
    from data_loader import load_battery_data
    from preprocessing import preprocess
    from feature_engineering import build_features

    raw_df, source, message = load_battery_data()
    processed = preprocess(raw_df, source, raw_df["battery_id"].iloc[0], save=False)
    X, y, feature_cols, full_df = build_features(processed)

    train_df, test_df, split_cycle, adapted = train_test_split_by_cycle(full_df)
    print(f"Split at cycle {split_cycle} (adapted={adapted})")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df["soh"]
    model = train_model(X_train, y_train)
    print("Model trained.")

    forecast_df = forecast_to_cycle(full_df, model, feature_cols, target_cycle=3000)
    print(forecast_df.tail())
