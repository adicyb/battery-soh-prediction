"""
model.py

Machine learning model for battery State-of-Health (SoH) prediction.

Primary ML model:
    RandomForestRegressor

Long-term forecast model:
    Empirical linear + exponential degradation model

        SoH(n) = 100
                 - a * (n - 1)
                 - b * (1 - exp(-(n - 1) / tau))

The Random Forest is used for the machine-learning SoH prediction task.

The long-term degradation curve is a separate empirical model because
tree-based ML models are not reliable extrapolators outside the feature
range observed during training.

Important:
    Any prediction beyond the experimentally measured cycle range is an
    extrapolation and must not be presented as measured experimental data.
"""

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Train/test splitting
# ---------------------------------------------------------------------------

def train_test_split_by_cycle(
    df: pd.DataFrame,
    train_up_to_cycle: int = 1500,
):
    """
    Chronological early-life -> future split.

    If the requested training cutoff exceeds the available data, the first
    70% of available cycles are used for training and the remaining 30% for
    testing.
    """
    max_cycle = df["cycle"].max()

    if max_cycle > train_up_to_cycle:
        split_cycle = train_up_to_cycle
        was_adapted = False
    else:
        split_cycle = int(max_cycle * 0.7)
        was_adapted = True

    train_df = (
        df[df["cycle"] <= split_cycle]
        .reset_index(drop=True)
    )

    test_df = (
        df[df["cycle"] > split_cycle]
        .reset_index(drop=True)
    )

    return train_df, test_df, split_cycle, was_adapted


def train_test_split_interpolation(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Secondary random interpolation benchmark.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = (
        train_df
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    return train_df, test_df


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_model(
    X_train,
    y_train,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train the Random Forest SoH regression model.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------------
# Empirical long-term degradation curve
# ---------------------------------------------------------------------------

def _linear_exponential_degradation_curve(
    n,
    a,
    b,
    tau,
):
    """
    Empirical battery degradation model:

        SoH(n) =
            100
            - a * (n - 1)
            - b * (1 - exp(-(n - 1) / tau))

    Parameters
    ----------
    n:
        Cycle number.

    a:
        Long-term approximately linear degradation rate.

    b:
        Magnitude of the nonlinear degradation component.

    tau:
        Characteristic cycle scale controlling the nonlinear component.

    Interpretation
    --------------
    The linear term represents continued cycle-dependent degradation.

    The exponential term captures an additional nonlinear aging component.

    This is an empirical trend model, not a first-principles
    electrochemical model.
    """
    n = np.asarray(
        n,
        dtype=float,
    )

    x = np.maximum(
        n - 1.0,
        0.0,
    )

    soh = (
        100.0
        - a * x
        - b * (
            1.0
            - np.exp(
                -x / tau
            )
        )
    )

    return soh


# ---------------------------------------------------------------------------
# Long-term forecast
# ---------------------------------------------------------------------------

def forecast_to_cycle(
    df: pd.DataFrame,
    model,
    feature_cols: list,
    target_cycle: int = 3000,
) -> pd.DataFrame:
    """
    Forecast SoH beyond the last measured cycle.

    The Random Forest is intentionally NOT used for long-range
    extrapolation.

    Instead, the measured SoH-vs-cycle trajectory is fitted using:

        SoH(n) =
            100
            - a * (n - 1)
            - b * (1 - exp(-(n - 1) / tau))

    The forecast is then evaluated for cycles after the final measured
    cycle through target_cycle.

    IMPORTANT:
        The forecast is empirical extrapolation and is not experimentally
        validated beyond the measured cycle range.
    """
    # Retained for compatibility with the existing pipeline.
    del model, feature_cols

    if "cycle" not in df.columns:
        raise ValueError(
            "Forecasting requires a 'cycle' column."
        )

    if "soh" not in df.columns:
        raise ValueError(
            "Forecasting requires a 'soh' column."
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "cycle",
                "soh_forecast",
            ]
        )

    measured = (
        df[
            [
                "cycle",
                "soh",
            ]
        ]
        .dropna()
        .sort_values("cycle")
        .drop_duplicates(
            subset=["cycle"]
        )
        .reset_index(drop=True)
    )

    if len(measured) < 8:
        print(
            "[forecast] WARNING: Fewer than 8 measured cycles are available."
        )

        print(
            "[forecast] Using a simple linear degradation fallback."
        )

        cycles = measured["cycle"].to_numpy(
            dtype=float
        )

        soh = measured["soh"].to_numpy(
            dtype=float
        )

        last_cycle = int(
            cycles[-1]
        )

        if target_cycle <= last_cycle:
            return pd.DataFrame(
                columns=[
                    "cycle",
                    "soh_forecast",
                ]
            )

        cycle_span = (
            cycles[-1]
            - cycles[0]
        )

        if cycle_span > 0:
            slope = max(
                0.0,
                (
                    soh[0]
                    - soh[-1]
                )
                / cycle_span,
            )
        else:
            slope = 0.0

        future_cycles = np.arange(
            last_cycle + 1,
            target_cycle + 1,
            dtype=int,
        )

        forecast = (
            float(soh[-1])
            - slope
            * (
                future_cycles
                - last_cycle
            )
        )

        forecast = np.clip(
            forecast,
            0.0,
            100.0,
        )

        forecast = np.minimum.accumulate(
            np.concatenate(
                [
                    [float(soh[-1])],
                    forecast,
                ]
            )
        )[1:]

        return pd.DataFrame(
            {
                "cycle": future_cycles,
                "soh_forecast": forecast,
                "forecast_a": slope,
                "forecast_b": np.nan,
                "forecast_tau": np.nan,
            }
        )

    cycles_measured = measured[
        "cycle"
    ].to_numpy(
        dtype=float
    )

    soh_measured = measured[
        "soh"
    ].to_numpy(
        dtype=float
    )

    soh_measured = np.clip(
        soh_measured,
        0.0,
        100.0,
    )

    last_cycle = int(
        cycles_measured[-1]
    )

    if target_cycle <= last_cycle:
        return pd.DataFrame(
            columns=[
                "cycle",
                "soh_forecast",
            ]
        )

    # ---------------------------------------------------------------
    # Initial parameter guesses
    # ---------------------------------------------------------------

    x_span = max(
        1.0,
        cycles_measured[-1]
        - cycles_measured[0],
    )

    total_drop = max(
        0.0,
        soh_measured[0]
        - soh_measured[-1],
    )

    # Split initial degradation estimate between:
    #   - linear component
    #   - nonlinear component
    initial_a = max(
        1e-6,
        0.25
        * total_drop
        / x_span,
    )

    initial_b = max(
        0.01,
        0.50
        * total_drop,
    )

    initial_tau = max(
        20.0,
        0.50
        * cycles_measured[-1],
    )

    p0 = [
        initial_a,
        initial_b,
        initial_tau,
    ]

    # Parameter bounds:
    #
    # a:
    #   non-negative linear degradation
    #
    # b:
    #   non-negative nonlinear component
    #
    # tau:
    #   positive characteristic cycle scale
    lower_bounds = [
        0.0,
        0.0,
        1.0,
    ]

    upper_bounds = [
        1.0,
        100.0,
        100000.0,
    ]

    # ---------------------------------------------------------------
    # Fit empirical degradation model
    # ---------------------------------------------------------------

    try:

        popt, _ = curve_fit(
            _linear_exponential_degradation_curve,
            cycles_measured,
            soh_measured,
            p0=p0,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            maxfev=100000,
        )

        a_fit = float(
            popt[0]
        )

        b_fit = float(
            popt[1]
        )

        tau_fit = float(
            popt[2]
        )

        print(
            "[forecast] Fitted empirical linear + exponential "
            "degradation model:"
        )

        print(
            "[forecast] "
            f"SoH(n) = 100 - "
            f"{a_fit:.6f}(n-1) - "
            f"{b_fit:.4f}(1-exp(-(n-1)/{tau_fit:.2f}))"
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ) as exc:

        print(
            "[forecast] WARNING: Nonlinear degradation fit was unstable."
        )

        print(
            f"[forecast] {exc}"
        )

        print(
            "[forecast] Using a simple linear degradation fallback."
        )

        cycle_span = max(
            1.0,
            cycles_measured[-1]
            - cycles_measured[0],
        )

        slope = max(
            0.0,
            (
                soh_measured[0]
                - soh_measured[-1]
            )
            / cycle_span,
        )

        future_cycles = np.arange(
            last_cycle + 1,
            target_cycle + 1,
            dtype=int,
        )

        forecast = (
            soh_measured[-1]
            - slope
            * (
                future_cycles
                - last_cycle
            )
        )

        forecast = np.clip(
            forecast,
            0.0,
            100.0,
        )

        forecast = np.minimum.accumulate(
            np.concatenate(
                [
                    [float(soh_measured[-1])],
                    forecast,
                ]
            )
        )[1:]

        return pd.DataFrame(
            {
                "cycle": future_cycles,
                "soh_forecast": forecast,
                "forecast_a": slope,
                "forecast_b": np.nan,
                "forecast_tau": np.nan,
            }
        )

    # ---------------------------------------------------------------
    # Generate forecast cycles
    # ---------------------------------------------------------------

    future_cycles = np.arange(
        last_cycle + 1,
        target_cycle + 1,
        dtype=int,
    )

    soh_forecast = (
        _linear_exponential_degradation_curve(
            future_cycles,
            a_fit,
            b_fit,
            tau_fit,
        )
    )

    # ---------------------------------------------------------------
    # Anchor continuation to final measured point
    # ---------------------------------------------------------------

    last_soh = float(
        soh_measured[-1]
    )

    first_fitted = float(
        soh_forecast[0]
    )

    # Shift the complete forecast so the first predicted point continues
    # from the final measured value.
    soh_forecast = (
        soh_forecast
        + (
            last_soh
            - first_fitted
        )
    )

    # The battery should not become healthier as cycles increase.
    soh_forecast = np.minimum.accumulate(
        np.concatenate(
            [
                [last_soh],
                soh_forecast,
            ]
        )
    )[1:]

    # Final physical bounds.
    soh_forecast = np.clip(
        soh_forecast,
        0.0,
        100.0,
    )

    return pd.DataFrame(
        {
            "cycle": future_cycles,
            "soh_forecast": soh_forecast,
            "forecast_a": a_fit,
            "forecast_b": b_fit,
            "forecast_tau": tau_fit,
        }
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from data_loader import load_battery_data
    from preprocessing import preprocess
    from feature_engineering import build_features

    raw_df, source, message = (
        load_battery_data()
    )

    print(message)

    processed = preprocess(
        raw_df,
        source,
        raw_df["battery_id"].iloc[0],
        save=False,
    )

    X, y, feature_cols, full_df = (
        build_features(processed)
    )

    (
        train_df,
        test_df,
        split_cycle,
        adapted,
    ) = train_test_split_by_cycle(
        full_df
    )

    print(
        f"Split at cycle {split_cycle} "
        f"(adapted={adapted})"
    )

    print(
        f"Train size: {len(train_df)}, "
        f"Test size: {len(test_df)}"
    )

    model = train_model(
        train_df[feature_cols],
        train_df["soh"],
    )

    print(
        "Model trained."
    )

    forecast_df = forecast_to_cycle(
        full_df,
        model,
        feature_cols,
        target_cycle=3000,
    )

    print(
        forecast_df.tail()
    )