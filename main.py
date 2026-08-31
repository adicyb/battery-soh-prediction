"""
main.py

Lithium-Ion Battery State of Health (SoH) Prediction
Using Early-Life Battery Operation Data

Main workflow
-------------
1. Load NASA .mat or generic CSV data.
2. Validate and preprocess the data.
3. Calculate/preserve SoH.
4. Engineer model features.
5. Perform chronological early-life -> future evaluation.
6. Perform a secondary random interpolation benchmark.
7. Train Random Forest regression.
8. Generate a long-term empirical forecast toward cycle 3000.
9. Generate report-quality plots.
10. Save predictions and metrics.

Examples
--------
Use automatic NASA discovery:

    python main.py

Use a specific NASA file:

    python main.py --input data/raw/B0005.mat

Use a generic CSV:

    python main.py --input data/raw/battery.csv

Choose a NASA battery when several .mat files are present:

    python main.py --battery B0006
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src",
)

PREDICTIONS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "predictions",
)

# Allow imports from src/
sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from data_loader import load_battery_data
from preprocessing import preprocess
from feature_engineering import build_features

from model import (
    train_test_split_by_cycle,
    train_test_split_interpolation,
    train_model,
    forecast_to_cycle,
)

from evaluation import (
    evaluate,
    print_metrics_report,
    save_metrics,
)

from visualization import (
    plot_degradation_curve,
    plot_actual_vs_predicted,
    plot_residuals,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_FORECAST_CYCLE = 3000
DEFAULT_TRAIN_UP_TO_CYCLE = 1500


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Lithium-ion battery State-of-Health prediction using "
            "early-life operational data."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to a battery dataset (.mat or .csv). "
            "If omitted, supported NASA .mat files are searched "
            "automatically in data/raw/."
        ),
    )

    parser.add_argument(
        "--battery",
        type=str,
        default="B0005",
        choices=["B0005", "B0006", "B0007", "B0018"],
        help=(
            "Preferred NASA battery ID when multiple NASA .mat files "
            "are present. Default: B0005."
        ),
    )

    parser.add_argument(
        "--train-until",
        type=int,
        default=DEFAULT_TRAIN_UP_TO_CYCLE,
        help=(
            "Maximum cycle used for the early-life training window. "
            "If the dataset is shorter, the split is automatically "
            "adapted. Default: 1500."
        ),
    )

    parser.add_argument(
        "--forecast-to",
        type=int,
        default=TARGET_FORECAST_CYCLE,
        help=(
            "Final cycle for long-term empirical forecasting. "
            "Default: 3000."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_initial_capacity(processed_df: pd.DataFrame):
    """
    Safely retrieve the initial capacity if available.

    A generic dataset may provide SoH directly without providing battery
    capacity, so initial_capacity_ah is optional.
    """
    if "initial_capacity_ah" not in processed_df.columns:
        return None

    values = processed_df["initial_capacity_ah"].dropna()

    if values.empty:
        return None

    return float(values.iloc[0])


def get_battery_id(
    raw_df: pd.DataFrame,
    fallback: str = "BATTERY",
) -> str:
    """
    Safely obtain the battery identifier from the standardized dataframe.
    """
    if "battery_id" not in raw_df.columns:
        return fallback

    values = raw_df["battery_id"].dropna()

    if values.empty:
        return fallback

    return str(values.iloc[0])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    print("\n" + "#" * 60)
    print("# Lithium-Ion Battery SoH Prediction Pipeline")
    print("#" * 60)

    # ---------------------------------------------------------------
    # Configuration checks
    # ---------------------------------------------------------------

    if args.train_until < 3:
        print(
            "[main] ERROR: --train-until must be at least 3 cycles."
        )
        return 1

    if args.forecast_to < 3:
        print(
            "[main] ERROR: --forecast-to must be at least 3 cycles."
        )
        return 1

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------

    try:
        raw_df, data_source, message = load_battery_data(
            preferred_battery_id=args.battery,
            input_path=args.input,
        )
    except Exception as exc:
        print("\n[main] ERROR while loading dataset:")
        print(f"[main] {exc}")
        return 1

    print(message)

    if raw_df.empty:
        print("[main] ERROR: Loaded dataset is empty.")
        return 1

    battery_id = get_battery_id(
        raw_df,
        fallback=(
            os.path.splitext(
                os.path.basename(args.input)
            )[0]
            if args.input
            else args.battery
        ),
    )

    # ---------------------------------------------------------------
    # Step 2: Preprocess and calculate/preserve SoH
    # ---------------------------------------------------------------

    try:
        processed_df = preprocess(
            raw_df,
            data_source,
            battery_id,
        )
    except Exception as exc:
        print("\n[main] ERROR during preprocessing:")
        print(f"[main] {exc}")
        return 1

    if processed_df.empty:
        print("[main] ERROR: No valid processed data remains.")
        return 1

    print(
        f"\n[main] Data source for this run: {data_source}"
    )

    print(
        f"[main] Battery ID: {battery_id}"
    )

    print(
        "[main] Total measured cycles available: "
        f"{processed_df['cycle'].max()}"
    )

    initial_capacity = get_initial_capacity(
        processed_df
    )

    if initial_capacity is not None:
        print(
            "[main] Initial reference capacity: "
            f"{initial_capacity:.4f} Ah"
        )
    else:
        print(
            "[main] Initial reference capacity: "
            "not supplied/available (SoH provided directly)"
        )

    print(
        "[main] Final measured SoH: "
        f"{processed_df['soh'].iloc[-1]:.2f}% "
        f"at cycle {processed_df['cycle'].iloc[-1]}"
    )

    # ---------------------------------------------------------------
    # Step 3: Feature engineering
    # ---------------------------------------------------------------

    try:
        X, y, feature_cols, full_df = build_features(
            processed_df
        )
    except Exception as exc:
        print("\n[main] ERROR during feature engineering:")
        print(f"[main] {exc}")
        return 1

    print(
        f"\n[main] Features used ({len(feature_cols)}): "
        f"{feature_cols}"
    )

    if len(full_df) < 6:
        print(
            "[main] ERROR: At least 6 valid cycles are recommended "
            "for train/test evaluation."
        )
        return 1

    # ---------------------------------------------------------------
    # Step 4: Chronological early-life -> future split
    # ---------------------------------------------------------------

    try:
        (
            train_df,
            test_df,
            split_cycle,
            was_adapted,
        ) = train_test_split_by_cycle(
            full_df,
            train_up_to_cycle=args.train_until,
        )
    except Exception as exc:
        print("\n[main] ERROR during chronological split:")
        print(f"[main] {exc}")
        return 1

    max_cycle = int(full_df["cycle"].max())

    if was_adapted:
        print(
            f"\n[main] NOTE: Dataset has only {max_cycle} measured "
            f"cycles, fewer than the requested "
            f"{args.train_until}-cycle training window."
        )

        print(
            "[main] Adapted split: training on cycles "
            f"1-{split_cycle} (70%), testing on cycles "
            f"{split_cycle + 1}-{max_cycle} (30%)."
        )
    else:
        print(
            f"\n[main] Training on cycles 1-{split_cycle}, "
            f"testing on cycles "
            f"{split_cycle + 1}-{max_cycle}."
        )

    X_train = train_df[feature_cols]
    y_train = train_df["soh"]

    X_test = test_df[feature_cols]
    y_test = test_df["soh"]

    if len(X_train) < 2 or len(X_test) == 0:
        print(
            "[main] ERROR: Insufficient train/test samples "
            "for evaluation."
        )
        return 1

    # ---------------------------------------------------------------
    # Step 5: Train Random Forest
    # ---------------------------------------------------------------

    try:
        model = train_model(
            X_train,
            y_train,
        )
    except Exception as exc:
        print("\n[main] ERROR while training Random Forest:")
        print(f"[main] {exc}")
        return 1

    print(
        "\n[main] Random Forest model trained on "
        f"{len(X_train)} samples "
        "(early-cycle / future-cycle evaluation)."
    )

    # ---------------------------------------------------------------
    # Step 6: Primary future-cycle evaluation
    # ---------------------------------------------------------------

    try:
        y_pred_test = model.predict(
            X_test
        )

        metrics_extrapolation = evaluate(
            y_test,
            y_pred_test,
        )
    except Exception as exc:
        print(
            "\n[main] ERROR during future-cycle evaluation:"
        )
        print(f"[main] {exc}")
        return 1

    print(
        "\n[main] === Evaluation 1: "
        "EARLY-CYCLE -> FUTURE-CYCLE PREDICTION ==="
    )

    print(
        "[main] Training uses only early-life cycles; "
        "testing uses later, never-seen cycles."
    )

    print_metrics_report(
        metrics_extrapolation,
        split_cycle,
        was_adapted,
    )

    # ---------------------------------------------------------------
    # Step 7: Secondary interpolation benchmark
    # ---------------------------------------------------------------

    try:
        train_df_i, test_df_i = (
            train_test_split_interpolation(
                full_df
            )
        )

        model_interp = train_model(
            train_df_i[feature_cols],
            train_df_i["soh"],
        )

        y_pred_interp = model_interp.predict(
            test_df_i[feature_cols]
        )

        metrics_interp = evaluate(
            test_df_i["soh"],
            y_pred_interp,
        )

    except Exception as exc:
        print(
            "\n[main] ERROR during interpolation benchmark:"
        )
        print(f"[main] {exc}")
        return 1

    print(
        "\n[main] === Evaluation 2: "
        "RANDOM (INTERPOLATION) SPLIT ==="
    )

    print(
        "[main] This is a secondary benchmark showing "
        "performance within the measured cycle range."
    )

    print_metrics_report(
        metrics_interp,
        split_cycle=None,
        was_adapted=False,
    )

    # The chronological future-cycle experiment is the primary result.
    metrics = metrics_extrapolation

    # ---------------------------------------------------------------
    # Step 8: Long-term empirical forecast
    # ---------------------------------------------------------------

    try:
        forecast_df = forecast_to_cycle(
            full_df,
            model,
            feature_cols,
            target_cycle=args.forecast_to,
        )
    except Exception as exc:
        print(
            "\n[main] ERROR while generating long-term forecast:"
        )
        print(f"[main] {exc}")
        return 1

    if len(forecast_df) > 0:

        print(
            f"\n[main] Forecast generated from cycle "
            f"{max_cycle + 1} to cycle "
            f"{args.forecast_to}."
        )

        print(
            f"[main] Forecasted SoH at cycle "
            f"{args.forecast_to}: "
            f"{forecast_df['soh_forecast'].iloc[-1]:.2f}%"
        )

        forecast_horizon = args.forecast_to - max_cycle

        print(
            f"[forecast] Measured data ends at cycle {max_cycle}."
        )

        print(
            f"[forecast] Forecast extends {forecast_horizon} cycles "
            f"beyond the measured data."
        )

        if forecast_horizon >= max(100, int(max_cycle * 2)):
            print(
                "[forecast] WARNING: This is a long-range empirical "
                "extrapolation far beyond the measured data."
            )
            print(
                "[forecast] The forecast is not experimentally validated "
                "beyond the observed cycle range."
            )

        # -----------------------------------------------------------
        # EOL detection
        # -----------------------------------------------------------

        measured_eol = full_df[
            full_df["soh"] <= 80.0
        ]

        if len(measured_eol) > 0:

            print(
                "[main] Measured data crosses 80% EOL "
                "threshold at cycle "
                f"{int(measured_eol['cycle'].iloc[0])}."
            )

        else:

            forecast_eol = forecast_df[
                forecast_df["soh_forecast"] <= 80.0
            ]

            if len(forecast_eol) > 0:

                print(
                    "[main] Forecast crosses 80% EOL "
                    "threshold at approx. cycle "
                    f"{int(forecast_eol['cycle'].iloc[0])}."
                )

            else:

                print(
                    "[main] Forecast does not reach "
                    "80% EOL threshold by cycle "
                    f"{args.forecast_to}."
                )

    else:

        print(
            f"\n[main] No forecast needed -- measured data "
            f"already extends to/past "
            f"cycle {args.forecast_to}."
        )

    # ---------------------------------------------------------------
    # Step 9: Generate plots
    # ---------------------------------------------------------------

    try:

        plot_degradation_curve(
            full_df,
            forecast_df,
            data_source,
            battery_id,
        )

        plot_actual_vs_predicted(
            test_df["soh"].values,
            y_pred_test,
            test_df["cycle"].values,
            filename="actual_vs_predicted_future.png",
        )

        plot_actual_vs_predicted(
            test_df_i["soh"].values,
            y_pred_interp,
            test_df_i["cycle"].values,
            filename="actual_vs_predicted_interpolation.png",
        )

        plot_residuals(
            test_df["soh"].values,
            y_pred_test,
            test_df["cycle"].values,
            filename="prediction_residuals_future.png",
        )

    except Exception as exc:

        print(
            "\n[main] ERROR while generating plots:"
        )

        print(
            f"[main] {exc}"
        )

        return 1

    # ---------------------------------------------------------------
    # Step 10: Save prediction files
    # ---------------------------------------------------------------

    os.makedirs(
        PREDICTIONS_DIR,
        exist_ok=True,
    )

    # Secondary interpolation predictions.
    interp_predictions_df = (
        test_df_i[["cycle"]].copy()
    )

    interp_predictions_df["actual_soh"] = (
        test_df_i["soh"].values
    )

    interp_predictions_df["predicted_soh"] = (
        y_pred_interp
    )

    interp_predictions_df["absolute_error"] = (
        interp_predictions_df["actual_soh"]
        - interp_predictions_df["predicted_soh"]
    ).abs()

    interp_predictions_df["type"] = (
        "random_split_measured_vs_predicted"
    )

    interp_pred_path = os.path.join(
        PREDICTIONS_DIR,
        "test_set_predictions.csv",
    )

    interp_predictions_df.to_csv(
        interp_pred_path,
        index=False,
    )

    print(
        "\n[main] Saved random-split test predictions -> "
        f"{interp_pred_path}"
    )

    # Primary early -> future predictions.
    extrapolation_predictions_df = (
        test_df[["cycle"]].copy()
    )

    extrapolation_predictions_df["actual_soh"] = (
        y_test.values
    )

    extrapolation_predictions_df["predicted_soh"] = (
        y_pred_test
    )

    extrapolation_predictions_df["absolute_error"] = (
        extrapolation_predictions_df["actual_soh"]
        - extrapolation_predictions_df["predicted_soh"]
    ).abs()

    extrapolation_predictions_df["type"] = (
        "early_to_future_cycle_prediction"
    )

    extrap_pred_path = os.path.join(
        PREDICTIONS_DIR,
        "early_to_future_extrapolation_predictions.csv",
    )

    extrapolation_predictions_df.to_csv(
        extrap_pred_path,
        index=False,
    )

    print(
        "[main] Saved early->future predictions -> "
        f"{extrap_pred_path}"
    )

    # Long-term forecast.
    if len(forecast_df) > 0:

        forecast_out = forecast_df[
            ["cycle", "soh_forecast"]
        ].copy()

        forecast_out["type"] = (
            "forecast_beyond_measured_data"
        )

        forecast_path = os.path.join(
            PREDICTIONS_DIR,
            "forecast_3000_cycles.csv",
        )

        forecast_out.to_csv(
            forecast_path,
            index=False,
        )

        print(
            "[main] Saved long-term forecast -> "
            f"{forecast_path}"
        )

    # ---------------------------------------------------------------
    # Step 11: Save metrics and metadata
    # ---------------------------------------------------------------

    extra_info = {
        "battery_id": battery_id,
        "data_source": data_source,
        "is_synthetic_data": (
            "SYNTHETIC" in data_source
        ),
        "total_measured_cycles": int(
            full_df["cycle"].max()
        ),
        "primary_metrics_are_from": (
            "chronological_early_to_future_split"
        ),
        "interpolation_split_metrics": (
            metrics_interp
        ),
        "early_to_future_metrics": (
            metrics_extrapolation
        ),
        "training_cycle_cutoff_requested": int(
            args.train_until
        ),
        "training_cycle_cutoff_actual": int(
            split_cycle
        ),
        "split_was_adapted": bool(
            was_adapted
        ),
        "forecast_target_cycle": int(
            args.forecast_to
        ),
        "forecast_method": (
            "bounded empirical degradation forecast"
        ),
        "features_used": feature_cols,
        "model": "RandomForestRegressor",
    }

    try:

        save_metrics(
            metrics,
            extra_info,
        )

    except Exception as exc:

        print(
            "\n[main] ERROR while saving metrics:"
        )

        print(
            f"[main] {exc}"
        )

        return 1

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    print(
        "\n" + "#" * 60
    )

    print(
        "# Pipeline complete. Check the outputs/ folder for:"
    )

    print(
        "#   outputs/figures/      -> degradation + prediction/error plots"
    )

    print(
        "#   outputs/predictions/  -> test set + forecast CSVs"
    )

    print(
        "#   outputs/metrics/      -> metrics.json"
    )

    print(
        "#   data/processed/       -> processed_battery_data.csv"
    )

    print(
        "#" * 60
        + "\n"
    )

    if "SYNTHETIC" in data_source:

        print(
            "*** REMINDER: This run used SYNTHETIC fallback data."
        )

        print(
            "*** Use a real NASA .mat or compatible CSV for"
        )

        print(
            "*** final experimental results.\n"
        )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(
        main()
    )