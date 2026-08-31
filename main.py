"""
main.py

Lithium-Ion Battery State of Health (SoH) Prediction Using Machine Learning
Based on Charge Cycle Data.

Runs the complete pipeline end to end:
    1. Load battery data (real NASA PCoE .mat if present in data/raw/,
       otherwise a clearly-labeled synthetic fallback dataset).
    2. Preprocess data and compute SoH.
    3. Engineer ML features.
    4. Split into early-life train / later-life test (train on cycles up
       to ~1500, test on the remainder -- adapted if fewer cycles exist).
    5. Train a Random Forest regression model.
    6. Evaluate on the test set (MAE, R2, RMSE).
    7. Forecast SoH out to cycle 3000 (clearly labeled as forecast).
    8. Generate all required plots.
    9. Save processed data, predictions, and metrics to outputs/.

Run with:  python main.py
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_battery_data
from preprocessing import preprocess
from feature_engineering import build_features
from model import (train_test_split_by_cycle, train_test_split_interpolation,
                    train_model, forecast_to_cycle)
from evaluation import evaluate, print_metrics_report, save_metrics
from visualization import (plot_degradation_curve, plot_actual_vs_predicted,
                             plot_residuals)

PREDICTIONS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "predictions")
TARGET_FORECAST_CYCLE = 3000
TRAIN_UP_TO_CYCLE = 1500


def main():
    print("\n" + "#" * 60)
    print("# Lithium-Ion Battery SoH Prediction Pipeline")
    print("#" * 60)

    # --- Step 1: Load data ---
    raw_df, data_source, message = load_battery_data(preferred_battery_id="B0005")
    print(message)
    battery_id = raw_df["battery_id"].iloc[0]

    # --- Step 2 & 3: Preprocess + compute SoH ---
    processed_df = preprocess(raw_df, data_source, battery_id)
    print(f"\n[main] Data source for this run: {data_source}")
    print(f"[main] Battery ID: {battery_id}")
    print(f"[main] Total measured cycles available: {len(processed_df)}")
    print(f"[main] Initial reference capacity: {processed_df['initial_capacity_ah'].iloc[0]:.4f} Ah")
    print(f"[main] Final measured SoH: {processed_df['soh'].iloc[-1]:.2f}% "
          f"at cycle {processed_df['cycle'].iloc[-1]}")

    # --- Step 4: Feature engineering ---
    X, y, feature_cols, full_df = build_features(processed_df)
    print(f"\n[main] Features used ({len(feature_cols)}): {feature_cols}")

    # --- Step 5: Train/test split (early-life -> later-life) ---
    train_df, test_df, split_cycle, was_adapted = train_test_split_by_cycle(
        full_df, train_up_to_cycle=TRAIN_UP_TO_CYCLE
    )
    if was_adapted:
        print(f"\n[main] NOTE: Dataset has only {full_df['cycle'].max()} measured cycles, "
              f"fewer than the requested {TRAIN_UP_TO_CYCLE}-cycle training window.")
        print(f"[main] Adapted split: training on cycles 1-{split_cycle} (70%), "
              f"testing on cycles {split_cycle + 1}-{full_df['cycle'].max()} (30%).")
    else:
        print(f"\n[main] Training on cycles 1-{split_cycle}, "
              f"testing on cycles {split_cycle + 1}-{full_df['cycle'].max()}.")

    X_train, y_train = train_df[feature_cols], train_df["soh"]
    X_test, y_test = test_df[feature_cols], test_df["soh"]

    if len(X_test) == 0:
        print("[main] ERROR: Test set is empty. Not enough cycles to evaluate. Exiting.")
        return

    # --- Step 6: Train model (future-extrapolation split) ---
    model = train_model(X_train, y_train)
    print(f"\n[main] Random Forest model trained on {len(X_train)} samples "
          f"(early-cycle / future-extrapolation split).")

    # --- Step 7: Predict on test set & evaluate (future-extrapolation) ---
    y_pred_test = model.predict(X_test)
    metrics_extrapolation = evaluate(y_test, y_pred_test)
    print("\n[main] === Evaluation 1: EARLY-CYCLE -> FUTURE-CYCLE EXTRAPOLATION ===")
    print("[main] (train on early cycles only, test on later, never-seen cycles --")
    print("[main]  this is the experiment from the project notes; tree-based models")
    print("[main]  are fundamentally limited at extrapolating past their training range,")
    print("[main]  which is reflected honestly in these numbers)")
    print_metrics_report(metrics_extrapolation, split_cycle, was_adapted)

    # --- Step 6b/7b: A second, complementary interpolation-style evaluation ---
    train_df_i, test_df_i = train_test_split_interpolation(full_df)
    model_interp = train_model(train_df_i[feature_cols], train_df_i["soh"])
    y_pred_interp = model_interp.predict(test_df_i[feature_cols])
    metrics_interp = evaluate(test_df_i["soh"], y_pred_interp)
    print("[main] === Evaluation 2: RANDOM (INTERPOLATION) SPLIT ===")
    print("[main] (test cycles scattered across the full measured range -- the")
    print("[main]  setting Random Forest is genuinely well suited to)")
    print_metrics_report(metrics_interp, split_cycle=None, was_adapted=False)

    # Primary reported metrics = the time-ordered early->future split because
    # it matches the project's forecasting objective.  The random split is
    # retained as a secondary interpolation benchmark for model sanity-checking.
    metrics = metrics_extrapolation

    # --- Step 8: Forecast to 3000 cycles ---
    forecast_df = forecast_to_cycle(full_df, model, feature_cols, target_cycle=TARGET_FORECAST_CYCLE)
    if len(forecast_df) > 0:
        print(f"[main] Forecast generated from cycle {full_df['cycle'].max() + 1} "
              f"to cycle {TARGET_FORECAST_CYCLE}.")
        print(f"[main] Forecasted SoH at cycle {TARGET_FORECAST_CYCLE}: "
              f"{forecast_df['soh_forecast'].iloc[-1]:.2f}%")
        # Check measured data FIRST.  B0005 is already below 80% SoH by the
        # end of the experiment, so reporting a forecast EOL at cycle 169
        # would be misleading.
        measured_eol = full_df[full_df["soh"] <= 80.0]
        if len(measured_eol) > 0:
            print(f"[main] Measured data crosses 80% EOL threshold at cycle "
                  f"{int(measured_eol['cycle'].iloc[0])}.")
        else:
            forecast_eol = forecast_df[forecast_df["soh_forecast"] <= 80.0]
            if len(forecast_eol) > 0:
                print(f"[main] Forecast crosses 80% EOL threshold at approx. cycle "
                      f"{int(forecast_eol['cycle'].iloc[0])}.")
            else:
                print(f"[main] Forecast does not reach 80% EOL threshold by cycle "
                      f"{TARGET_FORECAST_CYCLE}.")
    else:
        print(f"[main] No forecast needed -- measured data already extends to/past "
              f"{TARGET_FORECAST_CYCLE} cycles.")

    # --- Step 9: Plots ---
    # Main Actual-vs-Predicted and residual plots now correspond to the
    # primary time-ordered future-cycle evaluation.  A second interpolation
    # plot is also saved to show the RF's performance within the measured range.
    plot_degradation_curve(full_df, forecast_df, data_source, battery_id)
    plot_actual_vs_predicted(
        test_df["soh"].values, y_pred_test, test_df["cycle"].values,
        filename="actual_vs_predicted_future.png"
    )
    plot_actual_vs_predicted(
        test_df_i["soh"].values, y_pred_interp, test_df_i["cycle"].values,
        filename="actual_vs_predicted_interpolation.png"
    )
    plot_residuals(
        test_df["soh"].values, y_pred_test, test_df["cycle"].values,
        filename="prediction_residuals_future.png"
    )

    # --- Step 10: Save predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    interp_predictions_df = test_df_i[["cycle"]].copy()
    interp_predictions_df["actual_soh"] = test_df_i["soh"].values
    interp_predictions_df["predicted_soh"] = y_pred_interp
    interp_predictions_df["absolute_error"] = (
        interp_predictions_df["actual_soh"] - interp_predictions_df["predicted_soh"]
    ).abs()
    interp_predictions_df["type"] = "random_split_measured_vs_predicted"
    interp_pred_path = os.path.join(PREDICTIONS_DIR, "test_set_predictions.csv")
    interp_predictions_df.to_csv(interp_pred_path, index=False)
    print(f"\n[main] Saved random-split test predictions -> {interp_pred_path}")

    extrapolation_predictions_df = test_df[["cycle"]].copy()
    extrapolation_predictions_df["actual_soh"] = y_test.values
    extrapolation_predictions_df["predicted_soh"] = y_pred_test
    extrapolation_predictions_df["absolute_error"] = (
        extrapolation_predictions_df["actual_soh"] - extrapolation_predictions_df["predicted_soh"]
    ).abs()
    extrapolation_predictions_df["type"] = "early_to_future_cycle_extrapolation"
    extrap_pred_path = os.path.join(PREDICTIONS_DIR, "early_to_future_extrapolation_predictions.csv")
    extrapolation_predictions_df.to_csv(extrap_pred_path, index=False)
    print(f"[main] Saved early->future extrapolation predictions -> {extrap_pred_path}")

    if len(forecast_df) > 0:
        forecast_out = forecast_df[["cycle", "soh_forecast"]].copy()
        forecast_out["type"] = "forecast_beyond_measured_data"
        forecast_path = os.path.join(PREDICTIONS_DIR, "forecast_3000_cycles.csv")
        forecast_out.to_csv(forecast_path, index=False)
        print(f"[main] Saved 3000-cycle forecast -> {forecast_path}")

    # --- Step 11: Save metrics ---
    extra_info = {
        "battery_id": battery_id,
        "data_source": data_source,
        "is_synthetic_data": "SYNTHETIC" in data_source,
        "total_measured_cycles": int(full_df["cycle"].max()),
        "primary_metrics_are_from": "early_to_future_cycle_extrapolation_split",
        "interpolation_split_metrics": metrics_interp,
        "early_to_future_extrapolation_split_metrics": metrics_extrapolation,
        "extrapolation_train_test_split_cycle": int(split_cycle),
        "extrapolation_split_was_adapted": bool(was_adapted),
        "forecast_target_cycle": TARGET_FORECAST_CYCLE,
        "forecast_method": "bounded empirical stretched-exponential curve fit (scipy curve_fit), NOT the Random Forest",
        "features_used": feature_cols,
        "model": "RandomForestRegressor",
    }
    save_metrics(metrics, extra_info)

    print("\n" + "#" * 60)
    print("# Pipeline complete. Check the outputs/ folder for:")
    print("#   outputs/figures/      -> degradation + prediction/error plots")
    print("#   outputs/predictions/  -> test set + forecast CSVs")
    print("#   outputs/metrics/      -> metrics.json")
    print("#   data/processed/       -> processed_battery_data.csv")
    print("#" * 60 + "\n")

    if "SYNTHETIC" in data_source:
        print("*** REMINDER: This run used SYNTHETIC fallback data, not real NASA")
        print("*** measurements, because no .mat file was found in data/raw/.")
        print("*** Download B0005.mat (or similar) and re-run for real results.\n")


if __name__ == "__main__":
    main()
