"""
visualization.py

Generates all required plots for the project:
  1. SoH vs Cycle degradation curve (measured + forecast to 3000 cycles, 80% EOL line)
  2. Actual vs Predicted SoH line plot (on a selected test set)
  3. Prediction error / residual plot (on a selected test set)

All figures are saved as high-resolution PNGs into outputs/figures/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
EOL_THRESHOLD = 80.0


def _setup():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_degradation_curve(full_df, forecast_df, data_source: str,
                            battery_id: str, filename: str = "soh_vs_cycles.png"):
    """
    Plot SoH (%) vs Number of Charge Cycles.
    - Measured data plotted as a solid line/markers.
    - Forecast (if any) plotted as a dashed line, clearly labeled.
    - 80% EOL reference line included.
    - Extends x-axis to 3000 cycles.
    """
    _setup()
    fig, ax = plt.subplots(figsize=(11, 6.5))

    is_synthetic = "SYNTHETIC" in data_source
    measured_label = ("Measured SoH (SYNTHETIC DEMO DATA - not real NASA measurements)"
                       if is_synthetic else f"Measured SoH (NASA PCoE, battery {battery_id})")

    ax.plot(full_df["cycle"], full_df["soh"], color="#1f6feb", linewidth=1.8,
            label=measured_label, zorder=3)
    ax.scatter(full_df["cycle"].iloc[::max(1, len(full_df)//60)],
               full_df["soh"].iloc[::max(1, len(full_df)//60)],
               color="#1f6feb", s=14, zorder=4)

    if forecast_df is not None and len(forecast_df) > 0:
        measured_end = float(full_df["cycle"].iloc[-1])
        ax.axvspan(measured_end, 3000, color="#f3f4f6", alpha=0.45, zorder=0,
                   label="Forecast region (not experimentally measured)")
        # connect the last measured point to the forecast for visual continuity
        connect_cycle = np.concatenate([[measured_end], forecast_df["cycle"].values])
        connect_soh = np.concatenate([[full_df["soh"].iloc[-1]], forecast_df["soh_forecast"].values])
        ax.plot(connect_cycle, connect_soh, color="#d97706", linewidth=1.8,
                linestyle="--", label="Forecast / Predicted SoH", zorder=3)

    ax.axhline(EOL_THRESHOLD, color="#dc2626", linestyle=":", linewidth=1.6,
               label=f"{EOL_THRESHOLD:.0f}% End-of-Life (EOL) threshold", zorder=2)

    ax.axvline(full_df["cycle"].iloc[-1], color="gray", linestyle="-", linewidth=0.8,
               alpha=0.6, zorder=1)
    ax.text(full_df["cycle"].iloc[-1] + 25, 6, "last measured\ncycle", fontsize=8,
            color="gray", va="bottom")
    ax.text(1550, 103, "3000-cycle region is extrapolated; not measured",
            fontsize=8.5, color="dimgray", ha="center", va="top")

    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Number of Charge Cycles")
    ax.set_ylabel("State of Health (SoH) [%]")
    title_suffix = " -- SYNTHETIC DEMO DATA" if is_synthetic else ""
    ax.set_title(f"Battery State of Health vs. Charge Cycles{title_suffix}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved -> {out_path}")
    return out_path


def plot_actual_vs_predicted(y_true, y_pred, cycles, filename: str = "actual_vs_predicted.png"):
    """Plot Actual SoH vs Predicted SoH on the held-out test cycles."""
    _setup()
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(cycles, y_true, color="#1f6feb", linewidth=1.8, marker="o", markersize=3,
            label="Actual SoH (measured, held-out test cycles)")
    ax.plot(cycles, y_pred, color="#dc2626", linewidth=1.8, linestyle="--", marker="x",
            markersize=3, label="Predicted SoH (ML model)")

    ax.set_xlabel("Number of Charge Cycles")
    ax.set_ylabel("State of Health (SoH) [%]")
    ax.set_title("Actual vs Predicted SoH (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved -> {out_path}")
    return out_path


def plot_residuals(y_true, y_pred, cycles, filename: str = "prediction_residuals.png"):
    """Plot prediction error (residuals) = Actual - Predicted, vs cycle number."""
    _setup()
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].scatter(cycles, residuals, color="#7c3aed", s=18, alpha=0.8)
    axes[0].set_xlabel("Number of Charge Cycles")
    axes[0].set_ylabel("Residual (Actual - Predicted) [%]")
    axes[0].set_title("Prediction Error vs Cycle")

    axes[1].hist(residuals, bins=15, color="#7c3aed", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Residual (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    fig.suptitle("Prediction Error Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved -> {out_path}")
    return out_path
