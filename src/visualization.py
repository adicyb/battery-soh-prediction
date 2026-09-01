"""
visualization.py

Generates all required plots for the project:
  1. SoH vs Cycle degradation curve (measured + configurable forecast horizon, 80% EOL line)
  2. Actual vs Predicted SoH line plot (on a selected test set)
  3. Prediction error / residual plot (on a selected test set)

All figures are saved as high-resolution PNGs into the
dataset-specific output directory supplied by the pipeline.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EOL_THRESHOLD = 80.0


def _setup(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_degradation_curve(
    full_df,
    forecast_df,
    data_source: str,
    battery_id: str,
    forecast_target_cycle: int = 3000,
    filename: str = "soh_vs_cycles.png",
    output_dir: str = ".",
):
    """
    Plot SoH (%) vs Number of Charge Cycles.

    Parameters
    ----------
    full_df:
        Measured battery data.

    forecast_df:
        Long-term model forecast beyond the measured range.

    data_source:
        Dataset source label.

    battery_id:
        Battery/cell identifier.

    forecast_target_cycle:
        Final cycle shown on the forecast axis.
        Defaults to 3000 to preserve the original project behavior.

    filename:
        Output PNG filename.

    The measured and forecast regions are clearly distinguished.
    """
    _setup(output_dir)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    is_synthetic = "SYNTHETIC" in data_source

    measured_label = (
        "Measured SoH (SYNTHETIC DEMO DATA - not real NASA measurements)"
        if is_synthetic
        else f"Measured SoH (NASA PCoE, battery {battery_id})"
    )

    # ---------------------------------------------------------------
    # Measured data
    # ---------------------------------------------------------------

    ax.plot(
        full_df["cycle"],
        full_df["soh"],
        color="#1f6feb",
        linewidth=1.8,
        label=measured_label,
        zorder=3,
    )

    step = max(1, len(full_df) // 60)

    ax.scatter(
        full_df["cycle"].iloc[::step],
        full_df["soh"].iloc[::step],
        color="#1f6feb",
        s=14,
        zorder=4,
    )

    # ---------------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------------

    if forecast_df is not None and len(forecast_df) > 0:

        measured_end = float(
            full_df["cycle"].iloc[-1]
        )

        # Prevent the forecast shading from extending backwards if
        # the requested target is smaller than the measured range.
        forecast_end = max(
            float(forecast_target_cycle),
            measured_end,
        )

        ax.axvspan(
            measured_end,
            forecast_end,
            color="#f3f4f6",
            alpha=0.45,
            zorder=0,
            label="Forecast region (not experimentally measured)",
        )

        # Connect final measured point to first forecast point for
        # visual continuity.
        connect_cycle = np.concatenate(
            [
                [measured_end],
                forecast_df["cycle"].values,
            ]
        )

        connect_soh = np.concatenate(
            [
                [full_df["soh"].iloc[-1]],
                forecast_df["soh_forecast"].values,
            ]
        )

        ax.plot(
            connect_cycle,
            connect_soh,
            color="#d97706",
            linewidth=1.8,
            linestyle="--",
            label="Forecast / Predicted SoH",
            zorder=3,
        )

    # ---------------------------------------------------------------
    # EOL reference
    # ---------------------------------------------------------------

    ax.axhline(
        EOL_THRESHOLD,
        color="#dc2626",
        linestyle=":",
        linewidth=1.6,
        label=f"{EOL_THRESHOLD:.0f}% End-of-Life (EOL) threshold",
        zorder=2,
    )

    # ---------------------------------------------------------------
    # Measured/forecast boundary
    # ---------------------------------------------------------------

    measured_end = float(
        full_df["cycle"].iloc[-1]
    )

    ax.axvline(
        measured_end,
        color="gray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.6,
        zorder=1,
    )

    # Keep annotation inside the actual graph area.
    annotation_offset = max(
        10,
        int(forecast_target_cycle * 0.01),
    )

    ax.text(
        measured_end + annotation_offset,
        6,
        "last measured\ncycle",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    # ---------------------------------------------------------------
    # Forecast annotation
    # ---------------------------------------------------------------

    if forecast_df is not None and len(forecast_df) > 0:

        annotation_x = (
            measured_end
            + 0.50
            * max(
                0,
                forecast_target_cycle - measured_end,
            )
        )

        ax.text(
            annotation_x,
            103,
            f"{int(forecast_target_cycle)}-cycle region is "
            "extrapolated; not measured",
            fontsize=8.5,
            color="dimgray",
            ha="center",
            va="top",
        )

    # ---------------------------------------------------------------
    # Axes
    # ---------------------------------------------------------------

    ax.set_xlim(
        0,
        max(
            300,
            float(forecast_target_cycle),
        ),
    )

    ax.set_ylim(
        0,
        108,
    )

    ax.set_xlabel(
        "Number of Charge Cycles"
    )

    ax.set_ylabel(
        "State of Health (SoH) [%]"
    )

    title_suffix = (
        " -- SYNTHETIC DEMO DATA"
        if is_synthetic
        else ""
    )

    ax.set_title(
        f"Battery State of Health vs. Charge Cycles{title_suffix}",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.tight_layout()

    out_path = os.path.join(
        output_dir,
        filename,
    )

    fig.savefig(
        out_path
    )

    plt.close(fig)

    print(
        f"[visualization] Saved -> {out_path}"
    )

    return out_path


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    cycles,
    filename: str = "actual_vs_predicted.png",
    title: str = "Actual vs Predicted SoH",
    output_dir: str = ".",
):
    """Plot Actual SoH vs Predicted SoH on the held-out test cycles."""
    _setup(output_dir)
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(cycles, y_true, color="#1f6feb", linewidth=1.8, marker="o", markersize=3,
            label="Actual SoH (measured, held-out test cycles)")
    ax.plot(cycles, y_pred, color="#dc2626", linewidth=1.8, linestyle="--", marker="x",
            markersize=3, label="Predicted SoH (ML model)")

    ax.set_xlabel("Number of Charge Cycles")
    ax.set_ylabel("State of Health (SoH) [%]")
    ax.set_title(
    title,
    fontsize=13,
    fontweight="bold",
)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(
    output_dir,
    filename,
)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved -> {out_path}")
    return out_path


def plot_residuals(
    y_true,
    y_pred,
    cycles,
    filename: str = "prediction_residuals.png",
    output_dir: str = ".",
):
    """Plot prediction error (residuals) = Actual - Predicted, vs cycle number."""
    _setup(output_dir)
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

    fig.suptitle(
    "Prediction Residuals — Early-Life → Future-Cycle Test",
    fontsize=13,
    fontweight="bold",
)
    fig.tight_layout()
    out_path = os.path.join(
    output_dir,
    filename,
)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[visualization] Saved -> {out_path}")
    return out_path
