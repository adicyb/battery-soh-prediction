"""
evaluation.py

Computes evaluation metrics for the SoH prediction model.

Primary metric required by the project: MAE (Mean Absolute Error)

    MAE = mean(|Actual SoH - Predicted SoH|)

Also computes:
    - R²
    - RMSE

The output directory is supplied by the main pipeline so that results from
different batteries/datasets can be stored independently.
"""

import json
import os

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate(y_true, y_pred) -> dict:
    """
    Compute MAE, R² and RMSE.

    Returns a dictionary containing ordinary Python numeric types so that it
    can be serialized directly to JSON.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    return {
        "MAE_percent": round(
            float(mae),
            4,
        ),
        "R2": round(
            float(r2),
            4,
        ),
        "RMSE_percent": round(
            rmse,
            4,
        ),
        "n_test_samples": int(
            len(y_true)
        ),
    }


def print_metrics_report(
    metrics: dict,
    split_cycle=None,
    was_adapted: bool = False,
):
    """
    Print a human-readable metrics report.
    """
    print(
        "\n" + "=" * 50
    )

    print(
        "           MODEL PERFORMANCE REPORT"
    )

    print(
        "=" * 50
    )

    if split_cycle is not None:

        print(
            f"Train/test split at cycle: {split_cycle}"
            f"{'  (ADAPTED - see note below)' if was_adapted else ''}"
        )

    else:

        print(
            "Train/test split: random 75/25 (interpolation)"
        )

    print(
        f"Test samples evaluated   : "
        f"{metrics['n_test_samples']}"
    )

    print(
        "-" * 50
    )

    print(
        f"MAE  (Mean Absolute Error) : "
        f"{metrics['MAE_percent']:.2f} %"
    )

    print(
        f"R2   (Coefficient of Det.) : "
        f"{metrics['R2']:.4f}"
    )

    print(
        f"RMSE (Root Mean Sq. Error) : "
        f"{metrics['RMSE_percent']:.2f} %"
    )

    print(
        "=" * 50
        + "\n"
    )


def save_metrics(
    metrics: dict,
    extra_info: dict,
    output_dir: str,
    filename: str = "metrics.json",
):
    """
    Save metrics and metadata to a caller-specified directory.

    Parameters
    ----------
    metrics:
        Evaluation metrics dictionary.

    extra_info:
        Additional experiment metadata.

    output_dir:
        Dataset-specific metrics directory.

    filename:
        Metrics filename.
    """
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    out_path = os.path.join(
        output_dir,
        filename,
    )

    payload = {
        **metrics,
        **extra_info,
    }

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    print(
        f"[evaluation] Saved metrics -> {out_path}"
    )

    return out_path


if __name__ == "__main__":

    # Quick smoke test.
    y_true = [
        100,
        95,
        90,
        85,
    ]

    y_pred = [
        99,
        94,
        91,
        83,
    ]

    metrics = evaluate(
        y_true,
        y_pred,
    )

    print(
        metrics
    )