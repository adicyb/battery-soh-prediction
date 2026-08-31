"""
evaluation.py

Computes evaluation metrics for the SoH prediction model.

Primary metric required by the project: MAE (Mean Absolute Error)
    MAE = mean(|Actual SoH - Predicted SoH|)

Also computes R^2 and RMSE as supporting metrics.
"""

import json
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")


def evaluate(y_true, y_pred) -> dict:
    """Compute MAE, R2, RMSE and return as a dict of Python floats."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        "MAE_percent": round(float(mae), 4),
        "R2": round(float(r2), 4),
        "RMSE_percent": round(rmse, 4),
        "n_test_samples": int(len(y_true)),
    }


def print_metrics_report(metrics: dict, split_cycle=None, was_adapted: bool = False):
    print("\n" + "=" * 50)
    print("           MODEL PERFORMANCE REPORT")
    print("=" * 50)
    if split_cycle is not None:
        print(f"Train/test split at cycle: {split_cycle}"
              f"{'  (ADAPTED - see note below)' if was_adapted else ''}")
    else:
        print("Train/test split: random 75/25 (interpolation)")
    print(f"Test samples evaluated   : {metrics['n_test_samples']}")
    print("-" * 50)
    print(f"MAE  (Mean Absolute Error) : {metrics['MAE_percent']:.2f} %")
    print(f"R2   (Coefficient of Det.) : {metrics['R2']:.4f}")
    print(f"RMSE (Root Mean Sq. Error) : {metrics['RMSE_percent']:.2f} %")
    print("=" * 50 + "\n")


def save_metrics(metrics: dict, extra_info: dict, filename: str = "metrics.json"):
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = os.path.join(METRICS_DIR, filename)
    payload = {**metrics, **extra_info}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[evaluation] Saved metrics -> {out_path}")
    return out_path


if __name__ == "__main__":
    # quick smoke test
    y_true = [100, 95, 90, 85]
    y_pred = [99, 94, 91, 83]
    m = evaluate(y_true, y_pred)
    print(m)
