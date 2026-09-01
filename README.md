# Lithium-Ion Battery State of Health (SoH) Prediction Using Machine Learning

A reusable machine-learning pipeline for predicting the **State of Health (SoH)** of lithium-ion batteries from early-life operating-cycle data.

The main objective is to determine whether battery health can be estimated from data collected during the early portion of a battery's lifetime, rather than waiting for the complete degradation history.

**The project supports:**

- NASA PCoE Li-ion battery `.mat` datasets
- Generic battery `.csv` datasets
- Multiple battery IDs without changing the code
- Capacity-based or directly supplied SoH
- Chronological early-life → future-cycle evaluation
- Random interpolation benchmarking
- Random Forest regression
- MAE, RMSE, and R² evaluation
- Long-term empirical degradation extrapolation
- Dataset-specific output folders
- Configurable training and forecast horizons

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objective](#2-objective)
3. [Proposed Solution](#3-proposed-solution)
4. [Dataset](#4-dataset)
5. [SoH Definition](#5-soh-definition)
6. [Features Used](#6-features-used)
7. [Machine Learning Model](#7-machine-learning-model)
8. [Training and Evaluation Strategy](#8-training-and-evaluation-strategy)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Long-Term Degradation Forecast](#10-long-term-degradation-forecast)
11. [End-of-Life Threshold](#11-end-of-life-threshold)
12. [Verified NASA B0005 Results](#12-verified-nasa-b0005-results)
13. [3000-Cycle Forecast Interpretation](#13-3000-cycle-forecast-interpretation)
14. [Project Structure](#14-project-structure)
15. [Running the Project](#15-running-the-project)
16. [Running a Specific Dataset](#16-running-a-specific-dataset)
17. [Output Files](#17-output-files)
18. [Data Integrity and Honesty](#18-data-integrity-and-honesty)
19. [Important Limitations](#19-important-limitations)
20. [Key Project Finding](#20-key-project-finding)
21. [Conclusion](#21-conclusion)

---

## 1. Problem Statement

Lithium-ion batteries gradually lose usable capacity as they undergo repeated charge/discharge cycles. This degradation reduces the battery's State of Health (SoH) and eventually limits its useful operating life.

The problem addressed by this project is:

> **Can the future State of Health of a lithium-ion battery be predicted using only early-life battery operating data, without observing the battery for its entire lifetime?**

This is relevant to applications such as:

- Battery Management Systems (BMS)
- Electric vehicles
- Energy storage systems
- Portable electronics
- Predictive maintenance

The project therefore focuses on **early-life prediction of future battery degradation**.

---

## 2. Objective

The project has three main objectives:

1. Calculate battery State of Health from measured battery capacity when required.
2. Train a machine-learning regression model using early-cycle battery operation data.
3. Evaluate the model on later, unseen cycles, and provide a separate long-term degradation extrapolation toward a configurable cycle horizon (3000 cycles by default).

The distinction between **future-cycle prediction** and **long-term extrapolation** is important and is maintained throughout the project.

---

## 3. Proposed Solution

```text
Battery Dataset
       │
       ▼
Data Loading
       │
       ├── NASA .mat
       └── Generic .csv
       │
       ▼
Dataset Validation / Standardization
       │
       ▼
Preprocessing
       │
       ├── Cleaning
       └── SoH calculation / preservation
       │
       ▼
Feature Engineering
       │
       ├── Voltage features
       ├── Current features
       ├── Temperature features
       └── Derived features
       │
       ▼
Chronological Train/Test Split
       │
       ├── Early-life cycles → Training
       └── Later cycles → Testing
       │
       ▼
Random Forest Regressor
       │
       ▼
Future-Cycle SoH Prediction
       │
       ▼
MAE / RMSE / R²
       │
       ▼
Separate Long-Term Empirical Forecast
       │
       ▼
Plots + CSV + Metrics
```

---

## 4. Dataset

### Primary Dataset

The project is designed to work with the **NASA Prognostics Center of Excellence (PCoE) Li-ion Battery Aging Dataset**.

Example batteries include: `B0005`, `B0006`, `B0007`, `B0018`.

The original NASA dataset contains battery cycling measurements including voltage, current, temperature, and discharge capacity information.

**NASA sources:**
- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

### Adding NASA Data

Place the downloaded `.mat` files inside `data/raw/`:

```
data/raw/B0005.mat
data/raw/B0006.mat
```

The program automatically searches `data/raw/` for supported NASA `.mat` files and selects the preferred battery when requested. A specific input can also be supplied:

```bash
python main.py --input data/raw/B0005.mat
```

### Generic CSV Support

The pipeline also accepts generic battery CSV datasets:

```bash
python main.py --input data/raw/battery.csv
```

The loader standardizes common alternative column names into the project's internal schema. Typical information includes:

- `cycle`
- `capacity`
- `voltage`
- `current`
- `temperature`

(or equivalent aliases). A dataset may also provide SoH directly, in which case capacity-based SoH calculation is not required.

---

## 5. SoH Definition

When capacity is available, State of Health is calculated as:

```
SoH(n) = (Capacity(n) / Capacity_initial) × 100
```

where:
- `Capacity(n)` = measured discharge capacity at cycle `n`
- `Capacity_initial` = capacity at the earliest valid measured cycle

Therefore:
- Initial valid cycle → approximately 100% SoH
- Later cycles → decreasing SoH as capacity decreases

The project uses the first valid measured capacity as the reference, because a separate manufacturer nameplate capacity is not supplied as an independent reference in the project data pipeline.

---

## 6. Features Used

The current feature pipeline uses cycle-level voltage, current, and temperature statistics.

| Feature | Description |
|---|---|
| `cycle` | Charge/discharge cycle number |
| `voltage_mean` | Mean measured discharge voltage |
| `voltage_min` | Minimum measured discharge voltage |
| `voltage_max` | Maximum measured discharge voltage |
| `current_mean` | Mean measured discharge current |
| `current_min` | Minimum measured discharge current |
| `temperature_mean` | Mean measured temperature |
| `temperature_max` | Maximum measured temperature |
| `voltage_range` | `voltage_max − voltage_min` |
| `temperature_rise` | `temperature_max − temperature_mean` |

**Target variable:** `soh` (measured in percent)

### Internal Resistance

Internal resistance/impedance is **not** currently used as a primary ML feature, because the discharge records used by the current pipeline do not provide one consistently comparable per-cycle impedance value across datasets. The architecture can be extended later to incorporate impedance information.

---

## 7. Machine Learning Model

### Random Forest Regressor

The primary machine-learning model is `RandomForestRegressor` from scikit-learn.

The current configuration uses:

```
n_estimators     = 300
max_depth        = 10
min_samples_leaf = 2
```

### Why Random Forest?

Random Forest was selected because this project is fundamentally a tabular regression problem with a relatively small number of engineered features.

**Advantages:**
- Handles nonlinear relationships
- Does not require feature scaling
- Robust to measurement noise
- Works well with mixed feature relationships
- Fast to train for this dataset size
- Straightforward to explain during a viva

It is also preferable to introducing a deep neural network without sufficient training data or a demonstrated need for deep learning.

---

## 8. Training and Evaluation Strategy

The project deliberately performs two different evaluations. They answer different questions.

### A. Early-Life → Future-Cycle Prediction (Primary Evaluation)

The data is split **chronologically**:

```
Early cycles
    │
    ├── Training
    │
    ▼
Random Forest
    │
    ▼
Later unseen cycles
    │
    └── Testing
```

The model is trained only on early-life data and evaluated on later cycles that were not present in the training set.

The default training target is **1500 cycles**. However, if a dataset contains fewer than 1500 cycles, the pipeline automatically adapts.

For example, NASA B0005 contains only 168 measured cycles, so the pipeline uses:

```
Training: cycles 1–117
Testing:  cycles 118–168
```

This is approximately a 70% / 30% split. The adaptation is printed explicitly by the program.

**Why this evaluation matters:** this is the experiment most closely related to the actual project objective — using early-life data to estimate future battery health. The test cycles are genuine later measured cycles, so predictions can be compared against real ground truth.

### B. Random Interpolation Benchmark (Secondary Evaluation)

A second 75/25 random split is also performed, where cycles are randomly distributed between training and testing:

```
Measured cycle range
       │
       ├── Random 75% → Training
       │
       └── Random 25% → Testing
```

This is an interpolation task rather than a true future-extrapolation task. It is included because Random Forest is generally much better suited to predicting values within the feature range represented during training.

> **Note:** "Early → Future" and "Random Interpolation" must not be interpreted as the same experiment. The chronological early-life → future-cycle result is the **primary** project evaluation. The random interpolation result is a **secondary** benchmark.

---

## 9. Evaluation Metrics

The main metric required by the project is **Mean Absolute Error (MAE)**:

```
MAE = mean(|Actual SoH − Predicted SoH|)
```

For example, `MAE = 2.0%` means the predictions differ from the measured SoH by an average of approximately 2 percentage points. MAE is useful because it is directly expressed in the same unit as the target (SoH, %).

**Additional metrics reported by the pipeline:**

| Metric | Description |
|---|---|
| **R²** | Measures how well the predictions explain the variance of the target |
| **RMSE** | Penalizes larger prediction errors more strongly than MAE |

All three metrics are generated by the pipeline and saved in the output metrics file.

---

## 10. Long-Term Degradation Forecast

A separate long-term degradation forecast is generated after the ML evaluation.

### Why Isn't Random Forest Used for 3000-Cycle Extrapolation?

Random Forest is excellent at interpolation but is not a reliable mathematical extrapolator outside the feature range represented during training. For example, in the B0005 experiment, training cycles span 1–117 — a Random Forest should not be expected to reliably infer behavior thousands of cycles beyond this range.

Therefore, the project intentionally separates **machine-learning prediction** from **long-term empirical extrapolation**.

### Forecast Model

The long-term trend uses an empirical linear + exponential degradation model:

```
SoH(n) = 100 − a(n − 1) − b(1 − exp(−(n − 1)/τ))
```

where:
- `n` = cycle number
- `a` = approximately linear degradation component
- `b` = nonlinear degradation component
- `τ` = characteristic cycle scale of the nonlinear term

The parameters are fitted to the measured SoH trajectory using nonlinear least squares. The forecast is then extended from the final measured cycle to the requested forecast horizon.

### Important Interpretation

This forecast is **model-based extrapolation**, and not **experimentally measured data**.

For NASA B0005, the current dataset contains 168 measured discharge cycles. Cycles after 168 are extrapolated rather than experimentally observed. The program explicitly reports this limitation.

---

## 11. End-of-Life Threshold

The project uses **80% SoH** as the End-of-Life (EOL) reference threshold. The pipeline checks whether the measured battery trajectory reaches this threshold.

For the current NASA B0005 run: **80% EOL ≈ cycle 101**.

This means the battery had already dropped below 80% SoH before the measured experiment ended at cycle 168.

---

## 12. Verified NASA B0005 Results

The current verified run uses:

| Parameter | Value |
|---|---|
| Dataset | NASA PCoE |
| Battery | B0005 |
| Measured cycles | 168 |
| Initial reference capacity | 1.8565 Ah |
| Final measured SoH | 71.38% |

### Primary Evaluation — Chronological Early-Life → Future-Cycle Prediction

Training cycles: 1–117 · Testing cycles: 118–168

| Metric | Value |
|---|---|
| MAE | 4.02% |
| RMSE | 4.52% |
| R² | −3.4805 |

### Secondary Evaluation — Random Interpolation Benchmark

| Metric | Value |
|---|---|
| MAE | 0.42% |
| RMSE | 0.58% |
| R² | 0.9968 |

### Interpretation

The two results demonstrate an important distinction. The model performs very well when predicting within the measured data distribution (random interpolation, MAE = 0.42%), but performance is substantially worse when trained only on early-life data and evaluated on later aging cycles (early → future, MAE = 4.02%).

This demonstrates the difficulty of predicting future aging states from limited early-life observations using a standard tree-based regressor. The negative R² in the chronological evaluation is therefore **not hidden or replaced** with the interpolation result — both results are reported separately.

---

## 13. 3000-Cycle Forecast Interpretation

The project can generate a forecast toward 3000 cycles, or another horizon supplied through the command line:

```bash
python main.py --forecast-to 3000
```

```bash
python main.py --forecast-to 500
```

The 3000-cycle output should be interpreted as **an empirical long-term degradation extrapolation from the measured data**, not as a validated experimental measurement.

For B0005 specifically:

```
Measured:     cycles 1–168
Extrapolated: cycles 169–3000
```

The software explicitly warns when the requested horizon is far beyond the measured range.

---

## 14. Project Structure

```
battery-soh-prediction/
│
├── data/
│   ├── raw/
│   │   ├── B0005.mat
│   │   └── ...
│   │
│   └── processed/
│       ├── B0005_processed.csv
│       └── ...
│
├── outputs/
│   ├── B0005/
│   │   ├── figures/
│   │   │   ├── soh_vs_cycles.png
│   │   │   ├── actual_vs_predicted_future.png
│   │   │   ├── actual_vs_predicted_interpolation.png
│   │   │   └── prediction_residuals_future.png
│   │   │
│   │   ├── predictions/
│   │   │   ├── test_set_predictions.csv
│   │   │   ├── early_to_future_predictions.csv
│   │   │   └── forecast_3000_cycles.csv
│   │   │
│   │   └── metrics/
│   │       └── metrics.json
│   │
│   └── <another-battery-or-dataset>/
│       ├── figures/
│       ├── predictions/
│       └── metrics/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluation.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── run.bat
├── .gitignore
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `data_loader.py` | Loads NASA `.mat` battery data, generic CSV datasets, and synthetic fallback data when no real dataset is available. Converts inputs into a common internal representation. |
| `preprocessing.py` | Cleans invalid records, computes SoH when capacity is provided, preserves directly supplied SoH, and saves processed data. |
| `feature_engineering.py` | Builds the ML feature matrix from available battery measurements. |
| `model.py` | Contains chronological train/test splitting, random interpolation splitting, Random Forest training, and long-term empirical degradation forecasting. |
| `evaluation.py` | Calculates MAE, RMSE, and R², and writes metrics to JSON. |
| `visualization.py` | Generates the SoH degradation curve, actual vs. predicted future-cycle graph, actual vs. predicted interpolation graph, and prediction residual analysis. |
| `main.py` | Controls the complete end-to-end pipeline. |

---

## 15. Running the Project

### Windows — One-Command Execution

From the project root:

```cmd
run.bat
```

This:
1. Creates the virtual environment if necessary.
2. Installs dependencies.
3. Runs the pipeline.

### Manual Execution

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 16. Running a Specific Dataset

**NASA B0005:**
```bash
python main.py --input data/raw/B0005.mat
```

**Generic CSV:**
```bash
python main.py --input data/raw/battery.csv
```

**Select a battery** (when multiple NASA `.mat` files are available):
```bash
python main.py --battery B0006
```

**Change the training cutoff** (default: 1500 cycles):
```bash
python main.py --train-until 200
```

**Change the forecast horizon** (default: 3000 cycles):
```bash
python main.py --forecast-to 1000
```

**Combine options:**
```bash
python main.py --input data/raw/B0005.mat --train-until 100 --forecast-to 1000
```

---

## 17. Output Files

Each dataset receives its own output directory — for example, `outputs/B0005/`.

### Figures

| File | Description |
|---|---|
| `soh_vs_cycles.png` | Measured battery degradation and the long-term empirical forecast |
| `actual_vs_predicted_future.png` | Primary chronological early-life → future-cycle ML prediction |
| `actual_vs_predicted_interpolation.png` | Secondary random interpolation benchmark |
| `prediction_residuals_future.png` | Prediction residuals for the chronological future-cycle evaluation |

### Prediction CSVs

| File | Description |
|---|---|
| `test_set_predictions.csv` | Random interpolation predictions |
| `early_to_future_predictions.csv` | Primary early-life → future-cycle predictions |
| `forecast_<N>_cycles.csv` | Long-term empirical extrapolation toward the selected cycle horizon |

The forecast filename automatically follows the selected horizon, e.g. `forecast_500_cycles.csv`, `forecast_1000_cycles.csv`.

### Metrics

`metrics.json` contains the primary metrics plus experiment metadata, including:
- Battery ID
- Dataset source
- Measured cycle count
- Training cutoff and actual split used
- Forecast horizon
- Feature list
- Model name
- Interpolation metrics
- Early-to-future metrics

---

## 18. Data Integrity and Honesty

The project distinguishes clearly between:

- `NASA_PCoE_MEASURED`
- `SYNTHETIC_FALLBACK_NOT_REAL`

Synthetic data exists only as a fallback to demonstrate that the software pipeline can run when real measurements are unavailable. Synthetic output must **not** be presented as experimental NASA battery data. Before reporting results, always verify the dataset source recorded in the generated metrics.

---

## 19. Important Limitations

- **Limited long-term measurement data** — NASA B0005 currently provides only 168 measured discharge cycles in this experiment. A forecast extending to 3000 cycles is therefore a long-range extrapolation.
- **Random Forest extrapolation** — Random Forest is fundamentally better suited to interpolation within the distribution represented during training than to long-range extrapolation into unseen aging states. This is why the project separates ML prediction from empirical degradation extrapolation.
- **Model uncertainty** — the project does not claim that the empirical 3000-cycle forecast is experimentally validated. The extrapolated trajectory should be interpreted as a modeled degradation scenario rather than a guaranteed battery lifetime prediction.
- **SoH reference** — SoH is referenced to the earliest valid measured capacity rather than an independently supplied manufacturer nameplate capacity.
- **Feature availability** — the pipeline uses only features that can actually be extracted from the supplied dataset. Datasets with different measurements may therefore produce a different final feature set.

---

## 20. Key Project Finding

One of the most important findings from the current experiment is the difference between interpolation and genuine future prediction:

```
                    Random Forest

             ┌──────────────────────┐
             │  Measured-range      │
             │  interpolation       │
             │                      │
             │  MAE = 0.42%         │
             └──────────────────────┘
                       │
                       │
                       ▼
             ┌──────────────────────┐
             │  Early-life →        │
             │  future-cycle        │
             │                      │
             │  MAE = 4.02%         │
             └──────────────────────┘
```

This demonstrates that predicting battery aging is more difficult than simply fitting the observed SoH curve. The result supports the central motivation of the project:

> Early-life battery data contains information related to future health, but accurately predicting later aging states remains substantially more difficult than interpolation within the measured operating range.

---

## 21. Conclusion

This project implements a complete, reusable battery SoH prediction pipeline. The system:

- Loads real battery measurements.
- Converts battery capacity into SoH when required.
- Extracts voltage, current, and temperature features.
- Trains a Random Forest regression model.
- Evaluates the model on genuinely later, unseen cycles.
- Provides a secondary interpolation benchmark.
- Generates an empirical long-term degradation extrapolation.
- Produces plots, prediction CSVs, and machine-readable metrics.
- Separates results by battery/dataset to support repeated experiments.

The primary experiment demonstrates the feasibility and limitations of predicting future battery health from early-life operational data.

**For the verified NASA B0005 experiment:**

| | |
|---|---|
| Measured cycles | 168 |
| Final measured SoH | 71.38% |
| 80% EOL | ≈ cycle 101 |
| Early-life → future MAE / RMSE / R² | 4.02% / 4.52% / −3.4805 |
| Random interpolation MAE / RMSE / R² | 0.42% / 0.58% / 0.9968 |

The project therefore provides both a practical ML prediction pipeline and an explicit demonstration of the challenge of long-term battery degradation forecasting.