# Lithium-Ion Battery State of Health (SoH) Prediction Using Machine Learning Based on Charge Cycle Data

## 1. Project Title
Lithium-Ion Battery State of Health (SoH) Prediction Using Machine Learning Based on Charge Cycle Data

## 2. Objective
To model and predict how the health of a lithium-ion 18650 cell degrades as
the number of charge/discharge cycles increases, using measured cycling
data and a machine learning regression model, and to forecast the
degradation trend forward to 3000 cycles.

## 3. Problem Statement
Lithium-ion batteries lose usable capacity every time they are charged and
discharged. This capacity loss (aging/degradation) determines the
remaining useful life of the battery. Being able to predict State of
Health (SoH) from early-cycle behaviour is valuable for battery management
systems (BMS), EVs, and portable electronics, because it allows
maintenance/replacement to be planned before failure.

## 4. Dataset

**Primary (preferred) dataset:** NASA Prognostics Center of Excellence
(PCoE) Li-ion Battery Aging Dataset — batteries `B0005`, `B0006`, `B0007`,
`B0018` (18650 cells, repeatedly charged/discharged at 24°C until
significant capacity fade).
- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Mirror: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets

**How to add it:** download `B0005.mat` (or `B0006.mat` / `B0007.mat` /
`B0018.mat`) and place it directly inside `data/raw/`. The program
(`main.py`) automatically detects and loads it — no code changes needed.

**Fallback dataset (used automatically if no `.mat` file is present):**
a synthetic degradation dataset generated from a standard two-stage
Li-ion capacity-fade model (linear fade + accelerating "knee"), with
realistic sensor noise added. This exists **only** so the pipeline can be
demonstrated end-to-end before the real NASA files are downloaded.

**This run's dataset is always clearly recorded** in every output file
(`data_source` column/field = `NASA_PCoE_MEASURED` or
`SYNTHETIC_FALLBACK_NOT_REAL`) so it is never ambiguous which one was used.
Check `outputs/metrics/metrics.json` → `"is_synthetic_data"` before citing
results anywhere.

> **Battery chemistry note:** the NASA PCoE documentation for these cells
> does not explicitly publish a specific NMC/LFP chemistry designation for
> every cell in a way this project can verify automatically, so this
> project does **not** claim a specific chemistry unless you have checked
> the official NASA documentation for the exact cell used and can confirm
> it yourself for your report.

## 5. Battery SoH Definition

State of Health is defined using the standard formula:

```
SoH(n) = ( Capacity(n) / Capacity_initial ) × 100
```

`Capacity_initial` is the discharge capacity measured at the **first
valid cycle** in the dataset (cycle 1), used as the reference/baseline
capacity — the conventional choice when a separate manufacturer
nameplate capacity is not provided in the raw data.

## 6. Features Used

Only features that actually exist in the loaded dataset are used — nothing
is fabricated. From the NASA `.mat` files (or the equivalently-structured
synthetic fallback), the following per-cycle features are extracted:

| Feature | Description |
|---|---|
| `cycle` | Charge/discharge cycle number |
| `voltage_mean/min/max` | Statistics of measured terminal voltage during discharge |
| `current_mean/min` | Statistics of measured discharge current |
| `temperature_mean/max` | Statistics of measured cell temperature during discharge |
| `voltage_range` | derived: voltage_max − voltage_min |
| `temperature_rise` | derived: temperature_max − temperature_mean |

Internal resistance/impedance is **not** used as a feature because the
`.mat` discharge-cycle records used here do not reliably provide a single
per-cycle impedance value in a form comparable across all four batteries;
if you specifically need it, NASA also provides separate `impedance`-type
cycle records that could be parsed as a future extension.

**Target:** `soh` (%)

## 7. ML Methodology

**Model: Random Forest Regressor** (scikit-learn).

Why Random Forest, and not deep learning:
- SoH-vs-cycle is a smooth, mostly monotonic tabular regression problem
  with a modest number of features and samples — not enough data to
  justify a neural network.
- Random Forest handles the non-linear (linear-fade + "knee")
  degradation shape well, needs no feature scaling, and is robust to
  per-cycle measurement noise.
- It trains in seconds and is easy to explain in a viva (an ensemble of
  decision trees, each trained on a random subset of data/features, with
  predictions averaged).

## 8. Train/Test Strategy — Two Complementary Evaluations

This project reports **two** honest evaluations, because they answer
different questions:

**(A) Early-cycle → future-cycle extrapolation split** (matches the
project notes: *"feed ~1500 cycles → predict future cycles"*)
- Train ONLY on early cycles (up to cycle 1500, or 70% of available
  cycles if the dataset has fewer than 1500 — the program prints exactly
  which was used and why).
- Test on the later, never-seen-during-training cycles.
- **Important, honestly reported limitation:** tree-based models such as
  Random Forest cannot extrapolate beyond the numeric range of `cycle`
  (and correlated sensor values) seen during training — they can only
  interpolate. So this split's accuracy is expected to be poor, and this
  is a genuine, well-documented property of tree ensembles, not a bug.
  This is exactly *why* Step 7 (3000-cycle forecast) does NOT use the
  Random Forest — it uses a bounded empirical degradation curve fit instead (see
  Section 10 below).

**(B) Random (interpolation) split** — 75% train / 25% test, cycles
shuffled randomly across the whole measured range.
- This is the setting Random Forest is genuinely well-suited to, and is
  reported as the **primary MAE/R² result** for the project, because it
  fairly reflects the model's ability to predict SoH from sensor +
  cycle-count features.

Both sets of metrics are saved in `outputs/metrics/metrics.json`.

## 9. MAE (Mean Absolute Error)

```
MAE = mean( | Actual SoH − Predicted SoH | )
```

MAE is the primary evaluation metric required for this project because it
is directly interpretable in the same units as SoH (percent) — an MAE of,
say, 0.2% means the model's SoH predictions are, on average, within 0.2
percentage points of the true measured SoH.

## 10. 3000-Cycle Forecast Methodology

Because Random Forest cannot extrapolate past its training range, the
3000-cycle forecast is produced differently and clearly separately:

1. A standard two-stage Li-ion degradation curve is fit to the **entire
   measured SoH-vs-cycle history** using non-linear least squares
   (`scipy.optimize.curve_fit`):
   `SoH(n) = C0 * (1 − a·n − b·(1 − e^(−n/τ)))`
   (slow linear fade + an accelerating exponential "knee" — the standard
   shape reported across Li-ion aging literature).
2. This fitted curve is extended out to cycle 3000.
3. The forecast is forced to be non-increasing (physically required for
   SoH) and anchored to the last real measured SoH value.

**On every plot and in every CSV, this forecast region is explicitly
labeled "Forecast / Predicted"** and is visually and numerically
distinguished from the "Measured" region. The project never claims the
battery was experimentally tested for 3000 cycles.

## 11. Results

Results depend on which dataset was actually used for the run — always
check `outputs/metrics/metrics.json` (`is_synthetic_data`) before quoting
numbers. A results summary is printed to the console at the end of every
run, and also saved as JSON. Example structure of what's reported:

```
Primary (interpolation-split) results:
  MAE  : X.XX %
  R²   : X.XXXX
  RMSE : X.XX %

Early-cycle -> future-cycle extrapolation results (documented limitation):
  MAE  : X.XX %
  R²   : X.XXXX
```

## 12. Limitations

- Random Forest (and tree ensembles in general) cannot extrapolate beyond
  the cycle-number range they were trained on — this is why 3000-cycle
  forecasting uses a separate curve-fit method, not the ML model directly.
- If real NASA data was not downloaded before running, results are based
  on synthetic fallback data and **do not represent real battery
  behaviour** — they only demonstrate that the pipeline works correctly.
  This is clearly labeled everywhere (`SYNTHETIC_FALLBACK_NOT_REAL`).
- Internal resistance/impedance was not used, since it isn't reliably
  available per-discharge-cycle in the parsed `.mat` structure used here.
- The synthetic fallback's specific fade-rate constants are illustrative,
  not derived from a specific real cell's datasheet.
- SoH here is computed relative to the first measured cycle's capacity,
  not the manufacturer's nameplate rated capacity (which NASA's raw data
  does not separately specify).

## 13. Project Structure

```
battery-soh-prediction/
├── data/
│   ├── raw/                 <- put NASA .mat file(s) here
│   └── processed/           <- processed_battery_data.csv (auto-generated)
├── outputs/
│   ├── figures/              <- 3 PNG plots (auto-generated)
│   ├── predictions/          <- prediction CSVs (auto-generated)
│   └── metrics/              <- metrics.json (auto-generated)
├── src/
│   ├── data_loader.py         Loads NASA .mat data or synthetic fallback
│   ├── preprocessing.py       Cleans data, computes SoH
│   ├── feature_engineering.py Builds ML feature matrix
│   ├── model.py                Random Forest + 3000-cycle curve-fit forecast
│   ├── evaluation.py           MAE / R² / RMSE
│   └── visualization.py        All 3 required plots
├── main.py                    Runs the full pipeline
├── requirements.txt
├── run.bat                    Windows one-click setup + run
└── README.md
```

## 14. How to Run (Windows)

### Option A — one command
```cmd
run.bat
```
This creates a virtual environment, installs dependencies, and runs the
pipeline automatically.

### Option B — manual steps
```cmd
cd battery-soh-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

REM (Optional but recommended) place a downloaded NASA .mat file, e.g.
REM B0005.mat, into the data\raw\ folder now.

python main.py
```

### Outputs produced after running
```
data/processed/processed_battery_data.csv
outputs/figures/soh_vs_cycles.png
outputs/figures/actual_vs_predicted.png
outputs/figures/prediction_residuals.png
outputs/predictions/test_set_predictions.csv
outputs/predictions/early_to_future_extrapolation_predictions.csv
outputs/predictions/forecast_3000_cycles.csv
outputs/metrics/metrics.json
```


## Important interpretation of the 3000-cycle forecast

The NASA B0005 experiment contains 168 measured discharge cycles. The plot therefore shows measured NASA data only through the final observed cycle, followed by a dashed **model-based extrapolation** toward cycle 3000. The extrapolation uses a bounded stretched-exponential degradation curve:

\[ \mathrm{SoH}(n) = S_\mathrm{floor} + (100 - S_\mathrm{floor}) \exp\left(-\left(\frac{n}{\tau}\right)^p\right) \]

where `S_floor`, `tau`, and `p` are fitted from the measured SoH trajectory. The bounded form prevents the forecast from becoming negative or collapsing artificially to 0% simply because the curve was extrapolated far beyond the measured range. It is an empirical trend model, **not a first-principles electrochemical model and not a claim that B0005 was experimentally tested to 3000 cycles**.
