# Car Price Benchmark — Project Guide

## Purpose

This project builds a French used-car price benchmark to assess whether cars imported from African countries are over- or underpriced at customs declaration. Importers have an incentive to declare lower values to reduce customs duties — this model provides a defensible reference price range.

## Business Context

- **Client:** Customs and trade compliance analysts evaluating African car imports
- **Hypothesis:** Declared import prices are systematically lower than French market values
- **Output:** A price range — p50 (median benchmark) and p85 (upper confidence bound) — for any car described by brand, model, year, and optionally km/fuel/power
- **Current stage:** Research and model validation. Output is consumed via manual analysis. An API is planned once models are stable.

## Data

- **Source:** LeBonCoin (French second-hand marketplace), cars only
- **Extractions:** October (~700k listings, currently in use) and December (~700k listings, pending merge)
- **External data path:** `/Users/brunobrumbrum/Documents/data/car_price_prediction/`
  - Two subdirectories, one per extraction run
- **Deduplication rule:** Use the `url` column — same URL in both extractions = same listing. Keep the most recent.
- **Raw schema:** `marque`, `modele`, `annee_modele`, `kilometrage`, `price`, `energie`, `puissance_din`, `url`
- **Cleaned schema:** `price`, `year`, `km`, `brand`, `model`, `energie`, `horsepower`

## Tech Stack

- **Language:** Python
- **Data processing:** Polars (not pandas — use Polars for all bulk data operations)
- **ML:** LightGBM with `objective='quantile'`
- **Feature engineering:** scikit-learn compatible (`CarPriceFeatureEngineer` is a `BaseEstimator`/`TransformerMixin`)
- **Experiment tracking:** MLflow at `localhost:5001`
- **Notebooks:** Jupyter, used for research and experimentation only

## Pipeline Overview

```
Raw CSVs
  → load_car_data()          # src/data_processing.py
  → CarDataProcessor.clean_data()   # type conversion, outlier removal, rare brands
  → CarPriceFeatureEngineer  # src/features/feature_engineering.py
  → LightGBM quantile (p50 + p85)
  → MLflow logging
```

### Step 1 — Load
`load_car_data(data_dir: Path)` in `src/data_processing.py`. Reads all CSVs in a folder and parses horsepower from the `puissance_din` column (format: `"150 Ch"` → `150.0`).

### Step 2 — Clean (`CarDataProcessor.clean_data()`)
1. Type conversion and text normalization (lowercase, accent removal)
2. Remove antique cars (year < 1990)
3. Remove entries with `brand` or `model` == `"autre"` (unknown)
4. Clean horsepower: hard bounds [50, 1000 HP], then IQR per brand, then drop nulls
5. Drop rare brands (fewer than 400 listings)
6. IQR outlier removal on `log_price` and `km`, grouped per brand

### Step 3 — Feature Engineering (`CarPriceFeatureEngineer`)
Fit on training data only, transform train and test separately (no leakage).
- **Time:** `car_age`, `decade`, `age_category`, `is_almost_new`
- **Mileage:** `km_per_year`, `mileage_category`, `is_low_mileage`, `is_high_mileage`
- **Target encoding (brand/model):** `brand_mean_log_price`, `model_median_log_price`, etc.
- **Distribution:** `brand_p25_log_price`, `brand_iqr_log_price`, `model_p75_log_price`, etc.
- **Interaction:** `age_km_interaction`, `is_garage_queen`, `is_low_use_recent`
- **Optional:** horsepower features, fuel type OHE

### Step 4 — Train
Two LightGBM models per variant: `alpha=0.50` (median) and `alpha=0.85` (upper bound).

### Step 5 — Log
All experiments go to MLflow. Always log: variant, quantile, n_train, n_test, pinball loss, coverage rate, feature importances.

## Model Variants

| Variant | Features | Use case |
|---------|----------|----------|
| **Lean** | brand, model, year | Partner countries with minimal declaration data |
| **Extended** | + km, horsepower, fuel type | When more complete customs data is available |

Both use identical LightGBM quantile architecture. The only difference is the feature set passed to `CarPriceFeatureEngineer`.

## Key Files

| File | Role |
|------|------|
| `src/data_processing.py` | `CarDataProcessor` class + `load_car_data()` |
| `src/features/feature_engineering.py` | `CarPriceFeatureEngineer` (sklearn-compatible) |
| `notebooks/06_lean_model_training.ipynb` | Lean model training |
| `notebooks/07_lean_model_evaluation_customs.ipynb` | Evaluation against customs partner data |
| `notebooks/11_age_bias_analysis.ipynb` | Age bias investigation (currently active) |

## Workflow Convention

- **Notebooks are for research.** Experiment, validate, and compare results there.
- **`src/` is for stable code.** Only move code into `src/` after it has been validated in a notebook.
- **Always log to MLflow.** Even negative results — they prevent repeating the same experiment.
- **Never modify `src/` without a validated notebook proving the change is an improvement.**

## Known Issue — Age Bias in Outlier Removal (Active)

**Location:** `src/data_processing.py` → `CarDataProcessor._remove_outliers_iqr()` (~line 407)

**Problem:** IQR is computed grouped by `brand` only. Old cars (pre-2008) are genuinely cheaper, but they look like outliers compared to the modern cars that dominate each brand's distribution → drop rates of 25–30% for pre-2008 cars vs. ~0% for post-2015.

**Fix direction:** Group by `(brand, year_bucket)` where `year_bucket = (year // 5) * 5`. This ensures old cars are compared against other old cars of the same brand. Validate in notebook 11 before touching `src/`.

## Evaluation Metrics

- **Primary:** Pinball loss (quantile loss) — the correct metric for quantile regression
- **Calibration:** Coverage rate — % of held-out true prices below the p85 prediction (should be ~85%)
- **Bias check:** Pinball loss broken down by year group and brand — should be roughly uniform
- **Do not use RMSE or MAE** as primary metrics for quantile models

## Available Agent Skills

| Command | Purpose |
|---------|---------|
| `/data-pipeline` | Data loading, cleaning, and outlier removal |
| `/train-models` | Retrain lean/extended quantile models, log to MLflow |
| `/critic` | Methodology review: explainability, robustness, and "is this result defensible?" |
| `/research` | Open-ended experimentation: alternative models, features, cleaning improvements |
