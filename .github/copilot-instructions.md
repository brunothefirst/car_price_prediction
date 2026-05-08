# GitHub Copilot Instructions — Car Price Benchmark

Read CLAUDE.md at the root of this repository for full project context before making any suggestions. The content below mirrors the key context for quick access.

## What this project does

Builds a French used-car price benchmark (LeBonCoin data) to assess whether cars imported from African countries are over- or underpriced at customs declaration. The output is a price range: p50 (median) and p85 (upper bound) for any car described by brand, model, year, and optionally km/fuel/power.

## Tech rules — always follow these

- **Use Polars, not pandas** for all data processing. The codebase is Polars-native.
- **LightGBM with `objective='quantile'`** for all models. Do not suggest other loss functions.
- **Fit `CarPriceFeatureEngineer` on training data only.** Never fit on the full dataset — this causes target encoding leakage.
- **Log all experiments to MLflow** (`localhost:5001`). Always include: variant, quantile, pinball loss, coverage rate.
- **Primary metric is pinball loss**, not RMSE or MAE. Quantile models must be evaluated with quantile-appropriate metrics.
- **Notebooks for experimentation, `src/` for stable code.** Do not modify `src/` without a validated notebook proving the improvement.

## Data

- Raw path: `/Users/brunobrumbrum/Documents/data/car_price_prediction/` (two subdirectories: October and December extractions)
- Currently using October only. Deduplication across extractions uses the `url` column.
- Raw columns: `marque`, `modele`, `annee_modele`, `kilometrage`, `price`, `energie`, `puissance_din`, `url`
- Cleaned columns: `price`, `year`, `km`, `brand`, `model`, `energie`, `horsepower`

## Key classes

- `load_car_data(data_dir)` — `src/data_processing.py` — loads CSVs, parses horsepower
- `CarDataProcessor` — `src/data_processing.py` — cleaning pipeline (type conversion → antique removal → 'autre' removal → HP cleaning → rare brand filter → IQR outlier removal)
- `CarPriceFeatureEngineer` — `src/features/feature_engineering.py` — sklearn-compatible feature transformer

## Model variants

| Variant | Features |
|---------|----------|
| Lean | brand, model, year |
| Extended | + km, horsepower, fuel type |

## Active issue — age bias

`CarDataProcessor._remove_outliers_iqr()` groups IQR by `brand` only. Old cars (pre-2008) are over-removed (25–30% drop rate). Fix: group by `(brand, year_bucket)` where `year_bucket = (year // 5) * 5`.

## Agent skills (for Claude Code)

Four slash commands exist in `.claude/commands/`:
- `/data-pipeline` — data loading, cleaning, outlier removal
- `/train-models` — retrain lean/extended quantile models
- `/critic` — methodology review and explainability
- `/research` — open-ended experimentation

When a user asks to do any of these tasks, follow the guidelines in the corresponding `.claude/commands/*.md` file.
