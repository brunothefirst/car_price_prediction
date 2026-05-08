# Data Pipeline Skill

You are working on the data pipeline for a car price benchmark project. Your job is to load, clean, and prepare LeBonCoin used-car data for model training. Read CLAUDE.md for full project context.

## Your scope

Everything that happens between raw CSV files and a model-ready Polars DataFrame:
- Loading raw data
- Cleaning and normalising
- Outlier removal
- Deduplication (when merging extractions)

Feature engineering is handled separately by `CarPriceFeatureEngineer` — do not mix concerns.

## Data locations

- Raw data: `/Users/brunobrumbrum/Documents/data/car_price_prediction/`
  - Subdirectory 1: October extraction (~700k rows) — **currently in use**
  - Subdirectory 2: December extraction (~700k rows) — pending merge
- When merging both: deduplicate on the `url` column. Same URL = same listing. Keep most recent.

## Key code — `src/data_processing.py`

### `load_car_data(data_dir: Path)`
- Reads all CSVs in a directory
- Parses `puissance_din` column (format: `"150 Ch"`) into a float `horsepower` column
- Returns a concatenated Polars DataFrame

### `CarDataProcessor` cleaning steps (in order)
1. `_convert_data_types()` — cast numerics, normalize brand/model text (lowercase, accent removal)
2. `_remove_antique_cars()` — drop year < 1990
3. `_remove_autre_entries()` — drop `brand` or `model` == `"autre"`
4. `_clean_horsepower()` — hard bounds [50, 1000 HP] → IQR per brand → drop nulls
5. `_filter_rare_brands()` — drop brands with < 400 listings
6. `_remove_outliers_iqr()` — IQR on `log_price` and `km`, grouped per brand

After each run, inspect `processor.cleaning_stats` for row counts at each step.

## Known issue — age bias in step 6

**Problem:** `_remove_outliers_iqr()` groups IQR by `brand` only. Old cars (pre-2008) have low prices relative to the modern cars that dominate each brand's distribution. They fall below the lower IQR bound and are dropped at 25–30% rates vs. ~0% for post-2015 cars.

**Fix:** In `_remove_outliers_iqr()`, change the `group_by` from `'brand'` to `['brand', 'year_bucket']`, where:
```python
year_bucket = (pl.col('year') // 5) * 5  # 5-year bins: 1990, 1995, 2000, ...
```
Add this column before computing IQR bounds, then remove it after filtering.

**Before implementing:** validate in notebook 11 (`notebooks/11_age_bias_analysis.ipynb`). After implementing: verify drop rate by year is roughly uniform (< 5% variance across year groups).

## Rules

- Always use Polars, never pandas, for bulk data operations
- Always validate changes in a notebook before modifying `src/data_processing.py`
- After any cleaning change, plot drop rate by year and by brand to confirm no new bias
- Do not add cleaning steps that are not validated — prefer conservative defaults
- The `cleaning_stats` dict gives per-step diagnostics; always review it after running
