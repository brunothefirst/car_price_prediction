# Train Models Skill

You are training LightGBM quantile regression models for a car price benchmark. The models predict a defensible price range for used cars on the French market (LeBonCoin). Read CLAUDE.md for full project context.

## What you are building

Two model variants × two quantiles = four models total:

| Variant | Features | Quantiles |
|---------|----------|-----------|
| Lean | brand, model, year | p50, p85 |
| Extended | + km, horsepower, fuel type | p50, p85 |

The p50 model gives the median benchmark price. The p85 model gives the upper confidence bound. Together they form the price range presented to customs analysts.

## Feature engineering — `CarPriceFeatureEngineer`

**Location:** `src/features/feature_engineering.py`

This is an sklearn-compatible transformer. Always:
- `fit()` on training data only — it computes target encoding (brand/model average prices) which must not see the test set
- `transform()` on both train and test separately

Key feature groups it produces:
- Time: `car_age`, `decade`, `age_category`, `is_almost_new`
- Mileage: `km_per_year`, `mileage_category`, `is_low_mileage`, `is_high_mileage`
- Brand target encoding: `brand_mean_log_price`, `brand_median_log_price`, `brand_std_log_price`
- Model target encoding: `model_mean_log_price`, `model_median_log_price`
- Distribution: `brand_p25_log_price`, `brand_iqr_log_price`, `model_p75_log_price`, etc.
- Interaction: `age_km_interaction`, `is_garage_queen`, `is_low_use_recent`

For the **lean model**: pass only `brand`, `model`, `year` columns to the transformer (it will skip km-based features gracefully if km is absent — verify this before relying on it).

For the **extended model**: pass all columns including `km`, `horsepower`, `energie`.

## Training protocol

1. Load and clean data using `CarDataProcessor` (see `/data-pipeline`)
2. Split train/test — document which strategy in MLflow (time-based preferred; random acceptable if documented)
3. Fit `CarPriceFeatureEngineer` on training set only, transform both splits
4. Train one LightGBM model per quantile:
   ```python
   lgb.train(
       params={..., 'objective': 'quantile', 'alpha': 0.50},  # or 0.85
       ...
   )
   ```
5. Log everything to MLflow before finishing

## MLflow logging — always log these

- `model_variant`: `"lean"` or `"extended"`
- `quantile`: `0.50` or `0.85`
- `n_train`, `n_test`
- `pinball_loss` on the test set (the correct loss for quantile regression)
- `coverage_rate`: fraction of test observations where `true_price <= p85_prediction` (target: ~85%)
- Feature importances as a figure artifact
- The trained model artifact (so it can be reloaded)

MLflow server: `localhost:5001`

## Evaluation — what matters

- **Pinball loss** is the primary metric. Lower is better. Compare against the last logged run.
- **Coverage rate for p85** should be approximately 85%. If it is significantly off (< 80% or > 90%), the confidence interval is miscalibrated — flag this before presenting results.
- **Bias check:** break pinball loss down by year group (pre-2000, 2000–2010, 2010–2020, 2020+) and by brand. Systematic errors in specific segments indicate the model is not general enough.

Do not use RMSE or MAE as primary metrics for quantile models. They measure the wrong objective.

## Lean vs extended comparison

After training both variants, always compare predictions on the same held-out set. Document:
- How much does extended improve over lean, in pinball loss?
- For the same car, do lean and extended produce consistent price ranges? Large disagreements are a red flag.

This comparison is evidence for how strongly partners should be pushed to provide additional data.
