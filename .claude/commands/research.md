# Research Skill — Experimentation and Improvement

You are conducting structured data science research on a car price prediction project. Your job is to run experiments that genuinely improve the pipeline — and to distinguish real improvements from overfitting or measurement errors.

Read CLAUDE.md for full project context.

## The research workflow — always follow this order

1. **State your hypothesis first.** Before writing code, write one sentence: what do you expect to improve and why?
2. **Implement in a notebook.** Never prototype directly in `src/`.
3. **Compare against a baseline.** Always run the current pipeline as a baseline and compare with pinball loss and coverage rate. Use the last MLflow run as reference.
4. **Check for the right failure modes.** An improvement in overall pinball loss that introduces age or brand bias is not an improvement.
5. **Promote to `src/` only after validation.** Once confirmed better across all checks, update `src/data_processing.py` or `src/features/feature_engineering.py`.
6. **Log everything to MLflow.** Including negative results — they prevent repeating the same experiment.

## Open research areas

### Data cleaning improvements (high priority)
- **Age-stratified outlier removal** — fix the active bias issue. Group IQR by `(brand, year_bucket)` instead of `brand` only. Reference: `src/data_processing.py::_remove_outliers_iqr()`. Validate in `notebooks/11_age_bias_analysis.ipynb`.
- **Alternative outlier methods** — isolation forest or local outlier factor as a complement or replacement to IQR. Be careful: these are harder to interpret to a non-technical audience, which may be a problem for the critic.
- **Better rare model handling** — some common brands have rare models that get noisy target encoding. Explore smoothing strategies.

### Feature engineering improvements
- **New features to test:** listing age (days a car has been listed — proxy for overpricing), regional price variation (if region data exists in the raw data), VIN-based features (explored in `notebooks/10_vin_decoder_test.ipynb` — check if it added value)
- **Testing procedure:** compare pinball loss with feature included vs. excluded on a held-out set. Never judge a feature by training performance alone.
- **Data leakage check:** any feature that aggregates prices (target encoding) must be fit only on training data. `CarPriceFeatureEngineer.fit()` handles this — do not bypass it.

### Model architecture
- **Multi-output quantile:** train p50 and p85 jointly in a single model vs. current separate models. Can reduce inconsistency between quantiles.
- **CatBoost:** handles high-cardinality categoricals (brand, model) natively — may outperform LightGBM on the lean model where brand and model are the dominant features. Run a head-to-head comparison.
- **Rule:** do not switch architecture without a logged head-to-head comparison in MLflow showing improvement on both pinball loss and calibration.

### Dataset expansion
- **Merging October + December extractions:** deduplicate on `url` column, keep most recent listing. After merging, recheck all bias metrics — the size and distribution shift can introduce new problems.
- **Temporal split evaluation:** after merging, train on October and test on December to assess temporal generalisability before collapsing both into a single training set.

## Anti-patterns — do not do these

- Do not tune hyperparameters on the test set
- Do not evaluate quantile models with RMSE or MAE — always use pinball loss and coverage rate
- Do not add features to `CarPriceFeatureEngineer` without checking for leakage — target encoding must be fit only on training data
- Do not conclude a feature is useful based on training performance alone
- Do not modify `src/` without a validated notebook proving the improvement
- Do not declare an experiment successful without running the bias check (drop rate by year, pinball loss by year group and brand)

## Connecting research to the critic

Before marking any research task complete, run the `/critic` checklist on your result. The question is not just "does this have lower pinball loss?" but "is the methodology now more convincing to a skeptical customs analyst?"
