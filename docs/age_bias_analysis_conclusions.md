# Age Bias Analysis — Conclusions & Model Comparison Results

_Last updated: May 2026 — based on NB11 (age bias diagnosis) and NB12 (V1 vs V2 model comparison)_

---

## Problem: Global IQR Introduces Age Bias

The original data cleaning applied a single IQR filter per **brand**, removing any listing whose price fell outside 1.5× IQR of that brand's full distribution.

**Effect:** 18,743 listings (2.9% of total) were flagged as outliers — but the drop rate was heavily age-dependent:

- Pre-2010 cars: **20–25% drop rate**
- Post-2015 cars: **< 5% drop rate**

The global IQR was penalising cars for being old, not for being genuine price outliers. A cheap 1998 Renault Clio is not an outlier — it is a cheap old car. When compared to the full brand distribution (dominated by newer, pricier cars), it looks anomalous and gets removed.

**Box plot evidence:** For cars 6+ years old, the median price of *removed* cars was consistently *below* the median of *kept* cars — confirming that old cheap cars were being systematically excluded, not true outliers.

---

## Fix: Brand × Year IQR with Fallback

The new three-stage pipeline replaces the single brand IQR:

### Stage 1 — Remove cars older than 25 years (pre-2000)
Any car more than 25 years old is excluded before any other filter.
- **Removed:** 12,250 cars (1.9% of dataset)

### Stage 2 — Hard price boundaries
- **Floor €500:** listings below this are non-representative (removed 1,891 cars, 0.3%)
- **Ceiling €250,000:** 99.95th percentile; extreme high-end outliers (removed 320 cars)

### Stage 3 — Brand × Year IQR with ±1 fallback

| Group size | Action |
|-----------|--------|
| ≥ 50 cars in brand × year | Apply standard IQR directly |
| < 50 cars in brand × year | Expand to year ±1 window and retry |
| < 10 cars after expansion | Skip IQR entirely (too few to compute a meaningful range) |

Groups skipped entirely: **41 vehicles** — an acceptable edge-case count.

**Result:** Drop rate is flat at **3–7% across all model years** (vs. 23% for year-2000 cars under the old method). The age bias is eliminated.

---

## Model Comparison: V1 vs V2

### Experimental Setup

| | V1 | V2 |
|---|---|---|
| **Cleaning** | Brand-only IQR (original) | Brand+year IQR (new method) |
| **Features** | Lean: brand/model/year stats (31 cols) | Identical |
| **Hyperparameters** | `lr=0.1`, `n_estimators=5000` | Identical |
| **Training data** | 100% of October 2025 extraction | 100% of October 2025 extraction |
| **Test set** | December 2025 extraction (held-out) | Same |
| **Model artefacts** | `models/lean_quantile/` | `models/lean_quantile_v2_lean/` |

The December test set was filtered to brand+model combinations seen by V2, ensuring an apples-to-apples comparison rather than penalising V1 for combos V2 never trained on.

### Results

| Segment | V1 Median APE | V2 Median APE | Delta |
|---------|:------------:|:------------:|:-----:|
| Overall (weighted mean) | 15.1% | 15.0% | −0.1 pp |
| 16–20 yr (2005–09) | 26.1% | 25.7% | −0.4 pp |
| 21+ yr (pre-2005) | 36.0% | 34.5% | **−1.5 pp** |
| Post-2010 | ≈ identical | ≈ identical | ~0 |

**Key finding:** The cleaning improvement has no meaningful effect on newer cars (plentiful, well-represented data), but provides a consistent and measurable improvement on older inventory. The pre-2005 segment, which was most affected by the original bias, improves by 1.5 percentage points.

---

## Remaining Issue: High Absolute Error on Older Cars

Old cars still carry a significantly higher absolute % error (34–36% for pre-2005 vs. ~9% for 2024–25 cars). Three likely drivers:

1. **Fewer training examples** — old cars are a small fraction of LeBonCoin listings.
2. **Higher intrinsic price variability** — condition, history, and rarity create wide price ranges that are hard to predict from brand/model/year alone.
3. **Lower average prices** — a €500 error on a €3,000 car is 17%; the same error on a €20,000 car is 2.5%.

**Candidate remedy:** Oversample older car examples during training to prevent the model from implicitly optimising for the more numerous newer cars.

---

## Required Standard Metrics for All Future Evaluations

After this analysis, the following charts/tables are **mandatory** in every model evaluation notebook:

1. **Median APE by calendar year** (line chart, V1 vs V2 or baseline vs candidate) — detects regressions on older inventory that aggregate metrics would hide.
2. **Median APE by age bucket** (grouped bar chart, same 6 buckets: 0–2, 3–5, 6–10, 11–15, 16–20, 21+ yr).
3. **Q15–Q85 coverage by year** — confirms interval calibration is not degrading for specific vintages.

These are implemented in NB12 (`compute_year_metrics`, `compute_bucket_metrics`) and should be reused in future notebooks.

---

## Next Steps

1. **Re-run Cameroon and Côte d'Ivoire customs predictions with V2 models** and compare against V1 results to quantify the real-world improvement in the customs use case.
2. **Evaluate mileage (km) as a feature** — for countries that provide odometer data, adding km could meaningfully improve accuracy on older cars where mileage is a stronger price driver than model year alone.
3. **Explore oversampling older car examples** (e.g., weighted loss or resampling in training) to reduce the residual high APE on pre-2005 inventory.
4. **Merge October and December extractions** — deduplication via the `url` column; the combined dataset would roughly double the training data and should improve coverage for thin segments (old cars, rare brands).

---

## Reference Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/11_age_bias_analysis.ipynb`](../notebooks/11_age_bias_analysis.ipynb) | Diagnosed the IQR age bias; compared per-year drop rates old vs new cleaning |
| [`notebooks/12_lean_v1_vs_v2_cleaning.ipynb`](../notebooks/12_lean_v1_vs_v2_cleaning.ipynb) | Controlled V1 vs V2 comparison on December 2025 held-out set |
