# Next Steps Roadmap — Post Age-Bias Fix (June 2026)

_Assessment based on NB11, NB12, NB12b, `age_bias_analysis_conclusions.md`, and customs evaluation results. This document is the working plan for the next development phase._

---

## Current State

- **Done:** Age bias in IQR cleaning diagnosed (NB11) and fixed with brand+year IQR (NB12). Drop rate now flat 3–7% across years (was 23% for year-2000 cars). V2 models saved and logged to MLflow.
- **Honest read of NB12:** V2 improved pre-2005 Median APE by only ~1.5pp (36.0 → 34.5%). The cleaning fix was necessary hygiene but is **not** the lever for old-car performance. Remaining drivers: training imbalance, intrinsic price variance of old cars, low absolute prices inflating %.
- **Error profile (Dec 2025 held-out):** Median APE 8% (0–2yr) → 34% (21+yr). Q15–Q85 coverage ~69% overall but 55–73% depending on year.
- **Critical gap found in NB12b:** 50 of 102 Côte d'Ivoire customs vehicles dropped for unknown brand/model — the model cannot price ~half of real customs cases.

---

## Note on customs match rate (deprioritized)

The low match rate in NB12b (50/102 CI vehicles unmatched) is largely explained by **trucks** in the customs data, which are out of scope for a passenger-car benchmark. No dedicated matching notebook needed now. Keep two light checks for later: (a) tag and exclude trucks explicitly in customs evaluations so the match rate on *cars* is reported correctly; (b) verify the pre-2000 age cap against the age distribution of customs car declarations.

---

## Priority 1 — Merge October + December extractions (with a proper validation design)

December is currently the only true held-out test set. Merging it into training **destroys it** — plan validation before merging.

1. **Dedup by `url`** (keep most recent extraction), per project convention.
2. **Near-duplicate audit (audit, not blind dedup):** count listings identical on `(brand, model, year, km, price)` with different urls — likely the same physical car relisted. This is NOT about removing common cars (many similar Clios are legitimate signal and must be kept). The only risk is train/test leakage under random splits. If the rate is <1–2%, document and skip; if higher, dedupe with strict exact-match criteria only. Once evaluation moves to an out-of-time extraction, this concern largely disappears.
3. **Drift check before merging:** compare Oct vs Dec price distributions per segment (brand × age bucket). Quantifies market drift / seasonality → informs model refresh cadence for the future API.
4. **Bonus analysis:** cars present in both extractions with price changes → free signal on listing-price negotiation drift (relevant caveat: LeBonCoin = asking price, not transaction price — document this for clients).
5. **New validation plan:** after merge, use random split for development, but **schedule a fresh LeBonCoin extraction (it is June 2026)** as the new out-of-time test set. Out-of-time evaluation is the only honest measure for the customs use case.

**Deliverable:** notebook 13 — merge + dedup audit + drift analysis. Acceptance: documented dedup counts, drift table per segment, merged dataset saved with version tag.

---

## Priority 2 — Cross-validation harness (use for every experiment from here on)

Past evaluations relied on single splits; NB12 deltas (~0.1pp) are within split noise. Build a proper CV harness on the merged dataset and make it the standard for all subsequent comparisons.

1. **5-fold CV with the feature engineer refit inside each fold.** Target encodings (`model_median_log_price`, etc.) are fitted statistics — fitting once on full data then CV-ing leaks. This is the most common CV mistake with target encoding.
2. **Cross-validate all quantiles, not just p50:** per fold, compute pinball loss for q50 and q85 *and* empirical Q15–Q85 coverage. The p85 interval is the product; CV must expose its stability across folds.
3. Report mean ± std per metric → changes are only accepted if the improvement exceeds fold noise.
4. CV is for **model comparison and hyperparameter tuning**. Final client-facing numbers always come from the out-of-time test set (fresh extraction).

**Deliverable:** notebook 14 — reusable CV utilities (move to `src/` after validation). Acceptance: CV results reproduce the V2 baseline within noise; harness reused by all later notebooks.

---

## Priority 3 — Imbalance: improve old-car performance

Test in this order (each as a separate MLflow run, evaluated with the CV harness from Priority 2):

1. **Sample weights** in LightGBM (`sample_weight` = inverse year-frequency, capped at e.g. 5×) — preferred over physical oversampling: no duplicated rows, no leakage risk, works directly with quantile loss.
2. **Monotonic constraints** (`monotone_constraints`: price ↑ with year, ↓ with km within model). Small accuracy cost, large trust gain: the model can never produce economically absurd orderings. Strong client-facing argument.
3. **Age-segment interaction features** or a separate old-car model (brand stats computed within age cohort) — only if 1–2 are insufficient.
4. **Accept residual variance, fix the intervals instead:** for customs, what matters is a *calibrated* p85, not low APE. Wider but honest intervals on old cars are defensible; overconfident narrow ones are not.

**Evaluation (mandatory, per `age_bias_analysis_conclusions.md`):** Median APE by year and age bucket, signed bias by year, coverage by year, pinball loss by segment. Compare against V2 baseline.

**Acceptance:** pre-2005 Median APE meaningfully reduced **without** degrading post-2015, OR documented negative result in MLflow.

---

## Priority 4 — Calibration as the product: conformal correction

The client-facing claim is "X% of true market prices fall below our pXX". Current conditional coverage varies 55–73% by year — not defensible per segment.

1. Implement **split-conformal calibration per age bucket** (or per year-group): hold out a calibration set, adjust the raw q85 predictions so empirical coverage hits the target in *every* bucket, not just overall.
2. Report conditional coverage by year, age bucket, brand in every evaluation.
3. Attach a **confidence tier** to each prediction based on training support: n listings for that (brand, model, year). Low-support predictions get flagged. This is cheap and dramatically improves trust.

**Acceptance:** p85 coverage within ±3pp of nominal in every age bucket on the out-of-time test set.

---

## Priority 5 — Trust & explainability (client-readiness)

1. **SHAP values** for individual predictions — "why is this Corolla benchmarked at €9,400" must be answerable in one chart for customs analysts.
2. **Model card** documenting: data source and its asking-price caveat, training period, segments in/out of scope, error by segment, calibration guarantees, refresh cadence.
3. **Re-run CI + CM customs evaluations** with each model iteration (after the matching layer from Priority 1) — this is the real KPI.

---

## Metric policy (decided)

- **Model selection:** pinball loss per quantile (the only correct loss for quantile models; per CLAUDE.md).
- **Business reporting:** Median APE per year / age bucket (robust, scale-independent). Business rationale: customs duties are ad valorem (% of declared value), so *relative* error is what matters proportionally — a 15% error costs the same in duty terms on a €3k car as on a €30k car.
- **Calibration:** Q15–Q85 coverage (and p85 exceedance rate) per segment.
- MAE in € may be shown as context but never drives model selection (it over-weights expensive cars).

---

## On feature engineering: why distribution stats plateaued (explained, not a bug)

The engineered distribution features (`model_p75_log_price`, `brand_iqr_log_price`, top 1%/95% stats, etc.) are **group-level constants** — every car with the same (brand, model, year) receives identical values. They cannot explain within-group price variance, and they are heavily correlated with the median, so trees extract the signal once from `model_median_log_price` and the rest are redundant. This is why only **car-level** variables (km, horsepower) produce real gains. Implication: prune redundant group stats for a leaner, more explainable model (validate no accuracy loss via CV); the next car-level lever is trim/version.

---

## Worth investigating (not yet analyzed)

- **Trim/version granularity:** the cleaned schema has no trim (e.g. 318i vs M3 within "serie 3"). The largest remaining car-level signal for newer cars. Check whether raw extractions contain listing titles; if so, trim extraction is a high-value feature project.
- **`energie` drift:** EV/hybrid share is growing fast between extractions; price dynamics differ (EV depreciation). Check fuel-type mix Oct vs Dec and error by fuel type in the extended model.
- **Rare-brand threshold (<400):** verify against customs data — if declarations include brands dropped at cleaning, the fallback hierarchy (Priority 1) must cover them.
- **km plausibility:** km vs age sanity bounds (e.g. >60,000 km/yr or <1,000 km/yr on a 10-yr car) — odometer-fraud-like listings may distort km features.

---

## Suggested execution order

| Step | Work | Notebook |
|------|------|----------|
| 1 | Oct+Dec merge, dedup audit, drift analysis | NB13 |
| 2 | CV harness (all quantiles, fold-internal feature fitting) | NB14 |
| 3 | Sample weights + monotonic constraints + feature pruning | NB15 |
| 4 | Conformal calibration per segment | NB16 |
| 5 | Customs re-evaluation (CI, CM; trucks excluded) + model card | NB17 |

Rules unchanged: validate in notebooks before touching `src/`, log every experiment (incl. negative results) to MLflow, always produce the mandatory per-year/per-bucket evaluation charts.
