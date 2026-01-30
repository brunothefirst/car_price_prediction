# Car Price Prediction - Data Processing Methodology Documentation

**Date:** January 30, 2026  
**Purpose:** Detailed documentation of data cleaning and feature engineering methodology for review

---

## Table of Contents

1. [Data Cleaning Process (`clean_car_data`)](#1-data-cleaning-process)
2. [Feature Engineering Process (`CarPriceFeatureEngineer`)](#2-feature-engineering-process)
3. [Design Decisions & Rationale](#3-design-decisions--rationale)
4. [Questions for Review](#4-questions-for-review)

---

## 1. Data Cleaning Process

**Location:** `src/data_processing.py` → `CarDataProcessor.clean_data()`

### Overview
The cleaning pipeline processes raw car listing data through 5 sequential steps, removing invalid data and outliers while handling rare categories intelligently.

---

### Step 1: Data Type Conversion

**What it does:**
- Converts string columns to numeric types (price, year, kilometers)
- Removes non-numeric characters from price (e.g., currency symbols, spaces)
- Handles missing/invalid values by setting them to `None`

**Why:**
- Raw CSV data contains formatted strings like "25,000 €" for price
- Need numeric types for statistical operations and modeling
- Explicitly handling nulls prevents silent errors downstream

**Key operations:**
```python
# Remove all non-digit characters from price, convert to float
price: "25,000 €" → 25000.0

# Clean year and kilometers similarly
year: "2020" → 2020.0
km: "150,000 km" → 150000.0
```

**Result:** 
- Renamed columns: `price`, `year`, `km`, `brand`, `model`
- **Preserved columns:** `energie` (fuel type), `puissance_din` (horsepower in format "150 Ch")
- Filters out rows where price is null (cannot train without target)

**Note:** `energie` and `puissance_din` are kept as-is (strings) for downstream processing - they are NOT cleaned or transformed in this step

---

### Step 2: Brand Filtering and Grouping

**What it does:**
- **Removes very rare brands** (< 50 cars): Not enough data for reliable statistics
- **Groups medium-frequency brands** (50-599 cars) into 3 price tiers:
  - `other_low_cost` (bottom 33% of prices)
  - `other_standard` (middle 33%)
  - `other_luxury` (top 33%)
- **Keeps popular brands unchanged** (≥ 600 cars)

**Why:**
- **Prevents overfitting:** Rare brands with few examples lead to unreliable price estimates
- **Reduces dimensionality:** From potentially hundreds of brands to ~50-60 categories
- **Preserves signal:** Grouping by price tier maintains price-related information
- **Threshold rationale:**
  - < 50 cars: Too few for any statistics (removes ~1-2% of data)
  - 50-599: Enough to be meaningful but benefits from grouping
  - ≥ 600: Major brands with solid statistics

**Example:**
```
"Bugatti" (8 cars) → REMOVED (too rare)
"Suzuki" (250 cars, avg €8k) → "other_low_cost"
"Lexus" (450 cars, avg €35k) → "other_luxury"
"Renault" (12,000 cars) → Unchanged
```

**Important detail:**
- When a brand is grouped, its `model` column is also set to the group name
- This prevents creating meaningless model categories like "other_low_cost Clio"

---

### Step 3: Remove Antique Cars

**What it does:**
- Removes cars manufactured before 1990 (default threshold)

**Why:**
- **Different market dynamics:** Antique/collector cars price differently than modern used cars
- **Sparse data:** Very few antique cars in typical used car listings
- **Age-based features:** Cars > 35 years old break normal depreciation patterns
- **Outlier prevention:** Prevents extreme age values from skewing statistics

**Typical impact:** Removes < 1% of data (most listings are modern cars)

---

### Step 4: Remove 'Autre' Entries

**What it does:**
- Removes rows where brand or model is "autre" (French for "other/unknown")

**Why:**
- **No predictive value:** "Other" doesn't tell us anything about the car
- **Cannot compute statistics:** Can't calculate brand/model averages for "unknown"
- **User data quality:** Indicates incomplete/lazy data entry
- **Better to exclude:** More honest than pretending we can predict these

**Typical impact:** Removes 2-5% of data

---

### Step 5: IQR Outlier Detection (Per-Brand)

**What it does:**
- For each brand separately, calculates:
  - **Price outliers** using log-transformed prices
  - **Mileage outliers** using raw kilometers
- **Does NOT filter** `energie` or `puissance_din` - these are preserved for feature engineering
- Uses IQR (Interquartile Range) method:
  ```
  Lower bound = Q1 - 1.5 × IQR
  Upper bound = Q3 + 1.5 × IQR
  ```
- Removes cars outside these bounds

**Why use per-brand IQR instead of global?**
- **Prevents unfair removal:** A €100k Mercedes is normal; a €100k Dacia is fraudulent
- **Brand-specific pricing:** Different brands have vastly different price distributions
- **Accounts for brand variance:** Luxury brands have higher variance than economy brands

**Why log-transform prices but not kilometers?**
- **Prices are log-normal:** 10x price differences are common, log-transform normalizes this
- **Kilometers are more linear:** Used for IQR calculation, though extreme values still detected
- **Better outlier detection:** Log-price IQR catches proportional outliers better

**Example:**
```
Dacia brand: mean ~€12k, std ~€5k
- €60k Dacia → OUTLIER (likely data error)
- €6k Dacia → Normal

Mercedes brand: mean ~€40k, std ~€20k
- €60k Mercedes → Normal
- €6k Mercedes → Might be outlier (potential salvage/scam)
```

**IQR multiplier = 1.5:**
- Standard choice in statistics (same as boxplot whiskers)
- Not too aggressive (removes ~5-10% of data)
- Can be adjusted if needed via parameter

---

## 2. Feature Engineering Process

**Location:** `src/features/feature_engineering.py` → `CarPriceFeatureEngineer`

### Overview
Sklearn-compatible transformer that creates 30+ features from raw columns while avoiding data leakage through proper fit/transform separation.

**Important Note:** The `CarPriceFeatureEngineer` class does NOT currently handle `energie` or `puissance_din`. These variables are:
- Preserved by `clean_car_data()` 
- Available in the cleaned dataframe
- Can be processed separately (see notebook 5_advanced_feature_engineering.ipynb for examples)
- Commonly one-hot encoded (energie) or converted to numeric (puissance_din)

---

### Critical Design: Preventing Data Leakage

**The Problem:**
- Target encoding (using mean prices) can leak information from test set to training set
- Computing statistics on combined train+test data artificially improves performance

**The Solution:**
```python
fe.fit(X_train, y_train)      # Learns statistics ONLY from training data
X_train_fe = fe.transform(X_train)  # Uses learned statistics
X_test_fe = fe.transform(X_test)    # Uses same statistics (no leakage)
```

**What's learned in fit() (only from training data):**
- Brand/model average prices (target encoding)
- Global mean/median/std for unseen categories
- Mileage percentiles for categorical bins

**What's computed in transform() (safe operations):**
- Brand/model counts (based on training data counts)
- Car age (current_year - year)
- Kilometers per year
- All derived features

---

### Feature Categories

#### A. Time-Based Features (4 features)

**1. `car_age`**
```python
car_age = current_year - year
```
- **Why:** Primary depreciation driver - cars lose value with age
- **Linear relationship:** Each year typically reduces value by 10-15%

**2. `decade`**
```python
decade = (year // 10) * 10  # 2024 → 2020, 2015 → 2010
```
- **Why:** Captures generational design/technology changes
- **Examples:** 2010s (smartphone integration), 2000s (stricter emissions), 1990s (pre-airbag standards)

**3. `is_almost_new`**
```python
is_almost_new = (year >= current_year - 1)
```
- **Why:** Current/previous year cars have premium pricing (like "new" cars)
- **Market behavior:** Steep depreciation after 2 years

**4. `age_category` (optional)**
- Bins: new (0-2yr), recent (3-5yr), mid-age (6-10yr), older (11-15yr), very old (16+)
- **Why:** Non-linear depreciation (steep early, flattens later)
- **Alternative to polynomials:** More interpretable than car_age²

---

#### B. Mileage-Based Features (5 features)

**1. `km_per_year`**
```python
km_per_year = km / car_age  (handles age=0 case)
```
- **Why:** Usage intensity - 10k km/yr is light, 30k km/yr is heavy
- **Better than raw km:** A 5-year-old car with 150k km is heavily used; a 15-year-old is normal

**2. `is_low_mileage`** (< 50,000 km)
- **Why:** Low-mileage cars command premium prices
- **Market threshold:** 50k is common "low mileage" advertising cutoff

**3. `is_high_mileage`** (> 75th percentile)
- **Why:** High-mileage cars sell at discount
- **Dynamic threshold:** Uses training data percentile (adapts to dataset)

**4. `is_nearly_new_mileage`** (< 10,000 km)
- **Why:** "Like new" category - barely driven cars
- **Premium indicator:** Often ex-demo or dealer cars

**5. `mileage_category` (optional)**
- 5 bins based on percentiles: very low, low, medium, high, very high
- **Why:** Categorical alternative for non-linear mileage effects

---

#### C. Brand Aggregate Features (6 features)

**Computed in fit(), applied in transform()**

**1-3. Count/Demographics (always safe - no leakage):**
- `brand_count`: Number of this brand in training data
- `brand_avg_km`: Average kilometers for this brand
- `brand_avg_age`: Average age for this brand

**Why these are safe:**
- Don't use target variable (price)
- Counts/averages from training data applied to test data
- Unseen brands get global averages

**4-6. Target Encoding (optional - requires fit on y):**
- `brand_avg_price`: Mean price for brand (from training only)
- `brand_median_price`: Median price for brand
- `brand_price_std`: Price variability for brand

**Why target encoding:**
- **Massive signal:** Mercedes prices ≠ Dacia prices
- **Better than one-hot:** Reduces dimensionality (50 brands → 1 numeric feature)
- **Handles unseen:** New brands get global mean (graceful degradation)

**Minimum samples threshold (default=5):**
- Brands with < 5 cars don't get their own encoding
- Prevents overfitting to small samples
- Uses global mean instead

---

#### D. Model Aggregate Features (3 features)

Similar to brand features but at model level:
- `model_count`: Popularity indicator
- `model_avg_price`: Average price for specific model (target encoding)
- `model_median_price`: Median price for specific model

**Why model level:**
- **More granular:** Audi A3 ≠ Audi A8
- **Market segments:** Sports vs. family vs. luxury models
- **Complements brand:** Captures within-brand variation

---

#### E. Relative Features (1 feature)

**`model_popularity_ratio`**
```python
ratio = model_count / brand_count
```
- **Why:** Captures flagship vs. niche models
- **Examples:**
  - Renault Clio: ~40% of Renault sales (popular)
  - Renault Twizy: ~0.1% of sales (rare)
- **Price signal:** Popular models often better value (economies of scale)

---

#### F. Interaction Features (4 features)

**1. `age_km_interaction`**
```python
interaction = car_age × km / 1000
```
- **Why:** Captures "total use" - old AND high mileage compounds wear
- **Synergy effect:** Age and mileage together worse than sum of parts

**2. `is_low_use_recent`** (age < 5 AND km < 50k)
- **Why:** Premium category - barely used recent cars
- **Market behavior:** Often ex-leases or weekend cars

**3. `is_high_use_new`** (age < 3 AND km > 150k)
- **Why:** Taxi/fleet cars - heavy usage hurts value
- **Red flag:** Unusual pattern needs separate handling

**4. `is_garage_queen`** (age > 15 AND km < 50k)
- **Why:** Collector cars or barely-driven classics
- **Mixed signal:** Could be maintained gem or problematic sitter

---

#### G. Log/Polynomial Features (4 features, optional)

**1-2. Log transforms:**
- `log_km`: log(km + 1)
- `log_km_per_year`: log(km_per_year + 1)

**Why log transforms:**
- **Handles wide ranges:** 10k to 500k km
- **Diminishing returns:** 0→50k km matters more than 200k→250k km
- **Normalization:** Many ML models prefer normally-distributed features

**3-4. Polynomial:**
- `sqrt_km`: √km
- `car_age_squared`: age²

**Why polynomials:**
- **Non-linear depreciation:** Value drops fast early, slower later
- **Curved relationships:** Age² captures parabolic patterns
- **Model flexibility:** Helps linear models capture curves

---

## 3. Design Decisions & Rationale

### A. Per-Brand Outlier Detection

**Decision:** Calculate IQR separately for each brand

**Alternatives considered:**
1. Global IQR across all cars
2. Per-model IQR
3. No outlier removal

**Why chosen:**
- Global IQR would remove normal luxury cars
- Per-model too granular (many models have < 30 cars)
- Per-brand balances granularity with sample size
- No removal keeps obviously fraudulent listings

---

### B. Brand Grouping Strategy

**Decision:** Group 50-599 car brands by price tier

**Alternatives considered:**
1. Remove all rare brands
2. Keep all brands as-is
3. Group by country of origin
4. Group by vehicle type (sedan, SUV, etc.)

**Why chosen:**
- Removing loses too much data
- Keeping all creates sparse one-hot encoding
- Price tier preserves economic signal
- Country/type info not always available

**Threshold choice (50-599):**
- 50: Statistical significance (enough for percentiles)
- 600: Computational (one-hot encoding manageable up to ~50-60 brands)

---

### C. Target Encoding vs One-Hot Encoding

**Decision:** Use target encoding for brands/models

**Why not one-hot:**
- Would create 50+ brand columns + 500+ model columns
- Sparse matrix (99% zeros)
- Curse of dimensionality
- Computationally expensive

**Why target encoding:**
- Single numeric column captures price information
- Dense representation
- Handles unseen categories gracefully
- Proven effective for high-cardinality categoricals

**Leakage mitigation:**
- Fit only on training data
- Minimum sample threshold
- Global mean fallback

---

### D. Log-Transform Prices in Outlier Detection

**Decision:** Use log(price) for IQR bounds, not raw price

**Why:**
- Prices are log-normally distributed (multiplicative process)
- €10k→€20k is similar to €50k→€100k (both 2x)
- Linear IQR would be too tight for luxury brands, too loose for budget brands
- Log-space IQR captures proportional outliers

**Example:**
```
BMW raw prices: Q1=€25k, Q3=€60k, IQR=€35k
→ Bounds: €-27.5k to €112.5k (too wide!)

BMW log prices: Q1=10.13, Q3=11.00, IQR=0.87
→ Bounds: €11k to €135k (reasonable!)
```

---

### E. Current Year Parameter

**Decision:** Make current_year a parameter (default 2025)

**Why not hardcode:**
- Model retraining over time
- Testing with historical data
- Reproducibility

**Why 2025 default:**
- Code written in early 2025
- Can update annually

---

## 4. Questions for Review

### Critical Questions

**1. Brand Filtering Thresholds**
- Is 50 cars minimum too aggressive? Too lenient?
- Should luxury brands have different threshold?
- Alternative: Use percentage of dataset instead of absolute count?

**2. Antique Car Threshold (1990)**
- Should this be dynamic (e.g., current_year - 35)?
- Are 1990-2000 cars in the same market as 2020s?
- Consider: separate model for classics?

**3. IQR Multiplier (1.5)**
- Standard choice, but removes 5-10% of data
- Too aggressive? (Try 2.0 for more lenient)
- Too lenient? (Try 1.0 for stricter)

**4. Target Encoding Minimum Samples (5)**
- 5 cars enough for reliable mean?
- Should vary by variance? (luxury brands need more samples)

**5. Feature Selection**
- Are all 30+ features necessary?
- Any redundant combinations?
- Consider: automated feature selection?

### Methodological Questions

**6. Log vs Raw Outlier Detection**
- Log(price) for bounds but raw km - is this consistent?
- Should km also be log-transformed?

**7. Brand Grouping by Price**
- Is price tier the right grouping?
- Alternative: group by segment (economy/mid/luxury)?
- Alternative: use unsupervised clustering?

**8. Interaction Terms**
- Only 4 interactions - are we missing important ones?
- e.g., brand × age, fuel type × year, etc.?

**9. Missing Data Handling**
- Currently: filter out nulls in price/km/year
- Alternative: impute missing values?
- How many records lost due to missing data?

**10. Feature Scaling**
- No scaling applied in transformer
- Should we add StandardScaler option?
- Or leave to user's pipeline?

### Technical Debt

**11. Hardcoded Column Names**
- `km`, `year`, `brand`, `model` expected
- Should we make this more flexible?

**12. Polars Dependency**
- Entire pipeline uses Polars
- Alternative: support pandas too?
- Performance vs. compatibility trade-off?

**13. Unit Tests**
- No automated tests for cleaning pipeline
- What scenarios need testing?

### Business Logic

**14. Market Segmentation**
- Should we have separate models per brand tier?
- Do budget and luxury cars follow different depreciation?

**15. Geographic Considerations**
- No region/country features currently
- Does location matter for pricing?

**16. Seasonal Effects**
- No time-of-sale features
- Convertibles worth more in summer?

**17. Energie & Puissance_din Processing**
- Currently preserved but not automatically processed
- Should these be in the CarPriceFeatureEngineer?
- How to handle missing HP values?
- Fuel type interactions with brand/year?

---

## 5. Recommendations for Review

### Priority 1 (Core Methodology)
- [ ] Review per-brand IQR approach vs. alternatives
- [ ] Validate brand filtering thresholds with business knowledge
- [ ] Check if log(price) IQR aligns with domain expertise

### Priority 2 (Feature Engineering)
- [ ] Audit feature correlation/redundancy
- [ ] Consider additional interaction terms
- [ ] Evaluate target encoding vs. alternatives

### Priority 3 (Technical Improvements)
- [ ] Add automated tests
- [ ] Document performance benchmarks
- [ ] Consider scaling/normalization strategy

---

                                      ↓
                    [price, year, km, brand, model, energie, puissance_din]
                                      ↓
           Feature Engineering (fit on train) → [30+ features]
                                      ↓
                    Transform (train/test) → Modeling
```

**Note:** `energie` and `puissance_din` require separate preprocessing before/during feature engineering
Raw CSV → Type Conversion → Brand Filtering → Antique Removal 
→ Remove 'Autre' → IQR Outliers → Clean Data

Clean Data → Feature Engineering (fit on train) 
→ Transform (train/test) → Modeling
```

### Key Thresholds
| Parameter | Value | Purpose |
|-----------|-------|---------|
| min_brand_threshold | 50 | Remove rare brands |
| rare_brand_threshold | 600 | Group medium brands |
| price_iqr_multiplier | 1.5 | Price outlier sensitivity |
| km_iqr_multiplier | 1.5 | Mileage outlier sensitivity |
| min_year | 1990 | Exclude antique cars |
| min_samples_for_encoding | 5 | Target encoding reliability |
| current_year | 2025 | Age calculations |

### Feature Count Summary

**CarPriceFeatureEngineer creates:**
- Time features: 4
- Mileage features: 5
- Brand features: 6
- Model features: 3
- Relative features: 1
- Interaction features: 4
- Log/Polynomial: 4 (optional)
- **Subtotal:** 23-27 features created

**Additional variables available but not processed by CarPriceFeatureEngineer:**
- `energie` (fuel type): Typically one-hot encoded → 4-6 features
- `puissance_din` (horsepower): Extract numeric value → 1-2 features
- **Potential Total:** 28-35 features when including energie/HP

---

**Document Version:** 1.0  
**Last Updated:** January 30, 2026  
**Author:** Documentation for methodology review
