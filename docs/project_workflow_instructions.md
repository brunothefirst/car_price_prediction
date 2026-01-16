# Comprehensive Instructions for Car Price Prediction Project

Use these instructions as a base prompt for future tasks to ensure consistency and correctness in the workflow.

## 🎯 DATA LOADING & PREPROCESSING

**1. Data Loading**
- Use `data_processing.load_car_data()` or `data_processing.clean_car_data()` from the existing pipeline
- Data comes with French column names: `marque`, `modele`, `annee_modele`, `kilometrage`, `energie`, `puissance_din`, `price`
- After cleaning, columns are renamed to: `brand`, `model`, `year`, `km`, `energie`, `puissance_din`, `price`

**2. Critical Standardization (ALWAYS DO THIS FIRST)**
```python
df = df.with_columns([
    pl.col('brand').str.to_lowercase(),
    pl.col('model').str.to_lowercase()
])
```
- This ensures consistency with production data later

**3. Target Variable Transformation**
- **ALWAYS log-transform the price target** for price prediction models:
```python
df = df.with_columns([
    pl.col('price').log().alias('log_price')
])
```
- Do this BEFORE feature engineering so aggregates (brand_avg_price, model_avg_price) are computed in log space
- Train models on `log_price`, then exponentiate predictions back: `np.exp(predictions)`

## 🔧 FEATURE ENGINEERING

**1. Use the Feature Engineering Class**
- Import: `from features.feature_engineering import CarPriceFeatureEngineer`
- The class expects **TWO parameters** for fit:
```python
feature_engineer = CarPriceFeatureEngineer()

# CORRECT - Separate X and y
X_train = df_train.drop(['price', 'log_price'])
y_train = df_train['log_price']  # Use log_price
feature_engineer.fit(X_train, y_train)

# Transform
df_train_features = feature_engineer.transform(df_train.drop(['price', 'log_price']))
```

**2. Lean Feature Set**
For "lean" models (only brand, model, year):
- **Keep**: `brand`, `model`, `year`, `car_age`, and all aggregates derived from these
- **Include**: `brand_avg_price`, `brand_median_price`, `brand_price_std`, `model_avg_price`, `model_median_price`, `brand_count`, `model_count`, etc.
- **Exclude**: Any features using `km`, `kilometrage`, `fuel_type`, `energie`, `horsepower`, `puissance_din`
- **Exclude**: Derived features like `km_per_year`, `age_km_interaction`, `is_low_use_recent`

**3. Handle Categorical Variables**
- **Brand**: Use one-hot encoding (OHE) with `pd.get_dummies()` - only ~47 brands
- **Model**: DO NOT include in final features - too many unique values (thousands)
- Remove both `brand` and `model` from final feature matrix:
```python
X = df.drop(['brand', 'model', 'price', 'log_price'])
```

## 📊 DATA TYPE HANDLING

**Critical: Fix Polars → Pandas Conversion**
When converting from Polars to pandas, numeric columns may become 'object' dtype. ALWAYS apply this fix:
```python
X_pd = df_polars.to_pandas()

# Fix numeric dtypes
numeric_cols = X_pd.select_dtypes(include=['object']).columns
for col in numeric_cols:
    X_pd[col] = pd.to_numeric(X_pd[col], errors='coerce')
```

## 🤖 MODEL TRAINING

**1. Train/Test Split**
- Fit feature engineer ONLY on training data to prevent leakage
- Transform both train and test with the fitted feature engineer

**2. LightGBM Configuration**
```python
lgb_model = lgb.LGBMRegressor(
    objective='quantile',  # or 'regression'
    alpha=0.50,  # for quantile regression
    learning_rate=0.1,
    n_estimators=5000,
    random_state=42,
    verbose=-1
)
# No categorical_feature parameter needed if using OHE
lgb_model.fit(X_train, y_train_log)
```

**3. Production Training**
After validation, retrain on 100% of data:
```python
# Create NEW feature engineer for production
feature_engineer_prod = CarPriceFeatureEngineer()
X_all = df.drop(['price', 'log_price'])
y_all_log = df['log_price']
feature_engineer_prod.fit(X_all, y_all_log)

# Transform and train
X_all_features = feature_engineer_prod.transform(X_all)
# ... apply same OHE and cleaning steps ...
model_prod.fit(X_all_features, y_all_log)
```

## 💾 SAVING & LOADING

**Save all artifacts together:**
```python
models_dir = Path(MODELS_PATH) / "lean_quantile"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(lgb_q15_prod, models_dir / "lgb_q15_model.pkl")
joblib.dump(lgb_q50_prod, models_dir / "lgb_q50_model.pkl")
joblib.dump(lgb_q85_prod, models_dir / "lgb_q85_model.pkl")
joblib.dump(feature_engineer_prod, models_dir / "feature_engineer.pkl")
```

## 🚀 PRODUCTION PREDICTIONS

**1. Load Production Data**
- Standardize brand/model to lowercase IMMEDIATELY
- Add dummy `km` column if using feature engineer (even if not in lean features):
```python
df_prod = df_prod.with_columns([
    pl.col('brand').str.to_lowercase(),
    pl.col('model').str.to_lowercase(),
    pl.lit(100000).alias('km')  # Dummy for compatibility
])
```

**2. Filter to Seen Brands**
```python
training_brands = set(df_train['brand'].unique())
df_prod = df_prod.filter(pl.col('brand').is_in(training_brands))
```

**3. Apply Feature Engineering**
```python
# Load saved feature engineer (DO NOT REFIT)
feature_engineer_prod = joblib.load("feature_engineer.pkl")

# Transform only
df_prod_features = feature_engineer_prod.transform(df_prod.drop(['price']))
# ... apply same OHE and dtype fixes as training ...
```

**4. Make Predictions & Convert Back**
```python
predictions_log = model.predict(X_prod)
predictions_eur = np.exp(predictions_log)  # Convert from log to EUR
```

## 📈 VISUALIZATIONS

**Use matplotlib, not Plotly** (avoids nbformat issues):
- Show top 10 features for feature importance (not all features)
- Use `np.maximum(0, error_values)` for error bars to avoid negative values
- Sample large datasets (e.g., 5000 points) for scatter plots

## ⚠️ COMMON PITFALLS TO AVOID

1. ❌ Calling `feature_engineer.fit(df)` without separating X and y
2. ❌ Forgetting to lowercase brand/model before feature engineering
3. ❌ Not log-transforming the target variable
4. ❌ Including 'model' column with too many categories in final features
5. ❌ Not fixing pandas dtypes after Polars conversion
6. ❌ Refitting feature engineer on production data (causes data leakage)
7. ❌ Forgetting to exponentiate predictions back from log space
8. ❌ Not filtering production data to only seen brands