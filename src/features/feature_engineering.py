"""
Feature engineering transformer for car price prediction.

This module provides a scikit-learn compatible transformer that creates
features for car price prediction while avoiding data leakage.

The key distinction:
- Safe features (computed in transform): counts, km averages, age averages
- Leaky features (learned in fit): price-based averages (target encoding)

Target encoding statistics are computed only on training data in fit()
and applied in transform() to avoid data leakage.

Features created (matching notebook 03_feature_engineering):
- Time features: car_age, age_category, is_almost_new, decade
- Mileage features: km_per_year, mileage_category, is_low_mileage, is_high_mileage, is_nearly_new_mileage
- Brand features: brand_count, brand_avg_price, brand_median_price, brand_price_std, brand_avg_km, brand_avg_age
- Model features: model_count, model_avg_price, model_median_price
- Relative features: model_popularity_ratio
- Interaction features: age_km_interaction, is_low_use_recent, is_high_use_new, is_garage_queen
- Log features (optional): log_km, log_km_per_year, sqrt_km, car_age_squared

Example:
    >>> from src.features import CarPriceFeatureEngineer
    >>> import polars as pl
    >>> 
    >>> # Initialize and fit on training data
    >>> fe = CarPriceFeatureEngineer()
    >>> fe.fit(X_train, y_train)
    >>> 
    >>> # Transform training and test data
    >>> X_train_fe = fe.transform(X_train)
    >>> X_test_fe = fe.transform(X_test)
    >>> 
    >>> # Use in sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('features', CarPriceFeatureEngineer()),
    ...     ('model', YourModel())
    ... ])
"""

import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Union, Tuple
import warnings


class CarPriceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for car prices.
    
    This transformer creates features for car price prediction while properly
    handling target encoding to avoid data leakage. Price-based statistics
    (target encoding) are computed only during fit() on training data.
    
    Features created:
    
    Time-based features (4):
        - car_age: Age of the car in years
        - age_category: Categorical age grouping (new_0-2yr, recent_3-5yr, etc.)
        - is_almost_new: Boolean for cars manufactured in current year
        - decade: Decade of manufacture (2020, 2010, etc.)
    
    Mileage-based features (5):
        - km_per_year: Average kilometers driven per year
        - mileage_category: Very low/low/medium/high/very high
        - is_low_mileage: Boolean for < 50,000 km
        - is_high_mileage: Boolean for > 75th percentile
        - is_nearly_new_mileage: Boolean for < 10,000 km
    
    Brand aggregate features (6) - Target encoding computed in fit():
        - brand_count: Number of cars of this brand in training data
        - brand_avg_price: Average price for this brand (from training)
        - brand_median_price: Median price for this brand (from training)
        - brand_price_std: Price standard deviation for this brand
        - brand_avg_km: Average kilometers for this brand
        - brand_avg_age: Average age for this brand
    
    Model aggregate features (3) - Target encoding computed in fit():
        - model_count: Number of cars of this model in training data
        - model_avg_price: Average price for this model (from training)
        - model_median_price: Median price for this model (from training)
    
    Relative features (1):
        - model_popularity_ratio: Model count / Brand count
    
    Interaction features (4):
        - age_km_interaction: Age × Mileage / 1000
        - is_low_use_recent: Recent car (< 5yr) with low mileage (< 50k)
        - is_high_use_new: New car (< 3yr) with very high mileage (> 150k)
        - is_garage_queen: Old car (> 15yr) with very low mileage (< 50k)
    
    Optional log/polynomial features:
        - log_km: Log-transformed kilometers
        - sqrt_km: Square root of kilometers
        - log_km_per_year: Log-transformed km per year
        - car_age_squared: Squared car age
    
    Parameters
    ----------
    current_year : int, optional
        The current year for calculating car age. Default is 2025.
    min_samples_for_encoding : int, optional
        Minimum number of samples required to compute target encoding
        for a category. Categories with fewer samples use global mean.
        Default is 5.
    add_log_features : bool, optional
        Whether to add log-transformed features. Default is True.
    add_polynomial_features : bool, optional
        Whether to add polynomial features (squared terms). Default is True.
    add_target_encoding : bool, optional
        Whether to add target encoding features. Default is True.
    add_categorical_features : bool, optional
        Whether to add categorical features (age_category, mileage_category). Default is True.
    add_interaction_features : bool, optional
        Whether to add interaction features. Default is True.
    
    Attributes
    ----------
    brand_price_stats_ : dict
        Dictionary mapping brand -> (mean_price, median_price, std_price, count)
    model_price_stats_ : dict
        Dictionary mapping model -> (mean_price, median_price, std_price, count)
    brand_km_stats_ : dict
        Dictionary mapping brand -> (mean_km, mean_age, count)
    model_km_stats_ : dict
        Dictionary mapping model -> (mean_km, mean_age, count)
    km_percentiles_ : dict
        Dictionary with km percentile thresholds (p25, p50, p75, p90)
    global_mean_ : float
        Global mean price from training data (used for unseen categories)
    global_median_ : float
        Global median price from training data
    global_std_ : float
        Global standard deviation from training data
    is_fitted_ : bool
        Whether the transformer has been fitted
    """
    
    def __init__(
        self,
        current_year: int = 2025,
        min_samples_for_encoding: int = 5,
        add_log_features: bool = True,
        add_polynomial_features: bool = True,
        add_target_encoding: bool = True,
        add_categorical_features: bool = True,
        add_interaction_features: bool = True,
        standardize: bool = False,
        brand_onehot: bool = False
    ):
        self.current_year = current_year
        self.min_samples_for_encoding = min_samples_for_encoding
        self.add_log_features = add_log_features
        self.add_polynomial_features = add_polynomial_features
        self.add_target_encoding = add_target_encoding
        self.add_categorical_features = add_categorical_features
        self.add_interaction_features = add_interaction_features
        self.standardize = standardize
        self.brand_onehot = brand_onehot
        
        # Initialize attributes for learned statistics
        self.brand_price_stats_: Dict[str, Tuple[float, float, float, int]] = {}  # (log_price_mean, log_price_median, log_price_std, count)
        self.model_price_stats_: Dict[str, Tuple[float, float, float, int]] = {}  # (log_price_mean, log_price_median, log_price_std, count)
        self.brand_km_stats_: Dict[str, Tuple[float, float, float, float, int]] = {}  # (km_mean, km_median, age_mean, age_median, count)
        self.model_km_stats_: Dict[str, Tuple[float, float, int]] = {}  # (km_mean, age_mean, count)
        self.km_percentiles_: Dict[str, float] = {}
        self.global_mean_: Optional[float] = None
        self.global_median_: Optional[float] = None
        self.global_std_: Optional[float] = None
        self.global_km_mean_: Optional[float] = None
        self.global_age_mean_: Optional[float] = None
        self.scaler: Optional[StandardScaler] = None
        self.brand_columns_: list = []  # Store brand dummy column names
        self.is_fitted_: bool = False
    
    def fit(self, X: Union[pl.DataFrame, np.ndarray], y: Union[pl.Series, np.ndarray] = None):
        """
        Fit the transformer by computing statistics from training data.
        
        This method computes:
        - Target encoding statistics (brand/model average/median prices)
        - Non-leaky aggregate statistics (brand/model avg km, age, counts)
        - Mileage percentiles for categorical features
        
        Parameters
        ----------
        X : polars.DataFrame or numpy.ndarray
            Training features. Must contain columns: 'brand', 'model', 
            'kilometers', 'year' (or 'annee')
        y : polars.Series or numpy.ndarray
            Target variable (price). Required for target encoding.
        
        Returns
        -------
        self : CarPriceFeatureEngineer
            The fitted transformer
        
        Raises
        ------
        ValueError
            If y is None and add_target_encoding is True
        """
        # Convert to polars if needed
        X_df = self._to_polars(X)
        
        # Validate required columns
        self._validate_columns(X_df)
        
        # Convert y to numpy array
        if y is not None:
            if isinstance(y, pl.Series):
                y_arr = y.to_numpy()
            elif isinstance(y, pl.DataFrame):
                y_arr = y.to_numpy().flatten()
            else:
                y_arr = np.array(y).flatten()
        else:
            y_arr = None
        
        # Compute global statistics (use log_price for brand aggregates)
        if y_arr is not None:
            # Compute log_price for aggregates
            log_price = np.log(y_arr)
            self.global_mean_ = float(np.mean(log_price))
            self.global_median_ = float(np.median(log_price))
            self.global_std_ = float(np.std(log_price))
        
        # Get column names (handle potential aliases)
        km_col ="km" #self._get_column_name(X_df, ['kilometers', 'km', 'kilometrage'])
        year_col = "year" #self._get_column_name(X_df, ['year', 'annee', 'année'])
        brand_col = "brand" #self._get_column_name(X_df, ['brand', 'marque'])
        model_col = "model" #self._get_column_name(X_df, ['model', 'modele', 'modèle'])

        # Compute car age for statistics
        km_values = X_df[km_col].to_numpy()
        year_values = X_df[year_col].to_numpy()
        age_values = self.current_year - year_values
        
        # Store km percentiles for mileage categories
        self.km_percentiles_ = {
            'p25': float(np.percentile(km_values, 25)),
            'p50': float(np.percentile(km_values, 50)),
            'p75': float(np.percentile(km_values, 75)),
            'p90': float(np.percentile(km_values, 90))
        }
        
        # Global km and age statistics
        self.global_km_mean_ = float(np.mean(km_values))
        self.global_age_mean_ = float(np.mean(age_values))
        
        # Get brand and model values
        brands = X_df[brand_col].to_list()
        models = X_df[model_col].to_list()
        
        # Compute brand-level statistics
        self._compute_brand_stats(brands, km_values, age_values, y_arr)
        
        # Compute model-level statistics (including brand for model_popularity_ratio)
        self._compute_model_stats(brands, models, km_values, age_values, y_arr)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: Union[pl.DataFrame, np.ndarray]) -> pl.DataFrame:
        """
        Transform the data by adding engineered features.
        
        Uses statistics learned during fit() to create features.
        Unseen categories receive the global mean from training data.
        
        Parameters
        ----------
        X : polars.DataFrame or numpy.ndarray
            Features to transform. Must contain same columns as fit.
        
        Returns
        -------
        polars.DataFrame
            Transformed dataframe with additional features
        
        Raises
        ------
        RuntimeError
            If transform is called before fit
        """
        if not self.is_fitted_:
            raise RuntimeError("Transform called before fit. Call fit() first.")
        
        # Convert to polars if needed
        X_df = self._to_polars(X).clone()
        
        # Get column names
        km_col ='km'# self._get_column_name(X_df, ['kilometers', 'km', 'kilometrage'])
        year_col = 'year' #self._get_column_name(X_df, ['year', 'annee', 'année'])
        brand_col = 'brand' #self._get_column_name(X_df, ['brand', 'marque'])
        model_col = 'model' #self._get_column_name(X_df, ['model', 'modele', 'modèle'])

        # ============================================================
        # TIME-BASED FEATURES (4)
        # ============================================================
        
        # Car age
        X_df = X_df.with_columns([
            (pl.lit(self.current_year) - pl.col(year_col)).alias('car_age')
        ])
        
        # Decade of manufacture
        X_df = X_df.with_columns([
            ((pl.col(year_col) // 10) * 10).alias('decade')
        ])
        
        # Is almost new (manufactured this year or last year)
        X_df = X_df.with_columns([
            (pl.col(year_col) >= self.current_year - 1).alias('is_almost_new')
        ])
        
        if self.add_categorical_features:
            # Age category
            X_df = X_df.with_columns([
                pl.when(pl.col(year_col) >= self.current_year - 2)
                    .then(pl.lit('new_0-2yr'))
                .when(pl.col(year_col) >= self.current_year - 5)
                    .then(pl.lit('recent_3-5yr'))
                .when(pl.col(year_col) >= self.current_year - 10)
                    .then(pl.lit('mid_age_6-10yr'))
                .when(pl.col(year_col) >= self.current_year - 15)
                    .then(pl.lit('older_11-15yr'))
                .otherwise(pl.lit('very_old_16+yr'))
                .alias('age_category')
            ])
        
        # ============================================================
        # MILEAGE-BASED FEATURES (5)
        # ============================================================
        
        # Km per year (avoid division by zero)
        X_df = X_df.with_columns([
            pl.when(pl.col('car_age') > 0)
                .then(pl.col(km_col) / pl.col('car_age'))
                .otherwise(pl.col(km_col))
                .alias('km_per_year')
        ])
        
        # Boolean mileage flags
        X_df = X_df.with_columns([
            (pl.col(km_col) < 50000).alias('is_low_mileage'),
            (pl.col(km_col) > self.km_percentiles_['p75']).alias('is_high_mileage'),
            (pl.col(km_col) < 10000).alias('is_nearly_new_mileage')
        ])
        
        if self.add_categorical_features:
            # Mileage category using learned percentiles
            X_df = X_df.with_columns([
                pl.when(pl.col(km_col) < self.km_percentiles_['p25'])
                    .then(pl.lit('very_low'))
                .when(pl.col(km_col) < self.km_percentiles_['p50'])
                    .then(pl.lit('low'))
                .when(pl.col(km_col) < self.km_percentiles_['p75'])
                    .then(pl.lit('medium'))
                .when(pl.col(km_col) < self.km_percentiles_['p90'])
                    .then(pl.lit('high'))
                .otherwise(pl.lit('very_high'))
                .alias('mileage_category')
            ])
        
        # ============================================================
        # LOG/POLYNOMIAL FEATURES (optional)
        # ============================================================
        
        if self.add_log_features:
            X_df = X_df.with_columns([
                (pl.col(km_col) + 1).log().alias('log_km'),
                (pl.col('km_per_year') + 1).log().alias('log_km_per_year')
            ])
        
        if self.add_polynomial_features:
            # Rename km to mileage for consistency
            X_df = X_df.with_columns([
                pl.col(km_col).alias('mileage')
            ])
            
            # Add polynomial features for age and mileage
            X_df = X_df.with_columns([
                (pl.col('car_age') ** 2).alias('age_squared'),
                (pl.col('car_age') ** 3).alias('age_cubed'),
                (pl.col('mileage') ** 2).alias('mileage_squared'),
                (pl.col('mileage') ** 3).alias('mileage_cubed'),
                # Legacy features
                (pl.col(km_col).sqrt()).alias('sqrt_km'),
                (pl.col('car_age') ** 2).alias('car_age_squared')
            ])
        
        # ============================================================
        # BRAND FEATURES: ONE-HOT ENCODING OR AGGREGATES (7)
        # ============================================================
        
        brand_values = X_df[brand_col].to_list()
        
        if self.brand_onehot:
            # Option 1: One-hot encode brand (skip aggregates)
            # Convert to pandas for get_dummies, then back to polars
            df_pandas = X_df.to_pandas()
            brand_dummies = pd.get_dummies(df_pandas[brand_col], prefix='brand', drop_first=True)
            
            # Store column names during fit
            if not self.is_fitted_:
                self.brand_columns_ = brand_dummies.columns.tolist()
            
            # During transform, ensure same columns as training
            if self.is_fitted_:
                # Add missing columns (unseen brands in test set)
                for col in self.brand_columns_:
                    if col not in brand_dummies.columns:
                        brand_dummies[col] = 0
                # Keep only training columns
                brand_dummies = brand_dummies[self.brand_columns_]
            
            # Add dummies to dataframe
            df_pandas = pd.concat([df_pandas.drop(columns=[brand_col]), brand_dummies], axis=1)
            X_df = pl.from_pandas(df_pandas)
            
            # Create empty brand_counts list for model_popularity_ratio calculation
            brand_counts = [1] * len(brand_values)  # Use 1 to avoid division issues
            
        else:
            # Option 2: Create brand aggregate features (default)
            brand_avg_km = []
            brand_median_km = []
            brand_avg_age = []
            brand_median_age = []
            brand_mean_log_price = []
            brand_median_log_price = []
            brand_std_log_price = []
            brand_counts = []
            
            for brand in brand_values:
                if brand in self.brand_km_stats_:
                    km_mean, km_median, age_mean, age_median, count = self.brand_km_stats_[brand]
                    brand_avg_km.append(km_mean)
                    brand_median_km.append(km_median)
                    brand_avg_age.append(age_mean)
                    brand_median_age.append(age_median)
                    brand_counts.append(count)
                else:
                    # Unseen brand - use global statistics
                    brand_avg_km.append(self.global_km_mean_)
                    brand_median_km.append(self.global_km_mean_)  # Use mean as fallback
                    brand_avg_age.append(self.global_age_mean_)
                    brand_median_age.append(self.global_age_mean_)  # Use mean as fallback
                    brand_counts.append(1)  # Minimal count for unseen brands
                
                # Target encoding (log_price statistics)
                if self.add_target_encoding:
                    if brand in self.brand_price_stats_:
                        log_price_mean, log_price_median, log_price_std, _ = self.brand_price_stats_[brand]
                        brand_mean_log_price.append(log_price_mean)
                        brand_median_log_price.append(log_price_median)
                        brand_std_log_price.append(log_price_std)
                    else:
                        # Unseen brand - use global statistics
                        brand_mean_log_price.append(self.global_mean_)
                        brand_median_log_price.append(self.global_median_)
                        brand_std_log_price.append(self.global_std_)
            
            # Add aggregate features
            X_df = X_df.with_columns([
                pl.Series('brand_avg_km', brand_avg_km),
                pl.Series('brand_median_km', brand_median_km),
                pl.Series('brand_avg_age', brand_avg_age),
                pl.Series('brand_median_age', brand_median_age)
            ])
            
            if self.add_target_encoding:
                X_df = X_df.with_columns([
                    pl.Series('brand_mean_log_price', brand_mean_log_price),
                    pl.Series('brand_median_log_price', brand_median_log_price),
                    pl.Series('brand_std_log_price', brand_std_log_price)
                ])
            
            # Drop original brand column
            X_df = X_df.drop(brand_col)
        
        # ============================================================
        # MODEL AGGREGATE FEATURES (3) + RELATIVE FEATURES (1)
        # ============================================================
        
        model_values = X_df[model_col].to_list()
        
        model_counts = []
        model_avg_price = []
        model_median_price = []
        model_popularity_ratio = []
        
        for i, model in enumerate(model_values):
            brand = brand_values[i]
            
            if model in self.model_km_stats_:
                _, _, count = self.model_km_stats_[model]
                model_counts.append(count)
            else:
                # Unseen model
                model_counts.append(0)
            
            # Model popularity ratio (model_count / brand_count)
            brand_count = brand_counts[i] if brand_counts[i] > 0 else 1
            model_count = model_counts[-1]
            model_popularity_ratio.append(model_count / brand_count)
            
            # Target encoding (price statistics)
            if self.add_target_encoding:
                if model in self.model_price_stats_:
                    price_mean, price_median, _, _ = self.model_price_stats_[model]
                    model_avg_price.append(price_mean)
                    model_median_price.append(price_median)
                else:
                    # Unseen model - use global statistics
                    model_avg_price.append(self.global_mean_)
                    model_median_price.append(self.global_median_)
        
        X_df = X_df.with_columns([
            pl.Series('model_count', model_counts),
            pl.Series('model_popularity_ratio', model_popularity_ratio)
        ])
        
        if self.add_target_encoding:
            X_df = X_df.with_columns([
                pl.Series('model_avg_price', model_avg_price),
                pl.Series('model_median_price', model_median_price)
            ])
                # Drop model column after features are created
        X_df = X_df.drop(model_col)
                # ============================================================
        # INTERACTION FEATURES (7)
        # ============================================================
        
        if self.add_interaction_features:
            # Ensure mileage column exists for interactions
            if 'mileage' not in X_df.columns:
                X_df = X_df.with_columns([pl.col(km_col).alias('mileage')])
            
            X_df = X_df.with_columns([
                # Polynomial interaction features (new)
                (pl.col('car_age') * pl.col('mileage')).alias('age_mileage'),
                ((pl.col('car_age') ** 2) * pl.col('mileage')).alias('age_squared_mileage'),
                (pl.col('car_age') * (pl.col('mileage') ** 2)).alias('age_mileage_squared'),
                
                # Legacy interaction (heavily used old cars)
                (pl.col('car_age') * pl.col(km_col) / 1000).alias('age_km_interaction'),
                
                # Boolean combinations
                ((pl.col('car_age') < 5) & (pl.col(km_col) < 50000)).alias('is_low_use_recent'),
                ((pl.col('car_age') < 3) & (pl.col(km_col) > 150000)).alias('is_high_use_new'),
                ((pl.col('car_age') > 15) & (pl.col(km_col) < 50000)).alias('is_garage_queen')
            ])
        
        # ============================================================
        # Z-SCORE STANDARDIZATION (optional, applied last)
        # ============================================================
        
        if self.standardize:
            # Convert to pandas for StandardScaler
            df_pandas = X_df.to_pandas()
            
            # Get numeric columns
            numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude target variables and ID columns from standardization
            exclude_cols = ['price', 'log_price', 'id', 'car_id', 'listing_id', year_col]
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            if self.is_fitted_ and self.scaler is not None:
                # Transform using fitted scaler
                df_pandas[numeric_cols] = self.scaler.transform(df_pandas[numeric_cols])
            else:
                # Fit and transform (during fit_transform)
                self.scaler = StandardScaler()
                df_pandas[numeric_cols] = self.scaler.fit_transform(df_pandas[numeric_cols])
            
            # Convert back to polars
            X_df = pl.from_pandas(df_pandas)
        else:
            self.scaler = None
        
        return X_df
    
    def fit_transform(self, X: Union[pl.DataFrame, np.ndarray], y: Union[pl.Series, np.ndarray] = None) -> pl.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        X : polars.DataFrame or numpy.ndarray
            Training features
        y : polars.Series or numpy.ndarray
            Target variable (price)
        
        Returns
        -------
        polars.DataFrame
            Transformed dataframe with additional features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> dict:
        """
        Get list of feature names created by this transformer, organized by category.
        
        Returns
        -------
        dict
            Dictionary with feature categories as keys and lists of feature names
        """
        features = {
            'time_features': ['car_age', 'decade', 'is_almost_new'],
            'mileage_features': ['km_per_year', 'is_low_mileage', 'is_high_mileage', 'is_nearly_new_mileage'],
            'brand_features': ['brand_count', 'brand_avg_km', 'brand_avg_age'],
            'model_features': ['model_count'],
            'relative_features': ['model_popularity_ratio']
        }
        
        if self.add_categorical_features:
            features['time_features'].append('age_category')
            features['mileage_features'].append('mileage_category')
        
        if self.add_log_features:
            features['log_features'] = ['log_km', 'log_km_per_year']
        
        if self.add_polynomial_features:
            features['polynomial_features'] = ['sqrt_km', 'car_age_squared']
        
        if self.add_target_encoding:
            features['brand_features'].extend(['brand_avg_price', 'brand_median_price', 'brand_price_std'])
            features['model_features'].extend(['model_avg_price', 'model_median_price'])
        
        if self.add_interaction_features:
            features['interaction_features'] = [
                'age_km_interaction', 'is_low_use_recent', 'is_high_use_new', 'is_garage_queen'
            ]
        
        return features
    
    def get_feature_names_flat(self) -> list:
        """
        Get a flat list of all feature names created by this transformer.
        
        Returns
        -------
        list
            List of all feature names
        """
        features_dict = self.get_feature_names()
        all_features = []
        for category_features in features_dict.values():
            all_features.extend(category_features)
        return all_features
    
    def _compute_brand_stats(
        self, 
        brands: list, 
        km_values: np.ndarray, 
        age_values: np.ndarray, 
        y_arr: Optional[np.ndarray]
    ):
        """Compute brand-level statistics."""
        from collections import defaultdict
        
        # Group by brand
        brand_km = defaultdict(list)
        brand_age = defaultdict(list)
        brand_price = defaultdict(list)
        
        for i, brand in enumerate(brands):
            brand_km[brand].append(km_values[i])
            brand_age[brand].append(age_values[i])
            if y_arr is not None:
                brand_price[brand].append(y_arr[i])
        
        # Compute statistics
        for brand in brand_km:
            km_mean = float(np.mean(brand_km[brand]))
            km_median = float(np.median(brand_km[brand]))
            age_mean = float(np.mean(brand_age[brand]))
            age_median = float(np.median(brand_age[brand]))
            count = len(brand_km[brand])
            self.brand_km_stats_[brand] = (km_mean, km_median, age_mean, age_median, count)
            
            # Price statistics (target encoding) - use log_price
            if y_arr is not None and brand in brand_price:
                prices = np.array(brand_price[brand])
                if len(prices) >= self.min_samples_for_encoding:
                    log_prices = np.log(prices)
                    log_price_mean = float(np.mean(log_prices))
                    log_price_median = float(np.median(log_prices))
                    log_price_std = float(np.std(log_prices)) if len(log_prices) > 1 else self.global_std_
                    self.brand_price_stats_[brand] = (log_price_mean, log_price_median, log_price_std, len(prices))
                # Categories with few samples don't get their own encoding
                # They'll use global mean in transform()
    
    def _compute_model_stats(
        self, 
        brands: list,
        models: list, 
        km_values: np.ndarray, 
        age_values: np.ndarray, 
        y_arr: Optional[np.ndarray]
    ):
        """Compute model-level statistics."""
        from collections import defaultdict
        
        # Group by model
        model_km = defaultdict(list)
        model_age = defaultdict(list)
        model_price = defaultdict(list)
        
        for i, model in enumerate(models):
            model_km[model].append(km_values[i])
            model_age[model].append(age_values[i])
            if y_arr is not None:
                model_price[model].append(y_arr[i])
        
        # Compute statistics
        for model in model_km:
            km_mean = float(np.mean(model_km[model]))
            age_mean = float(np.mean(model_age[model]))
            count = len(model_km[model])
            self.model_km_stats_[model] = (km_mean, age_mean, count)
            
            # Price statistics (target encoding)
            if y_arr is not None and model in model_price:
                prices = model_price[model]
                if len(prices) >= self.min_samples_for_encoding:
                    price_mean = float(np.mean(prices))
                    price_median = float(np.median(prices))
                    price_std = float(np.std(prices)) if len(prices) > 1 else self.global_std_
                    self.model_price_stats_[model] = (price_mean, price_median, price_std, len(prices))
    
    def _to_polars(self, X: Union[pl.DataFrame, np.ndarray]) -> pl.DataFrame:
        """Convert input to polars DataFrame."""
        if isinstance(X, pl.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            raise ValueError(
                "numpy array input not supported. Please provide a polars DataFrame "
                "with columns: brand, model, kilometers, year"
            )
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")
    
    def _validate_columns(self, df: pl.DataFrame):
        """Validate that required columns are present."""
        required_col_groups = [
            ['brand', 'marque'],
            ['model', 'modele', 'modèle'],
            ['kilometers', 'km', 'kilometrage'],
            ['year', 'annee', 'année']
        ]
        
        for col_group in required_col_groups:
            found = any(col in df.columns for col in col_group)
            if not found:
                raise ValueError(
                    f"Missing required column. Expected one of: {col_group}. "
                    f"Available columns: {df.columns}"
                )
    
    def _get_column_name(self, df: pl.DataFrame, possible_names: list) -> str:
        """Get the actual column name from a list of possibilities."""
        for name in possible_names:
            if name in df.columns:
                return name
        raise ValueError(f"Column not found. Expected one of: {possible_names}")
    
    def __repr__(self):
        return (
            f"CarPriceFeatureEngineer("
            f"current_year={self.current_year}, "
            f"min_samples_for_encoding={self.min_samples_for_encoding}, "
            f"add_log_features={self.add_log_features}, "
            f"add_polynomial_features={self.add_polynomial_features}, "
            f"add_target_encoding={self.add_target_encoding}, "
            f"add_categorical_features={self.add_categorical_features}, "
            f"add_interaction_features={self.add_interaction_features}, "
            f"standardize={self.standardize}, "
            f"brand_onehot={self.brand_onehot})"
        )
