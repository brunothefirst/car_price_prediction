"""
Data Processing Module for Car Price Prediction

This module handles:
1. Data cleaning (currently implemented)
2. Feature engineering (to be added)

All operations use Polars DataFrames for performance.
"""

import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
import re
from config import DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH

warnings.filterwarnings('ignore')


class CarDataProcessor:
    """
    Main class for processing car data through cleaning and feature engineering.
    
    Parameters
    ----------
    min_brand_count : int, default=400
        Minimum number of cars required to keep a brand
    price_iqr_multiplier : float, default=1.5
        IQR multiplier for price outlier detection
    km_iqr_multiplier : float, default=1.5
        IQR multiplier for kilometer outlier detection
    hp_iqr_multiplier : float, default=1.5
        IQR multiplier for horsepower outlier detection
    min_year : int, default=1990
        Minimum car year to keep (removes antique cars)
    price_floor : float or None, default=None
        Hard lower bound on price in €.  Rows below this value are dropped
        before IQR.  Set to None to skip.
    price_ceiling : float or None, default=None
        Hard upper bound on price in €.  Rows above this value are dropped
        before IQR.  Set to None to skip.
    iqr_large_group_threshold : int, default=50
        (brand, year) groups with at least this many cars use direct IQR;
        smaller groups pool year±1 neighbours before computing bounds.
    iqr_min_group_size : int, default=10
        Minimum pool size (after year±1 expansion) required to compute IQR
        bounds.  Groups below this threshold are dropped entirely.
    verbose : bool, default=True
        Whether to print progress messages
    """
    
    def __init__(
        self,
        min_brand_count: int = 400,
        price_iqr_multiplier: float = 1.5,
        km_iqr_multiplier: float = 1.5,
        hp_iqr_multiplier: float = 1.5,
        min_year: int = 1990,
        price_floor: Optional[float] = None,
        price_ceiling: Optional[float] = None,
        iqr_large_group_threshold: int = 50,
        iqr_min_group_size: int = 10,
        verbose: bool = True
    ):
        self.min_brand_count = min_brand_count
        self.price_iqr_multiplier = price_iqr_multiplier
        self.km_iqr_multiplier = km_iqr_multiplier
        self.hp_iqr_multiplier = hp_iqr_multiplier
        self.min_year = min_year
        self.price_floor = price_floor
        self.price_ceiling = price_ceiling
        self.iqr_large_group_threshold = iqr_large_group_threshold
        self.iqr_min_group_size = iqr_min_group_size
        self.verbose = verbose
        
        # Store statistics for later inspection
        self.cleaning_stats = {}
        
    def _log(self, message: str):
        """Print message if verbose is True."""
        if self.verbose:
            print(message)
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by converting to lowercase, removing accents, and cleaning special characters."""
        if not text or not isinstance(text, str):
            return text
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Convert common accented characters
        accent_map = {
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ä': 'a',
            'ù': 'u', 'û': 'u', 'ü': 'u',
            'ô': 'o', 'ö': 'o',
            'î': 'i', 'ï': 'i',
            'ç': 'c',
            'ñ': 'n'
        }
        for accent, replacement in accent_map.items():
            text = text.replace(accent, replacement)
        
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        return text
    
    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Parameters
        ----------
        df : pl.DataFrame
            Raw DataFrame with columns: price, marque, modele, annee_modele, kilometrage
            
        Returns
        -------
        pl.DataFrame
            Cleaned DataFrame with columns: price, year, km, brand, model
        """
        self._log("🧹 Starting data cleaning pipeline...")
        self._log("=" * 60)
        
        # Step 1: Data type conversion and text normalization
        df = self._convert_data_types(df)

        # Step 1b: Hard price bounds (optional)
        df = self._apply_price_bounds(df)
        
        # Step 2: Remove antique cars
        df = self._remove_antique_cars(df)
        
        # Step 3: Remove 'autre' entries
        df = self._remove_autre_entries(df)
        
        # Step 4: Clean horsepower (hard bounds + IQR per brand)
        df = self._clean_horsepower(df)
        
        # Step 5: Drop rare brands
        df = self._filter_rare_brands(df)
        
        # Step 6: IQR outlier detection for price and km
        df = self._remove_outliers_iqr(df)
        
        self._log("\n✅ Data cleaning completed!")
        self._log(f"Final dataset: {df.height:,} rows × {df.width} columns")
        
        return df
    
    def _convert_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert string columns to appropriate numeric types and normalize text.
        
        Parameters
        ----------
        df : pl.DataFrame
            Raw DataFrame
            
        Returns
        -------
        pl.DataFrame
            DataFrame with numeric types and normalized text
        """
        self._log("\n1️⃣ Converting data types and normalizing text...")
        
        # Clean price column
        df = df.with_columns(
            pl.when(pl.col('price').str.replace_all(r'[^\d.]', '') == "")
                .then(None)
                .otherwise(pl.col('price').str.replace_all(r'[^\d.]', ''))
                .cast(pl.Float64)
                .alias('price_numeric')
        )
        
        # Clean year and kilometers columns
        df = df.with_columns([
            pl.col("annee_modele").cast(pl.Float64, strict=False).alias("year_numeric"),
            pl.col("kilometrage").str.replace_all(r"[, km]", "").cast(pl.Float64, strict=False).alias("km_numeric")
        ])
        
        # Normalize brand and model text
        df = df.with_columns([
            pl.col('marque').map_elements(self._normalize_text, return_dtype=pl.Utf8).alias('brand_normalized'),
            pl.col('modele').map_elements(self._normalize_text, return_dtype=pl.Utf8).alias('model_normalized')
        ])
        
        # Create simplified DataFrame with renamed columns
        # Note: puissance_din is dropped after creating horsepower; energie (fuel type) is kept
        df_clean = df.select([
            pl.col('price_numeric').alias('price'),
            pl.col('year_numeric').alias('year'), 
            pl.col('km_numeric').alias('km'),
            pl.col('brand_normalized').alias('brand'),
            pl.col('model_normalized').alias('model'),
            pl.col('energie'),
            pl.col('horsepower')
        ]).filter(
            pl.col('price').is_not_null()
        )
        
        rows_before = df.height
        rows_after = df_clean.height
        
        # Log unique brands and models
        n_brands = df_clean['brand'].n_unique()
        n_models = df_clean['model'].n_unique()
        
        self.cleaning_stats['type_conversion'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after,
            'unique_brands': n_brands,
            'unique_models': n_models
        }
        
        self._log(f"   Original: {rows_before:,} rows")
        self._log(f"   After conversion: {rows_after:,} rows")
        self._log(f"   Removed (invalid price): {rows_before - rows_after:,}")
        self._log(f"   Unique brands: {n_brands}, Unique models: {n_models}")
        
        return df_clean
    
    def _apply_price_bounds(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop rows outside hard price bounds (floor / ceiling).

        Both bounds are optional (None = not applied).  Runs after type
        conversion so the price column is already numeric.
        """
        if self.price_floor is None and self.price_ceiling is None:
            return df

        self._log("\n1️⃣b Applying hard price bounds...")
        rows_before = df.height

        mask = pl.lit(True)
        if self.price_floor is not None:
            mask = mask & (pl.col('price') >= self.price_floor)
        if self.price_ceiling is not None:
            mask = mask & (pl.col('price') <= self.price_ceiling)

        df_bounded = df.filter(mask)
        n_below = (df['price'] < self.price_floor).sum() if self.price_floor is not None else 0
        n_above = (df['price'] > self.price_ceiling).sum() if self.price_ceiling is not None else 0

        self.cleaning_stats['price_bounds'] = {
            'rows_before': rows_before,
            'rows_after': df_bounded.height,
            'dropped_below_floor': int(n_below),
            'dropped_above_ceiling': int(n_above),
        }

        self._log(f"   Price floor  : {self.price_floor} € — dropped {int(n_below):,} rows")
        self._log(f"   Price ceiling: {self.price_ceiling} € — dropped {int(n_above):,} rows")
        self._log(f"   Remaining    : {df_bounded.height:,} rows")

        return df_bounded

    def _filter_rare_brands(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop brands with fewer than min_brand_count observations.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with brand column
            
        Returns
        -------
        pl.DataFrame
            DataFrame with rare brands removed
        """
        self._log(f"\n5️⃣ Dropping rare brands (<{self.min_brand_count} cars)...")
        
        # Calculate brand frequencies
        brand_counts = df.group_by('brand').len().rename({'len': 'count'})
        
        # Identify valid and rare brands
        valid_brands = brand_counts.filter(pl.col('count') >= self.min_brand_count)['brand'].to_list()
        rare_brands = brand_counts.filter(pl.col('count') < self.min_brand_count)
        
        n_dropped_brands = rare_brands.height
        n_dropped_cars = rare_brands['count'].sum() if n_dropped_brands > 0 else 0
        
        # Filter dataset
        df_filtered = df.filter(pl.col('brand').is_in(valid_brands))
        
        self.cleaning_stats['rare_brand_removal'] = {
            'brands_dropped': n_dropped_brands,
            'cars_dropped': n_dropped_cars,
            'brands_remaining': len(valid_brands)
        }
        
        self._log(f"   Dropped {n_dropped_cars:,} cars from {n_dropped_brands} rare brands (< {self.min_brand_count} observations)")
        self._log(f"   Remaining: {len(valid_brands)} brands")
        
        return df_filtered
    
    def _remove_antique_cars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove cars older than the minimum year threshold.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with year column
            
        Returns
        -------
        pl.DataFrame
            DataFrame with antique cars removed
        """
        self._log(f"\n2️⃣ Removing antique cars (pre-{self.min_year})...")
        
        rows_before = df.height
        df_modern = df.filter(pl.col('year') >= self.min_year)
        rows_after = df_modern.height
        
        self.cleaning_stats['antique_removal'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after
        }
        
        self._log(f"   Removed {rows_before - rows_after:,} antique cars")
        
        return df_modern
    
    def _remove_autre_entries(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove entries with 'autre' (other/unknown) in brand or model.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with brand and model columns
            
        Returns
        -------
        pl.DataFrame
            DataFrame with 'autre' entries removed
        """
        self._log("\n3️⃣ Removing 'autre' entries...")
        
        rows_before = df.height
        df_no_autre = df.filter(
            (pl.col('model').str.to_lowercase() != 'autre') &
            (pl.col('brand').str.to_lowercase() != 'autre')
        )
        rows_after = df_no_autre.height
        
        self.cleaning_stats['autre_removal'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after
        }
        
        self._log(f"   Removed {rows_before - rows_after:,} 'autre' entries")
        
        return df_no_autre
    
    def _clean_horsepower(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean horsepower column: hard bounds, IQR per brand, drop missing.
        
        Assumes 'horsepower' column already exists from load_car_data().
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with horsepower column
            
        Returns
        -------
        pl.DataFrame
            DataFrame with cleaned horsepower
        """
        self._log("\n4️⃣ Cleaning horsepower...")
        
        rows_initial = df.height
        
        # Step 1: Hard bounds (50 - 1000 HP)
        n_low = (df['horsepower'] < 50).sum()
        n_high = (df['horsepower'] > 1000).sum()
        df = df.filter(
            (df['horsepower'] >= 50) & (df['horsepower'] <= 1000)
        )
        
        # Step 2: IQR outlier removal per brand
        n_before_iqr = df.height
        
        # Calculate per-brand IQR boundaries
        hp_bounds = df.group_by('brand').agg([
            pl.col('horsepower').quantile(0.25).alias('q1_hp'),
            pl.col('horsepower').quantile(0.75).alias('q3_hp'),
        ]).with_columns([
            (pl.col('q3_hp') - pl.col('q1_hp')).alias('iqr_hp')
        ]).with_columns([
            (pl.col('q1_hp') - self.hp_iqr_multiplier * pl.col('iqr_hp')).alias('lower_bound_hp'),
            (pl.col('q3_hp') + self.hp_iqr_multiplier * pl.col('iqr_hp')).alias('upper_bound_hp')
        ])
        
        # Join and filter
        df = df.join(hp_bounds, on='brand', how='left')
        df = df.filter(
            (pl.col('horsepower') >= pl.col('lower_bound_hp')) &
            (pl.col('horsepower') <= pl.col('upper_bound_hp'))
        ).drop(['q1_hp', 'q3_hp', 'iqr_hp', 'lower_bound_hp', 'upper_bound_hp'])
        
        n_outliers = n_before_iqr - df.height
        
        # Step 3: Drop missing HP
        n_missing = df['horsepower'].is_null().sum()
        df = df.filter(pl.col('horsepower').is_not_null())
        
        self.cleaning_stats['horsepower_cleaning'] = {
            'rows_before': rows_initial,
            'rows_after': df.height,
            'dropped_low': n_low,
            'dropped_high': n_high,
            'dropped_outliers': n_outliers,
            'dropped_missing': n_missing,
            'mean_hp': df['horsepower'].mean(),
            'median_hp': df['horsepower'].median()
        }
        
        self._log(f"   HP cleaning: dropped {n_low} cars <50HP, {n_high} cars >1000HP, {n_outliers} outliers (IQR per brand), {n_missing} missing HP")
        self._log(f"   Remaining dataset - Mean HP: {df['horsepower'].mean():.1f}, Median HP: {df['horsepower'].median():.1f}")
        
        return df
    
    def _remove_outliers_iqr(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove outliers using per-(brand, year) IQR method for log-price and km.

        For each (brand, year) group:
        - n >= iqr_large_group_threshold : IQR computed directly on that group.
        - n < iqr_large_group_threshold  : pool the group with year-1 and year+1 rows
          of the same brand, then compute IQR on the wider pool.
        - pool < iqr_min_group_size      : group is dropped entirely (too sparse to
          compute meaningful bounds).  A line is printed for each dropped group.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with price, km, brand, year, model, energie, horsepower

        Returns
        -------
        pl.DataFrame
            DataFrame with outliers and sparse-group rows removed
        """
        self._log(
            f"\n6️⃣ Removing price/km outliers "
            f"(brand+year IQR {self.price_iqr_multiplier}×; "
            f"iqr_large_group≥{self.iqr_large_group_threshold}; "
            f"iqr_min_pool={self.iqr_min_group_size})..."
        )

        # ── Prepare: add log1p price, cast year, drop invalid rows ───────────
        df_prepared = (
            df
            .with_columns([
                (pl.col('price') + 1).log().alias('log_price'),
                pl.col('year').cast(pl.Int32).alias('year'),
            ])
            .filter(
                pl.col('price').is_not_null()
                & pl.col('km').is_not_null()
                & pl.col('year').is_not_null()
                & (pl.col('price') > 0)
                & (pl.col('km') >= 0)
            )
        )

        rows_before = df_prepared.height

        # ── Group sizes ───────────────────────────────────────────────────────
        group_sizes = (
            df_prepared
            .group_by(['brand', 'year'])
            .agg(pl.len().alias('n_group'))
        )

        large_keys = (
            group_sizes
            .filter(pl.col('n_group') >= self.iqr_large_group_threshold)
            .select(['brand', 'year'])
        )
        small_keys = (
            group_sizes
            .filter(pl.col('n_group') < self.iqr_large_group_threshold)
            .select(['brand', 'year'])
            .to_pandas()
        )

        # ── Helper: compute IQR bounds for a pandas Series ───────────────────
        def _iqr_bounds(series, mult):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            return q1 - mult * iqr, q3 + mult * iqr

        # ── Large groups: vectorised Polars ───────────────────────────────────
        large_bounds_pl = (
            df_prepared
            .join(large_keys, on=['brand', 'year'], how='inner')
            .group_by(['brand', 'year'])
            .agg([
                pl.col('log_price').quantile(0.25).alias('q1_lp'),
                pl.col('log_price').quantile(0.75).alias('q3_lp'),
                pl.col('km').quantile(0.25).alias('q1_km'),
                pl.col('km').quantile(0.75).alias('q3_km'),
            ])
            .with_columns([
                (pl.col('q3_lp') - pl.col('q1_lp')).alias('iqr_lp'),
                (pl.col('q3_km') - pl.col('q1_km')).alias('iqr_km'),
            ])
            .with_columns([
                (pl.col('q1_lp') - self.price_iqr_multiplier * pl.col('iqr_lp')).alias('lo_lp'),
                (pl.col('q3_lp') + self.price_iqr_multiplier * pl.col('iqr_lp')).alias('hi_lp'),
                (pl.col('q1_km') - self.km_iqr_multiplier * pl.col('iqr_km')).alias('lo_km'),
                (pl.col('q3_km') + self.km_iqr_multiplier * pl.col('iqr_km')).alias('hi_km'),
            ])
            .select(['brand', 'year', 'lo_lp', 'hi_lp', 'lo_km', 'hi_km'])
        )

        # ── Small groups: iterate, pool year±1, compute IQR in pandas ────────
        df_pd = df_prepared.select(['brand', 'year', 'log_price', 'km']).to_pandas()
        n_dropped_tiny = 0
        small_rows: list[dict] = []

        for _, row in small_keys.iterrows():
            brand = row['brand']
            year  = int(row['year'])
            pool  = df_pd[
                (df_pd['brand'] == brand)
                & (df_pd['year'] >= year - 1)
                & (df_pd['year'] <= year + 1)
            ]
            if len(pool) < self.iqr_min_group_size:
                n_dropped_tiny += (
                    df_pd[(df_pd['brand'] == brand) & (df_pd['year'] == year)].shape[0]
                )
                if self.verbose:
                    pool_size = len(pool)
                    print(
                        f"   brand={brand!r:20s}  year={year}  "
                        f"pool_size={pool_size}  → DROPPED (below iqr_min_group_size={self.iqr_min_group_size})"
                    )
                continue

            lo_lp, hi_lp = _iqr_bounds(pool['log_price'], self.price_iqr_multiplier)
            lo_km, hi_km = _iqr_bounds(pool['km'],        self.km_iqr_multiplier)
            small_rows.append({
                'brand': brand,
                'year':  year,
                'lo_lp': float(lo_lp), 'hi_lp': float(hi_lp),
                'lo_km': float(lo_km), 'hi_km': float(hi_km),
            })

        small_bounds_pl = pl.DataFrame(
            small_rows,
            schema={
                'brand': pl.Utf8,
                'year':  pl.Int32,
                'lo_lp': pl.Float64,
                'hi_lp': pl.Float64,
                'lo_km': pl.Float64,
                'hi_km': pl.Float64,
            },
        ) if small_rows else pl.DataFrame(
            schema={
                'brand': pl.Utf8,
                'year':  pl.Int32,
                'lo_lp': pl.Float64,
                'hi_lp': pl.Float64,
                'lo_km': pl.Float64,
                'hi_km': pl.Float64,
            }
        )

        # ── Combine bounds and apply filter ───────────────────────────────────
        all_bounds = pl.concat([large_bounds_pl, small_bounds_pl])

        df_with_bounds = df_prepared.join(all_bounds, on=['brand', 'year'], how='left')

        df_clean = (
            df_with_bounds
            .filter(
                pl.col('lo_lp').is_not_null()          # drops tiny groups
                & (pl.col('log_price') >= pl.col('lo_lp'))
                & (pl.col('log_price') <= pl.col('hi_lp'))
                & (pl.col('km') >= pl.col('lo_km'))
                & (pl.col('km') <= pl.col('hi_km'))
            )
            .select(['price', 'year', 'km', 'brand', 'model', 'energie', 'horsepower'])
        )

        rows_after   = df_clean.height
        n_iqr_removed = rows_before - rows_after - n_dropped_tiny

        self.cleaning_stats['outlier_removal'] = {
            'rows_before':           rows_before,
            'rows_after':            rows_after,
            'rows_removed':          rows_before - rows_after,
            'pct_removed':           ((rows_before - rows_after) / rows_before) * 100,
            'n_dropped_iqr':         n_iqr_removed,
            'n_dropped_small_groups': n_dropped_tiny,
        }

        self._log(f"   Before         : {rows_before:,} rows")
        self._log(f"   Removed by IQR : {n_iqr_removed:,}")
        self._log(f"   Removed (tiny groups < {self.iqr_min_group_size}): {n_dropped_tiny:,}")
        self._log(f"   After          : {rows_after:,} rows  ({((rows_before - rows_after) / rows_before) * 100:.1f}% removed total)")

        return df_clean
    
    def get_cleaning_summary(self) -> Dict:
        """
        Get a summary of all cleaning operations performed.
        
        Returns
        -------
        dict
            Dictionary with statistics for each cleaning step
        """
        return self.cleaning_stats
    
    def engineer_features(self, df: pl.DataFrame, current_year: int = 2025) -> pl.DataFrame:
        """
        Apply feature engineering to cleaned data.
        
        TO BE IMPLEMENTED: This will include age calculations, price ratios, etc.
        
        Parameters
        ----------
        df : pl.DataFrame
            Cleaned DataFrame
        current_year : int, default=2025
            Current year for age calculations
            
        Returns
        -------
        pl.DataFrame
            DataFrame with engineered features
        """
        self._log("\n🔧 Feature engineering...")
        self._log("⚠️  Feature engineering not yet implemented - returning cleaned data as-is")
        
        # TODO: Implement feature engineering
        # - car_age
        # - price_per_year
        # - km_per_year
        # - brand/model aggregations
        # - categorical encodings
        
        return df
    
    def process(
        self, 
        df: pl.DataFrame, 
        include_feature_engineering: bool = False
    ) -> pl.DataFrame:
        """
        Complete data processing pipeline.
        
        Parameters
        ----------
        df : pl.DataFrame
            Raw DataFrame
        include_feature_engineering : bool, default=False
            Whether to apply feature engineering after cleaning
            
        Returns
        -------
        pl.DataFrame
            Processed DataFrame
        """
        # Clean data
        df_clean = self.clean_data(df)
        
        # Apply feature engineering if requested
        if include_feature_engineering:
            df_clean = self.engineer_features(df_clean)
        
        return df_clean


def load_car_data(data_dir: Path, infer_schema_length: int = 0) -> pl.DataFrame:
    """
    Load car data from CSV files in a directory.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing CSV files
    infer_schema_length : int, default=0
        Number of rows to use for schema inference (0 = all rows)
        
    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame from all CSV files
    """
    #csv_files = list(data_dir.glob("*.csv"))
    #
    #if not csv_files:
    #    raise FileNotFoundError(f"No CSV files found in {data_dir}")
    #
    #print(f"📂 Found {len(csv_files)} CSV file(s)")
    #
    #dataframes = []
    #total_rows = 0
    #
    #for file_path in csv_files:
    #    df = pl.read_csv(file_path, infer_schema_length=infer_schema_length)
    #    dataframes.append(df)
    #    rows = df.height
    #    total_rows += rows
    #    print(f"   {file_path.name}: {rows:,} rows × {df.width} columns")
    #
    ## Concatenate all dataframes
    #df_combined = pl.concat(dataframes, how="vertical")
    #print(f"✅ Total: {df_combined.height:,} rows × {df_combined.width} columns\n")
    #
    #return df_combined

    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))

    dataframes = {}
    total_rows = 0

    for file_path in csv_files:

        df = pl.read_csv(
            file_path,
            infer_schema_length=0,
            #encoding="utf8",
        )

        dataframes[file_path.stem] = df

    df_combined = pl.concat(dataframes.values(), how="vertical")
    
    # Parse horsepower from puissance_din column (format: "150 Ch" → 150.0)
    print("📊 Parsing horsepower from puissance_din column...")
    df_combined = df_combined.with_columns(
        pl.col('puissance_din')
        .str.replace(' Ch', '')
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias('horsepower')
    )
    
    print(f"✅ Loaded {df_combined.height:,} rows with horsepower parsed")
    print(f"   Note: 'energie' column contains fuel type (kept as-is)")
    print(df_combined.shape)

    return df_combined



# Convenience function for quick processing
def clean_car_data(
    df: pl.DataFrame,
    min_brand_count: int = 400,
    price_iqr_multiplier: float = 1.5,
    km_iqr_multiplier: float = 1.5,
    hp_iqr_multiplier: float = 1.5,
    min_year: int = 1990,
    price_floor: Optional[float] = None,
    price_ceiling: Optional[float] = None,
    iqr_large_group_threshold: int = 50,
    iqr_min_group_size: int = 10,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Quick function to clean car data with default parameters.
    
    Parameters
    ----------
    df : pl.DataFrame
        Raw DataFrame (must have 'horsepower' column from load_car_data)
    min_brand_count : int, default=400
        Minimum cars per brand to keep
    price_iqr_multiplier : float, default=1.5
        IQR multiplier for price
    km_iqr_multiplier : float, default=1.5
        IQR multiplier for kilometers
    hp_iqr_multiplier : float, default=1.5
        IQR multiplier for horsepower
    min_year : int, default=1990
        Minimum car year
    price_floor : float or None, default=None
        Hard lower bound on price in €.  None = not applied.
    price_ceiling : float or None, default=None
        Hard upper bound on price in €.  None = not applied.
    iqr_large_group_threshold : int, default=50
        (brand, year) groups with at least this many cars use direct IQR
    iqr_min_group_size : int, default=10
        Minimum pool size (after year±1 expansion) to compute IQR bounds;
        groups below this threshold are dropped entirely
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame
    """
    processor = CarDataProcessor(
        min_brand_count=min_brand_count,
        price_iqr_multiplier=price_iqr_multiplier,
        km_iqr_multiplier=km_iqr_multiplier,
        hp_iqr_multiplier=hp_iqr_multiplier,
        min_year=min_year,
        price_floor=price_floor,
        price_ceiling=price_ceiling,
        iqr_large_group_threshold=iqr_large_group_threshold,
        iqr_min_group_size=iqr_min_group_size,
        verbose=verbose
    )
    
    return processor.clean_data(df)
