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
from src.config import DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH

warnings.filterwarnings('ignore')


class CarDataProcessor:
    """
    Main class for processing car data through cleaning and feature engineering.
    
    Parameters
    ----------
    min_brand_threshold : int, default=50
        Minimum number of cars required to keep a brand
    rare_brand_threshold : int, default=600
        Threshold below which brands are grouped by price tier
    price_iqr_multiplier : float, default=1.5
        IQR multiplier for price outlier detection
    km_iqr_multiplier : float, default=1.5
        IQR multiplier for kilometer outlier detection
    min_year : int, default=1990
        Minimum car year to keep (removes antique cars)
    verbose : bool, default=True
        Whether to print progress messages
    """
    
    def __init__(
        self,
        min_brand_threshold: int = 50,
        rare_brand_threshold: int = 600,
        price_iqr_multiplier: float = 1.5,
        km_iqr_multiplier: float = 1.5,
        min_year: int = 1990,
        verbose: bool = True
    ):
        self.min_brand_threshold = min_brand_threshold
        self.rare_brand_threshold = rare_brand_threshold
        self.price_iqr_multiplier = price_iqr_multiplier
        self.km_iqr_multiplier = km_iqr_multiplier
        self.min_year = min_year
        self.verbose = verbose
        
        # Store statistics for later inspection
        self.cleaning_stats = {}
        
    def _log(self, message: str):
        """Print message if verbose is True."""
        if self.verbose:
            print(message)
    
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
        
        # Step 1: Data type conversion
        df = self._convert_data_types(df)
        
        # Step 2: Brand filtering and grouping
        df = self._filter_and_group_brands(df)
        
        # Step 3: Remove antique cars
        df = self._remove_antique_cars(df)
        
        # Step 4: Remove 'autre' entries
        df = self._remove_autre_entries(df)
        
        # Step 5: IQR outlier detection
        df = self._remove_outliers_iqr(df)
        
        self._log("\n✅ Data cleaning completed!")
        self._log(f"Final dataset: {df.height:,} rows × {df.width} columns")
        
        return df
    
    def _convert_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert string columns to appropriate numeric types.
        
        Parameters
        ----------
        df : pl.DataFrame
            Raw DataFrame
            
        Returns
        -------
        pl.DataFrame
            DataFrame with numeric types
        """
        self._log("\n1️⃣ Converting data types...")
        
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
        
        # Create simplified DataFrame with renamed columns
        df_clean = df.select([
            pl.col('price_numeric').alias('price'),
            pl.col('year_numeric').alias('year'), 
            pl.col('km_numeric').alias('km'),
            pl.col('marque').alias('brand'),
            pl.col('modele').alias('model')
        ]).filter(
            pl.col('price').is_not_null()
        )
        
        rows_before = df.height
        rows_after = df_clean.height
        self.cleaning_stats['type_conversion'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after
        }
        
        self._log(f"   Original: {rows_before:,} rows")
        self._log(f"   After conversion: {rows_after:,} rows")
        self._log(f"   Removed (invalid price): {rows_before - rows_after:,}")
        
        return df_clean
    
    def _filter_and_group_brands(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out very rare brands and group medium-frequency brands by price tier.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with brand column
            
        Returns
        -------
        pl.DataFrame
            DataFrame with filtered and grouped brands
        """
        self._log("\n2️⃣ Filtering and grouping brands...")
        
        # Calculate brand frequencies
        brand_freq = df.group_by('brand').len().sort('len', descending=True)
        
        # Remove brands with very few cars
        brands_to_remove = brand_freq.filter(pl.col('len') < self.min_brand_threshold)['brand'].to_list()
        df_filtered = df.filter(~pl.col('brand').is_in(brands_to_remove))
        
        self.cleaning_stats['brand_removal'] = {
            'brands_removed': len(brands_to_remove),
            'cars_removed': brands_to_remove and brand_freq.filter(
                pl.col('brand').is_in(brands_to_remove)
            )['len'].sum() or 0
        }
        
        self._log(f"   Removed {len(brands_to_remove)} brands with <{self.min_brand_threshold} cars")
        
        # Group rare brands by price tier
        brands_to_group = brand_freq.filter(
            (pl.col('len') >= self.min_brand_threshold) & 
            (pl.col('len') < self.rare_brand_threshold)
        )['brand'].to_list()
        
        if len(brands_to_group) > 0:
            # Calculate price thresholds
            all_prices = df_filtered['price'].to_list()
            low_price_threshold = np.percentile(all_prices, 33)
            high_price_threshold = np.percentile(all_prices, 67)
            
            # Get average price per brand
            brand_avg_prices = df_filtered.filter(
                pl.col('brand').is_in(brands_to_group)
            ).group_by('brand').agg(
                pl.col('price').mean().alias('avg_price')
            )
            
            # Categorize brands
            low_cost_brands = brand_avg_prices.filter(
                pl.col('avg_price') <= low_price_threshold
            )['brand'].to_list()
            
            luxury_brands = brand_avg_prices.filter(
                pl.col('avg_price') >= high_price_threshold
            )['brand'].to_list()
            
            standard_brands = brand_avg_prices.filter(
                (pl.col('avg_price') > low_price_threshold) & 
                (pl.col('avg_price') < high_price_threshold)
            )['brand'].to_list()
            
            # Apply grouping
            df_grouped = df_filtered.with_columns(
                pl.when(pl.col('brand').is_in(low_cost_brands))
                .then(pl.lit('other_low_cost'))
                .when(pl.col('brand').is_in(standard_brands))
                .then(pl.lit('other_standard'))
                .when(pl.col('brand').is_in(luxury_brands))
                .then(pl.lit('other_luxury'))
                .otherwise(pl.col('brand'))
                .alias('brand')
            ).with_columns(
                pl.when(pl.col('brand').str.starts_with('other_'))
                .then(pl.col('brand'))
                .otherwise(pl.col('model'))
                .alias('model')
            )
            
            self._log(f"   Grouped {len(brands_to_group)} brands into price tiers")
        else:
            df_grouped = df_filtered
        
        return df_grouped
    
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
        self._log(f"\n3️⃣ Removing antique cars (pre-{self.min_year})...")
        
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
        self._log("\n4️⃣ Removing 'autre' entries...")
        
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
    
    def _remove_outliers_iqr(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove outliers using per-brand IQR method.
        
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with price, km, and brand columns
            
        Returns
        -------
        pl.DataFrame
            DataFrame with outliers removed
        """
        self._log(f"\n5️⃣ Removing outliers (IQR {self.price_iqr_multiplier}× for price, {self.km_iqr_multiplier}× for km)...")
        
        # Add log-transformed price and filter invalid data
        df_prepared = df.with_columns([
            (pl.col('price') + 1).log().alias('log_price')
        ]).filter(
            (pl.col('price').is_not_null()) & 
            (pl.col('km').is_not_null()) &
            (pl.col('year').is_not_null()) &
            (pl.col('price') > 0) &
            (pl.col('km') >= 0)
        )
        
        # Calculate per-brand IQR boundaries
        bounds = df_prepared.group_by('brand').agg([
            # Log price boundaries
            pl.col('log_price').quantile(0.25).alias('q1_log_price'),
            pl.col('log_price').quantile(0.75).alias('q3_log_price'),
            # Kilometers boundaries  
            pl.col('km').quantile(0.25).alias('q1_km'),
            pl.col('km').quantile(0.75).alias('q3_km'),
            pl.len().alias('brand_count')
        ]).with_columns([
            # Calculate IQR
            (pl.col('q3_log_price') - pl.col('q1_log_price')).alias('iqr_log_price'),
            (pl.col('q3_km') - pl.col('q1_km')).alias('iqr_km')
        ]).with_columns([
            # Calculate boundaries
            (pl.col('q1_log_price') - self.price_iqr_multiplier * pl.col('iqr_log_price')).alias('lower_bound_log_price'),
            (pl.col('q3_log_price') + self.price_iqr_multiplier * pl.col('iqr_log_price')).alias('upper_bound_log_price'),
            (pl.col('q1_km') - self.km_iqr_multiplier * pl.col('iqr_km')).alias('lower_bound_km'),
            (pl.col('q3_km') + self.km_iqr_multiplier * pl.col('iqr_km')).alias('upper_bound_km')
        ])
        
        # Join bounds and filter outliers
        df_with_bounds = df_prepared.join(bounds, on='brand', how='left')
        
        rows_before = df_with_bounds.height
        df_clean = df_with_bounds.filter(
            (pl.col('log_price') >= pl.col('lower_bound_log_price')) &
            (pl.col('log_price') <= pl.col('upper_bound_log_price')) &
            (pl.col('km') >= pl.col('lower_bound_km')) &
            (pl.col('km') <= pl.col('upper_bound_km'))
        ).select(['price', 'year', 'km', 'brand', 'model'])
        
        rows_after = df_clean.height
        
        self.cleaning_stats['outlier_removal'] = {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after,
            'pct_removed': ((rows_before - rows_after) / rows_before) * 100
        }
        
        self._log(f"   Before: {rows_before:,} rows")
        self._log(f"   After: {rows_after:,} rows")
        self._log(f"   Removed: {rows_before - rows_after:,} ({((rows_before - rows_after) / rows_before) * 100:.1f}%)")
        
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
    print(df_combined.shape)

    return df_combined



# Convenience function for quick processing
def clean_car_data(
    df: pl.DataFrame,
    min_brand_threshold: int = 50,
    rare_brand_threshold: int = 600,
    price_iqr_multiplier: float = 1.5,
    km_iqr_multiplier: float = 1.5,
    min_year: int = 1990,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Quick function to clean car data with default parameters.
    
    Parameters
    ----------
    df : pl.DataFrame
        Raw DataFrame
    min_brand_threshold : int, default=50
        Minimum cars per brand
    rare_brand_threshold : int, default=600
        Threshold for brand grouping
    price_iqr_multiplier : float, default=1.5
        IQR multiplier for price
    km_iqr_multiplier : float, default=1.5
        IQR multiplier for kilometers
    min_year : int, default=1990
        Minimum car year
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame
    """
    processor = CarDataProcessor(
        min_brand_threshold=min_brand_threshold,
        rare_brand_threshold=rare_brand_threshold,
        price_iqr_multiplier=price_iqr_multiplier,
        km_iqr_multiplier=km_iqr_multiplier,
        min_year=min_year,
        verbose=verbose
    )
    
    return processor.clean_data(df)
