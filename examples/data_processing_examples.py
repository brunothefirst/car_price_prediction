"""
Example usage of the data_processing module.

This script demonstrates how to use the CarDataProcessor class
to clean car data.
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_PATH, PROCESSED_DATA_PATH
from src.data_processing import load_car_data, CarDataProcessor, clean_car_data


def example_basic_usage():
    """Example 1: Quick cleaning with default parameters."""
    print("="*70)
    print("EXAMPLE 1: Quick cleaning with default parameters")
    print("="*70)
    
    # Load data
    data_dir = DATA_PATH / "le_boncoin_13_oct_2025"
    df_raw = load_car_data(data_dir)
    
    # Clean data (one-liner)
    df_clean = clean_car_data(df_raw, verbose=True)
    
    print(f"\n✅ Cleaned data: {df_clean.height:,} rows × {df_clean.width} columns")
    print(df_clean.head(3))


def example_custom_parameters():
    """Example 2: Using custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom cleaning parameters")
    print("="*70)
    
    # Load data
    data_dir = DATA_PATH / "le_boncoin_13_oct_2025"
    df_raw = load_car_data(data_dir)
    
    # Initialize processor with custom parameters
    processor = CarDataProcessor(
        min_brand_threshold=100,        # Stricter: require 100+ cars per brand
        rare_brand_threshold=1000,      # Higher threshold for grouping
        price_iqr_multiplier=1.7,       # More lenient for luxury brands
        km_iqr_multiplier=1.5,          # Standard for kilometers
        min_year=2000,                  # Only cars from 2000+
        verbose=True
    )
    
    # Clean data
    df_clean = processor.clean_data(df_raw)
    
    # Get cleaning statistics
    stats = processor.get_cleaning_summary()
    
    print("\n📊 Cleaning Statistics:")
    print("="*70)
    for step, metrics in stats.items():
        print(f"\n{step.upper()}:")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")
    
    return df_clean


def example_with_feature_engineering():
    """Example 3: Full pipeline with feature engineering (when implemented)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Full pipeline (cleaning + feature engineering)")
    print("="*70)
    
    # Load data
    data_dir = DATA_PATH / "le_boncoin_13_oct_2025"
    df_raw = load_car_data(data_dir)
    
    # Initialize processor
    processor = CarDataProcessor(verbose=True)
    
    # Process with feature engineering
    df_processed = processor.process(
        df_raw,
        include_feature_engineering=False  # Set to True when implemented
    )
    
    print(f"\n✅ Processed data: {df_processed.height:,} rows × {df_processed.width} columns")
    
    return df_processed


def example_save_cleaned_data():
    """Example 4: Clean and save data."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Clean and save data")
    print("="*70)
    
    # Load and clean
    data_dir = DATA_PATH / "le_boncoin_13_oct_2025"
    df_raw = load_car_data(data_dir)
    df_clean = clean_car_data(df_raw, verbose=True)
    
    # Save cleaned data
    output_path = PROCESSED_DATA_PATH / "cleaned_car_data.csv"
    df_clean.write_csv(output_path)
    
    print(f"\n💾 Saved cleaned data to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    # Run all examples
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Example 2: Custom parameters
    df_custom = example_custom_parameters()
    
    # Example 3: Full pipeline
    df_processed = example_with_feature_engineering()
    
    # Example 4: Save cleaned data
    example_save_cleaned_data()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
