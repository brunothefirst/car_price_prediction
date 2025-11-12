# Data Processing Module

Comprehensive data processing pipeline for car price prediction using Polars DataFrames.

## Features

### ✅ Currently Implemented

**Data Cleaning Pipeline:**
1. **Data Type Conversion**: Convert string columns to numeric types
2. **Brand Filtering**: Remove brands with insufficient data (<50 cars by default)
3. **Brand Grouping**: Group low-frequency brands by price tier
4. **Antique Car Removal**: Filter out pre-1990 vehicles
5. **Invalid Entry Removal**: Remove 'autre' (unknown) entries
6. **IQR Outlier Detection**: Per-brand outlier removal using IQR method

### 🔄 Future Implementation

**Feature Engineering:**
- Car age calculations
- Price and mileage ratios
- Brand/model aggregations
- Categorical encodings
- Interaction features

## Usage

### Quick Start

```python
from src.data_processing import load_car_data, clean_car_data
from pathlib import Path

# Load raw data
data_dir = Path("data/raw/le_boncoin_13_oct_2025")
df_raw = load_car_data(data_dir)

# Clean data with default parameters
df_clean = clean_car_data(df_raw)

print(f"Cleaned data: {df_clean.height:,} rows")
```

### Using the CarDataProcessor Class

```python
from src.data_processing import CarDataProcessor

# Initialize with custom parameters
processor = CarDataProcessor(
    min_brand_threshold=50,      # Min cars per brand
    rare_brand_threshold=600,    # Threshold for grouping
    price_iqr_multiplier=1.5,    # IQR multiplier for price
    km_iqr_multiplier=1.5,       # IQR multiplier for kilometers
    min_year=1990,               # Minimum car year
    verbose=True                 # Print progress
)

# Clean data
df_clean = processor.clean_data(df_raw)

# Get statistics
stats = processor.get_cleaning_summary()
print(stats)
```

### Custom Parameters

```python
# More lenient cleaning (for luxury brands)
processor_lenient = CarDataProcessor(
    price_iqr_multiplier=1.7,    # Less aggressive outlier removal
    min_year=2000                # Only newer cars
)

# Stricter cleaning
processor_strict = CarDataProcessor(
    min_brand_threshold=100,     # Require more data per brand
    price_iqr_multiplier=1.3,    # More aggressive outlier removal
    rare_brand_threshold=1000    # Higher grouping threshold
)
```

### Integration with Notebooks

**In Feature Engineering Notebook (03_feature_engineering.ipynb):**

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd().parent))

from src.data_processing import load_car_data, CarDataProcessor
from src.config import DATA_PATH

# Load and clean data
data_dir = DATA_PATH / "le_boncoin_13_oct_2025"
df_raw = load_car_data(data_dir)

processor = CarDataProcessor(verbose=True)
df_clean = processor.clean_data(df_raw)

# Now proceed with feature engineering...
```

## Parameters

### CarDataProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_brand_threshold` | int | 50 | Minimum cars required to keep a brand |
| `rare_brand_threshold` | int | 600 | Brands below this are grouped by price tier |
| `price_iqr_multiplier` | float | 1.5 | IQR multiplier for price outlier detection |
| `km_iqr_multiplier` | float | 1.5 | IQR multiplier for kilometer outlier detection |
| `min_year` | int | 1990 | Minimum car year (removes antique cars) |
| `verbose` | bool | True | Print progress messages |

## Cleaning Pipeline Details

### 1. Data Type Conversion
- Converts `price`, `annee_modele` (year), and `kilometrage` (km) to numeric
- Removes rows with null prices
- Renames columns to English: `marque` → `brand`, `modele` → `model`

### 2. Brand Filtering & Grouping

**Removal:**
- Brands with < `min_brand_threshold` cars are removed entirely

**Grouping:**
- Brands with `min_brand_threshold` ≤ count < `rare_brand_threshold` are grouped into:
  - `other_low_cost` (bottom 33% by average price)
  - `other_standard` (middle 33%)
  - `other_luxury` (top 33%)

**Preservation:**
- Brands with ≥ `rare_brand_threshold` cars keep their original names

### 3. Antique Car Removal
- Removes all cars with `year` < `min_year`
- Default: removes pre-1990 vehicles

### 4. Invalid Entry Removal
- Removes rows where `brand` or `model` is `'autre'` (other/unknown)
- Filters out remaining null values in key columns

### 5. IQR Outlier Detection

**Per-Brand Approach:**
- Calculates Q1, Q3, and IQR separately for each brand
- Uses log-transformed prices for better outlier detection
- Standard (non-log) scale for kilometers

**Boundaries:**
- Price: `Q1 - (multiplier × IQR)` to `Q3 + (multiplier × IQR)` (log scale)
- Kilometers: `Q1 - (multiplier × IQR)` to `Q3 + (multiplier × IQR)` (standard scale)

**Why Per-Brand?**
- Different brands have different price distributions
- Luxury brands naturally have higher variance
- Prevents unfair removal of legitimate premium/budget listings

## Cleaning Statistics

Access detailed statistics using `get_cleaning_summary()`:

```python
stats = processor.get_cleaning_summary()

# Returns:
{
    'type_conversion': {
        'rows_before': 250000,
        'rows_after': 248500,
        'rows_removed': 1500
    },
    'brand_removal': {
        'brands_removed': 45,
        'cars_removed': 800
    },
    'antique_removal': {
        'rows_before': 247700,
        'rows_after': 246500,
        'rows_removed': 1200
    },
    'autre_removal': {
        'rows_before': 246500,
        'rows_after': 245000,
        'rows_removed': 1500
    },
    'outlier_removal': {
        'rows_before': 245000,
        'rows_after': 235000,
        'rows_removed': 10000,
        'pct_removed': 4.08
    }
}
```

## Output Schema

**Cleaned DataFrame Columns:**
- `price` (Float64): Car price in euros
- `year` (Float64): Year of manufacture
- `km` (Float64): Mileage in kilometers
- `brand` (String): Brand name (or grouped category)
- `model` (String): Model name

## Examples

See `examples/data_processing_examples.py` for complete usage examples:

```bash
python examples/data_processing_examples.py
```

## Notes

⚠️ **Important Considerations:**

1. **IQR Multiplier**: The default 1.5× multiplier may be too aggressive for luxury brands with naturally high variance. Consider using 1.7-1.9× for premium brands.

2. **Brand Grouping**: Adjust `rare_brand_threshold` based on your dataset size. Larger datasets can support lower thresholds.

3. **Memory**: Polars is memory-efficient, but very large datasets (>1M rows) may benefit from lazy evaluation using `scan_csv()` instead of `read_csv()`.

4. **Future Enhancement**: Brand-specific IQR multipliers can be implemented via Excel configuration file for subject matter expert input.

## Testing

Run the module directly to see a simple test:

```bash
cd src
python data_processing.py
```

## Dependencies

- `polars`: DataFrame operations
- `numpy`: Statistical calculations
- `pathlib`: File path handling

No visualization libraries required for the core module (only used in notebooks).
