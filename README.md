# Statistics Calculator

A Python library for statistical analysis of numerical data with support for percentiles, reference ranges, autocorrelated time series, and data preprocessing.

## Features

- **Percentile Analysis**: Calculate percentiles and interquartile ranges
- **Reference Range Analysis**: Analyze data distribution across specified ranges and zones
- **Autocorrelated Time Series**: Statistical analysis accounting for temporal dependencies
- **Data Preprocessing**: Data cleaning and intelligent downsampling

## Requirements

- Python 3.10 - 3.12
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- lttb >= 0.3.2 (optional, for downsampling)

## Installation

```bash
git clone <your-repository-url>
cd stats-calculator
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

```python
import numpy as np
from stats_calculator import (
    calculate_percentiles,
    analyze_range,
    autocorrelated_mean
)

# Sample data
data = np.random.normal(100, 15, 200).tolist()

# Calculate percentiles
percentiles = calculate_percentiles(data, [25, 50, 75, 95])
print(percentiles["percentiles"]["p50"])  # Median

# Analyze reference range
range_analysis = analyze_range(data, 80, 120)
print(f"In range: {range_analysis['in_range']['percentage']:.1f}%")

# Autocorrelated mean
mean_result = autocorrelated_mean(data, 0.95)
print(f"Mean: {mean_result['value']:.1f}")
```

## Available Functions

### General Statistics

**`calculate_percentiles(data, percentiles=[25, 50, 75])`**

- Calculate specified percentiles from data
- Returns: Dictionary with percentiles and sample size

**`calculate_iqr(data)`**

- Calculate Q1, median, Q3, IQR, and outlier bounds
- Returns: Dictionary with quartile statistics

### Reference Range Analysis

**`analyze_range(data, lower_bound, upper_bound)`**

- Analyze values relative to a reference range
- Returns: Counts and percentages for below/within/above range

**`analyze_zones(data, zones, allow_overlaps=False)`**

- Analyze data distribution across multiple zones
- Zones format: `[("name", lower, upper), ...]`
- Returns: Count and percentage for each zone

**`analyze_multiple_ranges(data, ranges)`**

- Compare data against multiple reference ranges
- Ranges format: `{"name": (lower, upper), ...}`
- Returns: Analysis for each range

**`get_outliers(data, lower_bound, upper_bound)`**

- Extract outlier values outside specified bounds
- Returns: Lists of outlier values

### Autocorrelated Time Series

**`autocorrelated_mean(data, confidence_level=0.95)`**

- Mean with autocorrelation-adjusted confidence intervals
- Accounts for temporal dependencies in time series data
- Returns: Mean, CI bounds, autocorrelation coefficient, effective sample size

**`autocorrelated_median(data, confidence_level=0.95)`**

- Median using block bootstrap for autocorrelated data
- Returns: Median, CI bounds, method details

**`autocorrelated_std(data, confidence_level=0.95)`**

- Standard deviation with confidence intervals and sigma bands
- Returns: Std dev, CI bounds, sigma bands (1σ, 2σ, 3σ)

### Data Preprocessing

**`clean_data(data)`**

- Remove NaN values and convert to numpy array
- Accepts lists, numpy arrays, pandas Series

**`downsample_data(data, timestamps=None, target_points=100)`**

- Reduce dataset size using LTTB algorithm
- Preserves data structure while compressing
- Optional timestamp handling for time series

## Usage Examples

### Basic Statistics

```python
from stats_calculator import calculate_percentiles, calculate_iqr

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Percentiles
result = calculate_percentiles(data, [10, 50, 90])
print(result["percentiles"])

# IQR with outlier detection
iqr_result = calculate_iqr(data)
print(f"IQR: {iqr_result['iqr']}")
print(f"Outlier bounds: {iqr_result['outlier_bounds']}")
```

### Range Analysis

```python
from stats_calculator import analyze_range, analyze_zones

data = list(range(1, 101))  # 1 to 100

# Single range
range_result = analyze_range(data, 25, 75)
print(f"In range: {range_result['in_range']['count']} values")

# Multiple zones
zones = [("Low", 1, 33), ("Mid", 34, 66), ("High", 67, 100)]
zone_result = analyze_zones(data, zones)
for zone, stats in zone_result["zones"].items():
    print(f"{zone}: {stats['percentage']:.1f}%")
```

### Time Series Analysis

```python
from stats_calculator import autocorrelated_mean, autocorrelated_std

# Generate autocorrelated data
data = [50]
for i in range(1, 100):
    data.append(0.7 * data[i-1] + 0.3 * 50 + np.random.normal(0, 2))

# Analyze with autocorrelation adjustment
mean_result = autocorrelated_mean(data, 0.95)
std_result = autocorrelated_std(data, 0.95)

print(f"Mean: {mean_result['value']:.1f}")
print(f"Autocorrelation: {mean_result['lag1_autocorrelation']:.3f}")
print(f"Effective N: {mean_result['effective_n']} (from {mean_result['n_observations']})")
```

### Data Preprocessing

```python
from stats_calculator import clean_data, downsample_data
import pandas as pd

# Clean data with NaN values
messy_data = [1, 2, np.nan, 4, 5, np.nan, 7]
clean_array = clean_data(messy_data)

# Downsample large dataset
large_data = list(range(1000))
timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')

downsampled = downsample_data(large_data, timestamps, target_points=50)
print(f"Reduced from {downsampled['original_points']} to {downsampled['actual_points']} points")
```

## Error Handling

All functions return error dictionaries for invalid inputs:

```python
result = calculate_percentiles([], [25, 50, 75])
if "error" in result:
    print(f"Error: {result['error']}")
```

## Data Types

- **Input**: Lists, numpy arrays, pandas Series
- **Output**: JSON-serializable dictionaries
- **Missing Data**: Automatically handled with `clean_data()`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=stats_calculator tests/

# Run specific module tests
pytest tests/test_general_stats.py -v
pytest tests/test_reference_ranges.py -v
pytest tests/test_autocorrelated_timeseries.py -v
pytest tests/test_utils.py -v
```

## Project Structure

```
stats_calculator/
├── __init__.py                    # Package imports and aliases
├── general_stats.py              # Percentiles and IQR calculations
├── reference_ranges.py           # Range and zone analysis functions
├── autocorrelated_timeseries.py  # Time series with autocorrelation
└── utils.py                      # Data cleaning and downsampling

tests/
├── __init__.py
├── test_general_stats.py         # Tests for percentile functions
├── test_reference_ranges.py      # Tests for range analysis
├── test_autocorrelated_timeseries.py  # Tests for time series functions
└── test_utils.py                 # Tests for utility functions

requirements.txt                   # Production dependencies
requirements-dev.txt              # Development dependencies
README.md                         # This file
```

## Dependencies

- **numpy**: Array operations and mathematical functions
- **pandas**: Data structures and time series handling
- **scipy**: Statistical distributions for confidence intervals
- **lttb**: Downsampling algorithm (optional, graceful fallback)
