"""
Stats Calculator - Statistical analysis functions for biometric and time series data.

This package provides functions for:
- General statistics and percentiles
- Reference range analysis 
- Autocorrelated time series analysis
- Data preprocessing and downsampling
"""

# General statistics and percentiles
from .general_stats import calculate_percentiles, calculate_iqr

# Reference range analysis
from .reference_ranges import (
    analyze_range,
    analyze_zones, 
    analyze_multiple_ranges,
    get_outliers
)

# Autocorrelated time series analysis
from .autocorrelated_timeseries import (
    autocorrelated_mean,
    autocorrelated_median,
    autocorrelated_std
)

# Utilities
from .utils import clean_data, downsample_data

# Convenience aliases for commonly used functions
analyze_value_based_range = analyze_range
analyze_band_based_range = analyze_zones
