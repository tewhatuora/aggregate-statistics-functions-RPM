"""Stats Calculator - Simple statistical analysis functions."""

from .core import (
    calculate_general_stats,
    calculate_confidence_interval,
    calculate_percentiles,
    analyze_value_based_range,
    analyze_time_based_range,
    analyze_band_based_range,
    downsample_data,
    analyze_all
)

__version__ = "1.0.0"