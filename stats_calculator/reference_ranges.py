"""
Reference range analysis functions for numerical data.

Analyze how data points fall within or outside specified ranges,
useful for clinical reference ranges, normal values, and threshold analysis.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from .utils import clean_data


def analyze_range(
    data: Union[List, np.ndarray, pd.Series], lower_bound: float, upper_bound: float
) -> Dict:
    """
    Analyze how values fall relative to a reference range.

    Args:
        data: Numerical data
        lower_bound: Lower boundary of the range
        upper_bound: Upper boundary of the range

    Returns:
        Dictionary with range statistics
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) == 0:
        return {"error": "No valid data provided"}

    if lower_bound >= upper_bound:
        return {
            "error": f"Lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"
        }

    n = len(cleaned_data)

    # Categorize data
    below_mask = cleaned_data < lower_bound
    above_mask = cleaned_data > upper_bound
    in_range_mask = ~below_mask & ~above_mask

    below_count = int(np.sum(below_mask))
    above_count = int(np.sum(above_mask))
    in_range_count = int(np.sum(in_range_mask))

    # Calculate statistics for out-of-range values
    below_stats = None
    if below_count > 0:
        below_values = cleaned_data[below_mask]
        below_stats = {
            "count": below_count,
            "percentage": round(below_count / n * 100, 1),
            "most_extreme": float(np.min(below_values)),
            "average_deviation": round(float(lower_bound - np.mean(below_values)), 2),
        }

    above_stats = None
    if above_count > 0:
        above_values = cleaned_data[above_mask]
        above_stats = {
            "count": above_count,
            "percentage": round(above_count / n * 100, 1),
            "most_extreme": float(np.max(above_values)),
            "average_deviation": round(float(np.mean(above_values) - upper_bound), 2),
        }

    return {
        "range_bounds": {"lower": lower_bound, "upper": upper_bound},
        "total_observations": n,
        "in_range": {
            "count": in_range_count,
            "percentage": round(in_range_count / n * 100, 1),
        },
        "below_range": below_stats,
        "above_range": above_stats,
    }


def analyze_zones(
    data: Union[List, np.ndarray, pd.Series],
    zones: List[Tuple[str, float, float]],
    allow_overlaps: bool = False,
) -> Dict:
    """
    Analyze data distribution across multiple zones.

    Args:
        data: Numerical data
        zones: List of (zone_name, lower_bound, upper_bound) tuples
        allow_overlaps: If False, check for and report overlapping zones

    Returns:
        Dictionary with zone statistics
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) == 0:
        return {"error": "No valid data provided"}

    # Validate zones
    for zone_name, lower, upper in zones:
        if lower > upper:
            return {
                "error": f"Invalid zone '{zone_name}': lower ({lower}) > upper ({upper})"
            }

    # Check for overlaps if not allowed
    if not allow_overlaps:
        overlaps = []
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                name1, lower1, upper1 = zones[i]
                name2, lower2, upper2 = zones[j]
                if max(lower1, lower2) <= min(upper1, upper2):
                    overlaps.append((name1, name2))

        if overlaps:
            return {
                "error": f"Overlapping zones detected: {overlaps}. Set allow_overlaps=True to proceed."
            }

    n = len(cleaned_data)
    results = {"total_observations": n, "zones": {}}

    for zone_name, lower, upper in zones:
        # Use closed intervals [lower, upper]
        in_zone = (cleaned_data >= lower) & (cleaned_data <= upper)
        count = int(np.sum(in_zone))

        results["zones"][zone_name] = {
            "bounds": {"lower": lower, "upper": upper},
            "count": count,
            "percentage": round(count / n * 100, 1),
        }

    return results


def analyze_multiple_ranges(
    data: Union[List, np.ndarray, pd.Series], ranges: Dict[str, Tuple[float, float]]
) -> Dict:
    """
    Analyze multiple reference ranges for the same dataset.

    Args:
        data: Numerical data
        ranges: Dictionary of {range_name: (lower_bound, upper_bound)}

    Returns:
        Dictionary with analysis for each range
    """
    results = {}

    for range_name, (lower, upper) in ranges.items():
        results[range_name] = analyze_range(data, lower, upper)

    return results


def get_outliers(
    data: Union[List, np.ndarray, pd.Series], lower_bound: float, upper_bound: float
) -> Dict:
    """
    Extract actual outlier values from data.

    Args:
        data: Numerical data
        lower_bound: Lower boundary
        upper_bound: Upper boundary

    Returns:
        Dictionary with outlier values and indices
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) == 0:
        return {"error": "No valid data provided"}

    below_mask = cleaned_data < lower_bound
    above_mask = cleaned_data > upper_bound

    below_values = cleaned_data[below_mask] if np.any(below_mask) else np.array([])
    above_values = cleaned_data[above_mask] if np.any(above_mask) else np.array([])

    return {
        "below_range": below_values.tolist(),
        "above_range": above_values.tolist(),
        "total_outliers": len(below_values) + len(above_values),
    }
