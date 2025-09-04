from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .utils import clean_data


def calculate_percentiles(
    data: Union[np.ndarray, pd.Series, List], percentiles: List[float] = [25, 50, 75]
) -> Dict:
    """
    Calculate percentiles for time series data.

    Args:
        data: Time series data
        percentiles: List of percentiles (0-100)

    Returns:
        Dictionary with percentile values
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) == 0:
        return {"error": "No valid data provided"}

    # Validate percentiles
    if not all(0 <= p <= 100 for p in percentiles):
        return {"error": "Percentiles must be between 0 and 100"}

    values = np.percentile(cleaned_data, percentiles)
    percentile_dict = {f"p{p}": float(v) for p, v in zip(percentiles, values)}

    return {"percentiles": percentile_dict, "n_observations": len(cleaned_data)}


def calculate_iqr(data: Union[np.ndarray, pd.Series, List]) -> Dict:
    """
    Calculate Inter-Quartile Range (IQR) statistics using the percentiles function.

    Args:
        data: Time series data

    Returns:
        Dictionary with Q1, Q2 (median), Q3, IQR, and range information
    """
    percentile_result = calculate_percentiles(data, [25, 50, 75])

    if "error" in percentile_result:
        return percentile_result

    q1 = percentile_result["percentiles"]["p25"]
    q2 = percentile_result["percentiles"]["p50"]
    q3 = percentile_result["percentiles"]["p75"]

    iqr_value = q3 - q1

    lower_fence = q1 - 1.5 * iqr_value
    upper_fence = q3 + 1.5 * iqr_value

    return {
        "q1": q1,
        "median": q2,
        "q3": q3,
        "iqr": iqr_value,
        "outlier_bounds": {"lower_fence": lower_fence, "upper_fence": upper_fence},
        "n_observations": percentile_result["n_observations"],
    }
