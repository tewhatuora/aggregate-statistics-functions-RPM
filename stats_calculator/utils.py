"""
Utility functions for data processing and manipulation.

Core utilities for data cleaning, validation, and downsampling.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from lttb import downsample

    LTTB_AVAILABLE = True
except ImportError:
    LTTB_AVAILABLE = False


def clean_data(data: Union[np.ndarray, pd.Series, List]) -> np.ndarray:
    """Convert input to clean numpy array, removing NaN values."""
    if isinstance(data, pd.Series):
        return data.dropna().values
    elif isinstance(data, list):
        data = np.array(data)
    else:
        data = np.array(data)

    return data[~np.isnan(data)]


def downsample_data(
    data: Union[List, np.ndarray, pd.Series],
    timestamps: Optional[Union[List, pd.Series]] = None,
    target_points: int = 100,
) -> Dict:
    """
    Downsample data using LTTB (Largest Triangle Three Buckets) algorithm.

    Args:
        data: Numerical data
        timestamps: Optional timestamps (if None, uses indices)
        target_points: Target number of points after downsampling

    Returns:
        Dictionary with downsampled data and timestamps
    """
    if not LTTB_AVAILABLE:
        return {"error": "lttb library not installed. Install with: pip install lttb"}

    if not isinstance(target_points, int) or target_points <= 0:
        return {
            "error": f"target_points must be a positive integer, got {target_points}"
        }

    # Clean the data and track valid indices
    if isinstance(data, pd.Series):
        valid_mask = data.notna()
        clean_array = data[valid_mask].values
        original_indices = data.index[valid_mask]
    elif isinstance(data, list):
        clean_array = np.array(data)
        valid_mask = ~np.isnan(clean_array)
        clean_array = clean_array[valid_mask]
        original_indices = np.arange(len(data))[valid_mask]
    else:
        clean_array = np.array(data)
        valid_mask = ~np.isnan(clean_array)
        clean_array = clean_array[valid_mask]
        original_indices = np.arange(len(data))[valid_mask]

    if len(clean_array) == 0:
        return {"error": "No valid data provided"}

    # Handle timestamps
    if timestamps is not None:
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)

        clean_timestamps = timestamps[valid_mask].reset_index(drop=True)

        if pd.api.types.is_datetime64_any_dtype(clean_timestamps):
            numeric_timestamps = clean_timestamps.astype("int64") // 10**9
        else:
            numeric_timestamps = clean_timestamps.astype(float)

        xy_data = np.column_stack([numeric_timestamps, clean_array])
    else:
        clean_timestamps = None
        xy_data = np.column_stack([original_indices, clean_array])

    # Check if downsampling is needed
    if len(clean_array) <= target_points:
        if clean_timestamps is not None:
            result_timestamps = clean_timestamps.tolist()
        else:
            result_timestamps = original_indices.tolist()

        return {
            "warning": f"Data size ({len(clean_array)}) <= target points ({target_points})",
            "original_points": len(clean_array),
            "target_points": target_points,
            "downsampled_data": clean_array.tolist(),
            "downsampled_timestamps": result_timestamps,
        }

    # Perform LTTB downsampling
    try:
        downsampled_xy = downsample(xy_data, target_points)

        downsampled_x = downsampled_xy[:, 0]
        downsampled_y = downsampled_xy[:, 1]

        # Convert timestamps back to original format
        if clean_timestamps is not None and pd.api.types.is_datetime64_any_dtype(
            clean_timestamps
        ):
            downsampled_timestamps = pd.to_datetime(downsampled_x, unit="s").tolist()
        else:
            downsampled_timestamps = downsampled_x.tolist()

        return {
            "original_points": len(clean_array),
            "target_points": target_points,
            "actual_points": len(downsampled_y),
            "downsampled_data": downsampled_y.tolist(),
            "downsampled_timestamps": downsampled_timestamps,
        }

    except Exception as e:
        return {"error": f"Downsampling failed: {str(e)}"}
