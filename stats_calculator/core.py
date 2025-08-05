import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Union, Optional, Tuple

try:
    from lttb import downsample
except ImportError:
    print("Warning: lttb not installed. Install with: pip install lttb")
    downsample = None

def calculate_general_stats(data: Union[List, np.ndarray]) -> Dict:
    """
    Calculate general descriptive statistics.
    
    Args:
        data: Numerical data
        
    Returns:
        Dictionary with all basic statistics
    """
    clean_data = np.array(data)
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    n = len(clean_data)
    
    q25, q50, q75 = np.percentile(clean_data, [25, 50, 75])
    
    result = {
        "count": int(n),
        "mean": float(np.mean(clean_data)),
        "median": float(q50),
        "min": float(np.min(clean_data)),
        "max": float(np.max(clean_data)),
        "iqr": float(q75 - q25),
        "variance": float(np.var(clean_data, ddof=1)) if n > 1 else 0.0,
        "std_dev": float(np.std(clean_data, ddof=1)) if n > 1 else 0.0,
        "std_error": float(np.std(clean_data, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "skewness": float(stats.skew(clean_data)) if n > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(clean_data)) if n > 3 else 0.0
    }
    
    return result

def calculate_confidence_interval(data: Union[List, np.ndarray], 
                                confidence_level: float = 0.95) -> Dict:
    """
    Calculate confidence interval for the mean.
    
    Args:
        data: Numerical data
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary with confidence interval information
    """
    # Validate confidence level
    if not 0 < confidence_level < 1:
        return {"error": f"Confidence level {confidence_level} must be between 0 and 1 (exclusive)"}
    
    clean_data = np.array(data)
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) <= 1:
        return {"error": "Need at least 2 data points for confidence interval"}
    
    n = len(clean_data)
    mean_val = np.mean(clean_data)
    std_error = np.std(clean_data, ddof=1) / np.sqrt(n)
    
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    margin_of_error = z_score * std_error
    
    return {
        "confidence_level": float(confidence_level),
        "mean": float(mean_val),
        "std_error": float(std_error),
        "z_score": float(z_score),
        "margin_of_error": float(margin_of_error),
        "lower_bound": float(mean_val - margin_of_error),
        "upper_bound": float(mean_val + margin_of_error)
    }

def calculate_percentiles(data: Union[List, np.ndarray], 
                         percentiles: List[float]) -> Dict:
    """
    Calculate specified percentiles.
    
    Args:
        data: Numerical data
        percentiles: List of percentiles to calculate (0-100)
        
    Returns:
        Dictionary mapping percentile to value
    """
    clean_data = np.array(data)
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    percentiles_array = np.array(percentiles)
    if not np.all((0 <= percentiles_array) & (percentiles_array <= 100)):
        invalid_values = percentiles_array[(percentiles_array < 0) | (percentiles_array > 100)]
        return {"error": f"Percentiles {invalid_values.tolist()} must be between 0 and 100 (inclusive)"}
    
    values = np.percentile(clean_data, percentiles)
    
    return {f"p{p}": float(v) for p, v in zip(percentiles, values)}

def analyze_value_based_range(data: Union[List, np.ndarray], 
                             lower_bound: float, 
                             upper_bound: float) -> Dict:
    """
    Analyze values outside the specified range (value-based).
    
    Args:
        data: Numerical data
        lower_bound: Lower boundary of the range
        upper_bound: Upper boundary of the range
        
    Returns:
        Dictionary with below and above range statistics
    """
    clean_data = np.array(data)
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    if lower_bound >= upper_bound:
        return {"error": f"Invalid range: lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"}
    
    n = len(clean_data)
    
    below_values = clean_data[clean_data < lower_bound]
    below_count = len(below_values)
    
    below_max_dev = below_avg_dev = below_min_value = 0.0
    
    if below_count > 0:
        below_deviations = lower_bound - below_values
        below_max_dev = float(np.max(below_deviations))
        below_avg_dev = float(np.mean(below_deviations))
        below_min_value = float(np.min(below_values))
    
    above_values = clean_data[clean_data > upper_bound]
    above_count = len(above_values)
    
    above_max_dev = above_avg_dev = above_max_value = 0.0
    
    if above_count > 0:
        above_deviations = above_values - upper_bound
        above_max_dev = float(np.max(above_deviations))
        above_avg_dev = float(np.mean(above_deviations))
        above_max_value = float(np.max(above_values))
    
    return {
        "range": {"lower": float(lower_bound), "upper": float(upper_bound)},
        "below_range": {
            "count": int(below_count),
            "percentage": float(below_count / n * 100),
            "max_deviation": below_max_dev,
            "avg_deviation": below_avg_dev,
            "min_value": below_min_value
        },
        "above_range": {
            "count": int(above_count),
            "percentage": float(above_count / n * 100),
            "max_deviation": above_max_dev,
            "avg_deviation": above_avg_dev,
            "max_value": above_max_value
        }
    }

def analyze_time_based_range(data: Union[List, np.ndarray],
                           timestamps: Union[List, pd.Series],
                           lower_bound: float,
                           upper_bound: float,
                           duration_unit: str = 'minutes') -> Dict:
    """
    Analyze time-based episodes outside the specified range.
    
    Args:
        data: Numerical data
        timestamps: Timestamps corresponding to data
        lower_bound: Lower boundary of the range
        upper_bound: Upper boundary of the range
        duration_unit: Unit for duration ('seconds', 'minutes', 'hours')
        
    Returns:
        Dictionary with time-based range statistics
    """
    clean_data = np.array(data)
    
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)
    
    valid_mask = ~np.isnan(clean_data)
    clean_data = clean_data[valid_mask]
    timestamps = timestamps[valid_mask].reset_index(drop=True)
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    if len(timestamps) != len(clean_data):
        return {"error": "Timestamps and data must have same length"}
    
    if lower_bound >= upper_bound:
        return {"error": f"Invalid range: lower bound ({lower_bound}) must be less than upper bound ({upper_bound})"}
    
    def find_episodes(mask):
        """Find contiguous episodes where mask is True."""
        episodes = []
        in_episode = False
        episode_start_idx = None
        
        for i, is_out in enumerate(mask):
            if is_out and not in_episode:
                in_episode = True
                episode_start_idx = i
            elif not is_out and in_episode:
                in_episode = False
                start_time = timestamps.iloc[episode_start_idx]
                end_time = timestamps.iloc[i -1]  
                duration = calculate_duration(start_time, end_time, duration_unit)
                episodes.append({
                    'duration': duration,
                    'start_idx': episode_start_idx,
                    'end_idx': i - 1  
                })
        
        if in_episode:
            start_time = timestamps.iloc[episode_start_idx]
            end_time = timestamps.iloc[-1]
            duration = calculate_duration(start_time, end_time, duration_unit)
            episodes.append({
                'duration': duration,
                'start_idx': episode_start_idx,
                'end_idx': len(timestamps) - 1
            })
        
        return episodes
    
    def calculate_duration(start, end, unit):
        """Calculate duration between timestamps."""
        delta = end - start
        total_seconds = delta.total_seconds()
        
        if unit == 'seconds':
            return total_seconds
        elif unit == 'minutes':
            return total_seconds / 60
        elif unit == 'hours':
            return total_seconds / 3600
        else:
            raise ValueError(f"Unsupported duration unit: {unit}")
    
    def calculate_episode_stats(episodes, values, is_below):
        """Calculate statistics for episodes."""
        if not episodes:
            return {
                "episode_count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": 0.0,
                "avg_episode_value": 0.0,
                "most_extreme_value": 0.0
            }
        
        durations = [ep['duration'] for ep in episodes]
        
        if len(values) > 0:
            avg_episode_value = float(np.mean(values))
            if is_below:
                most_extreme_value = float(np.min(values))
            else:
                most_extreme_value = float(np.max(values))
        else:
            avg_episode_value = most_extreme_value = 0.0
        
        return {
            "episode_count": len(episodes),
            "total_duration": float(sum(durations)),
            "avg_duration": float(np.mean(durations)),
            "max_duration": float(max(durations)),
            "min_duration": float(min(durations)),
            "avg_episode_value": avg_episode_value,
            "most_extreme_value": most_extreme_value
        }
    
    below_episodes = find_episodes(clean_data < lower_bound)
    above_episodes = find_episodes(clean_data > upper_bound)  
    
    below_values = clean_data[clean_data < lower_bound]
    above_values = clean_data[clean_data > upper_bound] 
    
    return {
        "range": {"lower": float(lower_bound), "upper": float(upper_bound)},
        "duration_unit": duration_unit,
        "below_range": calculate_episode_stats(below_episodes, below_values, True),
        "above_range": calculate_episode_stats(above_episodes, above_values, False)
    }

def analyze_band_based_range(data: Union[List, np.ndarray], 
                           bands: List[Tuple[str, float, float]]) -> Dict:
    """
    Analyze data distribution across specified bands/zones using closed intervals [lower, upper].
    
    Args:
        data: Numerical data
        bands: List of (band_name, lower_bound, upper_bound) tuples
        
    Returns:
        Dictionary with band statistics, including overlap detection if present
    """
    for band_name, lower, upper in bands:
        if lower > upper:
            return {"error": f"Invalid band '{band_name}': lower bound ({lower}) must not exceed upper bound ({upper})"}
    
    clean_data = np.array(data)
    clean_data = clean_data[~np.isnan(clean_data)]
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    overlap_pairs = []
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            name1, lower1, upper1 = bands[i]
            name2, lower2, upper2 = bands[j]
            if max(lower1, lower2) <= min(upper1, upper2):
                overlap_pairs.append((name1, name2))
    
    n = len(clean_data)
    results = {}
    
    for band_name, lower, upper in bands:
        in_band = (clean_data >= lower) & (clean_data <= upper)
        count = int(np.sum(in_band))
        percentage = float(count / n * 100)
        
        results[band_name] = {
            "band": band_name,
            "range": {"lower": float(lower), "upper": float(upper)},
            "count": count,
            "percentage": percentage
        }
    
    if overlap_pairs:
        results["overlap_warning"] = f"Overlapping bands detected: {overlap_pairs}. Values in overlapping regions counted in multiple bands."
    
    return results

def downsample_data(data: Union[List, np.ndarray], 
                   timestamps: Optional[Union[List, pd.Series]] = None,
                   target_points: int = 100) -> Dict:
    """
    Downsample data using LTTB (Largest Triangle Three Buckets) algorithm.
    
    Args:
        data: Numerical data
        timestamps: Optional timestamps (if None, uses indices)
        target_points: Target number of points after downsampling
        
    Returns:
        Dictionary with downsampled data and timestamps
    """
    if downsample is None:
        return {"error": "lttb library not installed. Install with: pip install lttb"}
    
    if not isinstance(target_points, int) or target_points <= 0:
        return {"error": f"target_points must be a positive integer, got {target_points}"}
    
    clean_data = np.array(data)
    valid_mask = ~np.isnan(clean_data)
    clean_data = clean_data[valid_mask]
    
    if len(clean_data) == 0:
        return {"error": "No valid data provided"}
    
    if timestamps is not None:
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        clean_timestamps = timestamps[valid_mask].reset_index(drop=True)
        original_tz = getattr(clean_timestamps.dtype, 'tz', None)
        
        if pd.api.types.is_datetime64_any_dtype(clean_timestamps):
            numeric_timestamps = clean_timestamps.astype('int64') // 10**9
        else:
            numeric_timestamps = clean_timestamps.astype(float)
        
        xy_data = np.column_stack([numeric_timestamps, clean_data])
    else:
        clean_timestamps = None
        original_tz = None
        
    indices = np.arange(len(clean_data))
    if timestamps is None:
        xy_data = np.column_stack([indices, clean_data])
    
    if len(clean_data) <= target_points:
        if clean_timestamps is not None:
            warning_timestamps = clean_timestamps.tolist()
        else:
            warning_timestamps = indices.tolist()
            
        return {
            "warning": f"Data size ({len(clean_data)}) is smaller than target points ({target_points})",
            "original_points": len(clean_data),
            "target_points": target_points,
            "downsampled_data": clean_data.tolist(),
            "downsampled_timestamps": warning_timestamps
        }
    
    try:
        downsampled_xy = downsample(xy_data, target_points)
        
        downsampled_x = downsampled_xy[:, 0]
        downsampled_y = downsampled_xy[:, 1]
        
        if clean_timestamps is not None and pd.api.types.is_datetime64_any_dtype(clean_timestamps):
            if original_tz is not None:
                downsampled_timestamps = pd.to_datetime(downsampled_x, unit='s', utc=True).tz_convert(original_tz).tolist()
            else:
                downsampled_timestamps = pd.to_datetime(downsampled_x, unit='s').tolist()
        else:
            downsampled_timestamps = downsampled_x.tolist()
        
        return {
            "original_points": len(clean_data),
            "target_points": target_points,
            "actual_downsampled_points": len(downsampled_y),
            "downsampled_data": downsampled_y.tolist(),
            "downsampled_timestamps": downsampled_timestamps
        }
    
    except Exception as e:
        return {"error": f"Downsampling failed: {str(e)}"}

def analyze_all(data: Union[List, np.ndarray], 
               confidence_level: float = 0.95, 
               percentiles: List[float] = [25, 50, 75, 95]) -> Dict:
    """
    One function that returns all common statistics.
    
    Args:
        data: Numerical data
        confidence_level: Confidence level for interval
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary with all statistical analyses
    """
    return {
        'general': calculate_general_stats(data),
        'confidence_interval': calculate_confidence_interval(data, confidence_level),
        'percentiles': calculate_percentiles(data, percentiles)
    }