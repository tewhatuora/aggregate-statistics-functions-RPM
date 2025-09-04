"""
Statistical functions for autocorrelated time series data.

Accounts for temporal dependencies in biometric data where consecutive
measurements are correlated (heart rate, blood pressure, weight, etc.).
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats
from .utils import clean_data


def _estimate_lag1_autocorrelation(data: np.ndarray) -> float:
    """Estimate lag-1 autocorrelation coefficient."""
    if len(data) < 3:
        return 0.0

    mean_val = np.mean(data)
    numerator = np.sum((data[:-1] - mean_val) * (data[1:] - mean_val))
    denominator = np.sum((data - mean_val) ** 2)

    if denominator == 0:
        return 0.0

    autocorr = numerator / denominator
    return np.clip(autocorr, -0.99, 0.99)


def _effective_sample_size(n: int, autocorr: float) -> int:
    """Calculate effective sample size for AR(1) process."""
    if autocorr <= 0:
        return n

    eff_n = n * (1 - autocorr) / (1 + autocorr)
    return max(int(eff_n), 3)


def autocorrelated_mean(
    data: Union[np.ndarray, pd.Series, List], confidence_level: float = 0.95
) -> Dict:
    """
    Calculate mean with confidence intervals for autocorrelated data.

    Args:
        data: Time series data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with mean, confidence bounds, and metadata
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) < 3:
        return {"error": "Need at least 3 valid observations"}

    if not 0 < confidence_level < 1:
        return {"error": "Confidence level must be between 0 and 1"}

    n = len(cleaned_data)
    autocorr = _estimate_lag1_autocorrelation(cleaned_data)
    eff_n = _effective_sample_size(n, autocorr)

    mean_val = float(np.mean(cleaned_data))
    std_error = np.std(cleaned_data, ddof=1) / np.sqrt(eff_n)

    alpha = 1 - confidence_level
    t_val = stats.t.ppf(1 - alpha / 2, df=eff_n - 1)
    margin = t_val * std_error

    return {
        "value": mean_val,
        "lower_bound": mean_val - margin,
        "upper_bound": mean_val + margin,
        "confidence_level": confidence_level,
        "n_observations": n,
        "effective_n": eff_n,
        "lag1_autocorrelation": autocorr,
        "method": "t-distribution with autocorrelation adjustment",
    }


def autocorrelated_median(
    data: Union[np.ndarray, pd.Series, List], confidence_level: float = 0.95
) -> Dict:
    """
    Calculate median with confidence intervals using block bootstrap.

    Args:
        data: Time series data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with median, confidence bounds, and metadata
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) < 3:
        return {"error": "Need at least 3 valid observations"}

    n = len(cleaned_data)
    autocorr = _estimate_lag1_autocorrelation(cleaned_data)
    median_val = float(np.median(cleaned_data))

    # Block bootstrap
    block_size = max(1, int(np.ceil(2 / (1 - abs(autocorr)))))
    n_boot = 1000
    rng = np.random.default_rng(42)

    boot_medians = []
    for _ in range(n_boot):
        if block_size == 1:
            boot_sample = rng.choice(cleaned_data, size=n, replace=True)
        else:
            n_blocks = int(np.ceil(n / block_size))
            boot_sample = []
            for _ in range(n_blocks):
                start_idx = rng.integers(0, len(cleaned_data) - block_size + 1)
                boot_sample.extend(cleaned_data[start_idx : start_idx + block_size])
            boot_sample = np.array(boot_sample[:n])

        boot_medians.append(np.median(boot_sample))

    alpha = 1 - confidence_level
    lower_bound = float(np.percentile(boot_medians, 100 * alpha / 2))
    upper_bound = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))

    return {
        "value": median_val,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "confidence_level": confidence_level,
        "n_observations": n,
        "lag1_autocorrelation": autocorr,
        "method": f"block bootstrap (block_size={block_size})",
    }


def autocorrelated_std(
    data: Union[np.ndarray, pd.Series, List], confidence_level: float = 0.95
) -> Dict:
    """
    Calculate standard deviation with confidence intervals and sigma bands.

    Args:
        data: Time series data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary with std dev, confidence intervals, and sigma bands
    """
    cleaned_data = clean_data(data)

    if len(cleaned_data) < 3:
        return {"error": "Need at least 3 valid observations"}

    n = len(cleaned_data)
    autocorr = _estimate_lag1_autocorrelation(cleaned_data)
    eff_n = _effective_sample_size(n, autocorr)

    mean_val = float(np.mean(cleaned_data))
    std_val = float(np.std(cleaned_data, ddof=1))

    # Confidence interval using chi-square with effective n
    alpha = 1 - confidence_level
    chi2_lower = stats.chi2.ppf(alpha / 2, df=eff_n - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=eff_n - 1)

    std_lower = std_val * np.sqrt((eff_n - 1) / chi2_upper)
    std_upper = std_val * np.sqrt((eff_n - 1) / chi2_lower)

    # Sigma bands
    sigma_bands = {
        "mean": mean_val,
        "1_sigma": (mean_val - std_val, mean_val + std_val),
        "2_sigma": (mean_val - 2 * std_val, mean_val + 2 * std_val),
        "3_sigma": (mean_val - 3 * std_val, mean_val + 3 * std_val),
    }

    return {
        "value": std_val,
        "lower_bound": std_lower,
        "upper_bound": std_upper,
        "confidence_level": confidence_level,
        "sigma_bands": sigma_bands,
        "n_observations": n,
        "effective_n": eff_n,
        "lag1_autocorrelation": autocorr,
    }
