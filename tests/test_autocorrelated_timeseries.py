import numpy as np
import pandas as pd
import pytest

from stats_calculator.autocorrelated_timeseries import (autocorrelated_mean,
                                                        autocorrelated_median,
                                                        autocorrelated_std)


@pytest.fixture
def simple_data():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def autocorrelated_data():
    """Generate data with known autocorrelation"""
    np.random.seed(42)
    n = 100
    data = [0]
    for i in range(1, n):
        # AR(1): x_t = 0.7 * x_{t-1} + noise
        data.append(0.7 * data[i - 1] + np.random.normal(0, 1))
    return data


@pytest.fixture
def pandas_series_data():
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


class TestAutocorrelatedMean:

    def test_mean_calculation(self, simple_data):
        result = autocorrelated_mean(simple_data, 0.95)

        assert "error" not in result
        assert "value" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "confidence_level" in result
        assert "n_observations" in result
        assert "effective_n" in result
        assert "lag1_autocorrelation" in result
        assert "method" in result

        assert result["value"] == 5.5
        assert result["confidence_level"] == 0.95
        assert result["n_observations"] == 10
        assert result["lower_bound"] < result["value"] < result["upper_bound"]
        assert "t-distribution" in result["method"]

    def test_autocorrelated_data_adjustment(self, autocorrelated_data):
        result = autocorrelated_mean(autocorrelated_data, 0.95)

        assert result["lag1_autocorrelation"] > 0.5
        assert result["effective_n"] < result["n_observations"]

        # Compare to naive analysis
        naive_se = np.std(autocorrelated_data, ddof=1) / np.sqrt(
            len(autocorrelated_data)
        )
        adjusted_se = np.std(autocorrelated_data, ddof=1) / np.sqrt(
            result["effective_n"]
        )

        assert adjusted_se > naive_se

    def test_different_confidence_levels(self, simple_data):
        ci_90 = autocorrelated_mean(simple_data, 0.90)
        ci_95 = autocorrelated_mean(simple_data, 0.95)
        ci_99 = autocorrelated_mean(simple_data, 0.99)

        width_90 = ci_90["upper_bound"] - ci_90["lower_bound"]
        width_95 = ci_95["upper_bound"] - ci_95["lower_bound"]
        width_99 = ci_99["upper_bound"] - ci_99["lower_bound"]

        assert width_90 < width_95 < width_99

    def test_insufficient_data(self):
        result = autocorrelated_mean([1, 2], 0.95)
        assert "error" in result
        assert "at least 3" in result["error"]

    def test_invalid_confidence_level(self, simple_data):
        result = autocorrelated_mean(simple_data, 1.5)
        assert "error" in result
        assert "between 0 and 1" in result["error"]

    def test_pandas_series_input(self, pandas_series_data):
        result = autocorrelated_mean(pandas_series_data, 0.95)
        assert "error" not in result
        assert result["value"] == 5.5


class TestAutocorrelatedMedian:

    def test_median_calculation(self, simple_data):
        result = autocorrelated_median(simple_data, 0.95)

        assert "error" not in result
        assert result["value"] == 5.5
        assert result["confidence_level"] == 0.95
        assert "block bootstrap" in result["method"]
        assert result["lower_bound"] < result["value"] < result["upper_bound"]

    def test_autocorrelated_data_block_bootstrap(self, autocorrelated_data):
        result = autocorrelated_median(autocorrelated_data, 0.95)

        assert result["lag1_autocorrelation"] > 0.5
        assert "block_size=" in result["method"]

        block_size_str = result["method"].split("block_size=")[1].split(")")[0]
        block_size = int(block_size_str)
        assert block_size > 1

    def test_insufficient_data_median(self):
        result = autocorrelated_median([1], 0.95)
        assert "error" in result


class TestAutocorrelatedStd:

    def test_std_calculation(self, simple_data):
        result = autocorrelated_std(simple_data, 0.95)

        assert "error" not in result
        assert "value" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert "sigma_bands" in result
        assert "effective_n" in result

        expected_std = np.std(simple_data, ddof=1)
        assert abs(result["value"] - expected_std) < 0.001

        assert result["lower_bound"] < result["value"] < result["upper_bound"]

    def test_sigma_bands(self, simple_data):
        result = autocorrelated_std(simple_data, 0.95)

        bands = result["sigma_bands"]
        assert "mean" in bands
        assert "1_sigma" in bands
        assert "2_sigma" in bands
        assert "3_sigma" in bands

        mean_val = bands["mean"]
        std_val = result["value"]

        assert bands["1_sigma"] == (mean_val - std_val, mean_val + std_val)
        assert bands["2_sigma"] == (mean_val - 2 * std_val, mean_val + 2 * std_val)
        assert bands["3_sigma"] == (mean_val - 3 * std_val, mean_val + 3 * std_val)

    def test_autocorrelated_std_adjustment(self, autocorrelated_data):
        result = autocorrelated_std(autocorrelated_data, 0.95)

        assert result["effective_n"] < result["n_observations"]

        width = result["upper_bound"] - result["lower_bound"]
        assert width > 0

    def test_insufficient_data_std(self):
        result = autocorrelated_std([1, 2], 0.95)
        assert "error" in result
