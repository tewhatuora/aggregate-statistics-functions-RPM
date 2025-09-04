import numpy as np
import pytest

from stats_calculator.general_stats import calculate_iqr, calculate_percentiles


@pytest.fixture
def simple_data():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def uniform_data():
    return list(range(1, 101))


@pytest.fixture
def data_with_nans():
    return [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]


class TestCalculatePercentiles:

    def test_default_percentiles(self, simple_data):
        result = calculate_percentiles(simple_data)

        assert "percentiles" in result
        assert "n_observations" in result
        assert result["n_observations"] == 10

        percentiles = result["percentiles"]
        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles

        assert percentiles["p50"] == 5.5
        assert percentiles["p25"] < percentiles["p50"] < percentiles["p75"]

    def test_custom_percentiles(self, uniform_data):
        result = calculate_percentiles(uniform_data, [10, 90])

        percentiles = result["percentiles"]
        assert "p10" in percentiles
        assert "p90" in percentiles
        assert len(percentiles) == 2

        assert 10 < percentiles["p10"] < 12
        assert 89 < percentiles["p90"] < 91

    def test_extreme_percentiles(self, simple_data):
        result = calculate_percentiles(simple_data, [0, 100])

        percentiles = result["percentiles"]
        assert percentiles["p0"] == 1
        assert percentiles["p100"] == 10

    def test_decimal_percentiles(self, uniform_data):
        result = calculate_percentiles(uniform_data, [0.1, 99.9])

        percentiles = result["percentiles"]
        assert "p0.1" in percentiles
        assert "p99.9" in percentiles
        assert percentiles["p0.1"] < percentiles["p99.9"]

    def test_invalid_percentiles(self, simple_data):
        result = calculate_percentiles(simple_data, [-1, 101])
        assert "error" in result
        assert "between 0 and 100" in result["error"]

    def test_empty_data(self):
        result = calculate_percentiles([])
        assert "error" in result
        assert "No valid data" in result["error"]

    def test_data_with_nans(self, data_with_nans):
        result = calculate_percentiles(data_with_nans, [25, 50, 75])

        assert "error" not in result
        assert result["n_observations"] == 8

        percentiles = result["percentiles"]
        assert 1 <= percentiles["p25"] <= 10
        assert 1 <= percentiles["p50"] <= 10
        assert 1 <= percentiles["p75"] <= 10


class TestCalculateIQR:

    def test_iqr_calculation(self, simple_data):
        result = calculate_iqr(simple_data)

        assert "error" not in result
        assert "q1" in result
        assert "median" in result
        assert "q3" in result
        assert "iqr" in result
        assert "outlier_bounds" in result
        assert "n_observations" in result

        assert result["n_observations"] == 10
        assert result["median"] == 5.5
        assert result["q1"] < result["median"] < result["q3"]
        assert result["iqr"] == result["q3"] - result["q1"]

    def test_outlier_bounds(self, simple_data):
        result = calculate_iqr(simple_data)

        bounds = result["outlier_bounds"]
        assert "lower_fence" in bounds
        assert "upper_fence" in bounds

        expected_lower = result["q1"] - 1.5 * result["iqr"]
        assert abs(bounds["lower_fence"] - expected_lower) < 0.001

        expected_upper = result["q3"] + 1.5 * result["iqr"]
        assert abs(bounds["upper_fence"] - expected_upper) < 0.001

    def test_iqr_with_nans(self, data_with_nans):
        result = calculate_iqr(data_with_nans)

        assert "error" not in result
        assert result["n_observations"] == 8
        assert result["q1"] < result["median"] < result["q3"]

    def test_iqr_empty_data(self):
        result = calculate_iqr([])
        assert "error" in result
