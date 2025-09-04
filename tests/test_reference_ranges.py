import pytest

from stats_calculator.reference_ranges import (analyze_multiple_ranges,
                                               analyze_range, analyze_zones,
                                               get_outliers)


@pytest.fixture
def mixed_range_data():
    below = list(range(10, 50))
    in_range = list(range(50, 151))
    above = list(range(151, 189))
    return below + in_range + above


@pytest.fixture
def simple_data():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestAnalyzeRange:

    def test_mixed_range_analysis(self, mixed_range_data):
        result = analyze_range(mixed_range_data, 50, 150)

        assert "error" not in result
        assert result["total_observations"] == 179

        assert result["below_range"]["count"] == 40
        assert abs(result["below_range"]["percentage"] - 22.3) < 0.1
        assert result["below_range"]["most_extreme"] == 10
        assert result["below_range"]["average_deviation"] == 20.5

        assert result["in_range"]["count"] == 101
        assert abs(result["in_range"]["percentage"] - 56.4) < 0.1

        assert result["above_range"]["count"] == 38
        assert abs(result["above_range"]["percentage"] - 21.2) < 0.1
        assert result["above_range"]["most_extreme"] == 188

    def test_all_in_range(self, simple_data):
        result = analyze_range(simple_data, 0, 15)

        assert result["in_range"]["count"] == 10
        assert result["in_range"]["percentage"] == 100.0
        assert result["below_range"] is None
        assert result["above_range"] is None

    def test_all_below_range(self, simple_data):
        result = analyze_range(simple_data, 15, 20)

        assert result["below_range"]["count"] == 10
        assert result["below_range"]["percentage"] == 100.0
        assert result["in_range"]["count"] == 0
        assert result["above_range"] is None

    def test_invalid_range(self, simple_data):
        result = analyze_range(simple_data, 10, 5)
        assert "error" in result
        assert "must be less than" in result["error"]

    def test_empty_data(self):
        result = analyze_range([], 0, 10)
        assert "error" in result
        assert "No valid data" in result["error"]


class TestAnalyzeZones:

    def test_non_overlapping_zones(self, simple_data):
        zones = [("Low", 1, 3), ("Medium", 4, 7), ("High", 8, 10)]

        result = analyze_zones(simple_data, zones)

        assert "error" not in result
        assert result["total_observations"] == 10

        assert result["zones"]["Low"]["count"] == 3
        assert result["zones"]["Medium"]["count"] == 4
        assert result["zones"]["High"]["count"] == 3

        assert result["zones"]["Low"]["percentage"] == 30.0
        assert result["zones"]["Medium"]["percentage"] == 40.0
        assert result["zones"]["High"]["percentage"] == 30.0

    def test_overlapping_zones_rejected(self, simple_data):
        zones = [("Zone1", 1, 5), ("Zone2", 4, 8)]

        result = analyze_zones(simple_data, zones, allow_overlaps=False)
        assert "error" in result
        assert "Overlapping zones" in result["error"]

    def test_overlapping_zones_allowed(self, simple_data):
        zones = [("Zone1", 1, 5), ("Zone2", 4, 8)]

        result = analyze_zones(simple_data, zones, allow_overlaps=True)

        assert "error" not in result
        assert result["zones"]["Zone1"]["count"] == 5
        assert result["zones"]["Zone2"]["count"] == 5

    def test_empty_zones(self, simple_data):
        zones = [("Empty", 50, 60), ("Valid", 1, 5)]

        result = analyze_zones(simple_data, zones)

        assert result["zones"]["Empty"]["count"] == 0
        assert result["zones"]["Empty"]["percentage"] == 0.0
        assert result["zones"]["Valid"]["count"] == 5

    def test_invalid_zone_bounds(self, simple_data):
        zones = [("Invalid", 10, 5)]

        result = analyze_zones(simple_data, zones)
        assert "error" in result
        assert "Invalid zone" in result["error"]


class TestAnalyzeMultipleRanges:

    def test_multiple_ranges(self, simple_data):
        ranges = {"strict": (3, 7), "loose": (1, 10), "narrow": (4, 6)}

        result = analyze_multiple_ranges(simple_data, ranges)

        assert "strict" in result
        assert "loose" in result
        assert "narrow" in result

        assert result["loose"]["in_range"]["count"] == 10

        assert result["narrow"]["in_range"]["count"] == 3


class TestGetOutliers:

    def test_get_outliers(self, mixed_range_data):
        result = get_outliers(mixed_range_data, 50, 150)

        assert "below_range" in result
        assert "above_range" in result
        assert "total_outliers" in result

        assert len(result["below_range"]) == 40
        assert len(result["above_range"]) == 38
        assert result["total_outliers"] == 78

        assert min(result["below_range"]) == 10
        assert max(result["below_range"]) == 49
        assert min(result["above_range"]) == 151
        assert max(result["above_range"]) == 188

    def test_no_outliers(self, simple_data):
        result = get_outliers(simple_data, 0, 15)

        assert result["below_range"] == []
        assert result["above_range"] == []
        assert result["total_outliers"] == 0

    def test_empty_data_outliers(self):
        result = get_outliers([], 0, 10)
        assert "error" in result
