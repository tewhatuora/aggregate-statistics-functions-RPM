import numpy as np
import pandas as pd

from stats_calculator.utils import clean_data, downsample_data


class TestCleanData:

    def test_numpy_array_input(self):
        data = np.array([1, 2, np.nan, 4, 5])
        result = clean_data(data)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        assert not np.any(np.isnan(result))
        assert np.array_equal(result, [1, 2, 4, 5])

    def test_pandas_series_input(self):
        data = pd.Series([1, 2, np.nan, 4, 5])
        result = clean_data(data)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        assert not np.any(np.isnan(result))

    def test_list_input(self):
        data = [1, 2, np.nan, 4, 5]
        result = clean_data(data)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        assert not np.any(np.isnan(result))

    def test_no_nans(self):
        data = [1, 2, 3, 4, 5]
        result = clean_data(data)

        assert len(result) == 5
        assert np.array_equal(result, [1, 2, 3, 4, 5])

    def test_all_nans(self):
        data = [np.nan, np.nan, np.nan]
        result = clean_data(data)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)


class TestDownsampleData:

    def test_basic_downsampling(self):
        data = list(range(100))
        result = downsample_data(data, target_points=10)

        if "error" not in result:
            assert result["original_points"] == 100
            assert result["target_points"] == 10
            assert len(result["downsampled_data"]) <= 10

            downsampled = result["downsampled_data"]
            assert downsampled[0] == 0
            assert downsampled[-1] == 99

    def test_with_timestamps(self):
        data = list(range(50))
        timestamps = pd.date_range("2024-01-01", periods=50, freq="1min")

        result = downsample_data(data, timestamps, target_points=10)

        if "error" not in result:
            assert len(result["downsampled_data"]) == len(
                result["downsampled_timestamps"]
            )
            assert result["original_points"] == 50

    def test_small_dataset_warning(self):
        data = [1, 2, 3, 4, 5]
        result = downsample_data(data, target_points=10)

        assert "warning" in result
        assert result["original_points"] == 5
        assert result["downsampled_data"] == data

    def test_with_nans(self):
        data = [1, 2, np.nan, 4, 5, np.nan, 7, 8]
        result = downsample_data(data, target_points=4)

        if "error" not in result:
            assert result["original_points"] == 6
            downsampled = result["downsampled_data"]
            assert not any(np.isnan(x) for x in downsampled)

    def test_invalid_target_points(self):
        data = list(range(20))

        result = downsample_data(data, target_points=0)
        assert "error" in result
        assert "positive integer" in result["error"]

        result = downsample_data(data, target_points=-5)
        assert "error" in result

        result = downsample_data(data, target_points=10.5)
        assert "error" in result

    def test_empty_data(self):
        result = downsample_data([], target_points=10)
        assert "error" in result
        assert "No valid data" in result["error"]

    def test_lttb_not_available(self, monkeypatch):
        import stats_calculator.utils

        monkeypatch.setattr(stats_calculator.utils, "LTTB_AVAILABLE", False)

        data = list(range(20))
        result = downsample_data(data, target_points=10)

        assert "error" in result
        assert "lttb library not installed" in result["error"]
