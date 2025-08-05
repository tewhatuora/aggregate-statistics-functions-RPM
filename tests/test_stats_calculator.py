import pytest
import numpy as np
import pandas as pd
import math
import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stats_calculator import (
    calculate_general_stats,
    calculate_confidence_interval,
    calculate_percentiles,
    analyze_value_based_range,
    analyze_time_based_range,
    analyze_band_based_range,
    downsample_data
)

@pytest.fixture
def perfect_normal_data():
    np.random.seed(12345)  
    return np.random.normal(100, 10, 10000).tolist()

@pytest.fixture
def simple_test_data():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

@pytest.fixture
def data_with_nans():
    return [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]

@pytest.fixture
def uniform_data():
    return list(range(1, 101))  

@pytest.fixture
def timestamps_minute():
    return pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min', tz='UTC')

@pytest.fixture
def timestamps_second():
    return pd.date_range('2024-01-01 00:00:00', periods=100, freq='1s', tz='UTC')

@pytest.fixture
def range_test_data():
    below_range = list(range(10, 50))     
    in_range = list(range(50, 150))       
    above_range = list(range(150, 189))  
    return below_range + in_range + above_range

class TestGeneralStats:
    
    def test_simple_data_statistics(self, simple_test_data):
        result = calculate_general_stats(simple_test_data)
        
        assert result['count'] == 10
        assert result['mean'] == 5.5
        assert result['median'] == 5.5
        assert result['min'] == 1
        assert result['max'] == 10
        assert abs(result['iqr'] - 4.5) < 0.1  
        
        expected_var = 82.5 / 9 
        assert abs(result['variance'] - expected_var) < 0.01
        assert abs(result['std_dev'] - math.sqrt(expected_var)) < 0.01
        
        expected_se = math.sqrt(expected_var) / math.sqrt(10)
        assert abs(result['std_error'] - expected_se) < 0.01
    
    def test_uniform_data_properties(self, uniform_data):
        result = calculate_general_stats(uniform_data)
        
        assert result['count'] == 100
        assert result['mean'] == 50.5
        assert result['median'] == 50.5
        assert result['min'] == 1
        assert result['max'] == 100
        assert abs(result['iqr'] - 49.5) < 1.0
        assert abs(result['skewness']) < 0.1
    
    def test_data_with_nans_comprehensive(self, data_with_nans):
        result = calculate_general_stats(data_with_nans)
        
        assert result['count'] == 8
        assert result['min'] == 1
        assert result['max'] == 10
        
        valid_data = [1, 2, 4, 5, 7, 8, 9, 10]
        expected_mean = sum(valid_data) / len(valid_data)
        assert abs(result['mean'] - expected_mean) < 0.001
        
        expected_var = sum((x - expected_mean) ** 2 for x in valid_data) / (len(valid_data) - 1)
        assert abs(result['variance'] - expected_var) < 0.001
        assert abs(result['std_dev'] - math.sqrt(expected_var)) < 0.001
        
        expected_se = math.sqrt(expected_var) / math.sqrt(len(valid_data))
        assert abs(result['std_error'] - expected_se) < 0.001
    
    def test_empty_data_error_message(self):
        result = calculate_general_stats([])
        assert 'error' in result
        error_msg = result['error'].lower()
        assert any(keyword in error_msg for keyword in ['no valid data', 'no data', 'empty'])
    
    def test_all_nan_data_error_message(self):
        result = calculate_general_stats([np.nan, np.nan, np.nan])
        assert 'error' in result
        error_msg = result['error'].lower()
        assert any(keyword in error_msg for keyword in ['no valid data', 'no data', 'nan'])
    
    def test_single_value_comprehensive(self):
        result = calculate_general_stats([42])
        
        assert result['count'] == 1
        assert result['mean'] == 42
        assert result['median'] == 42
        assert result['min'] == 42
        assert result['max'] == 42
        
        assert result['iqr'] == 0
        assert result['variance'] == 0
        assert result['std_dev'] == 0
        assert result['std_error'] == 0
        
        assert result['skewness'] == 0 or math.isnan(result['skewness'])
        assert result['kurtosis'] == 0 or math.isnan(result['kurtosis'])

class TestConfidenceInterval:
    
    def test_confidence_interval_95(self, simple_test_data):
        result = calculate_confidence_interval(simple_test_data, 0.95)
        
        assert result['confidence_level'] == 0.95
        assert result['mean'] == 5.5
        assert abs(result['z_score'] - 1.96) < 0.01
        
        margin = result['margin_of_error']
        assert abs(result['lower_bound'] - (5.5 - margin)) < 0.01
        assert abs(result['upper_bound'] - (5.5 + margin)) < 0.01
    
    def test_confidence_interval_99_complete(self, simple_test_data):
        result = calculate_confidence_interval(simple_test_data, 0.99)
        
        assert result['confidence_level'] == 0.99
        assert result['mean'] == 5.5
        assert abs(result['z_score'] - 2.576) < 0.01
        
        expected_se = math.sqrt(82.5/9) / math.sqrt(10)
        expected_margin = 2.576 * expected_se
        assert abs(result['margin_of_error'] - expected_margin) < 0.01
        
        assert abs(result['lower_bound'] - (5.5 - expected_margin)) < 0.01
        assert abs(result['upper_bound'] - (5.5 + expected_margin)) < 0.01
        
        ci_95 = calculate_confidence_interval(simple_test_data, 0.95)
        assert result['margin_of_error'] > ci_95['margin_of_error']
    
    def test_confidence_interval_different_levels(self, uniform_data):
        ci_90 = calculate_confidence_interval(uniform_data, 0.90)
        ci_95 = calculate_confidence_interval(uniform_data, 0.95)
        ci_99 = calculate_confidence_interval(uniform_data, 0.99)
        
        assert ci_90['margin_of_error'] < ci_95['margin_of_error']
        assert ci_95['margin_of_error'] < ci_99['margin_of_error']
    
    def test_insufficient_data_boundary_conditions(self):
        result = calculate_confidence_interval([42])
        assert 'error' in result
        assert 'at least 2 data points' in result['error']
        
        result = calculate_confidence_interval([10, 20])
        assert 'error' not in result
        assert result['mean'] == 15
        assert result['confidence_level'] == 0.95
    
    def test_invalid_confidence_levels(self, simple_test_data):
        result = calculate_confidence_interval(simple_test_data, 1.5)
        assert 'error' in result
        assert 'between 0 and 1' in result['error']
        
        result = calculate_confidence_interval(simple_test_data, -0.1)
        assert 'error' in result
        assert 'between 0 and 1' in result['error']
        
        result = calculate_confidence_interval(simple_test_data, 0)
        assert 'error' in result
        assert 'between 0 and 1' in result['error']
        
        result = calculate_confidence_interval(simple_test_data, 1.0)
        assert 'error' in result
        assert 'between 0 and 1' in result['error']

class TestPercentiles:
    
    def test_uniform_percentiles_values(self, uniform_data):
        percentiles = [0, 25, 50, 75, 100]
        result = calculate_percentiles(uniform_data, percentiles)
        
        assert abs(result['p0'] - 1) < 0.1
        assert abs(result['p25'] - 25.75) < 1
        assert abs(result['p50'] - 50.5) < 1
        assert abs(result['p75'] - 75.25) < 1
        assert abs(result['p100'] - 100) < 0.1
    
    def test_simple_percentiles_exact(self, simple_test_data):
        result = calculate_percentiles(simple_test_data, [0, 25, 50, 75, 100])
        
        assert result['p0'] == 1
        assert result['p50'] == 5.5
        assert result['p100'] == 10
        
        assert 2.5 <= result['p25'] <= 3.5
        assert 7.5 <= result['p75'] <= 8.5
    
    def test_custom_percentiles_values(self, uniform_data):
        percentiles = [1, 5, 95, 99]
        result = calculate_percentiles(uniform_data, percentiles)
        
        assert 'p1' in result
        assert 'p5' in result
        assert 'p95' in result
        assert 'p99' in result
        
        assert abs(result['p1'] - 1.99) < 1
        assert abs(result['p5'] - 5.95) < 1
        assert abs(result['p95'] - 95.05) < 1
        assert abs(result['p99'] - 99.01) < 1
        
        assert result['p1'] < result['p5'] < result['p95'] < result['p99']
    
    def test_decimal_percentiles(self):
        data = list(range(1, 1001))
        percentiles = [0.1, 99.9, 50.5]
        result = calculate_percentiles(data, percentiles)
        
        assert 'p0.1' in result
        assert 'p99.9' in result
        assert 'p50.5' in result
        
        assert result['p0.1'] < 10
        assert result['p99.9'] > 990
        assert 500 < result['p50.5'] < 510
    
    def test_invalid_percentiles_specific(self, simple_test_data):
        result = calculate_percentiles(simple_test_data, [-1, -10])
        assert 'error' in result
        assert 'between 0 and 100' in result['error'] and 'inclusive' in result['error']
        
        result = calculate_percentiles(simple_test_data, [101, 150])
        assert 'error' in result
        assert 'between 0 and 100' in result['error']
        
        result = calculate_percentiles(simple_test_data, [25, -1, 75])
        assert 'error' in result
    
    def test_empty_percentiles(self, simple_test_data):
        result = calculate_percentiles(simple_test_data, [])
        assert result == {}
    
    def test_percentiles_with_nans(self):
        data_with_nans = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        result = calculate_percentiles(data_with_nans, [25, 50, 75])
        
        assert 'p25' in result
        assert 'p50' in result
        assert 'p75' in result
        
        assert 1 <= result['p25'] <= 5
        assert 4 <= result['p50'] <= 8
        assert 7 <= result['p75'] <= 10

class TestValueBasedRange:
    
    def test_range_analysis_known_data(self, range_test_data):
        result = analyze_value_based_range(range_test_data, 50, 150)
        
        total_count = len(range_test_data)
        
        assert result['below_range']['count'] == 40
        assert abs(result['below_range']['percentage'] - (40/179)*100) < 0.1
        
        assert result['above_range']['count'] == 38
        assert abs(result['above_range']['percentage'] - (38/179)*100) < 0.1
        
        assert result['below_range']['max_deviation'] == 40
        assert result['above_range']['max_deviation'] == 38
    
    def test_boundary_value_behavior(self):
        data = [49, 50, 150, 151]
        result = analyze_value_based_range(data, 50, 150)
        
        assert result['below_range']['count'] == 1
        assert result['above_range']['count'] == 1
        
        assert result['below_range']['max_deviation'] == 1
        assert result['above_range']['max_deviation'] == 1
    
    def test_no_values_outside_range(self, simple_test_data):
        result = analyze_value_based_range(simple_test_data, 0, 15)
        
        assert result['below_range']['count'] == 0
        assert result['above_range']['count'] == 0
        assert result['below_range']['percentage'] == 0
        assert result['above_range']['percentage'] == 0
        assert result['below_range']['max_deviation'] == 0
        assert result['above_range']['max_deviation'] == 0
    
    def test_all_values_outside_range_scenarios(self, simple_test_data):
        result = analyze_value_based_range(simple_test_data, 11, 12)
        assert result['below_range']['count'] == 10
        assert result['above_range']['count'] == 0
        assert result['below_range']['percentage'] == 100
        assert result['below_range']['max_deviation'] == 10
        
        result = analyze_value_based_range(simple_test_data, -5, 0)
        assert result['below_range']['count'] == 0
        assert result['above_range']['count'] == 10
        assert result['above_range']['percentage'] == 100
        assert result['above_range']['max_deviation'] == 10
    
    def test_invalid_range_specific_errors(self, simple_test_data):
        result = analyze_value_based_range(simple_test_data, 10, 5)
        assert 'error' in result
        assert 'Invalid range' in result['error'] and 'must be less than' in result['error']
        
        result = analyze_value_based_range(simple_test_data, 5, 5)
        assert 'error' in result
        assert 'Invalid range' in result['error'] and 'must be less than' in result['error']

class TestTimeBasedRange:
    
    def test_time_episodes_comprehensive(self):
        data = [60, 40, 40, 60, 70, 70, 60]  
        timestamps = pd.date_range('2024-01-01', periods=7, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 50, 65, 'minutes')
        
        assert result['below_range']['episode_count'] == 1
        assert result['above_range']['episode_count'] == 1
        
        assert result['below_range']['total_duration'] == 1.0
        assert result['above_range']['total_duration'] == 1.0
        
        if 'episodes' in result['below_range']:
            assert len(result['below_range']['episodes']) == 1
        if 'episodes' in result['above_range']:
            assert len(result['above_range']['episodes']) == 1
    
    def test_single_point_episodes(self):
        data = [60, 40, 60, 70, 60]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 50, 65, 'minutes')
        
        assert result['below_range']['episode_count'] == 1
        assert result['above_range']['episode_count'] == 1
        
        assert result['below_range']['total_duration'] <= 1.0
        assert result['above_range']['total_duration'] <= 1.0
    
    def test_adjacent_episodes(self):
        data = [40, 60, 40, 60, 70]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 50, 65, 'minutes')
        
        assert result['below_range']['episode_count'] == 2
        assert result['above_range']['episode_count'] == 1
    
    def test_time_duration_units_comprehensive(self):
        data = [40, 40, 40, 60]
        timestamps = pd.date_range('2024-01-01', periods=4, freq='1min', tz='UTC')
        
        result_min = analyze_time_based_range(data, timestamps, 50, 70, 'minutes')
        result_sec = analyze_time_based_range(data, timestamps, 50, 70, 'seconds')
        result_hours = analyze_time_based_range(data, timestamps, 50, 70, 'hours')
        
        assert result_min['below_range']['total_duration'] == 2.0
        assert result_sec['below_range']['total_duration'] == 120.0
        assert abs(result_hours['below_range']['total_duration'] - 2/60) < 0.001
    
    def test_no_episodes(self):
        data = [55, 56, 57, 58, 59]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 50, 60)
        
        assert result['below_range']['episode_count'] == 0
        assert result['above_range']['episode_count'] == 0
        assert result['below_range']['total_duration'] == 0
        assert result['above_range']['total_duration'] == 0
    
    def test_entire_period_out_of_range(self):
        data = [40, 41, 42, 43, 44]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 50, 60, 'minutes')
        
        assert result['below_range']['episode_count'] == 1
        assert result['above_range']['episode_count'] == 0
        assert result['below_range']['total_duration'] == 4.0
    
    def test_mismatched_lengths_specific_error(self):
        data = [1, 2, 3]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        with pytest.raises((IndexError, ValueError)) as excinfo:
            analyze_time_based_range(data, timestamps, 0, 10)
        
        error_msg = str(excinfo.value).lower()
        assert any(keyword in error_msg for keyword in ['length', 'mismatch', 'boolean index'])
    
    def test_invalid_time_range_bounds(self):
        data = [55, 56, 57, 58, 59]
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min', tz='UTC')
        
        result = analyze_time_based_range(data, timestamps, 60, 50)
        assert 'error' in result
        assert 'Invalid range' in result['error'] and 'must be less than' in result['error']

class TestBandBasedRange:
    
    def test_band_analysis_uniform(self, uniform_data):
        bands = [
            ("Low", 1, 26),     
            ("Medium", 26, 76), 
            ("High", 76, 101)   
        ]
        
        result = analyze_band_based_range(uniform_data, bands)
        
        assert result['Low']['count'] == 26
        assert result['Medium']['count'] == 51  
        assert result['High']['count'] == 25
        
        assert abs(result['Low']['percentage'] - 26.0) < 0.1
        assert abs(result['Medium']['percentage'] - 51.0) < 0.1
        assert abs(result['High']['percentage'] - 25.0) < 0.1
    
    def test_overlapping_bands_comprehensive(self):
        data = [10, 20, 30, 40, 50]
        bands = [
            ("Band1", 15, 35),
            ("Band2", 25, 45),
            ("Band3", 5, 25)
        ]
        
        result = analyze_band_based_range(data, bands)
        
        assert result['Band1']['count'] == 2
        assert result['Band2']['count'] == 2
        assert result['Band3']['count'] == 2
        
        total_in_bands = result['Band1']['count'] + result['Band2']['count'] + result['Band3']['count']
        assert total_in_bands == 6
        
        unique_values_in_any_band = len([x for x in data if any(
            lower <= x <= upper for _, lower, upper in bands
        )])
        assert total_in_bands > unique_values_in_any_band
        
        assert abs(result['Band1']['percentage'] - (2/5)*100) < 0.1
        assert abs(result['Band2']['percentage'] - (2/5)*100) < 0.1
        assert abs(result['Band3']['percentage'] - (2/5)*100) < 0.1
    
    def test_non_overlapping_bands_complete_coverage(self):
        data = list(range(1, 101))
        bands = [
            ("First", 1, 25),
            ("Second", 26, 50),
            ("Third", 51, 75),
            ("Fourth", 76, 100)
        ]
        
        result = analyze_band_based_range(data, bands)
        
        for band_name in ["First", "Second", "Third", "Fourth"]:
            assert result[band_name]['count'] == 25
            assert abs(result[band_name]['percentage'] - 25.0) < 0.1
        
        total_coverage = sum(result[band]['count'] for band in ["First", "Second", "Third", "Fourth"])
        assert total_coverage == 100
    
    def test_empty_bands_specific(self):
        data = [10, 20, 30]
        bands = [
            ("Empty1", 50, 60),
            ("Empty2", 100, 200),
            ("Valid", 15, 25),
            ("AlsoEmpty", -10, 0)
        ]
        
        result = analyze_band_based_range(data, bands)
        
        assert result['Empty1']['count'] == 0
        assert result['Empty1']['percentage'] == 0.0
        assert result['Empty2']['count'] == 0
        assert result['AlsoEmpty']['count'] == 0
        
        assert result['Valid']['count'] == 1
        assert abs(result['Valid']['percentage'] - (1/3)*100) < 0.1
    
    def test_invalid_band_bounds_comprehensive(self, simple_test_data):
        bands = [("Invalid", 10, 5)]
        result = analyze_band_based_range(simple_test_data, bands)
        assert 'error' in result
        assert "Invalid band 'Invalid'" in result['error'] and 'must not exceed' in result['error']
        
        bands = [
            ("Valid", 1, 5),
            ("Invalid", 10, 5),
            ("AlsoValid", 6, 10)
        ]
        result = analyze_band_based_range(simple_test_data, bands)
        assert 'error' in result
        assert "Invalid band 'Invalid'" in result['error']
        
        bands = [("ZeroWidth", 5, 5)]
        result = analyze_band_based_range(simple_test_data, bands)
        assert 'error' not in result
        assert result['ZeroWidth']['count'] == 1
    
    def test_band_with_extreme_values(self):
        data = [1, 100, 1000, -50]
        bands = [
            ("Negative", -100, 0),
            ("Small", 0, 50),
            ("Large", 50, 500),
            ("Huge", 500, 2000)
        ]
        
        result = analyze_band_based_range(data, bands)
        
        assert result['Negative']['count'] == 1
        assert result['Small']['count'] == 1
        assert result['Large']['count'] == 1
        assert result['Huge']['count'] == 1

class TestDownsampleData:
    
    def test_downsample_algorithm_quality(self):
        data = list(range(100))
        result = downsample_data(data, target_points=10)
        
        if 'error' not in result:
            downsampled = result['downsampled_data']
            
            assert result['original_points'] == 100
            assert result['target_points'] == 10
            assert len(downsampled) <= 10
            
            assert downsampled[0] == data[0]
            assert downsampled[-1] == data[-1]
            
            if len(downsampled) > 2:
                gaps = [downsampled[i+1] - downsampled[i] for i in range(len(downsampled)-1)]
                avg_gap = sum(gaps) / len(gaps)
                
                for gap in gaps:
                    assert abs(gap - avg_gap) < avg_gap * 1.0, f"Gap {gap} too far from average {avg_gap}"
            
            assert len(set(downsampled)) == len(downsampled)
            
            assert downsampled == sorted(downsampled)
    
    def test_downsample_with_timestamps_comprehensive(self, uniform_data, timestamps_minute):
        result = downsample_data(uniform_data[:50], timestamps_minute[:50], target_points=10)
        
        if 'error' not in result:
            assert len(result['downsampled_data']) == len(result['downsampled_timestamps'])
            
            timestamps = result['downsampled_timestamps']
            assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            
            original_timestamps = timestamps_minute[:50]
            assert result['downsampled_timestamps'][0] == original_timestamps[0]
            assert result['downsampled_timestamps'][-1] == original_timestamps[-1]
            
            data = result['downsampled_data']
            assert data[0] == uniform_data[0]
            assert data[-1] == uniform_data[49]
    
    def test_downsample_small_dataset_warning(self):
        small_data = [1, 2, 3, 4, 5]
        result = downsample_data(small_data, target_points=10)
        
        assert 'warning' in result
        assert result['original_points'] == 5
        assert len(result['downsampled_data']) == 5
        assert result['downsampled_data'] == small_data
        assert 'smaller than target points' in result['warning']
    
    def test_downsample_with_nans_comprehensive(self):
        data_with_nans = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        result = downsample_data(data_with_nans, target_points=5)
        
        if 'error' not in result:
            assert result['original_points'] == 8
            
            downsampled = result['downsampled_data']
            assert all(not (isinstance(x, float) and math.isnan(x)) for x in downsampled)
            
            valid_data = [1, 2, 4, 5, 7, 8, 9, 10]
            assert downsampled[0] == valid_data[0]
            assert downsampled[-1] == valid_data[-1]
            
            assert all(val in valid_data for val in downsampled)
    
    def test_downsample_identical_values(self):
        identical_data = [42] * 100
        result = downsample_data(identical_data, target_points=10)
        
        if 'error' not in result:
            downsampled = result['downsampled_data']
            
            assert all(val == 42 for val in downsampled)
            assert len(downsampled) <= 10
            
            assert result['original_points'] == 100
    
    def test_downsample_extreme_target_points(self):
        data = list(range(50))
        
        result = downsample_data(data, target_points=1)
        if 'error' not in result:
            assert len(result['downsampled_data']) == 1
            assert result['downsampled_data'][0] in [0, 24, 49]
        
        result = downsample_data(data, target_points=0)
        assert 'error' in result
        assert 'positive integer' in result['error']
        
        result = downsample_data(data, target_points=-5)
        assert 'error' in result
        assert 'positive integer' in result['error']
        
        result = downsample_data(data, target_points=10.5)
        assert 'error' in result
        assert 'positive integer' in result['error']
        
        result = downsample_data(data, target_points=1000)
        if 'warning' in result:
            assert len(result['downsampled_data']) == 50
        elif 'error' not in result:
            assert len(result['downsampled_data']) <= 50

class TestIntegration:
    
    def test_complete_analysis_workflow(self, perfect_normal_data):
        general = calculate_general_stats(perfect_normal_data)
        ci = calculate_confidence_interval(perfect_normal_data, 0.95)
        percentiles = calculate_percentiles(perfect_normal_data, [25, 50, 75])
        value_range = analyze_value_based_range(perfect_normal_data, 90, 110)
        
        assert abs(general['mean'] - ci['mean']) < 0.01
        assert abs(general['median'] - percentiles['p50']) < 0.01
        
        assert abs(general['mean'] - 100) < 1  
        assert abs(general['std_dev'] - 10) < 1 
        assert abs(general['skewness']) < 0.1   
    
    def test_json_serialization(self, simple_test_data):
        results = {
            'general': calculate_general_stats(simple_test_data),
            'ci': calculate_confidence_interval(simple_test_data),
            'percentiles': calculate_percentiles(simple_test_data, [25, 50, 75]),
            'value_range': analyze_value_based_range(simple_test_data, 3, 8),
            'bands': analyze_band_based_range(simple_test_data, [("Low", 1, 5), ("High", 6, 10)])
        }
        
        try:
            json_str = json.dumps(results)
            assert len(json_str) > 100
        except TypeError as e:
            pytest.fail(f"JSON serialization failed: {e}")
        
        parsed = json.loads(json_str)
        
        expected_keys = ['general', 'ci', 'percentiles', 'value_range', 'bands']
        for key in expected_keys:
            assert key in parsed, f"Missing key: {key}"
        
        assert parsed['general']['count'] == 10
        assert parsed['general']['mean'] == 5.5
        assert parsed['ci']['confidence_level'] == 0.95
        assert 'p25' in parsed['percentiles']
        assert 'p50' in parsed['percentiles']
        assert 'p75' in parsed['percentiles']
        
        assert 'below_range' in parsed['value_range']
        assert 'above_range' in parsed['value_range']
        assert 'count' in parsed['value_range']['below_range']
        assert 'percentage' in parsed['value_range']['below_range']
        
        assert 'Low' in parsed['bands']
        assert 'High' in parsed['bands']
        assert 'count' in parsed['bands']['Low']
        assert 'percentage' in parsed['bands']['Low']
        
    def test_json_with_edge_cases(self):
        edge_case_data = [1, 2, 3]
        
        results = {
            'general': calculate_general_stats(edge_case_data),
            'ci': calculate_confidence_interval(edge_case_data),
            'percentiles': calculate_percentiles(edge_case_data, [0, 50, 100])
        }
        
        try:
            json_str = json.dumps(results)
            parsed = json.loads(json_str)
            
            if 'error' in results['ci']:
                assert 'error' in parsed['ci']
                
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed with edge cases: {e}")

class TestValidationImprovements:
    
    def test_confidence_interval_boundary_values(self, simple_test_data):
        result = calculate_confidence_interval(simple_test_data, 0.01)
        assert 'error' not in result
        assert result['confidence_level'] == 0.01
        
        result = calculate_confidence_interval(simple_test_data, 0.99)
        assert 'error' not in result
        assert result['confidence_level'] == 0.99
        
        result = calculate_confidence_interval(simple_test_data, 0.0)
        assert 'error' in result
        
        result = calculate_confidence_interval(simple_test_data, 1.0)
        assert 'error' in result
    
    def test_downsample_validation_comprehensive(self):
        data = list(range(100))
        
        invalid_targets = [0, -1, -10, 'string', None, [], {}]
        
        for invalid_target in invalid_targets:
            try:
                result = downsample_data(data, target_points=invalid_target)
                assert 'error' in result
                assert 'positive integer' in result['error']
            except TypeError:
                pass
    
    def test_percentiles_boundary_validation(self, simple_test_data):
        result = calculate_percentiles(simple_test_data, [0, 100])
        assert 'error' not in result
        assert 'p0' in result and 'p100' in result
        
        result = calculate_percentiles(simple_test_data, [0.1, 99.9])
        assert 'error' not in result
        assert 'p0.1' in result and 'p99.9' in result
        
        result = calculate_percentiles(simple_test_data, [-0.1])
        assert 'error' in result
        
        result = calculate_percentiles(simple_test_data, [100.1])
        assert 'error' in result

class TestEdgeCases:
    
    def test_extreme_values_calculation_correctness(self):
        extreme_data = [1e-10, 1e10, -1e10]
        result = calculate_general_stats(extreme_data)
        
        assert result['count'] == 3
        
        expected_mean = 1e-10 / 3
        assert abs(result['mean'] - expected_mean) < 1e-9
        
        assert not math.isnan(result['mean'])
        assert not math.isinf(result['mean'])
        assert not math.isnan(result['std_dev'])
        assert not math.isinf(result['std_dev'])
        
        assert result['variance'] > 0
        assert not math.isinf(result['variance'])
    
    def test_identical_values_comprehensive(self):
        identical_data = [42] * 100
        
        general = calculate_general_stats(identical_data)
        assert general['count'] == 100
        assert general['mean'] == 42
        assert general['median'] == 42
        assert general['min'] == 42
        assert general['max'] == 42
        assert general['variance'] == 0
        assert general['std_dev'] == 0
        assert general['iqr'] == 0
        assert general['std_error'] == 0
        
        ci = calculate_confidence_interval(identical_data, 0.95)
        if 'error' not in ci:
            assert ci['mean'] == 42
            assert ci['margin_of_error'] == 0
            assert ci['lower_bound'] == 42
            assert ci['upper_bound'] == 42
        
        percentiles = calculate_percentiles(identical_data, [0, 25, 50, 75, 100])
        for p in ['p0', 'p25', 'p50', 'p75', 'p100']:
            assert percentiles[p] == 42
        
        range_result = analyze_value_based_range(identical_data, 40, 45)
        assert range_result['below_range']['count'] == 0
        assert range_result['above_range']['count'] == 0
    
    def test_two_values_comprehensive(self):
        two_values = [10, 20]
        
        general = calculate_general_stats(two_values)
        assert general['count'] == 2
        assert general['mean'] == 15
        assert general['median'] == 15
        assert general['min'] == 10
        assert general['max'] == 20
        assert general['variance'] == 50
        assert abs(general['std_dev'] - math.sqrt(50)) < 0.001
        
        ci = calculate_confidence_interval(two_values, 0.95)
        if 'error' not in ci:
            assert ci['mean'] == 15
            assert ci['margin_of_error'] > 0
            assert ci['lower_bound'] < 15
            assert ci['upper_bound'] > 15
        else:
            assert 'insufficient' in ci['error'].lower()
    
    def test_large_dataset_performance(self):
        large_data = list(range(100000))
        
        general = calculate_general_stats(large_data)
        assert general['count'] == 100000
        assert general['mean'] == 49999.5
        
        downsampled = downsample_data(large_data, target_points=1000)
        if 'error' not in downsampled:
            assert len(downsampled['downsampled_data']) <= 1000
            assert downsampled['original_points'] == 100000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])