# Statistics Calculator

A collection of functions for statistical analysis of numerical data with support for confidence intervals, percentiles, range analysis, and time-based episode detection.

## Requirements
- Python 3.10 - 3.12
- See `requirements.txt` for package dependencies

## Quick Setup

```bash
git clone https://github.com/tewhatuora/aggregate-statistics-functions-RPM.git
cd aggregate-statistics-functions-RPM
pip install -r requirements.txt

# For development/testing
pip install -r requirements-dev.txt
```

## Quick Start

```python
from stats_calculator import calculate_general_stats, calculate_confidence_interval, analyze_all
import numpy as np

# Your data (example with 1000 points)
np.random.seed(42)
data = np.random.normal(100, 15, 1000).tolist()

# Get comprehensive statistics
stats = calculate_general_stats(data)
print(f"Mean: {stats['mean']:.2f} ± {stats['std_dev']:.2f}")

# Get confidence interval
ci = calculate_confidence_interval(data, 0.95)
print(f"95% CI: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]")

# All-in-one analysis
results = analyze_all(data, confidence_level=0.95, percentiles=[25, 50, 75, 95])
```

## Available Functions

### General Statistics
```python
from stats_calculator import calculate_general_stats

stats = calculate_general_stats(data)
# Returns: count, mean, median, min, max, iqr, variance, std_dev, std_error, skewness, kurtosis
```

### Confidence Intervals
```python
from stats_calculator import calculate_confidence_interval

ci = calculate_confidence_interval(data, 0.95)  # 95% confidence
# Returns: confidence_level, mean, std_error, z_score, margin_of_error, lower_bound, upper_bound
```

### Percentiles
```python
from stats_calculator import calculate_percentiles

percentiles = calculate_percentiles(data, [5, 25, 50, 75, 95])
# Returns: {"p5": 73.21, "p25": 89.87, "p50": 99.34, "p75": 109.23, "p95": 125.18}
```

### Value-Based Range Analysis
```python
from stats_calculator import analyze_value_based_range

# Analyze values outside normal range (85-115)
range_analysis = analyze_value_based_range(data, 85, 115)
# Returns: count, percentage, and deviation statistics for values outside range
print(f"Below range: {range_analysis['below_range']['count']} ({range_analysis['below_range']['percentage']:.1f}%)")
```

### Time-Based Episode Analysis
```python
import pandas as pd
from stats_calculator import analyze_time_based_range

timestamps = pd.date_range('2024-01-01', periods=len(data), freq='1min')
episodes = analyze_time_based_range(data, timestamps, 85, 115, 'minutes')
# Returns: episode count, duration statistics, peak/trough values
print(f"Episodes above range: {episodes['above_range']['episode_count']}")
print(f"Total time above: {episodes['above_range']['total_duration']:.1f} minutes")
```

### Band/Zone Analysis
```python
from stats_calculator import analyze_band_based_range

bands = [
    ("Very Low", 0, 70),
    ("Low", 70, 85), 
    ("Normal", 85, 115),
    ("High", 115, 130),
    ("Very High", 130, 200)
]
band_stats = analyze_band_based_range(data, bands)
# Returns: count and percentage for each band
for band_name, stats in band_stats.items():
    print(f"{band_name}: {stats['count']} values ({stats['percentage']:.1f}%)")
```

### Data Downsampling
```python
from stats_calculator import downsample_data

# Reduce large dataset using LTTB algorithm
downsampled = downsample_data(data, timestamps, target_points=50)
print(f"Reduced {downsampled['original_points']} to {len(downsampled['downsampled_data'])} points")
```

### All-in-One Analysis
```python
from stats_calculator import analyze_all

# Get all analysis in one call
results = analyze_all(data, confidence_level=0.95, percentiles=[25, 50, 75, 95])
# Returns: general stats, confidence interval, and percentiles combined
```

## Usage Example

See `examples/example_usage.py`:

```bash
python examples/example_usage.py
```

**Output includes:**
- General statistics (mean, std dev, skewness, etc.)
- Confidence intervals (95%, 99%)
- Percentile analysis
- Range analysis with deviation statistics
- Time-based episode detection
- Band/zone classification
- Data downsampling demonstration
- JSON serialization example

## Running Tests

```bash
python -m pytest tests/     
```

## JSON Serialization

All function outputs are JSON-serializable:

```python
import json
results = calculate_general_stats(data)
json_output = json.dumps(results, default=str)
```

## Project Structure

```
stats-calculator/
├── stats_calculator/           
│   ├── __init__.py              
│   └── core.py                
├── tests/                     
│   ├── __init__.py
│   └── test_stats.py          
├── examples/                  
│   └── example_usage.py 
├── requirements.txt           
├── requirements-dev.txt       
└── README.md                  
```
