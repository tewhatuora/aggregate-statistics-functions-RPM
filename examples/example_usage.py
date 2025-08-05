import sys
import os
import numpy as np
import pandas as pd
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from stats_calculator import (
    calculate_general_stats,
    calculate_confidence_interval,
    calculate_percentiles,
    analyze_value_based_range,
    analyze_time_based_range,
    analyze_band_based_range,
    downsample_data,
    analyze_all
)

def main():
    print("=== Statistics Calculator - Usage Example ===\n")
    
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000).tolist()  
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1000, freq='1min')
    
    print("Sample dataset: 1000 blood glucose readings (mg/dL)")
    print(f"Simulated data: Normal distribution (μ=100, σ=15)")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}\n")
    
    print("1. GENERAL STATISTICS")
    print("=" * 50)
    stats = calculate_general_stats(data)
    print(f"Count: {stats['count']:,}")
    print(f"Mean: {stats['mean']:.2f} mg/dL")
    print(f"Median: {stats['median']:.2f} mg/dL")
    print(f"Standard Deviation: {stats['std_dev']:.2f}")
    print(f"Range: {stats['min']:.1f} - {stats['max']:.1f} mg/dL")
    print(f"IQR: {stats['iqr']:.2f}")
    print(f"Skewness: {stats['skewness']:.3f} (normal ≈ 0)")
    print(f"Kurtosis: {stats['kurtosis']:.3f} (normal ≈ 0)")
    print()
    
    print("2. CONFIDENCE INTERVALS")
    print("=" * 50)
    ci_95 = calculate_confidence_interval(data, 0.95)
    ci_99 = calculate_confidence_interval(data, 0.99)
    print(f"95% CI: [{ci_95['lower_bound']:.2f}, {ci_95['upper_bound']:.2f}] mg/dL")
    print(f"99% CI: [{ci_99['lower_bound']:.2f}, {ci_99['upper_bound']:.2f}] mg/dL")
    print(f"95% Margin of Error: ±{ci_95['margin_of_error']:.2f} mg/dL")
    print(f"Standard Error: {ci_95['std_error']:.3f}")
    print()
    
    print("3. PERCENTILE ANALYSIS")
    print("=" * 50)
    percentiles = calculate_percentiles(data, [5, 10, 25, 50, 75, 90, 95])
    for p, v in percentiles.items():
        pct = p.replace('p', '')
        print(f"{pct:>2}th percentile: {v:6.2f} mg/dL")
    print()
    
    print("4. RANGE ANALYSIS (Normal: 70-140 mg/dL)")
    print("=" * 50)
    range_analysis = analyze_value_based_range(data, 70, 140)
    below = range_analysis['below_range']
    above = range_analysis['above_range']
    
    print(f"Below normal (<70): {below['count']:3d} readings ({below['percentage']:4.1f}%)")
    if below['count'] > 0:
        print(f"  Lowest value: {below['min_value']:.1f} mg/dL")
        print(f"  Max deviation: {below['max_deviation']:.1f} mg/dL below range")
    
    print(f"Above normal (>140): {above['count']:3d} readings ({above['percentage']:4.1f}%)")
    if above['count'] > 0:
        print(f"  Highest value: {above['max_value']:.1f} mg/dL")
        print(f"  Max deviation: {above['max_deviation']:.1f} mg/dL above range")
    print()
    
    print("5. TIME-BASED EPISODE ANALYSIS (Normal: 70-140 mg/dL)")
    print("=" * 50)
    time_analysis = analyze_time_based_range(data, timestamps, 70, 140, 'minutes')
    below_time = time_analysis['below_range']
    above_time = time_analysis['above_range']
    
    print(f"Hypoglycemic episodes (<70): {below_time['episode_count']}")
    if below_time['episode_count'] > 0:
        print(f"  Total time low: {below_time['total_duration']:.1f} minutes")
        print(f"  Average duration: {below_time['avg_duration']:.1f} minutes")
        print(f"  Longest episode: {below_time['max_duration']:.1f} minutes")
    
    print(f"Hyperglycemic episodes (>140): {above_time['episode_count']}")
    if above_time['episode_count'] > 0:
        print(f"  Total time high: {above_time['total_duration']:.1f} minutes") 
        print(f"  Average duration: {above_time['avg_duration']:.1f} minutes")
        print(f"  Longest episode: {above_time['max_duration']:.1f} minutes")
    print()
    
    print("6. GLYCEMIC ZONE ANALYSIS")
    print("=" * 50)
    bands = [
        ("Very Low", 0, 54),    
        ("Low", 54, 70),        
        ("Normal", 70, 140),     
        ("High", 140, 180),      
        ("Very High", 180, 400)   
    ]
    
    band_stats = analyze_band_based_range(data, bands)
    for band_name, stats_data in band_stats.items():
        print(f"{band_name:10}: {stats_data['count']:3d} readings ({stats_data['percentage']:4.1f}%)")
    print()
    
    print("7. DATA DOWNSAMPLING (1000 → 50 points)")
    print("=" * 50)  
    downsampled = downsample_data(data, timestamps, 50)
    if 'error' not in downsampled:
        print(f"Original points: {downsampled['original_points']:,}")
        print(f"Target points: {downsampled['target_points']}")
        print(f"Actual downsampled: {len(downsampled['downsampled_data'])}")
        print(f"Compression ratio: {downsampled['original_points'] / len(downsampled['downsampled_data']):.1f}:1")
        print(f"Sample downsampled values: {[f'{x:.1f}' for x in downsampled['downsampled_data'][:5]]}...")
    else:
        print(f"Downsampling error: {downsampled['error']}")
    print()
    
    print("8. ALL-IN-ONE COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    all_results = analyze_all(data, confidence_level=0.95, percentiles=[25, 50, 75, 95])
    print("✓ General statistics calculated")
    print("✓ 95% confidence interval calculated") 
    print("✓ Key percentiles calculated")    
    print(f"Quick summary:")
    print(f"  Mean: {all_results['general']['mean']:.2f} mg/dL")
    print(f"  95% CI: [{all_results['confidence_interval']['lower_bound']:.2f}, "
          f"{all_results['confidence_interval']['upper_bound']:.2f}]")
    print(f"  IQR: {all_results['percentiles']['p25']:.1f} - {all_results['percentiles']['p75']:.1f} mg/dL")
    print()
    
    print("9. JSON SERIALIZATION & EXPORT")
    print("=" * 50)
    export_data = {
        'metadata': {
            'dataset_size': len(data),
            'time_range': f"{timestamps[0]} to {timestamps[-1]}",
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'general_stats': stats,
        'confidence_intervals': {
            '95%': ci_95,
            '99%': ci_99
        },
        'percentiles': percentiles,
        'range_analysis': range_analysis,
        'band_analysis': band_stats
    }
    
    json_str = json.dumps(export_data, default=str, indent=2)
    print(f"✓ All results JSON-serialized successfully")
    print(f"✓ JSON output size: {len(json_str):,} characters")
    print()
    
    print("\n📊 OUTPUT EXAMPLE:")
    print(f"• Patient shows mean glucose of {stats['mean']:.1f} mg/dL")
    print(f"• {(100 - below['percentage'] - above['percentage']):.1f}% of readings in normal range (70-140)")
    if below_time['episode_count'] > 0:
        print(f"• {below_time['episode_count']} hypoglycemic episodes detected")
    if above_time['episode_count'] > 0:
        print(f"• {above_time['episode_count']} hyperglycemic episodes detected")
    
if __name__ == "__main__":
    main()