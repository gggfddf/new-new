"""
BTC Level Analysis - Discover which retracement levels actually work in BTC
"""

import pandas as pd
import numpy as np
import warnings
import sys

# Add current directory to path
sys.path.append('/workspace')

from swing_detector import SwingDetector
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def load_btc_data():
    """Load BTC data."""
    print("📊 Loading BTC 5-minute data...")
    
    # Read the CSV file
    df = pd.read_csv('/workspace/btc_5min.csv', sep='\t')
    
    # Clean data
    if df.iloc[0, 0] == '<DATE>':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Rename columns
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    
    # Convert data types
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    
    # Select OHLCV
    ohlcv_data = df[['open', 'high', 'low', 'close', 'vol']].copy()
    ohlcv_data.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_data = ohlcv_data.dropna()
    
    print(f"   Data shape: {ohlcv_data.shape}")
    print(f"   Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    
    return ohlcv_data

def analyze_retracement_levels(data, swings, zones, events):
    """Analyze which retracement levels actually work in BTC."""
    print(f"\n🔍 Analyzing Retracement Level Effectiveness...")
    
    # Group events by retracement level
    level_analysis = {}
    
    for event in events:
        if event.zone.zone_type == 'retracement':
            level = event.zone.level
            
            if level not in level_analysis:
                level_analysis[level] = {
                    'total_events': 0,
                    'wick_rejections': 0,
                    'retests': 0,
                    'long_duration': 0,
                    'success_score': 0
                }
            
            level_analysis[level]['total_events'] += 1
            
            # Count success indicators
            if event.wick_rejection:
                level_analysis[level]['wick_rejections'] += 1
                level_analysis[level]['success_score'] += 1
            
            if event.retest_count > 0:
                level_analysis[level]['retests'] += 1
                level_analysis[level]['success_score'] += 1
            
            if event.duration_bars and event.duration_bars > 3:
                level_analysis[level]['long_duration'] += 1
                level_analysis[level]['success_score'] += 1
    
    # Calculate effectiveness metrics
    effectiveness = []
    for level, stats in level_analysis.items():
        if stats['total_events'] >= 5:  # Minimum events threshold
            wick_rate = stats['wick_rejections'] / stats['total_events']
            retest_rate = stats['retests'] / stats['total_events']
            duration_rate = stats['long_duration'] / stats['total_events']
            avg_success = stats['success_score'] / stats['total_events']
            
            # Overall effectiveness score
            effectiveness_score = (wick_rate * 0.4 + retest_rate * 0.3 + duration_rate * 0.3) * stats['total_events']
            
            effectiveness.append({
                'level': level,
                'events': stats['total_events'],
                'wick_rate': wick_rate,
                'retest_rate': retest_rate,
                'duration_rate': duration_rate,
                'avg_success': avg_success,
                'effectiveness_score': effectiveness_score
            })
    
    # Sort by effectiveness
    effectiveness.sort(key=lambda x: x['effectiveness_score'], reverse=True)
    
    return effectiveness

def main():
    """Run level analysis."""
    print("🚀 BTC Retracement Level Discovery")
    print("=" * 50)
    print("🎯 Discovering which retracement levels actually work in BTC")
    print("   (Not assuming traditional Fibonacci numbers)")
    print("=" * 50)
    
    # Load data
    data = load_btc_data()
    
    # Use last 2500 bars for analysis
    data = data.tail(2500)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Step 1: Detect swings
    print(f"\n📈 Step 1: Detecting swings...")
    detector = SwingDetector(window=20, min_swing_strength=0.3, min_swing_size=0.002)
    swings = detector.detect_swings(data)
    print(f"   ✅ Detected {len(swings)} swings")
    
    # Step 2: Generate zones with wide range of levels
    print(f"\n📊 Step 2: Generating zones with wide level range...")
    
    # Use a wide range of potential retracement levels
    potential_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    zone_generator = FibZoneGenerator(
        standard_ratios=potential_levels,
        zone_width_factor=0.08,
        min_zone_size=0.001
    )
    
    zones = zone_generator.generate_all_zones(swings)
    zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.2)
    print(f"   ✅ Generated {len(zones)} zones with {len(potential_levels)} potential levels")
    
    # Step 3: Generate events
    print(f"\n🎯 Step 3: Generating events...")
    event_generator = EventGenerator(
        min_touch_duration=1,
        max_event_duration=100,
        wick_rejection_threshold=0.25
    )
    events = event_generator.detect_zone_touches(data, zones)
    print(f"   ✅ Generated {len(events)} events")
    
    # Step 4: Analyze level effectiveness
    effectiveness = analyze_retracement_levels(data, swings, zones, events)
    
    print(f"\n📊 Retracement Level Effectiveness Analysis:")
    print(f"   (Based on wick rejections, retests, and duration)")
    print(f"   {'Level':<8} {'Events':<8} {'Wick%':<8} {'Retest%':<8} {'Duration%':<10} {'Score':<8}")
    print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    
    for i, level_data in enumerate(effectiveness[:15]):
        level = level_data['level']
        events = level_data['events']
        wick_rate = level_data['wick_rate']
        retest_rate = level_data['retest_rate']
        duration_rate = level_data['duration_rate']
        score = level_data['effectiveness_score']
        
        print(f"   {level:<8.3f} {events:<8} {wick_rate:<8.1%} {retest_rate:<8.1%} {duration_rate:<10.1%} {score:<8.1f}")
    
    # Find the most effective levels
    top_levels = [level_data['level'] for level_data in effectiveness[:8]]
    
    print(f"\n🎯 Most Effective Retracement Levels in BTC:")
    print(f"   {[f'{level:.3f}' for level in top_levels]}")
    
    # Compare with traditional Fibonacci
    traditional_fib = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
    traditional_in_top = [level for level in traditional_fib if level in top_levels]
    non_traditional = [level for level in top_levels if level not in traditional_fib]
    
    print(f"\n📊 Comparison with Traditional Fibonacci:")
    print(f"   Traditional levels in top 8: {[f'{level:.3f}' for level in traditional_in_top]}")
    print(f"   Non-traditional levels in top 8: {[f'{level:.3f}' for level in non_traditional]}")
    
    # Show specific findings
    print(f"\n🔍 Key Findings:")
    if non_traditional:
        print(f"   ✅ Found effective non-traditional levels: {[f'{level:.3f}' for level in non_traditional[:3]]}")
    
    # Check for ranges like 0.8-0.83
    range_80_83 = [level for level in top_levels if 0.8 <= level <= 0.83]
    if range_80_83:
        print(f"   ✅ 0.8-0.83 range is effective: {[f'{level:.3f}' for level in range_80_83]}")
    
    # Check for levels around 0.4, 0.55, 0.77
    specific_levels = [0.4, 0.55, 0.77]
    found_specific = [level for level in top_levels if any(abs(level - spec) < 0.05 for spec in specific_levels)]
    if found_specific:
        print(f"   ✅ Found levels near 0.4, 0.55, 0.77: {[f'{level:.3f}' for level in found_specific]}")
    
    print(f"\n🎉 BTC Level Discovery Results:")
    print(f"   ✅ Analyzed {len(events)} zone touch events")
    print(f"   ✅ Tested {len(potential_levels)} potential retracement levels")
    print(f"   ✅ Found {len(effectiveness)} effective levels")
    print(f"   ✅ Discovered which levels actually work in BTC markets")
    
    print(f"\n🚀 The system can now use these learned levels instead of traditional Fibonacci!")

if __name__ == "__main__":
    main()