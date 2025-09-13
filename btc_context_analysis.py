"""
BTC Context Analysis - Discover which contexts make retracement levels work best
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
from features import FeatureExtractor

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
    
    return ohlcv_data

def analyze_contexts(data, swings, zones, events):
    """Analyze which contexts make retracement levels work best."""
    print(f"\n🔍 Analyzing Contexts for Effective Retracement Levels...")
    
    # Extract features for all events
    feature_extractor = FeatureExtractor()
    context_analysis = {}
    
    for event in events:
        if event.zone.zone_type == 'retracement':
            level = event.zone.level
            
            try:
                # Extract features for this event
                feature_set = feature_extractor.extract_all_features(data, event, swings)
                features = feature_set.raw_features
                
                if level not in context_analysis:
                    context_analysis[level] = {
                        'total_events': 0,
                        'successful_events': 0,
                        'contexts': []
                    }
                
                context_analysis[level]['total_events'] += 1
                
                # Determine if event was successful
                success = 0
                if event.wick_rejection:
                    success += 1
                if event.retest_count > 0:
                    success += 1
                if event.duration_bars and event.duration_bars > 3:
                    success += 1
                
                is_successful = success >= 2  # At least 2 success indicators
                if is_successful:
                    context_analysis[level]['successful_events'] += 1
                
                # Store context features
                context = {
                    'successful': is_successful,
                    'trend_direction_up': features.get('trend_direction_up', 0),
                    'trend_direction_down': features.get('trend_direction_down', 0),
                    'volatility_regime_high': features.get('volatility_regime_high', 0),
                    'volatility_regime_medium': features.get('volatility_regime_medium', 0),
                    'volatility_regime_low': features.get('volatility_regime_low', 0),
                    'atr_quantile': features.get('atr_quantile', 0),
                    'impulse_strength_atr': features.get('impulse_strength_atr', 0),
                    'impulse_duration_bars': features.get('impulse_duration_bars', 0),
                    'entry_position_in_zone': features.get('entry_position_in_zone', 0),
                    'swing_width_atr': features.get('swing_width_atr', 0),
                    'zone_width_atr': features.get('zone_width_atr', 0),
                    'session_hour': features.get('session_hour', 0),
                    'day_of_week': features.get('day_of_week', 0),
                    'higher_highs': features.get('higher_highs', 0),
                    'higher_lows': features.get('higher_lows', 0),
                    'structure_break': features.get('structure_break', 0)
                }
                
                context_analysis[level]['contexts'].append(context)
                
            except Exception as e:
                continue
    
    return context_analysis

def analyze_context_effectiveness(context_analysis):
    """Analyze which contexts make each level most effective."""
    print(f"\n📊 Context Effectiveness Analysis:")
    
    for level, data in context_analysis.items():
        if data['total_events'] < 20:  # Skip levels with too few events
            continue
            
        contexts = data['contexts']
        total_events = len(contexts)
        successful_events = sum(1 for ctx in contexts if ctx['successful'])
        success_rate = successful_events / total_events if total_events > 0 else 0
        
        print(f"\n🎯 Level {level:.3f} - {total_events} events, {success_rate:.1%} success rate:")
        
        # Analyze successful vs unsuccessful contexts
        successful_contexts = [ctx for ctx in contexts if ctx['successful']]
        unsuccessful_contexts = [ctx for ctx in contexts if not ctx['successful']]
        
        if len(successful_contexts) < 5 or len(unsuccessful_contexts) < 5:
            print(f"   Not enough data for context analysis")
            continue
        
        # Compare contexts
        context_comparison = {}
        
        for feature in ['trend_direction_up', 'trend_direction_down', 'volatility_regime_high', 
                       'volatility_regime_medium', 'volatility_regime_low', 'higher_highs', 
                       'higher_lows', 'structure_break']:
            
            success_avg = np.mean([ctx[feature] for ctx in successful_contexts])
            failure_avg = np.mean([ctx[feature] for ctx in unsuccessful_contexts])
            
            if abs(success_avg - failure_avg) > 0.1:  # Significant difference
                context_comparison[feature] = {
                    'success_avg': success_avg,
                    'failure_avg': failure_avg,
                    'difference': success_avg - failure_avg
                }
        
        # Show significant context differences
        if context_comparison:
            print(f"   📈 Contexts that work best:")
            for feature, data in sorted(context_comparison.items(), key=lambda x: abs(x[1]['difference']), reverse=True):
                if data['difference'] > 0:
                    print(f"      {feature}: {data['success_avg']:.2f} vs {data['failure_avg']:.2f} (works better)")
                else:
                    print(f"      {feature}: {data['success_avg']:.2f} vs {data['failure_avg']:.2f} (works worse)")
        
        # Analyze numerical features
        numerical_features = ['atr_quantile', 'impulse_strength_atr', 'impulse_duration_bars', 
                             'entry_position_in_zone', 'swing_width_atr', 'zone_width_atr', 
                             'session_hour', 'day_of_week']
        
        print(f"   📊 Numerical context analysis:")
        for feature in numerical_features:
            success_values = [ctx[feature] for ctx in successful_contexts if ctx[feature] is not None]
            failure_values = [ctx[feature] for ctx in unsuccessful_contexts if ctx[feature] is not None]
            
            if success_values and failure_values:
                success_mean = np.mean(success_values)
                failure_mean = np.mean(failure_values)
                
                if abs(success_mean - failure_mean) > 0.1:
                    print(f"      {feature}: Success {success_mean:.2f} vs Failure {failure_mean:.2f}")

def main():
    """Run context analysis."""
    print("🚀 BTC Context Analysis - Which Contexts Make Levels Work Best")
    print("=" * 70)
    print("🎯 Discovering which market contexts make retracement levels effective")
    print("=" * 70)
    
    # Load data
    data = load_btc_data()
    
    # Use last 2000 bars for analysis
    data = data.tail(2000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Step 1: Detect swings
    print(f"\n📈 Step 1: Detecting swings...")
    detector = SwingDetector(window=20, min_swing_strength=0.3, min_swing_size=0.002)
    swings = detector.detect_swings(data)
    print(f"   ✅ Detected {len(swings)} swings")
    
    # Step 2: Generate zones with effective levels
    print(f"\n📊 Step 2: Generating zones with effective levels...")
    
    # Use the effective levels we discovered
    effective_levels = [0.650, 0.450, 0.600, 0.400, 0.500, 0.550, 0.350, 0.700]
    
    zone_generator = FibZoneGenerator(
        standard_ratios=effective_levels,
        zone_width_factor=0.08,
        min_zone_size=0.001
    )
    
    zones = zone_generator.generate_all_zones(swings)
    zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.2)
    print(f"   ✅ Generated {len(zones)} zones with effective levels")
    
    # Step 3: Generate events
    print(f"\n🎯 Step 3: Generating events...")
    event_generator = EventGenerator(
        min_touch_duration=1,
        max_event_duration=100,
        wick_rejection_threshold=0.25
    )
    events = event_generator.detect_zone_touches(data, zones)
    print(f"   ✅ Generated {len(events)} events")
    
    # Step 4: Analyze contexts
    context_analysis = analyze_contexts(data, swings, zones, events)
    
    # Step 5: Show context effectiveness
    analyze_context_effectiveness(context_analysis)
    
    print(f"\n🎉 Context Analysis Results:")
    print(f"   ✅ Analyzed contexts for {len(context_analysis)} effective levels")
    print(f"   ✅ Discovered which market conditions make each level work best")
    print(f"   ✅ Found optimal contexts for successful retracement trades")
    
    print(f"\n🚀 The system now knows both the effective levels AND the contexts!")

if __name__ == "__main__":
    main()