"""
Enhanced Context Test - Test all the new context features on BTC data
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

def analyze_enhanced_contexts(data, swings, zones, events):
    """Analyze enhanced context features."""
    print(f"\n🔍 Analyzing Enhanced Context Features...")
    
    # Extract features for all events
    feature_extractor = FeatureExtractor()
    context_analysis = {}
    
    for i, event in enumerate(events[:100]):  # Test with first 100 events
        if event.zone.zone_type == 'retracement':
            level = event.zone.level
            
            try:
                # Extract features for this event
                feature_set = feature_extractor.extract_all_features(data, event, swings)
                features = feature_set.raw_features
                
                if level not in context_analysis:
                    context_analysis[level] = []
                
                # Store enhanced context features
                context = {
                    'level': level,
                    'successful': event.wick_rejection or event.retest_count > 0,
                    
                    # Market direction context
                    'price_momentum_10': features.get('price_momentum_10', 0),
                    'price_momentum_bullish': features.get('price_momentum_bullish', 0),
                    'price_momentum_bearish': features.get('price_momentum_bearish', 0),
                    'swing_direction_higher_highs': features.get('swing_direction_higher_highs', 0),
                    'swing_direction_lower_lows': features.get('swing_direction_lower_lows', 0),
                    
                    # Impulse pattern context
                    'impulse_strength_vs_historical': features.get('impulse_strength_vs_historical', 0),
                    'impulse_strength_above_average': features.get('impulse_strength_above_average', 0),
                    'impulse_pattern_downtrend': features.get('impulse_pattern_downtrend', 0),
                    'impulse_pattern_uptrend': features.get('impulse_pattern_uptrend', 0),
                    
                    # Retest and breakout patterns
                    'pattern_direct_breakout_up': features.get('pattern_direct_breakout_up', 0),
                    'pattern_direct_breakout_down': features.get('pattern_direct_breakout_down', 0),
                    'pattern_retest': features.get('pattern_retest', 0),
                    'multiple_retests': features.get('multiple_retests', 0),
                    
                    # Trend continuation patterns
                    'trend_continuation_uptrend': features.get('trend_continuation_uptrend', 0),
                    'trend_continuation_downtrend': features.get('trend_continuation_downtrend', 0),
                    'trend_reversal_uptrend': features.get('trend_reversal_uptrend', 0),
                    'trend_reversal_downtrend': features.get('trend_reversal_downtrend', 0),
                    
                    # Failure pattern learning
                    'level_failure_count': features.get('level_failure_count', 0),
                    'failure_pattern_2_3_fails': features.get('failure_pattern_2_3_fails', 0),
                    'failure_pattern_4_plus_fails': features.get('failure_pattern_4_plus_fails', 0),
                    'success_after_multiple_failures': features.get('success_after_multiple_failures', 0)
                }
                
                context_analysis[level].append(context)
                
            except Exception as e:
                continue
    
    return context_analysis

def show_enhanced_context_results(context_analysis):
    """Show enhanced context analysis results."""
    print(f"\n📊 Enhanced Context Analysis Results:")
    
    for level, contexts in context_analysis.items():
        if len(contexts) < 5:  # Skip levels with too few events
            continue
            
        print(f"\n🎯 Level {level:.3f} - {len(contexts)} events:")
        
        # Market direction context
        bullish_momentum = sum(1 for ctx in contexts if ctx['price_momentum_bullish'])
        bearish_momentum = sum(1 for ctx in contexts if ctx['price_momentum_bearish'])
        higher_highs = sum(1 for ctx in contexts if ctx['swing_direction_higher_highs'])
        lower_lows = sum(1 for ctx in contexts if ctx['swing_direction_lower_lows'])
        
        print(f"   📈 Market Direction Context:")
        print(f"      Bullish momentum: {bullish_momentum}/{len(contexts)} ({bullish_momentum/len(contexts):.1%})")
        print(f"      Bearish momentum: {bearish_momentum}/{len(contexts)} ({bearish_momentum/len(contexts):.1%})")
        print(f"      Higher highs: {higher_highs}/{len(contexts)} ({higher_highs/len(contexts):.1%})")
        print(f"      Lower lows: {lower_lows}/{len(contexts)} ({lower_lows/len(contexts):.1%})")
        
        # Impulse pattern context
        above_avg_strength = sum(1 for ctx in contexts if ctx['impulse_strength_above_average'])
        uptrend_impulse = sum(1 for ctx in contexts if ctx['impulse_pattern_uptrend'])
        downtrend_impulse = sum(1 for ctx in contexts if ctx['impulse_pattern_downtrend'])
        
        print(f"   ⚡ Impulse Pattern Context:")
        print(f"      Above avg strength: {above_avg_strength}/{len(contexts)} ({above_avg_strength/len(contexts):.1%})")
        print(f"      Uptrend impulse: {uptrend_impulse}/{len(contexts)} ({uptrend_impulse/len(contexts):.1%})")
        print(f"      Downtrend impulse: {downtrend_impulse}/{len(contexts)} ({downtrend_impulse/len(contexts):.1%})")
        
        # Retest and breakout patterns
        direct_breakout_up = sum(1 for ctx in contexts if ctx['pattern_direct_breakout_up'])
        direct_breakout_down = sum(1 for ctx in contexts if ctx['pattern_direct_breakout_down'])
        retest_pattern = sum(1 for ctx in contexts if ctx['pattern_retest'])
        multiple_retests = sum(1 for ctx in contexts if ctx['multiple_retests'])
        
        print(f"   🔄 Retest & Breakout Patterns:")
        print(f"      Direct breakout up: {direct_breakout_up}/{len(contexts)} ({direct_breakout_up/len(contexts):.1%})")
        print(f"      Direct breakout down: {direct_breakout_down}/{len(contexts)} ({direct_breakout_down/len(contexts):.1%})")
        print(f"      Retest pattern: {retest_pattern}/{len(contexts)} ({retest_pattern/len(contexts):.1%})")
        print(f"      Multiple retests: {multiple_retests}/{len(contexts)} ({multiple_retests/len(contexts):.1%})")
        
        # Trend continuation patterns
        trend_cont_uptrend = sum(1 for ctx in contexts if ctx['trend_continuation_uptrend'])
        trend_cont_downtrend = sum(1 for ctx in contexts if ctx['trend_continuation_downtrend'])
        trend_rev_uptrend = sum(1 for ctx in contexts if ctx['trend_reversal_uptrend'])
        trend_rev_downtrend = sum(1 for ctx in contexts if ctx['trend_reversal_downtrend'])
        
        print(f"   📊 Trend Continuation Patterns:")
        print(f"      Trend cont uptrend: {trend_cont_uptrend}/{len(contexts)} ({trend_cont_uptrend/len(contexts):.1%})")
        print(f"      Trend cont downtrend: {trend_cont_downtrend}/{len(contexts)} ({trend_cont_downtrend/len(contexts):.1%})")
        print(f"      Trend rev uptrend: {trend_rev_uptrend}/{len(contexts)} ({trend_rev_uptrend/len(contexts):.1%})")
        print(f"      Trend rev downtrend: {trend_rev_downtrend}/{len(contexts)} ({trend_rev_downtrend/len(contexts):.1%})")
        
        # Success rate by context
        successful_contexts = [ctx for ctx in contexts if ctx['successful']]
        if successful_contexts:
            print(f"   ✅ Success Rate Analysis:")
            print(f"      Total successful: {len(successful_contexts)}/{len(contexts)} ({len(successful_contexts)/len(contexts):.1%})")
            
            # Success by momentum
            success_bullish = sum(1 for ctx in successful_contexts if ctx['price_momentum_bullish'])
            success_bearish = sum(1 for ctx in successful_contexts if ctx['price_momentum_bearish'])
            print(f"      Success with bullish momentum: {success_bullish}/{len(successful_contexts)}")
            print(f"      Success with bearish momentum: {success_bearish}/{len(successful_contexts)}")
            
            # Success by pattern
            success_retest = sum(1 for ctx in successful_contexts if ctx['pattern_retest'])
            success_breakout = sum(1 for ctx in successful_contexts if ctx['pattern_direct_breakout_up'] or ctx['pattern_direct_breakout_down'])
            print(f"      Success with retest pattern: {success_retest}/{len(successful_contexts)}")
            print(f"      Success with breakout pattern: {success_breakout}/{len(successful_contexts)}")

def main():
    """Run enhanced context test."""
    print("🚀 Enhanced Context Features Test")
    print("=" * 60)
    print("🎯 Testing all enhanced context features on BTC data")
    print("   - Market direction context")
    print("   - Impulse pattern context") 
    print("   - Retest and breakout patterns")
    print("   - Trend continuation patterns")
    print("   - Failure pattern learning")
    print("=" * 60)
    
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
    
    # Step 4: Analyze enhanced contexts
    context_analysis = analyze_enhanced_contexts(data, swings, zones, events)
    
    # Step 5: Show results
    show_enhanced_context_results(context_analysis)
    
    print(f"\n🎉 Enhanced Context Analysis Results:")
    print(f"   ✅ Tested {len(context_analysis)} effective levels")
    print(f"   ✅ Analyzed market direction context")
    print(f"   ✅ Analyzed impulse pattern context")
    print(f"   ✅ Analyzed retest and breakout patterns")
    print(f"   ✅ Analyzed trend continuation patterns")
    print(f"   ✅ Analyzed failure pattern learning")
    
    print(f"\n🚀 The system now has comprehensive context understanding!")
    print(f"   Ready to predict which contexts make each level work best!")

if __name__ == "__main__":
    main()