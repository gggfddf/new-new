"""
Fibonacci Level Analysis - Detailed analysis of which levels work best
"""

import pandas as pd
import numpy as np
import warnings
import sys
import time

# Add current directory to path
sys.path.append('/workspace')

from fib_ml_system import FibMLSystem, SystemConfig

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

def analyze_fib_levels(data, system):
    """Analyze which Fibonacci levels work best."""
    print(f"\n🔍 Analyzing Fibonacci Level Effectiveness...")
    
    # Get training data
    training_data = system.last_training_data
    if not training_data:
        print("   ❌ No training data available")
        return
    
    events = training_data['events']
    features_df = training_data['features_df']
    labels_df = training_data['labels_df']
    
    # Merge features and labels
    merged_df = pd.merge(features_df, labels_df, on='event_id', how='inner')
    
    # Analyze by Fibonacci level
    level_analysis = {}
    
    for _, row in merged_df.iterrows():
        # Get Fibonacci level from context tags or features
        fib_level = None
        if 'fib_level' in row:
            fib_level = row['fib_level']
        elif 'context_tags' in row:
            context_tags = str(row['context_tags'])
            # Extract fib level from context tags
            import re
            fib_match = re.search(r'fib_([0-9.]+)', context_tags)
            if fib_match:
                fib_level = float(fib_match.group(1))
        
        if fib_level is None:
            continue
            
        if fib_level not in level_analysis:
            level_analysis[fib_level] = {
                'total_events': 0,
                'successful_events': 0,
                'reversal_events': 0,
                'continuation_events': 0,
                'breakout_events': 0,
                'contexts': []
            }
        
        level_analysis[fib_level]['total_events'] += 1
        
        # Count outcomes
        outcome = row.get('outcome_type', '')
        if outcome == 'Reversal':
            level_analysis[fib_level]['reversal_events'] += 1
        elif outcome == 'Continuation':
            level_analysis[fib_level]['continuation_events'] += 1
        elif outcome == 'Breakout':
            level_analysis[fib_level]['breakout_events'] += 1
        
        # Count successful events (high confidence)
        confidence = row.get('confidence', 0)
        if confidence > 0.7:
            level_analysis[fib_level]['successful_events'] += 1
        
        # Store context information
        context = {
            'confidence': confidence,
            'outcome': outcome,
            'volatility_regime_high': row.get('volatility_regime_high', 0),
            'volatility_regime_medium': row.get('volatility_regime_medium', 0),
            'volatility_regime_low': row.get('volatility_regime_low', 0),
            'trend_direction_up': row.get('trend_direction_up', 0),
            'trend_direction_down': row.get('trend_direction_down', 0),
            'impulse_strength_atr': row.get('impulse_strength_atr', 0),
            'wick_rejection': row.get('wick_rejection', 0),
            'retest_count': row.get('retest_count', 0),
            'session_hour': row.get('session_hour', 0),
            'day_of_week': row.get('day_of_week', 0)
        }
        level_analysis[fib_level]['contexts'].append(context)
    
    return level_analysis

def show_level_analysis(level_analysis):
    """Show detailed level analysis."""
    print(f"\n📊 Fibonacci Level Effectiveness Analysis:")
    print("=" * 80)
    
    # Sort levels by total events
    sorted_levels = sorted(level_analysis.items(), key=lambda x: x[1]['total_events'], reverse=True)
    
    for level, data in sorted_levels:
        if data['total_events'] < 10:  # Skip levels with too few events
            continue
            
        success_rate = data['successful_events'] / data['total_events'] if data['total_events'] > 0 else 0
        reversal_rate = data['reversal_events'] / data['total_events'] if data['total_events'] > 0 else 0
        continuation_rate = data['continuation_events'] / data['total_events'] if data['total_events'] > 0 else 0
        breakout_rate = data['breakout_events'] / data['total_events'] if data['total_events'] > 0 else 0
        
        print(f"\n🎯 Level {level:.3f} - {data['total_events']} events:")
        print(f"   📈 Success Rate: {success_rate:.1%} ({data['successful_events']}/{data['total_events']})")
        print(f"   📊 Outcome Distribution:")
        print(f"      Reversal: {reversal_rate:.1%} ({data['reversal_events']})")
        print(f"      Continuation: {continuation_rate:.1%} ({data['continuation_events']})")
        print(f"      Breakout: {breakout_rate:.1%} ({data['breakout_events']})")
        
        # Analyze contexts
        contexts = data['contexts']
        if contexts:
            # Volatility context
            high_vol = sum(1 for ctx in contexts if ctx['volatility_regime_high'])
            medium_vol = sum(1 for ctx in contexts if ctx['volatility_regime_medium'])
            low_vol = sum(1 for ctx in contexts if ctx['volatility_regime_low'])
            
            print(f"   🌊 Volatility Context:")
            print(f"      High: {high_vol}/{len(contexts)} ({high_vol/len(contexts):.1%})")
            print(f"      Medium: {medium_vol}/{len(contexts)} ({medium_vol/len(contexts):.1%})")
            print(f"      Low: {low_vol}/{len(contexts)} ({low_vol/len(contexts):.1%})")
            
            # Trend context
            uptrend = sum(1 for ctx in contexts if ctx['trend_direction_up'])
            downtrend = sum(1 for ctx in contexts if ctx['trend_direction_down'])
            
            print(f"   📈 Trend Context:")
            print(f"      Uptrend: {uptrend}/{len(contexts)} ({uptrend/len(contexts):.1%})")
            print(f"      Downtrend: {downtrend}/{len(contexts)} ({downtrend/len(contexts):.1%})")
            
            # Impulse context
            avg_impulse = np.mean([ctx['impulse_strength_atr'] for ctx in contexts])
            strong_impulse = sum(1 for ctx in contexts if ctx['impulse_strength_atr'] > avg_impulse)
            
            print(f"   ⚡ Impulse Context:")
            print(f"      Average strength: {avg_impulse:.2f}")
            print(f"      Strong impulses: {strong_impulse}/{len(contexts)} ({strong_impulse/len(contexts):.1%})")
            
            # Wick rejection and retests
            wick_rejection = sum(1 for ctx in contexts if ctx['wick_rejection'])
            retests = sum(1 for ctx in contexts if ctx['retest_count'] > 0)
            
            print(f"   🔄 Pattern Context:")
            print(f"      Wick rejection: {wick_rejection}/{len(contexts)} ({wick_rejection/len(contexts):.1%})")
            print(f"      Retests: {retests}/{len(contexts)} ({retests/len(contexts):.1%})")
            
            # Timing context
            avg_hour = np.mean([ctx['session_hour'] for ctx in contexts])
            avg_day = np.mean([ctx['day_of_week'] for ctx in contexts])
            
            print(f"   🕐 Timing Context:")
            print(f"      Average hour: {avg_hour:.1f}")
            print(f"      Average day: {avg_day:.1f}")

def main():
    """Run Fibonacci level analysis."""
    print("🚀 Fibonacci Level Analysis")
    print("=" * 60)
    print("🎯 Analyzing which Fibonacci levels work best and under which contexts")
    print("=" * 60)
    
    # Load BTC data
    data = load_btc_data()
    
    # Use last 5000 bars for analysis
    data = data.tail(5000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Create system with traditional Fibonacci levels
    config = SystemConfig(
        swing_window=30,
        min_swing_strength=0.5,
        min_swing_size=0.002,
        learn_levels=False,  # Use traditional levels
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.1,
        min_zone_size=0.001,
        min_touch_duration=2,
        max_event_duration=80,
        wick_rejection_threshold=0.3,
        enable_enhanced_context=True,
        context_momentum_period=15,
        context_historical_period=60,
        context_failure_tracking=True,
        atr_period=20,
        trend_period=30,
        volatility_period=30,
        lookforward_window=40,
        reversal_threshold=2.5,
        continuation_threshold=1.5,
        atr_multiplier=1.2,
        min_confidence=0.7,
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_splits=3,
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=2.0,
        commission_bps=1.0
    )
    
    # Initialize and train system
    print(f"\n🔧 Training system...")
    system = FibMLSystem(config)
    
    try:
        model_results = system.train(data, save_models=False)
        print(f"   ✅ Training completed!")
        
        # Analyze Fibonacci levels
        level_analysis = analyze_fib_levels(data, system)
        
        if level_analysis:
            show_level_analysis(level_analysis)
            
            print(f"\n🎉 Fibonacci Level Analysis Complete!")
            print(f"   ✅ Analyzed effectiveness of traditional Fibonacci levels")
            print(f"   ✅ Identified which contexts make each level work best")
            print(f"   ✅ Ready to optimize trading strategies!")
        else:
            print(f"   ❌ No level analysis data available")
            
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()