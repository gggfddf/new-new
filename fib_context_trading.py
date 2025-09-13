"""
Fibonacci Context Trading Analysis - WHEN each level reverses vs continues
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

def analyze_trading_contexts(data, system):
    """Analyze WHEN each level reverses vs continues."""
    print(f"\n🔍 Analyzing Trading Contexts - WHEN to trade each level...")
    
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
    
    # Analyze by Fibonacci level and outcome
    trading_contexts = {}
    
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
            
        outcome = row.get('outcome_type', '')
        if outcome not in ['Reversal', 'Continuation']:
            continue
            
        if fib_level not in trading_contexts:
            trading_contexts[fib_level] = {
                'Reversal': {'contexts': [], 'count': 0},
                'Continuation': {'contexts': [], 'count': 0}
            }
        
        trading_contexts[fib_level][outcome]['count'] += 1
        
        # Store detailed context
        context = {
            'confidence': row.get('confidence', 0),
            'volatility_regime_high': row.get('volatility_regime_high', 0),
            'volatility_regime_medium': row.get('volatility_regime_medium', 0),
            'volatility_regime_low': row.get('volatility_regime_low', 0),
            'trend_direction_up': row.get('trend_direction_up', 0),
            'trend_direction_down': row.get('trend_direction_down', 0),
            'impulse_strength_atr': row.get('impulse_strength_atr', 0),
            'wick_rejection': row.get('wick_rejection', 0),
            'retest_count': row.get('retest_count', 0),
            'session_hour': row.get('session_hour', 0),
            'day_of_week': row.get('day_of_week', 0),
            'zone_strength': row.get('zone_strength', 0),
            'touch_duration': row.get('touch_duration', 0),
            'volume_ratio': row.get('volume_ratio', 0),
            'price_momentum': row.get('price_momentum', 0),
            'atr_ratio': row.get('atr_ratio', 0)
        }
        trading_contexts[fib_level][outcome]['contexts'].append(context)
    
    return trading_contexts

def show_trading_contexts(trading_contexts):
    """Show detailed trading contexts for each level."""
    print(f"\n📊 FIBONACCI TRADING CONTEXTS - WHEN TO TRADE:")
    print("=" * 100)
    
    # Sort levels by total events
    sorted_levels = sorted(trading_contexts.items(), 
                          key=lambda x: x[1]['Reversal']['count'] + x[1]['Continuation']['count'], 
                          reverse=True)
    
    for level, outcomes in sorted_levels:
        total_events = outcomes['Reversal']['count'] + outcomes['Continuation']['count']
        if total_events < 50:  # Skip levels with too few events
            continue
            
        reversal_rate = outcomes['Reversal']['count'] / total_events
        continuation_rate = outcomes['Continuation']['count'] / total_events
        
        print(f"\n🎯 LEVEL {level:.3f} - {total_events} total events")
        print(f"   📊 Reversal: {reversal_rate:.1%} ({outcomes['Reversal']['count']}) | Continuation: {continuation_rate:.1%} ({outcomes['Continuation']['count']})")
        print("=" * 80)
        
        # Analyze REVERSAL contexts
        if outcomes['Reversal']['count'] > 10:
            reversal_contexts = outcomes['Reversal']['contexts']
            print(f"\n🔄 REVERSAL CONTEXTS ({outcomes['Reversal']['count']} events):")
            
            # Volatility for reversals
            high_vol_rev = sum(1 for ctx in reversal_contexts if ctx['volatility_regime_high'])
            medium_vol_rev = sum(1 for ctx in reversal_contexts if ctx['volatility_regime_medium'])
            low_vol_rev = sum(1 for ctx in reversal_contexts if ctx['volatility_regime_low'])
            
            print(f"   🌊 Volatility: High {high_vol_rev}/{len(reversal_contexts)} ({high_vol_rev/len(reversal_contexts):.1%}) | "
                  f"Medium {medium_vol_rev}/{len(reversal_contexts)} ({medium_vol_rev/len(reversal_contexts):.1%}) | "
                  f"Low {low_vol_rev}/{len(reversal_contexts)} ({low_vol_rev/len(reversal_contexts):.1%})")
            
            # Trend for reversals
            uptrend_rev = sum(1 for ctx in reversal_contexts if ctx['trend_direction_up'])
            downtrend_rev = sum(1 for ctx in reversal_contexts if ctx['trend_direction_down'])
            
            print(f"   📈 Trend: Uptrend {uptrend_rev}/{len(reversal_contexts)} ({uptrend_rev/len(reversal_contexts):.1%}) | "
                  f"Downtrend {downtrend_rev}/{len(reversal_contexts)} ({downtrend_rev/len(reversal_contexts):.1%})")
            
            # Impulse for reversals
            avg_impulse_rev = np.mean([ctx['impulse_strength_atr'] for ctx in reversal_contexts])
            strong_impulse_rev = sum(1 for ctx in reversal_contexts if ctx['impulse_strength_atr'] > avg_impulse_rev)
            
            print(f"   ⚡ Impulse: Avg {avg_impulse_rev:.2f} | Strong {strong_impulse_rev}/{len(reversal_contexts)} ({strong_impulse_rev/len(reversal_contexts):.1%})")
            
            # Patterns for reversals
            wick_rejection_rev = sum(1 for ctx in reversal_contexts if ctx['wick_rejection'])
            retests_rev = sum(1 for ctx in reversal_contexts if ctx['retest_count'] > 0)
            
            print(f"   🔄 Patterns: Wick rejection {wick_rejection_rev}/{len(reversal_contexts)} ({wick_rejection_rev/len(reversal_contexts):.1%}) | "
                  f"Retests {retests_rev}/{len(reversal_contexts)} ({retests_rev/len(reversal_contexts):.1%})")
            
            # Timing for reversals
            avg_hour_rev = np.mean([ctx['session_hour'] for ctx in reversal_contexts])
            avg_day_rev = np.mean([ctx['day_of_week'] for ctx in reversal_contexts])
            
            print(f"   🕐 Timing: Hour {avg_hour_rev:.1f} | Day {avg_day_rev:.1f}")
            
            # Zone strength for reversals
            avg_zone_strength_rev = np.mean([ctx['zone_strength'] for ctx in reversal_contexts])
            avg_touch_duration_rev = np.mean([ctx['touch_duration'] for ctx in reversal_contexts])
            
            print(f"   🎯 Zone: Strength {avg_zone_strength_rev:.2f} | Duration {avg_touch_duration_rev:.1f}")
            
            # Volume for reversals
            avg_volume_rev = np.mean([ctx['volume_ratio'] for ctx in reversal_contexts])
            avg_momentum_rev = np.mean([ctx['price_momentum'] for ctx in reversal_contexts])
            
            print(f"   📊 Volume: Ratio {avg_volume_rev:.2f} | Momentum {avg_momentum_rev:.2f}")
        
        # Analyze CONTINUATION contexts
        if outcomes['Continuation']['count'] > 10:
            continuation_contexts = outcomes['Continuation']['contexts']
            print(f"\n➡️ CONTINUATION CONTEXTS ({outcomes['Continuation']['count']} events):")
            
            # Volatility for continuations
            high_vol_cont = sum(1 for ctx in continuation_contexts if ctx['volatility_regime_high'])
            medium_vol_cont = sum(1 for ctx in continuation_contexts if ctx['volatility_regime_medium'])
            low_vol_cont = sum(1 for ctx in continuation_contexts if ctx['volatility_regime_low'])
            
            print(f"   🌊 Volatility: High {high_vol_cont}/{len(continuation_contexts)} ({high_vol_cont/len(continuation_contexts):.1%}) | "
                  f"Medium {medium_vol_cont}/{len(continuation_contexts)} ({medium_vol_cont/len(continuation_contexts):.1%}) | "
                  f"Low {low_vol_cont}/{len(continuation_contexts)} ({low_vol_cont/len(continuation_contexts):.1%})")
            
            # Trend for continuations
            uptrend_cont = sum(1 for ctx in continuation_contexts if ctx['trend_direction_up'])
            downtrend_cont = sum(1 for ctx in continuation_contexts if ctx['trend_direction_down'])
            
            print(f"   📈 Trend: Uptrend {uptrend_cont}/{len(continuation_contexts)} ({uptrend_cont/len(continuation_contexts):.1%}) | "
                  f"Downtrend {downtrend_cont}/{len(continuation_contexts)} ({downtrend_cont/len(continuation_contexts):.1%})")
            
            # Impulse for continuations
            avg_impulse_cont = np.mean([ctx['impulse_strength_atr'] for ctx in continuation_contexts])
            strong_impulse_cont = sum(1 for ctx in continuation_contexts if ctx['impulse_strength_atr'] > avg_impulse_cont)
            
            print(f"   ⚡ Impulse: Avg {avg_impulse_cont:.2f} | Strong {strong_impulse_cont}/{len(continuation_contexts)} ({strong_impulse_cont/len(continuation_contexts):.1%})")
            
            # Patterns for continuations
            wick_rejection_cont = sum(1 for ctx in continuation_contexts if ctx['wick_rejection'])
            retests_cont = sum(1 for ctx in continuation_contexts if ctx['retest_count'] > 0)
            
            print(f"   🔄 Patterns: Wick rejection {wick_rejection_cont}/{len(continuation_contexts)} ({wick_rejection_cont/len(continuation_contexts):.1%}) | "
                  f"Retests {retests_cont}/{len(continuation_contexts)} ({retests_cont/len(continuation_contexts):.1%})")
            
            # Timing for continuations
            avg_hour_cont = np.mean([ctx['session_hour'] for ctx in continuation_contexts])
            avg_day_cont = np.mean([ctx['day_of_week'] for ctx in continuation_contexts])
            
            print(f"   🕐 Timing: Hour {avg_hour_cont:.1f} | Day {avg_day_cont:.1f}")
            
            # Zone strength for continuations
            avg_zone_strength_cont = np.mean([ctx['zone_strength'] for ctx in continuation_contexts])
            avg_touch_duration_cont = np.mean([ctx['touch_duration'] for ctx in continuation_contexts])
            
            print(f"   🎯 Zone: Strength {avg_zone_strength_cont:.2f} | Duration {avg_touch_duration_cont:.1f}")
            
            # Volume for continuations
            avg_volume_cont = np.mean([ctx['volume_ratio'] for ctx in continuation_contexts])
            avg_momentum_cont = np.mean([ctx['price_momentum'] for ctx in continuation_contexts])
            
            print(f"   📊 Volume: Ratio {avg_volume_cont:.2f} | Momentum {avg_momentum_cont:.2f}")
        
        # TRADING RECOMMENDATIONS
        print(f"\n💡 TRADING RECOMMENDATIONS FOR {level:.3f}:")
        
        if outcomes['Reversal']['count'] > 10 and outcomes['Continuation']['count'] > 10:
            # Compare contexts to find differences
            rev_contexts = outcomes['Reversal']['contexts']
            cont_contexts = outcomes['Continuation']['contexts']
            
            # Find key differences
            rev_high_vol = sum(1 for ctx in rev_contexts if ctx['volatility_regime_high']) / len(rev_contexts)
            cont_high_vol = sum(1 for ctx in cont_contexts if ctx['volatility_regime_high']) / len(cont_contexts)
            
            rev_wick = sum(1 for ctx in rev_contexts if ctx['wick_rejection']) / len(rev_contexts)
            cont_wick = sum(1 for ctx in cont_contexts if ctx['wick_rejection']) / len(cont_contexts)
            
            rev_impulse = np.mean([ctx['impulse_strength_atr'] for ctx in rev_contexts])
            cont_impulse = np.mean([ctx['impulse_strength_atr'] for ctx in cont_contexts])
            
            print(f"   🔄 TRADE REVERSAL when:")
            if rev_high_vol > cont_high_vol + 0.1:
                print(f"      ✅ High volatility ({rev_high_vol:.1%} vs {cont_high_vol:.1%})")
            if rev_wick > cont_wick + 0.1:
                print(f"      ✅ Strong wick rejection ({rev_wick:.1%} vs {cont_wick:.1%})")
            if rev_impulse > cont_impulse + 1.0:
                print(f"      ✅ Strong impulse ({rev_impulse:.1f} vs {cont_impulse:.1f})")
            
            print(f"   ➡️ TRADE CONTINUATION when:")
            if cont_high_vol > rev_high_vol + 0.1:
                print(f"      ✅ High volatility ({cont_high_vol:.1%} vs {rev_high_vol:.1%})")
            if cont_wick > rev_wick + 0.1:
                print(f"      ✅ Strong wick rejection ({cont_wick:.1%} vs {rev_wick:.1%})")
            if cont_impulse > rev_impulse + 1.0:
                print(f"      ✅ Strong impulse ({cont_impulse:.1f} vs {rev_impulse:.1f})")

def main():
    """Run Fibonacci context trading analysis."""
    print("🚀 Fibonacci Context Trading Analysis")
    print("=" * 80)
    print("🎯 WHEN each level reverses vs continues - TRADING CONTEXTS")
    print("=" * 80)
    
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
        
        # Analyze trading contexts
        trading_contexts = analyze_trading_contexts(data, system)
        
        if trading_contexts:
            show_trading_contexts(trading_contexts)
            
            print(f"\n🎉 Trading Context Analysis Complete!")
            print(f"   ✅ Know WHEN to trade each Fibonacci level")
            print(f"   ✅ Know WHAT context makes reversals vs continuations")
            print(f"   ✅ Ready for profitable trading!")
        else:
            print(f"   ❌ No trading context data available")
            
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()