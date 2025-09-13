"""
Fibonacci Trading Signals - Clear trading rules for each level
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

def analyze_trading_signals(data, system):
    """Analyze trading signals for each Fibonacci level."""
    print(f"\n🔍 Analyzing Trading Signals - WHEN to trade each level...")
    
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
    trading_signals = {}
    
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
            
        if fib_level not in trading_signals:
            trading_signals[fib_level] = {
                'Reversal': {'contexts': [], 'count': 0},
                'Continuation': {'contexts': [], 'count': 0}
            }
        
        trading_signals[fib_level][outcome]['count'] += 1
        
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
        trading_signals[fib_level][outcome]['contexts'].append(context)
    
    return trading_signals

def show_trading_signals(trading_signals):
    """Show clear trading signals for each level."""
    print(f"\n📊 FIBONACCI TRADING SIGNALS - WHEN TO TRADE:")
    print("=" * 100)
    
    # Sort levels by total events
    sorted_levels = sorted(trading_signals.items(), 
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
        
        # Analyze contexts
        if outcomes['Reversal']['count'] > 10 and outcomes['Continuation']['count'] > 10:
            rev_contexts = outcomes['Reversal']['contexts']
            cont_contexts = outcomes['Continuation']['contexts']
            
            # Calculate key metrics
            rev_high_vol = sum(1 for ctx in rev_contexts if ctx['volatility_regime_high']) / len(rev_contexts)
            cont_high_vol = sum(1 for ctx in cont_contexts if ctx['volatility_regime_high']) / len(cont_contexts)
            
            rev_medium_vol = sum(1 for ctx in rev_contexts if ctx['volatility_regime_medium']) / len(rev_contexts)
            cont_medium_vol = sum(1 for ctx in cont_contexts if ctx['volatility_regime_medium']) / len(cont_contexts)
            
            rev_low_vol = sum(1 for ctx in rev_contexts if ctx['volatility_regime_low']) / len(rev_contexts)
            cont_low_vol = sum(1 for ctx in cont_contexts if ctx['volatility_regime_low']) / len(cont_contexts)
            
            rev_uptrend = sum(1 for ctx in rev_contexts if ctx['trend_direction_up']) / len(rev_contexts)
            cont_uptrend = sum(1 for ctx in cont_contexts if ctx['trend_direction_up']) / len(cont_contexts)
            
            rev_downtrend = sum(1 for ctx in rev_contexts if ctx['trend_direction_down']) / len(rev_contexts)
            cont_downtrend = sum(1 for ctx in cont_contexts if ctx['trend_direction_down']) / len(cont_contexts)
            
            rev_wick = sum(1 for ctx in rev_contexts if ctx['wick_rejection']) / len(rev_contexts)
            cont_wick = sum(1 for ctx in cont_contexts if ctx['wick_rejection']) / len(cont_contexts)
            
            rev_impulse = np.mean([ctx['impulse_strength_atr'] for ctx in rev_contexts])
            cont_impulse = np.mean([ctx['impulse_strength_atr'] for ctx in cont_contexts])
            
            rev_hour = np.mean([ctx['session_hour'] for ctx in rev_contexts])
            cont_hour = np.mean([ctx['session_hour'] for ctx in cont_contexts])
            
            rev_zone_strength = np.mean([ctx['zone_strength'] for ctx in rev_contexts])
            cont_zone_strength = np.mean([ctx['zone_strength'] for ctx in cont_contexts])
            
            # Show key differences
            print(f"\n🔍 KEY DIFFERENCES:")
            print(f"   🌊 Volatility:")
            print(f"      High: Reversal {rev_high_vol:.1%} vs Continuation {cont_high_vol:.1%} (diff: {rev_high_vol-cont_high_vol:+.1%})")
            print(f"      Medium: Reversal {rev_medium_vol:.1%} vs Continuation {cont_medium_vol:.1%} (diff: {rev_medium_vol-cont_medium_vol:+.1%})")
            print(f"      Low: Reversal {rev_low_vol:.1%} vs Continuation {cont_low_vol:.1%} (diff: {rev_low_vol-cont_low_vol:+.1%})")
            
            print(f"   📈 Trend:")
            print(f"      Uptrend: Reversal {rev_uptrend:.1%} vs Continuation {cont_uptrend:.1%} (diff: {rev_uptrend-cont_uptrend:+.1%})")
            print(f"      Downtrend: Reversal {rev_downtrend:.1%} vs Continuation {cont_downtrend:.1%} (diff: {rev_downtrend-cont_downtrend:+.1%})")
            
            print(f"   ⚡ Impulse: Reversal {rev_impulse:.2f} vs Continuation {cont_impulse:.2f} (diff: {rev_impulse-cont_impulse:+.2f})")
            print(f"   🔄 Wick Rejection: Reversal {rev_wick:.1%} vs Continuation {cont_wick:.1%} (diff: {rev_wick-cont_wick:+.1%})")
            print(f"   🕐 Timing: Reversal {rev_hour:.1f}h vs Continuation {cont_hour:.1f}h (diff: {rev_hour-cont_hour:+.1f}h)")
            print(f"   🎯 Zone Strength: Reversal {rev_zone_strength:.2f} vs Continuation {cont_zone_strength:.2f} (diff: {rev_zone_strength-cont_zone_strength:+.2f})")
            
            # Generate trading signals
            print(f"\n💡 TRADING SIGNALS FOR {level:.3f}:")
            
            # Reversal signals
            reversal_signals = []
            if rev_high_vol > cont_high_vol + 0.05:
                reversal_signals.append(f"✅ High volatility ({rev_high_vol:.1%} vs {cont_high_vol:.1%})")
            if rev_medium_vol > cont_medium_vol + 0.05:
                reversal_signals.append(f"✅ Medium volatility ({rev_medium_vol:.1%} vs {cont_medium_vol:.1%})")
            if rev_low_vol > cont_low_vol + 0.05:
                reversal_signals.append(f"✅ Low volatility ({rev_low_vol:.1%} vs {cont_low_vol:.1%})")
            if rev_downtrend > cont_downtrend + 0.05:
                reversal_signals.append(f"✅ Downtrend context ({rev_downtrend:.1%} vs {cont_downtrend:.1%})")
            if rev_uptrend > cont_uptrend + 0.05:
                reversal_signals.append(f"✅ Uptrend context ({rev_uptrend:.1%} vs {cont_uptrend:.1%})")
            if rev_impulse > cont_impulse + 0.5:
                reversal_signals.append(f"✅ Strong impulse ({rev_impulse:.1f} vs {cont_impulse:.1f})")
            if rev_wick > cont_wick + 0.05:
                reversal_signals.append(f"✅ Wick rejection ({rev_wick:.1%} vs {cont_wick:.1%})")
            if rev_zone_strength > cont_zone_strength + 0.1:
                reversal_signals.append(f"✅ Strong zone ({rev_zone_strength:.2f} vs {cont_zone_strength:.2f})")
            if rev_hour > cont_hour + 0.5:
                reversal_signals.append(f"✅ Later timing ({rev_hour:.1f}h vs {cont_hour:.1f}h)")
            
            # Continuation signals
            continuation_signals = []
            if cont_high_vol > rev_high_vol + 0.05:
                continuation_signals.append(f"✅ High volatility ({cont_high_vol:.1%} vs {rev_high_vol:.1%})")
            if cont_medium_vol > rev_medium_vol + 0.05:
                continuation_signals.append(f"✅ Medium volatility ({cont_medium_vol:.1%} vs {rev_medium_vol:.1%})")
            if cont_low_vol > rev_low_vol + 0.05:
                continuation_signals.append(f"✅ Low volatility ({cont_low_vol:.1%} vs {rev_low_vol:.1%})")
            if cont_downtrend > rev_downtrend + 0.05:
                continuation_signals.append(f"✅ Downtrend context ({cont_downtrend:.1%} vs {rev_downtrend:.1%})")
            if cont_uptrend > rev_uptrend + 0.05:
                continuation_signals.append(f"✅ Uptrend context ({cont_uptrend:.1%} vs {rev_uptrend:.1%})")
            if cont_impulse > rev_impulse + 0.5:
                continuation_signals.append(f"✅ Strong impulse ({cont_impulse:.1f} vs {rev_impulse:.1f})")
            if cont_wick > rev_wick + 0.05:
                continuation_signals.append(f"✅ Wick rejection ({cont_wick:.1%} vs {rev_wick:.1%})")
            if cont_zone_strength > rev_zone_strength + 0.1:
                continuation_signals.append(f"✅ Strong zone ({cont_zone_strength:.2f} vs {rev_zone_strength:.2f})")
            if cont_hour > rev_hour + 0.5:
                continuation_signals.append(f"✅ Later timing ({cont_hour:.1f}h vs {rev_hour:.1f}h)")
            
            print(f"   🔄 TRADE REVERSAL when:")
            if reversal_signals:
                for signal in reversal_signals:
                    print(f"      {signal}")
            else:
                print(f"      ⚠️ No clear reversal signals (contexts too similar)")
            
            print(f"   ➡️ TRADE CONTINUATION when:")
            if continuation_signals:
                for signal in continuation_signals:
                    print(f"      {signal}")
            else:
                print(f"      ⚠️ No clear continuation signals (contexts too similar)")
            
            # Overall recommendation
            if reversal_rate > 0.4:
                print(f"   🎯 OVERALL: This level favors REVERSALS ({reversal_rate:.1%})")
            elif continuation_rate > 0.7:
                print(f"   🎯 OVERALL: This level favors CONTINUATIONS ({continuation_rate:.1%})")
            else:
                print(f"   🎯 OVERALL: This level is BALANCED (Reversal {reversal_rate:.1%}, Continuation {continuation_rate:.1%})")

def main():
    """Run Fibonacci trading signals analysis."""
    print("🚀 Fibonacci Trading Signals Analysis")
    print("=" * 80)
    print("🎯 WHEN to trade each level - REVERSAL vs CONTINUATION signals")
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
        
        # Analyze trading signals
        trading_signals = analyze_trading_signals(data, system)
        
        if trading_signals:
            show_trading_signals(trading_signals)
            
            print(f"\n🎉 Trading Signals Analysis Complete!")
            print(f"   ✅ Know WHEN to trade each Fibonacci level")
            print(f"   ✅ Know WHAT context makes reversals vs continuations")
            print(f"   ✅ Ready for profitable trading!")
        else:
            print(f"   ❌ No trading signals data available")
            
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()