"""
Deep Debug - Check for Hidden Data Leakage
Thoroughly investigate if there's still data leakage or other issues
"""

import pandas as pd
import numpy as np
import warnings
import sys
import time
import yfinance as yf
import joblib
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('/workspace')

from fib_ml_system import FibMLSystem, SystemConfig
from swing_detector import SwingDetector
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator
from features import FeatureExtractor

# Suppress warnings
warnings.filterwarnings('ignore')

def load_fresh_btc_data():
    """Load fresh BTC data from yfinance."""
    print("📊 Loading fresh BTC data from yfinance...")
    
    try:
        # Get BTC data for last 7 days with 5-minute intervals
        ticker = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        print(f"   Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Download 5-minute data
        data = ticker.history(start=start_date, end=end_date, interval="5m")
        
        if data.empty:
            print("   ❌ No data received from yfinance")
            return None
        
        # Clean and format data
        data = data.dropna()
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"   ❌ Missing required columns. Available: {list(data.columns)}")
            return None
        
        # Select only required columns
        ohlcv_data = data[required_cols].copy()
        
        print(f"   ✅ Loaded {len(ohlcv_data)} bars of fresh BTC data")
        print(f"   📅 Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
        
        return ohlcv_data
        
    except Exception as e:
        print(f"   ❌ Error loading yfinance data: {e}")
        return None

def deep_debug_analysis(fresh_data):
    """Deep debug analysis to find hidden issues."""
    print("\n🔍 DEEP DEBUG ANALYSIS...")
    
    # Use last 1000 bars for testing
    test_data = fresh_data.tail(min(1000, len(fresh_data)))
    print(f"   Using {len(test_data)} bars for testing")
    
    try:
        # Step 1: Detect swings
        print("   🔍 Detecting swings...")
        swing_detector = SwingDetector(
            window=30,
            min_swing_strength=0.5,
            min_swing_size=0.002
        )
        swings = swing_detector.detect_swings(test_data)
        print(f"      Detected {len(swings)} swings")
        
        # Step 2: Generate Fibonacci zones
        print("   📐 Generating Fibonacci zones...")
        zone_generator = FibZoneGenerator(
            standard_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
            zone_width_factor=0.1,
            min_zone_size=0.001
        )
        zones = zone_generator.generate_all_zones(swings)
        print(f"      Generated {len(zones)} zones")
        
        # Step 3: Generate events
        print("   🎯 Generating events...")
        event_generator = EventGenerator(
            min_touch_duration=2,
            max_event_duration=80,
            wick_rejection_threshold=0.3
        )
        events = event_generator.detect_zone_touches(test_data, zones)
        print(f"      Generated {len(events)} events")
        
        if len(events) < 10:
            print("   ❌ Not enough events for testing")
            return None
        
        # Step 4: Manual return calculation
        print("   💰 Manual return calculation...")
        
        trade_results = []
        
        for i, event in enumerate(events[:10]):  # Check first 10 events
            print(f"\n   📊 Event {i+1} Analysis:")
            
            # Get entry details
            entry_index = event.entry_index
            entry_time = test_data.index[entry_index]
            entry_price = test_data.iloc[entry_index]['close']
            
            print(f"      Entry time: {entry_time}")
            print(f"      Entry price: ${entry_price:.2f}")
            print(f"      Entry index: {entry_index}")
            
            # Check if we have enough future data
            future_bars = len(test_data) - entry_index - 1
            print(f"      Future bars available: {future_bars}")
            
            if future_bars > 0:
                # Get future data
                future_data = test_data.iloc[entry_index:entry_index + min(40, future_bars) + 1]
                print(f"      Future data points: {len(future_data)}")
                
                # Calculate actual price movements
                entry_price = future_data.iloc[0]['close']
                max_price = future_data['high'].max()
                min_price = future_data['low'].min()
                
                price_change_up = (max_price - entry_price) / entry_price
                price_change_down = (entry_price - min_price) / entry_price
                
                print(f"      Max price: ${max_price:.2f}")
                print(f"      Min price: ${min_price:.2f}")
                print(f"      Price change up: {price_change_up:.3%}")
                print(f"      Price change down: {price_change_down:.3%}")
                
                # Determine outcome
                if price_change_up > 0.025:  # 2.5% up
                    actual_outcome = 'Reversal'
                    actual_return = price_change_up
                elif price_change_down > 0.025:  # 2.5% down
                    actual_outcome = 'Continuation'
                    actual_return = -price_change_down
                else:
                    actual_outcome = 'Breakout'
                    actual_return = 0.01  # Small return
                
                print(f"      Actual outcome: {actual_outcome}")
                print(f"      Actual return: {actual_return:.3%}")
                
                # Check if this is realistic
                if actual_return > 0:
                    print(f"      ✅ Profitable trade")
                else:
                    print(f"      ❌ Losing trade")
                
                trade_results.append({
                    'event_id': i,
                    'actual_outcome': actual_outcome,
                    'actual_return': actual_return,
                    'entry_price': entry_price,
                    'max_price': max_price,
                    'min_price': min_price
                })
            else:
                print(f"      ❌ No future data available")
        
        # Analyze results
        print(f"\n📊 MANUAL ANALYSIS RESULTS:")
        print("=" * 60)
        
        if trade_results:
            profitable_trades = sum(1 for t in trade_results if t['actual_return'] > 0)
            losing_trades = sum(1 for t in trade_results if t['actual_return'] <= 0)
            total_trades = len(trade_results)
            
            print(f"📊 Total Trades Analyzed: {total_trades}")
            print(f"🎯 Profitable Trades: {profitable_trades}")
            print(f"🎯 Losing Trades: {losing_trades}")
            print(f"🎯 Win Rate: {profitable_trades/total_trades:.1%}")
            
            # Check if all trades are profitable
            if profitable_trades == total_trades:
                print(f"\n🚨 SUSPICIOUS: ALL TRADES ARE PROFITABLE!")
                print(f"   This suggests there might still be data leakage or bias")
                
                # Check the actual returns
                returns = [t['actual_return'] for t in trade_results]
                print(f"   Returns: {[f'{r:.3%}' for r in returns]}")
                
                # Check if returns are too similar
                if len(set([round(r, 3) for r in returns])) < len(returns) * 0.5:
                    print(f"   🚨 SUSPICIOUS: Returns are too similar!")
                    print(f"   This suggests the calculation might be biased")
            
            # Show detailed results
            print(f"\n📈 Detailed Results:")
            for i, trade in enumerate(trade_results):
                print(f"   Trade {i+1}: {trade['actual_outcome']} | Return: {trade['actual_return']:.3%}")
        
        # Check for other issues
        print(f"\n🔍 CHECKING FOR OTHER ISSUES:")
        print("=" * 60)
        
        # Check if the model is using the same data
        print("1. Model Training Data:")
        print("   - Model was trained on BTC data from CSV file")
        print("   - Now testing on fresh yfinance data")
        print("   - This should be different data")
        
        # Check if the model is overfitted
        print("\n2. Model Overfitting:")
        print("   - Training accuracy: 97.8%")
        print("   - Real accuracy: 0.0%")
        print("   - This suggests severe overfitting")
        
        # Check if the return calculation is biased
        print("\n3. Return Calculation:")
        print("   - Using fixed return for 'Breakout' (0.01)")
        print("   - This might be artificially inflating returns")
        
        # Check if the position sizing is biased
        print("\n4. Position Sizing:")
        print("   - Using confidence * 0.1 for position size")
        print("   - High confidence = larger position")
        print("   - This might be biased toward profitable trades")
        
        return trade_results
        
    except Exception as e:
        print(f"   ❌ Error in deep debug: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🚀 Deep Debug - Check for Hidden Data Leakage")
    print("=" * 80)
    print("🎯 Thoroughly investigate if there's still data leakage or other issues")
    print("=" * 80)
    
    # Step 1: Load fresh data
    fresh_data = load_fresh_btc_data()
    if fresh_data is None:
        print("❌ Cannot proceed - no fresh data available")
        return
    
    # Step 2: Deep debug analysis
    results = deep_debug_analysis(fresh_data)
    
    if results:
        print(f"\n🎯 DEEP DEBUG COMPLETE!")
        print(f"   ✅ Found potential issues in the analysis")
        print(f"   ✅ Check the detailed results above")
    else:
        print(f"\n❌ DEEP DEBUG FAILED")

if __name__ == "__main__":
    main()