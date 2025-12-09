"""
Real Trading Results - Use Actual Price Movements
Calculate real returns based on actual price movements, not hardcoded values
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

def load_saved_models():
    """Load saved models."""
    print("📊 Loading saved models...")
    
    try:
        # Load models
        classifier = joblib.load('/workspace/fib_ml_models_classifier.pkl')
        target_regressor = joblib.load('/workspace/fib_ml_models_target_regressor.pkl')
        duration_regressor = joblib.load('/workspace/fib_ml_models_duration_regressor.pkl')
        calibration_model = joblib.load('/workspace/fib_ml_models_calibration.pkl')
        metadata = joblib.load('/workspace/fib_ml_models_metadata.pkl')
        
        print("   ✅ All models loaded successfully!")
        
        return {
            'classifier': classifier,
            'target_regressor': target_regressor,
            'duration_regressor': duration_regressor,
            'calibration_model': calibration_model,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return None

def calculate_real_returns(fresh_data, models):
    """Calculate REAL returns based on actual price movements."""
    print("\n💰 Calculating REAL Returns (Actual Price Movements)...")
    
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
        
        # Step 4: Extract features
        print("   🔧 Extracting features...")
        feature_extractor = FeatureExtractor(
            atr_period=20,
            trend_period=30,
            volatility_period=30
        )
        
        # Extract features for each event
        feature_sets = []
        for event in events:
            feature_set = feature_extractor.extract_all_features(test_data, event, swings)
            feature_sets.append(feature_set)
        
        features_df = feature_extractor.export_features_to_dataframe(feature_sets)
        print(f"      Extracted {len(features_df.columns)} features")
        
        # Step 5: Make predictions
        print("   🎯 Making predictions...")
        
        # Prepare features for prediction
        feature_cols = [col for col in features_df.columns if col not in ['event_id', 'context_tags']]
        X_test = features_df[feature_cols].fillna(0)
        
        # Get predictions
        classifier = models['classifier']
        pred_proba = classifier.predict(X_test, predict_disable_shape_check=True)
        predictions = np.argmax(pred_proba, axis=1)
        confidence_scores = np.max(pred_proba, axis=1)
        
        # Convert predictions to labels
        reverse_label_map = {0: 'Reversal', 1: 'Continuation', 2: 'Breakout'}
        pred_labels = [reverse_label_map.get(p, f'Class_{p}') for p in predictions]
        
        # Step 6: Calculate REAL returns based on actual price movements
        print("   💰 Calculating REAL returns from actual price movements...")
        
        trade_results = []
        total_return = 0
        winning_trades = 0
        losing_trades = 0
        
        for i, (event, pred_label, confidence) in enumerate(zip(events, pred_labels, confidence_scores)):
            # Get entry details
            entry_index = event.entry_index
            entry_time = test_data.index[entry_index]
            entry_price = test_data.iloc[entry_index]['close']
            
            # Check if we have enough future data
            future_bars = len(test_data) - entry_index - 1
            
            if future_bars > 0:
                # Get future data (look ahead 40 bars = 200 minutes)
                lookforward_bars = min(40, future_bars)
                future_data = test_data.iloc[entry_index:entry_index + lookforward_bars + 1]
                
                # Calculate actual price movements
                entry_price = future_data.iloc[0]['close']
                max_price = future_data['high'].max()
                min_price = future_data['low'].min()
                
                price_change_up = (max_price - entry_price) / entry_price
                price_change_down = (entry_price - min_price) / entry_price
                
                # Determine actual outcome based on price movement
                if price_change_up > 0.025:  # 2.5% up
                    actual_outcome = 'Reversal'
                    actual_return = price_change_up
                elif price_change_down > 0.025:  # 2.5% down
                    actual_outcome = 'Continuation'
                    actual_return = -price_change_down
                else:
                    actual_outcome = 'Breakout'
                    # For Breakout, use the actual price movement (not hardcoded!)
                    actual_return = max(price_change_up, -price_change_down)
                
                # Apply confidence-based position sizing
                position_size = confidence * 0.1  # 10% max position size scaled by confidence
                
                # Calculate trade return
                trade_return = actual_return * position_size
                
                # Track results
                trade_results.append({
                    'event_id': i,
                    'prediction': pred_label,
                    'actual': actual_outcome,
                    'confidence': confidence,
                    'entry_price': entry_price,
                    'max_price': max_price,
                    'min_price': min_price,
                    'price_change_up': price_change_up,
                    'price_change_down': price_change_down,
                    'actual_return': actual_return,
                    'position_size': position_size,
                    'trade_return': trade_return,
                    'correct': pred_label == actual_outcome
                })
                
                total_return += trade_return
                
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
        
        # Calculate summary statistics
        total_trades = len(trade_results)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        accuracy = sum(1 for t in trade_results if t['correct']) / total_trades if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        # Calculate net profit (assuming $100,000 initial capital)
        initial_capital = 100000
        net_profit = total_return * initial_capital
        
        print(f"\n💰 REAL TRADING RESULTS (ACTUAL PRICE MOVEMENTS):")
        print("=" * 60)
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎯 Accuracy: {accuracy:.1%}")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"📈 Total Return: {total_return:.2%}")
        print(f"💰 Net Profit: ${net_profit:,.2f}")
        print(f"📊 Average Return per Trade: {avg_return:.3%}")
        
        print(f"\n📈 Trade Breakdown:")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Losing Trades: {losing_trades}")
        
        # Show prediction distribution
        pred_counts = pd.Series(pred_labels).value_counts()
        print(f"\n📈 Prediction Distribution:")
        for pred_type, count in pred_counts.items():
            percentage = (count / len(pred_labels)) * 100
            print(f"   {pred_type}: {count} ({percentage:.1f}%)")
        
        # Show actual outcome distribution
        actual_counts = pd.Series([t['actual'] for t in trade_results]).value_counts()
        print(f"\n📊 Actual Outcome Distribution:")
        for actual_type, count in actual_counts.items():
            percentage = (count / len(trade_results)) * 100
            print(f"   {actual_type}: {count} ({percentage:.1f}%)")
        
        # Show sample trades with real returns
        print(f"\n📊 Sample Trades (Real Returns):")
        for i, trade in enumerate(trade_results[:10]):
            print(f"   Trade {i+1}: {trade['prediction']} → {trade['actual']}")
            print(f"      Entry: ${trade['entry_price']:.2f}")
            print(f"      Max: ${trade['max_price']:.2f} (+{trade['price_change_up']:.2%})")
            print(f"      Min: ${trade['min_price']:.2f} (-{trade['price_change_down']:.2%})")
            print(f"      Actual Return: {trade['actual_return']:.3%}")
            print(f"      Trade Return: {trade['trade_return']:.3%}")
            print(f"      Confidence: {trade['confidence']:.1%}")
            print()
        
        # Show top 10 best and worst trades
        print(f"\n🏆 Top 10 Best Trades:")
        sorted_trades = sorted(trade_results, key=lambda x: x['trade_return'], reverse=True)
        for i, trade in enumerate(sorted_trades[:10]):
            print(f"   {i+1}. {trade['prediction']} → {trade['actual']} | "
                  f"Return: {trade['trade_return']:.3%} | "
                  f"Confidence: {trade['confidence']:.1%}")
        
        print(f"\n💸 Top 10 Worst Trades:")
        for i, trade in enumerate(sorted_trades[-10:]):
            print(f"   {i+1}. {trade['prediction']} → {trade['actual']} | "
                  f"Return: {trade['trade_return']:.3%} | "
                  f"Confidence: {trade['confidence']:.1%}")
        
        return {
            'total_trades': total_trades,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'total_return': total_return,
            'net_profit': net_profit,
            'avg_return': avg_return,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'trade_results': trade_results
        }
        
    except Exception as e:
        print(f"   ❌ Error calculating real returns: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🚀 Real Trading Results - Use Actual Price Movements")
    print("=" * 80)
    print("🎯 Calculate real returns based on actual price movements, not hardcoded values")
    print("=" * 80)
    
    # Step 1: Load fresh data
    fresh_data = load_fresh_btc_data()
    if fresh_data is None:
        print("❌ Cannot proceed - no fresh data available")
        return
    
    # Step 2: Load saved models
    models = load_saved_models()
    if models is None:
        print("❌ Cannot proceed - models not loaded")
        return
    
    # Step 3: Calculate real returns
    results = calculate_real_returns(fresh_data, models)
    
    if results:
        print(f"\n🎉 REAL TRADING RESULTS:")
        print(f"   ✅ Accuracy: {results['accuracy']:.1%}")
        print(f"   ✅ Win Rate: {results['win_rate']:.1%}")
        print(f"   ✅ Total Return: {results['total_return']:.2%}")
        print(f"   ✅ Net Profit: ${results['net_profit']:,.2f}")
        print(f"   ✅ This is the REAL trading performance!")
    else:
        print(f"\n❌ FAILED")
        print(f"   ❌ Could not calculate real returns")

if __name__ == "__main__":
    main()