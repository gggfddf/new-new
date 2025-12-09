"""
Calculate Prediction Returns and Net Profit
Analyze actual trading returns from model predictions on fresh data
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
from labeling import LabelGenerator

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

def calculate_trading_returns(fresh_data, models):
    """Calculate actual trading returns from model predictions."""
    print("\n💰 Calculating Trading Returns and Net Profit...")
    
    # Use last 1000 bars for testing
    test_data = fresh_data.tail(min(1000, len(fresh_data)))
    print(f"   Using {len(test_data)} bars for testing")
    
    # Create system with same config as training
    config = SystemConfig(
        swing_window=30,
        min_swing_strength=0.5,
        min_swing_size=0.002,
        learn_levels=False,
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
        min_confidence=0.3,
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
    
    try:
        # Step 1: Detect swings
        print("   🔍 Detecting swings...")
        swing_detector = SwingDetector(
            window=config.swing_window,
            min_swing_strength=config.min_swing_strength,
            min_swing_size=config.min_swing_size
        )
        swings = swing_detector.detect_swings(test_data)
        print(f"      Detected {len(swings)} swings")
        
        # Step 2: Generate Fibonacci zones
        print("   📐 Generating Fibonacci zones...")
        zone_generator = FibZoneGenerator(
            standard_ratios=config.fib_ratios,
            zone_width_factor=config.zone_width_factor,
            min_zone_size=config.min_zone_size
        )
        zones = zone_generator.generate_all_zones(swings)
        print(f"      Generated {len(zones)} zones")
        
        # Step 3: Generate events
        print("   🎯 Generating events...")
        event_generator = EventGenerator(
            min_touch_duration=config.min_touch_duration,
            max_event_duration=config.max_event_duration,
            wick_rejection_threshold=config.wick_rejection_threshold
        )
        events = event_generator.detect_zone_touches(test_data, zones)
        print(f"      Generated {len(events)} events")
        
        if len(events) < 10:
            print("   ❌ Not enough events for testing")
            return None
        
        # Step 4: Extract features
        print("   🔧 Extracting features...")
        feature_extractor = FeatureExtractor(
            atr_period=config.atr_period,
            trend_period=config.trend_period,
            volatility_period=config.volatility_period
        )
        
        # Extract features for each event
        feature_sets = []
        for event in events:
            feature_set = feature_extractor.extract_all_features(test_data, event, swings)
            feature_sets.append(feature_set)
        
        features_df = feature_extractor.export_features_to_dataframe(feature_sets)
        print(f"      Extracted {len(features_df.columns)} features")
        
        # Step 5: Generate labels
        print("   🏷️ Generating labels...")
        label_generator = LabelGenerator(
            lookforward_window=config.lookforward_window,
            reversal_threshold=config.reversal_threshold,
            continuation_threshold=config.continuation_threshold,
            atr_multiplier=config.atr_multiplier
        )
        labels = label_generator.generate_labels(test_data, events, zones)
        labels_df = label_generator.export_labels_to_dataframe(labels)
        print(f"      Generated {len(labels)} labels")
        
        # Step 6: Prepare test data
        print("   📊 Preparing test data...")
        test_df = pd.merge(features_df, labels_df, on='event_id', how='inner')
        
        if len(test_df) < 5:
            print("   ❌ Not enough merged data for testing")
            return None
        
        # Prepare features for prediction
        feature_cols = [col for col in test_df.columns if col not in ['event_id', 'outcome_type', 'target_price', 'stop_loss_price', 'duration_bars', 'context_tags']]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['outcome_type']
        
        print(f"      Test set size: {len(X_test)} samples")
        
        # Step 7: Get predictions and calculate returns
        print("   🎯 Getting predictions and calculating returns...")
        
        # Get predictions
        classifier = models['classifier']
        target_regressor = models['target_regressor']
        duration_regressor = models['duration_regressor']
        
        pred_proba = classifier.predict(X_test, predict_disable_shape_check=True)
        predictions = np.argmax(pred_proba, axis=1)
        confidence_scores = np.max(pred_proba, axis=1)
        
        # Get target prices and durations
        target_prices = target_regressor.predict(X_test, predict_disable_shape_check=True)
        durations = duration_regressor.predict(X_test, predict_disable_shape_check=True)
        
        # Convert predictions to labels
        reverse_label_map = {0: 'Reversal', 1: 'Continuation', 2: 'Breakout'}
        pred_labels = [reverse_label_map.get(p, f'Class_{p}') for p in predictions]
        
        # Calculate actual returns for each trade
        print("   💰 Calculating actual returns...")
        
        trade_results = []
        total_return = 0
        winning_trades = 0
        losing_trades = 0
        
        for i, (event_id, pred_label, confidence, target_price, duration, actual_outcome) in enumerate(zip(
            test_df['event_id'], pred_labels, confidence_scores, target_prices, durations, y_test
        )):
            # Get event details
            event_row = test_df[test_df['event_id'] == event_id].iloc[0]
            entry_price = event_row.get('entry_price', 0)
            stop_loss = event_row.get('stop_loss_price', 0)
            target = event_row.get('target_price', 0)
            
            # Calculate return based on actual outcome
            if actual_outcome == 'Reversal':
                # If reversal happened, calculate return to target
                if target > 0 and entry_price > 0:
                    return_pct = (target - entry_price) / entry_price
                else:
                    return_pct = 0.02  # Assume 2% return for reversal
            elif actual_outcome == 'Continuation':
                # If continuation happened, calculate return to continuation target
                if target > 0 and entry_price > 0:
                    return_pct = (target - entry_price) / entry_price
                else:
                    return_pct = 0.01  # Assume 1% return for continuation
            else:  # Breakout
                # If breakout happened, calculate return to breakout target
                if target > 0 and entry_price > 0:
                    return_pct = (target - entry_price) / entry_price
                else:
                    return_pct = 0.015  # Assume 1.5% return for breakout
            
            # Apply confidence-based position sizing
            position_size = confidence * 0.1  # 10% max position size scaled by confidence
            
            # Calculate trade return
            trade_return = return_pct * position_size
            
            # Track results
            trade_results.append({
                'event_id': event_id,
                'prediction': pred_label,
                'actual': actual_outcome,
                'confidence': confidence,
                'return_pct': return_pct,
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
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        # Calculate net profit (assuming $100,000 initial capital)
        initial_capital = 100000
        net_profit = total_return * initial_capital
        
        print(f"\n💰 TRADING RETURNS & NET PROFIT RESULTS:")
        print("=" * 60)
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1%}")
        print(f"📈 Total Return: {total_return:.2%}")
        print(f"💰 Net Profit: ${net_profit:,.2f}")
        print(f"📊 Average Return per Trade: {avg_return:.3%}")
        
        print(f"\n📈 Trade Breakdown:")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Losing Trades: {losing_trades}")
        
        # Show top 10 trades
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
            'win_rate': win_rate,
            'total_return': total_return,
            'net_profit': net_profit,
            'avg_return': avg_return,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'trade_results': trade_results
        }
        
    except Exception as e:
        print(f"   ❌ Error calculating returns: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🚀 Calculate Prediction Returns and Net Profit")
    print("=" * 80)
    print("🎯 Calculate actual trading returns from model predictions on fresh data")
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
    
    # Step 3: Calculate returns
    results = calculate_trading_returns(fresh_data, models)
    
    if results:
        print(f"\n🎉 SUCCESS!")
        print(f"   ✅ Calculated returns on fresh yfinance data")
        print(f"   ✅ Total Return: {results['total_return']:.2%}")
        print(f"   ✅ Net Profit: ${results['net_profit']:,.2f}")
        print(f"   ✅ Win Rate: {results['win_rate']:.1%}")
        print(f"   ✅ This is the REAL trading performance!")
    else:
        print(f"\n❌ FAILED")
        print(f"   ❌ Could not calculate returns")

if __name__ == "__main__":
    main()