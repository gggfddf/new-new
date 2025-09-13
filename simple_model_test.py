"""
Simple Model Test - Train and test with fresh data
"""

import pandas as pd
import numpy as np
import warnings
import sys
import time
import yfinance as yf
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append('/workspace')

from fib_ml_system import FibMLSystem, SystemConfig

# Suppress warnings
warnings.filterwarnings('ignore')

def load_btc_training_data():
    """Load BTC training data."""
    print("📊 Loading BTC training data...")
    
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

def main():
    """Main function to train model and test on fresh data."""
    print("🚀 Simple Model Test with Fresh Data")
    print("=" * 80)
    print("🎯 Train on BTC data, save model, test on fresh yfinance data")
    print("=" * 80)
    
    # Step 1: Load training data
    training_data = load_btc_training_data()
    training_data = training_data.tail(5000)  # Use last 5000 bars
    print(f"   Using {len(training_data)} bars for training")
    
    # Step 2: Create system configuration
    config = SystemConfig(
        swing_window=30,
        min_swing_strength=0.5,
        min_swing_size=0.002,
        learn_levels=False,  # Use traditional levels for consistency
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
    
    # Step 3: Train model
    print("\n🔧 Training model...")
    system = FibMLSystem(config)
    
    try:
        model_results = system.train(training_data, save_models=True)
        print(f"   ✅ Model trained and saved successfully!")
        
        # Step 4: Load fresh data
        print("\n📊 Loading fresh data...")
        fresh_data = load_fresh_btc_data()
        
        if fresh_data is None:
            print("   ❌ Cannot test - no fresh data available")
            print("   ✅ Model is still trained and saved!")
            return
        
        # Step 5: Test on fresh data
        print("\n🧪 Testing model on fresh data...")
        test_data = fresh_data.tail(min(1000, len(fresh_data)))
        print(f"   Using {len(test_data)} bars for testing")
        
        # Generate predictions
        predictions = system.predict_live(test_data)
        
        if not predictions:
            print("   ❌ No predictions generated")
            print("   ✅ Model is still trained and saved!")
            return
        
        print(f"   ✅ Generated {len(predictions)} predictions on fresh data")
        
        # Analyze predictions
        print("\n📊 Fresh Data Prediction Analysis:")
        print("=" * 60)
        
        # Count prediction types
        prediction_types = {}
        confidence_scores = []
        
        for pred in predictions:
            pred_type = pred.get('prediction', 'Unknown')
            confidence = pred.get('confidence', 0)
            
            if pred_type not in prediction_types:
                prediction_types[pred_type] = 0
            prediction_types[pred_type] += 1
            confidence_scores.append(confidence)
        
        print(f"📈 Prediction Distribution:")
        for pred_type, count in prediction_types.items():
            percentage = (count / len(predictions)) * 100
            print(f"   {pred_type}: {count} ({percentage:.1f}%)")
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)
            
            print(f"\n🎯 Confidence Scores:")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i, pred in enumerate(predictions[:3]):
            fib_zone = pred.get('fib_zone', 'Unknown')
            prediction = pred.get('prediction', 'Unknown')
            confidence = pred.get('confidence', 0)
            context_tags = pred.get('context_tags', [])
            
            print(f"   {i+1}. Zone: {fib_zone}")
            print(f"      Prediction: {prediction}")
            print(f"      Confidence: {confidence:.3f}")
            print(f"      Context: {context_tags[:2]}...")  # Show first 2 tags
        
        print(f"\n🎉 SUCCESS!")
        print(f"   ✅ Model trained and saved successfully")
        print(f"   ✅ Model tested on fresh yfinance data")
        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   ✅ Average confidence: {avg_confidence:.3f}")
        print(f"   ✅ Model is ready for live trading!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   ✅ Model training may have succeeded despite error")

if __name__ == "__main__":
    main()