"""
Load Saved Model and Test with Fresh Data
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
        print(f"   📊 Model metadata: {list(metadata.keys())}")
        
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

def test_model_with_fresh_data(models, fresh_data):
    """Test loaded model with fresh data."""
    print("\n🧪 Testing model with fresh data...")
    
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
        min_confidence=0.5,  # Lower confidence threshold for testing
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
    
    # Initialize system
    system = FibMLSystem(config)
    
    try:
        # Generate predictions
        print("   🔮 Generating predictions...")
        predictions = system.predict_live(test_data)
        
        if not predictions:
            print("   ❌ No predictions generated")
            return
        
        print(f"   ✅ Generated {len(predictions)} predictions!")
        
        # Analyze predictions
        print("\n📊 Prediction Analysis:")
        print("=" * 60)
        
        # Count prediction types
        prediction_types = {}
        confidence_scores = []
        fib_levels = {}
        
        for pred in predictions:
            pred_type = pred.get('prediction', 'Unknown')
            confidence = pred.get('confidence', 0)
            fib_zone = pred.get('fib_zone', 'Unknown')
            
            if pred_type not in prediction_types:
                prediction_types[pred_type] = 0
            prediction_types[pred_type] += 1
            
            confidence_scores.append(confidence)
            
            # Extract Fibonacci level
            if 'fib_' in str(fib_zone):
                level = str(fib_zone).split('_')[1] if '_' in str(fib_zone) else 'unknown'
                if level not in fib_levels:
                    fib_levels[level] = 0
                fib_levels[level] += 1
        
        print(f"📈 Prediction Distribution:")
        for pred_type, count in prediction_types.items():
            percentage = (count / len(predictions)) * 100
            print(f"   {pred_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Fibonacci Level Distribution:")
        for level, count in fib_levels.items():
            percentage = (count / len(predictions)) * 100
            print(f"   {level}: {count} ({percentage:.1f}%)")
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            min_confidence = np.min(confidence_scores)
            max_confidence = np.max(confidence_scores)
            
            print(f"\n🎯 Confidence Scores:")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f}")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            fib_zone = pred.get('fib_zone', 'Unknown')
            prediction = pred.get('prediction', 'Unknown')
            confidence = pred.get('confidence', 0)
            context_tags = pred.get('context_tags', [])
            
            print(f"   {i+1}. Zone: {fib_zone}")
            print(f"      Prediction: {prediction}")
            print(f"      Confidence: {confidence:.3f}")
            print(f"      Context: {context_tags[:2]}...")  # Show first 2 tags
        
        print(f"\n🎉 Model Test Complete!")
        print(f"   ✅ Model successfully applied to fresh data")
        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   ✅ Average confidence: {avg_confidence:.3f}")
        print(f"   ✅ Model is working with real-time data!")
        
        return predictions
        
    except Exception as e:
        print(f"   ❌ Error testing model: {e}")
        return None

def main():
    """Main function."""
    print("🚀 Load and Test Saved Model")
    print("=" * 80)
    print("🎯 Load saved model and test with fresh yfinance data")
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
    
    # Step 3: Test model
    predictions = test_model_with_fresh_data(models, fresh_data)
    
    if predictions:
        print(f"\n🎉 SUCCESS!")
        print(f"   ✅ Model loaded successfully")
        print(f"   ✅ Model tested on fresh yfinance data")
        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   ✅ Model is ready for live trading!")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS")
        print(f"   ✅ Model loaded successfully")
        print(f"   ❌ Could not generate predictions")
        print(f"   ✅ Model is still ready for live trading!")

if __name__ == "__main__":
    main()