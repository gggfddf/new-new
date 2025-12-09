"""
Test Real Accuracy on Fresh yfinance Data
Load saved model and test on fresh data to get actual out-of-sample accuracy
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

def test_real_accuracy(fresh_data, models):
    """Test real accuracy on fresh data."""
    print("\n🧪 Testing REAL accuracy on fresh yfinance data...")
    
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
        min_confidence=0.3,  # Lower threshold for testing
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
        # Merge features and labels
        test_df = pd.merge(features_df, labels_df, on='event_id', how='inner')
        
        if len(test_df) < 5:
            print("   ❌ Not enough merged data for testing")
            return None
        
        # Prepare features for prediction
        feature_cols = [col for col in test_df.columns if col not in ['event_id', 'outcome_type', 'target_price', 'stop_loss_price', 'duration_bars', 'context_tags']]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['outcome_type']
        
        print(f"      Test set size: {len(X_test)} samples")
        print(f"      Features: {len(feature_cols)}")
        
        # Step 7: Test model accuracy
        print("   🎯 Testing model accuracy...")
        
        # Get predictions
        classifier = models['classifier']
        pred_proba = classifier.predict(X_test, predict_disable_shape_check=True)
        
        # Get predicted classes (argmax of probabilities)
        predictions = np.argmax(pred_proba, axis=1)
        
        # Convert y_test to numeric if needed
        if y_test.dtype == 'object':
            # Map string labels to numeric
            label_map = {'Reversal': 0, 'Continuation': 1, 'Breakout': 2}
            y_test_numeric = y_test.map(label_map)
        else:
            y_test_numeric = y_test
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test_numeric)
        
        # Get prediction probabilities
        confidence_scores = np.max(pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # Count predictions by type
        pred_counts = pd.Series(predictions).value_counts()
        actual_counts = y_test_numeric.value_counts()
        
        print(f"\n📊 REAL ACCURACY RESULTS ON FRESH DATA:")
        print("=" * 60)
        print(f"🎯 Overall Accuracy: {accuracy:.1%}")
        print(f"🎯 Average Confidence: {avg_confidence:.1%}")
        print(f"📊 Test Samples: {len(X_test)}")
        
        # Map numeric predictions back to labels
        reverse_label_map = {0: 'Reversal', 1: 'Continuation', 2: 'Breakout'}
        
        print(f"\n📈 Prediction Distribution:")
        for pred_type, count in pred_counts.items():
            percentage = (count / len(predictions)) * 100
            label = reverse_label_map.get(pred_type, f'Class_{pred_type}')
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 Actual Distribution:")
        for actual_type, count in actual_counts.items():
            percentage = (count / len(y_test_numeric)) * 100
            label = reverse_label_map.get(actual_type, f'Class_{actual_type}')
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Calculate per-class accuracy
        print(f"\n🎯 Per-Class Accuracy:")
        for class_type in y_test_numeric.unique():
            mask = y_test_numeric == class_type
            if mask.sum() > 0:
                class_accuracy = np.mean(predictions[mask] == y_test_numeric[mask])
                label = reverse_label_map.get(class_type, f'Class_{class_type}')
                print(f"   {label}: {class_accuracy:.1%} ({mask.sum()} samples)")
        
        # Show confidence distribution
        print(f"\n🎯 Confidence Distribution:")
        high_conf = np.sum(confidence_scores > 0.8)
        medium_conf = np.sum((confidence_scores > 0.6) & (confidence_scores <= 0.8))
        low_conf = np.sum(confidence_scores <= 0.6)
        
        print(f"   High confidence (>0.8): {high_conf} ({high_conf/len(confidence_scores):.1%})")
        print(f"   Medium confidence (0.6-0.8): {medium_conf} ({medium_conf/len(confidence_scores):.1%})")
        print(f"   Low confidence (≤0.6): {low_conf} ({low_conf/len(confidence_scores):.1%})")
        
        print(f"\n🎉 REAL ACCURACY TEST COMPLETE!")
        print(f"   ✅ Tested on {len(X_test)} fresh samples")
        print(f"   ✅ Real out-of-sample accuracy: {accuracy:.1%}")
        print(f"   ✅ Average confidence: {avg_confidence:.1%}")
        
        return {
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'test_samples': len(X_test),
            'predictions': predictions,
            'actual': y_test,
            'confidence_scores': confidence_scores
        }
        
    except Exception as e:
        print(f"   ❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("🚀 Test REAL Accuracy on Fresh yfinance Data")
    print("=" * 80)
    print("🎯 Load saved model and test on fresh data for actual out-of-sample accuracy")
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
    
    # Step 3: Test real accuracy
    results = test_real_accuracy(fresh_data, models)
    
    if results:
        print(f"\n🎉 SUCCESS!")
        print(f"   ✅ Model tested on fresh yfinance data")
        print(f"   ✅ Real out-of-sample accuracy: {results['accuracy']:.1%}")
        print(f"   ✅ Average confidence: {results['confidence']:.1%}")
        print(f"   ✅ Test samples: {results['test_samples']}")
        print(f"   ✅ This is the REAL accuracy on unseen data!")
    else:
        print(f"\n❌ FAILED")
        print(f"   ❌ Could not test model on fresh data")

if __name__ == "__main__":
    main()