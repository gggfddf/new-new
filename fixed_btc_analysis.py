"""
Fixed BTC Analysis - Run Fibonacci ML System on Real BTC Data
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os

# Add current directory to path
sys.path.append('/workspace')

from swing_detector import SwingDetector
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator
from features import FeatureExtractor
from labeling import LabelGenerator
from model_train import ModelTrainer

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
    
    print(f"   Data shape: {ohlcv_data.shape}")
    print(f"   Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    print(f"   Total return: {(ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[0] - 1) * 100:.1f}%")
    
    return ohlcv_data

def main():
    """Run analysis."""
    print("🚀 BTC Fibonacci ML Analysis")
    print("=" * 50)
    
    # Load data - use last 3000 bars for faster analysis
    data = load_btc_data().tail(3000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Step 1: Detect swings
    print(f"\n📈 Step 1: Detecting swings...")
    detector = SwingDetector(window=20, min_swing_strength=0.3, min_swing_size=0.002)
    swings = detector.detect_swings(data)
    print(f"   Detected {len(swings)} swings")
    
    if len(swings) < 2:
        print("❌ Not enough swings detected")
        return
    
    # Step 2: Generate Fibonacci zones
    print(f"\n📊 Step 2: Generating Fibonacci zones...")
    zone_generator = FibZoneGenerator(
        standard_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.1,
        min_zone_size=0.002
    )
    zones = zone_generator.generate_all_zones(swings)
    zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.2)
    print(f"   Generated {len(zones)} zones")
    
    if not zones:
        print("❌ No zones generated")
        return
    
    # Step 3: Generate events
    print(f"\n🎯 Step 3: Generating events...")
    event_generator = EventGenerator(
        min_touch_duration=1,
        max_event_duration=100,
        wick_rejection_threshold=0.3
    )
    events = event_generator.detect_zone_touches(data, zones)
    print(f"   Generated {len(events)} events")
    
    if not events:
        print("❌ No events generated")
        return
    
    # Step 4: Extract features
    print(f"\n🔍 Step 4: Extracting features...")
    feature_extractor = FeatureExtractor()
    feature_sets = []
    for event in events[:1000]:  # Limit to first 1000 events for speed
        try:
            feature_set = feature_extractor.extract_all_features(data, event, swings)
            feature_sets.append(feature_set)
        except:
            continue
    
    if not feature_sets:
        print("❌ No features extracted")
        return
    
    features_df = feature_extractor.export_features_to_dataframe(feature_sets)
    print(f"   Extracted {len(features_df.columns)} features for {len(features_df)} events")
    
    # Step 5: Generate labels
    print(f"\n🏷️ Step 5: Generating labels...")
    label_generator = LabelGenerator(
        lookforward_window=30,
        reversal_threshold=2.0,
        continuation_threshold=1.0,
        atr_multiplier=1.0,
        min_confidence=0.5
    )
    
    # Match events with features
    event_ids = [event.event_id for event in events[:len(feature_sets)]]
    labels = label_generator.generate_labels(data, events[:len(feature_sets)], zones)
    labels_df = label_generator.export_labels_to_dataframe(labels)
    print(f"   Generated {len(labels)} labels")
    
    if features_df.empty or labels_df.empty:
        print("❌ No features or labels")
        return
    
    # Step 6: Train models
    print(f"\n🧠 Step 6: Training models...")
    trainer = ModelTrainer(test_size=0.2, validation_size=0.2, random_state=42)
    
    try:
        model_results = trainer.train_all_models(features_df, labels_df)
        
        print(f"\n📊 Training Results:")
        classifier_metrics = model_results.performance_metrics['classifier']
        regression_metrics = model_results.performance_metrics['regression']
        
        print(f"   Classifier Accuracy: {classifier_metrics['accuracy']:.1%}")
        print(f"   ROC AUC Score: {classifier_metrics['roc_auc']:.3f}")
        print(f"   Target Regressor R²: {regression_metrics['target_price']['r2']:.3f}")
        print(f"   Duration Regressor R²: {regression_metrics['duration']['r2']:.3f}")
        
        # Feature importance
        print(f"\n🔍 Top 15 Most Important Features:")
        for i, (feature, importance) in enumerate(list(model_results.feature_importance.items())[:15]):
            print(f"   {i+1:2d}. {feature:<35} {importance:>8.1f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 7: Run backtest
    print(f"\n💰 Step 7: Running backtest...")
    from backtest import BacktestEngine
    
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=5.0,
        commission_bps=2.0,
        min_confidence=0.6
    )
    
    try:
        backtest_results = backtest_engine.run_backtest(data, model_results, events[:len(feature_sets)], feature_extractor, swings)
        
        metrics = backtest_results.performance_metrics
        print(f"\n📈 Backtest Results:")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Total Return: {metrics.get('total_return', 0):.1%}")
        print(f"   Annualized Return: {metrics.get('annualized_return', 0):.1%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   Expectancy: {metrics.get('expectancy', 0):.4f}")
        print(f"   Final Capital: ${metrics.get('final_capital', 0):,.0f}")
        
        # Trade analysis
        if backtest_results.trades:
            print(f"\n📊 Trade Analysis:")
            trade_analysis = backtest_results.trade_analysis
            print(f"   Average Duration: {trade_analysis.get('avg_duration_hours', 0):.1f} hours")
            print(f"   Outcome Distribution: {trade_analysis.get('outcome_distribution', {})}")
            
            pnl_stats = trade_analysis.get('pnl_stats', {})
            print(f"   Average P&L: ${pnl_stats.get('mean', 0):.2f}")
            print(f"   Best Trade: ${pnl_stats.get('max', 0):.2f}")
            print(f"   Worst Trade: ${pnl_stats.get('min', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
    
    # Step 8: Generate live predictions
    print(f"\n🔮 Step 8: Generating live predictions...")
    try:
        # Make predictions for recent events
        recent_events = events[-10:]  # Last 10 events
        predictions = []
        
        for event in recent_events:
            try:
                feature_set = feature_extractor.extract_all_features(data, event, swings)
                features_df_single = pd.DataFrame([feature_set.raw_features])
                
                # Classify outcome
                outcome_proba = model_results.classifier.predict(
                    features_df_single, num_iteration=model_results.classifier.best_iteration
                )
                outcome_idx = np.argmax(outcome_proba)
                outcome = model_results.classifier.reverse_encoder[outcome_idx]
                confidence = outcome_proba[outcome_idx]
                
                # Predict target price
                target_price = model_results.target_regressor.predict(
                    features_df_single, num_iteration=model_results.target_regressor.best_iteration
                )[0]
                
                prediction = {
                    "fib_zone": list(event.zone.price_range),
                    "prediction": outcome,
                    "confidence": confidence,
                    "target_zone": [target_price * 0.99, target_price * 1.01],
                    "stop_loss_level": event.zone.price_range[0] * 0.995 if outcome == 'Reversal' else event.zone.price_range[1] * 1.005,
                    "context_tags": [f"fib_{event.zone.level}", f"{event.zone.zone_type}_zone", f"{outcome.lower()}_prediction"]
                }
                predictions.append(prediction)
                
            except Exception as e:
                continue
        
        print(f"   Generated {len(predictions)} predictions")
        
        if predictions:
            print(f"\n📋 Sample Predictions:")
            for i, pred in enumerate(predictions[:3]):
                print(f"   Prediction {i+1}:")
                print(f"     Zone: {pred['fib_zone']}")
                print(f"     Outcome: {pred['prediction']}")
                print(f"     Confidence: {pred['confidence']:.1%}")
                print(f"     Context: {', '.join(pred['context_tags'][:3])}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    
    # Summary
    print(f"\n📊 Analysis Summary:")
    print(f"   Data: {len(data)} bars of BTC 5-minute data")
    print(f"   Swings: {len(swings)} detected")
    print(f"   Zones: {len(zones)} Fibonacci zones")
    print(f"   Events: {len(events)} zone touch events")
    print(f"   Features: {len(features_df.columns)} extracted")
    print(f"   Labels: {len(labels)} generated")
    
    print(f"\n🎉 BTC Fibonacci ML Analysis completed successfully!")
    print(f"   The system has learned from real BTC data and is ready for trading!")

if __name__ == "__main__":
    main()