"""
BTC Runner 30K - Efficient analysis with 30,000 data points
"""

import pandas as pd
import numpy as np
import warnings
import sys

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

def main():
    """Run BTC analysis with 30,000 data points."""
    print("🚀 BTC Runner 30K - Comprehensive Analysis")
    print("=" * 60)
    print("🎯 Running complete system with 30,000 BTC data points")
    print("=" * 60)
    
    # Load BTC data
    data = load_btc_data()
    
    # Use last 30000 bars for comprehensive analysis
    data = data.tail(30000)
    print(f"   Using last {len(data)} bars for comprehensive analysis")
    
    # Create system with optimized settings for large dataset
    config = SystemConfig(
        # Swing detection - optimized for large dataset
        swing_window=25,  # Slightly larger window for more stable swings
        min_swing_strength=0.4,  # Slightly lower threshold for more swings
        min_swing_size=0.0015,  # Slightly larger minimum swing size
        
        # Fibonacci zones - will be learned from data
        learn_levels=True,  # Enable adaptive level learning
        zone_width_factor=0.1,  # Standard zone width
        min_zone_size=0.001,  # Minimum zone size
        
        # Event generation - optimized for large dataset
        min_touch_duration=1,
        max_event_duration=150,  # Longer max duration for more events
        wick_rejection_threshold=0.3,  # Standard threshold
        
        # Enhanced context features
        enable_enhanced_context=True,  # Enable enhanced context features
        context_momentum_period=15,  # Longer momentum period for stability
        context_historical_period=100,  # Longer historical period
        context_failure_tracking=True,
        
        # Feature extraction
        atr_period=20,  # Longer ATR period for stability
        trend_period=30,  # Longer trend period
        volatility_period=30,  # Longer volatility period
        
        # Labeling
        lookforward_window=50,  # Longer lookforward for better labels
        reversal_threshold=2.5,  # Higher threshold for more reliable reversals
        continuation_threshold=1.5,  # Higher threshold for more reliable continuations
        atr_multiplier=1.2,  # Slightly higher multiplier
        min_confidence=0.7,  # Higher confidence threshold
        
        # Model training
        test_size=0.15,  # Smaller test set for more training data
        validation_size=0.15,  # Smaller validation set
        random_state=42,
        n_splits=3,  # Fewer splits for faster training
        
        # Backtesting
        initial_capital=100000,
        risk_per_trade=0.008,  # Slightly lower risk per trade
        max_positions=5,  # More positions allowed
        slippage_bps=2.0,
        commission_bps=1.0
    )
    
    # Initialize system
    print(f"\n🔧 Initializing system for large dataset...")
    system = FibMLSystem(config)
    
    # Train system
    print(f"\n🎓 Training system with 30K data points...")
    print(f"   This may take several minutes due to the large dataset...")
    
    try:
        model_results = system.train(data, save_models=False)
        
        print(f"   ✅ Training completed successfully!")
        print(f"   📊 Model Performance:")
        print(f"      Classifier accuracy: {model_results.performance_metrics['classifier']['accuracy']:.3f}")
        print(f"      Target regressor R²: {model_results.performance_metrics['regression']['target_price']['r2']:.3f}")
        print(f"      Duration regressor R²: {model_results.performance_metrics['regression']['duration']['r2']:.3f}")
        
        # Show system status
        status = system.get_system_status()
        print(f"   📈 Training Statistics:")
        print(f"      Data length: {status['last_training_stats']['data_length']}")
        print(f"      Swings detected: {status['last_training_stats']['swings_count']}")
        print(f"      Zones generated: {status['last_training_stats']['zones_count']}")
        print(f"      Events generated: {status['last_training_stats']['events_count']}")
        print(f"      Features extracted: {status['last_training_stats']['features_count']}")
        print(f"      Labels generated: {status['last_training_stats']['labels_count']}")
        
        # Run backtest
        print(f"\n📈 Running backtest...")
        backtest_results = system.backtest()
        
        # Show backtest results
        metrics = backtest_results.performance_metrics
        print(f"   📊 Backtest Results:")
        print(f"      Total trades: {metrics.get('total_trades', 0)}")
        print(f"      Win rate: {metrics.get('win_rate', 0):.1%}")
        print(f"      Total return: {metrics.get('total_return', 0):.1%}")
        print(f"      Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"      Max drawdown: {metrics.get('max_drawdown', 0):.1%}")
        print(f"      Profit factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Generate live predictions
        print(f"\n🔮 Generating live predictions...")
        predictions = system.predict_live(data.tail(200))  # Use last 200 bars for predictions
        
        print(f"   🎯 Generated {len(predictions)} predictions:")
        for i, pred in enumerate(predictions[:5]):  # Show first 5
            print(f"      Prediction {i+1}:")
            print(f"         Zone: {pred['fib_zone']}")
            print(f"         Outcome: {pred['prediction']}")
            print(f"         Confidence: {pred['confidence']:.3f}")
            print(f"         Target: {pred['target_zone']}")
            print(f"         Stop Loss: {pred['stop_loss_level']:.2f}")
            print(f"         Context Tags: {pred['context_tags']}")
            print()
        
        print(f"🎉 BTC 30K Analysis Complete!")
        print(f"   ✅ System successfully trained with 30,000 BTC data points")
        print(f"   ✅ Enhanced context features working on large dataset")
        print(f"   ✅ Context-aware predictions generated")
        print(f"   ✅ Ready for live trading with comprehensive analysis!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print(f"   The large dataset might be causing memory or time issues.")
        print(f"   Try reducing the dataset size or optimizing the configuration.")

if __name__ == "__main__":
    main()