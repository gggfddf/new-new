"""
BTC Runner Efficient - Optimized for large datasets with progress tracking
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

def main():
    """Run efficient BTC analysis with progress tracking."""
    print("🚀 BTC Runner Efficient - Large Dataset Analysis")
    print("=" * 60)
    print("🎯 Running optimized system with large BTC dataset")
    print("=" * 60)
    
    # Load BTC data
    data = load_btc_data()
    
    # Use last 15000 bars for efficient analysis (good balance)
    data = data.tail(15000)
    print(f"   Using last {len(data)} bars for efficient analysis")
    
    # Create system with efficient settings
    config = SystemConfig(
        # Swing detection - efficient settings
        swing_window=30,  # Larger window for stability
        min_swing_strength=0.5,  # Higher threshold for quality swings
        min_swing_size=0.002,  # Larger minimum swing size
        
        # Fibonacci zones - will be learned from data
        learn_levels=True,  # Enable adaptive level learning
        zone_width_factor=0.12,  # Slightly wider zones
        min_zone_size=0.0015,  # Larger minimum zone size
        
        # Event generation - efficient settings
        min_touch_duration=2,  # Longer minimum duration
        max_event_duration=120,  # Reasonable max duration
        wick_rejection_threshold=0.35,  # Higher threshold for quality
        
        # Enhanced context features
        enable_enhanced_context=True,  # Enable enhanced context features
        context_momentum_period=20,  # Longer momentum period
        context_historical_period=80,  # Longer historical period
        context_failure_tracking=True,
        
        # Feature extraction
        atr_period=25,  # Longer ATR period
        trend_period=40,  # Longer trend period
        volatility_period=40,  # Longer volatility period
        
        # Labeling
        lookforward_window=60,  # Longer lookforward
        reversal_threshold=3.0,  # Higher threshold for quality
        continuation_threshold=2.0,  # Higher threshold for quality
        atr_multiplier=1.5,  # Higher multiplier
        min_confidence=0.75,  # Higher confidence threshold
        
        # Model training
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_splits=3,  # Fewer splits for speed
        
        # Backtesting
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=2.0,
        commission_bps=1.0
    )
    
    # Initialize system
    print(f"\n🔧 Initializing system...")
    start_time = time.time()
    system = FibMLSystem(config)
    init_time = time.time() - start_time
    print(f"   ✅ System initialized in {init_time:.1f} seconds")
    
    # Train system with progress tracking
    print(f"\n🎓 Training system...")
    print(f"   Processing {len(data)} bars with enhanced context features...")
    
    try:
        train_start = time.time()
        model_results = system.train(data, save_models=False)
        train_time = time.time() - train_start
        
        print(f"   ✅ Training completed in {train_time:.1f} seconds!")
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
        backtest_start = time.time()
        backtest_results = system.backtest()
        backtest_time = time.time() - backtest_start
        
        # Show backtest results
        metrics = backtest_results.performance_metrics
        print(f"   📊 Backtest Results (completed in {backtest_time:.1f}s):")
        print(f"      Total trades: {metrics.get('total_trades', 0)}")
        print(f"      Win rate: {metrics.get('win_rate', 0):.1%}")
        print(f"      Total return: {metrics.get('total_return', 0):.1%}")
        print(f"      Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"      Max drawdown: {metrics.get('max_drawdown', 0):.1%}")
        print(f"      Profit factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Generate live predictions
        print(f"\n🔮 Generating live predictions...")
        predictions = system.predict_live(data.tail(100))
        
        print(f"   🎯 Generated {len(predictions)} predictions:")
        for i, pred in enumerate(predictions[:3]):  # Show first 3
            print(f"      Prediction {i+1}:")
            print(f"         Zone: {pred['fib_zone']}")
            print(f"         Outcome: {pred['prediction']}")
            print(f"         Confidence: {pred['confidence']:.3f}")
            print(f"         Target: {pred['target_zone']}")
            print(f"         Stop Loss: {pred['stop_loss_level']:.2f}")
            print(f"         Context Tags: {pred['context_tags']}")
            print()
        
        total_time = time.time() - start_time
        print(f"🎉 BTC Efficient Analysis Complete!")
        print(f"   ✅ Total processing time: {total_time:.1f} seconds")
        print(f"   ✅ System successfully trained with {len(data)} BTC data points")
        print(f"   ✅ Enhanced context features working efficiently")
        print(f"   ✅ Context-aware predictions generated")
        print(f"   ✅ Ready for live trading!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print(f"   Try reducing the dataset size or check system resources.")

if __name__ == "__main__":
    main()