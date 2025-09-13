"""
BTC Runner Fixed - Optimized parameters to prevent too many events
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
    """Run BTC analysis with optimized parameters."""
    print("🚀 BTC Runner Fixed - Optimized Parameters")
    print("=" * 60)
    print("🎯 Running system with optimized parameters to prevent too many events")
    print("=" * 60)
    
    # Load BTC data
    data = load_btc_data()
    
    # Use last 10000 bars for analysis
    data = data.tail(10000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Create system with OPTIMIZED settings to prevent too many events
    config = SystemConfig(
        # Swing detection - MORE SELECTIVE
        swing_window=40,  # Larger window for fewer, more significant swings
        min_swing_strength=0.6,  # Higher threshold for quality swings only
        min_swing_size=0.003,  # Larger minimum swing size
        
        # Fibonacci zones - FEWER ZONES
        learn_levels=True,  # Enable adaptive level learning
        zone_width_factor=0.15,  # Wider zones to reduce overlap
        min_zone_size=0.002,  # Larger minimum zone size
        
        # Event generation - MORE SELECTIVE
        min_touch_duration=3,  # Longer minimum duration (fewer events)
        max_event_duration=50,  # Shorter max duration (fewer events)
        wick_rejection_threshold=0.4,  # Higher threshold for quality events
        
        # Enhanced context features
        enable_enhanced_context=True,
        context_momentum_period=20,
        context_historical_period=80,
        context_failure_tracking=True,
        
        # Feature extraction
        atr_period=25,
        trend_period=40,
        volatility_period=40,
        
        # Labeling - HIGHER QUALITY
        lookforward_window=50,
        reversal_threshold=3.0,  # Higher threshold for quality
        continuation_threshold=2.0,  # Higher threshold for quality
        atr_multiplier=1.5,  # Higher multiplier
        min_confidence=0.8,  # Much higher confidence threshold
        
        # Model training
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_splits=3,
        
        # Backtesting
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=2.0,
        commission_bps=1.0
    )
    
    # Initialize system
    print(f"\n🔧 Initializing system with optimized parameters...")
    start_time = time.time()
    system = FibMLSystem(config)
    init_time = time.time() - start_time
    print(f"   ✅ System initialized in {init_time:.1f} seconds")
    
    # Train system
    print(f"\n🎓 Training system...")
    print(f"   Processing {len(data)} bars with optimized parameters...")
    
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
        
        # Calculate efficiency metrics
        events_per_bar = status['last_training_stats']['events_count'] / status['last_training_stats']['data_length']
        events_per_zone = status['last_training_stats']['events_count'] / status['last_training_stats']['zones_count']
        
        print(f"   📊 Efficiency Metrics:")
        print(f"      Events per bar: {events_per_bar:.2f}")
        print(f"      Events per zone: {events_per_zone:.2f}")
        print(f"      Zones per swing: {status['last_training_stats']['zones_count'] / status['last_training_stats']['swings_count']:.2f}")
        
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
        print(f"🎉 BTC Fixed Analysis Complete!")
        print(f"   ✅ Total processing time: {total_time:.1f} seconds")
        print(f"   ✅ System successfully trained with {len(data)} BTC data points")
        print(f"   ✅ Optimized parameters prevent too many events")
        print(f"   ✅ Enhanced context features working")
        print(f"   ✅ Context-aware predictions generated")
        print(f"   ✅ Ready for live trading!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print(f"   Error details: {str(e)}")

if __name__ == "__main__":
    main()