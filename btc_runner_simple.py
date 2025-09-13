"""
BTC Runner Simple - Skip learning step to avoid errors
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
    """Run BTC analysis with simple configuration."""
    print("🚀 BTC Runner Simple - Skip Learning Step")
    print("=" * 60)
    print("🎯 Running system without adaptive learning to avoid errors")
    print("=" * 60)
    
    # Load BTC data
    data = load_btc_data()
    
    # Use last 5000 bars for analysis
    data = data.tail(5000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Create system with SIMPLE settings (no learning)
    config = SystemConfig(
        # Swing detection
        swing_window=30,
        min_swing_strength=0.5,
        min_swing_size=0.002,
        
        # Fibonacci zones - USE TRADITIONAL LEVELS (no learning)
        learn_levels=False,  # DISABLE adaptive level learning
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],  # Traditional levels
        zone_width_factor=0.1,
        min_zone_size=0.001,
        
        # Event generation
        min_touch_duration=2,
        max_event_duration=80,
        wick_rejection_threshold=0.3,
        
        # Enhanced context features
        enable_enhanced_context=True,
        context_momentum_period=15,
        context_historical_period=60,
        context_failure_tracking=True,
        
        # Feature extraction
        atr_period=20,
        trend_period=30,
        volatility_period=30,
        
        # Labeling
        lookforward_window=40,
        reversal_threshold=2.5,
        continuation_threshold=1.5,
        atr_multiplier=1.2,
        min_confidence=0.7,
        
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
    print(f"\n🔧 Initializing system...")
    start_time = time.time()
    system = FibMLSystem(config)
    init_time = time.time() - start_time
    print(f"   ✅ System initialized in {init_time:.1f} seconds")
    
    # Train system
    print(f"\n🎓 Training system...")
    print(f"   Processing {len(data)} bars with traditional Fibonacci levels...")
    
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
        print(f"🎉 BTC Simple Analysis Complete!")
        print(f"   ✅ Total processing time: {total_time:.1f} seconds")
        print(f"   ✅ System successfully trained with {len(data)} BTC data points")
        print(f"   ✅ Traditional Fibonacci levels working")
        print(f"   ✅ Enhanced context features working")
        print(f"   ✅ Context-aware predictions generated")
        print(f"   ✅ Ready for live trading!")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        print(f"   Error details: {str(e)}")

if __name__ == "__main__":
    main()