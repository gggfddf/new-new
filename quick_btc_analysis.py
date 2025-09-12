"""
Quick BTC Analysis - Run Fibonacci ML System on Real BTC Data
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os

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
    
    print(f"   Data shape: {ohlcv_data.shape}")
    print(f"   Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    
    return ohlcv_data

def main():
    """Run analysis."""
    print("🚀 BTC Fibonacci ML Analysis")
    print("=" * 50)
    
    # Load data
    data = load_btc_data()
    
    # Use last 5000 bars for faster analysis
    data = data.tail(5000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Configure system
    config = SystemConfig(
        swing_window=20,
        min_swing_strength=0.3,
        min_zone_size=0.002,
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.1,
        min_confidence=0.6,
        initial_capital=100000,
        risk_per_trade=0.01
    )
    
    # Initialize system
    system = FibMLSystem(config)
    
    # Train system
    print(f"\n🧠 Training system...")
    try:
        model_results = system.train(data, save_models=False)
        
        print(f"\n📊 Training Results:")
        classifier_metrics = model_results.performance_metrics['classifier']
        print(f"   Accuracy: {classifier_metrics['accuracy']:.1%}")
        print(f"   ROC AUC: {classifier_metrics['roc_auc']:.3f}")
        
        # Feature importance
        print(f"\n🔍 Top 10 Features:")
        for i, (feature, importance) in enumerate(list(model_results.feature_importance.items())[:10]):
            print(f"   {i+1:2d}. {feature:<30} {importance:>8.1f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Run backtest
    print(f"\n💰 Running backtest...")
    try:
        backtest_results = system.backtest()
        
        metrics = backtest_results.performance_metrics
        print(f"\n📈 Backtest Results:")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   Total Return: {metrics.get('total_return', 0):.1%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
    
    # Generate predictions
    print(f"\n🔮 Generating predictions...")
    try:
        predictions = system.predict_live(data.tail(100))
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
    
    # System status
    print(f"\n⚙️ System Status:")
    status = system.get_system_status()
    if status['last_training_data_available']:
        stats = status['last_training_stats']
        print(f"   Swings Detected: {stats['swings_count']}")
        print(f"   Zones Generated: {stats['zones_count']}")
        print(f"   Events Created: {stats['events_count']}")
        print(f"   Features Extracted: {stats['features_count']}")
        print(f"   Labels Generated: {stats['labels_count']}")
    
    print(f"\n🎉 Analysis completed!")

if __name__ == "__main__":
    main()