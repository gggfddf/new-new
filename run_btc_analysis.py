"""
Run Complete Fibonacci ML System on Real BTC 5-Minute Data

This script loads the real BTC data and runs the complete system analysis.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append('/workspace')

from fib_ml_system import FibMLSystem, SystemConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_btc_data(file_path):
    """Load and prepare BTC data."""
    print("📊 Loading BTC 5-minute data...")
    
    # Read the CSV file
    df = pd.read_csv(file_path, sep='\t')  # Tab-separated based on the format
    
    print(f"   Raw data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Clean and prepare the data
    # Remove the first row if it's a header
    if df.iloc[0, 0] == '<DATE>':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Rename columns to standard format
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    
    # Convert data types
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['vol'], errors='coerce')
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    
    # Select only OHLCV columns
    ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Remove any rows with NaN values
    ohlcv_data = ohlcv_data.dropna()
    
    print(f"   Cleaned data shape: {ohlcv_data.shape}")
    print(f"   Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    print(f"   Total return: {(ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[0] - 1) * 100:.1f}%")
    
    return ohlcv_data

def run_complete_analysis(data):
    """Run the complete Fibonacci ML system analysis."""
    print("\n🚀 Starting Complete Fibonacci ML System Analysis")
    print("=" * 60)
    
    # Configure system for BTC data
    config = SystemConfig(
        # Swing detection - adjusted for 5-minute data
        swing_window=24,  # 2 hours of 5-min bars
        min_swing_strength=0.3,
        min_swing_size=0.002,
        
        # Fibonacci zones
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.414, 1.618],
        zone_width_factor=0.08,  # Tighter zones for crypto
        min_zone_size=0.001,
        
        # Event generation
        min_touch_duration=1,
        max_event_duration=200,  # ~16 hours max
        wick_rejection_threshold=0.25,
        
        # Feature extraction
        atr_period=14,
        trend_period=20,
        volatility_period=20,
        
        # Labeling
        lookforward_window=60,  # 5 hours lookforward
        reversal_threshold=1.5,  # Adjusted for crypto volatility
        continuation_threshold=0.8,
        atr_multiplier=1.0,
        min_confidence=0.6,
        
        # Backtesting
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=5.0,  # Higher slippage for crypto
        commission_bps=2.0,  # Higher commission for crypto
        
        # Model training
        test_size=0.2,
        validation_size=0.2,
        random_state=42
    )
    
    print(f"📋 System Configuration:")
    print(f"   Swing window: {config.swing_window} bars ({config.swing_window * 5} minutes)")
    print(f"   Fibonacci ratios: {config.fib_ratios}")
    print(f"   Lookforward window: {config.lookforward_window} bars ({config.lookforward_window * 5} minutes)")
    print(f"   Risk per trade: {config.risk_per_trade:.1%}")
    
    # Initialize system
    system = FibMLSystem(config)
    
    # Train the system
    print(f"\n🧠 Training the system on BTC data...")
    try:
        model_results = system.train(data, save_models=False)
        
        print(f"\n📊 Training Results:")
        classifier_metrics = model_results.performance_metrics['classifier']
        regression_metrics = model_results.performance_metrics['regression']
        
        print(f"   Classifier Accuracy: {classifier_metrics['accuracy']:.1%}")
        print(f"   ROC AUC Score: {classifier_metrics['roc_auc']:.3f}")
        print(f"   Target Regressor R²: {regression_metrics['target_price']['r2']:.3f}")
        print(f"   Duration Regressor R²: {regression_metrics['duration']['r2']:.3f}")
        
        # Show feature importance
        print(f"\n🔍 Top 15 Most Important Features:")
        for i, (feature, importance) in enumerate(list(model_results.feature_importance.items())[:15]):
            print(f"   {i+1:2d}. {feature:<35} {importance:>8.1f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None
    
    # Run backtest
    print(f"\n💰 Running backtest on BTC data...")
    try:
        backtest_results = system.backtest()
        
        # Show backtest results
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
        
        # Show trade analysis
        if backtest_results.trades:
            print(f"\n📊 Trade Analysis:")
            trade_analysis = backtest_results.trade_analysis
            print(f"   Average Duration: {trade_analysis.get('avg_duration_hours', 0):.1f} hours")
            print(f"   Outcome Distribution: {trade_analysis.get('outcome_distribution', {})}")
            
            # P&L statistics
            pnl_stats = trade_analysis.get('pnl_stats', {})
            print(f"   Average P&L: ${pnl_stats.get('mean', 0):.2f}")
            print(f"   P&L Std Dev: ${pnl_stats.get('std', 0):.2f}")
            print(f"   Best Trade: ${pnl_stats.get('max', 0):.2f}")
            print(f"   Worst Trade: ${pnl_stats.get('min', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        backtest_results = None
    
    # Generate live predictions
    print(f"\n🔮 Generating live predictions...")
    try:
        # Use last 200 bars for prediction
        recent_data = data.tail(200)
        predictions = system.predict_live(recent_data)
        
        print(f"   Generated {len(predictions)} predictions")
        
        if predictions:
            print(f"\n📋 Sample Predictions:")
            for i, pred in enumerate(predictions[:5]):
                print(f"   Prediction {i+1}:")
                print(f"     Zone: {pred['fib_zone']}")
                print(f"     Outcome: {pred['prediction']}")
                print(f"     Confidence: {pred['confidence']:.1%}")
                print(f"     Context: {', '.join(pred['context_tags'][:3])}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        predictions = []
    
    # Show system status and data statistics
    print(f"\n⚙️ System Status:")
    status = system.get_system_status()
    print(f"   Trained: {status['trained']}")
    print(f"   Training Data Available: {status['last_training_data_available']}")
    
    if status['last_training_data_available']:
        stats = status['last_training_stats']
        print(f"   Data Length: {stats['data_length']} bars")
        print(f"   Swings Detected: {stats['swings_count']}")
        print(f"   Zones Generated: {stats['zones_count']}")
        print(f"   Events Created: {stats['events_count']}")
        print(f"   Features Extracted: {stats['features_count']}")
        print(f"   Labels Generated: {stats['labels_count']}")
    
    return {
        'model_results': model_results,
        'backtest_results': backtest_results,
        'predictions': predictions,
        'system': system
    }

def main():
    """Main function to run the complete analysis."""
    print("🚀 BTC 5-Minute Data Analysis with Fibonacci ML System")
    print("=" * 70)
    
    # Load BTC data
    try:
        btc_data = load_btc_data('/workspace/btc_5min.csv')
    except Exception as e:
        print(f"❌ Failed to load BTC data: {e}")
        return
    
    # Run complete analysis
    try:
        results = run_complete_analysis(btc_data)
        
        if results:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"   The Fibonacci ML system has been trained on real BTC data")
            print(f"   and is ready for live trading predictions!")
        else:
            print(f"\n⚠️ Analysis completed with some issues")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()