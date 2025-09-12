"""
Demo Script for Fibonacci ML System

Demonstrates the complete system with sample data and shows results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fib_ml_system import FibMLSystem, SystemConfig
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_realistic_sample_data(length=1000, start_price=100.0, volatility=0.02):
    """Create more realistic sample data with trends and patterns."""
    np.random.seed(42)
    
    # Generate trend components
    trend = np.linspace(0, 0.2, length)  # 20% upward trend
    cycle = 0.1 * np.sin(2 * np.pi * np.arange(length) / 100)  # Cyclical component
    
    # Generate random walk with trend
    returns = np.random.normal(0, volatility, length)
    prices = start_price * np.exp(np.cumsum(returns + trend/100 + cycle/100))
    
    # Create OHLCV data
    data = pd.DataFrame(index=pd.date_range('2023-01-01', periods=length, freq='1H'))
    data['close'] = prices
    
    # Generate realistic OHLC from close prices
    noise = np.random.normal(0, volatility/4, length)
    data['high'] = data['close'] * (1 + np.abs(noise) + volatility/2)
    data['low'] = data['close'] * (1 - np.abs(noise) - volatility/2)
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['volume'] = np.random.lognormal(10, 1, length)
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def plot_results(system, sample_data, backtest_results):
    """Plot system results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Price data with swings
    ax1 = axes[0, 0]
    ax1.plot(sample_data.index, sample_data['close'], label='Close Price', alpha=0.7)
    
    # Add swings
    swings = system.last_training_data['swings']
    for swing in swings:
        color = 'red' if swing.swing_type == 'high' else 'green'
        ax1.scatter(swing.timestamp, swing.price, color=color, s=50, alpha=0.8)
    
    ax1.set_title('Price Data with Detected Swings')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity curve
    ax2 = axes[0, 1]
    if not backtest_results.equity_curve.empty:
        ax2.plot(backtest_results.equity_curve.index, backtest_results.equity_curve['equity'])
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Equity')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown curve
    ax3 = axes[1, 0]
    if not backtest_results.drawdown_curve.empty:
        ax3.fill_between(backtest_results.drawdown_curve.index, 
                        backtest_results.drawdown_curve['drawdown'], 
                        alpha=0.3, color='red')
        ax3.set_title('Drawdown Curve')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    ax4 = axes[1, 1]
    metrics = backtest_results.performance_metrics
    metric_names = ['Win Rate', 'Total Return', 'Sharpe Ratio', 'Profit Factor']
    metric_values = [
        metrics.get('win_rate', 0) * 100,
        metrics.get('total_return', 0) * 100,
        metrics.get('sharpe_ratio', 0),
        metrics.get('profit_factor', 0)
    ]
    
    bars = ax4.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'purple'])
    ax4.set_title('Performance Metrics')
    ax4.set_ylabel('Value')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main demo function."""
    print("🚀 Fibonacci ML System Demo")
    print("=" * 60)
    
    # Create system configuration
    config = SystemConfig(
        swing_window=15,
        min_swing_strength=0.4,
        min_zone_size=0.003,
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.12,
        min_confidence=0.65,
        initial_capital=100000,
        risk_per_trade=0.015,
        max_positions=3
    )
    
    print(f"📊 Configuration:")
    print(f"   Swing window: {config.swing_window}")
    print(f"   Min swing strength: {config.min_swing_strength}")
    print(f"   Fibonacci ratios: {config.fib_ratios}")
    print(f"   Initial capital: ${config.initial_capital:,.0f}")
    print(f"   Risk per trade: {config.risk_per_trade:.1%}")
    
    # Initialize system
    system = FibMLSystem(config)
    
    # Create realistic sample data
    print(f"\n📈 Creating sample data...")
    sample_data = create_realistic_sample_data(1200, 100.0, 0.025)
    print(f"   Generated {len(sample_data)} bars of data")
    print(f"   Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    print(f"   Total return: {(sample_data['close'].iloc[-1] / sample_data['close'].iloc[0] - 1) * 100:.1f}%")
    
    # Train the system
    print(f"\n🧠 Training the system...")
    model_results = system.train(sample_data, save_models=False)
    
    # Show training results
    print(f"\n📊 Training Results:")
    classifier_metrics = model_results.performance_metrics['classifier']
    regression_metrics = model_results.performance_metrics['regression']
    
    print(f"   Classifier Accuracy: {classifier_metrics['accuracy']:.1%}")
    print(f"   ROC AUC Score: {classifier_metrics['roc_auc']:.3f}")
    print(f"   Target Regressor R²: {regression_metrics['target_price']['r2']:.3f}")
    print(f"   Duration Regressor R²: {regression_metrics['duration']['r2']:.3f}")
    
    # Show feature importance
    print(f"\n🔍 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(model_results.feature_importance.items())[:10]):
        print(f"   {i+1:2d}. {feature:<30} {importance:>8.1f}")
    
    # Run backtest
    print(f"\n💰 Running backtest...")
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
    
    # Generate live predictions
    print(f"\n🔮 Generating live predictions...")
    predictions = system.predict_live(sample_data.tail(100))
    
    print(f"   Generated {len(predictions)} predictions")
    
    if predictions:
        print(f"\n📋 Sample Predictions:")
        for i, pred in enumerate(predictions[:3]):
            print(f"   Prediction {i+1}:")
            print(f"     Zone: {pred['fib_zone']}")
            print(f"     Outcome: {pred['prediction']}")
            print(f"     Confidence: {pred['confidence']:.1%}")
            print(f"     Context: {', '.join(pred['context_tags'][:3])}")
    
    # Show system status
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
    
    # Plot results
    print(f"\n📊 Generating plots...")
    try:
        plot_results(system, sample_data, backtest_results)
        print("   Plots displayed successfully!")
    except Exception as e:
        print(f"   Could not display plots: {e}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   The Fibonacci ML system is ready for live trading!")

if __name__ == "__main__":
    main()