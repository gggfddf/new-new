# Adaptive Fibonacci ML System

A machine learning system that learns which Fibonacci retracement/extension ranges matter in different market contexts and predicts reversal vs continuation outcomes with calibrated probabilities.

## 🎯 Overview

This system uses **price-action only** analysis (OHLCV + swings) to:
- Detect market swings and pivots
- Generate Fibonacci retracement/extension zones
- Extract comprehensive price-action features
- Train LightGBM models for outcome prediction
- Provide real-time trading signals with risk management

## 🏗️ Architecture

```
[Live Feed] → [Swing Detector] → [Zone Generator] → [Feature Extractor] 
→ [LightGBM Model] → [Risk Filter] → [Execution Engine]
```

## ✨ Features

- **Price-Action Only**: No lagging technical indicators (RSI, MACD, etc.)
- **Adaptive Learning**: Models learn which Fibonacci levels work in different contexts
- **Comprehensive Features**: Market regime, impulse, zone interaction, and structural features
- **Multi-Task Learning**: Classification (outcome) + Regression (targets, duration)
- **Probability Calibration**: Calibrated confidence scores for live trading
- **Realistic Backtesting**: Includes slippage, commissions, and risk management
- **Live Pipeline**: Real-time inference and execution capabilities

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from fib_ml_system import FibMLSystem, SystemConfig

# Create system with configuration
config = SystemConfig(
    swing_window=20,
    min_swing_strength=0.5,
    fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
    min_confidence=0.65
)

system = FibMLSystem(config)

# Train on historical data
model_results = system.train(ohlcv_data)

# Run backtest
backtest_results = system.backtest()

# Generate live predictions
predictions = system.predict_live(current_data)
```

### Demo

```bash
python demo.py
```

### Run Tests

```bash
python test_system.py
```

## 📊 System Components

### 1. Swing Detection (`swing_detector.py`)
- Detects market swings using rolling window pivot analysis
- Filters swings by strength and significance
- Groups swings into sequences for zone generation

### 2. Fibonacci Zones (`fib_zones.py`)
- Generates retracement and extension zones from swing pairs
- Creates zones as price ranges (not single levels)
- Filters zones by strength and price range

### 3. Event Generation (`event_generator.py`)
- Detects when price touches Fibonacci zones
- Tracks zone entry, exit, and interaction patterns
- Calculates event metrics (duration, volume, wick rejection)

### 4. Feature Extraction (`features.py`)
- **Market Regime**: Trend direction, volatility regime, ATR analysis
- **Impulse Features**: Duration, strength, acceleration, volume
- **Zone Interaction**: Entry position, wick rejection, retest count
- **Structural**: Swing width, zone width, Fibonacci level
- **Temporal**: Session hour, day of week, cyclical patterns

### 5. Labeling (`labeling.py`)
- Assigns outcome labels based on future price action
- **Reversal**: 2R+ gain before 1R loss
- **Continuation**: 1R+ loss before 2R gain
- **Breakout**: Slow drift beyond zone
- Includes target levels and stop losses

### 6. Model Training (`model_train.py`)
- **LightGBM Classifier**: Outcome prediction (Reversal/Continuation/Breakout)
- **LightGBM Regressors**: Target price and duration prediction
- **Probability Calibration**: Isotonic calibration for confidence scores
- **Feature Importance**: Identifies most predictive features

### 7. Backtesting (`backtest.py`)
- Realistic simulation with slippage and commissions
- Risk management (position sizing, stop losses)
- Comprehensive performance metrics
- Equity curve and drawdown analysis

### 8. Live Pipeline (`live_pipeline.py`)
- Real-time data processing and inference
- Trade execution and monitoring
- Performance tracking and logging
- Risk management integration

## 📈 Output Format

The system produces predictions in this format:

```python
{
    "fib_zone": [0.60, 0.65],          # Zone touched
    "prediction": "Reversal",          # {Reversal, Continuation, Breakout}
    "confidence": 0.82,                # Calibrated probability
    "target_zone": [0.3, 1.5, 0.6],   # Next Fib extension target
    "stop_loss_level": 0.58,           # Suggested SL
    "context_tags": ["bull_trend", "high_vol", "impulse"]  
}
```

## ⚙️ Configuration

```python
config = SystemConfig(
    # Swing Detection
    swing_window=20,
    min_swing_strength=0.5,
    min_swing_size=0.001,
    
    # Fibonacci Zones
    fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.414, 1.618],
    zone_width_factor=0.1,
    min_zone_size=0.001,
    
    # Event Generation
    min_touch_duration=1,
    max_event_duration=100,
    wick_rejection_threshold=0.3,
    
    # Feature Extraction
    atr_period=14,
    trend_period=20,
    volatility_period=20,
    
    # Labeling
    lookforward_window=30,
    reversal_threshold=2.0,
    continuation_threshold=1.0,
    atr_multiplier=1.0,
    min_confidence=0.6,
    
    # Backtesting
    initial_capital=100000,
    risk_per_trade=0.01,
    max_positions=3,
    slippage_bps=2.0,
    commission_bps=1.0
)
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit / Gross loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Expectancy**: Expected return per trade
- **Risk-Reward Ratio**: Average reward / Average risk

## 🔧 Advanced Usage

### Custom Data Sources

```python
# Load your own OHLCV data
data = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
data.columns = ['open', 'high', 'low', 'close', 'volume']

# Train system
system = FibMLSystem()
model_results = system.train(data)
```

### Model Persistence

```python
# Save trained models
system.train(data, save_models=True, model_path="my_models")

# Load pre-trained models
system.load_models("my_models")
```

### Live Trading Integration

```python
# Initialize live pipeline
pipeline = LivePipeline(model_results, config)

# Start live trading (requires data feed and execution engine)
await pipeline.start_live_trading(data_feed, execution_engine)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_system.py
```

Tests cover:
- Individual component functionality
- Integration testing
- Performance validation
- Edge case handling

## 📚 Examples

See the `demo.py` script for a complete example showing:
- System initialization and configuration
- Training on sample data
- Backtesting results
- Live prediction generation
- Performance visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.