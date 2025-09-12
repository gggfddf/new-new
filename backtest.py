"""
Backtesting Module

Simulates trading strategy with realistic slippage, commissions, and risk management.
Evaluates strategy performance with comprehensive metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from model_train import ModelResults
from features import FeatureExtractor
from event_generator import ZoneEvent
from labeling import OutcomeLabel


@dataclass
class Trade:
    """Represents a single trade."""
    trade_id: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: str = 'long'  # 'long' or 'short'
    size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    outcome: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration_bars: Optional[int] = None
    max_drawdown: Optional[float] = None
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    commission: float = 0.0
    context: Dict = None


@dataclass
class BacktestResults:
    """Container for backtest results."""
    trades: List[Trade]
    performance_metrics: Dict
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    trade_analysis: Dict


class BacktestEngine:
    """
    Backtesting engine for the Fibonacci ML strategy.
    
    Simulates realistic trading with slippage, commissions, and risk management.
    Evaluates strategy performance with comprehensive metrics.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.01,
                 max_positions: int = 3,
                 slippage_bps: float = 2.0,
                 commission_bps: float = 1.0,
                 min_confidence: float = 0.65):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            risk_per_trade: Risk per trade as fraction of capital
            max_positions: Maximum concurrent positions
            slippage_bps: Slippage in basis points
            commission_bps: Commission in basis points
            min_confidence: Minimum confidence threshold for trades
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.min_confidence = min_confidence
        
        self.current_capital = initial_capital
        self.active_trades = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, 
                    data: pd.DataFrame,
                    model_results: ModelResults,
                    events: List[ZoneEvent],
                    feature_extractor: FeatureExtractor,
                    swings: List) -> BacktestResults:
        """
        Run the backtest simulation.
        
        Args:
            data: OHLCV DataFrame
            model_results: Trained model results
            events: List of zone events
            feature_extractor: Feature extractor
            swings: List of swings
            
        Returns:
            BacktestResults object
        """
        print("Starting backtest simulation...")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.active_trades = []
        self.trades = []
        self.equity_curve = []
        
        # Process each bar
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Update active trades
            self._update_active_trades(data, i, timestamp, row)
            
            # Check for new trading opportunities
            self._check_trading_opportunities(
                data, i, timestamp, row, model_results, events, feature_extractor, swings
            )
            
            # Record equity
            self._record_equity(timestamp, i)
        
        # Close any remaining trades
        self._close_all_trades(data.iloc[-1])
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Generate equity and drawdown curves
        equity_curve = self._generate_equity_curve()
        drawdown_curve = self._generate_drawdown_curve()
        monthly_returns = self._calculate_monthly_returns()
        
        # Analyze trades
        trade_analysis = self._analyze_trades()
        
        return BacktestResults(
            trades=self.trades,
            performance_metrics=performance_metrics,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
            trade_analysis=trade_analysis
        )
    
    def _update_active_trades(self, 
                            data: pd.DataFrame,
                            bar_index: int,
                            timestamp: pd.Timestamp,
                            row: pd.Series) -> None:
        """Update active trades and check for exits."""
        trades_to_close = []
        
        for trade in self.active_trades:
            # Check stop loss
            if trade.stop_loss:
                if trade.direction == 'long' and row['low'] <= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.outcome = 'stop_loss'
                    trades_to_close.append(trade)
                elif trade.direction == 'short' and row['high'] >= trade.stop_loss:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.stop_loss
                    trade.outcome = 'stop_loss'
                    trades_to_close.append(trade)
            
            # Check take profit
            if trade.take_profit:
                if trade.direction == 'long' and row['high'] >= trade.take_profit:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.take_profit
                    trade.outcome = 'take_profit'
                    trades_to_close.append(trade)
                elif trade.direction == 'short' and row['low'] <= trade.take_profit:
                    trade.exit_time = timestamp
                    trade.exit_price = trade.take_profit
                    trade.outcome = 'take_profit'
                    trades_to_close.append(trade)
            
            # Check time-based exit (max duration)
            if trade.entry_time and (timestamp - trade.entry_time).total_seconds() > 86400 * 7:  # 7 days
                trade.exit_time = timestamp
                trade.exit_price = row['close']
                trade.outcome = 'time_exit'
                trades_to_close.append(trade)
        
        # Close trades
        for trade in trades_to_close:
            self._close_trade(trade, row)
    
    def _check_trading_opportunities(self, 
                                   data: pd.DataFrame,
                                   bar_index: int,
                                   timestamp: pd.Timestamp,
                                   row: pd.Series,
                                   model_results: ModelResults,
                                   events: List[ZoneEvent],
                                   feature_extractor: FeatureExtractor,
                                   swings: List) -> None:
        """Check for new trading opportunities."""
        # Skip if at max positions
        if len(self.active_trades) >= self.max_positions:
            return
        
        # Find events at current bar
        current_events = [e for e in events if e.entry_index == bar_index]
        
        for event in current_events:
            # Extract features
            feature_set = feature_extractor.extract_all_features(data, event, swings)
            
            # Make prediction
            prediction = self._make_prediction(model_results, feature_set)
            
            if prediction and prediction['confidence'] >= self.min_confidence:
                # Create trade
                trade = self._create_trade(event, prediction, timestamp, row)
                if trade:
                    self.active_trades.append(trade)
    
    def _make_prediction(self, 
                        model_results: ModelResults,
                        feature_set) -> Optional[Dict]:
        """Make prediction using trained models."""
        try:
            # Prepare features
            features_df = pd.DataFrame([feature_set.raw_features])
            
            # Classify outcome
            outcome_proba = model_results.classifier.predict(
                features_df, num_iteration=model_results.classifier.best_iteration
            )
            outcome_idx = np.argmax(outcome_proba)
            outcome = model_results.classifier.reverse_encoder[outcome_idx]
            confidence = outcome_proba[outcome_idx]
            
            # Predict target price
            target_price = model_results.target_regressor.predict(
                features_df, num_iteration=model_results.target_regressor.best_iteration
            )[0]
            
            # Predict duration
            duration = model_results.duration_regressor.predict(
                features_df, num_iteration=model_results.duration_regressor.best_iteration
            )[0]
            
            return {
                'outcome': outcome,
                'confidence': confidence,
                'target_price': target_price,
                'duration': duration
            }
        except Exception as e:
            warnings.warn(f"Prediction failed: {e}")
            return None
    
    def _create_trade(self, 
                     event: ZoneEvent,
                     prediction: Dict,
                     timestamp: pd.Timestamp,
                     row: pd.Series) -> Optional[Trade]:
        """Create a new trade based on prediction."""
        # Determine trade direction based on outcome
        if prediction['outcome'] == 'Reversal':
            # Reversal trade - opposite to trend
            if event.zone.swing_start.swing_type == 'high':
                direction = 'long'  # Buy the dip
            else:
                direction = 'short'  # Sell the rally
        elif prediction['outcome'] == 'Continuation':
            # Continuation trade - same as trend
            if event.zone.swing_start.swing_type == 'high':
                direction = 'short'  # Continue downtrend
            else:
                direction = 'long'  # Continue uptrend
        else:  # Breakout
            # Breakout trade - follow the break
            if row['close'] > event.zone.price_range[1]:
                direction = 'long'
            else:
                direction = 'short'
        
        # Calculate position size
        risk_amount = self.current_capital * self.risk_per_trade
        entry_price = row['close']
        
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(event, prediction, direction, entry_price)
        
        # Calculate take profit
        take_profit = self._calculate_take_profit(event, prediction, direction, entry_price)
        
        # Calculate position size based on risk
        if stop_loss:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                size = risk_amount / risk_per_share
            else:
                return None
        else:
            size = risk_amount / entry_price
        
        # Apply slippage
        slippage = entry_price * self.slippage_bps / 10000
        if direction == 'long':
            entry_price += slippage
        else:
            entry_price -= slippage
        
        # Calculate commission
        commission = entry_price * size * self.commission_bps / 10000
        
        # Create trade
        trade = Trade(
            trade_id=f"trade_{len(self.trades) + 1}",
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            slippage_entry=slippage,
            commission=commission,
            context={
                'event_id': event.event_id,
                'zone_type': event.zone.zone_type,
                'zone_level': event.zone.level,
                'prediction': prediction
            }
        )
        
        return trade
    
    def _calculate_stop_loss(self, 
                           event: ZoneEvent,
                           prediction: Dict,
                           direction: str,
                           entry_price: float) -> float:
        """Calculate stop loss level."""
        zone_lower, zone_upper = event.zone.price_range
        
        if direction == 'long':
            # Long trade - stop below zone
            stop_loss = zone_lower * 0.995  # 0.5% below zone
        else:
            # Short trade - stop above zone
            stop_loss = zone_upper * 1.005  # 0.5% above zone
        
        return stop_loss
    
    def _calculate_take_profit(self, 
                             event: ZoneEvent,
                             prediction: Dict,
                             direction: str,
                             entry_price: float) -> float:
        """Calculate take profit level."""
        target_price = prediction.get('target_price', entry_price)
        
        if direction == 'long':
            # Long trade - target above entry
            take_profit = max(target_price, entry_price * 1.02)  # At least 2% gain
        else:
            # Short trade - target below entry
            take_profit = min(target_price, entry_price * 0.98)  # At least 2% gain
        
        return take_profit
    
    def _close_trade(self, trade: Trade, row: pd.Series) -> None:
        """Close a trade and calculate P&L."""
        if trade.exit_price is None:
            trade.exit_price = row['close']
        
        # Apply slippage to exit
        slippage = trade.exit_price * self.slippage_bps / 10000
        if trade.direction == 'long':
            trade.exit_price -= slippage
        else:
            trade.exit_price += slippage
        
        trade.slippage_exit = slippage
        
        # Calculate P&L
        if trade.direction == 'long':
            pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - trade.exit_price) * trade.size
        
        # Subtract commission
        exit_commission = trade.exit_price * trade.size * self.commission_bps / 10000
        pnl -= trade.commission + exit_commission
        
        trade.pnl = pnl
        trade.pnl_pct = pnl / (trade.entry_price * trade.size)
        trade.duration_bars = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Hours
        
        # Update capital
        self.current_capital += pnl
        
        # Move to completed trades
        self.active_trades.remove(trade)
        self.trades.append(trade)
    
    def _close_all_trades(self, last_row: pd.Series) -> None:
        """Close all remaining active trades."""
        for trade in self.active_trades[:]:
            trade.exit_time = last_row.name
            trade.exit_price = last_row['close']
            trade.outcome = 'end_of_data'
            self._close_trade(trade, last_row)
    
    def _record_equity(self, timestamp: pd.Timestamp, bar_index: int) -> None:
        """Record current equity."""
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for trade in self.active_trades:
            # This is simplified - in reality you'd need current price
            unrealized_pnl += 0  # Placeholder
        
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'bar_index': bar_index,
            'equity': total_equity,
            'capital': self.current_capital,
            'unrealized_pnl': unrealized_pnl
        })
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        # Risk metrics
        returns = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / self.initial_capital
        
        # Return metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        annualized_return = total_return * 252 / len(self.equity_curve) if self.equity_curve else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy,
            'final_capital': self.current_capital
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.equity_curve:
            return 0
        
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _generate_equity_curve(self) -> pd.DataFrame:
        """Generate equity curve DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _generate_drawdown_curve(self) -> pd.DataFrame:
        """Generate drawdown curve DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        equity_values = [point['equity'] for point in self.equity_curve]
        timestamps = [point['timestamp'] for point in self.equity_curve]
        
        peak = equity_values[0]
        drawdowns = []
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            drawdowns.append(dd)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'drawdown': drawdowns
        })
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        
        monthly_equity = df['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return pd.DataFrame({
            'month': monthly_returns.index,
            'return': monthly_returns.values
        })
    
    def _analyze_trades(self) -> Dict:
        """Analyze trade characteristics."""
        if not self.trades:
            return {}
        
        # Trade duration analysis
        durations = [t.duration_bars for t in self.trades if t.duration_bars is not None]
        
        # Outcome analysis
        outcomes = [t.outcome for t in self.trades if t.outcome]
        outcome_counts = pd.Series(outcomes).value_counts().to_dict()
        
        # Direction analysis
        directions = [t.direction for t in self.trades]
        direction_counts = pd.Series(directions).value_counts().to_dict()
        
        # P&L distribution
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        
        return {
            'avg_duration_hours': np.mean(durations) if durations else 0,
            'outcome_distribution': outcome_counts,
            'direction_distribution': direction_counts,
            'pnl_stats': {
                'mean': np.mean(pnls) if pnls else 0,
                'std': np.std(pnls) if pnls else 0,
                'min': np.min(pnls) if pnls else 0,
                'max': np.max(pnls) if pnls else 0
            }
        }


def test_backtest_engine():
    """Test the backtest engine with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    from event_generator import EventGenerator
    from features import FeatureExtractor
    from labeling import LabelGenerator
    from model_train import ModelTrainer
    
    print("Testing Backtest Engine...")
    
    # Create sample data
    sample_data = create_sample_data(1000, 100.0, 0.02)
    print(f"Created sample data with {len(sample_data)} bars")
    
    # Detect swings
    detector = SwingDetector(window=8, min_swing_strength=0.2, min_swing_size=0.003)
    swings = detector.detect_swings(sample_data)
    print(f"Detected {len(swings)} swings")
    
    if len(swings) < 2:
        print("Not enough swings for zone generation")
        return
    
    # Generate Fibonacci zones
    zone_generator = FibZoneGenerator(
        standard_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.15,
        min_zone_size=0.002
    )
    
    zones = zone_generator.generate_all_zones(swings)
    zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
    print(f"Generated {len(zones)} zones")
    
    # Generate events
    event_generator = EventGenerator(
        min_touch_duration=1,
        max_event_duration=50,
        wick_rejection_threshold=0.3
    )
    
    events = event_generator.detect_zone_touches(sample_data, zones)
    print(f"Generated {len(events)} events")
    
    if not events:
        print("No events generated for backtesting")
        return
    
    # Extract features and generate labels
    feature_extractor = FeatureExtractor()
    feature_sets = []
    for event in events:
        feature_set = feature_extractor.extract_all_features(sample_data, event, swings)
        feature_sets.append(feature_set)
    
    features_df = feature_extractor.export_features_to_dataframe(feature_sets)
    
    label_generator = LabelGenerator()
    labels = label_generator.generate_labels(sample_data, events, zones)
    labels_df = label_generator.export_labels_to_dataframe(labels)
    
    if features_df.empty or labels_df.empty:
        print("No features or labels for training")
        return
    
    # Train models
    trainer = ModelTrainer()
    model_results = trainer.train_all_models(features_df, labels_df)
    
    # Run backtest
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        risk_per_trade=0.01,
        max_positions=3,
        slippage_bps=2.0,
        commission_bps=1.0,
        min_confidence=0.6
    )
    
    results = backtest_engine.run_backtest(
        sample_data, model_results, events, feature_extractor, swings
    )
    
    # Print results
    print("\nBacktest Results:")
    metrics = results.performance_metrics
    print(f"Total trades: {metrics.get('total_trades', 0)}")
    print(f"Win rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Total return: {metrics.get('total_return', 0):.1%}")
    print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max drawdown: {metrics.get('max_drawdown', 0):.1%}")
    print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Expectancy: {metrics.get('expectancy', 0):.4f}")
    
    print("Backtest test completed successfully!")


if __name__ == "__main__":
    test_backtest_engine()