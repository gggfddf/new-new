"""
Live Pipeline Module

Real-time inference and execution pipeline for the Fibonacci ML system.
Handles live data feeds, model inference, and trade execution.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import json
import time
from datetime import datetime, timedelta
import warnings
import redis
import sqlite3
from model_train import ModelResults, ModelTrainer
from features import FeatureExtractor
from event_generator import EventGenerator, ZoneEvent
from fib_zones import FibZoneGenerator
from swing_detector import SwingDetector
from backtest import BacktestEngine, Trade


@dataclass
class LivePrediction:
    """Container for live prediction results."""
    event_id: str
    fib_zone: Tuple[float, float]
    prediction: str  # 'Reversal', 'Continuation', 'Breakout'
    confidence: float
    target_zone: Optional[Tuple[float, float]]
    stop_loss_level: float
    context_tags: List[str]
    timestamp: datetime
    entry_price: float
    risk_reward_ratio: float


@dataclass
class ExecutionSignal:
    """Trading signal for execution."""
    signal_id: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    confidence: float
    context: Dict
    timestamp: datetime


class LivePipeline:
    """
    Live trading pipeline for the Fibonacci ML system.
    
    Handles real-time data processing, model inference, and trade execution.
    Includes risk management and position monitoring.
    """
    
    def __init__(self, 
                 model_results: ModelResults,
                 config: Dict,
                 data_store: str = 'sqlite',
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        """
        Initialize live pipeline.
        
        Args:
            model_results: Trained model results
            config: Configuration dictionary
            data_store: Data storage backend ('sqlite' or 'redis')
            redis_host: Redis host for caching
            redis_port: Redis port
        """
        self.model_results = model_results
        self.config = config
        
        # Initialize components
        self.swing_detector = SwingDetector(
            window=config.get('swing_window', 20),
            min_swing_strength=config.get('min_swing_strength', 0.5),
            min_swing_size=config.get('min_swing_size', 0.001)
        )
        
        self.zone_generator = FibZoneGenerator(
            standard_ratios=config.get('fib_ratios', [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]),
            zone_width_factor=config.get('zone_width_factor', 0.1),
            min_zone_size=config.get('min_zone_size', 0.001)
        )
        
        self.event_generator = EventGenerator(
            min_touch_duration=config.get('min_touch_duration', 1),
            max_event_duration=config.get('max_event_duration', 100),
            wick_rejection_threshold=config.get('wick_rejection_threshold', 0.3)
        )
        
        self.feature_extractor = FeatureExtractor(
            atr_period=config.get('atr_period', 14),
            trend_period=config.get('trend_period', 20),
            volatility_period=config.get('volatility_period', 20)
        )
        
        # Data storage
        self.data_store = data_store
        if data_store == 'redis':
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        else:
            self.redis_client = None
        
        # State management
        self.current_data = pd.DataFrame()
        self.current_swings = []
        self.current_zones = []
        self.active_events = []
        self.active_trades = []
        
        # Performance tracking
        self.predictions_log = []
        self.trades_log = []
        self.performance_metrics = {}
        
        # Risk management
        self.max_positions = config.get('max_positions', 3)
        self.risk_per_trade = config.get('risk_per_trade', 0.01)
        self.min_confidence = config.get('min_confidence', 0.65)
        
    async def start_live_trading(self, 
                               data_feed,
                               execution_engine=None) -> None:
        """
        Start the live trading pipeline.
        
        Args:
            data_feed: Data feed object (e.g., CCXT exchange)
            execution_engine: Trade execution engine
        """
        print("Starting live trading pipeline...")
        
        try:
            while True:
                # Get latest data
                latest_data = await self._get_latest_data(data_feed)
                
                if latest_data is not None:
                    # Update pipeline state
                    await self._update_pipeline_state(latest_data)
                    
                    # Generate predictions
                    predictions = await self._generate_predictions()
                    
                    # Process predictions
                    if predictions:
                        await self._process_predictions(predictions, execution_engine)
                    
                    # Monitor active trades
                    await self._monitor_trades(execution_engine)
                    
                    # Log performance
                    await self._log_performance()
                
                # Wait for next update
                await asyncio.sleep(self.config.get('update_interval', 60))
                
        except KeyboardInterrupt:
            print("Stopping live trading pipeline...")
        except Exception as e:
            print(f"Error in live trading pipeline: {e}")
            raise
    
    async def _get_latest_data(self, data_feed) -> Optional[pd.DataFrame]:
        """Get latest market data from feed."""
        try:
            # This would be implemented based on your data feed
            # For example, using CCXT for exchange data
            ohlcv = await data_feed.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            warnings.warn(f"Failed to get latest data: {e}")
            return None
    
    async def _update_pipeline_state(self, data: pd.DataFrame) -> None:
        """Update pipeline state with latest data."""
        self.current_data = data
        
        # Detect swings
        self.current_swings = self.swing_detector.detect_swings(data)
        
        # Generate zones
        if len(self.current_swings) >= 2:
            self.current_zones = self.zone_generator.generate_all_zones(self.current_swings)
            self.current_zones = self.zone_generator.filter_zones_by_strength(
                self.current_zones, min_strength=0.3
            )
        
        # Generate events
        if self.current_zones:
            new_events = self.event_generator.detect_zone_touches(data, self.current_zones)
            
            # Add new events to active events
            for event in new_events:
                if event.event_id not in [e.event_id for e in self.active_events]:
                    self.active_events.append(event)
    
    async def _generate_predictions(self) -> List[LivePrediction]:
        """Generate predictions for active events."""
        predictions = []
        
        for event in self.active_events:
            try:
                # Extract features
                feature_set = self.feature_extractor.extract_all_features(
                    self.current_data, event, self.current_swings
                )
                
                # Make prediction
                prediction = self._make_live_prediction(feature_set, event)
                
                if prediction:
                    predictions.append(prediction)
                    
            except Exception as e:
                warnings.warn(f"Failed to generate prediction for event {event.event_id}: {e}")
        
        return predictions
    
    def _make_live_prediction(self, 
                            feature_set,
                            event: ZoneEvent) -> Optional[LivePrediction]:
        """Make prediction for a single event."""
        try:
            # Prepare features
            features_df = pd.DataFrame([feature_set.raw_features])
            
            # Classify outcome
            outcome_proba = self.model_results.classifier.predict(
                features_df, num_iteration=self.model_results.classifier.best_iteration
            )
            outcome_idx = np.argmax(outcome_proba)
            outcome = self.model_results.classifier.reverse_encoder[outcome_idx]
            confidence = outcome_proba[outcome_idx]
            
            # Predict target price
            target_price = self.model_results.target_regressor.predict(
                features_df, num_iteration=self.model_results.target_regressor.best_iteration
            )[0]
            
            # Predict duration
            duration = self.model_results.duration_regressor.predict(
                features_df, num_iteration=self.model_results.duration_regressor.best_iteration
            )[0]
            
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(event, outcome)
            
            # Calculate target zone
            target_zone = self._calculate_target_zone(event, target_price, outcome)
            
            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                event.entry_price, target_price, stop_loss
            )
            
            # Generate context tags
            context_tags = self._generate_context_tags(event, outcome, confidence)
            
            return LivePrediction(
                event_id=event.event_id,
                fib_zone=event.zone.price_range,
                prediction=outcome,
                confidence=confidence,
                target_zone=target_zone,
                stop_loss_level=stop_loss,
                context_tags=context_tags,
                timestamp=datetime.now(),
                entry_price=event.entry_price,
                risk_reward_ratio=risk_reward_ratio
            )
            
        except Exception as e:
            warnings.warn(f"Prediction failed for event {event.event_id}: {e}")
            return None
    
    def _calculate_stop_loss(self, event: ZoneEvent, outcome: str) -> float:
        """Calculate stop loss level."""
        zone_lower, zone_upper = event.zone.price_range
        
        if outcome == 'Reversal':
            # Reversal trade - stop beyond zone
            if event.zone.swing_start.swing_type == 'high':
                stop_loss = zone_lower * 0.995  # Below zone
            else:
                stop_loss = zone_upper * 1.005  # Above zone
        elif outcome == 'Continuation':
            # Continuation trade - stop beyond zone
            if event.zone.swing_start.swing_type == 'high':
                stop_loss = zone_upper * 1.005  # Above zone
            else:
                stop_loss = zone_lower * 0.995  # Below zone
        else:  # Breakout
            # Breakout trade - stop back inside zone
            stop_loss = (zone_lower + zone_upper) / 2
        
        return stop_loss
    
    def _calculate_target_zone(self, 
                             event: ZoneEvent, 
                             target_price: float, 
                             outcome: str) -> Tuple[float, float]:
        """Calculate target zone."""
        if outcome == 'Reversal':
            # Target is back to swing extreme
            target_price = event.zone.swing_start.price
        
        # Create zone around target
        zone_width = abs(target_price) * 0.01  # 1% zone width
        return (target_price - zone_width, target_price + zone_width)
    
    def _calculate_risk_reward_ratio(self, 
                                   entry_price: float, 
                                   target_price: float, 
                                   stop_loss: float) -> float:
        """Calculate risk-reward ratio."""
        reward = abs(target_price - entry_price)
        risk = abs(entry_price - stop_loss)
        return reward / risk if risk > 0 else 0
    
    def _generate_context_tags(self, 
                             event: ZoneEvent, 
                             outcome: str, 
                             confidence: float) -> List[str]:
        """Generate context tags for the prediction."""
        tags = []
        
        # Zone type
        tags.append(f"{event.zone.zone_type}_zone")
        tags.append(f"fib_{event.zone.level}")
        
        # Outcome
        tags.append(f"{outcome.lower()}_prediction")
        
        # Confidence level
        if confidence > 0.8:
            tags.append("high_confidence")
        elif confidence > 0.6:
            tags.append("medium_confidence")
        else:
            tags.append("low_confidence")
        
        # Market context
        if event.zone.swing_start.swing_type == 'high':
            tags.append("downtrend_context")
        else:
            tags.append("uptrend_context")
        
        # Volume context
        if event.volume_in_zone and event.volume_in_zone > self.current_data['volume'].mean() * 1.5:
            tags.append("high_volume")
        
        # Wick rejection
        if event.wick_rejection:
            tags.append("wick_rejection")
        
        return tags
    
    async def _process_predictions(self, 
                                 predictions: List[LivePrediction],
                                 execution_engine) -> None:
        """Process predictions and generate trading signals."""
        for prediction in predictions:
            # Check if we should trade this prediction
            if self._should_trade(prediction):
                # Generate execution signal
                signal = self._generate_execution_signal(prediction)
                
                if signal:
                    # Execute trade
                    await self._execute_trade(signal, execution_engine)
                    
                    # Log prediction
                    self.predictions_log.append(prediction)
    
    def _should_trade(self, prediction: LivePrediction) -> bool:
        """Determine if we should trade this prediction."""
        # Check confidence threshold
        if prediction.confidence < self.min_confidence:
            return False
        
        # Check position limit
        if len(self.active_trades) >= self.max_positions:
            return False
        
        # Check risk-reward ratio
        if prediction.risk_reward_ratio < 1.5:
            return False
        
        # Check if we already have a trade for this event
        for trade in self.active_trades:
            if trade.context.get('event_id') == prediction.event_id:
                return False
        
        return True
    
    def _generate_execution_signal(self, 
                                 prediction: LivePrediction) -> Optional[ExecutionSignal]:
        """Generate execution signal from prediction."""
        try:
            # Determine trade direction
            if prediction.prediction == 'Reversal':
                # Reversal trade - opposite to trend
                if 'downtrend_context' in prediction.context_tags:
                    direction = 'long'
                else:
                    direction = 'short'
            elif prediction.prediction == 'Continuation':
                # Continuation trade - same as trend
                if 'downtrend_context' in prediction.context_tags:
                    direction = 'short'
                else:
                    direction = 'long'
            else:  # Breakout
                # Breakout trade - follow the break
                if prediction.entry_price > prediction.fib_zone[1]:
                    direction = 'long'
                else:
                    direction = 'short'
            
            # Calculate position size
            risk_amount = self.config.get('initial_capital', 100000) * self.risk_per_trade
            entry_price = prediction.entry_price
            stop_loss = prediction.stop_loss_level
            
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                size = risk_amount / risk_per_share
            else:
                return None
            
            # Calculate take profit
            if prediction.target_zone:
                take_profit = (prediction.target_zone[0] + prediction.target_zone[1]) / 2
            else:
                take_profit = entry_price * (1.02 if direction == 'long' else 0.98)
            
            return ExecutionSignal(
                signal_id=f"signal_{len(self.predictions_log) + 1}",
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=size,
                confidence=prediction.confidence,
                context={
                    'event_id': prediction.event_id,
                    'prediction': prediction.prediction,
                    'context_tags': prediction.context_tags
                },
                timestamp=prediction.timestamp
            )
            
        except Exception as e:
            warnings.warn(f"Failed to generate execution signal: {e}")
            return None
    
    async def _execute_trade(self, 
                           signal: ExecutionSignal,
                           execution_engine) -> None:
        """Execute a trade based on signal."""
        try:
            if execution_engine:
                # Execute through external engine
                trade_result = await execution_engine.execute_trade(signal)
                
                if trade_result:
                    # Create trade object
                    trade = Trade(
                        trade_id=signal.signal_id,
                        entry_time=signal.timestamp,
                        entry_price=signal.entry_price,
                        direction=signal.direction,
                        size=signal.size,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        context=signal.context
                    )
                    
                    self.active_trades.append(trade)
                    self.trades_log.append(trade)
                    
                    print(f"Executed trade: {signal.direction} {signal.size:.2f} @ {signal.entry_price:.2f}")
            else:
                # Simulate execution
                print(f"Simulated trade: {signal.direction} {signal.size:.2f} @ {signal.entry_price:.2f}")
                
        except Exception as e:
            warnings.warn(f"Failed to execute trade: {e}")
    
    async def _monitor_trades(self, execution_engine) -> None:
        """Monitor active trades and check for exits."""
        trades_to_close = []
        
        for trade in self.active_trades:
            try:
                # Get current price (simplified)
                current_price = self.current_data['close'].iloc[-1]
                
                # Check stop loss
                if trade.stop_loss:
                    if trade.direction == 'long' and current_price <= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.outcome = 'stop_loss'
                        trades_to_close.append(trade)
                    elif trade.direction == 'short' and current_price >= trade.stop_loss:
                        trade.exit_price = trade.stop_loss
                        trade.outcome = 'stop_loss'
                        trades_to_close.append(trade)
                
                # Check take profit
                if trade.take_profit:
                    if trade.direction == 'long' and current_price >= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.outcome = 'take_profit'
                        trades_to_close.append(trade)
                    elif trade.direction == 'short' and current_price <= trade.take_profit:
                        trade.exit_price = trade.take_profit
                        trade.outcome = 'take_profit'
                        trades_to_close.append(trade)
                
            except Exception as e:
                warnings.warn(f"Failed to monitor trade {trade.trade_id}: {e}")
        
        # Close trades
        for trade in trades_to_close:
            await self._close_trade(trade, execution_engine)
    
    async def _close_trade(self, trade: Trade, execution_engine) -> None:
        """Close a trade."""
        try:
            if execution_engine:
                await execution_engine.close_trade(trade)
            
            # Calculate P&L
            if trade.direction == 'long':
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
            else:
                trade.pnl = (trade.entry_price - trade.exit_price) * trade.size
            
            trade.exit_time = datetime.now()
            trade.duration_bars = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            
            # Remove from active trades
            self.active_trades.remove(trade)
            
            print(f"Closed trade: {trade.trade_id} P&L: {trade.pnl:.2f}")
            
        except Exception as e:
            warnings.warn(f"Failed to close trade {trade.trade_id}: {e}")
    
    async def _log_performance(self) -> None:
        """Log performance metrics."""
        try:
            # Calculate basic metrics
            total_trades = len(self.trades_log)
            if total_trades > 0:
                winning_trades = len([t for t in self.trades_log if t.pnl and t.pnl > 0])
                win_rate = winning_trades / total_trades
                
                total_pnl = sum(t.pnl for t in self.trades_log if t.pnl)
                
                self.performance_metrics = {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'active_trades': len(self.active_trades),
                    'timestamp': datetime.now()
                }
                
                # Log to storage
                await self._store_performance_metrics()
                
        except Exception as e:
            warnings.warn(f"Failed to log performance: {e}")
    
    async def _store_performance_metrics(self) -> None:
        """Store performance metrics to data store."""
        try:
            if self.data_store == 'redis' and self.redis_client:
                # Store in Redis
                key = f"fib_ml_performance:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.redis_client.set(key, json.dumps(self.performance_metrics, default=str))
            else:
                # Store in SQLite
                conn = sqlite3.connect('fib_ml_performance.db')
                df = pd.DataFrame([self.performance_metrics])
                df.to_sql('performance', conn, if_exists='append', index=False)
                conn.close()
                
        except Exception as e:
            warnings.warn(f"Failed to store performance metrics: {e}")
    
    def get_current_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'active_trades': len(self.active_trades),
            'active_events': len(self.active_events),
            'current_zones': len(self.current_zones),
            'current_swings': len(self.current_swings),
            'performance_metrics': self.performance_metrics,
            'last_update': datetime.now()
        }


def test_live_pipeline():
    """Test the live pipeline with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    from event_generator import EventGenerator
    from features import FeatureExtractor
    from labeling import LabelGenerator
    from model_train import ModelTrainer
    
    print("Testing Live Pipeline...")
    
    # Create sample data
    sample_data = create_sample_data(500, 100.0, 0.02)
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
        print("No events generated for testing")
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
    
    # Initialize live pipeline
    config = {
        'swing_window': 8,
        'min_swing_strength': 0.2,
        'min_zone_size': 0.003,
        'fib_ratios': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        'zone_width_factor': 0.15,
        'min_touch_duration': 1,
        'max_event_duration': 50,
        'wick_rejection_threshold': 0.3,
        'atr_period': 14,
        'trend_period': 20,
        'volatility_period': 20,
        'max_positions': 3,
        'risk_per_trade': 0.01,
        'min_confidence': 0.6,
        'initial_capital': 100000,
        'update_interval': 60
    }
    
    pipeline = LivePipeline(model_results, config)
    
    # Test prediction generation
    print("Testing prediction generation...")
    pipeline.current_data = sample_data
    pipeline.current_swings = swings
    pipeline.current_zones = zones
    pipeline.active_events = events[:5]  # Test with first 5 events
    
    predictions = asyncio.run(pipeline._generate_predictions())
    print(f"Generated {len(predictions)} predictions")
    
    if predictions:
        print("\nSample prediction:")
        pred = predictions[0]
        print(f"Event: {pred.event_id}")
        print(f"Prediction: {pred.prediction}")
        print(f"Confidence: {pred.confidence:.3f}")
        print(f"Risk-Reward: {pred.risk_reward_ratio:.2f}")
        print(f"Context tags: {pred.context_tags}")
    
    # Test status
    status = pipeline.get_current_status()
    print(f"\nPipeline status: {status}")
    
    print("Live pipeline test completed successfully!")


if __name__ == "__main__":
    test_live_pipeline()