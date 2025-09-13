"""
Main Fibonacci ML System

Complete system integration for adaptive Fibonacci machine learning.
Provides a unified interface for training, backtesting, and live trading.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from swing_detector import SwingDetector, create_sample_data
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator
from features import FeatureExtractor
from labeling import LabelGenerator
from model_train import ModelTrainer, ModelResults
from backtest import BacktestEngine, BacktestResults
from live_pipeline import LivePipeline


@dataclass
class SystemConfig:
    """Configuration for the Fibonacci ML system."""
    # Swing detection
    swing_window: int = 20
    min_swing_strength: float = 0.5
    min_swing_size: float = 0.001
    
    # Fibonacci zones (will be learned, not traditional)
    fib_ratios: List[float] = None
    zone_width_factor: float = 0.1
    min_zone_size: float = 0.001
    learn_levels: bool = True  # Enable adaptive level learning
    
    # Event generation
    min_touch_duration: int = 1
    max_event_duration: int = 100
    wick_rejection_threshold: float = 0.3
    
    # Feature extraction
    atr_period: int = 14
    trend_period: int = 20
    volatility_period: int = 20
    
    # Labeling
    lookforward_window: int = 30
    reversal_threshold: float = 2.0
    continuation_threshold: float = 1.0
    atr_multiplier: float = 1.0
    min_confidence: float = 0.6
    
    # Model training
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    
    # Backtesting
    initial_capital: float = 100000
    risk_per_trade: float = 0.01
    max_positions: int = 3
    slippage_bps: float = 2.0
    commission_bps: float = 1.0
    
    # Live trading
    update_interval: int = 60
    
    def __post_init__(self):
        if self.fib_ratios is None:
            self.fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.414, 1.618]


class FibMLSystem:
    """
    Main Fibonacci ML System class.
    
    Provides a unified interface for the complete system including
    training, backtesting, and live trading capabilities.
    """
    
    def __init__(self, config: SystemConfig = None):
        """
        Initialize the Fibonacci ML system.
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.swing_detector = SwingDetector(
            window=self.config.swing_window,
            min_swing_strength=self.config.min_swing_strength,
            min_swing_size=self.config.min_swing_size
        )
        
        self.zone_generator = FibZoneGenerator(
            standard_ratios=self.config.fib_ratios,
            zone_width_factor=self.config.zone_width_factor,
            min_zone_size=self.config.min_zone_size
        )
        
        self.event_generator = EventGenerator(
            min_touch_duration=self.config.min_touch_duration,
            max_event_duration=self.config.max_event_duration,
            wick_rejection_threshold=self.config.wick_rejection_threshold
        )
        
        self.feature_extractor = FeatureExtractor(
            atr_period=self.config.atr_period,
            trend_period=self.config.trend_period,
            volatility_period=self.config.volatility_period
        )
        
        self.label_generator = LabelGenerator(
            lookforward_window=self.config.lookforward_window,
            reversal_threshold=self.config.reversal_threshold,
            continuation_threshold=self.config.continuation_threshold,
            atr_multiplier=self.config.atr_multiplier,
            min_confidence=self.config.min_confidence
        )
        
        self.model_trainer = ModelTrainer(
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state,
            n_splits=self.config.n_splits
        )
        
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.initial_capital,
            risk_per_trade=self.config.risk_per_trade,
            max_positions=self.config.max_positions,
            slippage_bps=self.config.slippage_bps,
            commission_bps=self.config.commission_bps,
            min_confidence=self.config.min_confidence
        )
        
        # State
        self.trained_models = None
        self.last_training_data = None
        
    def train(self, 
              data: pd.DataFrame,
              save_models: bool = True,
              model_path: str = "fib_ml_models") -> ModelResults:
        """
        Train the complete system on historical data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            save_models: Whether to save trained models
            model_path: Path to save models
            
        Returns:
            ModelResults object
        """
        print("Starting system training...")
        
        # Step 1: Detect swings
        print("Step 1: Detecting swings...")
        swings = self.swing_detector.detect_swings(data)
        print(f"Detected {len(swings)} swings")
        
        if len(swings) < 2:
            raise ValueError("Not enough swings detected for training")
        
        # Step 2: Generate Fibonacci zones
        print("Step 2: Generating Fibonacci zones...")
        zones = self.zone_generator.generate_all_zones(swings)
        zones = self.zone_generator.filter_zones_by_strength(zones, min_strength=0.3)
        print(f"Generated {len(zones)} initial zones")
        
        # Step 2.5: Learn effective levels if enabled
        if self.config.learn_levels:
            print("Step 2.5: Learning effective retracement levels...")
            # Generate initial events to learn from
            temp_events = self.event_generator.detect_zone_touches(data, zones)
            learned_levels = self.zone_generator.learn_effective_levels(zones, temp_events, min_events=5)
            
            if learned_levels:
                print(f"Learned effective levels: {[f'{level:.3f}' for level in learned_levels[:10]]}")
                # Regenerate zones with learned levels
                zones = self.zone_generator.create_adaptive_zones(swings, learned_levels)
                zones = self.zone_generator.filter_zones_by_strength(zones, min_strength=0.3)
                print(f"Generated {len(zones)} zones using learned levels")
            else:
                print("No effective levels learned, using initial zones")
        
        if not zones:
            raise ValueError("No valid zones generated")
        
        # Step 3: Generate events
        print("Step 3: Generating events...")
        events = self.event_generator.detect_zone_touches(data, zones)
        print(f"Generated {len(events)} events")
        
        if not events:
            raise ValueError("No events generated")
        
        # Step 4: Extract features
        print("Step 4: Extracting features...")
        feature_sets = []
        for event in events:
            feature_set = self.feature_extractor.extract_all_features(data, event, swings)
            feature_sets.append(feature_set)
        
        features_df = self.feature_extractor.export_features_to_dataframe(feature_sets)
        print(f"Extracted {len(features_df.columns)} features")
        
        # Step 5: Generate labels
        print("Step 5: Generating labels...")
        labels = self.label_generator.generate_labels(data, events, zones)
        labels_df = self.label_generator.export_labels_to_dataframe(labels)
        print(f"Generated {len(labels)} labels")
        
        if features_df.empty or labels_df.empty:
            raise ValueError("No features or labels generated")
        
        # Step 6: Train models
        print("Step 6: Training models...")
        model_results = self.model_trainer.train_all_models(features_df, labels_df)
        
        # Save models
        if save_models:
            self.model_trainer.save_models(model_results, model_path)
            print(f"Models saved to {model_path}_*.pkl")
        
        # Store results
        self.trained_models = model_results
        self.last_training_data = {
            'data': data,
            'swings': swings,
            'zones': zones,
            'events': events,
            'features_df': features_df,
            'labels_df': labels_df
        }
        
        print("Training completed successfully!")
        return model_results
    
    def backtest(self, 
                 data: pd.DataFrame = None,
                 model_results: ModelResults = None) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame (uses last training data if None)
            model_results: Trained models (uses last trained models if None)
            
        Returns:
            BacktestResults object
        """
        if data is None:
            if self.last_training_data is None:
                raise ValueError("No training data available. Please train the system first.")
            data = self.last_training_data['data']
            swings = self.last_training_data['swings']
            zones = self.last_training_data['zones']
            events = self.last_training_data['events']
        else:
            # Generate new data
            swings = self.swing_detector.detect_swings(data)
            zones = self.zone_generator.generate_all_zones(swings)
            zones = self.zone_generator.filter_zones_by_strength(zones, min_strength=0.3)
            events = self.event_generator.detect_zone_touches(data, zones)
        
        if model_results is None:
            if self.trained_models is None:
                raise ValueError("No trained models available. Please train the system first.")
            model_results = self.trained_models
        
        print("Starting backtest...")
        
        # Run backtest
        results = self.backtest_engine.run_backtest(
            data, model_results, events, self.feature_extractor, swings
        )
        
        print("Backtest completed!")
        return results
    
    def predict_live(self, 
                    data: pd.DataFrame,
                    model_results: ModelResults = None) -> List[Dict]:
        """
        Generate live predictions for current market data.
        
        Args:
            data: Current OHLCV DataFrame
            model_results: Trained models (uses last trained models if None)
            
        Returns:
            List of prediction dictionaries
        """
        if model_results is None:
            if self.trained_models is None:
                raise ValueError("No trained models available. Please train the system first.")
            model_results = self.trained_models
        
        # Generate current state
        swings = self.swing_detector.detect_swings(data)
        zones = self.zone_generator.generate_all_zones(swings)
        zones = self.zone_generator.filter_zones_by_strength(zones, min_strength=0.3)
        events = self.event_generator.detect_zone_touches(data, zones)
        
        # Generate predictions
        predictions = []
        for event in events:
            try:
                # Extract features
                feature_set = self.feature_extractor.extract_all_features(data, event, swings)
                
                # Make prediction
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
                
                # Calculate stop loss
                stop_loss = self._calculate_stop_loss(event, outcome)
                
                # Generate context tags
                context_tags = self._generate_context_tags(event, outcome, confidence)
                
                prediction = {
                    "fib_zone": list(event.zone.price_range),
                    "prediction": outcome,
                    "confidence": confidence,
                    "target_zone": [target_price * 0.99, target_price * 1.01],
                    "stop_loss_level": stop_loss,
                    "context_tags": context_tags
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                warnings.warn(f"Failed to generate prediction for event {event.event_id}: {e}")
        
        return predictions
    
    def _calculate_stop_loss(self, event, outcome: str) -> float:
        """Calculate stop loss level."""
        zone_lower, zone_upper = event.zone.price_range
        
        if outcome == 'Reversal':
            if event.zone.swing_start.swing_type == 'high':
                stop_loss = zone_lower * 0.995
            else:
                stop_loss = zone_upper * 1.005
        elif outcome == 'Continuation':
            if event.zone.swing_start.swing_type == 'high':
                stop_loss = zone_upper * 1.005
            else:
                stop_loss = zone_lower * 0.995
        else:  # Breakout
            stop_loss = (zone_lower + zone_upper) / 2
        
        return stop_loss
    
    def _generate_context_tags(self, event, outcome: str, confidence: float) -> List[str]:
        """Generate context tags."""
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
        if event.volume_in_zone and event.volume_in_zone > 0:
            tags.append("volume_context")
        
        # Wick rejection
        if event.wick_rejection:
            tags.append("wick_rejection")
        
        return tags
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        status = {
            'trained': self.trained_models is not None,
            'last_training_data_available': self.last_training_data is not None,
            'config': self.config.__dict__
        }
        
        if self.trained_models:
            status['model_performance'] = self.trained_models.performance_metrics
        
        if self.last_training_data:
            status['last_training_stats'] = {
                'data_length': len(self.last_training_data['data']),
                'swings_count': len(self.last_training_data['swings']),
                'zones_count': len(self.last_training_data['zones']),
                'events_count': len(self.last_training_data['events']),
                'features_count': len(self.last_training_data['features_df'].columns),
                'labels_count': len(self.last_training_data['labels_df'])
            }
        
        return status
    
    def load_models(self, model_path: str) -> None:
        """
        Load pre-trained models.
        
        Args:
            model_path: Base path to model files
        """
        self.trained_models = self.model_trainer.load_models(model_path)
        print(f"Models loaded from {model_path}_*.pkl")
    
    def create_sample_data(self, 
                          length: int = 1000,
                          start_price: float = 100.0,
                          volatility: float = 0.02) -> pd.DataFrame:
        """
        Create sample OHLCV data for testing.
        
        Args:
            length: Number of bars
            start_price: Starting price
            volatility: Price volatility
            
        Returns:
            Sample OHLCV DataFrame
        """
        return create_sample_data(length, start_price, volatility)


def main():
    """Main function to demonstrate the system."""
    print("Fibonacci ML System Demo")
    print("=" * 50)
    
    # Create system
    config = SystemConfig(
        swing_window=10,
        min_swing_strength=0.3,
        min_zone_size=0.002,
        fib_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
        zone_width_factor=0.15,
        min_confidence=0.6
    )
    
    system = FibMLSystem(config)
    
    # Create sample data
    print("Creating sample data...")
    sample_data = system.create_sample_data(800, 100.0, 0.02)
    print(f"Created {len(sample_data)} bars of sample data")
    
    # Train system
    print("\nTraining system...")
    model_results = system.train(sample_data, save_models=False)
    
    # Show training results
    print(f"\nTraining Results:")
    print(f"Classifier accuracy: {model_results.performance_metrics['classifier']['accuracy']:.3f}")
    print(f"Target regressor R²: {model_results.performance_metrics['regression']['target_price']['r2']:.3f}")
    print(f"Duration regressor R²: {model_results.performance_metrics['regression']['duration']['r2']:.3f}")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = system.backtest()
    
    # Show backtest results
    metrics = backtest_results.performance_metrics
    print(f"\nBacktest Results:")
    print(f"Total trades: {metrics.get('total_trades', 0)}")
    print(f"Win rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Total return: {metrics.get('total_return', 0):.1%}")
    print(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max drawdown: {metrics.get('max_drawdown', 0):.1%}")
    print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
    
    # Generate live predictions
    print("\nGenerating live predictions...")
    predictions = system.predict_live(sample_data.tail(100))
    
    print(f"\nGenerated {len(predictions)} predictions:")
    for i, pred in enumerate(predictions[:3]):  # Show first 3
        print(f"Prediction {i+1}:")
        print(f"  Zone: {pred['fib_zone']}")
        print(f"  Outcome: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence']:.3f}")
        print(f"  Context: {pred['context_tags']}")
    
    # Show system status
    print(f"\nSystem Status:")
    status = system.get_system_status()
    print(f"Trained: {status['trained']}")
    print(f"Training data available: {status['last_training_data_available']}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()