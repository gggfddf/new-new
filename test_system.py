"""
Comprehensive Test Suite for Fibonacci ML System

Tests all components individually and as an integrated system.
"""

import pandas as pd
import numpy as np
import pytest
import warnings
from fib_ml_system import FibMLSystem, SystemConfig
from swing_detector import SwingDetector, create_sample_data
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator
from features import FeatureExtractor
from labeling import LabelGenerator
from model_train import ModelTrainer
from backtest import BacktestEngine
from live_pipeline import LivePipeline


class TestSwingDetector:
    """Test swing detection functionality."""
    
    def test_swing_detection(self):
        """Test basic swing detection."""
        # Create sample data
        data = create_sample_data(200, 100.0, 0.02)
        
        # Initialize detector
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        
        # Detect swings
        swings = detector.detect_swings(data)
        
        # Assertions
        assert len(swings) > 0, "Should detect at least some swings"
        assert all(hasattr(swing, 'swing_type') for swing in swings), "All swings should have type"
        assert all(hasattr(swing, 'price') for swing in swings), "All swings should have price"
        assert all(hasattr(swing, 'index') for swing in swings), "All swings should have index"
        
        # Check swing types
        swing_types = [swing.swing_type for swing in swings]
        assert 'high' in swing_types, "Should detect high swings"
        assert 'low' in swing_types, "Should detect low swings"
        
        print(f"✓ Swing detection test passed: {len(swings)} swings detected")
    
    def test_swing_sequences(self):
        """Test swing sequence generation."""
        data = create_sample_data(200, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        sequences = detector.get_swing_sequences(swings)
        
        assert len(sequences) > 0, "Should generate swing sequences"
        assert all(len(seq) > 0 for seq in sequences), "All sequences should be non-empty"
        
        print(f"✓ Swing sequences test passed: {len(sequences)} sequences generated")


class TestFibZones:
    """Test Fibonacci zone generation."""
    
    def test_zone_generation(self):
        """Test basic zone generation."""
        # Create sample data and swings
        data = create_sample_data(200, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        # Generate zones
        zone_generator = FibZoneGenerator(
            standard_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618],
            zone_width_factor=0.15,
            min_zone_size=0.002
        )
        
        zones = zone_generator.generate_all_zones(swings)
        
        assert len(zones) > 0, "Should generate zones"
        assert all(hasattr(zone, 'zone_type') for zone in zones), "All zones should have type"
        assert all(hasattr(zone, 'price_range') for zone in zones), "All zones should have price range"
        assert all(hasattr(zone, 'level') for zone in zones), "All zones should have level"
        
        # Check zone types
        zone_types = [zone.zone_type for zone in zones]
        assert 'retracement' in zone_types, "Should generate retracement zones"
        assert 'extension' in zone_types, "Should generate extension zones"
        
        print(f"✓ Zone generation test passed: {len(zones)} zones generated")
    
    def test_zone_filtering(self):
        """Test zone filtering functionality."""
        data = create_sample_data(200, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        
        # Filter by strength
        filtered_zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.3)
        
        assert len(filtered_zones) <= len(zones), "Filtered zones should be subset of all zones"
        assert all(zone.strength >= 0.3 for zone in filtered_zones), "All filtered zones should meet strength criteria"
        
        print(f"✓ Zone filtering test passed: {len(filtered_zones)}/{len(zones)} zones passed filter")


class TestEventGenerator:
    """Test event generation functionality."""
    
    def test_event_generation(self):
        """Test basic event generation."""
        # Create sample data, swings, and zones
        data = create_sample_data(300, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        # Generate events
        event_generator = EventGenerator(
            min_touch_duration=1,
            max_event_duration=50,
            wick_rejection_threshold=0.3
        )
        
        events = event_generator.detect_zone_touches(data, zones)
        
        assert len(events) > 0, "Should generate events"
        assert all(hasattr(event, 'event_id') for event in events), "All events should have ID"
        assert all(hasattr(event, 'zone') for event in events), "All events should have zone"
        assert all(hasattr(event, 'entry_time') for event in events), "All events should have entry time"
        assert all(hasattr(event, 'entry_price') for event in events), "All events should have entry price"
        
        print(f"✓ Event generation test passed: {len(events)} events generated")
    
    def test_event_metrics(self):
        """Test event metrics calculation."""
        data = create_sample_data(300, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        event_generator = EventGenerator()
        events = event_generator.detect_zone_touches(data, zones)
        
        metrics = event_generator.calculate_event_metrics(events)
        
        assert 'total_events' in metrics, "Metrics should include total events"
        assert 'events_with_exit' in metrics, "Metrics should include events with exit"
        assert 'avg_duration_bars' in metrics, "Metrics should include average duration"
        
        print(f"✓ Event metrics test passed: {metrics['total_events']} total events")


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def test_feature_extraction(self):
        """Test basic feature extraction."""
        # Create sample data, swings, zones, and events
        data = create_sample_data(400, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        event_generator = EventGenerator()
        events = event_generator.detect_zone_touches(data, zones)
        
        # Extract features
        feature_extractor = FeatureExtractor()
        feature_sets = []
        
        for event in events[:10]:  # Test with first 10 events
            feature_set = feature_extractor.extract_all_features(data, event, swings)
            feature_sets.append(feature_set)
        
        assert len(feature_sets) > 0, "Should extract features"
        assert all(hasattr(fs, 'raw_features') for fs in feature_sets), "All feature sets should have raw features"
        assert all(len(fs.raw_features) > 0 for fs in feature_sets), "All feature sets should have non-empty features"
        
        # Export to DataFrame
        features_df = feature_extractor.export_features_to_dataframe(feature_sets)
        
        assert not features_df.empty, "Features DataFrame should not be empty"
        assert len(features_df.columns) > 0, "Features DataFrame should have columns"
        
        print(f"✓ Feature extraction test passed: {len(features_df.columns)} features extracted")


class TestLabelGenerator:
    """Test label generation functionality."""
    
    def test_label_generation(self):
        """Test basic label generation."""
        # Create sample data, swings, zones, and events
        data = create_sample_data(500, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        event_generator = EventGenerator()
        events = event_generator.detect_zone_touches(data, zones)
        
        # Generate labels
        label_generator = LabelGenerator(
            lookforward_window=30,
            reversal_threshold=2.0,
            continuation_threshold=1.0,
            atr_multiplier=1.0,
            min_confidence=0.5
        )
        
        labels = label_generator.generate_labels(data, events, zones)
        
        assert len(labels) > 0, "Should generate labels"
        assert all(hasattr(label, 'outcome_type') for label in labels), "All labels should have outcome type"
        assert all(hasattr(label, 'confidence') for label in labels), "All labels should have confidence"
        
        # Check outcome types
        outcome_types = [label.outcome_type for label in labels]
        valid_outcomes = ['Reversal', 'Continuation', 'Breakout']
        assert all(outcome in valid_outcomes for outcome in outcome_types), "All outcomes should be valid"
        
        # Export to DataFrame
        labels_df = label_generator.export_labels_to_dataframe(labels)
        
        assert not labels_df.empty, "Labels DataFrame should not be empty"
        assert 'outcome_type' in labels_df.columns, "Labels DataFrame should have outcome_type column"
        
        print(f"✓ Label generation test passed: {len(labels)} labels generated")


class TestModelTrainer:
    """Test model training functionality."""
    
    def test_model_training(self):
        """Test basic model training."""
        # Create sample data and generate features/labels
        data = create_sample_data(600, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        event_generator = EventGenerator()
        events = event_generator.detect_zone_touches(data, zones)
        
        feature_extractor = FeatureExtractor()
        feature_sets = []
        for event in events:
            feature_set = feature_extractor.extract_all_features(data, event, swings)
            feature_sets.append(feature_set)
        
        features_df = feature_extractor.export_features_to_dataframe(feature_sets)
        
        label_generator = LabelGenerator()
        labels = label_generator.generate_labels(data, events, zones)
        labels_df = label_generator.export_labels_to_dataframe(labels)
        
        if features_df.empty or labels_df.empty:
            print("⚠ Skipping model training test - insufficient data")
            return
        
        # Train models
        trainer = ModelTrainer(
            test_size=0.2,
            validation_size=0.2,
            random_state=42,
            n_splits=3
        )
        
        model_results = trainer.train_all_models(features_df, labels_df)
        
        assert model_results.classifier is not None, "Classifier should be trained"
        assert model_results.target_regressor is not None, "Target regressor should be trained"
        assert model_results.duration_regressor is not None, "Duration regressor should be trained"
        assert model_results.calibration_model is not None, "Calibration model should be trained"
        
        # Check performance metrics
        assert 'classifier' in model_results.performance_metrics, "Should have classifier metrics"
        assert 'regression' in model_results.performance_metrics, "Should have regression metrics"
        
        print(f"✓ Model training test passed: All models trained successfully")


class TestBacktestEngine:
    """Test backtesting functionality."""
    
    def test_backtest_engine(self):
        """Test basic backtesting."""
        # Create sample data and train models
        data = create_sample_data(800, 100.0, 0.02)
        detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
        swings = detector.detect_swings(data)
        
        zone_generator = FibZoneGenerator()
        zones = zone_generator.generate_all_zones(swings)
        zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.1)
        
        event_generator = EventGenerator()
        events = event_generator.detect_zone_touches(data, zones)
        
        feature_extractor = FeatureExtractor()
        feature_sets = []
        for event in events:
            feature_set = feature_extractor.extract_all_features(data, event, swings)
            feature_sets.append(feature_set)
        
        features_df = feature_extractor.export_features_to_dataframe(feature_sets)
        
        label_generator = LabelGenerator()
        labels = label_generator.generate_labels(data, events, zones)
        labels_df = label_generator.export_labels_to_dataframe(labels)
        
        if features_df.empty or labels_df.empty:
            print("⚠ Skipping backtest test - insufficient data")
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
        
        results = backtest_engine.run_backtest(data, model_results, events, feature_extractor, swings)
        
        assert results.trades is not None, "Backtest should generate trades"
        assert results.performance_metrics is not None, "Backtest should generate performance metrics"
        assert results.equity_curve is not None, "Backtest should generate equity curve"
        
        # Check performance metrics
        metrics = results.performance_metrics
        assert 'total_trades' in metrics, "Should have total trades metric"
        assert 'win_rate' in metrics, "Should have win rate metric"
        assert 'total_return' in metrics, "Should have total return metric"
        
        print(f"✓ Backtest test passed: {metrics.get('total_trades', 0)} trades executed")


class TestIntegratedSystem:
    """Test the integrated system."""
    
    def test_complete_system(self):
        """Test the complete integrated system."""
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
        sample_data = system.create_sample_data(800, 100.0, 0.02)
        
        # Train system
        model_results = system.train(sample_data, save_models=False)
        
        assert model_results is not None, "System should train successfully"
        assert system.trained_models is not None, "System should store trained models"
        assert system.last_training_data is not None, "System should store training data"
        
        # Run backtest
        backtest_results = system.backtest()
        
        assert backtest_results is not None, "System should run backtest successfully"
        
        # Generate predictions
        predictions = system.predict_live(sample_data.tail(100))
        
        assert isinstance(predictions, list), "Should generate predictions list"
        
        # Check system status
        status = system.get_system_status()
        
        assert status['trained'], "System should report as trained"
        assert status['last_training_data_available'], "System should report training data available"
        
        print(f"✓ Integrated system test passed: Complete pipeline working")


def run_all_tests():
    """Run all tests."""
    print("Running Fibonacci ML System Tests")
    print("=" * 50)
    
    test_classes = [
        TestSwingDetector,
        TestFibZones,
        TestEventGenerator,
        TestFeatureExtractor,
        TestLabelGenerator,
        TestModelTrainer,
        TestBacktestEngine,
        TestIntegratedSystem
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {test_method} failed: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠ {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    run_all_tests()