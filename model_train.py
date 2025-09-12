"""
Model Training Module

Trains LightGBM models for classification and regression tasks.
Includes probability calibration and model evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
from features import FeatureSet
from labeling import OutcomeLabel


@dataclass
class ModelResults:
    """Container for model training results."""
    classifier: Any
    target_regressor: Any
    duration_regressor: Any
    calibration_model: Any
    feature_importance: Dict
    performance_metrics: Dict
    validation_scores: Dict


class ModelTrainer:
    """
    Trains LightGBM models for Fibonacci zone outcome prediction.
    
    Includes classification (outcome type), regression (target levels),
    and probability calibration for live trading.
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 validation_size: float = 0.2,
                 random_state: int = 42,
                 n_splits: int = 5):
        """
        Initialize model trainer.
        
        Args:
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            random_state: Random seed for reproducibility
            n_splits: Number of splits for time series cross-validation
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.n_splits = n_splits
        
        # LightGBM parameters
        self.classifier_params = {
            'objective': 'multiclass',
            'num_class': 3,  # Reversal, Continuation, Breakout
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        self.regressor_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
    
    def prepare_training_data(self, 
                            features_df: pd.DataFrame,
                            labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with proper splits.
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Merge features and labels
        merged_df = pd.merge(features_df, labels_df, on='event_id', how='inner')
        
        if merged_df.empty:
            raise ValueError("No matching features and labels found")
        
        # Sort by event_id to maintain temporal order
        merged_df = merged_df.sort_values('event_id')
        
        # Prepare features (X) and targets (y)
        feature_columns = [col for col in features_df.columns if col != 'event_id']
        X = merged_df[feature_columns]
        
        # Classification target
        y_class = merged_df['outcome_type']
        
        # Regression targets
        y_target = merged_df['target_price']
        y_duration = merged_df['duration_bars']
        
        # Handle missing values
        X = X.fillna(X.median())
        y_target = y_target.fillna(y_target.median())
        y_duration = y_duration.fillna(y_duration.median())
        
        # Time series split for temporal data
        n_samples = len(X)
        train_size = int(n_samples * (1 - self.test_size - self.validation_size))
        val_size = int(n_samples * self.validation_size)
        
        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]
        
        y_train_class = y_class.iloc[:train_size]
        y_val_class = y_class.iloc[train_size:train_size + val_size]
        y_test_class = y_class.iloc[train_size + val_size:]
        
        y_train_target = y_target.iloc[:train_size]
        y_val_target = y_target.iloc[train_size:train_size + val_size]
        y_test_target = y_target.iloc[train_size + val_size:]
        
        y_train_duration = y_duration.iloc[:train_size]
        y_val_duration = y_duration.iloc[train_size:train_size + val_size]
        y_test_duration = y_duration.iloc[train_size + val_size:]
        
        return (X_train, X_val, X_test, 
                y_train_class, y_val_class, y_test_class,
                y_train_target, y_val_target, y_test_target,
                y_train_duration, y_val_duration, y_test_duration)
    
    def train_classifier(self, 
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: pd.DataFrame,
                        y_val: pd.Series) -> Tuple[Any, Dict]:
        """
        Train the outcome classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (trained_model, performance_metrics)
        """
        # Encode labels
        label_encoder = {label: idx for idx, label in enumerate(y_train.unique())}
        reverse_encoder = {idx: label for label, idx in label_encoder.items()}
        
        y_train_encoded = y_train.map(label_encoder)
        y_val_encoded = y_val.map(label_encoder)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train_encoded)
        val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)
        
        # Train model
        model = lgb.train(
            self.classifier_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Make predictions
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        y_val_decoded = [reverse_encoder[pred] for pred in y_pred]
        metrics = self._calculate_classification_metrics(y_val, y_val_decoded, y_pred_proba, reverse_encoder)
        
        # Store encoders for later use
        model.label_encoder = label_encoder
        model.reverse_encoder = reverse_encoder
        
        return model, metrics
    
    def train_regressors(self, 
                        X_train: pd.DataFrame,
                        y_train_target: pd.Series,
                        y_train_duration: pd.Series,
                        X_val: pd.DataFrame,
                        y_val_target: pd.Series,
                        y_val_duration: pd.Series) -> Tuple[Any, Any, Dict]:
        """
        Train regression models for target price and duration.
        
        Args:
            X_train: Training features
            y_train_target: Training target prices
            y_train_duration: Training durations
            X_val: Validation features
            y_val_target: Validation target prices
            y_val_duration: Validation durations
            
        Returns:
            Tuple of (target_model, duration_model, metrics)
        """
        # Train target price regressor
        train_data_target = lgb.Dataset(X_train, label=y_train_target)
        val_data_target = lgb.Dataset(X_val, label=y_val_target, reference=train_data_target)
        
        target_model = lgb.train(
            self.regressor_params,
            train_data_target,
            valid_sets=[val_data_target],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Train duration regressor
        train_data_duration = lgb.Dataset(X_train, label=y_train_duration)
        val_data_duration = lgb.Dataset(X_val, label=y_val_duration, reference=train_data_duration)
        
        duration_model = lgb.train(
            self.regressor_params,
            train_data_duration,
            valid_sets=[val_data_duration],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Calculate regression metrics
        target_pred = target_model.predict(X_val, num_iteration=target_model.best_iteration)
        duration_pred = duration_model.predict(X_val, num_iteration=duration_model.best_iteration)
        
        metrics = self._calculate_regression_metrics(
            y_val_target, target_pred, y_val_duration, duration_pred
        )
        
        return target_model, duration_model, metrics
    
    def calibrate_probabilities(self, 
                               classifier: Any,
                               X_val: pd.DataFrame,
                               y_val: pd.Series) -> Any:
        """
        Calibrate classifier probabilities for better confidence estimates.
        
        Args:
            classifier: Trained classifier
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Calibrated classifier
        """
        # Get predictions from the classifier
        y_pred_proba = classifier.predict(X_val, num_iteration=classifier.best_iteration)
        
        # Encode labels
        y_val_encoded = y_val.map(classifier.label_encoder)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(
            base_estimator=classifier,
            method='isotonic',
            cv=3
        )
        
        # Fit calibration model
        calibrated_model.fit(X_val, y_val_encoded)
        
        return calibrated_model
    
    def train_all_models(self, 
                        features_df: pd.DataFrame,
                        labels_df: pd.DataFrame) -> ModelResults:
        """
        Train all models (classifier, regressors, calibration).
        
        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            
        Returns:
            ModelResults object with all trained models
        """
        print("Preparing training data...")
        (X_train, X_val, X_test, 
         y_train_class, y_val_class, y_test_class,
         y_train_target, y_val_target, y_test_target,
         y_train_duration, y_val_duration, y_test_duration) = self.prepare_training_data(features_df, labels_df)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train classifier
        print("Training classifier...")
        classifier, classifier_metrics = self.train_classifier(
            X_train, y_train_class, X_val, y_val_class
        )
        
        # Train regressors
        print("Training regressors...")
        target_regressor, duration_regressor, regression_metrics = self.train_regressors(
            X_train, y_train_target, y_train_duration,
            X_val, y_val_target, y_val_duration
        )
        
        # Calibrate probabilities
        print("Calibrating probabilities...")
        calibration_model = self.calibrate_probabilities(classifier, X_val, y_val_class)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(classifier, X_train.columns)
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = self._evaluate_test_set(
            classifier, target_regressor, duration_regressor, calibration_model,
            X_test, y_test_class, y_test_target, y_test_duration
        )
        
        # Combine all metrics
        performance_metrics = {
            'classifier': classifier_metrics,
            'regression': regression_metrics,
            'test': test_metrics
        }
        
        return ModelResults(
            classifier=classifier,
            target_regressor=target_regressor,
            duration_regressor=duration_regressor,
            calibration_model=calibration_model,
            feature_importance=feature_importance,
            performance_metrics=performance_metrics,
            validation_scores={}
        )
    
    def _calculate_classification_metrics(self, 
                                        y_true: pd.Series,
                                        y_pred: List[str],
                                        y_pred_proba: np.ndarray,
                                        reverse_encoder: Dict) -> Dict:
        """Calculate classification metrics."""
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (multiclass)
        try:
            y_true_encoded = y_true.map({v: k for k, v in reverse_encoder.items()})
            roc_auc = roc_auc_score(y_true_encoded, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = 0.0
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'accuracy': report['accuracy']
        }
    
    def _calculate_regression_metrics(self, 
                                    y_true_target: pd.Series,
                                    y_pred_target: np.ndarray,
                                    y_true_duration: pd.Series,
                                    y_pred_duration: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        # Target price metrics
        target_mae = np.mean(np.abs(y_true_target - y_pred_target))
        target_rmse = np.sqrt(np.mean((y_true_target - y_pred_target) ** 2))
        target_r2 = 1 - np.sum((y_true_target - y_pred_target) ** 2) / np.sum((y_true_target - y_true_target.mean()) ** 2)
        
        # Duration metrics
        duration_mae = np.mean(np.abs(y_true_duration - y_pred_duration))
        duration_rmse = np.sqrt(np.mean((y_true_duration - y_pred_duration) ** 2))
        duration_r2 = 1 - np.sum((y_true_duration - y_pred_duration) ** 2) / np.sum((y_true_duration - y_true_duration.mean()) ** 2)
        
        return {
            'target_price': {
                'mae': target_mae,
                'rmse': target_rmse,
                'r2': target_r2
            },
            'duration': {
                'mae': duration_mae,
                'rmse': duration_rmse,
                'r2': duration_r2
            }
        }
    
    def _calculate_feature_importance(self, 
                                    model: Any,
                                    feature_names: List[str]) -> Dict:
        """Calculate feature importance."""
        importance = model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def _evaluate_test_set(self, 
                          classifier: Any,
                          target_regressor: Any,
                          duration_regressor: Any,
                          calibration_model: Any,
                          X_test: pd.DataFrame,
                          y_test_class: pd.Series,
                          y_test_target: pd.Series,
                          y_test_duration: pd.Series) -> Dict:
        """Evaluate models on test set."""
        # Classifier predictions
        y_pred_proba = classifier.predict(X_test, num_iteration=classifier.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_decoded = [classifier.reverse_encoder[pred] for pred in y_pred]
        
        # Regression predictions
        target_pred = target_regressor.predict(X_test, num_iteration=target_regressor.best_iteration)
        duration_pred = duration_regressor.predict(X_test, num_iteration=duration_regressor.best_iteration)
        
        # Calibrated probabilities
        y_calibrated_proba = calibration_model.predict_proba(X_test)
        
        # Calculate metrics
        classifier_metrics = self._calculate_classification_metrics(
            y_test_class, y_pred_decoded, y_pred_proba, classifier.reverse_encoder
        )
        
        regression_metrics = self._calculate_regression_metrics(
            y_test_target, target_pred, y_test_duration, duration_pred
        )
        
        return {
            'classifier': classifier_metrics,
            'regression': regression_metrics,
            'calibrated_probabilities': y_calibrated_proba.tolist()
        }
    
    def save_models(self, 
                   model_results: ModelResults,
                   filepath: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            model_results: ModelResults object
            filepath: Base filepath for saving models
        """
        # Save individual models
        joblib.dump(model_results.classifier, f"{filepath}_classifier.pkl")
        joblib.dump(model_results.target_regressor, f"{filepath}_target_regressor.pkl")
        joblib.dump(model_results.duration_regressor, f"{filepath}_duration_regressor.pkl")
        joblib.dump(model_results.calibration_model, f"{filepath}_calibration.pkl")
        
        # Save metadata
        metadata = {
            'feature_importance': model_results.feature_importance,
            'performance_metrics': model_results.performance_metrics,
            'validation_scores': model_results.validation_scores
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        print(f"Models saved to {filepath}_*.pkl")
    
    def load_models(self, filepath: str) -> ModelResults:
        """
        Load trained models from disk.
        
        Args:
            filepath: Base filepath for loading models
            
        Returns:
            ModelResults object
        """
        classifier = joblib.load(f"{filepath}_classifier.pkl")
        target_regressor = joblib.load(f"{filepath}_target_regressor.pkl")
        duration_regressor = joblib.load(f"{filepath}_duration_regressor.pkl")
        calibration_model = joblib.load(f"{filepath}_calibration.pkl")
        metadata = joblib.load(f"{filepath}_metadata.pkl")
        
        return ModelResults(
            classifier=classifier,
            target_regressor=target_regressor,
            duration_regressor=duration_regressor,
            calibration_model=calibration_model,
            feature_importance=metadata['feature_importance'],
            performance_metrics=metadata['performance_metrics'],
            validation_scores=metadata['validation_scores']
        )


def test_model_trainer():
    """Test the model trainer with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    from event_generator import EventGenerator
    from features import FeatureExtractor
    from labeling import LabelGenerator
    
    print("Testing Model Trainer...")
    
    # Create sample data
    sample_data = create_sample_data(800, 100.0, 0.02)
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
        print("No events generated for training")
        return
    
    # Extract features
    feature_extractor = FeatureExtractor()
    feature_sets = []
    for event in events:
        feature_set = feature_extractor.extract_all_features(sample_data, event, swings)
        feature_sets.append(feature_set)
    
    features_df = feature_extractor.export_features_to_dataframe(feature_sets)
    print(f"Extracted features: {features_df.shape}")
    
    # Generate labels
    label_generator = LabelGenerator(
        lookforward_window=30,
        reversal_threshold=2.0,
        continuation_threshold=1.0,
        atr_multiplier=1.0,
        min_confidence=0.5
    )
    
    labels = label_generator.generate_labels(sample_data, events, zones)
    labels_df = label_generator.export_labels_to_dataframe(labels)
    print(f"Generated labels: {labels_df.shape}")
    
    if features_df.empty or labels_df.empty:
        print("No features or labels for training")
        return
    
    # Train models
    trainer = ModelTrainer(
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
        n_splits=5
    )
    
    model_results = trainer.train_all_models(features_df, labels_df)
    
    # Print results
    print("\nModel Training Results:")
    print(f"Classifier accuracy: {model_results.performance_metrics['classifier']['accuracy']:.3f}")
    print(f"Target regressor R²: {model_results.performance_metrics['regression']['target_price']['r2']:.3f}")
    print(f"Duration regressor R²: {model_results.performance_metrics['regression']['duration']['r2']:.3f}")
    
    print("\nTop 10 Feature Importance:")
    for i, (feature, importance) in enumerate(list(model_results.feature_importance.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.3f}")
    
    print("Model training test completed successfully!")


if __name__ == "__main__":
    test_model_trainer()