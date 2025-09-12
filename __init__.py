"""
Adaptive Fibonacci Machine Learning System

A system that learns which Fibonacci retracement/extension ranges matter
in different market contexts and predicts reversal vs continuation outcomes.
"""

__version__ = "1.0.0"
__author__ = "Quant Research Engineer"

from .swing_detector import SwingDetector
from .fib_zones import FibZoneGenerator
from .event_generator import EventGenerator
from .features import FeatureExtractor
from .labeling import LabelGenerator
from .model_train import ModelTrainer
from .backtest import BacktestEngine
from .live_pipeline import LivePipeline

__all__ = [
    "SwingDetector",
    "FibZoneGenerator", 
    "EventGenerator",
    "FeatureExtractor",
    "LabelGenerator",
    "ModelTrainer",
    "BacktestEngine",
    "LivePipeline"
]