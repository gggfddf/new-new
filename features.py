"""
Feature Extraction Module

Extracts price-action features from OHLCV data and zone events.
Focuses on market regime, impulse, zone interaction, and structural features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from event_generator import ZoneEvent
from fib_zones import FibZone
from swing_detector import Swing
import warnings


@dataclass
class FeatureSet:
    """Container for extracted features."""
    market_regime_features: Dict
    impulse_features: Dict
    zone_interaction_features: Dict
    structural_features: Dict
    temporal_features: Dict
    raw_features: Dict


class FeatureExtractor:
    """
    Extracts comprehensive price-action features from OHLCV data and zone events.
    
    Generates features for market regime, impulse characteristics, zone interactions,
    structural patterns, and temporal effects.
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 trend_period: int = 20,
                 volatility_period: int = 20):
        """
        Initialize feature extractor.
        
        Args:
            atr_period: Period for ATR calculation
            trend_period: Period for trend analysis
            volatility_period: Period for volatility analysis
        """
        self.atr_period = atr_period
        self.trend_period = trend_period
        self.volatility_period = volatility_period
    
    def extract_all_features(self, 
                           data: pd.DataFrame,
                           event: ZoneEvent,
                           swings: List[Swing]) -> FeatureSet:
        """
        Extract all features for a zone event.
        
        Args:
            data: OHLCV DataFrame
            event: ZoneEvent object
            swings: List of Swing objects
            
        Returns:
            FeatureSet object with all extracted features
        """
        # Extract market regime features
        market_regime = self._extract_market_regime_features(data, event, swings)
        
        # Extract impulse features
        impulse = self._extract_impulse_features(data, event, swings)
        
        # Extract zone interaction features
        zone_interaction = self._extract_zone_interaction_features(data, event)
        
        # Extract structural features
        structural = self._extract_structural_features(data, event, swings)
        
        # Extract temporal features
        temporal = self._extract_temporal_features(data, event)
        
        # Combine all features
        raw_features = {
            **market_regime,
            **impulse,
            **zone_interaction,
            **structural,
            **temporal
        }
        
        return FeatureSet(
            market_regime_features=market_regime,
            impulse_features=impulse,
            zone_interaction_features=zone_interaction,
            structural_features=structural,
            temporal_features=temporal,
            raw_features=raw_features
        )
    
    def _extract_market_regime_features(self, 
                                      data: pd.DataFrame,
                                      event: ZoneEvent,
                                      swings: List[Swing]) -> Dict:
        """Extract market regime features."""
        features = {}
        
        # Calculate ATR
        atr = self._calculate_atr(data, self.atr_period)
        current_atr = atr.iloc[event.entry_index] if event.entry_index < len(atr) else atr.iloc[-1]
        
        # Trend direction features
        trend_features = self._calculate_trend_features(data, event, swings)
        features.update(trend_features)
        
        # Volatility regime features
        volatility_features = self._calculate_volatility_features(data, event, atr)
        features.update(volatility_features)
        
        # ATR relative to swing size
        swing_size = abs(event.zone.swing_end.price - event.zone.swing_start.price)
        features['atr_to_swing_ratio'] = current_atr / swing_size if swing_size > 0 else 0
        
        # Market structure features
        structure_features = self._calculate_market_structure_features(data, event, swings)
        features.update(structure_features)
        
        return features
    
    def _extract_impulse_features(self, 
                                data: pd.DataFrame,
                                event: ZoneEvent,
                                swings: List[Swing]) -> Dict:
        """Extract impulse characteristics features."""
        features = {}
        
        # Impulse duration (bars from swing start to swing end)
        impulse_duration = abs(event.zone.swing_end.index - event.zone.swing_start.index)
        features['impulse_duration_bars'] = impulse_duration
        
        # Impulse strength (normalized by ATR)
        atr = self._calculate_atr(data, self.atr_period)
        swing_start_atr = atr.iloc[event.zone.swing_start.index] if event.zone.swing_start.index < len(atr) else atr.iloc[-1]
        
        swing_size = abs(event.zone.swing_end.price - event.zone.swing_start.price)
        features['impulse_strength_atr'] = swing_size / swing_start_atr if swing_start_atr > 0 else 0
        
        # Average bar range during impulse
        impulse_start = min(event.zone.swing_start.index, event.zone.swing_end.index)
        impulse_end = max(event.zone.swing_start.index, event.zone.swing_end.index)
        
        if impulse_start < len(data) and impulse_end < len(data):
            impulse_data = data.iloc[impulse_start:impulse_end+1]
            avg_range = (impulse_data['high'] - impulse_data['low']).mean()
            features['avg_impulse_bar_range'] = avg_range
            features['avg_impulse_bar_range_atr'] = avg_range / swing_start_atr if swing_start_atr > 0 else 0
        
        # Impulse acceleration (rate of change in momentum)
        acceleration = self._calculate_impulse_acceleration(data, event)
        features['impulse_acceleration'] = acceleration
        
        # Volume during impulse
        if impulse_start < len(data) and impulse_end < len(data):
            impulse_volume = data.iloc[impulse_start:impulse_end+1]['volume'].mean()
            features['avg_impulse_volume'] = impulse_volume
        
        return features
    
    def _extract_zone_interaction_features(self, 
                                         data: pd.DataFrame,
                                         event: ZoneEvent) -> Dict:
        """Extract zone interaction features."""
        features = {}
        
        # Entry position in zone (0=bottom, 1=top)
        zone_lower, zone_upper = event.zone.price_range
        zone_width = zone_upper - zone_lower
        
        if zone_width > 0:
            entry_position = (event.entry_price - zone_lower) / zone_width
            features['entry_position_in_zone'] = entry_position
        else:
            features['entry_position_in_zone'] = 0.5
        
        # Wick rejection ratio
        if event.wick_rejection is not None:
            features['wick_rejection'] = 1 if event.wick_rejection else 0
        else:
            features['wick_rejection'] = 0
        
        # Time spent in zone (bars)
        if event.duration_bars is not None:
            features['time_in_zone_bars'] = event.duration_bars
        else:
            features['time_in_zone_bars'] = 0
        
        # Retest count
        features['retest_count'] = event.retest_count
        
        # Break speed (bars to touch vs impulse duration)
        impulse_duration = abs(event.zone.swing_end.index - event.zone.swing_start.index)
        if impulse_duration > 0:
            features['break_speed_ratio'] = event.entry_index / impulse_duration
        else:
            features['break_speed_ratio'] = 0
        
        # Price range within zone
        if event.max_price_in_zone is not None and event.min_price_in_zone is not None:
            price_range_in_zone = event.max_price_in_zone - event.min_price_in_zone
            features['price_range_in_zone'] = price_range_in_zone
            features['price_range_in_zone_ratio'] = price_range_in_zone / zone_width if zone_width > 0 else 0
        
        # Volume in zone
        if event.volume_in_zone is not None:
            features['volume_in_zone'] = event.volume_in_zone
            # Normalize by average volume
            avg_volume = data['volume'].mean()
            features['volume_in_zone_ratio'] = event.volume_in_zone / avg_volume if avg_volume > 0 else 0
        
        return features
    
    def _extract_structural_features(self, 
                                   data: pd.DataFrame,
                                   event: ZoneEvent,
                                   swings: List[Swing]) -> Dict:
        """Extract structural pattern features."""
        features = {}
        
        # Swing width normalized by ATR
        atr = self._calculate_atr(data, self.atr_period)
        current_atr = atr.iloc[event.entry_index] if event.entry_index < len(atr) else atr.iloc[-1]
        
        swing_size = abs(event.zone.swing_end.price - event.zone.swing_start.price)
        features['swing_width_atr'] = swing_size / current_atr if current_atr > 0 else 0
        
        # Zone width normalized by ATR
        zone_width = event.zone.zone_width
        features['zone_width_atr'] = zone_width / current_atr if current_atr > 0 else 0
        
        # Zone strength
        features['zone_strength'] = event.zone.strength
        
        # Fibonacci level
        features['fib_level'] = event.zone.level
        
        # Zone type
        features['zone_type_retracement'] = 1 if event.zone.zone_type == 'retracement' else 0
        features['zone_type_extension'] = 1 if event.zone.zone_type == 'extension' else 0
        
        # Swing context (trend direction)
        if event.zone.swing_start.swing_type == 'high' and event.zone.swing_end.swing_type == 'low':
            features['swing_context_downtrend'] = 1
            features['swing_context_uptrend'] = 0
        elif event.zone.swing_start.swing_type == 'low' and event.zone.swing_end.swing_type == 'high':
            features['swing_context_uptrend'] = 1
            features['swing_context_downtrend'] = 0
        else:
            features['swing_context_uptrend'] = 0
            features['swing_context_downtrend'] = 0
        
        # Multiple swing analysis
        multi_swing_features = self._calculate_multi_swing_features(event, swings)
        features.update(multi_swing_features)
        
        return features
    
    def _extract_temporal_features(self, 
                                 data: pd.DataFrame,
                                 event: ZoneEvent) -> Dict:
        """Extract temporal features."""
        features = {}
        
        # Session hour (captures liquidity cycles)
        entry_hour = event.entry_time.hour
        features['session_hour'] = entry_hour
        
        # Day of week
        entry_dow = event.entry_time.dayofweek
        features['day_of_week'] = entry_dow
        
        # Time since market open (if available)
        if hasattr(event.entry_time, 'date'):
            market_open = event.entry_time.replace(hour=9, minute=30, second=0, microsecond=0)
            time_since_open = (event.entry_time - market_open).total_seconds() / 3600
            features['hours_since_market_open'] = max(0, time_since_open)
        
        # Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * entry_hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * entry_hour / 24)
        features['dow_sin'] = np.sin(2 * np.pi * entry_dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * entry_dow / 7)
        
        return features
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_trend_features(self, 
                                data: pd.DataFrame,
                                event: ZoneEvent,
                                swings: List[Swing]) -> Dict:
        """Calculate trend direction features."""
        features = {}
        
        # Compare closes vs prior swings
        current_close = data['close'].iloc[event.entry_index]
        
        # Find recent swings
        recent_swings = [s for s in swings if s.index < event.entry_index]
        if len(recent_swings) >= 2:
            # Compare with last two swings
            last_swing = recent_swings[-1]
            second_last_swing = recent_swings[-2]
            
            # Trend direction
            if last_swing.swing_type == 'high' and second_last_swing.swing_type == 'low':
                if last_swing.price > second_last_swing.price:
                    features['trend_direction_up'] = 1
                    features['trend_direction_down'] = 0
                else:
                    features['trend_direction_up'] = 0
                    features['trend_direction_down'] = 1
            elif last_swing.swing_type == 'low' and second_last_swing.swing_type == 'high':
                if last_swing.price < second_last_swing.price:
                    features['trend_direction_down'] = 1
                    features['trend_direction_up'] = 0
                else:
                    features['trend_direction_down'] = 0
                    features['trend_direction_up'] = 1
            else:
                features['trend_direction_up'] = 0
                features['trend_direction_down'] = 0
        
        # Moving average trend
        if event.entry_index >= self.trend_period:
            ma_short = data['close'].rolling(window=self.trend_period//2).mean().iloc[event.entry_index]
            ma_long = data['close'].rolling(window=self.trend_period).mean().iloc[event.entry_index]
            features['ma_trend_bullish'] = 1 if ma_short > ma_long else 0
            features['ma_trend_bearish'] = 1 if ma_short < ma_long else 0
        
        return features
    
    def _calculate_volatility_features(self, 
                                     data: pd.DataFrame,
                                     event: ZoneEvent,
                                     atr: pd.Series) -> Dict:
        """Calculate volatility regime features."""
        features = {}
        
        # ATR quantile
        if event.entry_index < len(atr):
            current_atr = atr.iloc[event.entry_index]
            atr_quantile = (atr.iloc[:event.entry_index+1] <= current_atr).mean()
            features['atr_quantile'] = atr_quantile
            
            # Volatility regime
            if atr_quantile > 0.8:
                features['volatility_regime_high'] = 1
                features['volatility_regime_medium'] = 0
                features['volatility_regime_low'] = 0
            elif atr_quantile < 0.2:
                features['volatility_regime_high'] = 0
                features['volatility_regime_medium'] = 0
                features['volatility_regime_low'] = 1
            else:
                features['volatility_regime_high'] = 0
                features['volatility_regime_medium'] = 1
                features['volatility_regime_low'] = 0
        
        return features
    
    def _calculate_market_structure_features(self, 
                                           data: pd.DataFrame,
                                           event: ZoneEvent,
                                           swings: List[Swing]) -> Dict:
        """Calculate market structure features."""
        features = {}
        
        # Higher highs, higher lows pattern
        recent_swings = [s for s in swings if s.index < event.entry_index]
        if len(recent_swings) >= 4:
            highs = [s for s in recent_swings[-4:] if s.swing_type == 'high']
            lows = [s for s in recent_swings[-4:] if s.swing_type == 'low']
            
            if len(highs) >= 2:
                features['higher_highs'] = 1 if highs[-1].price > highs[-2].price else 0
            if len(lows) >= 2:
                features['higher_lows'] = 1 if lows[-1].price > lows[-2].price else 0
        
        # Market structure break
        features['structure_break'] = self._detect_structure_break(data, event, swings)
        
        return features
    
    def _calculate_impulse_acceleration(self, 
                                      data: pd.DataFrame,
                                      event: ZoneEvent) -> float:
        """Calculate impulse acceleration."""
        swing_start = event.zone.swing_start.index
        swing_end = event.zone.swing_end.index
        
        if swing_start >= len(data) or swing_end >= len(data):
            return 0
        
        # Calculate momentum change
        impulse_data = data.iloc[swing_start:swing_end+1]
        if len(impulse_data) < 3:
            return 0
        
        # Calculate rate of change in momentum
        momentum = impulse_data['close'].diff()
        acceleration = momentum.diff().mean()
        
        return acceleration
    
    def _calculate_multi_swing_features(self, 
                                      event: ZoneEvent,
                                      swings: List[Swing]) -> Dict:
        """Calculate features from multiple swings."""
        features = {}
        
        # Find swings around the current event
        event_swings = [s for s in swings if abs(s.index - event.entry_index) <= 50]
        
        if len(event_swings) >= 3:
            # Swing frequency
            time_span = max(s.index for s in event_swings) - min(s.index for s in event_swings)
            features['swing_frequency'] = len(event_swings) / time_span if time_span > 0 else 0
            
            # Swing size variation
            swing_sizes = [abs(s.price - event_swings[i-1].price) for i, s in enumerate(event_swings[1:], 1)]
            if swing_sizes:
                features['swing_size_std'] = np.std(swing_sizes)
                features['swing_size_cv'] = np.std(swing_sizes) / np.mean(swing_sizes) if np.mean(swing_sizes) > 0 else 0
        
        return features
    
    def _detect_structure_break(self, 
                              data: pd.DataFrame,
                              event: ZoneEvent,
                              swings: List[Swing]) -> int:
        """Detect market structure break."""
        # Simple implementation - can be enhanced
        recent_swings = [s for s in swings if s.index < event.entry_index]
        
        if len(recent_swings) >= 2:
            last_swing = recent_swings[-1]
            second_last_swing = recent_swings[-2]
            
            # Check for break of previous swing
            if (last_swing.swing_type == 'high' and 
                event.entry_price > last_swing.price):
                return 1
            elif (last_swing.swing_type == 'low' and 
                  event.entry_price < last_swing.price):
                return 1
        
        return 0
    
    def export_features_to_dataframe(self, 
                                   feature_sets: List[FeatureSet]) -> pd.DataFrame:
        """
        Export feature sets to a pandas DataFrame.
        
        Args:
            feature_sets: List of FeatureSet objects
            
        Returns:
            DataFrame with all features
        """
        if not feature_sets:
            return pd.DataFrame()
        
        feature_data = []
        for feature_set in feature_sets:
            feature_data.append(feature_set.raw_features)
        
        return pd.DataFrame(feature_data)


def test_feature_extractor():
    """Test the feature extractor with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    from event_generator import EventGenerator
    
    print("Testing Feature Extractor...")
    
    # Create sample data
    sample_data = create_sample_data(400, 100.0, 0.02)
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
        print("No events generated for feature extraction")
        return
    
    # Extract features
    feature_extractor = FeatureExtractor(
        atr_period=14,
        trend_period=20,
        volatility_period=20
    )
    
    feature_sets = []
    for event in events[:10]:  # Test with first 10 events
        feature_set = feature_extractor.extract_all_features(sample_data, event, swings)
        feature_sets.append(feature_set)
    
    print(f"Extracted features for {len(feature_sets)} events")
    
    # Export to DataFrame
    features_df = feature_extractor.export_features_to_dataframe(feature_sets)
    print(f"Features DataFrame shape: {features_df.shape}")
    
    if not features_df.empty:
        print(f"Number of features: {len(features_df.columns)}")
        print("\nSample features:")
        print(features_df[['trend_direction_up', 'volatility_regime_high', 'wick_rejection', 'fib_level']].head())
    
    print("Feature extraction test completed successfully!")


if __name__ == "__main__":
    test_feature_extractor()