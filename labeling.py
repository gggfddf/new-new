"""
Labeling Module

Assigns outcome labels to zone events based on future price action.
Defines reversal, continuation, and breakout outcomes with target levels.
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
class OutcomeLabel:
    """Represents an outcome label for a zone event."""
    event_id: str
    outcome_type: str  # 'Reversal', 'Continuation', 'Breakout'
    confidence: float  # Label confidence (0-1)
    target_price: Optional[float] = None
    target_zone: Optional[Tuple[float, float]] = None
    stop_loss_level: Optional[float] = None
    duration_bars: Optional[int] = None
    max_adverse_excursion: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    context_tags: List[str] = None


class LabelGenerator:
    """
    Generates outcome labels for zone events based on future price action.
    
    Uses a look-forward window to determine if a zone touch resulted in
    reversal, continuation, or breakout behavior.
    """
    
    def __init__(self, 
                 lookforward_window: int = 30,
                 reversal_threshold: float = 2.0,
                 continuation_threshold: float = 1.0,
                 atr_multiplier: float = 1.0,
                 min_confidence: float = 0.6):
        """
        Initialize label generator.
        
        Args:
            lookforward_window: Number of bars to look forward for outcome
            reversal_threshold: R-multiple threshold for reversal (2R gain before 1R loss)
            continuation_threshold: R-multiple threshold for continuation (1R loss before 2R gain)
            atr_multiplier: ATR multiplier for risk calculation
            min_confidence: Minimum confidence for valid labels
        """
        self.lookforward_window = lookforward_window
        self.reversal_threshold = reversal_threshold
        self.continuation_threshold = continuation_threshold
        self.atr_multiplier = atr_multiplier
        self.min_confidence = min_confidence
    
    def generate_labels(self, 
                       data: pd.DataFrame,
                       events: List[ZoneEvent],
                       zones: List[FibZone]) -> List[OutcomeLabel]:
        """
        Generate outcome labels for all events.
        
        Args:
            data: OHLCV DataFrame
            events: List of ZoneEvent objects
            zones: List of FibZone objects
            
        Returns:
            List of OutcomeLabel objects
        """
        labels = []
        
        for event in events:
            label = self._generate_single_label(data, event, zones)
            if label and label.confidence >= self.min_confidence:
                labels.append(label)
        
        return labels
    
    def _generate_single_label(self, 
                              data: pd.DataFrame,
                              event: ZoneEvent,
                              zones: List[FibZone]) -> Optional[OutcomeLabel]:
        """
        Generate a single outcome label for an event.
        
        Args:
            data: OHLCV DataFrame
            event: ZoneEvent object
            zones: List of FibZone objects
            
        Returns:
            OutcomeLabel object or None if invalid
        """
        if event.entry_index >= len(data):
            return None
        
        # Calculate ATR for risk measurement
        atr = self._calculate_atr(data, 14)
        if event.entry_index >= len(atr):
            return None
        
        current_atr = atr.iloc[event.entry_index]
        risk_amount = current_atr * self.atr_multiplier
        
        # Get future price data
        future_start = event.entry_index + 1
        future_end = min(event.entry_index + self.lookforward_window + 1, len(data))
        
        if future_start >= future_end:
            return None
        
        future_data = data.iloc[future_start:future_end]
        
        # Determine outcome type
        outcome_type, confidence = self._determine_outcome_type(
            data, event, future_data, risk_amount
        )
        
        if outcome_type is None:
            return None
        
        # Calculate target and stop levels
        target_price, target_zone = self._calculate_target_levels(
            data, event, future_data, outcome_type, zones
        )
        
        stop_loss_level = self._calculate_stop_loss_level(
            data, event, outcome_type, risk_amount
        )
        
        # Calculate risk-reward ratio
        if target_price and stop_loss_level:
            entry_price = event.entry_price
            if outcome_type == 'Reversal':
                # For reversal, target is opposite direction
                if event.zone.zone_type == 'retracement':
                    # Retracement reversal - target is back to swing extreme
                    if event.zone.swing_start.swing_type == 'high':
                        target_price = event.zone.swing_start.price
                    else:
                        target_price = event.zone.swing_start.price
                else:
                    # Extension reversal - target is back to zone
                    target_price = (event.zone.price_range[0] + event.zone.price_range[1]) / 2
            
            reward = abs(target_price - entry_price)
            risk = abs(entry_price - stop_loss_level)
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # Calculate excursion metrics
        max_adverse, max_favorable = self._calculate_excursion_metrics(
            data, event, future_data
        )
        
        # Generate context tags
        context_tags = self._generate_context_tags(data, event, outcome_type)
        
        return OutcomeLabel(
            event_id=event.event_id,
            outcome_type=outcome_type,
            confidence=confidence,
            target_price=target_price,
            target_zone=target_zone,
            stop_loss_level=stop_loss_level,
            duration_bars=len(future_data),
            max_adverse_excursion=max_adverse,
            max_favorable_excursion=max_favorable,
            risk_reward_ratio=risk_reward_ratio,
            context_tags=context_tags
        )
    
    def _determine_outcome_type(self, 
                               data: pd.DataFrame,
                               event: ZoneEvent,
                               future_data: pd.DataFrame,
                               risk_amount: float) -> Tuple[Optional[str], float]:
        """
        Determine the outcome type based on future price action.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            future_data: Future price data
            risk_amount: Risk amount in price units
            
        Returns:
            Tuple of (outcome_type, confidence)
        """
        if future_data.empty:
            return None, 0.0
        
        entry_price = event.entry_price
        zone_lower, zone_upper = event.zone.price_range
        
        # Calculate price movements
        future_highs = future_data['high']
        future_lows = future_data['low']
        future_closes = future_data['close']
        
        # Calculate R-multiples
        max_gain = (future_highs.max() - entry_price) / risk_amount
        max_loss = (entry_price - future_lows.min()) / risk_amount
        
        # Determine outcome based on thresholds
        if max_gain >= self.reversal_threshold and max_loss < self.continuation_threshold:
            # Reversal: 2R+ gain before 1R loss
            confidence = min(1.0, max_gain / self.reversal_threshold)
            return 'Reversal', confidence
        
        elif max_loss >= self.continuation_threshold and max_gain < self.reversal_threshold:
            # Continuation: 1R+ loss before 2R gain
            confidence = min(1.0, max_loss / self.continuation_threshold)
            return 'Continuation', confidence
        
        elif max_gain >= self.reversal_threshold and max_loss >= self.continuation_threshold:
            # Both thresholds hit - determine by which came first
            gain_first = self._find_first_threshold_hit(
                future_data, entry_price, self.reversal_threshold, risk_amount, 'gain'
            )
            loss_first = self._find_first_threshold_hit(
                future_data, entry_price, self.continuation_threshold, risk_amount, 'loss'
            )
            
            if gain_first < loss_first:
                return 'Reversal', 0.8
            else:
                return 'Continuation', 0.8
        
        else:
            # Neither threshold hit - check for breakout
            breakout_type, breakout_confidence = self._detect_breakout(
                data, event, future_data, zone_lower, zone_upper
            )
            return breakout_type, breakout_confidence
    
    def _detect_breakout(self, 
                        data: pd.DataFrame,
                        event: ZoneEvent,
                        future_data: pd.DataFrame,
                        zone_lower: float,
                        zone_upper: float) -> Tuple[Optional[str], float]:
        """
        Detect breakout behavior.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            future_data: Future price data
            zone_lower: Zone lower bound
            zone_upper: Zone upper bound
            
        Returns:
            Tuple of (outcome_type, confidence)
        """
        # Check if price broke out of zone
        breakout_up = future_data['high'].max() > zone_upper * 1.01  # 1% buffer
        breakout_down = future_data['low'].min() < zone_lower * 0.99  # 1% buffer
        
        if breakout_up and not breakout_down:
            # Upward breakout
            breakout_strength = (future_data['high'].max() - zone_upper) / (zone_upper - zone_lower)
            confidence = min(1.0, breakout_strength)
            return 'Breakout', confidence
        
        elif breakout_down and not breakout_up:
            # Downward breakout
            breakout_strength = (zone_lower - future_data['low'].min()) / (zone_upper - zone_lower)
            confidence = min(1.0, breakout_strength)
            return 'Breakout', confidence
        
        elif breakout_up and breakout_down:
            # Both directions - check which is stronger
            up_strength = (future_data['high'].max() - zone_upper) / (zone_upper - zone_lower)
            down_strength = (zone_lower - future_data['low'].min()) / (zone_upper - zone_lower)
            
            if up_strength > down_strength:
                return 'Breakout', min(1.0, up_strength)
            else:
                return 'Breakout', min(1.0, down_strength)
        
        else:
            # No clear breakout - low confidence
            return 'Breakout', 0.3
    
    def _find_first_threshold_hit(self, 
                                 future_data: pd.DataFrame,
                                 entry_price: float,
                                 threshold: float,
                                 risk_amount: float,
                                 direction: str) -> int:
        """
        Find the first bar where a threshold was hit.
        
        Args:
            future_data: Future price data
            entry_price: Entry price
            threshold: R-multiple threshold
            risk_amount: Risk amount
            direction: 'gain' or 'loss'
            
        Returns:
            Bar index where threshold was first hit
        """
        target_movement = threshold * risk_amount
        
        for i, (_, row) in enumerate(future_data.iterrows()):
            if direction == 'gain':
                if row['high'] >= entry_price + target_movement:
                    return i
            else:  # loss
                if row['low'] <= entry_price - target_movement:
                    return i
        
        return len(future_data)  # Not hit
    
    def _calculate_target_levels(self, 
                                data: pd.DataFrame,
                                event: ZoneEvent,
                                future_data: pd.DataFrame,
                                outcome_type: str,
                                zones: List[FibZone]) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """
        Calculate target price and zone levels.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            future_data: Future price data
            outcome_type: Outcome type
            zones: List of FibZone objects
            
        Returns:
            Tuple of (target_price, target_zone)
        """
        if outcome_type == 'Reversal':
            # Target is back to swing extreme or next Fibonacci level
            if event.zone.zone_type == 'retracement':
                target_price = event.zone.swing_start.price
            else:
                # Extension reversal - target is back to zone
                target_price = (event.zone.price_range[0] + event.zone.price_range[1]) / 2
            
            # Find next Fibonacci zone as target
            target_zone = self._find_next_fib_zone(event, zones, target_price)
            
        elif outcome_type == 'Continuation':
            # Target is next Fibonacci extension
            target_zone = self._find_next_fib_zone(event, zones, None)
            if target_zone:
                target_price = (target_zone[0] + target_zone[1]) / 2
            else:
                target_price = None
        
        else:  # Breakout
            # Target is based on breakout direction
            if future_data['high'].max() > event.zone.price_range[1]:
                # Upward breakout
                target_price = future_data['high'].max()
                target_zone = (target_price * 0.99, target_price * 1.01)
            else:
                # Downward breakout
                target_price = future_data['low'].min()
                target_zone = (target_price * 0.99, target_price * 1.01)
        
        return target_price, target_zone
    
    def _find_next_fib_zone(self, 
                           event: ZoneEvent,
                           zones: List[FibZone],
                           target_price: Optional[float]) -> Optional[Tuple[float, float]]:
        """
        Find the next Fibonacci zone in the direction of movement.
        
        Args:
            event: ZoneEvent object
            zones: List of FibZone objects
            target_price: Target price (if known)
            
        Returns:
            Target zone tuple or None
        """
        # Find zones from the same swing pair
        swing_zones = [
            z for z in zones 
            if z.swing_start.index == event.zone.swing_start.index and 
               z.swing_end.index == event.zone.swing_end.index
        ]
        
        if not swing_zones:
            return None
        
        # Sort zones by level
        swing_zones.sort(key=lambda z: z.level)
        
        current_level = event.zone.level
        
        # Find next zone in sequence
        for zone in swing_zones:
            if zone.level > current_level:
                return zone.price_range
        
        return None
    
    def _calculate_stop_loss_level(self, 
                                  data: pd.DataFrame,
                                  event: ZoneEvent,
                                  outcome_type: str,
                                  risk_amount: float) -> float:
        """
        Calculate stop loss level.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            outcome_type: Outcome type
            risk_amount: Risk amount
            
        Returns:
            Stop loss level
        """
        entry_price = event.entry_price
        zone_lower, zone_upper = event.zone.price_range
        
        if outcome_type == 'Reversal':
            # Stop loss is beyond the zone
            if event.zone.zone_type == 'retracement':
                if event.zone.swing_start.swing_type == 'high':
                    # Downtrend retracement - stop above zone
                    stop_loss = zone_upper + risk_amount
                else:
                    # Uptrend retracement - stop below zone
                    stop_loss = zone_lower - risk_amount
            else:
                # Extension reversal - stop beyond extension
                stop_loss = entry_price + risk_amount if entry_price > zone_upper else entry_price - risk_amount
        
        elif outcome_type == 'Continuation':
            # Stop loss is beyond the zone in opposite direction
            if event.zone.swing_start.swing_type == 'high':
                # Downtrend - stop above zone
                stop_loss = zone_upper + risk_amount
            else:
                # Uptrend - stop below zone
                stop_loss = zone_lower - risk_amount
        
        else:  # Breakout
            # Stop loss is back inside the zone
            stop_loss = (zone_lower + zone_upper) / 2
        
        return stop_loss
    
    def _calculate_excursion_metrics(self, 
                                   data: pd.DataFrame,
                                   event: ZoneEvent,
                                   future_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate maximum adverse and favorable excursion.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            future_data: Future price data
            
        Returns:
            Tuple of (max_adverse, max_favorable)
        """
        entry_price = event.entry_price
        
        max_adverse = 0
        max_favorable = 0
        
        for _, row in future_data.iterrows():
            # Calculate adverse excursion (unfavorable movement)
            if event.zone.zone_type == 'retracement':
                if event.zone.swing_start.swing_type == 'high':
                    # Downtrend retracement - adverse is upward movement
                    adverse = max(0, row['high'] - entry_price)
                else:
                    # Uptrend retracement - adverse is downward movement
                    adverse = max(0, entry_price - row['low'])
            else:
                # Extension - adverse is movement away from extension
                adverse = max(0, abs(row['high'] - entry_price), abs(entry_price - row['low']))
            
            max_adverse = max(max_adverse, adverse)
            
            # Calculate favorable excursion (favorable movement)
            if event.zone.zone_type == 'retracement':
                if event.zone.swing_start.swing_type == 'high':
                    # Downtrend retracement - favorable is downward movement
                    favorable = max(0, entry_price - row['low'])
                else:
                    # Uptrend retracement - favorable is upward movement
                    favorable = max(0, row['high'] - entry_price)
            else:
                # Extension - favorable is movement toward extension
                favorable = max(0, abs(row['high'] - entry_price), abs(entry_price - row['low']))
            
            max_favorable = max(max_favorable, favorable)
        
        return max_adverse, max_favorable
    
    def _generate_context_tags(self, 
                              data: pd.DataFrame,
                              event: ZoneEvent,
                              outcome_type: str) -> List[str]:
        """
        Generate context tags for the outcome.
        
        Args:
            data: Full OHLCV DataFrame
            event: ZoneEvent object
            outcome_type: Outcome type
            
        Returns:
            List of context tags
        """
        tags = []
        
        # Zone type tags
        tags.append(f"{event.zone.zone_type}_zone")
        tags.append(f"fib_{event.zone.level}")
        
        # Outcome tags
        tags.append(f"{outcome_type.lower()}_outcome")
        
        # Market condition tags
        if event.zone.swing_start.swing_type == 'high':
            tags.append("downtrend_context")
        else:
            tags.append("uptrend_context")
        
        # Volume tags
        if event.volume_in_zone and event.volume_in_zone > data['volume'].mean() * 1.5:
            tags.append("high_volume")
        elif event.volume_in_zone and event.volume_in_zone < data['volume'].mean() * 0.5:
            tags.append("low_volume")
        
        # Wick rejection tags
        if event.wick_rejection:
            tags.append("wick_rejection")
        
        # Retest tags
        if event.retest_count > 0:
            tags.append(f"retest_{event.retest_count}")
        
        # Strength tags
        if event.zone.strength > 0.7:
            tags.append("strong_zone")
        elif event.zone.strength < 0.3:
            tags.append("weak_zone")
        
        return tags
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def export_labels_to_dataframe(self, labels: List[OutcomeLabel]) -> pd.DataFrame:
        """
        Export labels to a pandas DataFrame.
        
        Args:
            labels: List of OutcomeLabel objects
            
        Returns:
            DataFrame with label data
        """
        if not labels:
            return pd.DataFrame()
        
        label_data = []
        for label in labels:
            row = {
                'event_id': label.event_id,
                'outcome_type': label.outcome_type,
                'confidence': label.confidence,
                'target_price': label.target_price,
                'target_zone_lower': label.target_zone[0] if label.target_zone else None,
                'target_zone_upper': label.target_zone[1] if label.target_zone else None,
                'stop_loss_level': label.stop_loss_level,
                'duration_bars': label.duration_bars,
                'max_adverse_excursion': label.max_adverse_excursion,
                'max_favorable_excursion': label.max_favorable_excursion,
                'risk_reward_ratio': label.risk_reward_ratio,
                'context_tags': ','.join(label.context_tags) if label.context_tags else ''
            }
            label_data.append(row)
        
        return pd.DataFrame(label_data)


def test_label_generator():
    """Test the label generator with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    from event_generator import EventGenerator
    
    print("Testing Label Generator...")
    
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
        print("No events generated for labeling")
        return
    
    # Generate labels
    label_generator = LabelGenerator(
        lookforward_window=30,
        reversal_threshold=2.0,
        continuation_threshold=1.0,
        atr_multiplier=1.0,
        min_confidence=0.5
    )
    
    labels = label_generator.generate_labels(sample_data, events, zones)
    print(f"Generated {len(labels)} labels")
    
    # Export to DataFrame
    labels_df = label_generator.export_labels_to_dataframe(labels)
    print(f"Labels DataFrame shape: {labels_df.shape}")
    
    if not labels_df.empty:
        print("\nLabel distribution:")
        print(labels_df['outcome_type'].value_counts())
        print(f"\nAverage confidence: {labels_df['confidence'].mean():.3f}")
        print(f"Average risk-reward ratio: {labels_df['risk_reward_ratio'].mean():.3f}")
        
        print("\nSample labels:")
        print(labels_df[['outcome_type', 'confidence', 'target_price', 'stop_loss_level']].head())
    
    print("Label generation test completed successfully!")


if __name__ == "__main__":
    test_label_generator()