"""
Event Generation Module

Detects when price touches Fibonacci zones and generates events for ML training.
Tracks zone entry, exit, and interaction patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from fib_zones import FibZone
from swing_detector import Swing
import warnings


@dataclass
class ZoneEvent:
    """Represents a zone touch event with metadata."""
    event_id: str
    zone: FibZone
    entry_time: pd.Timestamp
    entry_price: float
    entry_index: int
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_index: Optional[int] = None
    duration_bars: Optional[int] = None
    max_price_in_zone: Optional[float] = None
    min_price_in_zone: Optional[float] = None
    volume_in_zone: Optional[float] = None
    wick_rejection: Optional[bool] = None
    retest_count: int = 0
    context: Dict = None


class EventGenerator:
    """
    Generates zone touch events from OHLCV data and Fibonacci zones.
    
    Tracks when price enters, stays in, and exits Fibonacci zones,
    creating events for ML model training.
    """
    
    def __init__(self, 
                 min_touch_duration: int = 1,
                 max_event_duration: int = 100,
                 wick_rejection_threshold: float = 0.3):
        """
        Initialize event generator.
        
        Args:
            min_touch_duration: Minimum bars price must stay in zone
            max_event_duration: Maximum bars to track an event
            wick_rejection_threshold: Threshold for wick rejection detection
        """
        self.min_touch_duration = min_touch_duration
        self.max_event_duration = max_event_duration
        self.wick_rejection_threshold = wick_rejection_threshold
    
    def detect_zone_touches(self, 
                           data: pd.DataFrame,
                           zones: List[FibZone]) -> List[ZoneEvent]:
        """
        Detect all zone touch events in the data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            zones: List of FibZone objects
            
        Returns:
            List of ZoneEvent objects
        """
        if data.empty or not zones:
            return []
        
        events = []
        active_events = {}  # zone_id -> ZoneEvent
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            high_price = row['high']
            low_price = row['low']
            volume = row['volume']
            
            # Check each zone for touches
            for zone in zones:
                zone_id = f"{zone.zone_type}_{zone.level:.3f}_{zone.swing_start.index % 1000}_{zone.swing_end.index % 1000}"
                lower_bound, upper_bound = zone.price_range
                
                # Check if price is in zone
                price_in_zone = lower_bound <= current_price <= upper_bound
                high_in_zone = lower_bound <= high_price <= upper_bound
                low_in_zone = lower_bound <= low_price <= upper_bound
                
                # Check for zone entry
                if zone_id not in active_events and price_in_zone:
                    # New zone entry
                    event = ZoneEvent(
                        event_id=f"{zone_id}_{i}",
                        zone=zone,
                        entry_time=timestamp,
                        entry_price=current_price,
                        entry_index=i,
                        max_price_in_zone=high_price,
                        min_price_in_zone=low_price,
                        volume_in_zone=volume,
                        context={}
                    )
                    
                    # Check for wick rejection
                    event.wick_rejection = self._detect_wick_rejection(
                        data, i, zone, high_price, low_price
                    )
                    
                    active_events[zone_id] = event
                    events.append(event)
                
                # Update active event
                elif zone_id in active_events:
                    event = active_events[zone_id]
                    
                    # Update price range in zone
                    if high_price > event.max_price_in_zone:
                        event.max_price_in_zone = high_price
                    if low_price < event.min_price_in_zone:
                        event.min_price_in_zone = low_price
                    
                    # Update volume
                    event.volume_in_zone += volume
                    
                    # Check for retest (exit and re-enter)
                    if not price_in_zone:
                        # Price exited zone
                        if event.exit_time is None:
                            event.exit_time = timestamp
                            event.exit_price = current_price
                            event.exit_index = i
                            event.duration_bars = i - event.entry_index
                        
                        # Check if this is a retest (re-entry within reasonable time)
                        if (i - event.exit_index) <= 10:  # Within 10 bars
                            if price_in_zone:
                                event.retest_count += 1
                                event.exit_time = None
                                event.exit_price = None
                                event.exit_index = None
                                event.duration_bars = None
                    
                    # Check for event completion
                    if (event.exit_time is not None and 
                        (i - event.exit_index) > 5):  # 5 bars after exit
                        del active_events[zone_id]
                    
                    # Check for max duration
                    if (i - event.entry_index) > self.max_event_duration:
                        if event.exit_time is None:
                            event.exit_time = timestamp
                            event.exit_price = current_price
                            event.exit_index = i
                            event.duration_bars = i - event.entry_index
                        del active_events[zone_id]
        
        # Clean up any remaining active events
        for event in active_events.values():
            if event.exit_time is None:
                event.exit_time = data.index[-1]
                event.exit_price = data['close'].iloc[-1]
                event.exit_index = len(data) - 1
                event.duration_bars = len(data) - 1 - event.entry_index
            events.append(event)
        
        # Filter events by minimum duration
        filtered_events = [
            event for event in events 
            if event.duration_bars is None or event.duration_bars >= self.min_touch_duration
        ]
        
        return filtered_events
    
    def _detect_wick_rejection(self, 
                              data: pd.DataFrame,
                              index: int,
                              zone: FibZone,
                              high_price: float,
                              low_price: float) -> bool:
        """
        Detect if there's a wick rejection at zone entry.
        
        Args:
            data: OHLCV DataFrame
            index: Current bar index
            zone: FibZone object
            high_price: Current bar high
            low_price: Current bar low
            
        Returns:
            True if wick rejection detected
        """
        lower_bound, upper_bound = zone.price_range
        
        # Check for upper wick rejection (price touched zone top and rejected)
        if high_price >= upper_bound * (1 - self.wick_rejection_threshold):
            # Check if close is significantly below the high
            close_price = data['close'].iloc[index]
            wick_size = high_price - close_price
            body_size = close_price - low_price
            
            if wick_size > body_size * 2:  # Long upper wick
                return True
        
        # Check for lower wick rejection (price touched zone bottom and rejected)
        if low_price <= lower_bound * (1 + self.wick_rejection_threshold):
            # Check if close is significantly above the low
            close_price = data['close'].iloc[index]
            wick_size = close_price - low_price
            body_size = high_price - close_price
            
            if wick_size > body_size * 2:  # Long lower wick
                return True
        
        return False
    
    def calculate_event_metrics(self, events: List[ZoneEvent]) -> Dict:
        """
        Calculate metrics for a list of events.
        
        Args:
            events: List of ZoneEvent objects
            
        Returns:
            Dictionary with event metrics
        """
        if not events:
            return {}
        
        # Basic metrics
        total_events = len(events)
        events_with_exit = len([e for e in events if e.exit_time is not None])
        events_with_retests = len([e for e in events if e.retest_count > 0])
        events_with_wick_rejection = len([e for e in events if e.wick_rejection])
        
        # Duration metrics
        durations = [e.duration_bars for e in events if e.duration_bars is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        # Volume metrics
        volumes = [e.volume_in_zone for e in events if e.volume_in_zone is not None]
        avg_volume = np.mean(volumes) if volumes else 0
        
        # Zone type breakdown
        retracement_events = len([e for e in events if e.zone.zone_type == 'retracement'])
        extension_events = len([e for e in events if e.zone.zone_type == 'extension'])
        
        metrics = {
            'total_events': total_events,
            'events_with_exit': events_with_exit,
            'events_with_retests': events_with_retests,
            'events_with_wick_rejection': events_with_wick_rejection,
            'avg_duration_bars': avg_duration,
            'avg_volume_in_zone': avg_volume,
            'retracement_events': retracement_events,
            'extension_events': extension_events,
            'retest_rate': events_with_retests / total_events if total_events > 0 else 0,
            'wick_rejection_rate': events_with_wick_rejection / total_events if total_events > 0 else 0
        }
        
        return metrics
    
    def filter_events_by_quality(self, 
                                events: List[ZoneEvent],
                                min_volume: float = 0,
                                min_duration: int = 1,
                                max_duration: int = 100) -> List[ZoneEvent]:
        """
        Filter events by quality criteria.
        
        Args:
            events: List of ZoneEvent objects
            min_volume: Minimum volume threshold
            min_duration: Minimum duration threshold
            max_duration: Maximum duration threshold
            
        Returns:
            Filtered list of ZoneEvent objects
        """
        filtered_events = []
        
        for event in events:
            # Check volume threshold
            if event.volume_in_zone is not None and event.volume_in_zone < min_volume:
                continue
            
            # Check duration threshold
            if event.duration_bars is not None:
                if event.duration_bars < min_duration or event.duration_bars > max_duration:
                    continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def get_events_by_zone_type(self, 
                               events: List[ZoneEvent],
                               zone_type: str) -> List[ZoneEvent]:
        """
        Get events filtered by zone type.
        
        Args:
            events: List of ZoneEvent objects
            zone_type: 'retracement' or 'extension'
            
        Returns:
            Filtered list of ZoneEvent objects
        """
        return [event for event in events if event.zone.zone_type == zone_type]
    
    def get_events_by_level(self, 
                           events: List[ZoneEvent],
                           level: float,
                           tolerance: float = 0.01) -> List[ZoneEvent]:
        """
        Get events filtered by Fibonacci level.
        
        Args:
            events: List of ZoneEvent objects
            level: Fibonacci level to filter by
            tolerance: Tolerance for level matching
            
        Returns:
            Filtered list of ZoneEvent objects
        """
        return [
            event for event in events 
            if abs(event.zone.level - level) <= tolerance
        ]
    
    def export_events_to_dataframe(self, events: List[ZoneEvent]) -> pd.DataFrame:
        """
        Export events to a pandas DataFrame for analysis.
        
        Args:
            events: List of ZoneEvent objects
            
        Returns:
            DataFrame with event data
        """
        if not events:
            return pd.DataFrame()
        
        event_data = []
        
        for event in events:
            row = {
                'event_id': event.event_id,
                'zone_type': event.zone.zone_type,
                'zone_level': event.zone.level,
                'zone_strength': event.zone.strength,
                'entry_time': event.entry_time,
                'entry_price': event.entry_price,
                'entry_index': event.entry_index,
                'exit_time': event.exit_time,
                'exit_price': event.exit_price,
                'exit_index': event.exit_index,
                'duration_bars': event.duration_bars,
                'max_price_in_zone': event.max_price_in_zone,
                'min_price_in_zone': event.min_price_in_zone,
                'volume_in_zone': event.volume_in_zone,
                'wick_rejection': event.wick_rejection,
                'retest_count': event.retest_count,
                'zone_lower_bound': event.zone.price_range[0],
                'zone_upper_bound': event.zone.price_range[1],
                'zone_width': event.zone.zone_width,
                'swing_start_index': event.zone.swing_start.index,
                'swing_end_index': event.zone.swing_end.index,
                'swing_start_type': event.zone.swing_start.swing_type,
                'swing_end_type': event.zone.swing_end.swing_type
            }
            event_data.append(row)
        
        return pd.DataFrame(event_data)


def test_event_generator():
    """Test the event generator with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    from fib_zones import FibZoneGenerator
    
    print("Testing Event Generator...")
    
    # Create sample data
    sample_data = create_sample_data(300, 100.0, 0.02)
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
    
    # Calculate metrics
    metrics = event_generator.calculate_event_metrics(events)
    print(f"Event metrics: {metrics}")
    
    # Filter events
    filtered_events = event_generator.filter_events_by_quality(
        events, min_volume=0, min_duration=1, max_duration=50
    )
    print(f"Filtered to {len(filtered_events)} events")
    
    # Export to DataFrame
    events_df = event_generator.export_events_to_dataframe(filtered_events)
    print(f"Events DataFrame shape: {events_df.shape}")
    
    if not events_df.empty:
        print("\nSample events:")
        print(events_df[['zone_type', 'zone_level', 'entry_price', 'duration_bars', 'wick_rejection']].head())
    
    print("Event generation test completed successfully!")


if __name__ == "__main__":
    test_event_generator()