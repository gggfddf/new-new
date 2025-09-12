"""
Fibonacci Zone Generation Module

Generates Fibonacci retracement and extension zones from swing data.
Creates zones as price ranges rather than single levels for better ML learning.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from swing_detector import Swing
import warnings


@dataclass
class FibZone:
    """Represents a Fibonacci zone with metadata."""
    zone_type: str  # 'retracement' or 'extension'
    level: float  # Fibonacci ratio (e.g., 0.618)
    price_range: Tuple[float, float]  # (lower, upper) price bounds
    swing_start: Swing  # Starting swing
    swing_end: Swing  # Ending swing
    zone_width: float  # Width of the zone
    strength: float  # Zone strength based on swing significance
    context: Dict  # Additional context


class FibZoneGenerator:
    """
    Generates Fibonacci retracement and extension zones from swing data.
    
    Creates zones as price ranges rather than single levels to account for
    market noise and improve ML model learning.
    """
    
    def __init__(self, 
                 standard_ratios: List[float] = None,
                 zone_width_factor: float = 0.1,
                 min_zone_size: float = 0.001):
        """
        Initialize Fibonacci zone generator.
        
        Args:
            standard_ratios: List of Fibonacci ratios to use
            zone_width_factor: Factor to determine zone width (as fraction of swing size)
            min_zone_size: Minimum zone size as fraction of price
        """
        if standard_ratios is None:
            self.standard_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.414, 1.618]
        else:
            self.standard_ratios = standard_ratios
            
        self.zone_width_factor = zone_width_factor
        self.min_zone_size = min_zone_size
    
    def calculate_retracement_zones(self, 
                                  swing_start: Swing, 
                                  swing_end: Swing) -> List[FibZone]:
        """
        Calculate Fibonacci retracement zones between two swings.
        
        Args:
            swing_start: Starting swing (high for downtrend, low for uptrend)
            swing_end: Ending swing (low for downtrend, high for uptrend)
            
        Returns:
            List of FibZone objects for retracements
        """
        zones = []
        
        # Determine trend direction
        if swing_start.swing_type == 'high' and swing_end.swing_type == 'low':
            # Downtrend - retrace from low back toward high
            trend_direction = 'downtrend'
            start_price = swing_end.price  # Low price
            end_price = swing_start.price  # High price
        elif swing_start.swing_type == 'low' and swing_end.swing_type == 'high':
            # Uptrend - retrace from high back toward low
            trend_direction = 'uptrend'
            start_price = swing_end.price  # High price
            end_price = swing_start.price  # Low price
        else:
            # Invalid swing pair
            return zones
        
        # Calculate swing size
        swing_size = abs(end_price - start_price)
        
        # Generate retracement zones
        for ratio in self.standard_ratios:
            if ratio >= 1.0:  # Skip extension ratios for retracements
                continue
                
            # Calculate retracement price
            retracement_price = start_price + (end_price - start_price) * ratio
            
            # Calculate zone width
            zone_width = max(swing_size * self.zone_width_factor, 
                           retracement_price * self.min_zone_size)
            
            # Create zone bounds
            lower_bound = retracement_price - zone_width / 2
            upper_bound = retracement_price + zone_width / 2
            
            # Calculate zone strength based on swing significance
            zone_strength = (swing_start.strength + swing_end.strength) / 2
            
            zone = FibZone(
                zone_type='retracement',
                level=ratio,
                price_range=(lower_bound, upper_bound),
                swing_start=swing_start,
                swing_end=swing_end,
                zone_width=zone_width,
                strength=zone_strength,
                context={
                    'trend_direction': trend_direction,
                    'swing_size': swing_size,
                    'swing_duration': abs(swing_end.index - swing_start.index)
                }
            )
            zones.append(zone)
        
        return zones
    
    def calculate_extension_zones(self, 
                                swing_start: Swing, 
                                swing_end: Swing) -> List[FibZone]:
        """
        Calculate Fibonacci extension zones beyond the swing.
        
        Args:
            swing_start: Starting swing
            swing_end: Ending swing
            
        Returns:
            List of FibZone objects for extensions
        """
        zones = []
        
        # Determine trend direction and extension direction
        if swing_start.swing_type == 'high' and swing_end.swing_type == 'low':
            # Downtrend - extend beyond the low
            trend_direction = 'downtrend'
            start_price = swing_start.price  # High price
            end_price = swing_end.price  # Low price
            extension_direction = -1  # Extend downward
        elif swing_start.swing_type == 'low' and swing_end.swing_type == 'high':
            # Uptrend - extend beyond the high
            trend_direction = 'uptrend'
            start_price = swing_start.price  # Low price
            end_price = swing_end.price  # High price
            extension_direction = 1  # Extend upward
        else:
            # Invalid swing pair
            return zones
        
        # Calculate swing size
        swing_size = abs(end_price - start_price)
        
        # Generate extension zones
        for ratio in self.standard_ratios:
            if ratio <= 1.0:  # Skip retracement ratios for extensions
                continue
                
            # Calculate extension price
            extension_price = end_price + (end_price - start_price) * (ratio - 1.0) * extension_direction
            
            # Calculate zone width
            zone_width = max(swing_size * self.zone_width_factor, 
                           extension_price * self.min_zone_size)
            
            # Create zone bounds
            lower_bound = extension_price - zone_width / 2
            upper_bound = extension_price + zone_width / 2
            
            # Calculate zone strength based on swing significance
            zone_strength = (swing_start.strength + swing_end.strength) / 2
            
            zone = FibZone(
                zone_type='extension',
                level=ratio,
                price_range=(lower_bound, upper_bound),
                swing_start=swing_start,
                swing_end=swing_end,
                zone_width=zone_width,
                strength=zone_strength,
                context={
                    'trend_direction': trend_direction,
                    'swing_size': swing_size,
                    'swing_duration': abs(swing_end.index - swing_start.index),
                    'extension_direction': extension_direction
                }
            )
            zones.append(zone)
        
        return zones
    
    def generate_zones_for_swing_pair(self, 
                                    swing_start: Swing, 
                                    swing_end: Swing) -> List[FibZone]:
        """
        Generate all Fibonacci zones for a swing pair.
        
        Args:
            swing_start: Starting swing
            swing_end: Ending swing
            
        Returns:
            List of all FibZone objects
        """
        zones = []
        
        # Generate retracement zones
        retracement_zones = self.calculate_retracement_zones(swing_start, swing_end)
        zones.extend(retracement_zones)
        
        # Generate extension zones
        extension_zones = self.calculate_extension_zones(swing_start, swing_end)
        zones.extend(extension_zones)
        
        return zones
    
    def generate_all_zones(self, swings: List[Swing]) -> List[FibZone]:
        """
        Generate Fibonacci zones for all valid swing pairs.
        
        Args:
            swings: List of Swing objects
            
        Returns:
            List of all FibZone objects
        """
        if len(swings) < 2:
            return []
        
        all_zones = []
        
        # Generate zones for consecutive swing pairs
        for i in range(len(swings) - 1):
            swing_start = swings[i]
            swing_end = swings[i + 1]
            
            # Only generate zones for alternating swing types
            if swing_start.swing_type != swing_end.swing_type:
                zones = self.generate_zones_for_swing_pair(swing_start, swing_end)
                all_zones.extend(zones)
        
        return all_zones
    
    def filter_zones_by_strength(self, 
                               zones: List[FibZone], 
                               min_strength: float = 0.3) -> List[FibZone]:
        """
        Filter zones by minimum strength threshold.
        
        Args:
            zones: List of FibZone objects
            min_strength: Minimum zone strength
            
        Returns:
            Filtered list of FibZone objects
        """
        return [zone for zone in zones if zone.strength >= min_strength]
    
    def filter_zones_by_price_range(self, 
                                  zones: List[FibZone], 
                                  data: pd.DataFrame) -> List[FibZone]:
        """
        Filter zones that are within the price range of the data.
        
        Args:
            zones: List of FibZone objects
            data: OHLCV DataFrame
            
        Returns:
            Filtered list of FibZone objects
        """
        if data.empty:
            return zones
        
        min_price = data[['high', 'low']].min().min()
        max_price = data[['high', 'low']].max().max()
        
        filtered_zones = []
        for zone in zones:
            zone_min, zone_max = zone.price_range
            
            # Keep zone if it overlaps with data price range
            if not (zone_max < min_price or zone_min > max_price):
                filtered_zones.append(zone)
        
        return filtered_zones
    
    def merge_overlapping_zones(self, zones: List[FibZone]) -> List[FibZone]:
        """
        Merge overlapping zones of the same type and level.
        
        Args:
            zones: List of FibZone objects
            
        Returns:
            List of merged FibZone objects
        """
        if not zones:
            return []
        
        # Group zones by type and level
        zone_groups = {}
        for zone in zones:
            key = (zone.zone_type, zone.level)
            if key not in zone_groups:
                zone_groups[key] = []
            zone_groups[key].append(zone)
        
        merged_zones = []
        
        for (zone_type, level), zone_list in zone_groups.items():
            if len(zone_list) == 1:
                merged_zones.append(zone_list[0])
            else:
                # Merge overlapping zones
                zone_list.sort(key=lambda z: z.price_range[0])  # Sort by lower bound
                
                merged = zone_list[0]
                for zone in zone_list[1:]:
                    # Check if zones overlap
                    if zone.price_range[0] <= merged.price_range[1]:
                        # Merge zones
                        new_lower = min(merged.price_range[0], zone.price_range[0])
                        new_upper = max(merged.price_range[1], zone.price_range[1])
                        merged.price_range = (new_lower, new_upper)
                        merged.zone_width = new_upper - new_lower
                        merged.strength = max(merged.strength, zone.strength)
                    else:
                        # No overlap, add previous merged zone and start new one
                        merged_zones.append(merged)
                        merged = zone
                
                merged_zones.append(merged)
        
        return merged_zones
    
    def get_zones_at_price(self, 
                          zones: List[FibZone], 
                          price: float) -> List[FibZone]:
        """
        Get all zones that contain a given price.
        
        Args:
            zones: List of FibZone objects
            price: Price to check
            
        Returns:
            List of zones containing the price
        """
        matching_zones = []
        
        for zone in zones:
            lower, upper = zone.price_range
            if lower <= price <= upper:
                matching_zones.append(zone)
        
        return matching_zones
    
    def get_zone_statistics(self, zones: List[FibZone]) -> Dict:
        """
        Calculate statistics for a list of zones.
        
        Args:
            zones: List of FibZone objects
            
        Returns:
            Dictionary with zone statistics
        """
        if not zones:
            return {}
        
        retracement_zones = [z for z in zones if z.zone_type == 'retracement']
        extension_zones = [z for z in zones if z.zone_type == 'extension']
        
        stats = {
            'total_zones': len(zones),
            'retracement_zones': len(retracement_zones),
            'extension_zones': len(extension_zones),
            'avg_zone_width': np.mean([z.zone_width for z in zones]),
            'avg_zone_strength': np.mean([z.strength for z in zones]),
            'zone_levels': list(set([z.level for z in zones])),
            'zone_types': list(set([z.zone_type for z in zones]))
        }
        
        return stats


def test_fib_zones():
    """Test the Fibonacci zone generator with sample data."""
    from swing_detector import create_sample_data, SwingDetector
    
    print("Testing Fibonacci Zone Generator...")
    
    # Create sample data
    sample_data = create_sample_data(200, 100.0, 0.02)
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
    
    all_zones = zone_generator.generate_all_zones(swings)
    print(f"Generated {len(all_zones)} total zones")
    
    # Filter and merge zones
    filtered_zones = zone_generator.filter_zones_by_strength(all_zones, min_strength=0.1)
    print(f"Filtered to {len(filtered_zones)} zones by strength")
    
    merged_zones = zone_generator.merge_overlapping_zones(filtered_zones)
    print(f"Merged to {len(merged_zones)} zones")
    
    # Show zone statistics
    stats = zone_generator.get_zone_statistics(merged_zones)
    print(f"Zone statistics: {stats}")
    
    # Show sample zones
    print("\nSample zones:")
    for i, zone in enumerate(merged_zones[:5]):
        print(f"Zone {i+1}: {zone.zone_type} {zone.level} "
              f"range [{zone.price_range[0]:.2f}, {zone.price_range[1]:.2f}] "
              f"strength {zone.strength:.3f}")
    
    print("Fibonacci zone generation test completed successfully!")


if __name__ == "__main__":
    test_fib_zones()