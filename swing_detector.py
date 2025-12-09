"""
Swing Detection Module

Detects market swings and pivots using rolling window analysis.
Identifies significant highs and lows for Fibonacci zone generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class Swing:
    """Represents a market swing with metadata."""
    swing_type: str  # 'high' or 'low'
    price: float
    index: int
    timestamp: pd.Timestamp
    strength: float  # Relative strength of the swing
    context: Dict  # Additional context (volume, ATR, etc.)


class SwingDetector:
    """
    Detects market swings using rolling window pivot analysis.
    
    Uses a rolling window to identify local highs and lows, then filters
    them based on strength and significance criteria.
    """
    
    def __init__(self, 
                 window: int = 20,
                 min_swing_strength: float = 0.5,
                 min_swing_size: float = 0.001):
        """
        Initialize swing detector.
        
        Args:
            window: Rolling window size for pivot detection
            min_swing_strength: Minimum relative strength for swing validation
            min_swing_size: Minimum price movement as fraction of price
        """
        self.window = window
        self.min_swing_strength = min_swing_strength
        self.min_swing_size = min_swing_size
        
    def detect_pivots(self, 
                     data: pd.DataFrame,
                     high_col: str = 'high',
                     low_col: str = 'low',
                     close_col: str = 'close',
                     volume_col: str = 'volume') -> pd.DataFrame:
        """
        Detect pivot highs and lows using rolling window analysis.
        
        Args:
            data: OHLCV DataFrame with datetime index
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            volume_col: Column name for volume
            
        Returns:
            DataFrame with pivot information
        """
        if len(data) < self.window * 2:
            warnings.warn(f"Data length {len(data)} is less than required {self.window * 2}")
            return pd.DataFrame()
            
        # Calculate rolling maximum and minimum
        rolling_high = data[high_col].rolling(window=self.window, center=True)
        rolling_low = data[low_col].rolling(window=self.window, center=True)
        
        # Identify pivot highs and lows
        pivot_highs = (data[high_col] == rolling_high.max()) & (data[high_col] > 0)
        pivot_lows = (data[low_col] == rolling_low.min()) & (data[low_col] > 0)
        
        # Create pivot DataFrame
        pivots = pd.DataFrame(index=data.index)
        pivots['pivot_high'] = pivot_highs
        pivots['pivot_low'] = pivot_lows
        pivots['high_price'] = data[high_col]
        pivots['low_price'] = data[low_col]
        pivots['close_price'] = data[close_col]
        pivots['volume'] = data[volume_col]
        
        return pivots
    
    def calculate_swing_strength(self, 
                               data: pd.DataFrame,
                               pivot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strength metrics for each pivot.
        
        Args:
            data: Original OHLCV data
            pivot_data: Pivot detection results
            
        Returns:
            DataFrame with strength metrics
        """
        pivot_data = pivot_data.copy()
        
        # Calculate ATR for volatility normalization
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=14).mean()
        
        pivot_data['atr'] = atr
        
        # Calculate swing strength for highs
        high_strength = np.zeros(len(pivot_data))
        for i in range(len(pivot_data)):
            if pivot_data.iloc[i]['pivot_high']:
                # Look left and right for comparison
                left_start = max(0, i - self.window)
                right_end = min(len(pivot_data), i + self.window + 1)
                
                left_highs = data['high'].iloc[left_start:i]
                right_highs = data['high'].iloc[i+1:right_end]
                
                if len(left_highs) > 0 and len(right_highs) > 0:
                    current_high = data['high'].iloc[i]
                    avg_surrounding = (left_highs.mean() + right_highs.mean()) / 2
                    strength = (current_high - avg_surrounding) / atr.iloc[i] if atr.iloc[i] > 0 else 0
                    high_strength[i] = strength
        
        # Calculate swing strength for lows
        low_strength = np.zeros(len(pivot_data))
        for i in range(len(pivot_data)):
            if pivot_data.iloc[i]['pivot_low']:
                # Look left and right for comparison
                left_start = max(0, i - self.window)
                right_end = min(len(pivot_data), i + self.window + 1)
                
                left_lows = data['low'].iloc[left_start:i]
                right_lows = data['low'].iloc[i+1:right_end]
                
                if len(left_lows) > 0 and len(right_lows) > 0:
                    current_low = data['low'].iloc[i]
                    avg_surrounding = (left_lows.mean() + right_lows.mean()) / 2
                    strength = (avg_surrounding - current_low) / atr.iloc[i] if atr.iloc[i] > 0 else 0
                    low_strength[i] = strength
        
        pivot_data['high_strength'] = high_strength
        pivot_data['low_strength'] = low_strength
        
        return pivot_data
    
    def filter_swings(self, 
                     pivot_data: pd.DataFrame,
                     data: pd.DataFrame) -> List[Swing]:
        """
        Filter pivots to identify significant swings.
        
        Args:
            pivot_data: Pivot data with strength metrics
            data: Original OHLCV data
            
        Returns:
            List of Swing objects
        """
        swings = []
        
        for i, row in pivot_data.iterrows():
            # Check for significant high
            if (row['pivot_high'] and 
                row['high_strength'] >= self.min_swing_strength and
                row['high_price'] > 0):
                
                # Calculate swing size
                swing_size = row['high_price'] * self.min_swing_size
                
                # Check if this is a significant swing
                if row['high_strength'] * row['atr'] >= swing_size:
                    swing = Swing(
                        swing_type='high',
                        price=row['high_price'],
                        index=pivot_data.index.get_loc(i),
                        timestamp=i,
                        strength=row['high_strength'],
                        context={
                            'volume': row['volume'],
                            'atr': row['atr'],
                            'close': row['close_price']
                        }
                    )
                    swings.append(swing)
            
            # Check for significant low
            if (row['pivot_low'] and 
                row['low_strength'] >= self.min_swing_strength and
                row['low_price'] > 0):
                
                # Calculate swing size
                swing_size = row['low_price'] * self.min_swing_size
                
                # Check if this is a significant swing
                if row['low_strength'] * row['atr'] >= swing_size:
                    swing = Swing(
                        swing_type='low',
                        price=row['low_price'],
                        index=pivot_data.index.get_loc(i),
                        timestamp=i,
                        strength=row['low_strength'],
                        context={
                            'volume': row['volume'],
                            'atr': row['atr'],
                            'close': row['close_price']
                        }
                    )
                    swings.append(swing)
        
        # Sort swings by timestamp
        swings.sort(key=lambda x: x.timestamp)
        
        return swings
    
    def detect_swings(self, 
                     data: pd.DataFrame,
                     high_col: str = 'high',
                     low_col: str = 'low',
                     close_col: str = 'close',
                     volume_col: str = 'volume') -> List[Swing]:
        """
        Complete swing detection pipeline.
        
        Args:
            data: OHLCV DataFrame with datetime index
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            volume_col: str = 'volume'
            
        Returns:
            List of detected Swing objects
        """
        # Step 1: Detect pivots
        pivot_data = self.detect_pivots(data, high_col, low_col, close_col, volume_col)
        
        if pivot_data.empty:
            return []
        
        # Step 2: Calculate swing strength
        pivot_data = self.calculate_swing_strength(data, pivot_data)
        
        # Step 3: Filter significant swings
        swings = self.filter_swings(pivot_data, data)
        
        return swings
    
    def get_swing_sequences(self, swings: List[Swing]) -> List[List[Swing]]:
        """
        Group swings into sequences (alternating highs and lows).
        
        Args:
            swings: List of Swing objects
            
        Returns:
            List of swing sequences
        """
        if not swings:
            return []
        
        sequences = []
        current_sequence = [swings[0]]
        
        for i in range(1, len(swings)):
            current_swing = swings[i]
            last_swing = current_sequence[-1]
            
            # Check if swing type alternates
            if current_swing.swing_type != last_swing.swing_type:
                current_sequence.append(current_swing)
            else:
                # Start new sequence
                sequences.append(current_sequence)
                current_sequence = [current_swing]
        
        # Add final sequence
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    def validate_swings(self, swings: List[Swing], data: pd.DataFrame) -> List[Swing]:
        """
        Validate swings for quality and remove duplicates.
        
        Args:
            swings: List of Swing objects
            data: Original OHLCV data
            
        Returns:
            Validated list of Swing objects
        """
        if not swings:
            return []
        
        validated_swings = []
        seen_indices = set()
        
        for swing in swings:
            # Check for duplicate indices
            if swing.index in seen_indices:
                continue
            
            # Validate price is within data range
            if (swing.index < len(data) and 
                swing.index >= 0 and
                data.iloc[swing.index]['high'] >= swing.price >= data.iloc[swing.index]['low']):
                
                validated_swings.append(swing)
                seen_indices.add(swing.index)
        
        return validated_swings


def create_sample_data(length: int = 1000, 
                      start_price: float = 100.0,
                      volatility: float = 0.02) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing.
    
    Args:
        length: Number of bars to generate
        start_price: Starting price
        volatility: Price volatility
        
    Returns:
        Sample OHLCV DataFrame
    """
    np.random.seed(42)
    
    # Generate random walk with trend
    returns = np.random.normal(0, volatility, length)
    trend = np.linspace(0, 0.1, length)  # Slight upward trend
    prices = start_price * np.exp(np.cumsum(returns + trend))
    
    # Create OHLCV data
    data = pd.DataFrame(index=pd.date_range('2023-01-01', periods=length, freq='1H'))
    data['close'] = prices
    
    # Generate realistic OHLC from close prices
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, volatility/2, length)))
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, volatility/2, length)))
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['volume'] = np.random.lognormal(10, 1, length)
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data


if __name__ == "__main__":
    # Test the swing detector
    print("Testing Swing Detector...")
    
    # Create sample data
    sample_data = create_sample_data(500, 100.0, 0.02)
    print(f"Created sample data with {len(sample_data)} bars")
    
    # Initialize detector
    detector = SwingDetector(window=10, min_swing_strength=0.3, min_swing_size=0.005)
    
    # Detect swings
    swings = detector.detect_swings(sample_data)
    print(f"Detected {len(swings)} swings")
    
    # Show swing details
    for i, swing in enumerate(swings[:5]):  # Show first 5 swings
        print(f"Swing {i+1}: {swing.swing_type} at {swing.price:.2f} "
              f"(strength: {swing.strength:.3f}, time: {swing.timestamp})")
    
    # Get swing sequences
    sequences = detector.get_swing_sequences(swings)
    print(f"Found {len(sequences)} swing sequences")
    
    print("Swing detection test completed successfully!")