"""
BTC Fibonacci Analysis Results - Show what the system found in real BTC data
"""

import pandas as pd
import numpy as np
import warnings
import sys

# Add current directory to path
sys.path.append('/workspace')

from swing_detector import SwingDetector
from fib_zones import FibZoneGenerator
from event_generator import EventGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def load_btc_data():
    """Load BTC data."""
    print("📊 Loading BTC 5-minute data...")
    
    # Read the CSV file
    df = pd.read_csv('/workspace/btc_5min.csv', sep='\t')
    
    # Clean data
    if df.iloc[0, 0] == '<DATE>':
        df = df.iloc[1:].reset_index(drop=True)
    
    # Rename columns
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    
    # Convert data types
    for col in ['open', 'high', 'low', 'close', 'vol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    
    # Select OHLCV
    ohlcv_data = df[['open', 'high', 'low', 'close', 'vol']].copy()
    ohlcv_data.columns = ['open', 'high', 'low', 'close', 'volume']
    ohlcv_data = ohlcv_data.dropna()
    
    print(f"   Data shape: {ohlcv_data.shape}")
    print(f"   Date range: {ohlcv_data.index[0]} to {ohlcv_data.index[-1]}")
    print(f"   Price range: ${ohlcv_data['close'].min():.2f} - ${ohlcv_data['close'].max():.2f}")
    print(f"   Total return: {(ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[0] - 1) * 100:.1f}%")
    
    return ohlcv_data

def main():
    """Run analysis and show results."""
    print("🚀 BTC Fibonacci ML System - Real Data Analysis")
    print("=" * 60)
    
    # Load data
    data = load_btc_data()
    
    # Use last 2000 bars for analysis
    data = data.tail(2000)
    print(f"   Using last {len(data)} bars for analysis")
    
    # Step 1: Detect swings
    print(f"\n📈 Step 1: Detecting Swings...")
    detector = SwingDetector(window=20, min_swing_strength=0.3, min_swing_size=0.002)
    swings = detector.detect_swings(data)
    print(f"   ✅ Detected {len(swings)} swings")
    
    if len(swings) > 0:
        print(f"   📊 Swing Statistics:")
        high_swings = [s for s in swings if s.swing_type == 'high']
        low_swings = [s for s in swings if s.swing_type == 'low']
        print(f"      High swings: {len(high_swings)}")
        print(f"      Low swings: {len(low_swings)}")
        
        # Show recent swings
        print(f"   📋 Recent Swings:")
        for i, swing in enumerate(swings[-5:]):
            print(f"      {i+1}. {swing.swing_type.upper()} at ${swing.price:.2f} (strength: {swing.strength:.3f})")
    
    # Step 2: Generate Fibonacci zones
    print(f"\n📊 Step 2: Generating Fibonacci Zones...")
    zone_generator = FibZoneGenerator(
        standard_ratios=[0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 1.0, 1.272, 1.414, 1.618],
        zone_width_factor=0.1,
        min_zone_size=0.002
    )
    zones = zone_generator.generate_all_zones(swings)
    zones = zone_generator.filter_zones_by_strength(zones, min_strength=0.2)
    print(f"   ✅ Generated {len(zones)} Fibonacci zones")
    
    if len(zones) > 0:
        # Zone statistics
        retracement_zones = [z for z in zones if z.zone_type == 'retracement']
        extension_zones = [z for z in zones if z.zone_type == 'extension']
        print(f"   📊 Zone Statistics:")
        print(f"      Retracement zones: {len(retracement_zones)}")
        print(f"      Extension zones: {len(extension_zones)}")
        
        # Show zone levels
        fib_levels = {}
        for zone in zones:
            level = zone.level
            if level not in fib_levels:
                fib_levels[level] = 0
            fib_levels[level] += 1
        
        print(f"   📋 Fibonacci Levels Found:")
        for level in sorted(fib_levels.keys()):
            print(f"      {level:.3f}: {fib_levels[level]} zones")
        
        # Show recent zones
        print(f"   📋 Recent Zones:")
        for i, zone in enumerate(zones[-5:]):
            print(f"      {i+1}. {zone.zone_type.upper()} {zone.level:.3f} at ${zone.price_range[0]:.2f}-${zone.price_range[1]:.2f}")
    
    # Step 3: Generate events
    print(f"\n🎯 Step 3: Generating Zone Touch Events...")
    event_generator = EventGenerator(
        min_touch_duration=1,
        max_event_duration=100,
        wick_rejection_threshold=0.3
    )
    events = event_generator.detect_zone_touches(data, zones)
    print(f"   ✅ Generated {len(events)} zone touch events")
    
    if len(events) > 0:
        # Event statistics
        events_with_exit = len([e for e in events if e.exit_time is not None])
        events_with_retests = len([e for e in events if e.retest_count > 0])
        events_with_wick_rejection = len([e for e in events if e.wick_rejection])
        
        print(f"   📊 Event Statistics:")
        print(f"      Events with exit: {events_with_exit}")
        print(f"      Events with retests: {events_with_retests}")
        print(f"      Events with wick rejection: {events_with_wick_rejection}")
        
        # Show recent events
        print(f"   📋 Recent Events:")
        for i, event in enumerate(events[-5:]):
            print(f"      {i+1}. Zone {event.zone.level:.3f} at ${event.entry_price:.2f} (duration: {event.duration_bars} bars)")
    
    # Step 4: Analyze Fibonacci effectiveness
    print(f"\n🔍 Step 4: Fibonacci Level Analysis...")
    if len(events) > 0:
        # Group events by Fibonacci level
        level_events = {}
        for event in events:
            level = event.zone.level
            if level not in level_events:
                level_events[level] = []
            level_events[level].append(event)
        
        print(f"   📊 Events by Fibonacci Level:")
        for level in sorted(level_events.keys()):
            events_at_level = level_events[level]
            avg_duration = np.mean([e.duration_bars for e in events_at_level if e.duration_bars is not None])
            print(f"      {level:.3f}: {len(events_at_level)} events (avg duration: {avg_duration:.1f} bars)")
    
    # Step 5: Show current market state
    print(f"\n📈 Step 5: Current Market State...")
    current_price = data['close'].iloc[-1]
    print(f"   Current BTC Price: ${current_price:.2f}")
    
    # Find zones near current price
    nearby_zones = []
    for zone in zones:
        lower, upper = zone.price_range
        if lower <= current_price <= upper:
            nearby_zones.append(zone)
    
    if nearby_zones:
        print(f"   🎯 Currently in {len(nearby_zones)} Fibonacci zones:")
        for zone in nearby_zones:
            print(f"      {zone.zone_type.upper()} {zone.level:.3f}: ${zone.price_range[0]:.2f}-${zone.price_range[1]:.2f}")
    else:
        print(f"   📊 No active Fibonacci zones at current price")
    
    # Summary
    print(f"\n📊 Analysis Summary:")
    print(f"   📈 Data: {len(data)} bars of BTC 5-minute data")
    print(f"   📊 Swings: {len(swings)} detected")
    print(f"   📊 Zones: {len(zones)} Fibonacci zones generated")
    print(f"   🎯 Events: {len(events)} zone touch events")
    print(f"   💰 Price Range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   📈 Total Return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.1f}%")
    
    print(f"\n🎉 BTC Fibonacci Analysis Results:")
    print(f"   ✅ The system successfully detected {len(swings)} market swings")
    print(f"   ✅ Generated {len(zones)} Fibonacci retracement/extension zones")
    print(f"   ✅ Found {len(events)} zone touch events for training")
    print(f"   ✅ Most active Fibonacci levels: {', '.join([f'{level:.3f}' for level in sorted(fib_levels.keys(), key=lambda x: fib_levels[x], reverse=True)[:5]])}")
    
    print(f"\n🚀 The Fibonacci ML system is ready to learn from this real BTC data!")
    print(f"   Next step: Train the ML models on these {len(events)} events")

if __name__ == "__main__":
    main()