import numpy as np
import pandas as pd
from typing import Dict, Optional, List


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels between a given high and low.
    Returns a dict with the fibonacci ratios and their corresponding price levels.
    """
    if high is None or low is None:
        return {}
    diff = high - low
    return {
        "high": high,
        "low": low,
        "23.6": low + diff * 0.236,
        "38.2": low + diff * 0.382,
        "50.0": low + diff * 0.5,
        "61.8": low + diff * 0.618,
        "78.6": low + diff * 0.786,
    }


def detect_ascending_triangle(
    df: pd.DataFrame, lookback: int = 40, tolerance: float = 0.002, min_points: int = 3
) -> Optional[Dict[str, any]]:
    """
    Detect an ascending triangle pattern in a price DataFrame.

    The pattern is identified by:
    - A sequence of highs clustering around a horizontal resistance level.
    - A sequence of lows trending upwards with a positive slope.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'high', 'low', 'close'. The index should be datetime-like.
        lookback (int): Number of recent bars to evaluate.
        tolerance (float): Relative tolerance for equal highs (e.g., 0.002 = 0.2%).
        min_points (int): Minimum number of highs touching the resistance to confirm the pattern.

    Returns:
        dict: A dictionary describing the pattern if found, otherwise None.
    """
    if df is None or df.empty or len(df) < lookback:
        return None
    recent = df.tail(lookback).copy()
    highs = recent['high'].values
    lows = recent['low'].values
    # Determine the potential resistance as the mode of the top highs within tolerance
    max_high = highs.max()
    # Count highs close to max_high within tolerance
    threshold = max_high * tolerance
    resistance_points = [h for h in highs if abs(h - max_high) <= threshold]
    if len(resistance_points) < min_points:
        return None
    # Check upward trend of lows
    x = np.arange(len(lows))
    # linear regression for lows
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, lows, 1)
    if slope <= 0:
        return None
    # Compute completion ratio: how far the close is between lowest low and resistance
    lowest_low = lows.min()
    last_close = recent['close'].iloc[-1]
    if max_high - lowest_low <= 0:
        completion = 0.0
    else:
        completion = (last_close - lowest_low) / (max_high - lowest_low)
    completion = float(np.clip(completion, 0.0, 1.0))
    # Target price: breakout above resistance by height of triangle
    height = max_high - lowest_low
    target = max_high + height
    # Stop loss at lowest low of pattern
    stop_loss = lowest_low
    # Confidence based on slope and number of resistance touches
    confidence = float(
        min(1.0, (len(resistance_points) / min_points) * (slope / (max_high * tolerance + 1e-9)))
    )
    return {
        "type": "ASCENDING_TRIANGLE",
        "completion": round(completion, 3),
        "target_price": round(target, 3),
        "stop_loss": round(stop_loss, 3),
        "confidence": round(confidence, 3),
    }


def recognize_patterns(df: pd.DataFrame) -> Dict[str, any]:
    """
    Recognize chart patterns and compute Fibonacci levels for the given price DataFrame.

    Currently implements:
        - Ascending Triangle detection
        - Fibonacci retracement levels between the highest high and lowest low in the recent window.

    Returns a dictionary with active patterns and fibonacci levels.
    """
    patterns: List[Dict[str, any]] = []
    # Detect ascending triangle on last 40 bars
    tri = detect_ascending_triangle(df, lookback=40, tolerance=0.002, min_points=3)
    if tri:
        patterns.append(tri)
    # Compute fibonacci levels on the entire DataFrame or last 100 bars
    lookback = min(len(df), 100)
    subset = df.tail(lookback)
    high = subset['high'].max() if 'high' in subset else None
    low = subset['low'].min() if 'low' in subset else None
    fib_levels = fibonacci_levels(high, low) if high is not None and low is not None else {}
    return {
        "active_patterns": patterns,
        "fibonacci_levels": fib_levels,
    }