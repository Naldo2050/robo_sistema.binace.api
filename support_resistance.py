import numpy as np
import pandas as pd

def calculate_pivot_points(high: float, low: float, close: float) -> dict:
    """
    Calculate classic pivot points and support/resistance levels for a given period.

    Returns a dictionary with pivot (P), R1, R2, S1, S2.
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {
        "pivot": pivot,
        "r1": r1,
        "s1": s1,
        "r2": r2,
        "s2": s2,
    }

def _period_high_low_close(df: pd.DataFrame, period: str) -> tuple[float, float, float]:
    """
    Helper to extract high, low, close for a specified period ('D', 'W', 'M').
    Expects df with datetime index and columns high, low, close.
    Returns (high, low, close) for the last complete period.
    """
    if df.empty:
        return (0.0, 0.0, 0.0)
    # Resample to specified period and take last row for the previous period
    resampled = df.resample(period).agg({'high': 'max', 'low': 'min', 'close': 'last'})
    if len(resampled) < 2:
        # Not enough data; use last values
        last = resampled.iloc[-1]
        return (float(last['high']), float(last['low']), float(last['close']))
    # Use second to last row for last complete period
    last_complete = resampled.iloc[-2]
    return (float(last_complete['high']), float(last_complete['low']), float(last_complete['close']))

def daily_pivot(df: pd.DataFrame) -> dict:
    """
    Compute daily pivot points using the last complete day's high, low and close.
    Requires df with datetime index and high, low, close columns.
    """
    high, low, close = _period_high_low_close(df, 'D')
    return calculate_pivot_points(high, low, close)

def weekly_pivot(df: pd.DataFrame) -> dict:
    """
    Compute weekly pivot points using the last complete week's high, low and close.
    """
    high, low, close = _period_high_low_close(df, 'W')
    return calculate_pivot_points(high, low, close)

def monthly_pivot(df: pd.DataFrame) -> dict:
    """
    Compute monthly pivot points using the last complete month's high, low and close.
    """
    high, low, close = _period_high_low_close(df, 'M')
    return calculate_pivot_points(high, low, close)

def detect_support_resistance(price_series: pd.Series, num_levels: int = 3) -> dict:
    """
    Detect immediate support and resistance levels from a price series.
    Returns a dictionary with lists of support and resistance levels and their strengths.

    The simplest method: take unique local minima (supports) and maxima (resistances) and
    sort them by proximity to the latest close.
    """
    if price_series.empty:
        return {
            "immediate_support": [],
            "immediate_resistance": [],
            "support_strength": [],
            "resistance_strength": [],
        }
    prices = price_series.astype(float).dropna().values
    # Identify local minima and maxima
    supports = []
    resistances = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            supports.append(prices[i])
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            resistances.append(prices[i])
    # Deduplicate and sort
    supports = sorted(set(supports))
    resistances = sorted(set(resistances), reverse=True)
    # Take nearest num_levels supports/resistances
    last_close = prices[-1]
    supports = sorted(supports, key=lambda x: abs(last_close - x))[:num_levels]
    resistances = sorted(resistances, key=lambda x: abs(last_close - x))[:num_levels]
    # Strength: stronger if closer to last_close and more touches
    def calc_strength(level: float) -> float:
        touches = np.sum(np.isclose(prices, level, atol=level * 0.001))
        proximity = 1 / (abs(last_close - level) + 1e-6)
        return touches * proximity
    support_strength = [calc_strength(level) for level in supports]
    resistance_strength = [calc_strength(level) for level in resistances]
    return {
        "immediate_support": supports,
        "immediate_resistance": resistances,
        "support_strength": support_strength,
        "resistance_strength": resistance_strength,
    }

def defense_zones(support_resistance: dict) -> dict:
    """
    Create bull and bear defense zones from support/resistance levels.
    The primary zone is the nearest support/resistance; secondary is the next one.
    Each zone includes price, width (difference between adjacent levels), and strength.
    """
    supports = support_resistance.get("immediate_support", [])
    resistances = support_resistance.get("immediate_resistance", [])
    support_strength = support_resistance.get("support_strength", [])
    resistance_strength = support_resistance.get("resistance_strength", [])
    bull_defense = {}
    bear_defense = {}
    # Primary and secondary bull (support) defense
    if supports:
        primary_price = supports[0]
        primary_strength = support_strength[0] if support_strength else 0.0
        if len(supports) > 1:
            secondary_price = supports[1]
            secondary_strength = support_strength[1] if len(support_strength) > 1 else 0.0
            width = abs(primary_price - secondary_price)
            bull_defense = {
                "primary": {"price": primary_price, "width": width, "strength": primary_strength},
                "secondary": {"price": secondary_price, "width": width, "strength": secondary_strength},
            }
        else:
            bull_defense = {
                "primary": {"price": primary_price, "width": 0.0, "strength": primary_strength},
                "secondary": None,
            }
    # Primary and secondary bear (resistance) defense
    if resistances:
        primary_price = resistances[0]
        primary_strength = resistance_strength[0] if resistance_strength else 0.0
        if len(resistances) > 1:
            secondary_price = resistances[1]
            secondary_strength = resistance_strength[1] if len(resistance_strength) > 1 else 0.0
            width = abs(primary_price - secondary_price)
            bear_defense = {
                "primary": {"price": primary_price, "width": width, "strength": primary_strength},
                "secondary": {"price": secondary_price, "width": width, "strength": secondary_strength},
            }
        else:
            bear_defense = {
                "primary": {"price": primary_price, "width": 0.0, "strength": primary_strength},
                "secondary": None,
            }
    # no man's land between last support and last resistance
    no_mans_land = None
    if supports and resistances:
        no_mans_land = {"start": supports[-1], "end": resistances[-1]}
    return {
        "bull_defense": bull_defense or None,
        "bear_defense": bear_defense or None,
        "no_mans_land": no_mans_land,
    }