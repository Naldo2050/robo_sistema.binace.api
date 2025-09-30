"""
alert_engine.py

This module provides a centralized engine to detect and generate alerts based on
market conditions. Alerts are triggered when specific conditions are met,
such as price tests of support/resistance levels, volume spikes, or volatility
squeezes.

Functions in this module are designed to be lightweight and easily invoked
within the data pipeline or other parts of the system.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def detect_support_resistance_test(
    price: float,
    support_levels: List[float],
    resistance_levels: List[float],
    tolerance_pct: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    Detects when the current price tests support or resistance levels within a
    relative tolerance.

    Args:
        price (float): The current price.
        support_levels (List[float]): Sorted list of support levels.
        resistance_levels (List[float]): Sorted list of resistance levels.
        tolerance_pct (float): Relative tolerance (e.g., 0.001 = 0.1%) used to
                               detect tests.

    Returns:
        List[Dict[str, Any]]: A list of alert dictionaries, one per level
                              tested. Each alert has keys: type, level,
                              severity, probability, action.
    """
    alerts = []
    if price <= 0:
        return alerts
    tolerance = price * tolerance_pct
    # Test supports
    for level in support_levels:
        if abs(price - level) <= tolerance:
            alerts.append({
                "type": "SUPPORT_TEST",
                "level": level,
                "severity": "HIGH",
                "probability": 0.75,
                "action": "MONITOR_CLOSELY",
            })
    # Test resistances
    for level in resistance_levels:
        if abs(price - level) <= tolerance:
            alerts.append({
                "type": "RESISTANCE_TEST",
                "level": level,
                "severity": "HIGH",
                "probability": 0.75,
                "action": "MONITOR_CLOSELY",
            })
    return alerts


def detect_volume_spike(
    current_volume: float,
    average_volume: float,
    threshold_factor: float = 3.0,
    duration: str = "5_MINUTES",
) -> Optional[Dict[str, Any]]:
    """
    Detects a volume spike when the current volume exceeds a multiple of the
    average volume.

    Args:
        current_volume (float): Volume in the current window.
        average_volume (float): Average volume over a reference period.
        threshold_factor (float): Factor by which the current volume must
                                  exceed the average to trigger an alert.
        duration (str): Description of the reference window (for metadata).

    Returns:
        dict or None: An alert dictionary if a spike is detected, otherwise
                      None.
    """
    if average_volume <= 0:
        return None
    ratio = current_volume / average_volume
    if ratio >= threshold_factor:
        return {
            "type": "VOLUME_SPIKE",
            "threshold_exceeded": round(ratio, 2),
            "severity": "MEDIUM" if ratio < 5 else "HIGH",
            "duration": duration,
        }
    return None


def detect_volatility_squeeze(
    current_vol: float,
    recent_vols: List[float],
    low_percentile: float = 10.0,
    high_percentile: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """
    Detects a volatility squeeze when current volatility is in the lower or
    upper percentiles of recent volatility values.

    Args:
        current_vol (float): Current realized volatility or similar measure.
        recent_vols (List[float]): A list of recent volatility values.
        low_percentile (float): Lower percentile threshold for low volatility.
        high_percentile (float): Upper percentile threshold for high volatility.

    Returns:
        dict or None: An alert dictionary indicating a volatility squeeze,
                      otherwise None.
    """
    if not recent_vols:
        return None
    vols = np.array([v for v in recent_vols if v is not None])
    if vols.size == 0:
        return None
    low_thresh = np.percentile(vols, low_percentile)
    high_thresh = np.percentile(vols, high_percentile)
    if current_vol <= low_thresh:
        return {
            "type": "VOLATILITY_SQUEEZE",
            "level": "LOW",
            "severity": "MEDIUM",
            "probability": 0.6,
            "action": "WATCH_FOR_BREAKOUT",
        }
    if current_vol >= high_thresh:
        return {
            "type": "VOLATILITY_SQUEEZE",
            "level": "HIGH",
            "severity": "MEDIUM",
            "probability": 0.6,
            "action": "WATCH_FOR_REVERSION",
        }
    return None


def generate_alerts(
    price: float,
    support_resistance: Dict[str, Any],
    current_volume: float,
    average_volume: float,
    current_volatility: float,
    recent_volatilities: List[float],
    volume_threshold: float = 3.0,
    tolerance_pct: float = 0.001,
) -> List[Dict[str, Any]]:
    """
    Generate a list of alerts based on support/resistance tests, volume spikes,
    and volatility squeezes.

    Args:
        price (float): Current price.
        support_resistance (dict): Contains immediate_support and
                                   immediate_resistance lists.
        current_volume (float): Volume in current window.
        average_volume (float): Reference average volume.
        current_volatility (float): Current realized volatility.
        recent_volatilities (List[float]): List of recent volatility values.
        volume_threshold (float): Factor threshold for volume spike.
        tolerance_pct (float): Tolerance for support/resistance tests.

    Returns:
        List[Dict[str, Any]]: List of alerts triggered.
    """
    alerts = []
    # Support/resistance tests
    supp = support_resistance.get("immediate_support", [])
    res = support_resistance.get("immediate_resistance", [])
    alerts.extend(detect_support_resistance_test(price, supp, res, tolerance_pct))
    # Volume spike
    vol_alert = detect_volume_spike(current_volume, average_volume, volume_threshold)
    if vol_alert:
        alerts.append(vol_alert)
    # Volatility squeeze
    vol_alert2 = detect_volatility_squeeze(current_volatility, recent_volatilities)
    if vol_alert2:
        alerts.append(vol_alert2)
    return alerts