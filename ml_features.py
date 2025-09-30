"""
ml_features.py

This module provides functions to compute engineered features from price,
volume and microstructure data for machine learning models. The goal is to
extract numerical descriptors from raw trading data that capture price
dynamics, volume behaviour and order book microstructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def calculate_price_features(df: pd.DataFrame, lookback_windows: List[int]) -> Dict[str, Any]:
    """
    Calculate price-based features such as returns and realized volatility for
    specified lookback windows (in number of rows).

    Args:
        df (pd.DataFrame): DataFrame with column 'close' and optional 'high', 'low'.
        lookback_windows (List[int]): List of window sizes (in rows) to compute
            returns and volatility. For example, [1, 5, 15] computes features for
            1, 5 and 15 periods.

    Returns:
        dict: A dictionary with keys 'returns_<window>', 'volatility_<window>'.
    """
    features: Dict[str, Any] = {}
    if df is None or df.empty or 'close' not in df:
        return features
    prices = df['close'].astype(float)
    for window in lookback_windows:
        if len(prices) >= window + 1:
            current_close = prices.iloc[-1]
            past_close = prices.iloc[-(window + 1)]
            if past_close > 0:
                ret = (current_close - past_close) / past_close
            else:
                ret = 0.0
            features[f'returns_{window}'] = float(ret)
            # Realized volatility as std of log returns over the window
            log_returns = np.log(prices.iloc[-(window + 1):].values[1:] / prices.iloc[-(window + 1):-1].values)
            vol = np.std(log_returns, ddof=0)
            features[f'volatility_{window}'] = float(vol)
        else:
            features[f'returns_{window}'] = 0.0
            features[f'volatility_{window}'] = 0.0
    # Momentum score: z-score of the longest return window
    last_window = max(lookback_windows)
    if f'returns_{last_window}' in features and f'volatility_{last_window}' in features:
        mu_ret = np.mean([features[f'returns_{w}'] for w in lookback_windows])
        sigma_ret = np.std([features[f'returns_{w}'] for w in lookback_windows], ddof=0)
        if sigma_ret > 0:
            momentum = (features[f'returns_{last_window}'] - mu_ret) / sigma_ret
        else:
            momentum = 0.0
        features['momentum_score'] = float(momentum)
    return features


def calculate_volume_features(df: pd.DataFrame, volume_ma_window: int) -> Dict[str, Any]:
    """
    Calculate volume-based features.
    Features include volume moving-average ratio, volume momentum and buy/sell pressure.

    Args:
        df (pd.DataFrame): DataFrame with columns 'q' (quantity), 'p' (price) and 'm'
                            (trade direction: True for sell, False for buy).
        volume_ma_window (int): Window size for moving average of volume.

    Returns:
        dict: Volume features.
    """
    features: Dict[str, Any] = {}
    if df is None or df.empty or 'q' not in df:
        return features
    vols = df['q'].astype(float)
    # Volume SMA ratio: current volume vs. SMA
    current_vol = vols.sum()
    if len(vols) >= volume_ma_window:
        sma_vol = vols.rolling(window=volume_ma_window, min_periods=volume_ma_window).sum().iloc[-1]
    else:
        sma_vol = vols.sum()
    sma_vol = sma_vol if sma_vol > 0 else 1e-9
    features['volume_sma_ratio'] = float(current_vol / sma_vol)
    # Volume momentum: difference in volume between halves of the window
    half = len(vols) // 2
    vol_first = vols.iloc[:half].sum() if half > 0 else 0.0
    vol_second = vols.iloc[half:].sum()
    if vol_first > 0:
        features['volume_momentum'] = float((vol_second - vol_first) / vol_first)
    else:
        features['volume_momentum'] = float(vol_second)
    # Buy/sell pressure
    if 'm' in df:
        sell_mask = df['m'].astype(bool)
        buy_mask = ~sell_mask
        buy_vol = vols[buy_mask].sum()
        sell_vol = vols[sell_mask].sum()
        total = buy_vol + sell_vol
        if total > 0:
            features['buy_sell_pressure'] = float((buy_vol - sell_vol) / total)
        else:
            features['buy_sell_pressure'] = 0.0
    else:
        features['buy_sell_pressure'] = 0.0
    # Liquidity gradient: based on volume in first half vs second half
    if vol_first > 0:
        features['liquidity_gradient'] = float((vol_second - vol_first) / vol_first)
    else:
        features['liquidity_gradient'] = float(vol_second)
    return features


def calculate_microstructure_features(orderbook_data: Dict[str, Any], flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute microstructure features from orderbook and flow metrics.
    - order_book_slope: measures how quickly liquidity diminishes deeper in the book.
    - flow_imbalance: ratio of net flow to total flow.
    - tick_rule_sum: sum of tick directions (sign of price change) over the window.
    - trade_intensity: trades per second.

    Args:
        orderbook_data (dict): Data from OrderBookAnalyzer with depth levels.
        flow_metrics (dict): Data from FlowAnalyzer (contains sector_flow and other metrics).

    Returns:
        dict: Microstructure features.
    """
    features: Dict[str, Any] = {}
    # Order book slope
    try:
        ob_depth = orderbook_data.get('order_book_depth', {})
        # Use L1 and L10 depths if available
        l1 = ob_depth.get('L1', {})
        l10 = ob_depth.get('L10', {})
        if l1 and l10:
            bid1 = l1.get('bids', 0.0)
            ask1 = l1.get('asks', 0.0)
            bid10 = l10.get('bids', 0.0)
            ask10 = l10.get('asks', 0.0)
            # slope: difference between top and deeper levels normalized by depth
            denom = (bid10 + ask10) if (bid10 + ask10) != 0 else 1e-9
            slope = ((bid1 + ask1) - (bid10 + ask10)) / denom
            features['order_book_slope'] = float(slope)
    except Exception:
        features['order_book_slope'] = 0.0
    # Flow imbalance
    try:
        # net_flow is difference of buy_notional_usdt and sell_notional_usdt in flow_metrics
        net_flow = 0.0
        total_flow = 0.0
        # flow_metrics might contain order_flow information if integrated
        if 'order_flow' in flow_metrics:
            f = flow_metrics['order_flow']
            net_flow = float(f.get('net_flow_1m', 0.0))
            # approximate total as absolute net_flow divided by ratio
            total_flow = abs(net_flow) / max(f.get('buy_sell_ratio', 1e-9), 1e-9)
        # fallback: use sector_flow from flow_metrics
        else:
            sector_flow = flow_metrics.get('sector_flow', {})
            buy_total = sum(v.get('buy', 0.0) for v in sector_flow.values())
            sell_total = sum(v.get('sell', 0.0) for v in sector_flow.values())
            net_flow = buy_total - sell_total
            total_flow = buy_total + sell_total
        if total_flow > 0:
            features['flow_imbalance'] = float(net_flow / total_flow)
        else:
            features['flow_imbalance'] = 0.0
    except Exception:
        features['flow_imbalance'] = 0.0
    # Tick rule sum
    try:
        # Approximate tick rule using orderbook_data's spread movement if available
        # We don't have raw ticks here, so this is a placeholder using spread sign
        spread = orderbook_data.get('spread_metrics', {}).get('spread', 0.0)
        if spread is not None:
            # Positive spread implies sellers dominating, negative implies buyers; sum over a window is stubbed here
            features['tick_rule_sum'] = float(np.sign(spread))
        else:
            features['tick_rule_sum'] = 0.0
    except Exception:
        features['tick_rule_sum'] = 0.0
    # Trade intensity
    try:
        # Use number of trades and window duration if available in flow_metrics
        bursts = flow_metrics.get('bursts', {})
        count = bursts.get('count', 0)
        # approximate using the burst window size (ms) and count
        window_ms = flow_metrics.get('metadata', {}).get('burst_window_ms', 1000)
        if window_ms > 0:
            intensity = count / (window_ms / 1000.0)
        else:
            intensity = 0.0
        features['trade_intensity'] = float(intensity)
    except Exception:
        features['trade_intensity'] = 0.0
    return features


def generate_ml_features(
    df: pd.DataFrame,
    orderbook_data: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    lookback_windows: List[int] = [1, 5, 15],
    volume_ma_window: int = 20,
) -> Dict[str, Any]:
    """
    Generate consolidated machine learning features from price, volume and
    microstructure data.

    Args:
        df (pd.DataFrame): DataFrame with raw trades or candle data; must include
                           'close', 'q', 'p', 'm' columns as applicable.
        orderbook_data (dict): Output from OrderBookAnalyzer.
        flow_metrics (dict): Output from FlowAnalyzer.
        lookback_windows (List[int]): List of windows for price features.
        volume_ma_window (int): Window for volume moving average.

    Returns:
        dict: Consolidated ML features grouped under keys 'price_features',
              'volume_features' and 'microstructure'.
    """
    price_feats = calculate_price_features(df[['close']], lookback_windows)
    volume_feats = calculate_volume_features(df[['q', 'p', 'm']], volume_ma_window)
    micro_feats = calculate_microstructure_features(orderbook_data, flow_metrics)
    return {
        "price_features": price_feats,
        "volume_features": volume_feats,
        "microstructure": micro_feats,
    }