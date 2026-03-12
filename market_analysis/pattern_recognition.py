import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any


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
) -> Optional[Dict[str, Any]]:
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
    highs = np.array(recent['high'].values, dtype=float)
    lows = np.array(recent['low'].values, dtype=float)
    # Determine the potential resistance as the mode of the top highs within tolerance
    max_high = float(np.max(highs))
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
    lowest_low = float(np.min(lows))
    last_close = float(recent['close'].iloc[-1])
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
        "target_price": round(float(target), 3),
        "stop_loss": round(float(stop_loss), 3),
        "confidence": round(confidence, 3),
    }


def detect_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    Detecta padrões de candlestick clássicos.
    
    Padrões detectados (10 mais relevantes para crypto):
        Single candle:
          1. Doji — Indecisão, possível reversão
          2. Hammer — Reversão bullish (fundo)
          3. Shooting Star — Reversão bearish (topo)
          4. Pin Bar — Rejeição forte
          
        Double candle:
          5. Bullish Engulfing — Reversão bullish forte
          6. Bearish Engulfing — Reversão bearish forte
          
        Triple candle:
          7. Morning Star — Reversão bullish forte (fundo)
          8. Evening Star — Reversão bearish forte (topo)
          9. Three White Soldiers — Continuação bullish forte
          10. Three Black Crows — Continuação bearish forte
    
    Args:
        df: DataFrame com colunas: open, high, low, close (ou o, h, l, c)
            Mínimo 3 rows, ideal 5+.
        lookback: Quantas candles analisar (default 5).
        
    Returns:
        Dict com padrões detectados, sinal dominante e confiança máxima.
    """
    patterns = []

    if df is None or len(df) < 2:
        return {
            "patterns_detected": 0,
            "patterns": [],
            "dominant_signal": "none",
            "max_confidence": 0,
        }

    # Normalizar nomes de colunas
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("open", "o"):
            col_map["open"] = col
        elif cl in ("high", "h"):
            col_map["high"] = col
        elif cl in ("low", "l"):
            col_map["low"] = col
        elif cl in ("close", "c"):
            col_map["close"] = col

    required = {"open", "high", "low", "close"}
    if not required.issubset(col_map.keys()):
        return {
            "patterns_detected": 0,
            "patterns": [],
            "dominant_signal": "none",
            "max_confidence": 0,
            "error": f"Missing columns. Found: {list(col_map.keys())}, need: {required}",
        }

    # Trabalhar com as últimas N candles
    data = df.tail(lookback).reset_index(drop=True)

    def _get(row_idx, field):
        """Helper para acessar dados de forma segura."""
        if row_idx < 0 or row_idx >= len(data):
            return None
        return float(data.iloc[row_idx][col_map[field]])

    def _is_bullish(idx):
        o = _get(idx, "open")
        c = _get(idx, "close")
        return c > o if o is not None and c is not None else None

    def _body(idx):
        o = _get(idx, "open")
        c = _get(idx, "close")
        return abs(c - o) if o is not None and c is not None else 0

    def _range(idx):
        h = _get(idx, "high")
        l = _get(idx, "low")
        return h - l if h is not None and l is not None else 0

    def _upper_shadow(idx):
        h = _get(idx, "high")
        o = _get(idx, "open")
        c = _get(idx, "close")
        if h is None or o is None or c is None:
            return 0
        return h - max(o, c)

    def _lower_shadow(idx):
        l = _get(idx, "low")
        o = _get(idx, "open")
        c = _get(idx, "close")
        if l is None or o is None or c is None:
            return 0
        return min(o, c) - l

    # Índice da candle mais recente
    last = len(data) - 1
    prev = last - 1
    prev2 = last - 2

    if last < 0:
        return {"patterns_detected": 0, "patterns": [], "dominant_signal": "none", "max_confidence": 0}

    c_body = _body(last)
    c_range = _range(last)
    c_upper = _upper_shadow(last)
    c_lower = _lower_shadow(last)
    c_bullish = _is_bullish(last)

    # Evitar divisão por zero
    if c_range == 0:
        c_range = 0.01

    # ════════════════════════════════════
    # 1. DOJI
    # ════════════════════════════════════
    if c_body < c_range * 0.10:
        doji_type = "doji"
        if c_lower > c_range * 0.6:
            doji_type = "dragonfly_doji"  # Bullish
        elif c_upper > c_range * 0.6:
            doji_type = "gravestone_doji"  # Bearish

        patterns.append({
            "name": doji_type,
            "type": "neutral" if doji_type == "doji" else ("bullish" if "dragonfly" in doji_type else "bearish"),
            "confidence": 0.65,
            "candles_used": 1,
            "implication": "Indecision - watch next candle for direction",
        })

    # ════════════════════════════════════
    # 2. HAMMER (bullish reversal at bottom)
    # ════════════════════════════════════
    if c_body > 0 and c_lower > c_body * 2 and c_upper < c_body * 0.5:
        patterns.append({
            "name": "hammer",
            "type": "bullish",
            "confidence": 0.70,
            "candles_used": 1,
            "implication": "Potential bottom reversal - buyers stepped in at lows",
        })

    # ════════════════════════════════════
    # 3. SHOOTING STAR (bearish reversal at top)
    # ════════════════════════════════════
    if c_body > 0 and c_upper > c_body * 2 and c_lower < c_body * 0.5:
        patterns.append({
            "name": "shooting_star",
            "type": "bearish",
            "confidence": 0.70,
            "candles_used": 1,
            "implication": "Potential top reversal - sellers rejected higher prices",
        })

    # ════════════════════════════════════
    # 4. PIN BAR (long wick rejection)
    # ════════════════════════════════════
    max_shadow = max(c_upper, c_lower)
    if c_range > 0 and max_shadow / c_range > 0.66 and c_body < c_range * 0.25:
        pin_dir = "bullish" if c_lower > c_upper else "bearish"
        patterns.append({
            "name": "pin_bar",
            "type": pin_dir,
            "confidence": 0.72,
            "candles_used": 1,
            "implication": f"Strong {pin_dir} rejection - {'lower' if pin_dir == 'bullish' else 'higher'} prices rejected",
        })

    # ════════════════════════════════════
    # Double candle patterns (need prev)
    # ════════════════════════════════════
    if prev >= 0:
        p_body = _body(prev)
        p_bullish = _is_bullish(prev)
        p_open = _get(prev, "open")
        p_close = _get(prev, "close")
        c_open = _get(last, "open")
        c_close = _get(last, "close")

        if p_open is not None and p_close is not None and c_open is not None and c_close is not None:

            # ════════════════════════════════════
            # 5. BULLISH ENGULFING
            # ════════════════════════════════════
            if (p_bullish is False and c_bullish is True and
                c_open <= p_close and c_close >= p_open and
                c_body > p_body * 0.8):
                patterns.append({
                    "name": "bullish_engulfing",
                    "type": "bullish",
                    "confidence": 0.78,
                    "candles_used": 2,
                    "implication": "Strong bullish reversal - buyers overwhelmed sellers",
                })

            # ════════════════════════════════════
            # 6. BEARISH ENGULFING
            # ════════════════════════════════════
            if (p_bullish is True and c_bullish is False and
                c_open >= p_close and c_close <= p_open and
                c_body > p_body * 0.8):
                patterns.append({
                    "name": "bearish_engulfing",
                    "type": "bearish",
                    "confidence": 0.78,
                    "candles_used": 2,
                    "implication": "Strong bearish reversal - sellers overwhelmed buyers",
                })

    # ════════════════════════════════════
    # Triple candle patterns (need prev2)
    # ════════════════════════════════════
    if prev2 >= 0:
        pp_bullish = _is_bullish(prev2)
        pp_open = _get(prev2, "open")
        pp_close = _get(prev2, "close")
        p_body_mid = _body(prev)
        p_range_mid = _range(prev)
        c_close_last = _get(last, "close")

        if pp_open is not None and pp_close is not None and c_close_last is not None:
            pp_midpoint = (pp_open + pp_close) / 2

            # ════════════════════════════════════
            # 7. MORNING STAR (bullish reversal)
            # ════════════════════════════════════
            if (pp_bullish is False and
                p_body_mid < p_range_mid * 0.35 and
                c_bullish is True and
                c_close_last > pp_midpoint):
                patterns.append({
                    "name": "morning_star",
                    "type": "bullish",
                    "confidence": 0.82,
                    "candles_used": 3,
                    "implication": "Strong bottom reversal - classic morning star formation",
                })

            # ════════════════════════════════════
            # 8. EVENING STAR (bearish reversal)
            # ════════════════════════════════════
            if (pp_bullish is True and
                p_body_mid < p_range_mid * 0.35 and
                c_bullish is False and
                c_close_last < pp_midpoint):
                patterns.append({
                    "name": "evening_star",
                    "type": "bearish",
                    "confidence": 0.82,
                    "candles_used": 3,
                    "implication": "Strong top reversal - classic evening star formation",
                })

        # ════════════════════════════════════
        # 9. THREE WHITE SOLDIERS
        # ════════════════════════════════════
        if (prev2 >= 0 and
            _is_bullish(prev2) is True and
            _is_bullish(prev) is True and
            c_bullish is True):
            c2_close = _get(prev2, "close")
            c1_close = _get(prev, "close")
            c0_close = _get(last, "close")
            if c2_close and c1_close and c0_close and c0_close > c1_close > c2_close:
                # Verificar que bodies são significativos
                if _body(prev2) > _range(prev2) * 0.3 and _body(prev) > _range(prev) * 0.3 and c_body > c_range * 0.3:
                    patterns.append({
                        "name": "three_white_soldiers",
                        "type": "bullish",
                        "confidence": 0.85,
                        "candles_used": 3,
                        "implication": "Strong bullish continuation - sustained buying pressure",
                    })

        # ════════════════════════════════════
        # 10. THREE BLACK CROWS
        # ════════════════════════════════════
        if (prev2 >= 0 and
            _is_bullish(prev2) is False and
            _is_bullish(prev) is False and
            c_bullish is False):
            c2_close = _get(prev2, "close")
            c1_close = _get(prev, "close")
            c0_close = _get(last, "close")
            if c2_close and c1_close and c0_close and c0_close < c1_close < c2_close:
                if _body(prev2) > _range(prev2) * 0.3 and _body(prev) > _range(prev) * 0.3 and c_body > c_range * 0.3:
                    patterns.append({
                        "name": "three_black_crows",
                        "type": "bearish",
                        "confidence": 0.85,
                        "candles_used": 3,
                        "implication": "Strong bearish continuation - sustained selling pressure",
                    })

    # ═══════════════════════════════════════════
    # RESULTADO
    # ═══════════════════════════════════════════
    # Determinar sinal dominante
    if patterns:
        # Ponderar por confiança
        bullish_weight = sum(p["confidence"] for p in patterns if p["type"] == "bullish")
        bearish_weight = sum(p["confidence"] for p in patterns if p["type"] == "bearish")
        neutral_weight = sum(p["confidence"] for p in patterns if p["type"] == "neutral")

        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            dominant = "bullish"
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            dominant = "bearish"
        elif neutral_weight > 0:
            dominant = "neutral"
        else:
            dominant = "mixed"

        max_conf = max(p["confidence"] for p in patterns)
    else:
        dominant = "none"
        max_conf = 0

    # Ordenar por confiança
    patterns.sort(key=lambda p: p["confidence"], reverse=True)

    return {
        "patterns_detected": len(patterns),
        "patterns": patterns,
        "dominant_signal": dominant,
        "max_confidence": round(max_conf, 4),
        "bullish_count": sum(1 for p in patterns if p["type"] == "bullish"),
        "bearish_count": sum(1 for p in patterns if p["type"] == "bearish"),
        "neutral_count": sum(1 for p in patterns if p["type"] == "neutral"),
    }


def recognize_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Reconhece padrões técnicos completos:
      - Triângulo ascendente (chart pattern)
      - Fibonacci levels
      - Candlestick patterns (10 padrões)
    
    Args:
        df: DataFrame com colunas OHLCV (open, high, low, close, volume)
        
    Returns:
        Dict com padrões ativos, fibonacci e candlestick patterns.
    """
    result: Dict[str, Any] = {
        "active_patterns": [],
        "fibonacci_levels": {},
        "candlestick_patterns": {
            "patterns_detected": 0,
            "patterns": [],
            "dominant_signal": "none",
            "max_confidence": 0,
        },
    }

    if df is None or df.empty:
        return result

    # 1. Triângulo ascendente
    try:
        tri = detect_ascending_triangle(df)
        if tri is not None:
            result["active_patterns"].append(tri)
    except Exception:
        pass

    # 2. Fibonacci levels
    try:
        # Determinar nomes das colunas
        high_col = "high" if "high" in df.columns else "h" if "h" in df.columns else None
        low_col = "low" if "low" in df.columns else "l" if "l" in df.columns else None

        if high_col and low_col:
            swing_high = float(df[high_col].max())
            swing_low = float(df[low_col].min())
            if swing_high > swing_low:
                result["fibonacci_levels"] = fibonacci_levels(swing_high, swing_low)
    except Exception:
        pass

    # 3. Candlestick patterns
    try:
        result["candlestick_patterns"] = detect_candlestick_patterns(df)
    except Exception:
        pass

    return result