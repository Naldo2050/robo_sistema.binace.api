import numpy as np
import pandas as pd
from typing import Optional

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.
    Returns a pandas Series with RSI values.
    """
    series = series.astype(float)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    return rsi

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """
    Compute the MACD line and signal line for a price series.
    Returns a tuple (macd_line, signal_line).
    """
    series = series.astype(float)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Helper to compute True Range, used in ATR and ADX calculations.
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) for given high, low, close series.
    Returns a pandas Series with ADX values.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    # Compute directional movements
    up_move = high.diff()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # True range
    tr = _true_range(high, low, close)
    # Smooth TR, plus_dm, minus_dm
    atr = tr.rolling(window=window, min_periods=window).sum().shift()  # to align
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window, min_periods=window).mean()
    return adx

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Compute the Stochastic Oscillator %K and %D lines.
    Returns a tuple (%K, %D).
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return k, d

def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute the Commodity Channel Index (CCI).
    Returns a pandas Series with CCI values.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    typical_price = (high + low + close) / 3.0
    sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
    mean_dev = typical_price.rolling(window=window, min_periods=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp) / (0.015 * mean_dev)
    return cci

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Williams %R - Oscilador de momentum.
    Varia de -100 a 0.
    Acima de -20 = overbought, abaixo de -80 = oversold.
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    denom = highest_high - lowest_low
    denom = denom.replace(0, float('nan'))
    wr = ((highest_high - close) / denom) * -100
    return wr.fillna(-50)

def stochastic_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI - RSI aplicado sobre o próprio RSI.
    Mais sensível que RSI ou Stochastic isolados.
    Retorna (K, D) onde ambos variam de 0 a 100.
    K > 80 = overbought, K < 20 = oversold.
    Crossover K/D gera sinais.
    """
    rsi_values = rsi(series, window=rsi_period)
    min_rsi = rsi_values.rolling(window=stoch_period).min()
    max_rsi = rsi_values.rolling(window=stoch_period).max()
    denom = max_rsi - min_rsi
    denom = denom.replace(0, float('nan'))
    stoch_rsi_raw = ((rsi_values - min_rsi) / denom) * 100
    stoch_rsi_raw = stoch_rsi_raw.fillna(50)
    k = stoch_rsi_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def twap(close_prices: pd.Series) -> float:
    """
    TWAP (Time-Weighted Average Price).
    Preço médio ponderado pelo tempo — cada barra tem peso igual.
    Divergência TWAP vs VWAP indica absorção institucional:
      - TWAP > VWAP → preço ficou mais tempo em cima (absorção compradora)
      - TWAP < VWAP → preço ficou mais tempo embaixo (absorção vendedora)

    Args:
        close_prices: Série de preços de fechamento da sessão/período

    Returns:
        Valor TWAP (float)
    """
    if close_prices.empty:
        return 0.0
    return float(close_prices.mean())


def twap_vwap_analysis(
    close_prices: pd.Series,
    volumes: pd.Series,
    current_vwap: Optional[float] = None
) -> dict:
    """
    Calcula TWAP e compara com VWAP para detectar absorção.

    Args:
        close_prices: Série de preços de fechamento
        volumes: Série de volumes correspondentes
        current_vwap: VWAP pré-calculado (opcional, calcula se não fornecido)

    Returns:
        Dict com TWAP, VWAP, divergência e sinal
    """
    if close_prices.empty or len(close_prices) < 2:
        return {
            "twap": 0.0,
            "vwap": 0.0,
            "divergence_pct": 0.0,
            "signal": "insufficient_data",
        }

    twap_value = float(close_prices.mean())

    if current_vwap is not None and current_vwap > 0:
        vwap_value = current_vwap
    else:
        # Calcular VWAP
        total_volume = volumes.sum()
        if total_volume > 0:
            vwap_value = float((close_prices * volumes).sum() / total_volume)
        else:
            vwap_value = twap_value

    # Divergência
    if vwap_value > 0:
        divergence_pct = round((twap_value - vwap_value) / vwap_value * 100, 6)
    else:
        divergence_pct = 0.0

    # Classificação do sinal
    if abs(divergence_pct) < 0.005:
        signal = "neutral"
    elif divergence_pct > 0.02:
        signal = "strong_buy_absorption"
    elif divergence_pct > 0:
        signal = "slight_buy_absorption"
    elif divergence_pct < -0.02:
        signal = "strong_sell_absorption"
    else:
        signal = "slight_sell_absorption"

    return {
        "twap": round(twap_value, 2),
        "vwap": round(vwap_value, 2),
        "divergence_pct": divergence_pct,
        "signal": signal,
        "interpretation": (
            "Price spent more time at higher levels (buy absorption)"
            if divergence_pct > 0
            else "Price spent more time at lower levels (sell absorption)"
            if divergence_pct < 0
            else "Price time-balanced (no absorption detected)"
        ),
    }


def realized_volatility(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Compute the realized volatility of a series of prices.
    Volatility is the standard deviation of log returns over the specified window.
    Returns a pandas Series of annualized volatility.
    """
    series = series.astype(float)
    log_returns = pd.Series(np.log(series / series.shift()), index=series.index)
    vol = log_returns.rolling(window=window, min_periods=window).std(ddof=0)
    # annualize assuming 365 days and 24*60/5 minute bars: adjust if sampling different frequency
    return vol * np.sqrt(window)

def detect_regime(adx_series: pd.Series, vol_series: pd.Series, adx_threshold: float = 25, vol_threshold: float = 0.02) -> pd.Series:
    """
    Simple regime detection combining ADX and volatility.
    If ADX > adx_threshold and volatility < vol_threshold: Trending.
    If ADX <= adx_threshold and volatility < vol_threshold: Range-bound.
    If volatility >= vol_threshold: High-volatility regime.
    Returns a pandas Series of strings.
    """
    def classify_row(row):
        if row['vol'] >= vol_threshold:
            return 'High Volatility'
        if row['adx'] > adx_threshold:
            return 'Trending'
        return 'Range'
    df = pd.DataFrame({'adx': adx_series, 'vol': vol_series})
    regime = df.apply(classify_row, axis=1)
    return regime