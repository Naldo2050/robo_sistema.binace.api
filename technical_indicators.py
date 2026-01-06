import numpy as np
import pandas as pd

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

def realized_volatility(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Compute the realized volatility of a series of prices.
    Volatility is the standard deviation of log returns over the specified window.
    Returns a pandas Series of annualized volatility.
    """
    series = series.astype(float)
    log_returns = np.log(series / series.shift())
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