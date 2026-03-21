# context_collector.py v3.1.0 - ASYNC MIGRATION
"""
Context Collector com migração para assíncrono.
Mantém interface síncrona externa, mas usa asyncio internamente.

🔹 MIGRAÇÃO v3.1.0:
   ✅ Interface pública síncrona mantida
   ✅ Internamente assíncrono com aiohttp + yfinance
   ✅ Loop dedicado em thread separada
   ✅ Compatibilidade com chaves v2.1.0
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import logging
import threading
import os
import random
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

# Importações de config com try/except para resolver erros do Pylance
try:
    from config import (
        CONTEXT_TIMEFRAMES, CONTEXT_EMA_PERIOD, CONTEXT_ATR_PERIOD,
        CONTEXT_UPDATE_INTERVAL_SECONDS, INTERMARKET_SYMBOLS, DERIVATIVES_SYMBOLS,
        VP_NUM_DAYS_HISTORY, VP_VALUE_AREA_PERCENT, LIQUIDATION_MAP_DEPTH,
        EXTERNAL_MARKETS
    )
except (ImportError, AttributeError):
    # Fallbacks para constantes do Context Collector
    CONTEXT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
    CONTEXT_EMA_PERIOD = 21
    CONTEXT_ATR_PERIOD = 14
    CONTEXT_UPDATE_INTERVAL_SECONDS = 300
    INTERMARKET_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    DERIVATIVES_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    VP_NUM_DAYS_HISTORY = 30
    VP_VALUE_AREA_PERCENT = 0.7
    LIQUIDATION_MAP_DEPTH = 500.0
    EXTERNAL_MARKETS = {
        "SP500": "^GSPC",
        "DXY": "DX-Y.NYB",
        "NASDAQ": "^IXIC",
        "TNX": "^TNX",
        "GOLD": "GC=F",
        "WTI": "CL=F",
    }
    ENABLE_ONCHAIN = False
    ONCHAIN_PROVIDERS = []
    STABLECOIN_FLOW_TRACKING = False

# [SUPPORT_RESISTANCE] Importa cálculo de Pivots
from support_resistance import daily_pivot, weekly_pivot, monthly_pivot

# Configurações opcionais - com try/except para compatibilidade com Pylance
try:
    from config import ENABLE_ALPHAVANTAGE
except (ImportError, AttributeError):
    ENABLE_ALPHAVANTAGE = True

try:
    from config import (
        CORRELATION_LOOKBACK,
        VOLATILITY_PERCENTILES,
        ADX_PERIOD,
        RSI_PERIODS,
        MACD_FAST_PERIOD,
        MACD_SLOW_PERIOD,
        MACD_SIGNAL_PERIOD,
    )
except (ImportError, AttributeError):
    CORRELATION_LOOKBACK = 50
    VOLATILITY_PERCENTILES = (0.35, 0.65)
    ADX_PERIOD = 14
    RSI_PERIODS = {"short": 14, "long": 21}
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9

from market_analysis.historical_profiler import HistoricalVolumeProfiler
from monitoring.time_manager import TimeManager
from fetchers.fred_fetcher import FREDFetcher
from monitoring.metrics_collector import record_fred_fallback

# Fetchers reais (on-chain e funding agregado)
try:
    from onchain_fetcher import OnchainFetcher
    _ONCHAIN_FETCHER_AVAILABLE = True
except ImportError:
    _ONCHAIN_FETCHER_AVAILABLE = False

try:
    from funding_aggregator import FundingAggregator
    _FUNDING_AGG_AVAILABLE = True
except ImportError:
    _FUNDING_AGG_AVAILABLE = False

# Mapeamento de tickers para yFinance
TICKER_MAPPING = {
    'BTC': 'BTC-USD',
    'DXY': 'DX-Y.NYB',      # US Dollar Index real (não ETF proxy)
    'NASDAQ': '^IXIC',      # NASDAQ Composite Index real (não QQQ ETF)
    'SP500': '^GSPC',       # S&P 500 Index real (não SPY ETF)
    'TNX': '^TNX',
    'GOLD': 'GC=F',
    'XAUUSD': 'GC=F',
    'CL': 'CL=F',
    'WTI': 'CL=F',
    'VIX': '^VIX',
}

# Configurações para yFinance
YFINANCE_CONFIG: Dict[str, Any] = {
    'timeout': 15,  # 🆕 Aumentado de 10 para 15
    'retries': 2,   # 🆕 Reduzido de 3 para 2 (já tem lock)
    'interval': '1d'
}
YFINANCE_RETRIES: int = int(YFINANCE_CONFIG['retries'])
YFINANCE_INTERVAL: str = str(YFINANCE_CONFIG['interval'])

logger = logging.getLogger("ContextCollector")


class ContextCollector:
    """
    Coletor de contexto macro assíncrono internamente, síncrono externamente.
    
    🆕 v3.1.0:
       - Assíncrono com aiohttp/yfinance
       - Thread dedicada para event loop
       - Interface compatível v2.1.0
    """

    def __init__(self, symbol):
        self.symbol = symbol
        self.timeframes = CONTEXT_TIMEFRAMES
        self.ema_period = CONTEXT_EMA_PERIOD
        self.atr_period = CONTEXT_ATR_PERIOD
        self.update_interval = CONTEXT_UPDATE_INTERVAL_SECONDS
        self.intermarket_symbols = INTERMARKET_SYMBOLS
        self.derivatives_symbols = DERIVATIVES_SYMBOLS
         
        # Endpoints Binance
        self.klines_api_url = "https://api.binance.com/api/v3/klines"
        self.funding_api_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.open_interest_api_url = "https://fapi.binance.com/fapi/v1/openInterest"
        self.long_short_ratio_api_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        self.liquidations_api_url = "https://fapi.binance.com/fapi/v1/allForceOrders"
        self.mark_price_api_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
         
        # Volume Profile
        self.historical_profiler = HistoricalVolumeProfiler(
            symbol=self.symbol,
            num_days=VP_NUM_DAYS_HISTORY,
            value_area_percent=VP_VALUE_AREA_PERCENT
        )
         
        self._context_data = {}
        self.time_manager = TimeManager()
         
        # Cache simples
        self._api_cache = {}
        self._cache_ttl = 300  # s (era 60s — reduz chamadas yFinance em 5x)
         
        # Alpha Vantage
        self.alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY", "KC4IE0MBOEXK88Y3")
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
         
        # 🆕 VALIDAÇÃO DE API KEY
        if ENABLE_ALPHAVANTAGE:
            if not self.alpha_vantage_api_key or self.alpha_vantage_api_key == "demo":
                logger.warning("⚠️ Alpha Vantage habilitado mas API key inválida/demo!")
         
        # Fetchers reais de on-chain e funding agregado
        self._onchain_fetcher = OnchainFetcher() if _ONCHAIN_FETCHER_AVAILABLE else None
        self._funding_aggregator = FundingAggregator() if _FUNDING_AGG_AVAILABLE else None

        # Fallbacks
        self._dxy_cache: Optional[Dict[str, Any]] = None
        self._dxy_cache_time: float = 0.0
        self._dxy_candidates = ["DXY"]
        self._fallback_map = {
            "S&P500": ["GSPC", "SPY"],
            "NASDAQ": ["IXIC", "QQQ"],
            "UST10Y": ["TNX", "TYX", "FVX"],
            "GOLD": ["GC", "XAUUSD"],
            "WTI": ["CL", "BZ"],
        }
         
        # Async internals
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = None
        self._update_task = None
        
        # Thread pool executor for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # FRED API (fallback para dados econômicos)
        self.fred_fetcher = FREDFetcher()
        
        # 🆕 Lock para yFinance (adicionar após self._executor)
        self._yfinance_lock = None  # Será criado na primeira chamada async
        
        logger.info(
            "✅ ContextCollector inicializado | Symbol: %s | Alpha Vantage: %s",
            symbol, "ENABLED" if ENABLE_ALPHAVANTAGE else "DISABLED"
        )

    # ---------- Helpers Alpha Vantage (async) ----------
    
    async def _build_retrying_session(self) -> aiohttp.ClientSession:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        # Armazenar referência para fechar no stop()
        self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _yfinance_history(self, symbol: str, period: str = "90d", interval: str = "1d") -> pd.DataFrame:
        """Busca dados históricos do yFinance com retry e proteção contra concorrência."""
        ticker = TICKER_MAPPING.get(symbol, symbol)
        logger.info(f"🔍 Buscando dados yFinance para {symbol} (ticker: {ticker})")

        # Garantir que o lock existe e pertence ao event loop correto
        if self._yfinance_lock is None:
            self._yfinance_lock = asyncio.Lock()

        async with self._yfinance_lock:
            def _sync_fetch():
                """Função síncrona para buscar dados do yfinance."""
                try:
                    import time
                    time.sleep(0.5)  # Rate limiting

                    ticker_obj = yf.Ticker(ticker)

                    # 🆕 Tentar múltiplos métodos
                    df = None

                    # Método 1: history() com raise_errors=False para yfinance 1.0
                    try:
                        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTE
                        def _yf_hist():
                            return ticker_obj.history(
                                period=period,
                                interval=interval,
                                raise_errors=False
                            )
                        # Hard timeout - yfinance timeout param nao funciona no v1.0
                        with ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(_yf_hist)
                            try:
                                df = fut.result(timeout=15)
                            except FTE:
                                logger.warning(f"⏰ Timeout (15s) ao buscar {ticker}")
                                df = None
                    except Exception as e:
                        logger.debug(f"history() falhou para {ticker}: {e}")

                    # NOTA: yf.download() removido - bugado no yfinance 1.0 para
                    # tickers com caracteres especiais (DX-Y.NYB, ^GSPC, etc.)
                    # Causa "'NoneType' object is not subscriptable"

                    # Verificar formato
                    if df is not None and not df.empty:
                        logger.debug(f"Colunas retornadas para {ticker}: {df.columns.tolist()}")

                        # Normalizar nomes de colunas
                        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]

                        # Se tem MultiIndex, pegar o primeiro nível
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)

                        return df

                    return pd.DataFrame()

                except Exception as e:
                    logger.debug(f"_sync_fetch erro para {ticker}: {e}")
                    return pd.DataFrame()

            for attempt in range(YFINANCE_RETRIES):
                try:
                    loop = asyncio.get_running_loop()
                    df = await loop.run_in_executor(self._executor, _sync_fetch)

                    if df is not None and not df.empty:
                        # 🆕 Verificar múltiplas variações de 'close'
                        close_col = None
                        for col in ['close', 'Close', 'adj close', 'Adj Close']:
                            if col in df.columns:
                                close_col = col
                                break

                        if close_col:
                            df = df.rename(columns={close_col: 'close'})
                            result_df = df[['close']].dropna()

                            if len(result_df) >= 1:
                                logger.debug(f"✅ Dados yFinance obtidos para {symbol}: {len(result_df)} pontos")
                                return result_df
                            else:
                                logger.warning(f"⚠️ Dados insuficientes yFinance para {symbol}: {len(result_df)} pontos")
                        else:
                            logger.warning(f"⚠️ Coluna 'close' não encontrada para {symbol}. Colunas: {df.columns.tolist()}")
                    else:
                        logger.warning(f"⚠️ DataFrame vazio para {symbol} (ticker: {ticker})")

                except Exception as e:
                    logger.warning(f"⚠️ Erro yFinance para {symbol} (tentativa {attempt+1}): {e}")

                if attempt < YFINANCE_RETRIES - 1:
                    await asyncio.sleep(3)

            logger.error(f"❌ Falha total yFinance para {symbol} após {YFINANCE_RETRIES} tentativas")
            return pd.DataFrame()

    async def _alpha_vantage_history(self, session: aiohttp.ClientSession, symbol: str, function: str = "ECONOMIC_INDICATORS", interval: str = "1min"):
        """Wrapper resiliente para Alpha Vantage com retry (async)."""
        logger.info(f"🔍 Tentando buscar dados Alpha Vantage para {symbol} com função {function}")
        try_count = 2
        for i in range(try_count):
            try:
                params = {
                    "function": function,
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_api_key
                }
                if function == "TIME_SERIES_INTRADAY":
                    params["interval"] = interval
                elif function == "TIME_SERIES_DAILY":
                    params["outputsize"] = "compact"
                elif function == "ECONOMIC_INDICATORS":
                    pass
                elif function == "FX_DAILY":
                    params["from_symbol"] = symbol.split("/")[0]
                    params["to_symbol"] = symbol.split("/")[1]

                async with session.get(self.alpha_vantage_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as res:
                    if res.status == 200:
                        data = await res.json()
                        if "Error Message" in data:
                            logger.warning(f"❌ Alpha Vantage erro para {symbol} (função {function}): {data['Error Message']}")
                            return pd.DataFrame()
                        if "Time Series (1min)" in data:
                            df = pd.DataFrame(data["Time Series (1min)"]).T
                            df.index = pd.to_datetime(df.index)
                            df.columns = ["open", "high", "low", "close", "volume"]
                            df = df.astype(float)
                            logger.info(f"✅ Dados Alpha Vantage obtidos para {symbol} (1min)")
                            return df
                        elif "Time Series (Daily)" in data:
                            df = pd.DataFrame(data["Time Series (Daily)"]).T
                            df.index = pd.to_datetime(df.index)
                            df.columns = ["open", "high", "low", "close", "volume"]
                            df = df.astype(float)
                            logger.info(f"✅ Dados Alpha Vantage obtidos para {symbol} (Daily)")
                            return df
                        elif "Time Series (Monthly)" in data:
                            df = pd.DataFrame(data["Time Series (Monthly)"]).T
                            df.index = pd.to_datetime(df.index)
                            df.columns = ["open", "high", "low", "close", "volume"]
                            df = df.astype(float)
                            logger.info(f"✅ Dados Alpha Vantage obtidos para {symbol} (Monthly)")
                            return df
                        elif "Data" in data:
                            df = pd.DataFrame(data["Data"]).T
                            df.index = pd.to_datetime(df.index)
                            df.columns = ["value"]
                            df = df.astype(float)
                            logger.info(f"✅ Dados Alpha Vantage obtidos para {symbol} (Data)")
                            return df
                        elif "Time Series FX (Daily)" in data:
                            df = pd.DataFrame(data["Time Series FX (Daily)"]).T
                            df.index = pd.to_datetime(df.index)
                            df.columns = ["open", "high", "low", "close"]
                            df = df.astype(float)
                            logger.info(f"✅ Dados Alpha Vantage obtidos para {symbol} (FX Daily)")
                            return df
                        else:
                            logger.warning(f"⚠️ Resposta Alpha Vantage inesperada para {symbol} (função {function}): {str(data)[:160]}")
                            return pd.DataFrame()
                    else:
                        logger.warning(f"⚠️ Alpha Vantage status {res.status} para {symbol} (função {function}): {(await res.text())[:160]}")
            except Exception as e:
                logger.warning(f"⚠️ Alpha Vantage erro para {symbol} (função {function}, tentativa {i+1}/{try_count}): {e}")

            if i < try_count - 1:
                await asyncio.sleep(0.6 + random.uniform(0, 0.5))
        logger.error(f"❌ Falha total em obter dados Alpha Vantage para {symbol} após {try_count} tentativas")
        return pd.DataFrame()

    # ---------- Cache genérico (async) ----------

    async def _async_fetch_with_cache(self, cache_key: str, fetch_fn, ttl_seconds: Optional[int] = None):
        ttl = ttl_seconds or self._cache_ttl
        now = asyncio.get_running_loop().time()
        if cache_key in self._api_cache:
            cached_data, timestamp = self._api_cache[cache_key]
            if now - timestamp < ttl:
                return cached_data
        data = await fetch_fn()
        self._api_cache[cache_key] = (data, now)
        return data

    # ---------- Binance (async) ----------
    
    async def _fetch_klines_uncached(self, session: aiohttp.ClientSession, symbol, timeframe, limit=200):
        max_retries = 3
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                params = {"symbol": symbol, "interval": timeframe, "limit": limit}
                async with session.get(self.klines_api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as res:
                    if res.status == 429:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limit klines {symbol} {timeframe}. Retry {attempt+1}/{max_retries} em {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    res.raise_for_status()
                    data = await res.json()
                    if not isinstance(data, list):
                        logger.debug(f"Resposta inesperada klines {symbol} {timeframe}: {str(data)[:160]}")
                        return pd.DataFrame()
                    df = pd.DataFrame(data, columns=[
                        'open_time','open','high','low','close','volume',
                        'close_time','qav','num_trades','tbbav','tbqav','ignore'
                    ])
                    for col in ['open','high','low','close','volume']:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
            except aiohttp.ClientError as e:
                logger.error(f"Req klines {symbol} {timeframe} ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Inesperado klines {symbol} {timeframe} ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        logger.error(f"Falha persistente klines {symbol} {timeframe}. Retornando vazio.")
        return pd.DataFrame()

    # TTL por timeframe: TFs curtos expiram mais rápido para RSI não congelar
    _KLINE_TTL = {"5m": 60, "15m": 120, "1h": 120, "4h": 300, "1d": 600, "1w": 900, "1M": 1800}

    async def _fetch_klines(self, session: aiohttp.ClientSession, symbol, timeframe, limit=200):
        cache_key = f"klines_{symbol}_{timeframe}_{limit}"
        ttl = self._KLINE_TTL.get(timeframe, self._cache_ttl)
        return await self._async_fetch_with_cache(cache_key, lambda: self._fetch_klines_uncached(session, symbol, timeframe, limit), ttl_seconds=ttl)

    async def _fetch_symbol_price(self, session: aiohttp.ClientSession, symbol: str) -> float:
        cache_key = f"mark_price_{symbol}"
        async def _do_fetch():
            try:
                async with session.get(self.mark_price_api_url, params={"symbol": symbol}, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    r.raise_for_status()
                    data = await r.json()
                    return float(data.get("markPrice", 0.0))
            except Exception as e:
                logger.debug(f"Falha markPrice {symbol}: {e}")
                return 0.0
        return float(await self._async_fetch_with_cache(cache_key, _do_fetch, ttl_seconds=15) or 0.0)

    # ---------- Tempo ----------
    
    async def _get_binance_server_time_async(self) -> int:
        """Versão async para buscar tempo do servidor Binance."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://fapi.binance.com/fapi/v1/time",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as res:
                        res.raise_for_status()
                        data = await res.json()
                        server_time_ms = data.get("serverTime")
                        if not server_time_ms:
                            raise ValueError("serverTime ausente")
                        now_ms = self.time_manager.now()
                        if abs(server_time_ms - now_ms) > 5000:
                            logger.debug(f"Skew {abs(server_time_ms - now_ms)}ms; retry...")
                            await asyncio.sleep(0.5)
                            continue
                        return server_time_ms
            except Exception as e:
                logger.debug(f"get time falha ({attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(0.5)
        return self.time_manager.now()

    def _get_binance_server_time(self) -> int:
        """Método síncrono que chama versão async."""
        try:
            return asyncio.run(self._get_binance_server_time_async())
        except RuntimeError:
            # Já existe event loop rodando nesta thread (ex.: Jupyter/FastAPI)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(self._get_binance_server_time_async()))
                return int(fut.result(timeout=10))
        except Exception as e:
            logger.debug(f"Erro ao buscar tempo Binance: {e}")
            return self.time_manager.now()

    # ---------- Cálculos técnicos (sync, wrap if needed) ----------
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        if df is None or df.empty:
            return 0.0
        df = df.copy()
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = (df['high'] - df['close'].shift()).abs()
        df['l-pc'] = (df['low'] - df['close'].shift()).abs()
        df['tr'] = df[['h-l','h-pc','l-pc']].max(axis=1)
        atr = df['tr'].ewm(span=period, adjust=False).mean()
        try:
            return float(atr.iloc[-1])
        except Exception:
            return 0.0

    def _classify_regime(self, df: pd.DataFrame, atr_value: float):
        if df.empty:
            return "Desconhecido"
        avg_volume = df["volume"].tail(50).mean() if len(df) >= 50 else df["volume"].mean()
        last_volume = df["volume"].iloc[-1]
        vol_change = (last_volume / avg_volume) if (avg_volume and avg_volume > 0) else 1
        price = float(df["close"].iloc[-1])
        atr_pct = (atr_value / price) if price > 0 else 0
        high_vol = atr_pct > 0.01
        high_volume = vol_change > 1.2
        low_volume = vol_change < 0.8
        if high_vol and low_volume:
            return "Manipulação"
        elif high_vol and high_volume:
            return "Institucional"
        elif not high_vol and low_volume:
            return "Range"
        else:
            return "Acumulação"

    def _calculate_rsi(self, series: pd.Series, period: int) -> float:
        try:
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.ewm(span=period, adjust=False).mean()
            roll_down = down.ewm(span=period, adjust=False).mean()
            rs = roll_up / roll_down
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.debug(f"Falha ao calcular RSI: {e}")
            return 0.0

    def _calculate_macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[float, float]:
        try:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return float(macd.iloc[-1]), float(signal_line.iloc[-1])
        except Exception as e:
            logger.debug(f"Falha ao calcular MACD: {e}")
            return 0.0, 0.0

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> float:
        try:
            if df is None or df.empty or len(df) < period + 1:
                return 0.0
            high = df['high']
            low = df['low']
            close = df['close']
            up_move = high.diff()
            down_move = low.shift() - low
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
            tr1 = (high - low)
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()
            return float(adx.iloc[-1]) if not adx.empty else 0.0
        except Exception as e:
            logger.debug(f"Falha ao calcular ADX: {e}")
            return 0.0

    def _calculate_realized_volatility(self, series: pd.Series) -> float:
        try:
            returns = pd.Series(np.log(series / series.shift())).dropna()
            if returns.empty:
                return 0.0
            return float(returns.std())
        except Exception as e:
            logger.debug(f"Falha ao calcular volatilidade realizada: {e}")
            return 0.0

    # ---------- Contexto de mercado ----------
    
    def _calculate_market_context(self) -> dict:
        try:
            now_ny_iso = self.time_manager.now_ny_iso(timespec="seconds")
            now_ny = datetime.fromisoformat(now_ny_iso)
            dow = now_ny.weekday()
            is_holiday = False  # Cripto 24/7

            hour = now_ny.hour
            if 13 <= hour < 17:
                session = "NY_OVERLAP"
                phase = "ACTIVE"
                close_dt = now_ny.replace(hour=17, minute=0, second=0, microsecond=0)
            elif 17 <= hour < 22:
                session = "NY"
                phase = "ACTIVE"
                close_dt = now_ny.replace(hour=22, minute=0, second=0, microsecond=0)
            elif 22 <= hour or hour < 8:
                session = "ASIA"
                phase = "ACTIVE" if 0 <= hour < 8 else "OFF"
                close_dt = (now_ny + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0) if hour >= 22 else now_ny.replace(hour=8, minute=0, second=0, microsecond=0)
            else:
                session = "EUROPE"
                phase = "ACTIVE"
                close_dt = now_ny.replace(hour=13, minute=0, second=0, microsecond=0)

            time_to_close = max(0, int((close_dt - now_ny).total_seconds()))
             
            return {
                "trading_session": session,
                "session_phase": phase,
                "time_to_session_close": time_to_close,
                "day_of_week": dow,
                "is_holiday": is_holiday,
                "market_hours_type": "EXTENDED"
            }
        except Exception as e:
            logger.debug(f"Falha ao calcular contexto de mercado: {e}")
            return {}

    # ---------- Ambiente de mercado ----------
    
    async def _calculate_market_environment(self, session: aiohttp.ClientSession) -> dict:
        env: Dict[str, Any] = {
            "volatility_regime": None,
            "trend_direction": None,
            "market_structure": None,
            "liquidity_environment": None,
            "risk_sentiment": None,
            "correlation_spy": None,
            "correlation_dxy": None,
            "correlation_gold": None,
        }
        try:
            df_daily = await self._fetch_klines(session, self.symbol, '1d', limit=CORRELATION_LOOKBACK + 10)
            if not df_daily.empty:
                df_daily = df_daily.copy()
                df_daily['close'] = pd.to_numeric(df_daily['close'], errors='coerce')
                vol = self._calculate_realized_volatility(df_daily['close'])
                low_pct, high_pct = VOLATILITY_PERCENTILES
                if vol < low_pct:
                    env["volatility_regime"] = "LOW"
                elif vol > high_pct:
                    env["volatility_regime"] = "HIGH"
                else:
                    env["volatility_regime"] = "NORMAL"
             
            mtf = await self._analyze_mtf_trends(session)
            ups = sum(1 for v in mtf.values() if v.get("tendencia") == "Alta")
            downs = sum(1 for v in mtf.values() if v.get("tendencia") == "Baixa")
            if ups > downs:
                env["trend_direction"] = "UP"
            elif downs > ups:
                env["trend_direction"] = "DOWN"
            else:
                env["trend_direction"] = "SIDEWAYS"
             
            regimes = [v.get("regime") for v in mtf.values()]
            if any(r == "Range" for r in regimes):
                env["market_structure"] = "RANGE_BOUND"
            elif any(r in ("Institucional", "Manipulação") for r in regimes):
                env["market_structure"] = "TRENDING"
            else:
                env["market_structure"] = "ACCUMULATION"
             
            env["liquidity_environment"] = "NORMAL"
             
            if ENABLE_ALPHAVANTAGE:
                try:
                    env["correlation_spy"] = await self._compute_correlation(session, "SP500", EXTERNAL_MARKETS.get("SP500", ""))
                except Exception:
                    pass
                try:
                    env["correlation_dxy"] = await self._compute_correlation(session, "DXY", "DXY")
                except Exception:
                    pass
                try:
                    env["correlation_gold"] = await self._compute_correlation(session, "GOLD", EXTERNAL_MARKETS.get("GOLD", ""))
                except Exception:
                    pass
             
            # FIX 5C: risk_sentiment must be consistent with Fear&Greed + VIX.
            # Correlation-based sentiment is overridden when macro data is extreme.
            corr_spy = env.get("correlation_spy")
            corr_dxy = env.get("correlation_dxy")
            if corr_spy is not None and corr_dxy is not None:
                if corr_spy > 0 and corr_dxy < 0:
                    env["risk_sentiment"] = "BULLISH"
                elif corr_spy < 0 and corr_dxy > 0:
                    env["risk_sentiment"] = "BEARISH"
                else:
                    env["risk_sentiment"] = "NEUTRAL"
            else:
                env["risk_sentiment"] = "NEUTRAL"
        except Exception as e:
            logger.debug(f"Falha ao calcular ambiente de mercado: {e}")
        return env

    async def _compute_correlation(
        self,
        session: aiohttp.ClientSession,
        name: str,
        ticker: str
    ) -> Optional[float]:
        """
        Calcula correlação entre BTCUSDT (Binance) e um ativo externo (DXY, SP500, GOLD, etc).

        Melhorias:
        - Usa retornos diários
        - Alinha as séries por posição (tamanho) em vez de timestamp, evitando NaN por falta de
          interseção perfeita de datas (BTC 24/7 vs índices apenas dias úteis).
        - Exige um número mínimo de pontos para ter correlação minimamente confiável.
        """
        try:
            if not ticker:
                return None

            # 1) Série principal (BTCUSDT) via Binance
            sym_df = await self._fetch_klines(
                session,
                self.symbol,
                '1d',
                limit=CORRELATION_LOOKBACK + 50
            )
            if sym_df.empty:
                return None

            sym_series = pd.to_numeric(sym_df['close'], errors='coerce').dropna()
            sym_returns = sym_series.pct_change().dropna()

            # 2) Série externa via yFinance
            #    Aqui usamos o "name" se ele estiver no TICKER_MAPPING (DXY, SP500, GOLD, etc.),
            #    senão usamos o próprio "ticker" vindo da config.
            symbol_key = name if name in TICKER_MAPPING else ticker
            ext_hist = await self._yfinance_history(
                symbol_key,
                period="180d",
                interval=YFINANCE_INTERVAL
            )
            if ext_hist.empty or 'close' not in ext_hist.columns:
                return None

            ext_series = pd.to_numeric(ext_hist['close'], errors='coerce').dropna()
            ext_returns = ext_series.pct_change().dropna()

            # 3) Alinhar por posição, não por timestamp
            max_len = min(len(sym_returns), len(ext_returns), CORRELATION_LOOKBACK)

            # Critério de qualidade: exigimos um número mínimo de pontos
            # para que a correlação não seja baseada em 2–3 candles apenas.
            MIN_POINTS = 10  # você pode ajustar para 5 ou expor isso em config.py
            if max_len < MIN_POINTS:
                # Muito pouca amostra para correlação estável
                logger.debug(
                    f"Correlação {name} ignorada: apenas {max_len} pontos (min={MIN_POINTS})"
                )
                return None

            sym_aligned = sym_returns.tail(max_len).reset_index(drop=True)
            ext_aligned = ext_returns.tail(max_len).reset_index(drop=True)

            corr = sym_aligned.corr(ext_aligned)
            if pd.isna(corr):
                return None

            corr_val = float(round(corr, 4))
            logger.debug(f"Correlação {name}: {corr_val} com n={max_len} pontos")
            return corr_val

        except Exception as e:
            logger.debug(f"Erro ao calcular correlação {name}: {e}")
            return None

    async def _analyze_mtf_trends(self, session: aiohttp.ClientSession):
        mtf_context = {}
        for tf in self.timeframes:
            limit_needed = max(self.ema_period, self.atr_period) * 3 + 20
            df = await self._fetch_klines(session, self.symbol, tf, limit=limit_needed)
            if not df.empty:
                df = df.copy()
                df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
                last_close = float(df['close'].iloc[-1])
                last_ema = float(df['ema'].iloc[-1])
                tendencia = "Alta" if last_close > last_ema else "Baixa"
                 
                rsi_short = self._calculate_rsi(df['close'], RSI_PERIODS.get('short', 14))
                rsi_long = self._calculate_rsi(df['close'], RSI_PERIODS.get('long', 21))
                macd_val, macd_sig = self._calculate_macd(
                    df['close'],
                    fast=MACD_FAST_PERIOD,
                    slow=MACD_SLOW_PERIOD,
                    signal=MACD_SIGNAL_PERIOD,
                )
                adx_val = self._calculate_adx(df[['high', 'low', 'close']], ADX_PERIOD)
                realized_vol = self._calculate_realized_volatility(df['close'])
                atr = self._calculate_atr(df, self.atr_period)
                regime = self._classify_regime(df, atr)
                 
                mtf_context[tf] = {
                    "tendencia": tendencia,
                    "preco_atual": last_close,
                    f"mme_{self.ema_period}": round(last_ema, 2),
                    "atr": round(atr, 2),
                    "regime": regime,
                    "rsi_short": round(rsi_short, 2) if rsi_short else 0.0,
                    "rsi_long": round(rsi_long, 2) if rsi_long else 0.0,
                    "macd": round(macd_val, 4) if macd_val else 0.0,
                    "macd_signal": round(macd_sig, 4) if macd_sig else 0.0,
                    "adx": round(adx_val, 2) if adx_val else 0.0,
                    "realized_vol": round(realized_vol, 4) if realized_vol else 0.0,
                }
        return mtf_context

    # ---------- Intermarket / External ----------
    
    async def _fetch_intermarket_data(self, session: aiohttp.ClientSession):
        data = {}
        
        # Binance intermarket (mantém como está)
        for sym in self.intermarket_symbols:
            df = await self._fetch_klines(session, sym, '5m', limit=2)
            if not df.empty:
                last, prev = float(df['close'].iloc[-1]), float(df['close'].iloc[-2])
                data[sym] = {"preco_atual": last, "movimento": "Alta" if last > prev else "Baixa"}
        
        if not ENABLE_ALPHAVANTAGE:
            return data
        
        # 🆕 Buscar DXY via FRED (fallback do yfinance) + cache
        try:
            dxy_result = None

            # Tentativa 1: FRED ou yfinance
            if self.fred_fetcher.is_failing("DXY"):
                logger.debug("FRED: DXY em modo fallback, usando yfinance diretamente")
                hist = await self._yfinance_history("DXY", period="5d", interval="1d")
                if not hist.empty:
                    last = float(hist["close"].iloc[-1])
                    dxy_result = {
                        "preco_atual": round(last, 2),
                        "movimento": "Neutro",
                        "source": "yfinance",
                        "ticker": "DX-Y.NYB"
                    }
                    logger.info(f"✅ DXY obtido via yfinance (fallback): {last:.2f}")
            else:
                dxy_value = await self.fred_fetcher.fetch_latest_value("DXY", session)

                if dxy_value is not None:
                    dxy_result = {
                        "preco_atual": round(dxy_value, 2),
                        "movimento": "Neutro",
                        "source": "FRED",
                        "ticker": "DTWEXBGS"
                    }
                    logger.info(f"✅ DXY obtido via FRED: {dxy_value:.2f}")
                else:
                    # Fallback para yfinance se FRED falhar
                    logger.debug("DXY via FRED falhou, tentando yfinance...")
                    record_fred_fallback()
                    hist = await self._yfinance_history("DXY", period="5d", interval="1d")
                    if not hist.empty:
                        last = float(hist["close"].iloc[-1])
                        dxy_result = {
                            "preco_atual": round(last, 2),
                            "movimento": "Neutro",
                            "source": "yfinance",
                            "ticker": "DX-Y.NYB"
                        }
                        logger.info(f"✅ DXY obtido via yfinance: {last:.2f}")

            # Salvar em cache se obtivemos resultado
            if dxy_result:
                data["DXY"] = dxy_result
                self._dxy_cache = dxy_result.copy()
                self._dxy_cache_time = time.time()
            elif self._dxy_cache and (time.time() - self._dxy_cache_time) < 3600:
                # Tentativa 2: Cache recente (< 1 hora)
                cached = self._dxy_cache.copy()
                age_min = round((time.time() - self._dxy_cache_time) / 60, 1)
                cached["source"] = f"{cached.get('source', 'unknown')}_cached"
                cached["cache_age_min"] = age_min
                data["DXY"] = cached
                logger.info(f"✅ DXY obtido via cache ({age_min}min): {cached.get('preco_atual')}")
            else:
                logger.warning("⚠️ DXY indisponível: FRED, yfinance e cache falharam")
        except Exception as e:
            # Em caso de exceção, tentar cache antes de desistir
            if self._dxy_cache and (time.time() - self._dxy_cache_time) < 3600:
                cached = self._dxy_cache.copy()
                cached["source"] = f"{cached.get('source', 'unknown')}_cached"
                cached["cache_age_min"] = round((time.time() - self._dxy_cache_time) / 60, 1)
                data["DXY"] = cached
                logger.info(f"✅ DXY via cache após erro: {cached.get('preco_atual')}")
            else:
                logger.warning(f"⚠️ Erro ao buscar DXY: {e}")
        
        return data

    async def _fetch_external_markets(self, session: aiohttp.ClientSession):
        """Busca dados externos com FRED como fallback."""
        ext_data: Dict[str, Any] = {}

        if not ENABLE_ALPHAVANTAGE:
            logger.info("External markets desabilitados")
            return ext_data
        
        # Símbolos que o FRED pode fornecer
        fred_symbols = ["DXY", "TNX"]
        
        # Símbolos que precisam do yfinance (inclui VIX)
        yfinance_symbols = {
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "GOLD": "GC=F",
            "WTI": "CL=F",
            "VIX": "^VIX",
        }

        # 1. Buscar via FRED primeiro (mais confiável para indicadores econômicos)
        for symbol in fred_symbols:
            try:
                # 🆕 Verificar se FRED está em modo fallback para este símbolo
                if self.fred_fetcher.is_failing(symbol):
                    logger.debug(f"FRED: {symbol} em modo fallback, pulando")
                    continue

                value = await self.fred_fetcher.fetch_latest_value(symbol, session)

                if value is not None:
                    ext_data[symbol] = {
                        "preco_atual": round(value, 4),
                        "movimento": "Neutro",
                        "source": "FRED",
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"✅ {symbol}: {value:.4f} (FRED)")
                else:
                    logger.debug(f"⚠️ {symbol} não disponível via FRED")
            except Exception as e:
                logger.warning(f"⚠️ Erro FRED para {symbol}: {e}")

        # 1b. Fallback yfinance para DXY se FRED falhou
        if "DXY" not in ext_data:
            try:
                hist = await self._yfinance_history("DXY", period="5d", interval="1d")
                if not hist.empty:
                    last = float(hist["close"].iloc[-1])
                    prev = float(hist["close"].iloc[-2]) if len(hist) > 1 else last
                    ext_data["DXY"] = {
                        "preco_atual": round(last, 4),
                        "movimento": "Alta" if last > prev else "Baixa" if last < prev else "Neutro",
                        "source": "yfinance",
                        "ticker": "DX-Y.NYB",
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"DXY: {last:.4f} (yfinance fallback)")
            except Exception as e:
                logger.warning(f"DXY yfinance fallback falhou: {e}")

        # 2. Buscar via yfinance (índices, commodities e VIX)
        for name, ticker in yfinance_symbols.items():
            try:
                hist = await self._yfinance_history(name, period="5d", interval="1d")

                if not hist.empty:
                    last = float(hist["close"].iloc[-1])
                    prev = float(hist["close"].iloc[-2]) if len(hist) > 1 else last

                    ext_data[name] = {
                        "preco_atual": round(last, 2),
                        "movimento": "Alta" if last > prev else "Baixa" if last < prev else "Neutro",
                        "source": "yfinance",
                        "ticker": ticker,
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"✅ {name}: ${last:.2f} (yfinance)")
                else:
                    logger.debug(f"⚠️ {name} ({ticker}) vazio via yfinance")

            except Exception as e:
                logger.warning(f"⚠️ Erro yfinance para {name}: {e}")
                continue

        # 3. Fear & Greed Index via alternative.me (gratuito, sem API key)
        try:
            async with session.get(
                "https://api.alternative.me/fng/?limit=2&format=json",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status == 200:
                    fng_data = await resp.json()
                    entries = fng_data.get("data", [])
                    if entries:
                        current = entries[0]
                        prev_entry = entries[1] if len(entries) > 1 else current
                        fng_value = int(current.get("value", 50))
                        fng_prev = int(prev_entry.get("value", fng_value))
                        ext_data["FEAR_GREED"] = {
                            "preco_atual": fng_value,
                            "prev": fng_prev,
                            "movimento": "Alta" if fng_value > fng_prev else "Baixa" if fng_value < fng_prev else "Neutro",
                            "classification": current.get("value_classification", "Unknown"),
                            "source": "alternative.me",
                            "timestamp": datetime.now().isoformat()
                        }
                        logger.info(f"✅ Fear&Greed: {fng_value} ({current.get('value_classification', '')})")
                else:
                    logger.debug(f"⚠️ Fear&Greed API status {resp.status}")
        except Exception as e:
            logger.debug(f"Fear&Greed indisponível: {e}")

        return ext_data

    # ---------- Derivativos ----------
    
    async def _fetch_liquidations_data(self, session: aiohttp.ClientSession, symbol, lookback_minutes=5):
        max_retries = 3
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                end_ms = self.time_manager.now() - 1000
                start_ms = end_ms - int(lookback_minutes) * 60_000
                now_ms = self.time_manager.now()
                if start_ms < (now_ms - 24 * 60 * 60 * 1000):
                    start_ms = now_ms - 24 * 60 * 60 * 1000
                if end_ms > now_ms + 5_000:
                    end_ms = now_ms
                params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}
                async with session.get(self.liquidations_api_url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 204 or not await r.text():
                        return []
                    if r.status == 200:
                        return await r.json()
                    logger.debug(f"ForceOrders {r.status}: {(await r.text())[:200]}")
                    if r.status == 429 and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit liquidations. Retry {attempt+1}/{max_retries} em {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    return []
            except aiohttp.ClientError as e:
                logger.error(f"Req liquidations ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logger.debug(f"ForceOrders exception ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
        logger.error(f"Falha persistente liquidations {symbol}. Retornando [].")
        return []

    def _build_liquidation_heatmap(self, liq_data):
        heatmap = {}
        if not liq_data:
            return heatmap
        for liq in liq_data:
            try:
                price = float(liq["price"])
                qty = float(liq["origQty"])
                side = liq.get("side", "")
                bucket = round(price / LIQUIDATION_MAP_DEPTH) * LIQUIDATION_MAP_DEPTH
                if bucket not in heatmap:
                    heatmap[bucket] = {"longs": 0.0, "shorts": 0.0}
                usd = price * qty
                if side == "SELL":
                    heatmap[bucket]["longs"] += usd
                elif side == "BUY":
                    heatmap[bucket]["shorts"] += usd
            except Exception:
                continue
        return {str(k): v for k, v in sorted(heatmap.items())}

    async def _fetch_derivatives_data(self, session: aiohttp.ClientSession):
        derivatives_data = {}
        for sym in self.derivatives_symbols:
            try:
                async with session.get(self.funding_api_url, params={'symbol': sym, 'limit': 1}, timeout=aiohttp.ClientTimeout(total=5)) as fr:
                    fr.raise_for_status()
                    fr_data = await fr.json()
                    funding_rate = float(fr_data[-1]["fundingRate"]) * 100 if fr_data else 0.0
                async with session.get(self.open_interest_api_url, params={'symbol': sym}, timeout=aiohttp.ClientTimeout(total=5)) as oi:
                    oi.raise_for_status()
                    oi_data = await oi.json()
                    open_interest = float(oi_data.get("openInterest", 0))
                async with session.get(
                    self.long_short_ratio_api_url,
                    params={'symbol': sym, 'period': '5m', 'limit': 1}, timeout=aiohttp.ClientTimeout(total=5)
                ) as ls:
                    ls.raise_for_status()
                    ls_data = await ls.json()
                    long_short_ratio = float(ls_data[0]["longShortRatio"]) if ls_data else 0.0
                liq = await self._fetch_liquidations_data(session, sym, lookback_minutes=int(self.update_interval / 60))
                heatmap = self._build_liquidation_heatmap(liq)
                totals = {
                    "longs_usd": sum(v["longs"] for v in heatmap.values()),
                    "shorts_usd": sum(v["shorts"] for v in heatmap.values())
                }
                price = await self._fetch_symbol_price(session, sym)
                open_interest_usd = open_interest * price if price else open_interest
                derivatives_data[sym] = {
                    "funding_rate_percent": round(funding_rate, 4),
                    "open_interest": open_interest,
                    "open_interest_usd": round(open_interest_usd, 2) if open_interest_usd else 0.0,
                    "long_short_ratio": long_short_ratio,
                    "liquidation_heatmap": heatmap,
                    **totals
                }
            except Exception as e:
                logger.debug(f"Erro derivativos {sym}: {e}")
        return derivatives_data

    # ---------- Pivots (Calculados) ----------

    async def _calculate_pivots(self, session: aiohttp.ClientSession) -> dict:
        """Calcula Pivot Points Clássicos (D/W/M) usando klines históricos."""
        pivots: Dict[str, Any] = {"daily": {}, "weekly": {}, "monthly": {}}
        try:
            # Daily (últimos 5 dias para garantir)
            df_d = await self._fetch_klines(session, self.symbol, '1d', limit=5)
            if not df_d.empty:
               # daily_pivot usa o último dia completo
               pivots["daily"] = daily_pivot(df_d)

            # Weekly
            df_w = await self._fetch_klines(session, self.symbol, '1w', limit=5)
            if not df_w.empty:
               pivots["weekly"] = weekly_pivot(df_w)
            
            # Monthly
            df_m = await self._fetch_klines(session, self.symbol, '1M', limit=5)
            if not df_m.empty:
               pivots["monthly"] = monthly_pivot(df_m)
               
        except Exception as e:
            logger.debug(f"Falha calculo pivots: {e}")
            
        return pivots

    # ---------- On-chain / Sentimento ----------
    
    async def _fetch_onchain_sentiment(self, session: Optional[aiohttp.ClientSession] = None):
        sentiment: Dict[str, Any] = {"onchain": {}, "funding_agg": {}}

        # On-chain REAL (blockchain.info + mempool.space - gratuito)
        if self._onchain_fetcher:
            try:
                sentiment["onchain"] = await self._onchain_fetcher.fetch_all(session)
            except Exception as e:
                logger.debug(f"Dados on-chain reais indisponíveis: {e}")
                sentiment["onchain"] = {"status": "fetch_error", "is_real_data": False}
        else:
            sentiment["onchain"] = {"status": "fetcher_not_available", "is_real_data": False}

        # Funding agregado REAL (Binance premiumIndex - gratuito)
        if self._funding_aggregator:
            try:
                sentiment["funding_agg"] = await self._funding_aggregator.fetch_all(session)
            except Exception as e:
                logger.debug(f"Funding agregado indisponível: {e}")
                sentiment["funding_agg"] = {"status": "fetch_error", "is_real_data": False}
        else:
            sentiment["funding_agg"] = {"status": "fetcher_not_available", "is_real_data": False}

        return sentiment

    # ---------- Consolidação ----------
    
    async def _async_build_full_context(self, session: aiohttp.ClientSession):
        return {
            "mtf": await self._analyze_mtf_trends(session),
            "intermarket": await self._fetch_intermarket_data(session),
            "external": await self._fetch_external_markets(session),
            "derivatives": await self._fetch_derivatives_data(session),
            "sentiment": await self._fetch_onchain_sentiment(session),
            "profile": await asyncio.to_thread(self.historical_profiler.update_profiles),
            "market_context": self._calculate_market_context(),
            "market_environment": await self._calculate_market_environment(session),
            "pivots": await self._calculate_pivots(session),
            "timestamp": self.time_manager.now_iso(),
        }

    @staticmethod
    def _adjust_risk_sentiment(ctx: dict) -> None:
        """FIX 5C: Override risk_sentiment when Fear&Greed / VIX are extreme."""
        env = ctx.get("market_environment")
        ext = ctx.get("external")
        if not isinstance(env, dict) or not isinstance(ext, dict):
            return

        fg_data = ext.get("FEAR_GREED", {}) or {}
        fg = fg_data.get("preco_atual")  # 0-100 scale
        vix_data = ext.get("VIX", {}) or {}
        vix = vix_data.get("preco_atual")

        # Fear & Greed extreme values dominate
        if fg is not None:
            try:
                fg = float(fg)
                if fg <= 20:
                    env["risk_sentiment"] = "RISK_OFF"
                    return
                if fg >= 80:
                    env["risk_sentiment"] = "RISK_ON"
                    return
            except (TypeError, ValueError):
                pass

        # High VIX = risk off
        if vix is not None:
            try:
                vix = float(vix)
                if vix > 25:
                    env["risk_sentiment"] = "RISK_OFF"
                    return
                if vix < 15:
                    env["risk_sentiment"] = "RISK_ON"
                    return
            except (TypeError, ValueError):
                pass

        # Otherwise keep correlation-based value (already set)

    @staticmethod
    def _validate_historical_vp(ctx: dict) -> None:
        """FIX 6A: Mark degenerate VP periods as insufficient_data."""
        hvp = ctx.get("historical_vp")
        if not isinstance(hvp, dict):
            return
        for period in ("daily", "weekly", "monthly"):
            vp = hvp.get(period)
            if not isinstance(vp, dict):
                continue
            poc = vp.get("poc", 0)
            vah = vp.get("vah", 0)
            val = vp.get("val", 0)
            if poc and vah and val and poc > 0:
                # Range < 0.01% of POC → degenerate (e.g. POC=VAH=VAL)
                if abs(vah - val) / poc < 0.0001:
                    vp["status"] = "insufficient_data"
                    vp["quality"] = "degenerate"

    async def _async_update_loop(self):
        logger.info("✅ Coletor de Contexto iniciado (async).")
        async with await self._build_retrying_session() as session:
            while True:
                try:
                    ctx = await self._async_build_full_context(session)
                    # Adapt to v2.1.0 keys
                    final_ctx = ctx
                    final_ctx["mtf_trends"] = final_ctx.get("mtf", {})
                    final_ctx["historical_vp"] = final_ctx.get("profile", {})
                    final_ctx.setdefault("intermarket", {})
                    final_ctx.setdefault("sentiment", {})

                    # FIX 5C: Post-process risk_sentiment using Fear&Greed + VIX
                    # (external and market_environment are gathered independently)
                    self._adjust_risk_sentiment(final_ctx)

                    # FIX 6A: Mark degenerate historical_vp as insufficient_data
                    self._validate_historical_vp(final_ctx)

                    self._context_data = final_ctx
                    logger.info("✅ Contexto Macro atualizado.")
                except Exception as e:
                    logger.error(f"❌ Erro crítico loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)

    async def _async_mtf_fast_loop(self):
        """Atualiza apenas mtf_trends a cada 120s (RSI, ATR, etc.)."""
        MTF_REFRESH_INTERVAL = 120  # 2 minutos
        await asyncio.sleep(MTF_REFRESH_INTERVAL)  # skip first (full context já fez)
        async with await self._build_retrying_session() as session:
            while True:
                try:
                    fresh_mtf = await self._analyze_mtf_trends(session)
                    if fresh_mtf and self._context_data:
                        self._context_data["mtf"] = fresh_mtf
                        self._context_data["mtf_trends"] = fresh_mtf
                        logger.debug("🔄 MTF trends atualizados (fast loop)")
                except Exception as e:
                    logger.debug(f"MTF fast loop erro: {e}")
                await asyncio.sleep(MTF_REFRESH_INTERVAL)

    async def _async_start(self):
        loop = asyncio.get_running_loop()
        if self._yfinance_lock is None:
            self._yfinance_lock = asyncio.Lock()
        self._update_task = loop.create_task(self._async_update_loop())
        self._mtf_fast_task = loop.create_task(self._async_mtf_fast_loop())

    async def _async_stop(self):
        for task in (self._update_task, getattr(self, '_mtf_fast_task', None)):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Fechar sessão aiohttp se existir
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Sessão aiohttp fechada")

    async def _async_get_context(self):
        return self._context_data.copy()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
            self._loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop=self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    def stop(self):
        if not self._loop:
            return
        
        # Verificar se o loop está rodando antes de tentar parar
        if not self._loop.is_running():
            logger.debug("Loop já não está rodando")
            return
            
        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
            fut.result(timeout=5)
        except Exception as e:
            logger.warning(f"Erro ao parar ContextCollector: {e}")
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception as e:
                logger.debug(f"Erro ao parar loop: {e}")
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            
            # Shutdown thread pool executor
            self._executor.shutdown(wait=False)

    def get_context(self) -> dict:
        if not self._loop:
            return {}
        fut = asyncio.run_coroutine_threadsafe(self._async_get_context(), self._loop)
        return fut.result(timeout=2) or {}