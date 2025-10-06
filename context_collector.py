import requests
import pandas as pd
import logging
import time
import threading
import os
import random  # ✅ Adicionado
import numpy as np  # ✅ Para cálculos numéricos
# Tipos para anotações
from typing import Tuple
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import (
    CONTEXT_TIMEFRAMES, CONTEXT_EMA_PERIOD, CONTEXT_ATR_PERIOD,
    CONTEXT_UPDATE_INTERVAL_SECONDS, INTERMARKET_SYMBOLS, DERIVATIVES_SYMBOLS,
    VP_NUM_DAYS_HISTORY, VP_VALUE_AREA_PERCENT, LIQUIDATION_MAP_DEPTH,
    EXTERNAL_MARKETS, ENABLE_ONCHAIN, ONCHAIN_PROVIDERS, STABLECOIN_FLOW_TRACKING
)
# Opcional: se quiser poder desligar Alpha Vantage rapidamente, adicione em config.py:
# ENABLE_ALPHAVANTAGE = True
try:
    from config import ENABLE_ALPHAVANTAGE
except Exception:
    ENABLE_ALPHAVANTAGE = True
# Novos parâmetros de configuração para regime, indicadores e correlações
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
except Exception:
    # Fallbacks caso não existam (defina-os em config.py para personalizar)
    CORRELATION_LOOKBACK = 50
    VOLATILITY_PERCENTILES = (0.35, 0.65)
    ADX_PERIOD = 14
    RSI_PERIODS = {"short": 14, "long": 21}
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
from historical_profiler import HistoricalVolumeProfiler
from time_manager import TimeManager

class ContextCollector:
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
        self.context_data = {}
        self._lock = threading.Lock()
        self.should_stop = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.time_manager = TimeManager()
        # Cache simples
        self._api_cache = {}
        self._cache_ttl = 60  # s
        # ---------- Alpha Vantage ----------
        self.alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY", "KC4IE0MBOEXK88Y3")
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        # Fallbacks por mercado
        self._dxy_candidates = ["DXY"]  # Alpha Vantage usa "DXY" para ECONOMIC_INDICATORS
        self._fallback_map = {
            "S&P500": ["GSPC", "SPY"],
            "NASDAQ": ["IXIC", "QQQ"],
            "UST10Y": ["TNX", "TYX", "FVX"],
            "GOLD": ["GC", "XAUUSD"],
            "WTI": ["CL", "BZ"],
        }

    # ---------- Helpers Alpha Vantage ----------
    def _build_retrying_session(self) -> requests.Session:
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        return sess

    def _alpha_vantage_history(self, symbol: str, function: str = "ECONOMIC_INDICATORS", interval: str = "1min"):
        """
        Wrapper resiliente para Alpha Vantage
        - Usa sessão com retries
        - 2 tentativas com pequeno backoff
        """
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
                    params["outputsize"] = "compact"  # ou "full"
                elif function == "ECONOMIC_INDICATORS":
                    # Para DXY, não precisa de interval
                    pass
                elif function == "FX_DAILY":
                    params["from_symbol"] = symbol.split("/")[0]
                    params["to_symbol"] = symbol.split("/")[1]
                res = requests.get(self.alpha_vantage_url, params=params, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if "Error Message" in data:
                        logging.debug(f"Alpha Vantage erro {symbol}: {data['Error Message']}")
                        return pd.DataFrame()
                    if "Time Series (1min)" in data:
                        df = pd.DataFrame(data["Time Series (1min)"]).T
                        df.index = pd.to_datetime(df.index)
                        df.columns = ["open", "high", "low", "close", "volume"]
                        df = df.astype(float)
                        return df
                    elif "Time Series (Daily)" in data:
                        df = pd.DataFrame(data["Time Series (Daily)"]).T
                        df.index = pd.to_datetime(df.index)
                        df.columns = ["open", "high", "low", "close", "volume"]
                        df = df.astype(float)
                        return df
                    elif "Time Series (Monthly)" in data:
                        df = pd.DataFrame(data["Time Series (Monthly)"]).T
                        df.index = pd.to_datetime(df.index)
                        df.columns = ["open", "high", "low", "close", "volume"]
                        df = df.astype(float)
                        return df
                    elif "Data" in data:
                        # Para ECONOMIC_INDICATORS
                        df = pd.DataFrame(data["Data"]).T
                        df.index = pd.to_datetime(df.index)
                        df.columns = ["value"]
                        df = df.astype(float)
                        return df
                    elif "Time Series FX (Daily)" in data:
                        df = pd.DataFrame(data["Time Series FX (Daily)"]).T
                        df.index = pd.to_datetime(df.index)
                        df.columns = ["open", "high", "low", "close"]
                        df = df.astype(float)
                        return df
                    else:
                        logging.debug(f"Resposta Alpha Vantage inesperada {symbol}: {str(data)[:160]}")
                        return pd.DataFrame()
                else:
                    logging.debug(f"Alpha Vantage status {res.status_code} para {symbol}: {res.text[:160]}")
            except Exception as e:
                logging.debug(f"Alpha Vantage erro {symbol} (tentativa {i+1}/{try_count}): {e}")
            # pequeno backoff com jitter
            if i < try_count - 1:
                time.sleep(0.6 + random.uniform(0, 0.5))
        return pd.DataFrame()

    # ---------- Cache genérico ----------
    def _fetch_with_cache(self, cache_key: str, fetch_fn, ttl_seconds: int = None):
        ttl = ttl_seconds or self._cache_ttl
        now = time.time()
        if cache_key in self._api_cache:
            cached_data, timestamp = self._api_cache[cache_key]
            if now - timestamp < ttl:
                return cached_data
        data = fetch_fn()
        self._api_cache[cache_key] = (data, now)
        return data

    # ---------- Binance ----------
    def _fetch_klines_uncached(self, symbol, timeframe, limit=200):
        max_retries = 3
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                params = {"symbol": symbol, "interval": timeframe, "limit": limit}
                res = requests.get(self.klines_api_url, params=params, timeout=10)
                if res.status_code == 429:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(
                        f"Rate limit klines {symbol} {timeframe}. Retry {attempt+1}/{max_retries} em {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                res.raise_for_status()
                data = res.json()
                if not isinstance(data, list):
                    logging.debug(f"Resposta inesperada klines {symbol} {timeframe}: {str(data)[:160]}")
                    return pd.DataFrame()
                df = pd.DataFrame(data, columns=[
                    'open_time','open','high','low','close','volume',
                    'close_time','qav','num_trades','tbbav','tbqav','ignore'
                ])
                for col in ['open','high','low','close','volume']:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
            except requests.exceptions.RequestException as e:
                logging.error(f"Req klines {symbol} {timeframe} ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logging.error(f"Inesperado klines {symbol} {timeframe} ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        logging.error(f"Falha persistente klines {symbol} {timeframe}. Retornando vazio.")
        return pd.DataFrame()

    def _fetch_klines(self, symbol, timeframe, limit=200):
        cache_key = f"klines_{symbol}_{timeframe}_{limit}"
        return self._fetch_with_cache(cache_key, lambda: self._fetch_klines_uncached(symbol, timeframe, limit))

    def _fetch_symbol_price(self, symbol: str) -> float:
        cache_key = f"mark_price_{symbol}"
        def _do_fetch():
            try:
                r = requests.get(self.mark_price_api_url, params={"symbol": symbol}, timeout=5)
                r.raise_for_status()
                data = r.json()
                return float(data.get("markPrice", 0.0))
            except Exception as e:
                logging.debug(f"Falha markPrice {symbol}: {e}")
                return 0.0
        return float(self._fetch_with_cache(cache_key, _do_fetch, ttl_seconds=15) or 0.0)

    # ---------- Tempo ----------
    def _get_binance_server_time(self) -> int:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = requests.get("https://fapi.binance.com/fapi/v1/time", timeout=5)
                res.raise_for_status()
                server_time_ms = res.json().get("serverTime")
                if not server_time_ms:
                    raise ValueError("serverTime ausente")
                now_ms = self.time_manager.now()
                if abs(server_time_ms - now_ms) > 5000:
                    logging.debug(f"Skew {abs(server_time_ms - now_ms)}ms; retry...")
                    time.sleep(0.5)
                    continue
                return server_time_ms
            except Exception as e:
                logging.debug(f"get time falha ({attempt+1}/{max_retries}): {e}")
                time.sleep(0.5)
        return self.time_manager.now()

    # ---------- Cálculos técnicos ----------
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

    # ---------- Cálculos de indicadores técnicos (RSI, MACD, ADX) ----------
    def _calculate_rsi(self, series: pd.Series, period: int) -> float:
        """
        Calcula o Relative Strength Index (RSI) com suavização exponencial.
        Retorna o valor do RSI mais recente ou 0.0 se não for possível.
        Fórmula: RSI = 100 - 100 / (1 + RS), onde RS = média ganhos / média perdas.
        """
        try:
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            # médias exponenciais
            roll_up = up.ewm(span=period, adjust=False).mean()
            roll_down = down.ewm(span=period, adjust=False).mean()
            rs = roll_up / roll_down
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception as e:
            logging.debug(f"Falha ao calcular RSI: {e}")
            return 0.0

    def _calculate_macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[float, float]:
        """
        Calcula o MACD (diferença das EMAs) e a linha de sinal. Retorna tupla (macd, sinal).
        """
        try:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return float(macd.iloc[-1]), float(signal_line.iloc[-1])
        except Exception as e:
            logging.debug(f"Falha ao calcular MACD: {e}")
            return 0.0, 0.0

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> float:
        """
        Calcula o Average Directional Movement Index (ADX) simplificado.
        Baseado na fórmula clássica: utiliza True Range e directional movements.
        """
        try:
            if df is None or df.empty or len(df) < period + 1:
                return 0.0
            high = df['high']
            low = df['low']
            close = df['close']
            # Movimentos direcionais
            up_move = high.diff()
            down_move = low.shift() - low
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
            # True Range
            tr1 = (high - low)
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # Suavizações exponenciais
            atr = tr.ewm(span=period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()
            return float(adx.iloc[-1]) if not adx.empty else 0.0
        except Exception as e:
            logging.debug(f"Falha ao calcular ADX: {e}")
            return 0.0

    def _calculate_realized_volatility(self, series: pd.Series) -> float:
        """
        Calcula a volatilidade realizada (desvio padrão dos retornos logarítmicos).
        """
        try:
            returns = np.log(series / series.shift()).dropna()
            if returns.empty:
                return 0.0
            # Volatilidade anualizada não é necessária; retornamos stdev simples
            return float(returns.std())
        except Exception as e:
            logging.debug(f"Falha ao calcular volatilidade realizada: {e}")
            return 0.0

    # ---------- Contexto de mercado (sessão, horários) ----------
    def _calculate_market_context(self) -> dict:
        """
        Define contexto de sessão e informações temporais (NY) para o ativo.
        Retorna um dicionário com:
        - trading_session: rótulo da sessão (ex.: NY_OVERLAP, ASIA, EUROPE)
        - session_phase: fase (PRE, ACTIVE, CLOSE, OFF)
        - time_to_session_close: segundos até o fechamento da sessão principal
        - day_of_week: dia da semana (0=segunda)
        - is_holiday: boolean indicando fim de semana/feriado
        - market_hours_type: tipo de negociação (EXTENDED ou REGULAR)
        """
        try:
            # Usa horário de Nova York para sessão (a maioria das análises institucionais se baseia no fuso NY)
            now_ny_iso = self.time_manager.now_ny_iso(timespec="seconds")
            now_ny = datetime.fromisoformat(now_ny_iso)
            dow = now_ny.weekday()
            # ✅ CORREÇÃO: Cripto opera 24/7 — não há feriados nem fins de semana relevantes
            is_holiday = False

            # Sessões simplificadas
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

            # Tempo até o fechamento da sessão, em segundos
            time_to_close = max(0, int((close_dt - now_ny).total_seconds()))
            context = {
                "trading_session": session,
                "session_phase": phase,
                "time_to_session_close": time_to_close,
                "day_of_week": dow,
                "is_holiday": is_holiday,  # ✅ Corrigido: sempre False para cripto
                "market_hours_type": "EXTENDED"  # Cripto negocia 24/7
            }
            return context
        except Exception as e:
            logging.debug(f"Falha ao calcular contexto de mercado: {e}")
            return {}

    # ---------- Ambiente de mercado (volatilidade, correlação, tendência) ----------
    def _calculate_market_environment(self) -> dict:
        """
        Calcula indicadores agregados de ambiente de mercado, incluindo volatilidade
        realizada, regime de volatilidade, direção de tendência, estrutura de
        mercado, correlações com SP500/DXY/GOLD e sentimento de risco.
        """
        env = {
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
            # Realized volatility usando dados diários
            df_daily = self._fetch_klines(self.symbol, '1d', limit=CORRELATION_LOOKBACK + 10)
            if not df_daily.empty:
                df_daily = df_daily.copy()
                df_daily['close'] = pd.to_numeric(df_daily['close'], errors='coerce')
                vol = self._calculate_realized_volatility(df_daily['close'])
                # Classificar volatilidade comparando com percentis fornecidos
                low_pct, high_pct = VOLATILITY_PERCENTILES
                if vol < low_pct:
                    env["volatility_regime"] = "LOW"
                elif vol > high_pct:
                    env["volatility_regime"] = "HIGH"
                else:
                    env["volatility_regime"] = "NORMAL"
            # Tendência e estrutura: usa multi-timeframe trends
            mtf = self._analyze_mtf_trends()
            # Direção: se a maioria das timeframes estiver em alta, então alta; se baixa; senão sideways
            ups = sum(1 for v in mtf.values() if v.get("tendencia") == "Alta")
            downs = sum(1 for v in mtf.values() if v.get("tendencia") == "Baixa")
            if ups > downs:
                env["trend_direction"] = "UP"
            elif downs > ups:
                env["trend_direction"] = "DOWN"
            else:
                env["trend_direction"] = "SIDEWAYS"
            # Estrutura: se algum regime retornado for "Range" -> RANGE_BOUND, se "Institucional" ou "Manipulação" -> TRENDING, senão ACCUMULATION
            regimes = [v.get("regime") for v in mtf.values()]
            if any(r == "Range" for r in regimes):
                env["market_structure"] = "RANGE_BOUND"
            elif any(r in ("Institucional", "Manipulação") for r in regimes):
                env["market_structure"] = "TRENDING"
            else:
                env["market_structure"] = "ACCUMULATION"
            # Liquidez: valor nominal. Aprofundar com análise de book e spreads; por ora, assume normal
            env["liquidity_environment"] = "NORMAL"
            # Correlações com SP500 (SPX), DXY e GOLD
            if ENABLE_ALPHAVANTAGE:
                # SP500
                try:
                    env["correlation_spy"] = self._compute_correlation("SP500", EXTERNAL_MARKETS.get("SP500", ""))
                except Exception:
                    pass
                # DXY
                try:
                    env["correlation_dxy"] = self._compute_correlation("DXY", "DXY")
                except Exception:
                    pass
                # GOLD
                try:
                    env["correlation_gold"] = self._compute_correlation("GOLD", EXTERNAL_MARKETS.get("GOLD", ""))
                except Exception:
                    pass
            # Sentimento de risco baseado nas correlações: se BTC acompanha SP500 (correlation>0) e diverge do DXY (correlation<0), risco ON; inverso -> OFF
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
            logging.debug(f"Falha ao calcular ambiente de mercado: {e}")
        return env

    def _compute_correlation(self, name: str, ticker: str) -> float:
        """
        Calcula a correlação de retornos diários entre o ativo principal e outro mercado.
        Se não houver dados suficientes, retorna None.
        """
        try:
            if not ticker:
                return None
            # Dados do ativo principal (BTCUSDT) - fecha diário
            sym_df = self._fetch_klines(self.symbol, '1d', limit=CORRELATION_LOOKBACK + 10)
            if sym_df.empty:
                return None
            sym_series = pd.to_numeric(sym_df['close'], errors='coerce')
            sym_returns = sym_series.pct_change().dropna()
            # Dados do mercado externo via Alpha Vantage (usando TIME_SERIES_DAILY)
            if name == "DXY":
                # Tenta ECONOMIC_INDICATORS, depois TIME_SERIES_DAILY
                ext_hist = self._alpha_vantage_history(ticker, function="ECONOMIC_INDICATORS")
                if ext_hist.empty:
                    ext_hist = self._alpha_vantage_history(ticker, function="TIME_SERIES_DAILY")
                    # Alguns tickers podem vir em 'value'
                    if 'value' in ext_hist.columns:
                        ext_series = ext_hist['value'].astype(float)
                    else:
                        ext_series = ext_hist['close'].astype(float)
                else:
                    ext_series = ext_hist['value'].astype(float)
            else:
                ext_hist = self._alpha_vantage_history(ticker, function="TIME_SERIES_DAILY")
                if 'close' in ext_hist.columns:
                    ext_series = ext_hist['close'].astype(float)
                elif 'value' in ext_hist.columns:
                    ext_series = ext_hist['value'].astype(float)
                else:
                    return None
            if ext_hist.empty:
                return None
            ext_returns = ext_series.pct_change().dropna()
            # Alinha retornos pelo índice (datas) e limita ao lookback
            sym_aligned = sym_returns.tail(CORRELATION_LOOKBACK)
            ext_aligned = ext_returns.tail(CORRELATION_LOOKBACK)
            # Combina índices comuns
            merged = pd.concat([sym_aligned, ext_aligned], axis=1, join='inner')
            if merged.shape[0] < 2:
                return None
            corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
            return float(round(corr, 4)) if corr is not None else None
        except Exception as e:
            logging.debug(f"Erro ao calcular correlação {name}: {e}")
            return None

    def _analyze_mtf_trends(self):
        mtf_context = {}
        for tf in self.timeframes:
            limit_needed = max(self.ema_period, self.atr_period) * 3 + 20
            df = self._fetch_klines(self.symbol, tf, limit=limit_needed)
            if not df.empty:
                df = df.copy()
                # EMA para tendência básica
                df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
                last_close = float(df['close'].iloc[-1])
                last_ema = float(df['ema'].iloc[-1])
                tendencia = "Alta" if last_close > last_ema else "Baixa"
                # Indicadores técnicos adicionais
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
                    # Indicadores avançados
                    "rsi_short": round(rsi_short, 2) if rsi_short else 0.0,
                    "rsi_long": round(rsi_long, 2) if rsi_long else 0.0,
                    "macd": round(macd_val, 4) if macd_val else 0.0,
                    "macd_signal": round(macd_sig, 4) if macd_sig else 0.0,
                    "adx": round(adx_val, 2) if adx_val else 0.0,
                    "realized_vol": round(realized_vol, 4) if realized_vol else 0.0,
                }
        return mtf_context

    # ---------- Intermarket / Alpha Vantage ----------
    def _fetch_intermarket_data(self):
        data = {}
        # symbols Binance (ex.: BTCUSDT, ETHUSDT...) continuam vindo dos klines
        for sym in self.intermarket_symbols:
            df = self._fetch_klines(sym, '5m', limit=2)
            if not df.empty:
                last, prev = float(df['close'].iloc[-1]), float(df['close'].iloc[-2])
                data[sym] = {"preco_atual": last, "movimento": "Alta" if last > prev else "Baixa"}
        if not ENABLE_ALPHAVANTAGE:
            return data
        # DXY via Alpha Vantage (ECONOMIC_INDICATORS)
        dxy_got = False
        for tkr in self._dxy_candidates:
            try:
                hist = self._alpha_vantage_history(tkr, function="ECONOMIC_INDICATORS")
                if not hist.empty:
                    last, prev = float(hist["value"].iloc[-1]), float(hist["value"].iloc[-2])
                    data["DXY"] = {
                        "preco_atual": round(last, 2),
                        "movimento": "Alta" if last > prev else "Baixa",
                        "ticker": tkr
                    }
                    dxy_got = True
                    break
            except Exception as e:
                logging.debug(f"DXY fallback {tkr} falhou: {e}")
        if not dxy_got:
            logging.debug("DXY indisponível em todos os fallbacks.")
        return data

    def _fetch_external_markets(self):
        ext_data = {}
        if not ENABLE_ALPHAVANTAGE:
            return ext_data
        for name, ticker in EXTERNAL_MARKETS.items():
            # Primeiro tenta o ticker fornecido
            if "/" in ticker:  # Pares de moedas (ex: USD/EUR)
                hist = self._alpha_vantage_history(ticker, function="FX_DAILY")
                if not hist.empty:
                    last, prev = float(hist["close"].iloc[-1]), float(hist["close"].iloc[-2])
                    ext_data[name] = {
                        "preco_atual": round(last, 2),
                        "movimento": "Alta" if last > prev else "Baixa",
                        "ticker": ticker
                    }
                    continue
            # Depois fallbacks (se houver para este nome)
            for alt in self._fallback_map.get(name, []):
                if "/" in alt:  # Pares de moedas
                    hist = self._alpha_vantage_history(alt, function="FX_DAILY")
                else:  # Ações/índices
                    hist = self._alpha_vantage_history(alt, function="TIME_SERIES_DAILY")
                if not hist.empty:
                    last, prev = float(hist["close"].iloc[-1]), float(hist["close"].iloc[-2])
                    ext_data[name] = {
                        "preco_atual": round(last, 2),
                        "movimento": "Alta" if last > prev else "Baixa",
                        "ticker": alt
                    }
                    break
            if name not in ext_data:
                logging.debug(f"Sem dados para {name} ({ticker}) e fallbacks.")
        return ext_data

    # ---------- Derivativos (Binance) ----------
    def _fetch_liquidations_data(self, symbol, lookback_minutes=5):
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
                r = requests.get(self.liquidations_api_url, params=params, timeout=5)
                if r.status_code == 204 or not r.text:
                    return []
                if r.status_code == 200:
                    return r.json()
                logging.debug(f"ForceOrders {r.status_code}: {r.text[:200]}")
                if r.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Rate limit liquidations. Retry {attempt+1}/{max_retries} em {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                return []
            except requests.exceptions.RequestException as e:
                logging.error(f"Req liquidations ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
            except Exception as e:
                logging.debug(f"ForceOrders exception ({attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        logging.error(f"Falha persistente liquidations {symbol}. Retornando [].")
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

    def _fetch_derivatives_data(self):
        derivatives_data = {}
        for sym in self.derivatives_symbols:
            try:
                fr = requests.get(self.funding_api_url, params={'symbol': sym}, timeout=5).json()
                funding_rate = float(fr[0]["fundingRate"]) * 100 if fr else 0.0
                oi = requests.get(self.open_interest_api_url, params={'symbol': sym}, timeout=5).json()
                open_interest = float(oi.get("openInterest", 0))
                ls = requests.get(
                    self.long_short_ratio_api_url,
                    params={'symbol': sym, 'period': '5m', 'limit': 1}, timeout=5
                ).json()
                long_short_ratio = float(ls[0]["longShortRatio"]) if ls else 0.0
                liq = self._fetch_liquidations_data(sym, lookback_minutes=int(self.update_interval / 60))
                heatmap = self._build_liquidation_heatmap(liq)
                totals = {
                    "longs_usd": sum(v["longs"] for v in heatmap.values()),
                    "shorts_usd": sum(v["shorts"] for v in heatmap.values())
                }
                price = self._fetch_symbol_price(sym)
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
                logging.debug(f"Erro derivativos {sym}: {e}")
        return derivatives_data

    # ---------- On-chain / Sentimento (placeholders) ----------
    def _fetch_onchain_sentiment(self):
        sentiment = {"onchain": {}, "funding_agg": {}}
        try:
            if ENABLE_ONCHAIN:
                sentiment["onchain"] = {
                    "btc_exchange_inflow": 1200,
                    "btc_exchange_outflow": 900,
                    "stablecoin_inflow": 5_000_000 if STABLECOIN_FLOW_TRACKING else 0,
                    "stablecoin_outflow": 4_500_000 if STABLECOIN_FLOW_TRACKING else 0
                }
        except Exception as e:
            logging.debug(f"Dados on-chain indisponíveis: {e}")
        try:
            sentiment["funding_agg"] = {
                "avg_funding": 0.02,
                "binance_funding": 0.025,
                "okx_funding": 0.018,
                "cme_basis": -0.005
            }
        except Exception as e:
            logging.debug(f"Sentimento funding indisponível: {e}")
        return sentiment

    # ---------- Consolidação ----------
    def _build_full_context(self):
        return {
            "mtf_trends": self._analyze_mtf_trends(),
            "intermarket": self._fetch_intermarket_data(),
            "external": self._fetch_external_markets(),
            "derivatives": self._fetch_derivatives_data(),
            "sentiment": self._fetch_onchain_sentiment(),
            "historical_vp": self.historical_profiler.update_profiles(),
            # Novos campos: contexto e ambiente de mercado
            "market_context": self._calculate_market_context(),
            "market_environment": self._calculate_market_environment(),
            "timestamp": self.time_manager.now_iso(),
        }

    def _update_loop(self):
        logging.info("Coletor de Contexto iniciado.")
        while not self.should_stop:
            try:
                ctx = self._build_full_context()
                with self._lock:
                    self.context_data = ctx
                logging.info("Contexto Macro atualizado.")
            except Exception as e:
                logging.error(f"Erro crítico loop: {e}", exc_info=True)
            time.sleep(self.update_interval)

    def get_context(self):
        with self._lock:
            return self.context_data.copy()

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        if self.thread.is_alive():
            logging.info("Parando Coletor de Contexto...")
            self.should_stop = True
            self.thread.join(timeout=5)