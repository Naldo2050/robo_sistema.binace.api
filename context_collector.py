import requests
import pandas as pd
import logging
import time
import threading
import yfinance as yf
from datetime import datetime, timezone, timedelta
from config import (
    CONTEXT_TIMEFRAMES, CONTEXT_EMA_PERIOD, CONTEXT_ATR_PERIOD,
    CONTEXT_UPDATE_INTERVAL_SECONDS, INTERMARKET_SYMBOLS, DERIVATIVES_SYMBOLS,
    VP_NUM_DAYS_HISTORY, VP_VALUE_AREA_PERCENT, LIQUIDATION_MAP_DEPTH,
    EXTERNAL_MARKETS, ENABLE_ONCHAIN, ONCHAIN_PROVIDERS, STABLECOIN_FLOW_TRACKING
)
from historical_profiler import HistoricalVolumeProfiler

class ContextCollector:
    def __init__(self, symbol):
        self.symbol = symbol
        self.timeframes = CONTEXT_TIMEFRAMES
        self.ema_period = CONTEXT_EMA_PERIOD
        self.atr_period = CONTEXT_ATR_PERIOD
        self.update_interval = CONTEXT_UPDATE_INTERVAL_SECONDS
        self.intermarket_symbols = INTERMARKET_SYMBOLS
        self.derivatives_symbols = DERIVATIVES_SYMBOLS

        # APIs Binance
        self.klines_api_url = "https://api.binance.com/api/v3/klines"
        self.funding_api_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.open_interest_api_url = "https://fapi.binance.com/fapi/v1/openInterest"
        self.long_short_ratio_api_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        self.liquidations_api_url = "https://fapi.binance.com/fapi/v1/allForceOrders"

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

    # ========== Funções de cálculo núcleo ==========
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        df['h-l'] = df['high'] - df['low']
        df['h-pc'] = abs(df['high'] - df['close'].shift())
        df['l-pc'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['h-l','h-pc','l-pc']].max(axis=1)
        atr = df['tr'].ewm(span=period, adjust=False).mean()
        return atr.iloc[-1]

    def _fetch_klines(self, symbol, timeframe, limit=200):
        try:
            params = {"symbol": symbol, "interval": timeframe, "limit": limit}
            res = requests.get(self.klines_api_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            df = pd.DataFrame(data, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','tbbav','tbqav','ignore'])
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            logging.error(f"Erro buscar klines {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def _classify_regime(self, df: pd.DataFrame, atr_value: float):
        if df.empty: return "Desconhecido"
        avg_volume = df["volume"].tail(50).mean() if len(df) >= 50 else df["volume"].mean()
        last_volume = df["volume"].iloc[-1]
        vol_change = (last_volume / avg_volume) if avg_volume > 0 else 1
        price = df["close"].iloc[-1]
        atr_pct = atr_value / price if price > 0 else 0
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

    def _analyze_mtf_trends(self):
        mtf_context = {}
        for tf in self.timeframes:
            df = self._fetch_klines(self.symbol, tf, limit=self.ema_period*3)
            if not df.empty:
                df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
                last_close = df['close'].iloc[-1]
                last_ema = df['ema'].iloc[-1]
                tendencia = "Alta" if last_close > last_ema else "Baixa"
                atr = self._calculate_atr(df, self.atr_period)
                regime = self._classify_regime(df, atr)
                mtf_context[tf] = {
                    "tendencia": tendencia,
                    "preco_atual": float(last_close),
                    f"mme_{self.ema_period}": float(round(last_ema, 2)),
                    "atr": float(round(atr, 2)),
                    "regime": regime
                }
        return mtf_context

    # ========== Intermarket ==========
    def _fetch_intermarket_data(self):
        data = {}
        for sym in self.intermarket_symbols:
            df = self._fetch_klines(sym, '5m', limit=2)
            if not df.empty:
                last, prev = df['close'].iloc[-1], df['close'].iloc[-2]
                data[sym] = {"preco_atual": float(last), "movimento": "Alta" if last > prev else "Baixa"}
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="2d", interval="5m")
            if not hist.empty:
                last, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
                data["DXY"] = {"preco_atual": float(round(last,2)), "movimento": "Alta" if last > prev else "Baixa"}
        except Exception as e:
            logging.warning(f"DXY indisponível: {e}")
        return data

    def _fetch_external_markets(self):
        ext_data = {}
        for name, ticker in EXTERNAL_MARKETS.items():
            try:
                asset = yf.Ticker(ticker)
                hist = asset.history(period="2d", interval="15m")
                if not hist.empty:
                    last, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
                    ext_data[name] = {
                        "preco_atual": float(round(last, 2)),
                        "movimento": "Alta" if last > prev else "Baixa"
                    }
            except Exception as e:
                logging.warning(f"Falhou buscar {name}: {e}")
        return ext_data

    # ========== Derivativos ==========
    def _fetch_liquidations_data(self, symbol, lookback_minutes=5):
        try:
            end_time_ms = int(datetime.now(timezone.utc).timestamp()*1000)
            start_time_ms = end_time_ms - int(timedelta(minutes=lookback_minutes).total_seconds()*1000)
            params = {'symbol': symbol, 'startTime': start_time_ms, 'limit': 1000}
            res = requests.get(self.liquidations_api_url, params=params, timeout=5)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logging.warning(f"Aviso liquidações {symbol}: {e}")
            return []

    def _build_liquidation_heatmap(self, liq_data):
        heatmap = {}
        if not liq_data: return heatmap
        for liq in liq_data:
            try:
                price = float(liq["price"])
                qty = float(liq["origQty"])
                side = liq["side"]
                bucket = round(price / LIQUIDATION_MAP_DEPTH) * LIQUIDATION_MAP_DEPTH
                if bucket not in heatmap:
                    heatmap[bucket] = {"longs": 0.0, "shorts": 0.0}
                usd = price * qty
                if side == "SELL": heatmap[bucket]["longs"] += usd
                elif side == "BUY": heatmap[bucket]["shorts"] += usd
            except Exception: continue
        return {str(k): v for k,v in sorted(heatmap.items())}

    def _fetch_derivatives_data(self):
        derivatives_data = {}
        for sym in self.derivatives_symbols:
            try:
                fr = requests.get(self.funding_api_url, params={'symbol': sym}, timeout=5).json()
                funding_rate = float(fr[0]["fundingRate"])*100 if fr else 0
                oi = requests.get(self.open_interest_api_url, params={'symbol': sym}, timeout=5).json()
                open_interest = float(oi.get("openInterest", 0))
                ls = requests.get(self.long_short_ratio_api_url, params={'symbol': sym,'period':'5m','limit':1},timeout=5).json()
                long_short_ratio = float(ls[0]["longShortRatio"]) if ls else 0
                liq = self._fetch_liquidations_data(sym, lookback_minutes=(CONTEXT_UPDATE_INTERVAL_SECONDS/60))
                heatmap = self._build_liquidation_heatmap(liq)
                totals = {
                    "longs_usd": sum(v["longs"] for v in heatmap.values()),
                    "shorts_usd": sum(v["shorts"] for v in heatmap.values())
                }
                derivatives_data[sym] = {
                    "funding_rate_percent": round(funding_rate,4),
                    "open_interest": open_interest,
                    "long_short_ratio": long_short_ratio,
                    "liquidation_heatmap": heatmap,
                    **totals
                }
            except Exception as e:
                logging.error(f"Erro derivativos {sym}: {e}")
        return derivatives_data

    # ========== Off-chain & Sentimento ==========
    def _fetch_onchain_sentiment(self):
        sentiment = {"onchain": {}, "funding_agg": {}}
        try:
            if ENABLE_ONCHAIN:
                # mock simples para BTC inflows/outflows
                sentiment["onchain"] = {
                    "btc_exchange_inflow": 1200,
                    "btc_exchange_outflow": 900,
                    "stablecoin_inflow": 5_000_000 if STABLECOIN_FLOW_TRACKING else 0,
                    "stablecoin_outflow": 4_500_000 if STABLECOIN_FLOW_TRACKING else 0
                }
        except Exception as e:
            logging.warning(f"Dados on-chain indisponíveis: {e}")

        try:
            # Sentimento de funding agregado multi-exchange (placeholder)
            sentiment["funding_agg"] = {
                "avg_funding": 0.02,
                "binance_funding": 0.025,
                "okx_funding": 0.018,
                "cme_basis": -0.005
            }
        except Exception as e:
            logging.warning(f"Sentimento funding indisponível: {e}")
        return sentiment

    # ========== Consolidação ==========
    def _build_full_context(self):
        return {
            "mtf_trends": self._analyze_mtf_trends(),
            "intermarket": self._fetch_intermarket_data(),
            "external": self._fetch_external_markets(),
            "derivatives": self._fetch_derivatives_data(),
            "sentiment": self._fetch_onchain_sentiment(),
            "historical_vp": self.historical_profiler.update_profiles()
        }

    def _update_loop(self):
        logging.info("✅ Coletor de Contexto iniciado.")
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
            logging.info("🛑 Parando Coletor de Contexto...")
            self.should_stop = True
            self.thread.join(timeout=5)