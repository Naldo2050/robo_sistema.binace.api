# context_collector.py
import requests
import pandas as pd
import logging
import time
import threading
import yfinance as yf
from datetime import datetime, timezone, timedelta
from config import CONTEXT_TIMEFRAMES, CONTEXT_EMA_PERIOD, CONTEXT_ATR_PERIOD, CONTEXT_UPDATE_INTERVAL_SECONDS, INTERMARKET_SYMBOLS, DERIVATIVES_SYMBOLS, VP_NUM_DAYS_HISTORY, VP_VALUE_AREA_PERCENT
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
        
        # APIs
        self.klines_api_url = "https://api.binance.com/api/v3/klines"
        self.funding_api_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.open_interest_api_url = "https://fapi.binance.com/fapi/v1/openInterest"
        self.long_short_ratio_api_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        self.liquidations_api_url = "https://fapi.binance.com/fapi/v1/allForceOrders"

        self.historical_profiler = HistoricalVolumeProfiler(
            symbol=self.symbol,
            num_days=VP_NUM_DAYS_HISTORY,
            value_area_percent=VP_VALUE_AREA_PERCENT
        )

        self.context_data = {}
        self._lock = threading.Lock()
        self.should_stop = False
        self.thread = threading.Thread(target=self._update_loop, daemon=True)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        df['h-l'] = df['high'] - df['low']; df['h-pc'] = abs(df['high'] - df['close'].shift()); df['l-pc'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1); atr = df['tr'].ewm(span=period, adjust=False).mean(); return atr.iloc[-1]
        
    def _fetch_klines(self, symbol, timeframe, limit=100):
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        try:
            response = requests.get(self.klines_api_url, params=params, timeout=10); response.raise_for_status(); data = response.json()
            df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Erro ao buscar klines para {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def _analyze_mtf_trends(self):
        mtf_context = {}
        for tf in self.timeframes:
            df = self._fetch_klines(self.symbol, tf, limit=self.ema_period * 2 + 1)
            if not df.empty:
                df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
                last_close, last_ema = df['close'].iloc[-1], df['ema'].iloc[-1]
                tendencia = "Alta" if last_close > last_ema else "Baixa"
                atr = self._calculate_atr(df, self.atr_period)
                mtf_context[tf] = {"tendencia": tendencia, "preco_atual": float(last_close), f"mme_{self.ema_period}": float(round(last_ema, 2)), "atr": float(round(atr, 2))}
        return mtf_context

    def _fetch_intermarket_data(self):
        intermarket_data = {}
        for symbol in self.intermarket_symbols:
            df = self._fetch_klines(symbol, '5m', limit=2)
            if not df.empty: last_close, prev_close = df['close'].iloc[-1], df['close'].iloc[-2]; intermarket_data[symbol] = {"preco_atual": float(last_close), "movimento": "Alta" if last_close > prev_close else "Baixa"}
        try:
            dxy = yf.Ticker("DX-Y.NYB"); hist = dxy.history(period="2d", interval="5m")
            if not hist.empty: last_close, prev_close = hist['Close'].iloc[-1], hist['Close'].iloc[-2]; intermarket_data["DXY"] = {"preco_atual": float(round(last_close, 2)), "movimento": "Alta" if last_close > prev_close else "Baixa"}
        except Exception as e: logging.warning(f"Não foi possível buscar dados do DXY: {e}")
        return intermarket_data

    def _fetch_liquidations_data(self, symbol, lookback_minutes=5):
        liquidations = {'longs_usd': 0.0, 'shorts_usd': 0.0}
        try:
            end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_time_ms = end_time_ms - int(timedelta(minutes=lookback_minutes).total_seconds() * 1000)
            params = {'symbol': symbol, 'startTime': start_time_ms, 'limit': 1000}
            response = requests.get(self.liquidations_api_url, params=params, timeout=5); response.raise_for_status(); data = response.json()
            for liq in data:
                liq_value_usd = float(liq['price']) * float(liq['origQty'])
                if liq['side'] == 'SELL': liquidations['longs_usd'] += liq_value_usd
                elif liq['side'] == 'BUY': liquidations['shorts_usd'] += liq_value_usd
            return liquidations
        except Exception as e:
            if '400' not in str(e): logging.warning(f"Aviso ao buscar dados de liquidação para {symbol}: {e}")
            return liquidations

    def _fetch_derivatives_data(self):
        derivatives_data = {}
        for symbol in self.derivatives_symbols:
            try:
                fr_params = {'symbol': symbol}; fr_response = requests.get(self.funding_api_url, params=fr_params, timeout=5); fr_response.raise_for_status(); fr_data = fr_response.json()
                funding_rate = float(fr_data[0]['fundingRate']) * 100
                oi_params = {'symbol': symbol}; oi_response = requests.get(self.open_interest_api_url, params=oi_params, timeout=5); oi_response.raise_for_status(); oi_data = oi_response.json()
                open_interest = float(oi_data['openInterest'])
                ls_params = {'symbol': symbol, 'period': '5m', 'limit': 1}; ls_response = requests.get(self.long_short_ratio_api_url, params=ls_params, timeout=5); ls_response.raise_for_status(); ls_data = ls_response.json()
                long_short_ratio = float(ls_data[0]['longShortRatio'])
                derivatives_data[symbol] = {"funding_rate_percent": round(funding_rate, 4),"open_interest": open_interest,"long_short_ratio": long_short_ratio}
                liquidations = self._fetch_liquidations_data(symbol, lookback_minutes=(CONTEXT_UPDATE_INTERVAL_SECONDS / 60))
                derivatives_data[symbol].update(liquidations)
            except Exception as e: logging.error(f"Erro ao buscar dados de derivativos para {symbol}: {e}")
        return derivatives_data

    def _build_full_context(self):
        # Removido os prints de DEBUG daqui para não poluir
        context = {
            "mtf_trends": self._analyze_mtf_trends(),
            "intermarket": self._fetch_intermarket_data(),
            "derivatives": self._fetch_derivatives_data(),
            "historical_vp": self.historical_profiler.update_profiles()
        }
        return context

    def _update_loop(self):
        logging.info("✅ Coletor de Contexto iniciado em segundo plano.")
        while not self.should_stop:
            try:
                print("\nDEBUG: Coletor vai iniciar um novo ciclo de busca de dados...")
                context = self._build_full_context()
                
                # <<< ALTERAÇÃO ETAPA 2.3: Adicionados prints de diagnóstico cirúrgico
                print(f"DEBUG: Coleta concluída. Conteúdo tem {len(context)} chaves. Prestes a salvar...")
                
                with self._lock:
                    self.context_data = context
                    print("DEBUG: DADOS SALVOS NA MEMÓRIA COM SUCESSO!")
                
                logging.info(f"Contexto Macro (Completo) atualizado.")
                print("DEBUG: Log de sucesso ('Contexto Macro...') foi enviado.")
                
            except Exception as e:
                logging.error(f"Erro CRÍTICO no loop de atualização do contexto: {e}", exc_info=True)
                print(f"DEBUG: OCORREU UM ERRO GRAVE NO LOOP DE ATUALIZAÇÃO.")
            
            time_to_sleep = self.update_interval
            while time_to_sleep > 0 and not self.should_stop:
                time.sleep(1)
                time_to_sleep -= 1

    def get_context(self):
        with self._lock:
            return self.context_data.copy()

    def start(self):
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self):
        if self.thread.is_alive():
            logging.info("🛑 Parando o Coletor de Contexto...")
            self.should_stop = True
            self.thread.join(timeout=5)