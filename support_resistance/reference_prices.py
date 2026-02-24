"""
Reference Prices — Previous Period Closes & Highs/Lows.

Mantém referências de preço dos períodos anteriores (dia, semana, mês).
Usado por desks institucionais como primeiro nível de referência.

Uso:
    rp = ReferencePrices()
    await rp.update("BTCUSDT", client)  # client = Binance async client
    ctx = rp.get_context(current_price=64892)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TZ_NY = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")


class ReferencePrices:
    """
    Calcula e mantém preços de referência dos períodos anteriores.
    
    Períodos:
        - Previous 1h, 4h
        - Previous Day (close, high, low, open)
        - Previous Week (close, high, low)
        - Previous Month (close, high, low)
    
    Atualiza via candles da Binance (klines).
    """

    def __init__(self):
        self._cache = {}
        self._last_update_ms = 0
        self._min_update_interval_ms = 60_000  # Atualiza no máximo 1x por minuto

    async def update(self, symbol: str, client, force: bool = False) -> bool:
        """
        Atualiza os preços de referência via Binance API.
        
        Args:
            symbol: Par de trading (ex: "BTCUSDT")
            client: Cliente Binance (async) com método get_klines()
            force: Forçar atualização mesmo dentro do intervalo mínimo
            
        Returns:
            True se atualizou, False se usou cache
        """
        import time
        now_ms = int(time.time() * 1000)
        
        if not force and (now_ms - self._last_update_ms) < self._min_update_interval_ms:
            return False

        try:
            # Buscar candles de diferentes timeframes
            # Binance klines: [open_time, open, high, low, close, volume, close_time, ...]
            
            # 1h — últimas 4 candles
            klines_1h = await self._safe_get_klines(client, symbol, "1h", limit=4)
            
            # 4h — últimas 4 candles
            klines_4h = await self._safe_get_klines(client, symbol, "4h", limit=4)
            
            # 1d — últimas 3 candles (hoje + 2 anteriores)
            klines_1d = await self._safe_get_klines(client, symbol, "1d", limit=3)
            
            # 1w — últimas 3 candles
            klines_1w = await self._safe_get_klines(client, symbol, "1w", limit=3)
            
            # 1M — últimas 3 candles
            klines_1M = await self._safe_get_klines(client, symbol, "1M", limit=3)

            refs = {}

            # Previous 1h (penúltima candle, pois a última é a atual)
            if klines_1h and len(klines_1h) >= 2:
                prev = klines_1h[-2]
                refs["prev_1h"] = {
                    "close": float(prev[4]),
                    "high": float(prev[2]),
                    "low": float(prev[3]),
                    "open": float(prev[1]),
                }

            # Previous 4h
            if klines_4h and len(klines_4h) >= 2:
                prev = klines_4h[-2]
                refs["prev_4h"] = {
                    "close": float(prev[4]),
                    "high": float(prev[2]),
                    "low": float(prev[3]),
                    "open": float(prev[1]),
                }

            # Previous Day
            if klines_1d and len(klines_1d) >= 2:
                prev = klines_1d[-2]
                refs["prev_day"] = {
                    "close": float(prev[4]),
                    "high": float(prev[2]),
                    "low": float(prev[3]),
                    "open": float(prev[1]),
                }

            # Previous Week
            if klines_1w and len(klines_1w) >= 2:
                prev = klines_1w[-2]
                refs["prev_week"] = {
                    "close": float(prev[4]),
                    "high": float(prev[2]),
                    "low": float(prev[3]),
                    "open": float(prev[1]),
                }

            # Previous Month
            if klines_1M and len(klines_1M) >= 2:
                prev = klines_1M[-2]
                refs["prev_month"] = {
                    "close": float(prev[4]),
                    "high": float(prev[2]),
                    "low": float(prev[3]),
                    "open": float(prev[1]),
                }

            self._cache[symbol] = refs
            self._last_update_ms = now_ms
            logger.info(f"ReferencePrices updated for {symbol}: {len(refs)} periods")
            return True

        except Exception as e:
            logger.error(f"ReferencePrices update failed: {e}")
            return False

    async def _safe_get_klines(self, client, symbol: str, interval: str, limit: int) -> list:
        """
        Busca klines com tratamento de erro.
        Tenta múltiplos métodos do client (compatibilidade).
        """
        try:
            # Tentar método async primeiro (python-binance)
            if hasattr(client, 'get_klines'):
                return await client.get_klines(symbol=symbol, interval=interval, limit=limit)
            elif hasattr(client, 'klines'):
                return await client.klines(symbol=symbol, interval=interval, limit=limit)
            elif hasattr(client, 'futures_klines'):
                return await client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                logger.warning(f"No klines method found on client for {interval}")
                return []
        except Exception as e:
            logger.warning(f"Failed to get {interval} klines for {symbol}: {e}")
            return []

    def get_context(self, symbol: str = "BTCUSDT", current_price: float = 0) -> dict:
        """
        Retorna contexto de preços de referência com distâncias calculadas.
        
        Args:
            symbol: Par de trading
            current_price: Preço atual para calcular distâncias
            
        Returns:
            Dict com referências e métricas de distância
        """
        refs = self._cache.get(symbol, {})
        
        if not refs:
            return {
                "status": "no_data",
                "reference_prices": {},
            }

        result = {}

        for period_key in ("prev_1h", "prev_4h", "prev_day", "prev_week", "prev_month"):
            period_data = refs.get(period_key)
            if not period_data:
                continue

            entry = {
                "close": period_data["close"],
                "high": period_data["high"],
                "low": period_data["low"],
                "open": period_data["open"],
            }

            # Calcular distâncias se tiver preço atual
            if current_price > 0 and period_data["close"] > 0:
                close = period_data["close"]
                entry["distance_from_close_pct"] = round(
                    (current_price - close) / close * 100, 4
                )
                entry["above_prev_close"] = current_price > close
                entry["distance_from_high_pct"] = round(
                    (current_price - period_data["high"]) / period_data["high"] * 100, 4
                )
                entry["distance_from_low_pct"] = round(
                    (current_price - period_data["low"]) / period_data["low"] * 100, 4
                )
                entry["within_prev_range"] = (
                    period_data["low"] <= current_price <= period_data["high"]
                )

            result[period_key] = entry

        # Resumo rápido
        summary = {}
        for period in ("prev_day", "prev_week", "prev_month"):
            if period in result and "above_prev_close" in result[period]:
                summary[f"above_{period}_close"] = result[period]["above_prev_close"]
                summary[f"{period}_close"] = result[period]["close"]

        return {
            "status": "ok",
            "reference_prices": result,
            "summary": summary,
        }

    def update_from_candles(self, symbol: str, candles: dict) -> None:
        """
        Alternativa: atualizar diretamente com candles já disponíveis
        no sistema (sem chamar API).
        
        Args:
            candles: Dict com chaves '1h', '4h', '1d', '1w', '1M',
                     cada uma com lista de candles [open, high, low, close]
        """
        refs = {}
        
        mapping = {
            "1h": "prev_1h",
            "4h": "prev_4h",
            "1d": "prev_day",
            "1w": "prev_week",
            "1M": "prev_month",
        }
        
        for tf, ref_key in mapping.items():
            tf_candles = candles.get(tf)
            if tf_candles and len(tf_candles) >= 2:
                prev = tf_candles[-2]  # Penúltima = período anterior completo
                if isinstance(prev, dict):
                    refs[ref_key] = {
                        "close": float(prev.get("close", 0)),
                        "high": float(prev.get("high", 0)),
                        "low": float(prev.get("low", 0)),
                        "open": float(prev.get("open", 0)),
                    }
                elif isinstance(prev, (list, tuple)) and len(prev) >= 5:
                    # Formato Binance kline: [open_time, open, high, low, close, ...]
                    refs[ref_key] = {
                        "close": float(prev[4]),
                        "high": float(prev[2]),
                        "low": float(prev[3]),
                        "open": float(prev[1]),
                    }

        if refs:
            self._cache[symbol] = refs
            logger.info(f"ReferencePrices updated from candles for {symbol}: {len(refs)} periods")
