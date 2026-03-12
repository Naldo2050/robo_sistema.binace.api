# funding_aggregator.py
"""
Agregador de funding rates REAIS da Binance Futures (API gratuita, sem chave extra).

Coleta funding rates de múltiplos símbolos e calcula métricas agregadas:
- Funding rate atual por símbolo
- Média ponderada por OI
- Tendência de funding (rising/falling/stable)
- Histórico de funding para detecção de extremos

Usa o endpoint premiumIndex que retorna funding + mark price em 1 request.
Rate limit: ~1200 req/min (Binance), chamamos 1x a cada 5 min = seguro.
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger("FundingAggregator")

_CACHE_TTL = 300  # 5 minutos
_REQUEST_TIMEOUT = 10

# Símbolos para monitorar funding (os mais relevantes)
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


class FundingAggregator:
    """
    Agrega funding rates reais da Binance Futures.
    """

    def __init__(self, symbols: Optional[List[str]] = None, cache_ttl: int = _CACHE_TTL):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_ts: float = 0.0
        self._history: deque = deque(maxlen=100)  # últimas 100 leituras
        self._premium_index_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        self._funding_history_url = "https://fapi.binance.com/fapi/v1/fundingRate"

    async def fetch_all(self, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        """
        Retorna funding rates agregados de todos os símbolos.
        """
        now = time.time()
        if self._cache and (now - self._cache_ts) < self.cache_ttl:
            return self._cache

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        try:
            result = await self._fetch_premium_index_all(session)
            if result:
                self._cache = result
                self._cache_ts = now
                self._history.append({"ts": now, "data": result})
            return result or self._cache
        finally:
            if own_session and session:
                await session.close()

    async def _fetch_premium_index_all(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Busca premiumIndex para todos os símbolos em 1 request (sem parâmetro symbol).
        Retorna funding rate, mark price, index price, next funding time.
        """
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        per_symbol: Dict[str, Dict[str, Any]] = {}

        try:
            # Um único request retorna todos os símbolos
            async with session.get(self._premium_index_url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning(f"premiumIndex HTTP {resp.status}")
                    return {}

                all_data = await resp.json()

                for item in all_data:
                    sym = item.get("symbol", "")
                    if sym in self.symbols:
                        funding_rate = float(item.get("lastFundingRate", 0))
                        mark_price = float(item.get("markPrice", 0))
                        index_price = float(item.get("indexPrice", 0))
                        next_funding_ts = int(item.get("nextFundingTime", 0))

                        per_symbol[sym] = {
                            "funding_rate": round(funding_rate, 8),
                            "funding_rate_pct": round(funding_rate * 100, 4),
                            "mark_price": round(mark_price, 2),
                            "index_price": round(index_price, 2),
                            "mark_index_diff_pct": round(
                                ((mark_price - index_price) / index_price * 100)
                                if index_price > 0 else 0, 4
                            ),
                            "next_funding_time_ms": next_funding_ts,
                            "time_to_funding_min": max(0, round(
                                (next_funding_ts / 1000 - time.time()) / 60, 1
                            )) if next_funding_ts > 0 else 0,
                        }

        except Exception as e:
            logger.warning(f"Erro ao buscar premiumIndex: {e}")
            return {}

        if not per_symbol:
            return {}

        # Calcular métricas agregadas
        funding_rates = [v["funding_rate"] for v in per_symbol.values()]
        avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0

        # Determinar sentimento baseado em funding
        btc_funding = per_symbol.get("BTCUSDT", {}).get("funding_rate", 0)
        eth_funding = per_symbol.get("ETHUSDT", {}).get("funding_rate", 0)

        if btc_funding > 0.0003:
            funding_sentiment = "EXTREME_GREED"
        elif btc_funding > 0.0001:
            funding_sentiment = "GREED"
        elif btc_funding > 0:
            funding_sentiment = "MILD_BULLISH"
        elif btc_funding > -0.0001:
            funding_sentiment = "MILD_BEARISH"
        elif btc_funding > -0.0003:
            funding_sentiment = "FEAR"
        else:
            funding_sentiment = "EXTREME_FEAR"

        # Tendência de funding (comparar com histórico)
        funding_trend = self._calculate_funding_trend(btc_funding)

        return {
            "per_symbol": per_symbol,
            "aggregated": {
                "avg_funding_rate": round(avg_funding, 8),
                "avg_funding_pct": round(avg_funding * 100, 4),
                "btc_funding_pct": round(btc_funding * 100, 4),
                "eth_funding_pct": round(eth_funding * 100, 4),
                "funding_sentiment": funding_sentiment,
                "funding_trend": funding_trend,
                "symbols_positive": sum(1 for r in funding_rates if r > 0),
                "symbols_negative": sum(1 for r in funding_rates if r < 0),
                "symbols_total": len(funding_rates),
            },
            "data_source": "binance_futures_premium_index",
            "is_real_data": True,
        }

    def _calculate_funding_trend(self, current_btc_funding: float) -> str:
        """Analisa tendência de funding comparando com histórico."""
        if len(self._history) < 2:
            return "insufficient_data"

        prev_data = self._history[-2].get("data", {})
        prev_btc = prev_data.get("aggregated", {}).get("btc_funding_pct", 0)
        curr = current_btc_funding * 100

        diff = curr - prev_btc
        if abs(diff) < 0.001:
            return "stable"
        elif diff > 0.005:
            return "rising_fast"
        elif diff > 0:
            return "rising"
        elif diff < -0.005:
            return "falling_fast"
        else:
            return "falling"