# fetchers/binance_positioning_fetcher.py
# -*- coding: utf-8 -*-
"""
Coletor Canônico de Posicionamento Binance USD-M Futures (API Pública).
Fase P1.1 (Arquitetura Context-Only).

Fonte intraday da Binance, independente do COT oficial semanal da CFTC/CME
(ver fetchers/cftc_cot_fetcher.py). "Global" = mercado geral Binance (todas
as contas; não equivale a "varejo" como fato). "Top Trader" = coorte Binance
por margem/volume (não equivale a "institucional"/"smart money" como fato).

Coleta e normaliza:
1. Global Long/Short Account Ratio (globalLongShortAccountRatio)
2. Top Trader Long/Short Account Ratio (topLongShortAccountRatio)
3. Top Trader Long/Short Position Ratio (topLongShortPositionRatio)
4. Open Interest History e Deltas Temporais 1h/4h (openInterestHist)

Garante:
- Separação semântica estrita entre account ratio e position ratio.
- Rastreamento de proveniência (source_timestamp, observed_at, age_seconds, is_stale).
- Fail-soft resiliente: erros e dados ausentes retornam None/UNKNOWN, nunca 0, 1.0 ou NEUTRAL.
- Cache com TTL de 300s (~5 minutos, alinhado à frequência de publicação da Binance).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("BinancePositioningFetcher")

_DEFAULT_CACHE_TTL = 300.0  # 5 minutos
_MAX_STALE_SECONDS = 900.0  # 15 minutos (após isso, dado é considerado STALE)
_REQUEST_TIMEOUT = 5.0      # 5 segundos por request


@dataclass
class BinancePositioningSnapshot:
    """Snapshot canônico normalizado de posicionamento Binance."""
    symbol: str
    period: str
    observed_at: float
    source: str = "binance_usdm"

    # 1. Global Account Ratio (Varejo + Geral)
    global_account_ratio: Optional[float] = None
    global_long_account_pct: Optional[float] = None
    global_short_account_pct: Optional[float] = None

    # 2. Top Trader Account Ratio (Top 20% contas)
    top_account_ratio: Optional[float] = None
    top_long_account_pct: Optional[float] = None
    top_short_account_pct: Optional[float] = None

    # 3. Top Trader Position Ratio (Top 20% volume financeiro)
    top_position_ratio: Optional[float] = None
    top_long_position_pct: Optional[float] = None
    top_short_position_pct: Optional[float] = None

    # 4. Open Interest e Deltas
    open_interest: Optional[float] = None
    open_interest_usd: Optional[float] = None
    open_interest_unit: str = "contracts"
    oi_delta_1h: Optional[float] = None  # Fração percentual relativa (ex: +0.024 = +2.4%)
    oi_delta_4h: Optional[float] = None  # Fração percentual relativa (ex: +0.051 = +5.1%)

    # 5. Divergências Derivadas (Top vs Global)
    top_account_vs_global: Optional[float] = None
    top_position_vs_global: Optional[float] = None

    # 6. Proveniência e Freshness
    # source_as_of  = instante da fonte (max source_timestamp, ISO UTC).
    # retrieved_at  = quando a resposta foi recebida/coletada (ISO UTC).
    # observed_at   = início da coleta (epoch float, legado).
    # analyzed_at   = quando a análise rodou (CryptoCOTAnalysis.observed_at).
    # retrieved_at NUNCA é preenchido com analyzed_at.
    source_timestamp: Optional[int] = None
    source_as_of: Optional[str] = None  # ISO UTC do source_timestamp (quando há)
    retrieved_at: Optional[str] = None  # ISO UTC do recebimento (quando há)
    age_seconds: Optional[float] = None
    is_stale: bool = False
    is_available: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converte snapshot para dicionário limpo para persistência e logging."""
        return asdict(self)


def _safe_float(val: Any) -> Optional[float]:
    """Converte valor para float finito, ou None se inválido."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (ValueError, TypeError):
        pass
    return None


def _safe_ratio(val: Any) -> Optional[float]:
    """Ratio L/S válido (>= 0) ou None. Negativo é semanticamente inválido."""
    f = _safe_float(val)
    if f is None or f < 0:
        return None
    return f


def _safe_level(val: Any) -> Optional[float]:
    """Nível (ex. open interest) válido (>= 0) ou None."""
    f = _safe_float(val)
    if f is None or f < 0:
        return None
    return f


class BinancePositioningFetcher:
    """
    Cliente assíncrono para coleta de dados de posicionamento da Binance Futures.
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        cache_ttl: float = _DEFAULT_CACHE_TTL,
        max_stale_seconds: float = _MAX_STALE_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_ttl = cache_ttl
        self.max_stale_seconds = max_stale_seconds

        # Cache em memória: symbol -> (timestamp, snapshot)
        self._cache: Dict[str, tuple[float, BinancePositioningSnapshot]] = {}
        self._lock = asyncio.Lock()

    def get_cached(self, symbol: str = "BTCUSDT") -> Optional[BinancePositioningSnapshot]:
        """Retorna snapshot em cache se ainda não expirou."""
        if symbol in self._cache:
            ts, snapshot = self._cache[symbol]
            now = time.time()
            if now - ts < self.cache_ttl:
                # Atualiza age_seconds e is_stale dinamicamente
                if snapshot.source_timestamp:
                    age = now - (snapshot.source_timestamp / 1000.0)
                    snapshot.age_seconds = round(age, 1)
                    snapshot.is_stale = age > self.max_stale_seconds
                return snapshot
        return None

    async def fetch_positioning(
        self,
        symbol: str = "BTCUSDT",
        session: Optional[aiohttp.ClientSession] = None,
        force_refresh: bool = False,
    ) -> BinancePositioningSnapshot:
        """
        Busca todos os indicadores de posicionamento da Binance Futures.
        Usa cache de 5 minutos caso disponível e válido.
        """
        if not force_refresh:
            cached = self.get_cached(symbol)
            if cached:
                return cached

        async with self._lock:
            # Dupla checagem sob lock
            if not force_refresh:
                cached = self.get_cached(symbol)
                if cached:
                    return cached

            own_session = session is None
            if own_session:
                connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    connector=connector,
                    headers={"User-Agent": "MarketBot-Positioning/1.0"},
                )

            try:
                snapshot = await self._fetch_all_endpoints(session, symbol)
                now = time.time()
                snapshot.retrieved_at = datetime.fromtimestamp(
                    now, tz=timezone.utc).isoformat()
                self._cache[symbol] = (now, snapshot)
                return snapshot
            finally:
                if own_session and session:
                    await session.close()

    async def _fetch_single_endpoint(
        self,
        session: aiohttp.ClientSession,
        path: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Busca um endpoint individual com retry limitado e timeout estrito."""
        url = f"{self.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT)
        max_retries = 2

        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                            return data["data"]
                        return []
                    elif resp.status == 429:
                        logger.warning(f"Binance Positioning HTTP 429 (Rate Limit) em {path}")
                        return []
                    elif resp.status >= 500:
                        logger.debug(f"Binance Positioning HTTP {resp.status} em {path} (tentativa {attempt+1})")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        return []
                    else:
                        logger.warning(f"Binance Positioning HTTP {resp.status} em {path}")
                        return []
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.debug(f"Falha de conexão em {path} (tentativa {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return []
            except Exception as e:
                logger.warning(f"Erro inesperado ao buscar {path}: {e}")
                return []
        return []

    async def _fetch_all_endpoints(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> BinancePositioningSnapshot:
        """Dispara requisições simultâneas para os 4 endpoints e consolida os dados."""
        now = time.time()
        period = "5m"
        limit = 60  # Para cobrir 4 horas (48 barras de 5m)

        tasks = [
            self._fetch_single_endpoint(
                session, "/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": limit}
            ),
            self._fetch_single_endpoint(
                session, "/futures/data/topLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": limit}
            ),
            self._fetch_single_endpoint(
                session, "/futures/data/topLongShortPositionRatio",
                {"symbol": symbol, "period": period, "limit": limit}
            ),
            self._fetch_single_endpoint(
                session, "/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit}
            ),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        global_acc_data = results[0] if isinstance(results[0], list) else []
        top_acc_data = results[1] if isinstance(results[1], list) else []
        top_pos_data = results[2] if isinstance(results[2], list) else []
        oi_hist_data = results[3] if isinstance(results[3], list) else []

        snapshot = BinancePositioningSnapshot(
            symbol=symbol,
            period=period,
            observed_at=now,
            source="binance_usdm",
        )

        source_timestamps = []

        # 1. Global Account Ratio
        if global_acc_data:
            latest = global_acc_data[-1]
            snapshot.global_account_ratio = _safe_ratio(latest.get("longShortRatio"))
            snapshot.global_long_account_pct = _safe_float(latest.get("longAccount"))
            snapshot.global_short_account_pct = _safe_float(latest.get("shortAccount"))
            if latest.get("timestamp"):
                source_timestamps.append(int(latest["timestamp"]))

        # 2. Top Trader Account Ratio
        if top_acc_data:
            latest = top_acc_data[-1]
            snapshot.top_account_ratio = _safe_ratio(latest.get("longShortRatio"))
            snapshot.top_long_account_pct = _safe_float(latest.get("longAccount"))
            snapshot.top_short_account_pct = _safe_float(latest.get("shortAccount"))
            if latest.get("timestamp"):
                source_timestamps.append(int(latest["timestamp"]))

        # 3. Top Trader Position Ratio
        if top_pos_data:
            latest = top_pos_data[-1]
            snapshot.top_position_ratio = _safe_ratio(latest.get("longShortRatio"))
            # Binance pode retornar 'longPosition' ou 'longAccount' neste endpoint
            l_pos = latest.get("longPosition") or latest.get("longAccount")
            s_pos = latest.get("shortPosition") or latest.get("shortAccount")
            snapshot.top_long_position_pct = _safe_float(l_pos)
            snapshot.top_short_position_pct = _safe_float(s_pos)
            if latest.get("timestamp"):
                source_timestamps.append(int(latest["timestamp"]))

        # 4. Open Interest e Deltas
        if oi_hist_data:
            latest = oi_hist_data[-1]
            oi_cur = _safe_level(latest.get("sumOpenInterest"))
            oi_cur_usd = _safe_level(latest.get("sumOpenInterestValue"))
            snapshot.open_interest = oi_cur
            snapshot.open_interest_usd = oi_cur_usd
            if latest.get("timestamp"):
                source_timestamps.append(int(latest["timestamp"]))

            # Delta 1h: 12 barras de 5m
            if len(oi_hist_data) >= 13 and oi_cur is not None:
                past_1h = oi_hist_data[-13]
                oi_1h_base = _safe_level(past_1h.get("sumOpenInterest"))
                if oi_1h_base and oi_1h_base > 0:
                    snapshot.oi_delta_1h = round((oi_cur - oi_1h_base) / oi_1h_base, 4)

            # Delta 4h: 48 barras de 5m
            if len(oi_hist_data) >= 49 and oi_cur is not None:
                past_4h = oi_hist_data[-49]
                oi_4h_base = _safe_level(past_4h.get("sumOpenInterest"))
                if oi_4h_base and oi_4h_base > 0:
                    snapshot.oi_delta_4h = round((oi_cur - oi_4h_base) / oi_4h_base, 4)

        # 5. Divergências Derivadas
        if snapshot.top_account_ratio is not None and snapshot.global_account_ratio is not None:
            snapshot.top_account_vs_global = round(
                snapshot.top_account_ratio - snapshot.global_account_ratio, 4
            )

        if snapshot.top_position_ratio is not None and snapshot.global_account_ratio is not None:
            snapshot.top_position_vs_global = round(
                snapshot.top_position_ratio - snapshot.global_account_ratio, 4
            )

        # 6. Freshness e Disponibilidade
        if source_timestamps:
            max_ts = max(source_timestamps)
            snapshot.source_timestamp = max_ts
            snapshot.source_as_of = datetime.fromtimestamp(
                max_ts / 1000.0, tz=timezone.utc).isoformat()
            age = now - (max_ts / 1000.0)
            snapshot.age_seconds = max(0.0, round(age, 1))
            snapshot.is_stale = age > self.max_stale_seconds
            snapshot.is_available = True
        else:
            snapshot.is_available = False
            snapshot.error = "no_data_from_endpoints"

        return snapshot
