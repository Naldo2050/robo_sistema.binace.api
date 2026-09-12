# institutional/session_vwap.py
# -*- coding: utf-8 -*-
"""
Session VWAP Canônico Ancorado em UTC 00:00:00.
Fase P1.2 (Arquitetura Context-Only).

Implementa:
1. Session VWAP diário ancorado estritamente em UTC 00:00:00.000.
2. Atualização incremental O(1) de acumuladores sum(Price * Volume) e sum(Volume).
3. Reconstrução determinística pós-restart via klines de 1m da Binance Futures.
4. Transição automática de sessão em 00:00 UTC sem dependência do fuso horário local.
5. Rastreamento rigoroso de proveniência (method='ohlcv_1m_typical_price', session_start, status).
6. Distância normalizada em fração decimal: (price - session_vwap) / session_vwap.

RESTRIÇÃO ARQUITETURAL:
Módulo estritamente CONTEXT-ONLY. Não gera sinais direcionais nem altera execução de trade.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

logger = logging.getLogger("SessionVWAP")

_MS_PER_DAY = 86_400_000  # 24 * 60 * 60 * 1000 ms
_MAX_STALE_SECONDS = 300.0  # 5 minutos sem novas barras torna o dado stale


class SessionVWAPStatus(str, Enum):
    """Status operacional do Session VWAP."""
    VALID = "VALID"
    WARMING_UP = "WARMING_UP"
    STALE = "STALE"
    ERROR = "ERROR"
    # Cobertura incompleta mas fresca: valor utilizável, NÃO rotular de sessão cheia.
    PARTIAL = "PARTIAL"
    # Único ausente é exatamente o último minuto fechado, ainda não transferido
    # ao tracker pela ordem do pipeline (enrich antes de update). Não é perda.
    PIPELINE_PENDING = "PIPELINE_PENDING"


@dataclass
class SessionVWAPSnapshot:
    """Snapshot canônico de Session VWAP com proveniência completa."""
    symbol: str
    session_vwap: Optional[float]
    current_price: Optional[float]
    distance_fraction: Optional[float]  # (price - vwap) / vwap
    side: str  # "ABOVE", "BELOW", "AT", "UNKNOWN"
    session_start_ms: int
    session_start_iso: str
    accumulated_volume: float
    accumulated_quote_volume: float
    bars_count: int
    status: SessionVWAPStatus
    method: str = "ohlcv_1m_typical_price"
    observed_at: float = field(default_factory=time.time)
    last_candle_ms: Optional[int] = None
    age_seconds: Optional[float] = None
    is_valid: bool = False
    # Cobertura da sessão (minutos distintos recebidos vs esperados).
    first_candle_ms: Optional[int] = None
    expected_bars: int = 0
    missing_bars: int = 0
    coverage_pct: Optional[float] = None
    coverage_status: str = "UNAVAILABLE"  # FULL | PARTIAL | UNAVAILABLE
    # Minutos ausentes (ordenados, capados) para identidade exata do gap.
    # Permite distinguir pendência estrutural (só o último fechado) de perda real.
    missing_minutes: list = field(default_factory=list)
    missing_minutes_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialização amigável em JSON."""
        d = asdict(self)
        d["status"] = self.status.value
        return d


def get_utc_session_start_ms(timestamp_ms: Optional[int] = None) -> int:
    """
    Retorna o epoch timestamp (ms) exato de UTC 00:00:00.000 do dia correspondente.
    Totalmente agnóstico ao fuso horário local do host.
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    return (timestamp_ms // _MS_PER_DAY) * _MS_PER_DAY


class SessionVWAPTracker:
    """
    Rastreador incremental O(1) de Session VWAP ancorado em UTC 00:00:00.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        base_url: str = "https://fapi.binance.com",
        max_stale_seconds: float = _MAX_STALE_SECONDS,
    ):
        self.symbol = symbol
        self.base_url = base_url.rstrip("/")
        self.max_stale_seconds = max_stale_seconds

        # Estado da Sessão Atual
        self._session_start_ms: int = get_utc_session_start_ms()
        self._sum_pv: float = 0.0       # Σ(Typical Price × Volume)
        self._sum_vol: float = 0.0      # Σ(Volume)
        self._bars_count: int = 0
        self._last_candle_ms: Optional[int] = None
        self._status: SessionVWAPStatus = SessionVWAPStatus.WARMING_UP
        # Minutos distintos (ts // 60000) aceitos na sessão atual.
        # Base da cobertura FULL/PARTIAL; limitado a ~1440/dia, limpo no reset.
        self._seen_minutes: set = set()
        self._lock = asyncio.Lock()

    @property
    def session_start_ms(self) -> int:
        return self._session_start_ms

    @property
    def current_vwap(self) -> Optional[float]:
        """Retorna o valor atual do VWAP se o volume acumulado for positivo."""
        if self._sum_vol > 0 and math.isfinite(self._sum_pv):
            val = self._sum_pv / self._sum_vol
            return round(val, 2) if math.isfinite(val) else None
        return None

    def reset_session(self, new_session_start_ms: Optional[int] = None) -> None:
        """Reinicia os acumuladores para uma nova sessão UTC."""
        if new_session_start_ms is None:
            new_session_start_ms = get_utc_session_start_ms()
        self._session_start_ms = new_session_start_ms
        self._sum_pv = 0.0
        self._sum_vol = 0.0
        self._bars_count = 0
        self._last_candle_ms = None
        self._seen_minutes = set()
        self._status = SessionVWAPStatus.WARMING_UP
        logger.info(
            f"Session VWAP resetado para nova sessão UTC: "
            f"{datetime.fromtimestamp(new_session_start_ms / 1000.0, timezone.utc).isoformat()}"
        )

    def _check_session_boundary(self, candle_timestamp_ms: int) -> None:
        """Verifica se a barra pertence à sessão atual ou se cruzou UTC 00:00."""
        expected_session_start = get_utc_session_start_ms(candle_timestamp_ms)
        if expected_session_start > self._session_start_ms:
            # Rollover UTC detectado
            self.reset_session(expected_session_start)

    def update_candle(
        self,
        timestamp_ms: int,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> Optional[float]:
        """
        Atualiza o Session VWAP incrementalmente com um candle de 1 minuto em O(1).
        Usa Typical Price = (High + Low + Close) / 3.0.
        """
        # Validação defensiva de dados
        if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(close) and math.isfinite(volume)):
            return self.current_vwap

        if high <= 0 or low <= 0 or close <= 0 or high < low:
            return self.current_vwap

        # Checagem de fronteira de sessão UTC
        self._check_session_boundary(timestamp_ms)

        # Se a barra for anterior ao início da sessão atual, ignora
        if timestamp_ms < self._session_start_ms:
            return self.current_vwap

        # Se a barra for duplicada ou fora de ordem (já processada), ignora
        if self._last_candle_ms is not None and timestamp_ms <= self._last_candle_ms:
            return self.current_vwap

        # Minuto vazio genuíno (OHLC válido, volume zero): peso zero no VWAP,
        # mas conta para cobertura (não é dado ausente).
        if volume <= 0:
            self._seen_minutes.add(timestamp_ms // 60000)
            return self.current_vwap

        typical_price = (high + low + close) / 3.0
        pv = typical_price * volume

        self._sum_pv += pv
        self._sum_vol += volume
        self._bars_count += 1
        self._last_candle_ms = timestamp_ms
        self._seen_minutes.add(timestamp_ms // 60000)

        if self._bars_count > 0 and self._status == SessionVWAPStatus.WARMING_UP:
            self._status = SessionVWAPStatus.VALID

        return self.current_vwap

    def update_batch(self, candles: List[Dict[str, Any]]) -> Optional[float]:
        """
        Processa um lote de candles ordenados cronologicamente (ex: reconstrução).
        """
        for c in candles:
            # Suporta dict ou kline Binance list: [t, o, h, l, c, v, T, q, n, ...]
            if isinstance(c, dict):
                ts = int(c.get("open_time") or c.get("timestamp") or c.get("t", 0))
                h = float(c.get("high") or c.get("h", 0))
                l = float(c.get("low") or c.get("l", 0))
                close = float(c.get("close") or c.get("c", 0))
                v = float(c.get("volume") or c.get("v", 0))
            elif isinstance(c, (list, tuple)) and len(c) >= 6:
                ts = int(c[0])
                h = float(c[2])
                l = float(c[3])
                close = float(c[4])
                v = float(c[5])
            else:
                continue

            self.update_candle(ts, h, l, close, v)

        return self.current_vwap

    async def rebuild_from_binance(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        now_ms: Optional[int] = None,
    ) -> bool:
        """
        Reconstrói os acumuladores desde UTC 00:00:00 via API REST da Binance Futures.
        Garante que o Session VWAP não comece do zero em caso de restart intraday.
        """
        async with self._lock:
            if now_ms is None:
                now_ms = int(time.time() * 1000)
            session_start = get_utc_session_start_ms(now_ms)
            self.reset_session(session_start)

            url = f"{self.base_url}/fapi/v1/klines"
            own_session = session is None
            if own_session:
                session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=15.0),
                    headers={"User-Agent": "MarketBot-SessionVWAP/1.0"}
                )

            current_start_ms = session_start
            total_fetched = 0

            try:
                # Paginação com limit=1000 para cobrir até 1440 barras de 24h com segurança
                while current_start_ms < now_ms:
                    params = {
                        "symbol": self.symbol,
                        "interval": "1m",
                        "startTime": current_start_ms,
                        "endTime": now_ms,
                        "limit": 1000,
                    }

                    async with session.get(url, params=params) as resp:
                        if resp.status != 200:
                            logger.warning(f"Rebuild Session VWAP HTTP {resp.status} em {url}")
                            self._status = SessionVWAPStatus.ERROR
                            return False

                        klines = await resp.json()
                        if not isinstance(klines, list) or len(klines) == 0:
                            break

                        self.update_batch(klines)
                        total_fetched += len(klines)

                        # Último timestamp recebido para avançar a paginação
                        last_ts = int(klines[-1][0])
                        if last_ts <= current_start_ms:
                            break
                        current_start_ms = last_ts + 60000  # Próximo minuto

                        if len(klines) < 1000:
                            # Chegou ao fim das klines disponíveis
                            break

                # Validação de cobertura da sessão:
                # Se a sessão já decorreu N minutos, precisamos ter pelo menos 90% das barras
                elapsed_minutes = max(1, (now_ms - session_start) // 60000)
                min_required_bars = max(1, int(elapsed_minutes * 0.90))

                if self._bars_count >= min_required_bars:
                    self._status = SessionVWAPStatus.VALID
                    logger.info(
                        f"Session VWAP reconstruído com sucesso para {self.symbol}: "
                        f"{self._bars_count}/{elapsed_minutes} barras de 1m desde 00:00 UTC | VWAP={self.current_vwap}"
                    )
                    return True
                else:
                    self._status = SessionVWAPStatus.WARMING_UP
                    logger.warning(
                        f"Session VWAP cobertura insuficiente: {self._bars_count}/{elapsed_minutes} barras. "
                        f"Mantendo status WARMING_UP."
                    )
                    return False

            except Exception as e:
                logger.warning(f"Falha ao reconstruir Session VWAP via Binance: {e}")
                self._status = SessionVWAPStatus.ERROR
                return False
            finally:
                if own_session and session:
                    await session.close()

    def _coverage(self, now_ms: int) -> Dict[str, Any]:
        """Cobertura da sessão em minutos distintos recebidos vs esperados.

        Esperado = minutos fechados em [session_start, último minuto fechado].
        O minuto corrente (ainda aberto) nunca conta como esperado.
        Retorna dict com expected_bars, missing_bars, coverage_pct,
        coverage_status (FULL|PARTIAL|UNAVAILABLE), first_candle_ms e
        missing_minutes (lista ordenada, capada em 32 + flag de truncamento).
        """
        last_closed = (now_ms // 60000 - 1) * 60000
        if last_closed < self._session_start_ms:
            return {"expected_bars": 0, "missing_bars": 0, "coverage_pct": None,
                    "coverage_status": "UNAVAILABLE", "first_candle_ms": None,
                    "missing_minutes": [], "missing_minutes_truncated": False,
                    "last_closed_ms": last_closed}
        expected = (last_closed - self._session_start_ms) // 60000 + 1
        seen = {m for m in self._seen_minutes
                if self._session_start_ms <= m * 60000 <= last_closed}
        missing = sorted(m * 60000 for m in range(
            self._session_start_ms // 60000, last_closed // 60000 + 1)
            if m not in seen)
        received = expected - len(missing)
        return {
            "expected_bars": expected,
            "missing_bars": len(missing),
            "coverage_pct": round(received / expected * 100, 1),
            "coverage_status": "FULL" if not missing else "PARTIAL",
            "first_candle_ms": min(seen) * 60000 if seen else None,
            "missing_minutes": missing[:32],
            "missing_minutes_truncated": len(missing) > 32,
            "last_closed_ms": last_closed,
        }

    def get_snapshot(self, current_price: Optional[float] = None,
                     now_ms: Optional[int] = None) -> SessionVWAPSnapshot:
        """
        Gera snapshot com métricas de distância, lado e frescor.
        """
        now_wall = time.time()
        if now_ms is None:
            now_ms = int(now_wall * 1000)

        # Checagem de virada de dia se o relógio passou de 00:00 UTC sem candles
        self._check_session_boundary(now_ms)

        vwap_val = self.current_vwap
        age_seconds = None
        status = self._status

        if self._last_candle_ms is not None:
            age_seconds = max(0.0, round(now_ms / 1000.0 - (self._last_candle_ms / 1000.0), 1))
            if age_seconds > self.max_stale_seconds and status == SessionVWAPStatus.VALID:
                status = SessionVWAPStatus.STALE

        # Completude: fresca mas incompleta => PARTIAL (nunca FULL silencioso).
        # WARMING_UP (sem barras) e ERROR/STALE permanecem como estão.
        cov = self._coverage(now_ms)
        if status == SessionVWAPStatus.VALID and cov["coverage_status"] == "PARTIAL":
            # PIPELINE_PENDING somente se o ÚNICO ausente for exatamente o
            # último minuto fechado (candle recém-fechada ainda não transferida
            # ao tracker pela ordem enrich-antes-de-update). Qualquer outro
            # padrão (gap antigo, múltiplos) permanece PARTIAL — nunca inferir
            # a partir de contagem (missing==1 sozinho NÃO basta).
            if (not cov["missing_minutes_truncated"]
                    and cov["missing_minutes"] == [cov["last_closed_ms"]]):
                status = SessionVWAPStatus.PIPELINE_PENDING
            else:
                status = SessionVWAPStatus.PARTIAL

        # Cálculo de distância fracionária: (P - VWAP) / VWAP
        dist_fraction = None
        side = "UNKNOWN"
        is_valid = (status in (SessionVWAPStatus.VALID,
                               SessionVWAPStatus.PARTIAL,
                               SessionVWAPStatus.PIPELINE_PENDING)
                    ) and (vwap_val is not None)

        if vwap_val and current_price and math.isfinite(current_price) and current_price > 0:
            dist_fraction = round((current_price - vwap_val) / vwap_val, 4)
            if dist_fraction > 0.0005:
                side = "ABOVE"
            elif dist_fraction < -0.0005:
                side = "BELOW"
            else:
                side = "AT"

        session_iso = datetime.fromtimestamp(
            self._session_start_ms / 1000.0, timezone.utc
        ).strftime("%Y-%m-%dT00:00:00Z")

        return SessionVWAPSnapshot(
            symbol=self.symbol,
            session_vwap=vwap_val,
            current_price=round(current_price, 2) if current_price else None,
            distance_fraction=dist_fraction,
            side=side,
            session_start_ms=self._session_start_ms,
            session_start_iso=session_iso,
            accumulated_volume=round(self._sum_vol, 4),
            accumulated_quote_volume=round(self._sum_pv, 2),
            bars_count=self._bars_count,
            status=status,
            method="ohlcv_1m_typical_price",
            observed_at=now_wall,
            last_candle_ms=self._last_candle_ms,
            age_seconds=age_seconds,
            is_valid=is_valid,
            first_candle_ms=cov["first_candle_ms"],
            expected_bars=cov["expected_bars"],
            missing_bars=cov["missing_bars"],
            coverage_pct=cov["coverage_pct"],
            coverage_status=cov["coverage_status"],
            missing_minutes=cov["missing_minutes"],
            missing_minutes_truncated=cov["missing_minutes_truncated"],
        )
