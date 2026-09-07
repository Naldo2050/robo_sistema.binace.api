# fetchers/onchain_updater.py
# -*- coding: utf-8 -*-
"""
Updater onchain fora do hot path da janela (FASE B).

Contrato:
  - HTTP onchain NUNCA roda no caminho da janela. Um único updater em
    background (thread + loop próprios) busca e publica snapshots.
  - A janela lê snapshotull via `read_view()` (µs, sem I/O, sem threads).
  - Publicação all-or-nothing: o leitor vê o snapshot completo anterior ou
    o novo completo — nunca parcial.
  - `last_error` é estado interno (logs/métricas). NUNCA vai ao payload/LLM.
  - Freshness por grupo (fast/slow) via `OnchainFreshnessPolicy` explícita;
    `age_seconds` (derivada de monotonic no leitor) é a verdade primária.

Grupos:
  - fast: fees/mempool (mudam a cada bloco, ~10min).
  - slow: difficulty/hash_rate e stats 24h (estáveis intradia).
  - never-evidence: exchange_netflow/whale_transactions/exchange_reserves/sopr
    (sem fonte real; NÃO entram no snapshot como número — fonte inexistente
    != valor zero; vão em `capabilities` como requires_paid_api).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import aiohttp

from fetchers.onchain_fetcher import OnchainFetcher

logger = logging.getLogger("OnchainUpdater")

# Campos que NUNCA são evidência observada (sem fonte real; só API paga).
NEVER_EVIDENCE_FIELDS = frozenset(
    {
        "exchange_netflow",
        "whale_transactions",
        "exchange_reserves",
        "sopr",
    }
)
NEVER_EVIDENCE_REASON = "requires_paid_api"

# Campos rápidos (mudam a cada bloco) vs lentos (estáveis intradia).
FAST_FIELDS = frozenset(
    {
        "fees_fastest_sat_vb",
        "fees_half_hour_sat_vb",
        "fees_hour_sat_vb",
        "fees_economy_sat_vb",
        "mempool_size",
        "mempool_vsize_mb",
        "mempool_total_fee_btc",
        "unconfirmed_txs",
    }
)
SLOW_FIELDS = frozenset(
    {
        "difficulty",
        "difficulty_adjustment",
        "hash_rate",
        "active_addresses",
        "total_btc_sent_24h",
        "total_fees_btc_24h",
        "trade_volume_btc_24h",
        "miner_flows",
        "minutes_between_blocks",
    }
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class OnchainFreshnessPolicy:
    """Política operacional INICIAL de freshness (segundos).

    Defaults documentados e ancorados:
      - fast fresh 300s / usable 1800s: fees/mempool movem-se por bloco
        (~10min); 300s preserva o TTL vigente; 1800s ≈ 3 blocos.
      - slow fresh 3600s / usable 21600s: difficulty retarget ~2 semanas,
        hash/receita estáveis intradia; stats 24h rolam devagar.
    Elegibilidade, não garantia de que o fenômeno não mudou.
    Configurável via env (ONCHAIN_FAST_FRESH_S, ONCHAIN_FAST_USABLE_S,
    ONCHAIN_SLOW_FRESH_S, ONCHAIN_SLOW_USABLE_S, ONCHAIN_REFRESH_INTERVAL_S).
    """

    fast_fresh_s: float = 300.0
    fast_usable_s: float = 1800.0
    slow_fresh_s: float = 3600.0
    slow_usable_s: float = 21600.0
    refresh_interval_s: float = 300.0

    @classmethod
    def from_env(cls) -> "OnchainFreshnessPolicy":
        return cls(
            fast_fresh_s=_env_float("ONCHAIN_FAST_FRESH_S", 300.0),
            fast_usable_s=_env_float("ONCHAIN_FAST_USABLE_S", 1800.0),
            slow_fresh_s=_env_float("ONCHAIN_SLOW_FRESH_S", 3600.0),
            slow_usable_s=_env_float("ONCHAIN_SLOW_USABLE_S", 21600.0),
            refresh_interval_s=_env_float("ONCHAIN_REFRESH_INTERVAL_S", 300.0),
        )


@dataclass
class OnchainSnapshot:
    """Snapshot imutável publicado pelo updater (all-or-nothing).

    Nunca armazena idade pré-computada: o leitor deriva
    age = monotonic_now - fetched_monotonic. `fetched_at_ms` (wall-clock UTC)
    serve só para auditoria. `last_error` é interno (logs/métricas).
    """

    fast: Dict[str, Any] = field(default_factory=dict)
    slow: Dict[str, Any] = field(default_factory=dict)
    fetched_at_ms: Optional[int] = None
    fetched_monotonic: Optional[float] = None
    last_error: Optional[str] = None


def _classify(age_s: Optional[float], fresh_s: float, usable_s: float) -> str:
    if age_s is None:
        return "warming_up"
    if age_s <= fresh_s:
        return "fresh"
    if age_s <= usable_s:
        return "stale"
    return "unavailable"


class OnchainUpdater:
    """Dono único do refresh onchain (1 instância por bot).

    Ciclo de vida: `start()` (não-bloqueante) / `stop()` (fecha task, sessão
    e thread). Janela usa só `read_view()`.
    """

    def __init__(
        self,
        policy: Optional[OnchainFreshnessPolicy] = None,
        fetcher: Optional[OnchainFetcher] = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.policy = policy or OnchainFreshnessPolicy.from_env()
        self._fetcher = fetcher or OnchainFetcher()
        self._monotonic_fn = monotonic_fn
        self._lock = threading.Lock()
        self._snapshot: Optional[OnchainSnapshot] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        # Métricas (observabilidade; last_error NÃO vai ao payload).
        self.updates_total = 0
        self.failures_total = 0
        self.last_duration_s: Optional[float] = None
        self.last_error: Optional[str] = None

    # -- ciclo de vida -------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run, name="onchain-updater", daemon=True
        )
        self._thread.start()
        logger.info("✅ OnchainUpdater iniciado (refresh a cada %.0fs)",
                    self.policy.refresh_interval_s)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_requested.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("⚠️ OnchainUpdater thread não parou em %.1fs",
                               timeout)
            else:
                logger.info("🛑 OnchainUpdater parado")

    def _run(self) -> None:
        try:
            asyncio.run(self._amain())
        except Exception as e:
            logger.error(f"❌ OnchainUpdater loop falhou: {e}")

    async def _amain(self) -> None:
        connector = aiohttp.TCPConnector(force_close=True,
                                         enable_cleanup_closed=True)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), connector=connector
        )
        try:
            while not self._stop_requested.is_set():
                await self._refresh_once_async(session)
                for _ in range(int(self.policy.refresh_interval_s)):
                    if self._stop_requested.is_set():
                        break
                    await asyncio.sleep(1.0)
        finally:
            await session.close()

    # -- refresh --------------------------------------------------------
    async def _refresh_once_async(self, session) -> bool:
        start = self._monotonic_fn()
        try:
            merged = await self._fetcher.fetch_all(session)
            self._store_snapshot_from_merged(merged or {})
            self.updates_total += 1
            self.last_duration_s = self._monotonic_fn() - start
            self.last_error = None
            return True
        except Exception as e:
            self.failures_total += 1
            self.last_duration_s = self._monotonic_fn() - start
            self.last_error = str(e)[:200]
            logger.warning(f"⚠️ Onchain refresh falhou (mantido snapshot anterior): {e}")
            return False

    def _refresh_once(self) -> bool:
        """Refresh síncrono (testes / tick manual). Abre sessão própria.

        Propositalmente SEM ThreadPoolExecutor: deve ser chamado fora de
        loop em execução (o updater usa `_refresh_once_async` no próprio loop).
        """
        async def _one() -> bool:
            connector = aiohttp.TCPConnector(force_close=True,
                                             enable_cleanup_closed=True)
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30), connector=connector
            )
            try:
                return await self._refresh_once_async(session)
            finally:
                await session.close()

        return asyncio.run(_one())

    def _store_snapshot_from_merged(self, merged: Dict[str, Any]) -> None:
        fast = {k: v for k, v in merged.items() if k in FAST_FIELDS}
        slow = {k: v for k, v in merged.items() if k in SLOW_FIELDS}
        self._store_snapshot(fast, slow)

    def _store_snapshot(self, fast: Dict[str, Any],
                        slow: Dict[str, Any]) -> None:
        snap = OnchainSnapshot(
            fast=dict(fast),
            slow=dict(slow),
            fetched_at_ms=int(time.time() * 1000),
            fetched_monotonic=self._monotonic_fn(),
            last_error=None,
        )
        with self._lock:
            self._snapshot = snap

    # -- leitura (hot path: sem I/O, sem threads) ------------------------
    def read_view(self) -> Dict[str, Any]:
        """Visão all-or-nothing para a janela. Nunca faz HTTP."""
        with self._lock:
            snap = self._snapshot
        now_mono = self._monotonic_fn()

        def _group(values: Dict[str, Any], fetched_mono: Optional[float],
                   fresh_s: float, usable_s: float) -> Dict[str, Any]:
            age = (now_mono - fetched_mono) if fetched_mono is not None else None
            return {
                "values": dict(values),
                "status": _classify(age, fresh_s, usable_s),
                "age_seconds": age,
            }

        fetched_at = snap.fetched_at_ms if snap else None
        fetched_mono = snap.fetched_monotonic if snap else None
        return {
            "fast": _group(snap.fast if snap else {}, fetched_mono,
                           self.policy.fast_fresh_s, self.policy.fast_usable_s),
            "slow": _group(snap.slow if snap else {}, fetched_mono,
                           self.policy.slow_fresh_s, self.policy.slow_usable_s),
            "fetched_at_ms": fetched_at,
            "capabilities": {f: NEVER_EVIDENCE_REASON
                             for f in sorted(NEVER_EVIDENCE_FIELDS)},
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Métricas internas (logs/observabilidade; last_error NÃO vai ao LLM)."""
        with self._lock:
            snap = self._snapshot
        return {
            "updates_total": self.updates_total,
            "failures_total": self.failures_total,
            "last_duration_s": self.last_duration_s,
            "last_error": self.last_error,
            "has_snapshot": snap is not None,
            "snapshot_age_s": (
                (self._monotonic_fn() - snap.fetched_monotonic)
                if snap and snap.fetched_monotonic is not None else None
            ),
        }
