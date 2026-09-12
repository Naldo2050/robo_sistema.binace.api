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
    age = monotonic_now - fetched_monotonic (por grupo). `fetched_at_ms`
    (wall-clock UTC, por grupo) serve só para auditoria. `last_error` é
    interno (logs/métricas).
    """

    fast: Dict[str, Any] = field(default_factory=dict)
    slow: Dict[str, Any] = field(default_factory=dict)
    fast_fetched_at_ms: Optional[int] = None
    slow_fetched_at_ms: Optional[int] = None
    fast_fetched_monotonic: Optional[float] = None
    slow_fetched_monotonic: Optional[float] = None
    last_error: Optional[str] = None
    # P04: proveniência por campo (VALID/REAL_ZERO/MISSING/API_ERROR).
    # Interno como last_error (nunca número); exposto na view para o evento.
    field_status: Dict[str, str] = field(default_factory=dict)


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
        # PF-S1: owner loop/task refs para cancelamento real no owner loop.
        # Guardados por _lifecycle_lock; session.close() SEMPRE no owner loop
        # (finally de _amain), nunca de thread estranha.
        self._lifecycle_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
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

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
        """PF-S1: cancelamento real do refresh async + join bounded.

        Retorna status estruturado honesto (nunca finge sucesso):
        {"clean_shutdown": bool, "thread_alive": bool, "elapsed_ms": int}.
        Idempotente: stop duplo / nunca iniciado não quebra.
        """
        start = time.monotonic()
        self._stop_requested.set()
        with self._lifecycle_lock:
            thread = self._thread
            self._thread = None
            loop = self._loop
            task = self._main_task
        if task is not None and loop is not None:
            try:
                if not task.done():
                    loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # Loop já fechado: thread vai sair sozinha via finally.
                pass
            except Exception as e:
                logger.debug(f"OnchainUpdater cancel falhou (ignorado): {e}")
        alive = False
        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)
                alive = thread.is_alive()
            else:
                alive = True
        elapsed_ms = int((time.monotonic() - start) * 1000)
        clean = not alive
        if thread is not None:
            if alive:
                logger.warning("⚠️ OnchainUpdater thread não parou em %.1fs "
                               "(elapsed=%dms, cancel solicitado)",
                               timeout, elapsed_ms)
            else:
                logger.info("🛑 OnchainUpdater parado (elapsed=%dms)", elapsed_ms)
        return {"clean_shutdown": clean, "thread_alive": alive,
                "elapsed_ms": elapsed_ms}

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lifecycle_lock:
            # stop() pode ter sido chamado antes do loop ficar ready:
            # registra o loop cedo para o cancel alcançar a task.
            self._loop = loop
            self._main_task = None
        if self._stop_requested.is_set():
            with self._lifecycle_lock:
                self._loop = None
            try:
                loop.close()
            except Exception:
                pass
            return
        task = loop.create_task(self._amain())
        with self._lifecycle_lock:
            self._main_task = task
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            # Cancelamento via stop(): caminho esperado, não é falha.
            pass
        except Exception as e:
            logger.error(f"❌ OnchainUpdater loop falhou: {e}")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            with self._lifecycle_lock:
                self._loop = None
                self._main_task = None

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
        except asyncio.CancelledError:
            # stop() cancelou a task no owner loop: propaga para
            # run_until_complete observar o cancelamento; finally fecha
            # a sessão no MESMO loop (nunca de thread estranha).
            raise
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
        except asyncio.CancelledError:
            # Cancelamento de stop(): nunca engolir, nunca contar como falha.
            raise
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
        # P04: preserva proveniência por campo (fora de FAST/SLOW por desenho).
        field_status = merged.get("_field_status")
        self._store_snapshot(
            fast, slow,
            field_status=dict(field_status) if isinstance(field_status, dict) else {},
        )

    def _store_snapshot(self, fast: Dict[str, Any],
                        slow: Dict[str, Any],
                        field_status: Optional[Dict[str, str]] = None) -> None:
        wall_ms = int(time.time() * 1000)
        mono = self._monotonic_fn()
        snap = OnchainSnapshot(
            fast=dict(fast),
            slow=dict(slow),
            fast_fetched_at_ms=wall_ms,
            slow_fetched_at_ms=wall_ms,
            fast_fetched_monotonic=mono,
            slow_fetched_monotonic=mono,
            last_error=None,
            field_status=dict(field_status or {}),
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

        return {
            "fast": _group(
                snap.fast if snap else {},
                snap.fast_fetched_monotonic if snap else None,
                self.policy.fast_fresh_s, self.policy.fast_usable_s),
            "slow": _group(
                snap.slow if snap else {},
                snap.slow_fetched_monotonic if snap else None,
                self.policy.slow_fresh_s, self.policy.slow_usable_s),
            "fetched_at_ms": (snap.fast_fetched_at_ms if snap else None),
            "capabilities": {f: NEVER_EVIDENCE_REASON
                             for f in sorted(NEVER_EVIDENCE_FIELDS)},
            # P04: proveniência por campo (não é evidência numérica).
            "field_status": dict(snap.field_status) if snap and snap.field_status else {},
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
                (self._monotonic_fn() - snap.fast_fetched_monotonic)
                if snap and snap.fast_fetched_monotonic is not None else None
            ),
        }
