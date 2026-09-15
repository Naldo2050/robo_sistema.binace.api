# fetchers/cftc_cot_updater.py
# -*- coding: utf-8 -*-
"""
Updater CFTC COT fora do hot path da janela.

Segue o padrão arquitetural do OnchainUpdater/CrossAssetUpdater:
  - HTTP/rede CFTC NUNCA roda no caminho da janela. Um único updater em
    background (thread + loop próprios) busca e publica snapshots.
  - A janela lê via `read_view()` (µs, sem I/O).
  - Publicação all-or-nothing: erro novo preserva o snapshot anterior;
    sua idade cresce naturalmente; a tentativa aparece só em observabilidade.
  - `last_error` é interno (logs/métricas). NUNCA vai ao payload/LLM.
  - Ausência CFTC nunca afeta Binance positioning (fontes independentes).

Refresh alinhado ao calendário semanal conhecido (sexta 15:30 ET):
default 6h — cobre a publicação semanal com folga sem agredir a API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from fetchers.cftc_cot_fetcher import CftcCotFetcher, SYMBOL_TO_CONTRACT
from institutional.cftc_cot import CftcCot

logger = logging.getLogger("CftcCotUpdater")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class CftcCotUpdater:
    """Dono único do refresh CFTC COT (1 instância por bot)."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        refresh_interval_s: Optional[float] = None,
        fetcher: Optional[CftcCotFetcher] = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.symbols = list(symbols) if symbols else ["BTCUSDT", "ETHUSDT"]
        # Valida mapa na construção (fail-fast em config, não em janela).
        for s in self.symbols:
            if s not in SYMBOL_TO_CONTRACT:
                raise ValueError(f"CftcCotUpdater: símbolo sem mapeamento CME: {s}")
        self.refresh_interval_s = (
            refresh_interval_s
            if refresh_interval_s is not None
            else _env_float("CFTC_REFRESH_INTERVAL_S", 21600.0)
        )
        self._fetcher = fetcher or CftcCotFetcher()
        self._cot = CftcCot()
        self._monotonic_fn = monotonic_fn
        self._lock = threading.Lock()
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self.updates_total = 0
        self.failures_total = 0
        self.last_duration_s: Optional[float] = None
        self.last_error: Optional[str] = None

    # -- ciclo de vida (mesmo padrão PF-S1 do OnchainUpdater) ------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run, name="cftc-cot-updater", daemon=True
        )
        self._thread.start()
        logger.info("✅ CftcCotUpdater iniciado (refresh a cada %.0fs)",
                    self.refresh_interval_s)

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
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
                pass
            except Exception as e:  # noqa: BLE001
                logger.debug(f"CftcCotUpdater cancel falhou (ignorado): {e}")
        alive = False
        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)
                alive = thread.is_alive()
            else:
                alive = True
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if thread is not None:
            if alive:
                logger.warning("⚠️ CftcCotUpdater thread não parou em %.1fs", timeout)
            else:
                logger.info("🛑 CftcCotUpdater parado (elapsed=%dms)", elapsed_ms)
        return {"clean_shutdown": not alive, "thread_alive": alive,
                "elapsed_ms": elapsed_ms}

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lifecycle_lock:
            self._loop = loop
            self._main_task = None
        if self._stop_requested.is_set():
            with self._lifecycle_lock:
                self._loop = None
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            return
        task = loop.create_task(self._amain())
        with self._lifecycle_lock:
            self._main_task = task
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ CftcCotUpdater loop falhou: {e}")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            with self._lifecycle_lock:
                self._loop = None
                self._main_task = None

    async def _amain(self) -> None:
        connector = aiohttp.TCPConnector(force_close=True,
                                         enable_cleanup_closed=True)
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), connector=connector,
            headers={"User-Agent": "MarketBot-CFTC-COT/1.0"},
        )
        try:
            while not self._stop_requested.is_set():
                await self._refresh_once_async(session)
                for _ in range(int(self.refresh_interval_s)):
                    if self._stop_requested.is_set():
                        break
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        finally:
            await session.close()

    # -- refresh --------------------------------------------------------
    async def _refresh_once_async(self, session) -> bool:
        start = self._monotonic_fn()
        ok_all = True
        for symbol in self.symbols:
            code, _ = SYMBOL_TO_CONTRACT[symbol]
            try:
                row, err = await self._fetcher.fetch_latest(code, session)
                now_iso = datetime.now(timezone.utc).isoformat()
                if row is None:
                    logger.warning("⚠️ CFTC COT %s sem linha (%s): preservado anterior",
                                   symbol, err)
                    ok_all = False
                    continue
                record, ing_err = self._fetcher.ingest_row(code, row, now_iso)
                if ing_err or record is None:
                    logger.warning("⚠️ CFTC COT %s linha inválida (%s)", symbol, ing_err)
                    ok_all = False
                    continue
                prev = self._fetcher.history(code)
                prev_row = None
                if len(prev) >= 2:
                    prev_row = prev[-2].raw
                snap = self._cot.analyze(
                    record.raw, symbol=symbol, contract_code=code,
                    first_seen_at=record.first_seen_at,
                    retrieved_at=record.retrieved_at,
                    prev_row=prev_row,
                )
                snap.provenance["cache_hit"] = False
                snap.provenance["content_hash"] = record.content_hash
                snap.provenance["revision"] = record.revision
                snap.quality["revision"] = record.revision
                snap.quality["is_revision"] = record.revision > 0
                with self._lock:
                    self._snapshots[symbol] = snap.to_dict()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - erro preserva anterior
                logger.warning("⚠️ CFTC COT refresh %s falhou (mantido anterior): %s",
                               symbol, e)
                ok_all = False
        self.last_duration_s = self._monotonic_fn() - start
        if ok_all:
            self.updates_total += 1
            self.last_error = None
        else:
            self.failures_total += 1
            if self.last_error is None:
                self.last_error = "partial_refresh_failure"
        return ok_all

    # -- leitura (hot path: sem I/O) ------------------------------------
    def read_view(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Visão all-or-nothing para a janela. Nunca faz HTTP."""
        with self._lock:
            snap = self._snapshots.get(symbol)
            return dict(snap) if snap else {}

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            symbols = sorted(self._snapshots.keys())
        return {
            "updates_total": self.updates_total,
            "failures_total": self.failures_total,
            "last_duration_s": self.last_duration_s,
            "last_error": self.last_error,
            "symbols": symbols,
        }
