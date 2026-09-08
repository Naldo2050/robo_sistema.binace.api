# market_analysis/cross_asset_updater.py
# -*- coding: utf-8 -*-
"""
Updater cross-asset fora do hot path da janela (FASE E3-B).

Segue o padrão arquitetural do OnchainUpdater (sem copiar detalhes
desnecessários: aqui os fetches são SÍNCRONOS, então não há sessão
async para gerenciar — só 1 thread dedicada).

Contrato:
  - HTTP/rede cross-asset (yfinance, klines Binance, TwelveData,
    AlphaVantage, CoinGecko, FRED) NUNCA roda no caminho da janela.
  - Publicação all-or-nothing: o leitor vê o último snapshot completo ou
    nada parcial. Refresh parcial/falho preserva o snapshot anterior; sua
    idade cresce naturalmente; a tentativa aparece só em observabilidade.
  - "Completo" = result["status"] == "ok" (veredito da própria
    get_enhanced_cross_asset_correlations). Placeholders deliberadamente
    None NÃO são exigidos para completude.
  - `last_error` é interno (logs/métricas). NUNCA vai ao payload/LLM.
  - Freshness: fresh <= 300s (preserva o TTL vigente); stale_usable <= 3600s
    (correlações 7d/30d/90d e valores macro movem-se devagar intradia).
    `age_seconds` (derivada de monotonic no leitor) é a verdade primária.

CONTRATO TEMPORAL (F5-C):
  A matemática agora é shared-session (closes nas mesmas datas, retornos após
  alinhamento, só sessões com availability <= decision). Cada snapshot carrega
  correlation_method/correlation_contract_version (2) + n/instrumento por
  campo; linhas sem a chave são positional_v1 legado — NÃO misturar em treino.
  CROSS_ASSET_TEMPORAL_AUDIT segue "pending": features cross-asset continuam
  BLOQUEADAS para treino ML (uso atual: leitura pela IA com method/n visível).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("CrossAssetUpdater")

# Auditoria temporal pendente: NÃO usar cross-asset para treino ML até E4.
CROSS_ASSET_TEMPORAL_AUDIT = "pending"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class CrossAssetFreshnessPolicy:
    """Política operacional INICIAL (segundos).

    fresh 300s preserva o TTL vigente (_CORR_CACHE_TTL). stale_usable 3600s:
    correlações 7d/30d/90d de closes diários movem-se de forma negligível
    intradia; valores pontuais (vix/gold/dominance) degradam em horas, não
    minutos. Elegibilidade, não garantia. Env: CROSS_FRESH_S,
    CROSS_USABLE_S, CROSS_REFRESH_INTERVAL_S.
    """

    fresh_s: float = 300.0
    usable_s: float = 3600.0
    refresh_interval_s: float = 300.0

    @classmethod
    def from_env(cls) -> "CrossAssetFreshnessPolicy":
        return cls(
            fresh_s=_env_float("CROSS_FRESH_S", 300.0),
            usable_s=_env_float("CROSS_USABLE_S", 3600.0),
            refresh_interval_s=_env_float("CROSS_REFRESH_INTERVAL_S", 300.0),
        )


@dataclass
class CrossAssetSnapshot:
    """Snapshot imutável (all-or-nothing). Sem idade pré-computada."""

    result: Dict[str, Any] = field(default_factory=dict)
    fetched_at_ms: Optional[int] = None
    fetched_monotonic: Optional[float] = None
    last_error: Optional[str] = None


def _classify(age_s: Optional[float], fresh_s: float,
              usable_s: float) -> str:
    if age_s is None:
        return "warming_up"
    if age_s <= fresh_s:
        return "fresh"
    if age_s <= usable_s:
        return "stale"
    return "unavailable"


class CrossAssetUpdater:
    """Dono único do refresh cross-asset (1 instância por bot)."""

    def __init__(
        self,
        policy: Optional[CrossAssetFreshnessPolicy] = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.policy = policy or CrossAssetFreshnessPolicy.from_env()
        self._monotonic_fn = monotonic_fn
        self._lock = threading.Lock()
        self._snapshot: Optional[CrossAssetSnapshot] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self.updates_total = 0
        self.partial_total = 0
        self.failures_total = 0
        self.last_duration_s: Optional[float] = None
        self.last_error: Optional[str] = None

    # -- ciclo de vida -------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run, name="crossasset-updater", daemon=True
        )
        self._thread.start()
        logger.info("✅ CrossAssetUpdater iniciado (refresh a cada %.0fs)",
                    self.policy.refresh_interval_s)

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
        """PF-S2: shutdown cooperativo + join bounded, sem matar thread.

        Marca stop_requested (nenhuma nova operação externa começa) e aguarda
        no máximo `timeout`. I/O sync já em andamento pode terminar no próprio
        timeout; nesse caso retorna clean=False (honesto, nunca finge).
        Join NÃO foi aumentado para 60/180s como solução.
        Idempotente: stop duplo / nunca iniciado não quebra.
        """
        start = self._monotonic_fn()
        self._stop_requested.set()
        thread, self._thread = self._thread, None
        alive = False
        if thread is not None and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)
                alive = thread.is_alive()
            else:
                alive = True
        elapsed_ms = int((self._monotonic_fn() - start) * 1000)
        clean = not alive
        if thread is not None:
            if alive:
                logger.warning("⚠️ CrossAssetUpdater thread não parou em %.1fs "
                               "(elapsed=%dms, I/O sync em andamento até "
                               "próprio timeout)", timeout, elapsed_ms)
            else:
                logger.info("🛑 CrossAssetUpdater parado (elapsed=%dms)",
                            elapsed_ms)
        return {"clean_shutdown": clean, "thread_alive": alive,
                "elapsed_ms": elapsed_ms}

    def _run(self) -> None:
        try:
            while not self._stop_requested.is_set():
                try:
                    self._refresh_once()
                except Exception as e:
                    logger.debug(f"CrossAsset refresh erro (ignorado): {e}")
                for _ in range(int(self.policy.refresh_interval_s)):
                    if self._stop_requested.is_set():
                        break
                    time.sleep(1.0)
        except Exception as e:
            logger.error(f"❌ CrossAssetUpdater loop falhou: {e}")

    # -- refresh --------------------------------------------------------
    def _refresh_once(self) -> bool:
        """Uma tentativa. True só com snapshot completo novo publicado.

        PF-S2: propaga stop_event para os fetches; "cancelled" preserva o
        snapshot anterior (como parcial), sem iniciar nova operação externa.
        """
        from market_analysis.cross_asset_correlations import (
            get_enhanced_cross_asset_correlations,
        )

        start = self._monotonic_fn()
        try:
            result = get_enhanced_cross_asset_correlations(
                stop_event=self._stop_requested)
        except Exception as e:
            self.failures_total += 1
            self.last_duration_s = self._monotonic_fn() - start
            self.last_error = str(e)[:200]
            logger.warning(f"⚠️ CrossAsset refresh falhou (mantido anterior): {e}")
            return False
        if isinstance(result, dict) and result.get("status") == "cancelled":
            # Stop pedido no meio do refresh: sem snapshot novo, sem erro.
            self.last_duration_s = self._monotonic_fn() - start
            logger.debug("CrossAsset refresh cancelado por stop (mantido anterior)")
            return False
        if not isinstance(result, dict) or result.get("status") != "ok":
            # Parcial/falha: NÃO publica mistura; anterior envelhece sozinho.
            self.partial_total += 1
            self.last_duration_s = self._monotonic_fn() - start
            logger.debug("CrossAsset refresh parcial (mantido anterior)")
            return False
        snap = CrossAssetSnapshot(
            result=dict(result),
            fetched_at_ms=int(time.time() * 1000),
            fetched_monotonic=self._monotonic_fn(),
            last_error=None,
        )
        with self._lock:
            self._snapshot = snap
        self.updates_total += 1
        self.last_duration_s = self._monotonic_fn() - start
        self.last_error = None
        return True

    def _store_snapshot_for_test(self, result: Dict[str, Any],
                                 age_s: float = 0.0) -> None:
        """Seam de teste: publica snapshot completo com idade controlada."""
        now_mono = self._monotonic_fn()
        with self._lock:
            self._snapshot = CrossAssetSnapshot(
                result=dict(result),
                fetched_at_ms=int(time.time() * 1000) - int(age_s * 1000),
                fetched_monotonic=now_mono - age_s,
                last_error=None,
            )

    # -- leitura (hot path: sem I/O, sem threads) ------------------------
    def read_view(self) -> Dict[str, Any]:
        """Visão all-or-nothing para a janela. Nunca faz rede."""
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return {
                "values": {},
                "status": "warming_up",
                "age_seconds": None,
                "fetched_at_ms": None,
                "temporal_audit": CROSS_ASSET_TEMPORAL_AUDIT,
            }
        age = self._monotonic_fn() - snap.fetched_monotonic
        return {
            "values": dict(snap.result),
            "status": _classify(age, self.policy.fresh_s,
                                self.policy.usable_s),
            "age_seconds": age,
            "fetched_at_ms": snap.fetched_at_ms,
            "temporal_audit": CROSS_ASSET_TEMPORAL_AUDIT,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Observabilidade interna (last_error NÃO vai ao LLM)."""
        with self._lock:
            snap = self._snapshot
        return {
            "updates_total": self.updates_total,
            "partial_total": self.partial_total,
            "failures_total": self.failures_total,
            "last_duration_s": self.last_duration_s,
            "last_error": self.last_error,
            "has_snapshot": snap is not None,
        }
