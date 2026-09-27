# flow_analyzer/effort_response_transport.py
"""
P1-F ETAPA 2 — Shadow Async Transport v1.

Transporte assíncrono e resolução causal de outcomes para o Shadow Dataset v1.

PRINCÍPIOS ARQUITETURAIS:
1. Zero I/O, zero locks pesados e zero dependência de IA/LLM no hot path.
2. Contrato temporal canônico:
   - causal_anchor_ms = window_end_ms lógico/exclusivo.
   - Feature window interval: trades T < causal_anchor_ms.
   - Future excursion interval: causal_anchor_ms < T <= target_timestamp_ms.
   - Trade com T == causal_anchor_ms é deliberadamente excluído de ambos:
     BOUNDARY_EXCLUDED_FOR_CAUSAL_SAFETY.
3. Concorrência: N producers -> bounded queue -> 1 single dedicated writer thread.
4. Backpressure: DROP_NEWEST não-bloqueante (put_nowait).
5. Pending Registry baseado em min-heap (heapq) O(log N) por target_timestamp_ms.
6. Separação formal de terminal future_price (FIRST_ON_OR_AFTER dentro de 1000ms)
   versus excursões de preço (apenas dentro do intervalo causal).
7. Eviction do buffer de observações por event time (15m + tolerância).
8. Failure isolation estrito: StorageCorruptionError desabilita apenas o shadow,
   preservando o trading e sem abrir circuit breaker.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from flow_analyzer.effort_response_dataset import (
    FEATURE_CONTRACT_VERSION,
    OUTCOME_BOUNDARY_TOLERANCE_MS,
    SHADOW_SCHEMA_VERSION,
    EffortResponseShadowRecord,
    EffortResponseShadowStorage,
    HorizonOutcome,
    StorageCorruptionError,
    StorageOrphanUpdateError,
    build_deterministic_record_id,
    build_shadow_record,
)

logger = logging.getLogger("ShadowAsyncTransport")

# Capacidade operacional conservadora padrão documentada
# 1.000 janelas de 1m representam ~16.6h de buffer com footprint ~1.5MB de RAM.
DEFAULT_OPERATIONAL_UNCALIBRATED_CAPACITY: int = 1000

# Retenção máxima do observation buffer em ms (15 minutos + 1000ms tolerância + margem de 50s)
OBSERVATION_BUFFER_MAX_AGE_MS: int = 15 * 60_000 + OUTCOME_BOUNDARY_TOLERANCE_MS + 50_000


# ─────────────────────────────────────────────────────────────────────────────
# 1. DTO MÍNIMO E FROZEN
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EffortResponseSnapshotDTO:
    """Snapshot imutável e minimalista de primitives extraídos no fechamento da janela.

    Não retém referências ao DataPipeline, DataFrames, WindowProcessor ou raw_event.
    Mutação posterior de qualquer objeto fonte não afeta este DTO.
    """
    symbol: str
    causal_anchor_ms: int
    observation_open_ms: int
    observation_close_ms: int
    buy_notional_usd: float
    sell_notional_usd: float
    open: float
    high: float
    low: float
    close: float
    window_duration_ms: int
    vwap: Optional[float] = None
    poc: Optional[float] = None
    context_data: Optional[Dict[str, Any]] = None
    source_event_id: Optional[str] = None
    orderbook_source_type: Optional[str] = None
    orderbook_snapshot_ms: Optional[int] = None
    flow_window_validity: Optional[str] = None
    latency_ms: Optional[float] = None
    freshness_ms: Optional[float] = None

    def __post_init__(self) -> None:
        # Cópia defensiva imutável do context_data para isolar de mutações externas
        if self.context_data is not None:
            ctx_copy = dict(self.context_data)
            object.__setattr__(self, "context_data", ctx_copy)

        # Validação causal fundamental:
        # Todos os trades da janela fechada devem ter ocorrido antes do causal_anchor_ms
        if self.observation_close_ms >= self.causal_anchor_ms:
            raise ValueError(
                f"Violação causal: observation_close_ms ({self.observation_close_ms}) "
                f"não pode ser >= causal_anchor_ms ({self.causal_anchor_ms}). "
                f"BOUNDARY_EXCLUDED_FOR_CAUSAL_SAFETY exige T < causal_anchor_ms."
            )
        if self.observation_open_ms > self.observation_close_ms:
            raise ValueError(
                f"Invariante temporal inválido: observation_open_ms ({self.observation_open_ms}) "
                f"> observation_close_ms ({self.observation_close_ms})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTRUTURAS DO PENDING REGISTRY E OBSERVAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class PendingHorizonEntry:
    """Item indexado no min-heap para resolução de outcomes.

    Ordenado primariamente por target_timestamp_ms, com desempate por seq monotônico.
    """
    target_timestamp_ms: int
    seq: int
    record_id: str = field(compare=False)
    horizon: str = field(compare=False)  # "1m", "5m", "15m"
    symbol: str = field(compare=False)
    close_price_at_t: float = field(compare=False)
    causal_anchor_ms: int = field(compare=False)


@dataclass(frozen=True)
class PriceObservation:
    """Observação de preço para cálculo de excursão e preço terminal."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    is_window: bool = False
    window_open_ms: Optional[int] = None
    window_close_ms: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# 3. GESTÃO DE MÉTRICAS PROMETHEUS DO SUBSISTEMA SHADOW
# ─────────────────────────────────────────────────────────────────────────────

class ShadowMetrics:
    """Contêiner de métricas Prometheus institucional para o subsistema Shadow.

    Evita duplicações no CollectorRegistry default durante testes unitários.
    """
    _instance: Optional["ShadowMetrics"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from monitoring.metrics_collector import create_counter, create_gauge

        self.queue_size = create_gauge(
            "shadow_queue_size",
            "Profundidade atual da fila do shadow writer",
        )
        self.queue_capacity = create_gauge(
            "shadow_queue_capacity",
            "Capacidade máxima da fila do shadow writer",
        )
        self.queue_high_watermark = create_gauge(
            "shadow_queue_high_watermark",
            "Pico máximo atingido de ocupação da fila do shadow writer",
        )
        self.records_enqueued_total = create_counter(
            "shadow_records_enqueued_total",
            "Total de registros shadow enfileirados no hot path",
        )
        self.records_written_total = create_counter(
            "shadow_records_written_total",
            "Total de registros shadow gravados no storage JSONL",
        )
        self.records_dropped_total = create_counter(
            "shadow_records_dropped_total",
            "Total de registros shadow descartados por fila cheia (DROP_NEWEST)",
        )
        self.write_errors_total = create_counter(
            "shadow_write_errors_total",
            "Total de erros de escrita ou corrupção no shadow storage",
            labelnames=["error_type"],
        )
        self.outcomes_resolved_total = create_counter(
            "shadow_outcomes_resolved_total",
            "Total de horizontes de outcome resolvidos com sucesso",
            labelnames=["horizon"],
        )
        self.outcomes_insufficient_total = create_counter(
            "shadow_outcomes_insufficient_total",
            "Total de horizontes de outcome não resolvidos por falta de dados/gap",
            labelnames=["horizon"],
        )
        self.writer_lag_ms = create_gauge(
            "shadow_writer_lag_ms",
            "Latência entre o causal anchor e o momento de escrita do registro",
        )

    @classmethod
    def get_instance(cls) -> "ShadowMetrics":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUBSISTEMA SHADOW ASYNC TRANSPORT
# ─────────────────────────────────────────────────────────────────────────────

class ShadowAsyncTransport:
    """Transporte assíncrono para o dataset de Effort/Response.

    Arquitetura:
    - N producers chamam submit_nowait() com EffortResponseSnapshotDTO (não-bloqueante).
    - Fila bounded thread-safe queue.Queue com política DROP_NEWEST.
    - Exatamente 1 worker thread consumidora executando operações no EffortResponseShadowStorage.
    - Resolução de outcomes desacoplada de timers, acionada por eventos/ticks e controlada via min-heap.
    """
    _singleton: Optional["ShadowAsyncTransport"] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        queue_capacity: Optional[int] = None,
        enabled: Optional[bool] = None,
        startup_event_time_ms: Optional[int] = None,
        start_worker: bool = True,
    ) -> None:
        import queue as _queue

        # 1. Determinação da flag de ativação
        if enabled is None:
            env_val = os.getenv("EFFORT_RESPONSE_SHADOW_ENABLED", "0").strip().lower()
            self._enabled = env_val in ("1", "true", "yes")
        else:
            self._enabled = bool(enabled)

        if filepath is None or str(filepath) == "dados/datasets/shadow_effort_response.jsonl":
            filepath = os.getenv("EFFORT_RESPONSE_SHADOW_FILEPATH", "dados/datasets/shadow_effort_response.jsonl")
        self.filepath = Path(filepath)
        self._disabled_due_to_corruption = False

        if not self._enabled:
            # Estado inativo: zero threads, zero filas, zero I/O
            self._queue: Any = None
            self._storage: Any = None
            self._worker_thread: Optional[threading.Thread] = None
            logger.info("Coleta shadow EFFORT_RESPONSE desativada (flag=0).")
            return

        # 2. Configuração de capacidade
        if queue_capacity is not None and queue_capacity > 0:
            self._capacity = int(queue_capacity)
        else:
            env_cap = os.getenv("EFFORT_RESPONSE_SHADOW_QUEUE_CAPACITY", "")
            try:
                self._capacity = int(env_cap) if env_cap else DEFAULT_OPERATIONAL_UNCALIBRATED_CAPACITY
            except ValueError:
                self._capacity = DEFAULT_OPERATIONAL_UNCALIBRATED_CAPACITY

        self._queue = _queue.Queue(maxsize=self._capacity)
        self._metrics = ShadowMetrics.get_instance()
        self._metrics.queue_capacity.set(float(self._capacity))

        # 3. Componentes internos do worker
        self._storage: Optional[EffortResponseShadowStorage] = None
        self._storage_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._drain_finished_event = threading.Event()
        self._worker_started = False

        # Estatísticas e high watermark
        self._high_watermark: int = 0
        self._dropped_count: int = 0
        self._enqueued_count: int = 0
        self._written_count: int = 0
        self._seq_counter: int = 0

        # Pending Registry (Min-Heap + Idempotency Map)
        self._pending_heap: List[PendingHorizonEntry] = []
        self._pending_map: Dict[Tuple[str, str], PendingHorizonEntry] = {}

        # Observation buffer para cálculo de excursões e terminal prices
        self._observations: deque[PriceObservation] = deque()
        self._last_event_time_ms: int = 0

        # Política de recuperação no startup (sem wall clock)
        self._recovery_pending_records: List[Any] = []
        self._recovery_pending_state: str = "RESOLVED"

        # Inicializa storage e worker
        self._init_storage()
        self._init_pending_from_storage(startup_event_time_ms)
        if start_worker:
            self.start()

    def _init_storage(self) -> None:
        """Inicializa storage fail-closed."""
        try:
            self._storage = EffortResponseShadowStorage(self.filepath)
        except StorageCorruptionError as e:
            self._handle_corruption(e)

    def _handle_corruption(self, exc: Exception) -> None:
        """Trata corrupção de storage desativando apenas o subsistema shadow."""
        self._disabled_due_to_corruption = True
        self._enabled = False
        try:
            self._metrics.write_errors_total.labels(error_type="corruption").inc()
        except Exception:
            pass
        logger.critical(
            "CRITICAL: Storage de shadow esforço/resposta corrompido em '%s'! "
            "Subsistema shadow desativado de forma fail-closed. "
            "O bot de trading continuará operando normalmente. Erro: %s",
            self.filepath,
            exc,
            exc_info=True,
        )

    def _init_pending_from_storage(self, startup_event_time_ms: Optional[int] = None) -> None:
        """Carrega registros com horizontes PENDING na inicialização com política temporal estrita."""
        if not self._storage or self._disabled_due_to_corruption:
            return

        try:
            records = self._storage.read_records()
        except StorageCorruptionError as e:
            self._handle_corruption(e)
            return

        self._recovery_pending_records = []
        for rec in records:
            outcomes = rec.outcomes_future or {}
            if outcomes.get("status") == "RESOLVED":
                continue
            horizons = outcomes.get("horizons") or {}
            has_pending = any(h_data.get("status") == "PENDING" for h_data in horizons.values())
            if has_pending:
                self._recovery_pending_records.append(rec)

        if startup_event_time_ms is not None:
            self.resolve_startup_recovery(startup_event_time_ms)
        else:
            if self._recovery_pending_records:
                self._recovery_pending_state = "RECOVERY_PENDING"
                logger.info(
                    "Shadow startup: %d registros carregados como RECOVERY_PENDING. "
                    "Aguardando primeiro watermark/event-time real da exchange.",
                    len(self._recovery_pending_records),
                )
            else:
                self._recovery_pending_state = "RESOLVED"

    def resolve_startup_recovery(self, event_time_ms: int) -> None:
        """Resolve registros em RECOVERY_PENDING a partir do primeiro event-time real da exchange.

        Contrato:
        - NUNCA usa wall clock local como event-time.
        - Se target < (event_time_ms - OUTCOME_BOUNDARY_TOLERANCE_MS): marca INSUFFICIENT_DATA (offline gap).
        - Se target >= (event_time_ms - OUTCOME_BOUNDARY_TOLERANCE_MS): reinserir pending heap.
        - Se já RESOLVED: não reinserir.
        """
        if not self._recovery_pending_records:
            self._recovery_pending_state = "RESOLVED"
            return

        recovered_count = 0
        gap_count = 0

        for rec in self._recovery_pending_records:
            outcomes = rec.outcomes_future or {}
            horizons = outcomes.get("horizons") or {}
            updates_needed: Dict[str, Any] = {"horizons": {}}
            has_offline_gap = False

            for h_name, h_data in horizons.items():
                if h_data.get("status") == "PENDING":
                    target_ms = int(h_data.get("target_timestamp_ms", 0))
                    # Se o alvo venceu durante o downtime:
                    if target_ms < (event_time_ms - OUTCOME_BOUNDARY_TOLERANCE_MS):
                        has_offline_gap = True
                        gap_count += 1
                        updates_needed["horizons"][h_name] = HorizonOutcome(
                            horizon=h_name,
                            status="INSUFFICIENT_DATA",
                            target_timestamp_ms=target_ms,
                            timing_error_ms=None,
                            future_price=None,
                            return_bps=None,
                            horizon_duration_ms=h_data.get("horizon_duration_ms"),
                            excursion_status="INSUFFICIENT_DATA",
                            resolved_at_ms=event_time_ms,
                        ).to_dict()
                        try:
                            self._metrics.outcomes_insufficient_total.labels(horizon=h_name).inc()
                        except Exception:
                            pass
                    else:
                        # Alvo ainda futuro em relação ao event_time_ms seguro:
                        recovered_count += 1
                        self._register_pending_entry(
                            target_timestamp_ms=target_ms,
                            record_id=rec.record_id,
                            horizon=h_name,
                            symbol=rec.provenance.symbol,
                            close_price_at_t=float(rec.features_at_t.get("price.close", 0.0)),
                            causal_anchor_ms=int(rec.provenance.causal_anchor_ms or rec.provenance.window_close_ms),
                        )

            if has_offline_gap:
                all_res = all(
                    updates_needed["horizons"].get(h, {}).get("status") in ("RESOLVED", "INSUFFICIENT_DATA")
                    for h in ("1m", "5m", "15m")
                )
                updates_needed["status"] = "PARTIALLY_RESOLVED" if not all_res else "RESOLVED"
                try:
                    with self._storage_lock:
                        self._storage.update_record_outcomes(rec.record_id, updates_needed)
                except Exception as e_up:
                    logger.warning("Falha ao registrar gap offline para %s: %s", rec.record_id, e_up)

        self._recovery_pending_records.clear()
        self._recovery_pending_state = "RESOLVED"
        logger.info(
            "Shadow startup recovery concluída para watermark %d: %d horizontes reinseridos, %d marcados como INSUFFICIENT_DATA.",
            event_time_ms,
            recovered_count,
            gap_count,
        )

    def _register_pending_entry(
        self,
        target_timestamp_ms: int,
        record_id: str,
        horizon: str,
        symbol: str,
        close_price_at_t: float,
        causal_anchor_ms: int,
    ) -> None:
        """Registra entrada no min-heap e mapa de idempotência."""
        self._seq_counter += 1
        entry = PendingHorizonEntry(
            target_timestamp_ms=target_timestamp_ms,
            seq=self._seq_counter,
            record_id=record_id,
            horizon=horizon,
            symbol=symbol,
            close_price_at_t=close_price_at_t,
            causal_anchor_ms=causal_anchor_ms,
        )
        self._pending_map[(record_id, horizon)] = entry
        heapq.heappush(self._pending_heap, entry)

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE E WORKER
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia a single writer thread do transporte."""
        if not self._enabled or self._disabled_due_to_corruption or self._worker_started:
            return

        self._stop_event.clear()
        self._drain_finished_event.clear()
        t = threading.Thread(target=self._worker_loop, name="shadow-writer", daemon=True)
        t.start()
        self._worker_thread = t
        self._worker_started = True
        logger.info("ShadowAsyncTransport iniciado com capacidade %d.", self._capacity)

    def submit_nowait(self, dto: EffortResponseSnapshotDTO) -> bool:
        """Submissão NÃO-BLOQUEANTE a partir do hot path.

        Nunca abre arquivo, nunca espera lock de disco, nunca propaga exceção.
        """
        if not self._enabled or self._disabled_due_to_corruption or self._stop_event.is_set():
            return False

        try:
            self._queue.put_nowait(dto)
            self._enqueued_count += 1
            curr_size = self._queue.qsize()

            if curr_size > self._high_watermark:
                self._high_watermark = curr_size
                self._metrics.queue_high_watermark.set(float(curr_size))

            self._metrics.queue_size.set(float(curr_size))
            self._metrics.records_enqueued_total.inc()
            return True

        except Exception as e_full:
            import queue as _q

            is_full = isinstance(e_full, _q.Full)
            self._dropped_count += 1
            self._metrics.records_dropped_total.inc()

            if not is_full:
                self._metrics.write_errors_total.labels(error_type="enqueue_error").inc()
                logger.warning("Falha inesperada no submit_nowait shadow: %s", e_full)
            else:
                logger.warning(
                    "Fila shadow cheia (capacidade=%d). DROP_NEWEST aplicado para janela %s (drop_total=%d)",
                    self._capacity,
                    getattr(dto, "symbol", "UNKNOWN"),
                    self._dropped_count,
                )
            return False

    def on_price_observation(
        self,
        timestamp_ms: int,
        open: float,
        high: float,
        low: float,
        close: float,
        is_window: bool = True,
        window_open_ms: Optional[int] = None,
        window_close_ms: Optional[int] = None,
    ) -> None:
        """Envia observação de preço para o buffer do worker resolver outcomes.

        Não bloqueia o chamador.
        """
        if not self._enabled or self._disabled_due_to_corruption or not self._queue or self._stop_event.is_set():
            return

        obs = PriceObservation(
            timestamp_ms=int(timestamp_ms),
            open=float(open),
            high=float(high),
            low=float(low),
            close=float(close),
            is_window=is_window,
            window_open_ms=window_open_ms,
            window_close_ms=window_close_ms,
        )
        try:
            self._queue.put_nowait(obs)
        except Exception:
            # Observações de preço para outcome podem ser descartadas sob congestionamento
            # sem interromper trading
            pass

    def _worker_loop(self) -> None:
        """Single writer worker loop."""
        import queue as _q

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.25)
            except _q.Empty:
                continue

            if item is None:
                # Sentinela de shutdown
                self._queue.task_done()
                break

            try:
                if isinstance(item, EffortResponseSnapshotDTO):
                    self._process_snapshot_dto(item)
                elif isinstance(item, PriceObservation):
                    self._process_observation(item)
            except StorageCorruptionError as e_corr:
                self._handle_corruption(e_corr)
                self._queue.task_done()
                break
            except Exception as e_proc:
                self._metrics.write_errors_total.labels(error_type="worker_error").inc()
                logger.warning("Erro no processamento do item shadow: %s", e_proc, exc_info=True)
            finally:
                try:
                    self._queue.task_done()
                    self._metrics.queue_size.set(float(self._queue.qsize()))
                except Exception:
                    pass

        self._drain_finished_event.set()

    def _process_snapshot_dto(self, dto: EffortResponseSnapshotDTO) -> None:
        """Constrói e persiste CREATE do EffortResponseShadowRecord."""
        if not self._storage:
            return

        window_data = {
            "buy_notional_usd": dto.buy_notional_usd,
            "sell_notional_usd": dto.sell_notional_usd,
            "open": dto.open,
            "high": dto.high,
            "low": dto.low,
            "close": dto.close,
            "window_duration_ms": dto.window_duration_ms,
            "vwap": dto.vwap,
            "poc": dto.poc,
        }

        record = build_shadow_record(
            symbol=dto.symbol,
            window_open_ms=dto.observation_open_ms,
            window_close_ms=dto.observation_close_ms,
            window_data=window_data,
            causal_anchor_ms=dto.causal_anchor_ms,
            observation_open_ms=dto.observation_open_ms,
            observation_close_ms=dto.observation_close_ms,
            context_data=dto.context_data,
            source_event_id=dto.source_event_id,
            orderbook_source_type=dto.orderbook_source_type,
            orderbook_snapshot_ms=dto.orderbook_snapshot_ms,
            flow_window_validity=dto.flow_window_validity,
            latency_ms=dto.latency_ms,
            freshness_ms=dto.freshness_ms,
        )

        with self._storage_lock:
            saved = self._storage.append_record(record)
            if saved:
                self._written_count += 1
                self._metrics.records_written_total.inc()
                now_wall_ms = int(time.time() * 1000)
                lag = max(0, now_wall_ms - dto.causal_anchor_ms)
                self._metrics.writer_lag_ms.set(float(lag))

        # Cadastra horizontes pendentes no min-heap indexados por causal_anchor_ms
        for h, duration_ms in (("1m", 60_000), ("5m", 300_000), ("15m", 900_000)):
            target_ms = dto.causal_anchor_ms + duration_ms
            self._register_pending_entry(
                target_timestamp_ms=target_ms,
                record_id=record.record_id,
                horizon=h,
                symbol=dto.symbol,
                close_price_at_t=dto.close,
                causal_anchor_ms=dto.causal_anchor_ms,
            )

    def _process_observation(self, obs: PriceObservation) -> None:
        """Adiciona observação ao buffer com eviction por event time e resolve min-heap."""
        if self._recovery_pending_state == "RECOVERY_PENDING" and obs.timestamp_ms > 0:
            self.resolve_startup_recovery(obs.timestamp_ms)

        self._last_event_time_ms = max(self._last_event_time_ms, obs.timestamp_ms)
        self._observations.append(obs)

        # Eviction estritamente temporal (horizonte máximo 15m + tolerância)
        cutoff_ms = self._last_event_time_ms - OBSERVATION_BUFFER_MAX_AGE_MS
        while self._observations and self._observations[0].timestamp_ms < cutoff_ms:
            self._observations.popleft()

        # Resolução do min-heap O(log N)
        self._resolve_pending_horizons(obs.timestamp_ms)

    def _resolve_pending_horizons(self, current_event_time_ms: int) -> None:
        """Verifica e resolve horizontes cujo target foi alcançado."""
        if not self._storage or not self._pending_heap:
            return

        # Avalia apenas enquanto o topo do heap já alcançou ou passou o tempo atual
        # Tolerância: target_timestamp_ms <= current_event_time_ms
        while self._pending_heap:
            top = self._pending_heap[0]
            if top.target_timestamp_ms > current_event_time_ms:
                # O menor elemento ainda está no futuro; encerra sem full-scan
                break

            entry = heapq.heappop(self._pending_heap)
            mapKey = (entry.record_id, entry.horizon)

            # Idempotência: se já foi removido do mapa, descarta entrada fantasma
            if mapKey not in self._pending_map:
                continue

            del self._pending_map[mapKey]

            # Resolução temporal precisa
            self._resolve_single_entry(entry, current_event_time_ms)

    def _resolve_single_entry(self, entry: PendingHorizonEntry, current_event_time_ms: int) -> None:
        """Calcula terminal future_price e excursões para uma entrada específica."""
        target_ms = entry.target_timestamp_ms
        drift_ms = current_event_time_ms - target_ms

        # 1. Contrato de Terminal Price: FIRST_ON_OR_AFTER dentro de [target, target + 1000ms]
        future_price: Optional[float] = None
        obs_price_ts: Optional[int] = None
        timing_error: Optional[int] = None

        candidate_terminal = [
            obs for obs in self._observations
            if target_ms <= obs.timestamp_ms <= (target_ms + OUTCOME_BOUNDARY_TOLERANCE_MS)
        ]

        if candidate_terminal:
            cand = candidate_terminal[0]
            obs_price_ts = cand.timestamp_ms
            future_price = cand.close
            timing_error = obs_price_ts - target_ms

        # 2. Contrato de Excursions:
        # Apenas observações estritamente dentro de (causal_anchor_ms, target_timestamp_ms]
        excursion_highs: List[float] = []
        excursion_lows: List[float] = []
        valid_obs_count = 0
        cov_start: Optional[int] = None
        cov_end: Optional[int] = None
        has_partial_coverage = False

        for obs in self._observations:
            # Se for candle ou observação agregada em janela:
            if obs.is_window and obs.window_open_ms is not None and obs.window_close_ms is not None:
                # Janela inteira contida estritamente no intervalo causal
                if obs.window_open_ms > entry.causal_anchor_ms and obs.window_close_ms <= target_ms:
                    excursion_highs.append(obs.high)
                    excursion_lows.append(obs.low)
                    valid_obs_count += 1
                    cov_start = obs.window_open_ms if cov_start is None else min(cov_start, obs.window_open_ms)
                    cov_end = obs.window_close_ms if cov_end is None else max(cov_end, obs.window_close_ms)
                # Janela intersecta parcialmente a fronteira
                elif (obs.window_open_ms <= entry.causal_anchor_ms < obs.window_close_ms) or \
                     (obs.window_open_ms <= target_ms < obs.window_close_ms):
                    has_partial_coverage = True
                    # NÃO incluir high/low de janela parcialmente sobreposta para evitar contaminação
            else:
                # Trade pontual: estritamente > anchor e <= target
                if entry.causal_anchor_ms < obs.timestamp_ms <= target_ms:
                    excursion_highs.append(obs.high)
                    excursion_lows.append(obs.low)
                    valid_obs_count += 1
                    cov_start = obs.timestamp_ms if cov_start is None else min(cov_start, obs.timestamp_ms)
                    cov_end = obs.timestamp_ms if cov_end is None else max(cov_end, obs.timestamp_ms)

        # 3. Determinação de Status do Horizonte
        if future_price is None or not math.isfinite(future_price):
            # Fora da tolerância ou sem observação terminal elegível
            h_outcome = HorizonOutcome(
                horizon=entry.horizon,
                status="INSUFFICIENT_DATA",
                target_timestamp_ms=target_ms,
                horizon_duration_ms=_get_horizon_duration_ms(entry.horizon),
                excursion_status="INSUFFICIENT_DATA" if valid_obs_count == 0 else "PARTIAL",
                excursion_coverage_start_ms=cov_start,
                excursion_coverage_end_ms=cov_end,
                excursion_observation_count=valid_obs_count,
            )
            self._metrics.outcomes_insufficient_total.labels(horizon=entry.horizon).inc()
        else:
            base_p = entry.close_price_at_t
            ret_bps = (future_price - base_p) / base_p * 10000.0 if base_p > 0 else 0.0

            max_high = max(excursion_highs) if excursion_highs else future_price
            min_low = min(excursion_lows) if excursion_lows else future_price
            max_up = (max_high - base_p) / base_p * 10000.0 if base_p > 0 else 0.0
            max_down = (min_low - base_p) / base_p * 10000.0 if base_p > 0 else 0.0

            exc_status = "PARTIAL" if has_partial_coverage or valid_obs_count == 0 else "FULL"

            h_outcome = HorizonOutcome(
                horizon=entry.horizon,
                status="RESOLVED",
                target_timestamp_ms=target_ms,
                observed_price_timestamp_ms=obs_price_ts,
                timing_error_ms=timing_error,
                future_price=future_price,
                return_bps=ret_bps,
                max_excursion_up_bps=max_up,
                max_excursion_down_bps=max_down,
                mfe_bps=max_up,
                mae_bps=max_down,
                max_high=max_high,
                min_low=min_low,
                observation_count=valid_obs_count,
                horizon_duration_ms=_get_horizon_duration_ms(entry.horizon),
                resolved_at_ms=obs_price_ts,
                excursion_status=exc_status,
                excursion_coverage_start_ms=cov_start,
                excursion_coverage_end_ms=cov_end,
                excursion_observation_count=valid_obs_count,
            )
            self._metrics.outcomes_resolved_total.labels(horizon=entry.horizon).inc()

        # 4. Grava atualização no storage
        update_data = {
            "horizons": {
                entry.horizon: h_outcome.to_dict()
            }
        }
        with self._storage_lock:
            try:
                self._storage.update_record_outcomes(entry.record_id, update_data, strict=False)
            except Exception as e_up:
                logger.warning("Falha ao persistir outcome para %s (%s): %s", entry.record_id, entry.horizon, e_up)

    # ─────────────────────────────────────────────────────────────────────────
    # FLUSH, SHUTDOWN E STATS
    # ─────────────────────────────────────────────────────────────────────────

    def flush(self, timeout: float = 5.0) -> bool:
        """Aguarda esvaziar itens pendentes na fila."""
        if not self._enabled or not self._queue or self._disabled_due_to_corruption:
            return True

        start = time.time()
        while self._queue.unfinished_tasks > 0:
            if time.time() - start > timeout:
                logger.warning("Timeout no flush do shadow transport (pendentes=%d)", self._queue.unfinished_tasks)
                return False
            time.sleep(0.02)
        return True

    def close(self, timeout: float = 5.0) -> bool:
        """Drena a fila e encerra a worker thread com segurança."""
        if not self._enabled or self._disabled_due_to_corruption:
            return True

        self._stop_event.set()
        if self._worker_started and self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass

            self.flush(timeout=timeout)

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)

        self._worker_started = False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Retorna telemetria operacional."""
        q_depth = self._queue.qsize() if self._queue else 0
        return {
            "enabled": self._enabled,
            "disabled_due_to_corruption": self._disabled_due_to_corruption,
            "queue_capacity": self._capacity if self._enabled else 0,
            "queue_depth": q_depth,
            "high_watermark": self._high_watermark,
            "records_enqueued": self._enqueued_count,
            "records_written": self._written_count,
            "records_dropped": self._dropped_count,
            "pending_heap_size": len(self._pending_heap),
            "pending_registry_size": len(self._pending_map),
            "observation_buffer_size": len(self._observations),
            "last_event_time_ms": self._last_event_time_ms,
            "recovery_pending_state": self._recovery_pending_state,
            "recovery_pending_count": len(self._recovery_pending_records),
        }

    @classmethod
    def get_if_initialized(cls) -> Optional["ShadowAsyncTransport"]:
        """Retorna o singleton apenas se já estiver instanciado, sem criar nova instância."""
        return cls._singleton

    @classmethod
    def get_instance(cls, **kwargs: Any) -> "ShadowAsyncTransport":
        """Obtém ou cria singleton da instância de transporte."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls(**kwargs)
        return cls._singleton

    @classmethod
    def reset_instance_for_testing(cls) -> None:
        """Método de utilidade para isolamento de testes unitários."""
        with cls._singleton_lock:
            if cls._singleton is not None:
                cls._singleton.close(timeout=1.0)
                cls._singleton = None


def _get_horizon_duration_ms(horizon: str) -> int:
    if horizon == "1m":
        return 60_000
    if horizon == "5m":
        return 300_000
    if horizon == "15m":
        return 900_000
    return 60_000
