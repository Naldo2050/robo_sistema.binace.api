# trade_buffer.py
"""
Buffer assíncrono de trades com backpressure e métricas de latência.

Resolve o problema de trades atrasados implementando:
- Buffer com tamanho limitado e backpressure
- Processamento assíncrono em background
- Métricas de latência end-to-end
- Alertas de buffer crítico
- Shutdown delimitado (bounded drain) e observabilidade completa de drops
"""

import asyncio
import inspect
import logging
import math
import time
from collections import deque
from threading import Lock, Event
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Importar métricas do sistema
try:
    from monitoring.metrics_collector import (
        record_trade_late,
        update_active_trades_buffer as set_trades_in_buffer,
        track_operation as track_latency,  # type: ignore[assignment]
        record_timeout,
        record_trades_dropped_overflow,
        record_trades_dropped_shutdown,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    def record_trade_late(latency_ms: float): pass
    def set_trades_in_buffer(count): pass
    def track_latency(operation_type: Any = None):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    def record_timeout(): pass
    def record_trades_dropped_overflow(count: int = 1): pass
    def record_trades_dropped_shutdown(count: int = 1): pass

# Importar validador de trades
try:
    from trading.trade_validator import validate_and_filter_trades
    TRADE_VALIDATOR_AVAILABLE = True
except ImportError:
    TRADE_VALIDATOR_AVAILABLE = False
    def validate_and_filter_trades(trades, max_age_seconds=30):
        return trades

# Importar monitor de latência (NOVO - não descarta trades)
try:
    from trading.trade_validator import latency_monitor
    LATENCY_MONITOR_AVAILABLE = True
except ImportError:
    LATENCY_MONITOR_AVAILABLE = False
    latency_monitor = None

# Importar filtro de trades atrasados (não existe mais - usando fallback sempre)
TRADE_FILTER_AVAILABLE = False
class DummyFilter:
    def is_trade_valid(self, trade): return True
trade_filter = DummyFilter()

# Importar validador de timestamp (não existe mais - usando fallback sempre)
TRADE_TIMESTAMP_VALIDATOR_AVAILABLE = False
trade_timestamp_validator = None


class BufferStatus(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERFLOW = "overflow"


@dataclass
class TradeLatencyMetrics:
    """Métricas de latência de trades."""
    buffer_size: int
    buffer_capacity: int
    buffer_fill_ratio: float
    avg_processing_time_ms: float
    p95_processing_time_ms: float
    max_processing_time_ms: float
    trades_per_second: float
    status: BufferStatus
    last_trade_latency_ms: float


class AsyncTradeBuffer:
    """
    Buffer assíncrono de trades com backpressure e bounded shutdown.

    Características:
    - Buffer com tamanho máximo configurável
    - Backpressure quando > 80% de capacidade (DROP_OLDEST 10%)
    - Processamento assíncrono em background
    - Shutdown bounded com timeout explícito (drenagem observável)
    - Contabilização canônica e sem sobreposição de todos os trades perdidos
    - Métricas de latência e integração com Prometheus e Forensic Context
    """

    def __init__(
        self,
        max_size: int = 5000,
        backpressure_threshold: float = 0.8,
        processing_batch_size: int = 50,
        processing_interval_ms: int = 10,
        max_processing_time_ms: float = 5.0,
        drain_timeout_ms: float = 2000.0,
        warning_callback: Optional[Callable[[BufferStatus, int], None]] = None,
        heartbeat_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Inicializa o buffer assíncrono com validação estrita de configurações.
        """
        # Validação de drain_timeout_ms
        if isinstance(drain_timeout_ms, bool):
            raise TypeError("drain_timeout_ms cannot be a boolean")
        if not isinstance(drain_timeout_ms, (int, float)):
            raise TypeError(f"drain_timeout_ms must be a number, got {type(drain_timeout_ms).__name__}")
        if math.isnan(drain_timeout_ms) or math.isinf(drain_timeout_ms):
            raise ValueError("drain_timeout_ms cannot be NaN or Inf")
        if drain_timeout_ms < 0.0:
            raise ValueError(f"drain_timeout_ms cannot be negative, got {drain_timeout_ms}")

        self.max_size = max_size
        self.backpressure_threshold = backpressure_threshold
        self.processing_batch_size = processing_batch_size
        self.processing_interval_ms = processing_interval_ms
        self.max_processing_time_ms = max_processing_time_ms
        self.drain_timeout_ms = float(drain_timeout_ms)
        self.warning_callback = warning_callback
        self.heartbeat_callback = heartbeat_callback

        # Buffer principal
        self._buffer: deque = deque(maxlen=max_size)
        self._buffer_lock = Lock()

        # Controle de lifecycle e processamento
        self._processing_lock = Lock()
        self._should_stop = Event()
        self._processing_task: Optional[asyncio.Task] = None
        self._accepting_trades: bool = True
        self._is_stopped: bool = False
        self._is_stopping: bool = False

        # Rastreamento de in-flight
        self._in_flight_count: int = 0
        self._current_in_flight_trades: list = []

        # Métricas de latência
        self._processing_times = deque(maxlen=1000)
        self._trade_count = 0
        self._last_trade_time = 0.0
        self._last_metrics_update = 0.0

        # Status atual
        self._current_status = BufferStatus.NORMAL

        # Estatísticas canônicas de perda, conservação de massa e processamento
        self._total_accepted_trades: int = 0
        self._overflow_events: int = 0
        self._dropped_overflow_trades: int = 0
        self._dropped_shutdown_trades: int = 0
        self._processor_error_trades: int = 0
        self._total_trades_processed: int = 0
        self._total_processing_time_ms: float = 0.0

        # Estatísticas legadas (compatibilidade)
        self._overflow_count: int = 0
        self._backpressure_count: int = 0
        self._dropped_trades_count: int = 0

        # Inicializa tempo de início para cálculo de TPS
        self._start_time = time.time()

        logging.info(
            f"✅ AsyncTradeBuffer inicializado | "
            f"Max size: {max_size} | "
            f"Backpressure: {backpressure_threshold*100:.0f}% | "
            f"Batch size: {processing_batch_size} | "
            f"Drain timeout: {drain_timeout_ms}ms"
        )

    async def start(self) -> None:
        """Inicia ou reinicia o processamento background."""
        with self._buffer_lock:
            if self._is_stopping:
                raise RuntimeError("Cannot start AsyncTradeBuffer while stop/drain is in progress")
            if self._processing_task and not self._processing_task.done():
                return
            self._should_stop.clear()
            self._accepting_trades = True
            self._is_stopped = False
            self._processing_task = asyncio.create_task(self._processing_loop())
        logging.info("✅ AsyncTradeBuffer iniciado")

    async def stop(self, timeout_ms: Optional[float] = None) -> None:
        """
        Para o processamento background com bounded drain (drenagem delimitada).

        1. Marca buffer como não aceitando novos trades (_accepting_trades = False).
        2. Sinaliza intenção de parada (_should_stop.set()).
        3. Permite que o worker continue drenando.
        4. Espera até:
           A) buffer vazio E batch in-flight concluído (shutdown limpo);
           OU
           B) deadline expirado.
        5. Se deadline expirar:
           - Contabiliza explicitamente trades restantes no deque em _dropped_shutdown_trades.
           - Contabiliza trades in-flight interrompidos por cancelamento em _dropped_shutdown_trades.
           - Limpa buffer e encerra o worker.
        """
        if timeout_ms is None:
            timeout_ms = self.drain_timeout_ms
        else:
            if isinstance(timeout_ms, bool):
                raise TypeError("timeout_ms cannot be a boolean")
            if not isinstance(timeout_ms, (int, float)):
                raise TypeError(f"timeout_ms must be a number, got {type(timeout_ms).__name__}")
            if math.isnan(timeout_ms) or math.isinf(timeout_ms):
                raise ValueError("timeout_ms cannot be NaN or Inf")
            if timeout_ms < 0.0:
                raise ValueError(f"timeout_ms cannot be negative, got {timeout_ms}")

        timeout_s = max(0.0, float(timeout_ms) / 1000.0)

        with self._buffer_lock:
            if self._is_stopped and (self._processing_task is None or self._processing_task.done()):
                return
            self._is_stopping = True
            self._accepting_trades = False
            self._should_stop.set()

        try:
            task = self._processing_task
            if task is None or task.done():
                with self._buffer_lock:
                    remaining = len(self._buffer)
                    self._buffer.clear()
                    if remaining > 0:
                        self._dropped_shutdown_trades += remaining
                        self._dropped_trades_count += remaining
                        if METRICS_AVAILABLE:
                            record_trades_dropped_shutdown(remaining)
                            set_trades_in_buffer(0)
                        try:
                            from audit_live.forensic_context import CTX as _fctx
                            _fctx.inc("trade_buffer_shutdown_drops", delta=remaining)
                        except Exception:
                            pass
                    self._is_stopped = True
                    self._processing_task = None
                logging.info("✅ AsyncTradeBuffer parado (sem worker ativo)")
                return

            deadline = time.monotonic() + timeout_s
            clean_drain = False

            while True:
                with self._buffer_lock:
                    buf_len = len(self._buffer)
                    in_flight = self._in_flight_count

                if buf_len == 0 and in_flight == 0:
                    clean_drain = True
                    break

                if time.monotonic() >= deadline:
                    break

                await asyncio.sleep(0.005)

            if clean_drain:
                remaining_wait = max(0.05, deadline - time.monotonic())
                try:
                    await asyncio.wait_for(task, timeout=remaining_wait)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                with self._buffer_lock:
                    self._is_stopped = True
                    self._processing_task = None
                    if METRICS_AVAILABLE:
                        set_trades_in_buffer(0)
                logging.info("✅ AsyncTradeBuffer finalizado com drain limpo")
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                with self._buffer_lock:
                    remaining_buf = len(self._buffer)
                    self._buffer.clear()
                    remaining_inflight = self._in_flight_count
                    self._in_flight_count = 0
                    self._current_in_flight_trades.clear()

                    dropped_shutdown = remaining_buf + remaining_inflight
                    if dropped_shutdown > 0:
                        self._dropped_shutdown_trades += dropped_shutdown
                        self._dropped_trades_count += dropped_shutdown
                        if METRICS_AVAILABLE:
                            record_trades_dropped_shutdown(dropped_shutdown)
                            set_trades_in_buffer(0)
                        try:
                            from audit_live.forensic_context import CTX as _fctx
                            _fctx.inc("trade_buffer_shutdown_drops", delta=dropped_shutdown)
                        except Exception:
                            pass
                        logging.warning(
                            f"🚨 AsyncTradeBuffer shutdown timeout expirado ({timeout_ms}ms): "
                            f"descartados {dropped_shutdown} trades "
                            f"(buffer={remaining_buf}, in_flight={remaining_inflight})"
                        )
                    self._is_stopped = True
                    self._processing_task = None
                    if METRICS_AVAILABLE:
                        set_trades_in_buffer(0)
                logging.info("✅ AsyncTradeBuffer parado após timeout de drain")
        finally:
            with self._buffer_lock:
                self._is_stopping = False

    def add_trade_sync(self, trade: Dict[str, Any], processor: Callable) -> bool:
        """
        Versão thread-safe síncrona para adicionar trade ao buffer.
        Ponto único canônico de admissão, medição de latência e append no buffer.

        Args:
            trade: Dados do trade
            processor: Função para processar o trade

        Returns:
            True: trade NOVO foi aceito no buffer (mesmo se trades antigos foram ejetados por overflow).
            False: trade NOVO foi rejeitado (ex: buffer não aceita trades durante ou após shutdown).
        """
        start_time = time.perf_counter()

        if LATENCY_MONITOR_AVAILABLE and latency_monitor:
            latency_monitor.record_trade(trade)

        trade['_received_at_ms'] = int(time.time() * 1000)

        with self._buffer_lock:
            if not self._accepting_trades or self._is_stopped:
                return False

            current_size = len(self._buffer)

            # Verifica overflow
            if current_size >= self.max_size:
                self._overflow_events += 1
                self._overflow_count += 1
                # Remover trades mais antigos se necessário (backpressure) - DROP_OLDEST de 10%
                if current_size >= self.max_size * 0.95:
                    trades_to_remove = max(1, int(self.max_size * 0.1))
                    removed = 0
                    for _ in range(trades_to_remove):
                        if self._buffer:
                            self._buffer.popleft()
                            removed += 1
                    if removed:
                        self._backpressure_count += 1
                        self._dropped_overflow_trades += removed
                        self._dropped_trades_count += removed
                        if METRICS_AVAILABLE:
                            record_trades_dropped_overflow(removed)
                        try:
                            from audit_live.forensic_context import CTX as _fctx
                            _fctx.inc("trade_buffer_overflow_drops", delta=removed)
                        except Exception:
                            pass
                        logging.warning(
                            f"🚨 Buffer overflow: removidos {removed} trades antigos "
                            f"(total descartados overflow: {self._dropped_overflow_trades})"
                        )
                current_size = len(self._buffer)

            # ÚNICO ponto de append no buffer em todo o módulo (trade NOVO aceito)
            self._buffer.append((trade, processor))
            self._total_accepted_trades += 1
            current_size += 1

            if METRICS_AVAILABLE:
                set_trades_in_buffer(current_size)

            fill_ratio = current_size / self.max_size
            new_status = self._get_buffer_status(fill_ratio)

            if new_status != self._current_status:
                self._current_status = new_status
                if self.warning_callback:
                    self.warning_callback(new_status, current_size)

                if new_status == BufferStatus.WARNING:
                    logging.warning(
                        f"⚠️ Buffer approaching limit: {current_size}/{self.max_size} "
                        f"({fill_ratio*100:.1f}%)"
                    )
                elif new_status == BufferStatus.CRITICAL:
                    logging.warning(
                        f"🚨 Buffer critical: {current_size}/{self.max_size} "
                        f"({fill_ratio*100:.1f}%)"
                    )
                elif new_status == BufferStatus.NORMAL:
                    logging.info("✅ Buffer level normalized")

        self._trade_count += 1
        self._last_trade_time = start_time
        return True

    async def add_trade(self, trade: Dict[str, Any], processor: Callable) -> bool:
        """
        Adiciona trade ao buffer (versão async).

        Args:
            trade: Dados do trade
            processor: Função para processar o trade

        Returns:
            True se o trade foi aceito no buffer, False se rejeitado.
        """
        return self.add_trade_sync(trade, processor)

    def add_trades_batch_sync(self, trades: list, processor: Callable, max_age_seconds: int = 30) -> int:
        """
        Adiciona múltiplos trades ao buffer iterando por add_trade_sync.
        A latência é registrada exatamente uma vez por trade via add_trade_sync.

        Args:
            trades: Lista de trades para adicionar
            processor: Função para processar cada trade
            max_age_seconds: Parâmetro mantido para compatibilidade (não utilizado)

        Returns:
            Número de trades aceitos pelo buffer.
        """
        if not trades:
            return 0

        added_count = 0
        for trade in trades:
            if self.add_trade_sync(trade, processor):
                added_count += 1

        return added_count

    async def add_trades_batch(self, trades: list, processor: Callable, max_age_seconds: int = 30) -> int:
        """
        Versão assíncrona de add_trades_batch_sync.
        """
        return self.add_trades_batch_sync(trades, processor, max_age_seconds)

    async def _processing_loop(self):
        """Loop de processamento background com suporte a bounded drain."""
        while True:
            try:
                with self._buffer_lock:
                    should_stop = self._should_stop.is_set()
                    buf_len = len(self._buffer)
                    in_flight = self._in_flight_count

                # Encerramento gracioso quando stop é sinalizado e não há itens pendentes nem in-flight
                if should_stop and buf_len == 0 and in_flight == 0:
                    break

                # Se não há itens para processar no buffer:
                if buf_len == 0:
                    if should_stop:
                        break
                    await asyncio.sleep(self.processing_interval_ms / 1000.0)
                    continue

                batch = self._get_batch()
                if batch:
                    batch_start = time.perf_counter()

                    if len(batch) >= 20:
                        await self._process_batch_parallel(batch)
                    else:
                        await self._process_batch_sequential(batch)

                    batch_time = (time.perf_counter() - batch_start) * 1000

                    avg_time = batch_time / len(batch) if batch else 0.0
                    if avg_time > self.max_processing_time_ms * 3:
                        logging.warning(
                            f"Batch processing slow: {batch_time:.2f}ms "
                            f"for {len(batch)} trades "
                            f"(avg: {avg_time:.2f}ms/trade)"
                        )
                    elif batch_time > self.max_processing_time_ms * len(batch):
                        logging.debug(
                            f"Batch processing above target: {batch_time:.2f}ms "
                            f"for {len(batch)} trades "
                            f"(avg: {avg_time:.2f}ms/trade)"
                        )

                    if self.heartbeat_callback:
                        try:
                            self.heartbeat_callback('trade_buffer')
                        except Exception as e:
                            logging.debug(f"Erro ao enviar heartbeat: {e}")

                    with self._buffer_lock:
                        if len(self._buffer) > 0:
                            continue

                await asyncio.sleep(self.processing_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Erro no processing loop: {e}")
                await asyncio.sleep(0.05)

    async def _process_batch_sequential(self, batch: list) -> None:
        """Processa batch sequencialmente com rastreamento preciso de in-flight e erros."""
        for item in batch:
            trade, processor = item
            trade_start = time.perf_counter()
            proc_error = False
            completed = False
            try:
                with track_latency("trade_processing"):
                    result = processor(trade)
                    if inspect.isawaitable(result):
                        await result
                completed = True
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                proc_error = True
                completed = True
                if METRICS_AVAILABLE:
                    record_timeout()
                logging.error("Timeout ao processar trade")
            except Exception as e:
                proc_error = True
                completed = True
                logging.error(f"Erro ao processar trade: {e}")
            finally:
                if completed:
                    trade_time = (time.perf_counter() - trade_start) * 1000
                    with self._buffer_lock:
                        if proc_error:
                            self._processor_error_trades += 1
                        if item in self._current_in_flight_trades:
                            self._current_in_flight_trades.remove(item)
                        self._in_flight_count = max(0, self._in_flight_count - 1)
                    with self._processing_lock:
                        self._processing_times.append(trade_time)
                        if not proc_error:
                            self._total_trades_processed += 1
                        self._total_processing_time_ms += trade_time

    async def _process_batch_parallel(self, batch: list) -> None:
        """Processa batch em paralelo particionando em workers."""
        num_workers = 4
        chunk_size = max(1, len(batch) // num_workers)
        chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]

        await asyncio.gather(*[
            self._process_chunk(chunk) for chunk in chunks
        ])

    async def _process_chunk(self, chunk: list) -> None:
        """Processa um chunk de trades em paralelo com rastreamento de in-flight e erros."""
        for item in chunk:
            trade, processor = item
            trade_start = time.perf_counter()
            proc_error = False
            completed = False
            try:
                with track_latency("trade_processing"):
                    result = processor(trade)
                    if inspect.isawaitable(result):
                        await result
                completed = True
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                proc_error = True
                completed = True
                if METRICS_AVAILABLE:
                    record_timeout()
                logging.error("Timeout ao processar trade")
            except Exception as e:
                proc_error = True
                completed = True
                logging.error(f"Erro ao processar trade: {e}")
            finally:
                if completed:
                    trade_time = (time.perf_counter() - trade_start) * 1000
                    with self._buffer_lock:
                        if proc_error:
                            self._processor_error_trades += 1
                        if item in self._current_in_flight_trades:
                            self._current_in_flight_trades.remove(item)
                        self._in_flight_count = max(0, self._in_flight_count - 1)
                    with self._processing_lock:
                        self._processing_times.append(trade_time)
                        if not proc_error:
                            self._total_trades_processed += 1
                        self._total_processing_time_ms += trade_time

    def _get_batch(self):
        """Obtém batch de trades do buffer e atualiza contagem in-flight."""
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return []

            batch_size = min(self.processing_batch_size, len(self._buffer))
            batch = []

            for _ in range(batch_size):
                if self._buffer:
                    batch.append(self._buffer.popleft())

            self._in_flight_count += len(batch)
            self._current_in_flight_trades.extend(batch)

            if METRICS_AVAILABLE:
                set_trades_in_buffer(len(self._buffer))

            return batch

    def _get_buffer_status(self, fill_ratio: float) -> BufferStatus:
        """Determina status do buffer baseado no fill ratio."""
        if fill_ratio >= 1.0:
            return BufferStatus.OVERFLOW
        elif fill_ratio >= 0.9:
            return BufferStatus.CRITICAL
        elif fill_ratio >= self.backpressure_threshold:
            return BufferStatus.WARNING
        else:
            return BufferStatus.NORMAL

    def get_metrics(self) -> TradeLatencyMetrics:
        """Retorna métricas atuais do buffer."""
        with self._buffer_lock:
            buffer_size = len(self._buffer)
            buffer_capacity = self.max_size
            buffer_fill_ratio = buffer_size / buffer_capacity if buffer_capacity > 0 else 0.0

        with self._processing_lock:
            processing_times = list(self._processing_times)
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            p95_time = sorted(processing_times)[int(len(processing_times) * 0.95)] if processing_times else 0.0
            max_time = max(processing_times) if processing_times else 0.0

            trades_per_sec = self._calculate_trades_per_second()

        last_latency = 0.0
        if self._last_trade_time > 0:
            last_latency = (time.perf_counter() - self._last_trade_time) * 1000

        return TradeLatencyMetrics(
            buffer_size=buffer_size,
            buffer_capacity=buffer_capacity,
            buffer_fill_ratio=buffer_fill_ratio,
            avg_processing_time_ms=avg_time,
            p95_processing_time_ms=p95_time,
            max_processing_time_ms=max_time,
            trades_per_second=trades_per_sec,
            status=self._current_status,
            last_trade_latency_ms=last_latency
        )

    def _calculate_trades_per_second(self) -> float:
        """Calcula trades processados por segundo sem reentrar em processing_lock."""
        now = time.time()
        time_window = 60.0

        if hasattr(self, '_start_time'):
            elapsed = now - self._start_time
            if elapsed > 0:
                return self._total_trades_processed / elapsed

        return self._trade_count / time_window if time_window > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas detalhadas com contabilidade precisa e thread-safe.

        Nota sobre contabilidade:
        - dropped_overflow_trades: descartes de trades antigos por backpressure (DROP_OLDEST).
        - dropped_shutdown_trades: descartes de trades do buffer e in-flight por expiração do timeout de drain.
        - dropped_trades_count: campo legado que contabiliza estritamente os 'buffer drops'
          (dropped_overflow_trades + dropped_shutdown_trades), NÃO incluindo falhas internas
          do processador (processor_error_trades).
        - processor_error_trades: contabiliza falhas/exceções disparadas durante o processamento do trade.
        """
        metrics = self.get_metrics()
        with self._buffer_lock:
            accepted_total = self._total_accepted_trades
            overflow_events = self._overflow_events
            dropped_overflow = self._dropped_overflow_trades
            dropped_shutdown = self._dropped_shutdown_trades
            dropped_total = dropped_overflow + dropped_shutdown
            processor_errors = self._processor_error_trades
            in_flight = self._in_flight_count
            accepting = self._accepting_trades
            is_stopped = self._is_stopped
            backpressure_cnt = self._backpressure_count

        with self._processing_lock:
            total_proc = self._total_trades_processed

        return {
            "buffer": {
                "current_size": metrics.buffer_size,
                "capacity": metrics.buffer_capacity,
                "fill_ratio": metrics.buffer_fill_ratio,
                "status": metrics.status.value,
                "accepting_trades": accepting,
                "is_stopped": is_stopped,
                "in_flight_count": in_flight,
                "total_accepted_trades": accepted_total,
                "overflow_events": overflow_events,
                "overflow_count": overflow_events,  # compatibilidade
                "backpressure_count": backpressure_cnt,
                "dropped_trades_count": dropped_total,  # compatibilidade
                "dropped_overflow_trades": dropped_overflow,
                "dropped_shutdown_trades": dropped_shutdown,
                "processor_error_trades": processor_errors,
            },
            "processing": {
                "total_trades": total_proc,
                "avg_time_ms": metrics.avg_processing_time_ms,
                "p95_time_ms": metrics.p95_processing_time_ms,
                "max_time_ms": metrics.max_processing_time_ms,
                "trades_per_second": metrics.trades_per_second,
            },
            "latency": {
                "last_trade_ms": metrics.last_trade_latency_ms,
            }
        }

    def is_healthy(self) -> bool:
        """Verifica se o buffer está saudável."""
        metrics = self.get_metrics()
        return (
            metrics.status not in [BufferStatus.CRITICAL, BufferStatus.OVERFLOW] and
            metrics.avg_processing_time_ms < self.max_processing_time_ms and
            metrics.p95_processing_time_ms < self.max_processing_time_ms * 2
        )
