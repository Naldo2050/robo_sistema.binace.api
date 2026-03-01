# trade_buffer.py
"""
Buffer assíncrono de trades com backpressure e métricas de latência.

Resolve o problema de trades atrasados implementando:
- Buffer com tamanho limitado e backpressure
- Processamento assíncrono em background
- Métricas de latência end-to-end
- Alertas de buffer crítico
"""

import asyncio
import inspect
import logging
import time
from collections import deque
from threading import Lock, Event
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Importar métricas do sistema
try:
    from metrics_collector import (
        record_trade_late,
        update_active_trades_buffer as set_trades_in_buffer,
        track_operation as track_latency,  # type: ignore[assignment]
        record_timeout
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

# Importar validador de trades
try:
    from trade_validator import validate_and_filter_trades
    TRADE_VALIDATOR_AVAILABLE = True
except ImportError:
    TRADE_VALIDATOR_AVAILABLE = False
    def validate_and_filter_trades(trades, max_age_seconds=30):
        return trades

# Importar monitor de latência (NOVO - não descarta trades)
try:
    from trade_validator import latency_monitor
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
    Buffer assíncrono de trades com backpressure.
    
    Características:
    - Buffer com tamanho máximo configurável
    - Backpressure quando > 80% de capacidade
    - Processamento assíncrono em background
    - Métricas de latência em tempo real
    - Alertas automáticos de overflow
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        backpressure_threshold: float = 0.8,
        processing_batch_size: int = 50,
        processing_interval_ms: int = 10,
        max_processing_time_ms: float = 5.0,
        warning_callback: Optional[Callable[[BufferStatus, int], None]] = None,
        heartbeat_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Inicializa o buffer assíncrono.
        
        Args:
            max_size: Tamanho máximo do buffer (aumentado para 5000)
            backpressure_threshold: Limite para ativar backpressure (0.8 = 80%)
            processing_batch_size: Número de trades para processar por batch
            processing_interval_ms: Intervalo entre processamentos em ms
            max_processing_time_ms: Tempo máximo aceitável para processar 1 trade
            warning_callback: Callback para alertas de buffer
            heartbeat_callback: Callback para enviar heartbeats do módulo
        """
        self.max_size = max_size
        self.backpressure_threshold = backpressure_threshold
        self.processing_batch_size = processing_batch_size
        self.processing_interval_ms = processing_interval_ms
        self.max_processing_time_ms = max_processing_time_ms
        self.warning_callback = warning_callback
        self.heartbeat_callback = heartbeat_callback
        
        # Buffer principal
        self._buffer: deque = deque(maxlen=max_size)
        self._buffer_lock = Lock()
        
        # Controle de processamento
        self._processing_lock = Lock()
        self._should_stop = Event()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Métricas de latência
        self._processing_times = deque(maxlen=1000)
        self._trade_count = 0
        self._last_trade_time = 0.0
        self._last_metrics_update = 0.0
        
        # Status atual
        self._current_status = BufferStatus.NORMAL
        
        # Estatísticas
        self._overflow_count = 0
        self._backpressure_count = 0
        self._total_trades_processed = 0
        self._total_processing_time_ms = 0.0
        
        # Inicializa tempo de início para cálculo de TPS
        self._start_time = time.time()
        
        logging.info(
            f"✅ AsyncTradeBuffer inicializado | "
            f"Max size: {max_size} | "
            f"Backpressure: {backpressure_threshold*100:.0f}% | "
            f"Batch size: {processing_batch_size}"
        )
    
    async def start(self):
        """Inicia o processamento background."""
        if self._processing_task and not self._processing_task.done():
            return
        self._should_stop.clear()
        self._processing_task = asyncio.create_task(self._processing_loop())
        logging.info("✅ AsyncTradeBuffer iniciado")
    
    async def stop(self):
        """Para o processamento background."""
        if not self._processing_task:
            return
        self._processing_task.cancel()
        try:
            await self._processing_task
        except asyncio.CancelledError:
            pass
        self._processing_task = None
        logging.info("✅ AsyncTradeBuffer parado")
    
    def add_trade_sync(self, trade: Dict[str, Any], processor: Callable) -> bool:
        """
        Versão thread-safe síncrona para adicionar trade ao buffer.
        
        Args:
            trade: Dados do trade
            processor: Função para processar o trade

        Returns:
            True sempre - trades nunca são descartados por latência
        """
        start_time = time.perf_counter()

        # NOVO: Registrar latência para monitoramento (NÃO descarta trade)
        if LATENCY_MONITOR_AVAILABLE and latency_monitor:
            latency_monitor.record_trade(trade)

        # Adiciona timestamp de recebimento
        trade['_received_at_ms'] = int(time.time() * 1000)
        
        with self._buffer_lock:
            current_size = len(self._buffer)
            
            # Verifica overflow
            if current_size >= self.max_size:
                self._overflow_count += 1
                # Remover trades mais antigos se necessário (backpressure)
                if current_size >= self.max_size * 0.95:
                    trades_to_remove = int(self.max_size * 0.1)
                    for _ in range(trades_to_remove):
                        if self._buffer:
                            self._buffer.popleft()
                    logging.debug(f"Backpressure: removidos {trades_to_remove} trades antigos")
                current_size = len(self._buffer)
                
            # Adiciona ao buffer
            self._buffer.append((trade, processor))
            current_size += 1
            
            # Atualizar métrica de trades no buffer
            if METRICS_AVAILABLE:
                set_trades_in_buffer(current_size)
            
            # Atualiza status
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
        
        # Estatísticas
        self._trade_count += 1
        self._last_trade_time = start_time
        
        return True
    
    async def add_trade(self, trade: Dict[str, Any], processor: Callable) -> bool:
        """
        Adiciona trade ao buffer com controle de backpressure (versão async).
        
        Args:
            trade: Dados do trade
            processor: Função para processar o trade

        Returns:
            True sempre - trades nunca são descartados por latência
        """
        start_time = time.perf_counter()

        # NOVO: Registrar latência para monitoramento (NÃO descarta trade)
        if LATENCY_MONITOR_AVAILABLE and latency_monitor:
            latency_monitor.record_trade(trade)

        # Adiciona timestamp de recebimento
        trade['_received_at_ms'] = int(time.time() * 1000)
        
        with self._buffer_lock:
            current_size = len(self._buffer)
            
            # Verifica overflow
            if current_size >= self.max_size:
                self._overflow_count += 1
                # Remover trades mais antigos se necessário (backpressure)
                if current_size >= self.max_size * 0.95:
                    trades_to_remove = int(self.max_size * 0.1)
                    for _ in range(trades_to_remove):
                        if self._buffer:
                            self._buffer.popleft()
                    logging.debug(f"Backpressure: removidos {trades_to_remove} trades antigos")
                current_size = len(self._buffer)
                
            # Adiciona ao buffer
            self._buffer.append((trade, processor))
            current_size += 1
            
            # Atualizar métrica de trades no buffer
            if METRICS_AVAILABLE:
                set_trades_in_buffer(current_size)
            
            # Atualiza status
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
        
        # Estatísticas
        self._trade_count += 1
        self._last_trade_time = start_time
        
        return True

    def add_trades_batch_sync(self, trades: list, processor: Callable, max_age_seconds: int = 30) -> int:
        """
        Adiciona múltiplos trades ao buffer. TODOS os trades são aceitos.
        A latência é monitorada mas nenhum trade é descartado.

        Args:
            trades: Lista de trades para adicionar
            processor: Função para processar cada trade
            max_age_seconds: Parâmetro mantido para compatibilidade (não utilizado)

        Returns:
            Número de trades adicionados (todos os trades são aceitos)
        """
        if not trades:
            return 0

        # Registrar latência para todos os trades (NÃO descarta)
        if LATENCY_MONITOR_AVAILABLE and latency_monitor:
            for trade in trades:
                latency_monitor.record_trade(trade)

        # Adicionar todos os trades ao buffer
        added_count = 0
        for trade in trades:
            if self.add_trade_sync(trade, processor):
                added_count += 1

        return added_count

    async def add_trades_batch(self, trades: list, processor: Callable, max_age_seconds: int = 30) -> int:
        """
        Versão assíncrona de add_trades_batch_sync.
        TODOS os trades são aceitos - apenas monitoramos latência.

        Args:
            trades: Lista de trades para adicionar
            processor: Função para processar cada trade
            max_age_seconds: Parâmetro mantido para compatibilidade (não utilizado)

        Returns:
            Número de trades adicionados (todos os trades são aceitos)
        """
        if not trades:
            return 0

        # Registrar latência para todos os trades (NÃO descarta)
        if LATENCY_MONITOR_AVAILABLE and latency_monitor:
            for trade in trades:
                latency_monitor.record_trade(trade)

        # Adicionar todos os trades ao buffer
        added_count = 0
        for trade in trades:
            if await self.add_trade(trade, processor):
                added_count += 1

        return added_count

    async def _processing_loop(self):
        """Loop de processamento background."""
        while not self._should_stop.is_set():
            try:
                # Enviar heartbeat para o módulo buffer_critical
                if self.heartbeat_callback:
                    try:
                        self.heartbeat_callback('buffer_critical')
                    except Exception as e:
                        logging.debug(f"Erro ao enviar heartbeat: {e}")
                
                # Processa batch de trades
                batch = self._get_batch()
                if batch:
                    batch_start = time.perf_counter()
                    
                    # OTIMIZAÇÃO: Processamento paralelo para batches grandes
                    if len(batch) >= 20:
                        await self._process_batch_parallel(batch)
                    else:
                        # Processamento sequencial para batches pequenos
                        await self._process_batch_sequential(batch)
                    
                    batch_time = (time.perf_counter() - batch_start) * 1000
                    
                    # Log de performance se muito lento
                    if batch_time > self.max_processing_time_ms * len(batch):
                        logging.warning(
                            f"Batch processing slow: {batch_time:.2f}ms "
                            f"for {len(batch)} trades "
                            f"(avg: {batch_time/len(batch):.2f}ms/trade)"
                        )

                    # Se ainda houver backlog, drena sem dormir
                    if len(self._buffer) > 0:
                        continue

                # Intervalo entre processamentos
                await asyncio.sleep(self.processing_interval_ms / 1000.0)

            except Exception as e:
                logging.error(f"Erro no processing loop: {e}")
                await asyncio.sleep(0.05)
        
    async def _process_batch_sequential(self, batch: list) -> None:
        """Processa batch sequencialmente (otimizado para batches pequenos)."""
        for trade, processor in batch:
            trade_start = time.perf_counter()
            try:
                with track_latency("trade_processing"):
                    result = processor(trade)
                    if inspect.isawaitable(result):
                        await result
            except asyncio.TimeoutError:
                if METRICS_AVAILABLE:
                    record_timeout()
                logging.error(f"Timeout ao processar trade")
            except Exception as e:
                logging.error(f"Erro ao processar trade: {e}")
            
            # Métricas de tempo de processamento
            trade_time = (time.perf_counter() - trade_start) * 1000
            with self._processing_lock:
                self._processing_times.append(trade_time)
                self._total_trades_processed += 1
                self._total_processing_time_ms += trade_time
    
    async def _process_batch_parallel(self, batch: list) -> None:
        """
        Processa batch em paralelo (otimização para reduzir latência).
        
        Args:
            batch: Lista de tuples (trade, processor)
        """
        # Divide em 4 chunks para processamento paralelo
        num_workers = 4
        chunk_size = max(1, len(batch) // num_workers)
        chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]
        
        # Processa todos os chunks em paralelo
        await asyncio.gather(*[
            self._process_chunk(chunk) for chunk in chunks
        ])
    
    async def _process_chunk(self, chunk: list) -> None:
        """
        Processa um chunk de trades.
        
        Args:
            chunk: Lista de tuples (trade, processor)
        """
        for trade, processor in chunk:
            trade_start = time.perf_counter()
            try:
                with track_latency("trade_processing"):
                    result = processor(trade)
                    if inspect.isawaitable(result):
                        await result
            except asyncio.TimeoutError:
                if METRICS_AVAILABLE:
                    record_timeout()
                logging.error(f"Timeout ao processar trade")
            except Exception as e:
                logging.error(f"Erro ao processar trade: {e}")
            
            # Métricas de tempo de processamento
            trade_time = (time.perf_counter() - trade_start) * 1000
            with self._processing_lock:
                self._processing_times.append(trade_time)
                self._total_trades_processed += 1
                self._total_processing_time_ms += trade_time
        
    def _get_batch(self):
        """Obtem batch de trades para processar."""
        with self._buffer_lock:
            if len(self._buffer) == 0:
                return []
            
            # Determina tamanho do batch
            batch_size = min(self.processing_batch_size, len(self._buffer))
            batch = []
            
            # Extrai trades do buffer
            for _ in range(batch_size):
                if self._buffer:
                    batch.append(self._buffer.popleft())
            
            # Atualizar métrica após remover batch
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
        
        # Latência do último trade
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
        """Calcula trades processados por segundo."""
        now = time.time()
        time_window = 60.0  # 1 minuto
        
        # Estima baseado no total e tempo decorrido
        if hasattr(self, '_start_time'):
            elapsed = now - self._start_time
            if elapsed > 0:
                return self._total_trades_processed / elapsed
        
        # Fallback simples
        return self._trade_count / time_window if time_window > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas."""
        metrics = self.get_metrics()
        
        return {
            "buffer": {
                "current_size": metrics.buffer_size,
                "capacity": metrics.buffer_capacity,
                "fill_ratio": metrics.buffer_fill_ratio,
                "status": metrics.status.value,
                "overflow_count": self._overflow_count,
                "backpressure_count": self._backpressure_count,
            },
            "processing": {
                "total_trades": self._total_trades_processed,
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
