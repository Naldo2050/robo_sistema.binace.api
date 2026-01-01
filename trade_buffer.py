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
        max_size: int = 2000,
        backpressure_threshold: float = 0.8,
        processing_batch_size: int = 50,
        processing_interval_ms: int = 10,
        max_processing_time_ms: float = 5.0,
        warning_callback: Optional[Callable[[BufferStatus, int], None]] = None
    ):
        """
        Inicializa o buffer assíncrono.
        
        Args:
            max_size: Tamanho máximo do buffer
            backpressure_threshold: Limite para ativar backpressure (0.8 = 80%)
            processing_batch_size: Número de trades para processar por batch
            processing_interval_ms: Intervalo entre processamentos em ms
            max_processing_time_ms: Tempo máximo aceitável para processar 1 trade
            warning_callback: Callback para alertas de buffer
        """
        self.max_size = max_size
        self.backpressure_threshold = backpressure_threshold
        self.processing_batch_size = processing_batch_size
        self.processing_interval_ms = processing_interval_ms
        self.max_processing_time_ms = max_processing_time_ms
        self.warning_callback = warning_callback
        
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
            True se adicionado com sucesso, False se descartado por overflow
        """
        start_time = time.perf_counter()
        
        # Adiciona timestamp de recebimento
        trade['_received_at_ms'] = int(time.time() * 1000)
        
        with self._buffer_lock:
            current_size = len(self._buffer)
            
            # Verifica overflow
            if current_size >= self.max_size:
                self._overflow_count += 1
                logging.warning(
                    f"⚠️ Buffer overflow! Descartando trade. "
                    f"Size: {current_size}/{self.max_size}, "
                    f"Overflows: {self._overflow_count}"
                )
                return False
            
            # Adiciona ao buffer
            self._buffer.append((trade, processor))
            current_size += 1
            
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
            True se adicionado com sucesso, False se descartado por overflow
        """
        start_time = time.perf_counter()
        
        # Adiciona timestamp de recebimento
        trade['_received_at_ms'] = int(time.time() * 1000)
        
        with self._buffer_lock:
            current_size = len(self._buffer)
            
            # Verifica overflow
            if current_size >= self.max_size:
                self._overflow_count += 1
                logging.warning(
                    f"⚠️ Buffer overflow! Descartando trade. "
                    f"Size: {current_size}/{self.max_size}, "
                    f"Overflows: {self._overflow_count}"
                )
                return False
            
            # Adiciona ao buffer
            self._buffer.append((trade, processor))
            current_size += 1
            
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
    
    async def _processing_loop(self):
        """Loop de processamento background."""
        while not self._should_stop.is_set():
            try:
                # Processa batch de trades
                batch = self._get_batch()
                if batch:
                    batch_start = time.perf_counter()
                    
                    # Processa trades do batch
                    for trade, processor in batch:
                        trade_start = time.perf_counter()
                        try:
                            result = processor(trade)
                            if inspect.isawaitable(result):
                                await result
                        except Exception as e:
                            logging.error(f"❌ Erro ao processar trade: {e}")
                        
                        # Métricas de tempo de processamento
                        trade_time = (time.perf_counter() - trade_start) * 1000
                        with self._processing_lock:
                            self._processing_times.append(trade_time)
                            self._total_trades_processed += 1
                            self._total_processing_time_ms += trade_time
                    
                    batch_time = (time.perf_counter() - batch_start) * 1000
                    
                    # Log de performance se muito lento
                    if batch_time > self.max_processing_time_ms * len(batch):
                        logging.warning(
                            f"⚠️ Batch processing slow: {batch_time:.2f}ms "
                            f"for {len(batch)} trades "
                            f"(avg: {batch_time/len(batch):.2f}ms/trade)"
                        )
                
                # Intervalo entre processamentos
                await asyncio.sleep(self.processing_interval_ms / 1000.0)
                
            except Exception as e:
                logging.error(f"❌ Erro no processing loop: {e}")
                await asyncio.sleep(0.1)
    
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
