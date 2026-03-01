#!/usr/bin/env python3
"""
Coletor de Métricas para Monitoramento do Sistema de Trading
Implementa métricas Prometheus e integração com AlertManager
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Sequence
from collections import deque
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

# Tentar importar prometheus_client, com fallback para implementação simulada
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server, REGISTRY, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client não disponível, usando implementação simulada")

# ----------------------------------------------------------------------
# Classes de métricas simuladas (fallback)
# ----------------------------------------------------------------------

class SimulatedMetric:
    """Base para métricas simuladas quando Prometheus não está disponível"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0.0
        self._labels: Dict[str, str] = {}
        self._histogram_values: List[float] = []
    
    def inc(self, amount: float = 1.0):
        self._value += amount
    
    def dec(self, amount: float = 1.0):
        self._value -= amount
    
    def set(self, value: float):
        self._value = value
    
    def observe(self, value: float):
        self._value += value
        self._histogram_values.append(value)
    
    def labels(self, **kwargs):
        self._labels = kwargs
        return self


class SimulatedCounter(SimulatedMetric):
    """Contador simulado"""
    pass


class SimulatedHistogram(SimulatedMetric):
    """Histograma simulado"""
    
    def __init__(self, name: str, description: str, buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)):
        super().__init__(name, description)
        self.buckets = buckets
        self._bucket_counts = {b: 0 for b in buckets}
    
    def observe(self, value: float):
        super().observe(value)
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1


class SimulatedGauge(SimulatedMetric):
    """Gauge simulado"""
    pass


class SimulatedSummary(SimulatedMetric):
    """Summary simulada"""
    pass


# ----------------------------------------------------------------------
# Factory de métricas
# ----------------------------------------------------------------------

def create_counter(name: str, description: str):
    """Cria um contador Prometheus ou simulado"""
    if PROMETHEUS_AVAILABLE:
        return Counter(name, description)
    return SimulatedCounter(name, description)


def create_histogram(name: str, description: str, buckets: Optional[Sequence[float]] = None):
    """Cria um histograma Prometheus ou simulado"""
    default_buckets = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
    bucket_list = buckets or default_buckets
    if PROMETHEUS_AVAILABLE:
        return Histogram(name, description, buckets=bucket_list)
    return SimulatedHistogram(name, description, buckets=bucket_list)


def create_gauge(name: str, description: str):
    """Cria um gauge Prometheus ou simulado"""
    if PROMETHEUS_AVAILABLE:
        return Gauge(name, description)
    return SimulatedGauge(name, description)


# ----------------------------------------------------------------------
# Métricas do Sistema de Trading
# ----------------------------------------------------------------------

# Contadores de eventos
coroutine_warnings = create_counter(
    'trading_coroutine_warnings_total',
    'Warnings de corotinas não aguardadas'
)

trades_late = create_counter(
    'trading_trades_late_total',
    'Trades recebidos atrasados'
)

timeouts_occurred = create_counter(
    'trading_timeouts_total',
    'Timeouts ocorridos'
)

fred_fallback_used = create_counter(
    'trading_fred_fallback_used_total',
    'Fallback do FRED utilizado'
)

data_corrections = create_counter(
    'trading_data_corrections_total',
    'Correções de dados aplicadas'
)

parse_errors = create_counter(
    'trading_parse_errors_total',
    'Erros de parsing de dados'
)

enrich_errors = create_counter(
    'trading_enrich_errors_total',
    'Erros no enrichment de eventos'
)

connection_losses = create_counter(
    'trading_connection_losses_total',
    'Perdas de conexão'
)

# Histogramas de latência
processing_latency = create_histogram(
    'trading_processing_latency_seconds',
    'Latência do processamento',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

trade_latency = create_histogram(
    'trading_trade_latency_seconds',
    'Latência de processamento de trades',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

enrich_latency = create_histogram(
    'trading_enrich_latency_seconds',
    'Latência do enrichment de eventos',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

api_latency = create_histogram(
    'trading_api_latency_seconds',
    'Latência de chamadas de API',
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

# Gauges para estado atual
active_trades_buffer = create_gauge(
    'trading_active_trades_buffer',
    'Número de trades no buffer'
)

active_connections = create_gauge(
    'trading_active_connections',
    'Número de conexões ativas'
)

memory_usage_mb = create_gauge(
    'trading_memory_usage_mb',
    'Uso de memória em MB'
)

last_trade_timestamp = create_gauge(
    'trading_last_trade_timestamp_seconds',
    'Timestamp do último trade processado'
)

# ----------------------------------------------------------------------
# Decoradores de instrumentação
# ----------------------------------------------------------------------

def track_latency(metric_name: str):
    """
    Decorador para medir latência de funções síncronas.
    
    Args:
        metric_name: Nome da métrica a ser atualizada
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if metric_name == 'processing':
                    processing_latency.observe(elapsed)
                elif metric_name == 'trade':
                    trade_latency.observe(elapsed)
                elif metric_name == 'enrich':
                    enrich_latency.observe(elapsed)
                elif metric_name == 'api':
                    api_latency.observe(elapsed)
        return wrapper
    return decorator


def track_async_latency(metric_name: str):
    """
    Decorador para medir latência de coroutines.
    
    Args:
        metric_name: Nome da métrica a ser atualizada
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                if metric_name == 'processing':
                    processing_latency.observe(elapsed)
                elif metric_name == 'trade':
                    trade_latency.observe(elapsed)
                elif metric_name == 'enrich':
                    enrich_latency.observe(elapsed)
                elif metric_name == 'api':
                    api_latency.observe(elapsed)
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# Funções de registro de métricas
# ----------------------------------------------------------------------

def record_coroutine_warning(context: Optional[Dict[str, Any]] = None):
    """Registra um warning de corotina"""
    coroutine_warnings.inc()
    logger.warning(f"⚠️ Coroutine warning registrado{': ' + str(context) if context else ''}")


def record_trade_late(latency_ms: float):
    """Registra um trade atrasado"""
    trades_late.inc()
    logger.warning(f"🕐 Trade atrasado: {latency_ms}ms")


def record_timeout(timeout_type: str = "general"):
    """Registra um timeout"""
    timeouts_occurred.inc()
    logger.warning(f"⏱️ Timeout registrado: {timeout_type}")


def record_fred_fallback(reason: Optional[str] = None):
    """Registra uso de fallback FRED"""
    fred_fallback_used.inc()
    reason_str = f" ({reason})" if reason else ""
    logger.warning(f"🔄 FRED fallback usado{reason_str}")


def record_data_correction(correction_type: str, details: Optional[Dict] = None):
    """Registra uma correção de dados"""
    data_corrections.labels(type=correction_type).inc()
    logger.info(f"📊 Correção de dados: {correction_type}")


def record_parse_error(data_type: str, error: Optional[str] = None):
    """Registra um erro de parsing"""
    parse_errors.labels(type=data_type).inc()
    logger.error(f"❌ Erro de parsing: {data_type}{' - ' + error if error else ''}")


def record_enrich_error(event_type: str, error: Optional[str] = None):
    """Registra um erro de enrichment"""
    enrich_errors.labels(type=event_type).inc()
    logger.error(f"❌ Erro de enrichment: {event_type}{' - ' + error if error else ''}")


def record_connection_loss(connection_type: str = "websocket"):
    """Registra uma perda de conexão"""
    connection_losses.labels(type=connection_type).inc()
    logger.warning(f"🔌 Perda de conexão: {connection_type}")


def update_active_trades_buffer(count: int):
    """Atualiza o número de trades no buffer"""
    active_trades_buffer.set(count)


def update_active_connections(count: int):
    """Atualiza o número de conexões ativas"""
    active_connections.set(count)


def update_memory_usage(mb: float):
    """Atualiza o uso de memória"""
    memory_usage_mb.set(mb)


def update_last_trade_timestamp():
    """Atualiza o timestamp do último trade"""
    last_trade_timestamp.set(time.time())


# ----------------------------------------------------------------------
# Context Manager para medição de latência
# ----------------------------------------------------------------------

class LatencyTracker:
    """Context manager para medir latência de operações"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            if self.metric_name == 'processing':
                processing_latency.observe(elapsed)
            elif self.metric_name == 'trade':
                trade_latency.observe(elapsed)
            elif self.metric_name == 'enrich':
                enrich_latency.observe(elapsed)
            elif self.metric_name == 'api':
                api_latency.observe(elapsed)
        return False


class AsyncLatencyTracker:
    """Context manager async para medir latência de operações"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            if self.metric_name == 'processing':
                processing_latency.observe(elapsed)
            elif self.metric_name == 'trade':
                trade_latency.observe(elapsed)
            elif self.metric_name == 'enrich':
                enrich_latency.observe(elapsed)
            elif self.metric_name == 'api':
                api_latency.observe(elapsed)
        return False
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            if self.metric_name == 'processing':
                processing_latency.observe(elapsed)
            elif self.metric_name == 'trade':
                trade_latency.observe(elapsed)
            elif self.metric_name == 'enrich':
                enrich_latency.observe(elapsed)
            elif self.metric_name == 'api':
                api_latency.observe(elapsed)
        return False


def track_operation(metric_name: str):
    """Retorna o tracker apropriado para o contexto"""
    if hasattr(asyncio, 'get_running_loop') and asyncio.get_running_loop().is_running():
        return AsyncLatencyTracker(metric_name)
    return LatencyTracker(metric_name)


# ----------------------------------------------------------------------
# Coletor de métricas com buffer
# ----------------------------------------------------------------------

class MetricsCollector:
    """
    Coletor de métricas com buffer para análise de tendências.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer: Dict[str, deque] = {
            'trade_latency_ms': deque(maxlen=buffer_size),
            'enrich_latency_ms': deque(maxlen=buffer_size),
            'processing_latency_ms': deque(maxlen=buffer_size),
            'trades_late_count': deque(maxlen=buffer_size),
            'timeout_count': deque(maxlen=buffer_size),
            'correction_count': deque(maxlen=buffer_size)
        }
        self._start_time = time.time()
    
    def record_trade_latency(self, latency_ms: float):
        """Registra latência de trade"""
        self.metrics_buffer['trade_latency_ms'].append({
            'value': latency_ms,
            'timestamp': time.time()
        })
    
    def record_enrich_latency(self, latency_ms: float):
        """Registra latência de enrichment"""
        self.metrics_buffer['enrich_latency_ms'].append({
            'value': latency_ms,
            'timestamp': time.time()
        })
    
    def record_processing_latency(self, latency_ms: float):
        """Registra latência de processamento"""
        self.metrics_buffer['processing_latency_ms'].append({
            'value': latency_ms,
            'timestamp': time.time()
        })
    
    def record_trade_late_event(self):
        """Registra evento de trade atrasado"""
        self.metrics_buffer['trades_late_count'].append({
            'value': 1,
            'timestamp': time.time()
        })
    
    def record_timeout_event(self):
        """Registra evento de timeout"""
        self.metrics_buffer['timeout_count'].append({
            'value': 1,
            'timestamp': time.time()
        })
    
    def record_correction_event(self):
        """Registra evento de correção"""
        self.metrics_buffer['correction_count'].append({
            'value': 1,
            'timestamp': time.time()
        })
    
    def get_statistics(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Retorna estatísticas das métricas em uma janela de tempo"""
        now = time.time()
        cutoff = now - window_seconds
        
        stats = {
            'uptime_seconds': now - self._start_time,
            'windows_seconds': window_seconds
        }
        
        for metric_name, buffer in self.metrics_buffer.items():
            values = [entry['value'] for entry in buffer if entry['timestamp'] >= cutoff]
            if values:
                stats[metric_name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
            else:
                stats[metric_name] = {'count': 0, 'sum': 0, 'avg': 0, 'min': 0, 'max': 0}
        
        return stats
    
    def get_rate(self, metric_name: str, window_seconds: int = 60) -> float:
        """Calcula a taxa de eventos por segundo em uma janela"""
        if metric_name not in self.metrics_buffer:
            return 0.0
        
        buffer = self.metrics_buffer[metric_name]
        now = time.time()
        cutoff = now - window_seconds
        
        count = sum(1 for entry in buffer if entry['timestamp'] >= cutoff)
        return count / window_seconds if window_seconds > 0 else 0.0


# Singleton do collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(buffer_size: int = 1000) -> MetricsCollector:
    """Retorna a instância singleton do MetricsCollector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(buffer_size)
    return _metrics_collector


def reset_metrics_collector():
    """Reseta o singleton do MetricsCollector"""
    global _metrics_collector
    _metrics_collector = None


# ----------------------------------------------------------------------
# Servidor HTTP para métricas Prometheus
# ----------------------------------------------------------------------

_metrics_server_port = 8000
_metrics_server_started = False


def start_metrics_server(port: int = 8000) -> bool:
    """
    Inicia o servidor HTTP para expor métricas Prometheus.
    
    Args:
        porta: Porta do servidor
        
    Returns:
        True se o servidor foi iniciado com sucesso
    """
    global _metrics_server_port, _metrics_server_started
    
    if _metrics_server_started:
        logger.warning("Servidor de métricas já está em execução")
        return True
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus não disponível, servidor não pode ser iniciado")
        return False
    
    try:
        start_http_server(port)
        _metrics_server_port = port
        _metrics_server_started = True
        logger.info(f"✅ Servidor de métricas Prometheus iniciado na porta {port}")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar servidor de métricas: {e}")
        return False


def stop_metrics_server():
    """Para o servidor de métricas"""
    global _metrics_server_started
    _metrics_server_started = False
    logger.info("Servidor de métricas parado")


# ----------------------------------------------------------------------
# Integração com AlertManager
# ----------------------------------------------------------------------

def integrate_with_alert_manager(alert_manager=None):
    """
    Integra o coletor de métricas com o AlertManager.
    
    Args:
        alert_manager: Instância do AlertManager (opcional)
    """
    try:
        from alert_manager import get_alert_manager, AlertType
        
        if alert_manager is None:
            alert_manager = get_alert_manager()
        
        # Registrar callbacks de alertas baseados em métricas
        def on_trade_late(alert):
            record_trade_late(alert.metrics.get('rate', 0) * 1000)
        
        def on_timeout(alert):
            record_timeout()

        # adicionar callbacks
        alert_manager.add_notification_callback(on_trade_late)
        alert_manager.add_notification_callback(on_timeout)
        
        logger.info("✅ Métricas integradas com AlertManager")
    except ImportError:
        logger.warning("AlertManager não disponível, integração ignorada")


# ----------------------------------------------------------------------
# Funções de conveniência para logging estruturado
# ----------------------------------------------------------------------

def log_operation_start(operation: str, **kwargs):
    """Log do início de uma operação"""
    logger.info(f"▶️ Iniciando {operation}", extra={'operation': operation, **kwargs})


def log_operation_end(operation: str, elapsed_ms: float, success: bool = True, **kwargs):
    """Log do fim de uma operação"""
    status = "✅" if success else "❌"
    logger.info(f"{status} {operation} concluído em {elapsed_ms:.2f}ms", 
                extra={'operation': operation, 'elapsed_ms': elapsed_ms, **kwargs})


def log_operation_error(operation: str, error: Exception, **kwargs):
    """Log de erro em uma operação"""
    logger.error(f"❌ Erro em {operation}: {error}", 
                 extra={'operation': operation, 'error': str(error), **kwargs})


# ----------------------------------------------------------------------
# Exemplo de uso
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- Teste do Metrics Collector ---")
    
    # Testar registro de métricas
    record_coroutine_warning()
    record_trade_late(150)
    record_timeout("api")
    record_fred_fallback("timeout")
    
    # Testar tracker de latência
    with track_operation('trade') as tracker:
        time.sleep(0.01)  # Simular operação
    
    # Testar estatísticas
    collector = get_metrics_collector()
    collector.record_trade_latency(100)
    collector.record_trade_latency(200)
    collector.record_trade_late_event()
    
    stats = collector.get_statistics()
    print(f"\nEstatísticas: {stats}")
    
    # Mostrar valores das métricas
    print(f"\nMétricas registradas:")
    print(f"  coroutine_warnings: {coroutine_warnings._value if hasattr(coroutine_warnings, '_value') else 'N/A'}")
    print(f"  trades_late: {trades_late._value if hasattr(trades_late, '_value') else 'N/A'}")
    print(f"  timeouts: {timeouts_occurred._value if hasattr(timeouts_occurred, '_value') else 'N/A'}")
    print(f"  fred_fallback: {fred_fallback_used._value if hasattr(fred_fallback_used, '_value') else 'N/A'}")
    
    print("\n✅ Metrics Collector funcionando corretamente")
