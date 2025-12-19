# flow_analyzer/prometheus_metrics.py
"""
Métricas Prometheus para FlowAnalyzer.

DEPENDÊNCIA OPCIONAL:
    pip install prometheus-client

Este módulo funciona sem prometheus_client instalado.
Quando não disponível, PrometheusMetrics opera em modo "noop".
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional, TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info

# Tenta importar prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info  # type: ignore[import-untyped]
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    # Dummy classes para type hints quando prometheus não está instalado
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]
    Summary = None  # type: ignore[misc, assignment]
    Info = None  # type: ignore[misc, assignment]


class PrometheusMetrics:
    """
    Coletor de métricas Prometheus para FlowAnalyzer.
    
    Métricas expostas (quando prometheus_client está disponível):
    - flow_analyzer_trades_total: Total de trades processados
    - flow_analyzer_trades_invalid_total: Trades inválidos
    - flow_analyzer_cvd: CVD atual
    - flow_analyzer_processing_seconds: Tempo de processamento
    - flow_analyzer_whale_delta: Delta de whales
    - flow_analyzer_ooo_total: Trades out-of-order
    
    Quando prometheus_client NÃO está instalado, opera em modo "noop"
    (todas as operações são ignoradas silenciosamente).
    
    Example:
        >>> metrics = PrometheusMetrics()
        >>> if metrics.enabled:
        ...     metrics.record_trade(is_valid=True, processing_time=0.001)
        ...     metrics.set_cvd(1.5)
    """
    
    def __init__(self, prefix: str = "flow_analyzer"):
        self._prefix = prefix
        self._enabled = HAS_PROMETHEUS
        
        if not self._enabled:
            # Modo noop
            self.trades_total = None
            self.trades_invalid = None
            self.ooo_total = None
            self.bursts_total = None
            self.cvd = None
            self.whale_delta = None
            self.flow_trades_count = None
            self.memory_bytes = None
            self.processing_time = None
            self.trade_size = None
            self.info = None
            return
        
        # Counters
        self.trades_total = Counter(
            f'{prefix}_trades_total',
            'Total trades processed',
            ['side', 'sector']
        )
        
        self.trades_invalid = Counter(
            f'{prefix}_trades_invalid_total',
            'Invalid trades',
            ['reason']
        )
        
        self.ooo_total = Counter(
            f'{prefix}_ooo_total',
            'Out-of-order trades detected'
        )
        
        self.bursts_total = Counter(
            f'{prefix}_bursts_total',
            'Volume bursts detected'
        )
        
        # Gauges
        self.cvd = Gauge(
            f'{prefix}_cvd',
            'Cumulative Volume Delta'
        )
        
        self.whale_delta = Gauge(
            f'{prefix}_whale_delta',
            'Whale Delta (buy - sell)'
        )
        
        self.flow_trades_count = Gauge(
            f'{prefix}_flow_trades_count',
            'Number of trades in window'
        )
        
        self.memory_bytes = Gauge(
            f'{prefix}_memory_bytes',
            'Memory usage in bytes'
        )
        
        # Histograms
        self.processing_time = Histogram(
            f'{prefix}_processing_seconds',
            'Trade processing time',
            buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0]
        )
        
        self.trade_size = Histogram(
            f'{prefix}_trade_size_btc',
            'Trade size in BTC',
            buckets=[.01, .1, .5, 1, 2, 5, 10, 50, 100]
        )
        
        # Info
        self.info = Info(
            f'{prefix}_info',
            'FlowAnalyzer information'
        )
    
    @property
    def enabled(self) -> bool:
        """Retorna True se prometheus_client está disponível."""
        return self._enabled
    
    def record_trade(
        self,
        is_valid: bool,
        processing_time: float,
        side: Optional[str] = None,
        sector: Optional[str] = None,
        size_btc: Optional[float] = None,
        invalid_reason: Optional[str] = None,
    ) -> None:
        """
        Registra processamento de trade.
        
        Args:
            is_valid: Se o trade foi válido
            processing_time: Tempo de processamento em segundos
            side: Lado do trade (buy/sell)
            sector: Setor (retail/mid/whale)
            size_btc: Tamanho em BTC
            invalid_reason: Razão de invalidação (se inválido)
        """
        if not self._enabled:
            return
        
        self.processing_time.observe(processing_time)
        
        if is_valid:
            self.trades_total.labels(
                side=side or 'unknown',
                sector=sector or 'unknown'
            ).inc()
            
            if size_btc is not None:
                self.trade_size.observe(size_btc)
        else:
            self.trades_invalid.labels(
                reason=invalid_reason or 'unknown'
            ).inc()
    
    def set_cvd(self, value: float) -> None:
        """Atualiza CVD."""
        if self._enabled and self.cvd is not None:
            self.cvd.set(value)
    
    def set_whale_delta(self, value: float) -> None:
        """Atualiza whale delta."""
        if self._enabled and self.whale_delta is not None:
            self.whale_delta.set(value)
    
    def set_flow_trades_count(self, value: int) -> None:
        """Atualiza contagem de trades na janela."""
        if self._enabled and self.flow_trades_count is not None:
            self.flow_trades_count.set(value)
    
    def set_memory_bytes(self, value: int) -> None:
        """Atualiza uso de memória."""
        if self._enabled and self.memory_bytes is not None:
            self.memory_bytes.set(value)
    
    def record_ooo(self) -> None:
        """Registra trade out-of-order."""
        if self._enabled and self.ooo_total is not None:
            self.ooo_total.inc()
    
    def record_burst(self) -> None:
        """Registra burst detectado."""
        if self._enabled and self.bursts_total is not None:
            self.bursts_total.inc()
    
    def set_info(self, version: str, config_version: int) -> None:
        """Define informações do analyzer."""
        if self._enabled and self.info is not None:
            self.info.info({
                'version': version,
                'config_version': str(config_version),
            })


class MetricsCollector:
    """
    Coletor de métricas agnóstico (funciona sem Prometheus).
    
    Mantém métricas internamente e pode exportar para diferentes backends.
    Útil quando prometheus_client não está instalado.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.inc_counter("trades_total", side="buy")
        >>> collector.set_gauge("cvd", 1.5)
        >>> collector.observe_histogram("processing_time", 0.005)
        >>> stats = collector.get_all()
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._start_time = time.time()
    
    def inc_counter(self, name: str, value: int = 1, **labels) -> None:
        """Incrementa contador."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, **labels) -> None:
        """Define gauge."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, **labels) -> None:
        """Adiciona observação ao histograma."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
    
    def _make_key(self, name: str, labels: Dict[str, Any]) -> str:
        """Cria chave única para métrica."""
        if not labels:
            return name
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f'{name}{{{label_str}}}'
    
    def get_all(self) -> Dict[str, Any]:
        """Retorna todas as métricas."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histogram_counts': {k: len(v) for k, v in self._histograms.items()},
            'uptime_sec': time.time() - self._start_time,
        }
    
    def get_histogram_stats(self, name: str, **labels) -> Dict[str, Any]:
        """Retorna estatísticas de um histograma."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {'count': 0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return {
            'count': n,
            'min': sorted_vals[0],
            'max': sorted_vals[-1],
            'avg': sum(sorted_vals) / n,
            'p50': sorted_vals[int(n * 0.5)],
            'p95': sorted_vals[min(int(n * 0.95), n - 1)],
            'p99': sorted_vals[min(int(n * 0.99), n - 1)],
        }
    
    def reset(self) -> None:
        """Reseta todas as métricas."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._start_time = time.time()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def is_prometheus_available() -> bool:
    """Verifica se prometheus_client está disponível."""
    return HAS_PROMETHEUS


def get_default_metrics() -> PrometheusMetrics:
    """Retorna instância padrão de métricas."""
    return PrometheusMetrics()


def get_fallback_collector() -> MetricsCollector:
    """Retorna coletor fallback quando Prometheus não está disponível."""
    return MetricsCollector()