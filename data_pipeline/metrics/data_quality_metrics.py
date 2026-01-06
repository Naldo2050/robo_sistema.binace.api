# -*- coding: utf-8 -*-
"""
Métricas de Qualidade de Dados v1.0.0

Sistema de instrumentação para monitorar:
- Eventos válidos, corrigidos e descartados
- Taxa de correção por tipo
- Latência de processamento (p50, p95)

Emite logs estruturados periódicos e alertas.
Projetado para ser facilmente estendido para Prometheus.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


@dataclass
class QualityStats:
    """Estatísticas de qualidade de dados."""
    total_events: int = 0
    valid_events: int = 0
    corrected_events: int = 0
    discarded_events: int = 0
    corrections_by_type: Dict[str, int] = field(default_factory=dict)
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))


class DataQualityMetrics:
    """
    Sistema de métricas de qualidade de dados.
    
    Thread-safe, leve e extensível para Prometheus.
    
    Uso:
        metrics = DataQualityMetrics()
        
        # Registrar evento
        metrics.record_event(
            status="valid",  # "valid", "corrected", "discarded"
            correction_types=["recalculated_delta"],
            latency_ms=1.5
        )
        
        # Obter estatísticas
        stats = metrics.get_stats()
        
        # Log periódico (chamado automaticamente a cada N eventos)
        metrics.emit_periodic_log()
    """
    
    # Configurações de alerta
    CORRECTION_RATE_WARNING = 5.0   # %
    CORRECTION_RATE_ERROR = 10.0    # %
    DISCARD_RATE_WARNING = 2.0      # %
    LATENCY_P95_WARNING_MS = 10.0   # ms
    
    # Intervalo de log periódico
    LOG_INTERVAL_EVENTS = 100
    LOG_INTERVAL_SECONDS = 60
    
    def __init__(self, logger_name: str = "DataQualityMetrics"):
        """
        Inicializa sistema de métricas.
        
        Args:
            logger_name: Nome do logger para identificação
        """
        self.logger = logging.getLogger(logger_name)
        self._lock = Lock()
        self._stats = QualityStats()
        self._last_log_time = time.time()
        self._last_log_events = 0
        
    def record_event(
        self,
        status: str,
        correction_types: Optional[List[str]] = None,
        latency_ms: float = 0.0
    ) -> None:
        """
        Registra um evento processado.
        
        Args:
            status: "valid", "corrected" ou "discarded"
            correction_types: Lista de tipos de correção aplicados
            latency_ms: Latência de processamento em milissegundos
        
        Thread-safe e não-bloqueante.
        """
        with self._lock:
            self._stats.total_events += 1
            
            if status == "valid":
                self._stats.valid_events += 1
            elif status == "corrected":
                self._stats.corrected_events += 1
                self._stats.valid_events += 1  # Corrigido também é válido
            elif status == "discarded":
                self._stats.discarded_events += 1
            
            # Registrar tipos de correção
            if correction_types:
                for ctype in correction_types:
                    self._stats.corrections_by_type[ctype] = \
                        self._stats.corrections_by_type.get(ctype, 0) + 1
            
            # Registrar latência
            if latency_ms > 0:
                self._stats.latencies_ms.append(latency_ms)
            
            # Verificar se deve emitir log periódico
            events_since_log = self._stats.total_events - self._last_log_events
            time_since_log = time.time() - self._last_log_time
            
            if events_since_log >= self.LOG_INTERVAL_EVENTS or \
               time_since_log >= self.LOG_INTERVAL_SECONDS:
                self._emit_log_unlocked()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas atuais.
        
        Returns:
            Dicionário com todas as métricas.
        """
        with self._lock:
            return self._compute_stats()
    
    def _compute_stats(self) -> Dict[str, Any]:
        """Calcula estatísticas (não thread-safe, usar com lock)."""
        total = self._stats.total_events
        
        # Taxas
        correction_rate = (self._stats.corrected_events / total * 100) if total > 0 else 0.0
        discard_rate = (self._stats.discarded_events / total * 100) if total > 0 else 0.0
        
        # Latência
        latencies = list(self._stats.latencies_ms)
        if latencies:
            latencies_sorted = sorted(latencies)
            p50_idx = int(len(latencies_sorted) * 0.50)
            p95_idx = int(len(latencies_sorted) * 0.95)
            latency_p50 = latencies_sorted[p50_idx] if p50_idx < len(latencies_sorted) else 0.0
            latency_p95 = latencies_sorted[p95_idx] if p95_idx < len(latencies_sorted) else 0.0
            latency_max = max(latencies_sorted)
            latency_avg = sum(latencies) / len(latencies)
        else:
            latency_p50 = latency_p95 = latency_max = latency_avg = 0.0
        
        # Top correções
        top_corrections = dict(
            sorted(
                self._stats.corrections_by_type.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_events": total,
                "valid_events": self._stats.valid_events,
                "corrected_events": self._stats.corrected_events,
                "discarded_events": self._stats.discarded_events,
            },
            "rates": {
                "correction_rate_pct": round(correction_rate, 2),
                "discard_rate_pct": round(discard_rate, 2),
            },
            "corrections_by_type": top_corrections,
            "latency_ms": {
                "p50": round(latency_p50, 3),
                "p95": round(latency_p95, 3),
                "max": round(latency_max, 3),
                "avg": round(latency_avg, 3),
            },
            "alerts": self._compute_alerts(correction_rate, discard_rate, latency_p95),
        }
    
    def _compute_alerts(
        self,
        correction_rate: float,
        discard_rate: float,
        latency_p95: float
    ) -> List[Dict[str, str]]:
        """Computa alertas baseados nos thresholds."""
        alerts = []
        
        if correction_rate >= self.CORRECTION_RATE_ERROR:
            alerts.append({
                "level": "ERROR",
                "type": "HIGH_CORRECTION_RATE",
                "message": f"Taxa de correção crítica: {correction_rate:.1f}% (limite: {self.CORRECTION_RATE_ERROR}%)"
            })
        elif correction_rate >= self.CORRECTION_RATE_WARNING:
            alerts.append({
                "level": "WARNING",
                "type": "HIGH_CORRECTION_RATE",
                "message": f"Taxa de correção elevada: {correction_rate:.1f}% (limite: {self.CORRECTION_RATE_WARNING}%)"
            })
        
        if discard_rate >= self.DISCARD_RATE_WARNING:
            alerts.append({
                "level": "WARNING",
                "type": "HIGH_DISCARD_RATE",
                "message": f"Taxa de descarte elevada: {discard_rate:.1f}% (limite: {self.DISCARD_RATE_WARNING}%)"
            })
        
        if latency_p95 >= self.LATENCY_P95_WARNING_MS:
            alerts.append({
                "level": "WARNING",
                "type": "HIGH_LATENCY",
                "message": f"Latência P95 alta: {latency_p95:.1f}ms (limite: {self.LATENCY_P95_WARNING_MS}ms)"
            })
        
        return alerts
    
    def emit_periodic_log(self) -> None:
        """Emite log periódico de métricas."""
        with self._lock:
            self._emit_log_unlocked()
    
    def _emit_log_unlocked(self) -> None:
        """Emite log (não thread-safe, usar com lock)."""
        stats = self._compute_stats()
        
        # Log estruturado JSON
        log_entry = {
            "type": "DATA_QUALITY_METRICS",
            **stats
        }
        
        # Emitir log principal
        self.logger.info(f"📊 {json.dumps(log_entry, ensure_ascii=False)}")
        
        # Emitir alertas separadamente
        for alert in stats["alerts"]:
            if alert["level"] == "ERROR":
                self.logger.error(f"🚨 [DATA_QUALITY_ALERT] {alert['message']}")
            else:
                self.logger.warning(f"⚠️ [DATA_QUALITY_ALERT] {alert['message']}")
        
        # Atualizar contadores de log
        self._last_log_time = time.time()
        self._last_log_events = self._stats.total_events
    
    def reset(self) -> None:
        """Reseta todas as métricas."""
        with self._lock:
            self._stats = QualityStats()
            self._last_log_time = time.time()
            self._last_log_events = 0
    
    # =========================================
    # Métodos para extensão futura (Prometheus)
    # =========================================
    
    def get_prometheus_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas no formato compatível com Prometheus.
        
        Para uso futuro com prometheus_client:
        
        ```python
        from prometheus_client import Counter, Histogram
        
        events_total = Counter('data_events_total', 'Total de eventos', ['status'])
        latency_histogram = Histogram('data_latency_seconds', 'Latência de processamento')
        ```
        """
        stats = self.get_stats()
        return {
            "data_events_total{status=\"valid\"}": stats["summary"]["valid_events"],
            "data_events_total{status=\"corrected\"}": stats["summary"]["corrected_events"],
            "data_events_total{status=\"discarded\"}": stats["summary"]["discarded_events"],
            "data_correction_rate": stats["rates"]["correction_rate_pct"],
            "data_latency_p50_ms": stats["latency_ms"]["p50"],
            "data_latency_p95_ms": stats["latency_ms"]["p95"],
        }


# =========================================
# Instância global (singleton)
# =========================================

_global_metrics: Optional[DataQualityMetrics] = None
_global_lock = Lock()


def get_quality_metrics() -> DataQualityMetrics:
    """
    Retorna instância global de métricas.
    
    Uso:
        from data_pipeline.metrics.data_quality_metrics import get_quality_metrics
        
        metrics = get_quality_metrics()
        metrics.record_event("valid", latency_ms=1.5)
    """
    global _global_metrics
    
    if _global_metrics is None:
        with _global_lock:
            if _global_metrics is None:
                _global_metrics = DataQualityMetrics()
    
    return _global_metrics
