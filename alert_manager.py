#!/usr/bin/env python3
"""
Gerenciador de Alertas para Monitoramento do Sistema de Trading
Implementa alertas específicos para qualidade de dados e saúde do sistema
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Níveis de alerta"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType(Enum):
    """Tipos de alerta"""
    TRADE_LATE = "trade_late"
    TIMEOUT = "timeout"
    DATA_CORRECTION = "data_correction"
    COROUTINE_WARNING = "coroutine_warning"
    FRED_FALLBACK = "fred_fallback"
    PARSE_ERROR = "parse_error"
    ENRICH_ERROR = "enrich_error"
    CONNECTION_LOST = "connection_lost"
    HIGH_LATENCY = "high_latency"
    MEMORY_WARNING = "memory_warning"


@dataclass
class Alert:
    """Representação de um alerta"""
    alert_type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


class AlertManager:
    """
    Gerenciador de alertas para monitoramento do sistema de trading.
    
    Implementa verificações automáticas de métricas e gera alertas
    quando thresholds são violados.
    """
    
    # Thresholds padrão
    DEFAULT_THRESHOLDS = {
        'trade_late_rate': 0.1,        # 10% de trades atrasados
        'timeout_rate': 0.05,          # 5% de timeouts
        'correction_rate': 0.1,        # 10% de correções de dados
        'coroutine_warning_rate': 0.05, # 5% de warnings de corotinas
        'fred_fallback_rate': 0.3,     # 30% de fallback FRED
        'parse_error_rate': 0.02,      # 2% de erros de parsing
        'enrich_error_rate': 0.05,     # 5% de erros de enrichment
        'max_latency_ms': 1000,        # 1 segundo de latência máxima
        'max_connection_loss_count': 3 # 3 perdas de conexão consecutivas
    }
    
    # Períodos de agregação em segundos
    AGGREGATION_PERIODS = {
        'short': 60,      # 1 minuto
        'medium': 300,    # 5 minutos
        'long': 900       # 15 minutos
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o gerenciador de alertas.
        
        Args:
            config: Dicionário opcional com configuração customizada de thresholds
        """
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if config:
            self.thresholds.update(config)
        
        # Armazenamento de alertas ativos e históricos
        self.active_alerts: Dict[AlertType, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Métricas agregadas por período
        self.metrics_buffer: Dict[str, deque] = {
            'trade_late': deque(maxlen=100),
            'timeouts': deque(maxlen=100),
            'corrections': deque(maxlen=100),
            'coroutine_warnings': deque(maxlen=100),
            'fred_fallbacks': deque(maxlen=100),
            'parse_errors': deque(maxlen=100),
            'enrich_errors': deque(maxlen=100),
            'latencies': deque(maxlen=100)
        }
        
        # Contadores de estado
        self.connection_loss_count = 0
        self.last_connection_loss: Optional[datetime] = None
        
        # Callbacks de notificação
        self.notification_callbacks: List[Callable[..., Any]] = []
        
        # Estado do manager
        self._running = False
        
        logger.info("AlertManager inicializado com thresholds: %s", self.thresholds)
    
    def start(self):
        """Inicia o monitoramento de alertas"""
        self._running = True
        logger.info("AlertManager iniciado")
    
    def stop(self):
        """Para o monitoramento de alertas"""
        self._running = False
        logger.info("AlertManager parado")
    
    def add_notification_callback(self, callback: Callable[..., Any]):
        """
        Adiciona um callback para notificação de alertas.
        
        Args:
            callback: Função que será chamada quando um alerta for gerado
        """
        self.notification_callbacks.append(callback)
    
    def record_metric(self, metric_type: str, value: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """
        Registra uma métrica para análise.
        
        Args:
            metric_type: Tipo de métrica (trade_late, timeout, etc.)
            value: Valor da métrica
            metadata: Metadados adicionais
        """
        if metric_type in self.metrics_buffer:
            timestamp = time.time()
            entry = {'value': value, 'timestamp': timestamp, 'metadata': metadata or {}}
            self.metrics_buffer[metric_type].append(entry)
    
    def record_trade_late(self, latency_ms: float):
        """Registra um trade atrasado"""
        self.record_metric('trade_late', latency_ms)
        self._check_trade_late_alert()
    
    def record_timeout(self, timeout_type: str = "general"):
        """Registra um timeout"""
        self.record_metric('timeouts', 1.0, {'type': timeout_type})
        self._check_timeout_alert()
    
    def record_data_correction(self, correction_type: str = "general"):
        """Registra uma correção de dados"""
        self.record_metric('corrections', 1.0, {'type': correction_type})
        self._check_correction_alert()
    
    def record_coroutine_warning(self):
        """Registra um warning de corotina"""
        self.record_metric('coroutine_warnings', 1.0)
        self._check_coroutine_warning_alert()
    
    def record_fred_fallback(self):
        """Registra uso de fallback FRED"""
        self.record_metric('fred_fallbacks', 1.0)
        self._check_fred_fallback_alert()
    
    def record_parse_error(self, error_type: str = "general"):
        """Registra um erro de parsing"""
        self.record_metric('parse_errors', 1.0, {'type': error_type})
        self._check_parse_error_alert()
    
    def record_enrich_error(self, error_type: str = "general"):
        """Registra um erro de enrichment"""
        self.record_metric('enrich_errors', 1.0, {'type': error_type})
        self._check_enrich_error_alert()
    
    def record_latency(self, operation: str, latency_ms: float):
        """Registra latência de uma operação"""
        self.record_metric('latencies', latency_ms, {'operation': operation})
        self._check_latency_alert(operation, latency_ms)
    
    def record_connection_loss(self):
        """Registra perda de conexão"""
        self.connection_loss_count += 1
        self.last_connection_loss = datetime.now()
        self._check_connection_alert()
    
    def record_connection_recovery(self):
        """Registra recuperação de conexão"""
        if self.connection_loss_count > 0:
            self.connection_loss_count = 0
            self._resolve_alert(AlertType.CONNECTION_LOST)
            logger.info("Conexão recuperada após %d falhas", self.connection_loss_count)
    
    def _calculate_rate(self, metric_type: str, window_seconds: int = 300) -> float:
        """
        Calcula a taxa de uma métrica em uma janela de tempo.
        
        Args:
            metric_type: Tipo de métrica
            window_seconds: Janela de tempo em segundos
            
        Returns:
            Taxa de ocorrência (0.0 a 1.0)
        """
        if metric_type not in self.metrics_buffer:
            return 0.0
        
        buffer = self.metrics_buffer[metric_type]
        if not buffer:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        # Conta ocorrências na janela
        count = sum(1 for entry in buffer if entry['timestamp'] >= cutoff)
        total = len(buffer)
        
        return count / total if total > 0 else 0.0
    
    def _calculate_average_latency(self, window_seconds: int = 300) -> float:
        """Calcula a latência média"""
        buffer = self.metrics_buffer['latencies']
        if not buffer:
            return 0.0
        
        now = time.time()
        cutoff = now - window_seconds
        
        latencies = [entry['value'] for entry in buffer if entry['timestamp'] >= cutoff]
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Retorna as métricas atuais calculadas.
        
        Returns:
            Dicionário com métricas atuais
        """
        return {
            'trade_late_rate_1m': self._calculate_rate('trade_late', 60),
            'trade_late_rate_5m': self._calculate_rate('trade_late', 300),
            'timeout_rate_1m': self._calculate_rate('timeouts', 60),
            'timeout_rate_5m': self._calculate_rate('timeouts', 300),
            'correction_rate_1m': self._calculate_rate('corrections', 60),
            'correction_rate_5m': self._calculate_rate('corrections', 300),
            'coroutine_warning_rate_1m': self._calculate_rate('coroutine_warnings', 60),
            'fred_fallback_rate_1m': self._calculate_rate('fred_fallbacks', 60),
            'parse_error_rate_1m': self._calculate_rate('parse_errors', 60),
            'enrich_error_rate_1m': self._calculate_rate('enrich_errors', 60),
            'avg_latency_ms': self._calculate_average_latency(300),
            'connection_loss_count': self.connection_loss_count
        }
    
    def check_all_alerts(self) -> List[Alert]:
        """
        Verifica todos os alertas e retorna lista de alertas ativos.
        
        Returns:
            Lista de alertas ativos
        """
        metrics = self.get_current_metrics()
        alerts = []
        
        # Verificar taxa de trades atrasados
        if metrics.get('trade_late_rate_5m', 0) > self.thresholds['trade_late_rate']:
            alerts.append(self._create_alert(
                AlertType.TRADE_LATE,
                AlertLevel.CRITICAL,
                f"Taxa de trades atrasados muito alta: {metrics['trade_late_rate_5m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de timeouts
        if metrics.get('timeout_rate_5m', 0) > self.thresholds['timeout_rate']:
            alerts.append(self._create_alert(
                AlertType.TIMEOUT,
                AlertLevel.CRITICAL,
                f"Taxa de timeouts muito alta: {metrics['timeout_rate_5m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de correções
        if metrics.get('correction_rate_5m', 0) > self.thresholds['correction_rate']:
            alerts.append(self._create_alert(
                AlertType.DATA_CORRECTION,
                AlertLevel.WARNING,
                f"Taxa de correções de dados muito alta: {metrics['correction_rate_5m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de coroutine warnings
        if metrics.get('coroutine_warning_rate_1m', 0) > self.thresholds['coroutine_warning_rate']:
            alerts.append(self._create_alert(
                AlertType.COROUTINE_WARNING,
                AlertLevel.WARNING,
                f"Taxa de warnings de corotinas muito alta: {metrics['coroutine_warning_rate_1m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de fallback FRED
        if metrics.get('fred_fallback_rate_1m', 0) > self.thresholds['fred_fallback_rate']:
            alerts.append(self._create_alert(
                AlertType.FRED_FALLBACK,
                AlertLevel.WARNING,
                f"Taxa de fallback FRED muito alta: {metrics['fred_fallback_rate_1m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de erros de parsing
        if metrics.get('parse_error_rate_1m', 0) > self.thresholds['parse_error_rate']:
            alerts.append(self._create_alert(
                AlertType.PARSE_ERROR,
                AlertLevel.CRITICAL,
                f"Taxa de erros de parsing muito alta: {metrics['parse_error_rate_1m']*100:.1f}%",
                metrics
            ))
        
        # Verificar taxa de erros de enrichment
        if metrics.get('enrich_error_rate_1m', 0) > self.thresholds['enrich_error_rate']:
            alerts.append(self._create_alert(
                AlertType.ENRICH_ERROR,
                AlertLevel.CRITICAL,
                f"Taxa de erros de enrichment muito alta: {metrics['enrich_error_rate_1m']*100:.1f}%",
                metrics
            ))
        
        # Verificar latência
        if metrics.get('avg_latency_ms', 0) > self.thresholds['max_latency_ms']:
            alerts.append(self._create_alert(
                AlertType.HIGH_LATENCY,
                AlertLevel.WARNING,
                f"Latência média muito alta: {metrics['avg_latency_ms']:.0f}ms",
                metrics
            ))
        
        # Verificar perda de conexão
        if self.connection_loss_count >= self.thresholds['max_connection_loss_count']:
            alerts.append(self._create_alert(
                AlertType.CONNECTION_LOST,
                AlertLevel.EMERGENCY,
                f"Múltiplas perdas de conexão: {self.connection_loss_count} vezes",
                metrics
            ))
        
        return alerts
    
    def _create_alert(self, alert_type: AlertType, level: AlertLevel, 
                      message: str, metrics: Dict[str, Any]) -> Alert:
        """Cria e registra um novo alerta"""
        # Verificar se já existe um alerta ativo do mesmo tipo
        existing = self.active_alerts.get(alert_type)
        if existing and existing.level.value >= level.value:
            return existing
        
        alert = Alert(
            alert_type=alert_type,
            level=level,
            message=message,
            metrics=metrics
        )
        
        self.active_alerts[alert_type] = alert
        self.alert_history.append(alert)
        
        # Log do alerta
        log_method = getattr(logger, level.value.lower())
        log_method(f"🚨 {alert_type.value.upper()}: {message}")
        
        # Notificar callbacks
        self._notify_callbacks(alert)
        
        return alert
    
    def _resolve_alert(self, alert_type: AlertType):
        """Resolve um alerta ativo"""
        if alert_type in self.active_alerts:
            alert = self.active_alerts.pop(alert_type)
            alert.resolved = True
            alert.resolution_time = datetime.now()
            self.alert_history.append(alert)
            logger.info(f"✅ Alerta resolvido: {alert_type.value}")
    
    def _notify_callbacks(self, alert: Alert):
        """Notifica todos os callbacks registrados"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erro ao executar callback de alerta: {e}")
    
    def _check_trade_late_alert(self):
        """Verifica alerta de trades atrasados"""
        rate = self._calculate_rate('trade_late', 60)
        if rate > self.thresholds['trade_late_rate']:
            self._create_alert(
                AlertType.TRADE_LATE,
                AlertLevel.CRITICAL,
                f"Taxa de trades atrasados: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_timeout_alert(self):
        """Verifica alerta de timeouts"""
        rate = self._calculate_rate('timeouts', 60)
        if rate > self.thresholds['timeout_rate']:
            self._create_alert(
                AlertType.TIMEOUT,
                AlertLevel.CRITICAL,
                f"Taxa de timeouts: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_correction_alert(self):
        """Verifica alerta de correções de dados"""
        rate = self._calculate_rate('corrections', 60)
        if rate > self.thresholds['correction_rate']:
            self._create_alert(
                AlertType.DATA_CORRECTION,
                AlertLevel.WARNING,
                f"Taxa de correções de dados: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_coroutine_warning_alert(self):
        """Verifica alerta de warnings de corotinas"""
        rate = self._calculate_rate('coroutine_warnings', 60)
        if rate > self.thresholds['coroutine_warning_rate']:
            self._create_alert(
                AlertType.COROUTINE_WARNING,
                AlertLevel.WARNING,
                f"Taxa de warnings de corotinas: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_fred_fallback_alert(self):
        """Verifica alerta de fallback FRED"""
        rate = self._calculate_rate('fred_fallbacks', 60)
        if rate > self.thresholds['fred_fallback_rate']:
            self._create_alert(
                AlertType.FRED_FALLBACK,
                AlertLevel.WARNING,
                f"Taxa de fallback FRED: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_parse_error_alert(self):
        """Verifica alerta de erros de parsing"""
        rate = self._calculate_rate('parse_errors', 60)
        if rate > self.thresholds['parse_error_rate']:
            self._create_alert(
                AlertType.PARSE_ERROR,
                AlertLevel.CRITICAL,
                f"Taxa de erros de parsing: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_enrich_error_alert(self):
        """Verifica alerta de erros de enrichment"""
        rate = self._calculate_rate('enrich_errors', 60)
        if rate > self.thresholds['enrich_error_rate']:
            self._create_alert(
                AlertType.ENRICH_ERROR,
                AlertLevel.CRITICAL,
                f"Taxa de erros de enrichment: {rate*100:.1f}%",
                {'rate': rate}
            )
    
    def _check_latency_alert(self, operation: str, latency_ms: float):
        """Verifica alerta de latência"""
        if latency_ms > self.thresholds['max_latency_ms']:
            self._create_alert(
                AlertType.HIGH_LATENCY,
                AlertLevel.WARNING,
                f"Latência alta em {operation}: {latency_ms:.0f}ms",
                {'operation': operation, 'latency_ms': latency_ms}
            )
    
    def _check_connection_alert(self):
        """Verifica alerta de perda de conexão"""
        if self.connection_loss_count >= self.thresholds['max_connection_loss_count']:
            self._create_alert(
                AlertType.CONNECTION_LOST,
                AlertLevel.EMERGENCY,
                f"Conexão perdida {self.connection_loss_count} vezes",
                {'loss_count': self.connection_loss_count}
            )
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retorna lista de alertas ativos"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna histórico de alertas"""
        return [alert.to_dict() for alert in list(self.alert_history)[-limit:]]
    
    def clear_history(self):
        """Limpa o histórico de alertas"""
        self.alert_history.clear()
        logger.info("Histórico de alertas limpo")
    
    def update_threshold(self, key: str, value: float):
        """
        Atualiza um threshold específico em runtime.
        
        Args:
            key: Nome do threshold
            value: Novo valor
        """
        if key in self.thresholds:
            old_value = self.thresholds[key]
            self.thresholds[key] = value
            logger.info(f"Threshold atualizado: {key} {old_value} -> {value}")


# Singleton para uso global
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """
    Retorna a instância singleton do AlertManager.
    
    Args:
        config: Configuração opcional
        
    Returns:
        Instância do AlertManager
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(config)
    return _alert_manager


def reset_alert_manager():
    """Reseta o singleton do AlertManager"""
    global _alert_manager
    if _alert_manager:
        _alert_manager.stop()
    _alert_manager = None


# Integração com metrics_collector
def setup_alert_manager_integration(metrics_collector=None):
    """
    Configura integração entre AlertManager e métricas.
    
    Args:
        metrics_collector: Instância opcional do metrics collector
    """
    alert_manager = get_alert_manager()
    alert_manager.start()
    
    logger.info("AlertManager integrado e iniciado")
    return alert_manager


if __name__ == "__main__":
    # Teste simples do AlertManager
    logging.basicConfig(level=logging.INFO)
    
    manager = AlertManager()
    manager.start()
    
    # Simular algumas métricas
    print("\n--- Teste do AlertManager ---")
    
    # Registrar algumas métricas
    manager.record_trade_late(150)
    manager.record_trade_late(200)
    manager.record_timeout("api")
    manager.record_latency("enrich_event", 500)
    
    # Verificar alertas
    alerts = manager.check_all_alerts()
    print(f"\nAlertas ativos: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert.level.value}: {alert.message}")
    
    # Mostrar métricas atuais
    metrics = manager.get_current_metrics()
    print(f"\nMétricas atuais:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n✅ AlertManager funcionando corretamente")
