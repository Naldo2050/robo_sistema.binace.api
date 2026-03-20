# flow_analyzer/metrics.py
"""
Métricas e observabilidade do FlowAnalyzer.

Inclui:
- PerformanceMonitor: Tracking de latência com percentis
- CircuitBreaker: Proteção contra falhas em cascata
- HealthChecker: Verificação de saúde
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Any, Optional, List
from enum import Enum

from .constants import (
    PERF_MONITOR_WINDOW_SIZE,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIME_MS,
    HIGH_LATENCY_P99_MS,
    MEMORY_USAGE_WARNING_RATIO,
)
from .utils import lazy_log, get_current_time_ms


# ==============================================================================
# PERFORMANCE MONITOR
# ==============================================================================

class PerformanceMonitor:
    """
    Monitor de performance com percentis.
    
    Thread-safe para uso em ambiente multithread.
    Mantém janela deslizante de tempos de processamento.
    
    Example:
        >>> monitor = PerformanceMonitor(window_size=1000)
        >>> monitor.record(5.5)  # 5.5ms
        >>> monitor.record(3.2)
        >>> stats = monitor.get_stats()
        >>> print(f"P99: {stats['p99']:.1f}ms")
    """
    
    def __init__(self, window_size: int = PERF_MONITOR_WINDOW_SIZE):
        self.window_size = window_size
        self.times: deque = deque(maxlen=window_size)
        self._lock = Lock()
        self._total_recorded = 0
    
    def record(self, duration_ms: float) -> None:
        """
        Registra tempo de processamento.
        
        Args:
            duration_ms: Duração em milliseconds
        """
        with self._lock:
            self.times.append(duration_ms)
            self._total_recorded += 1
    
    def get_stats(self) -> Dict[str, float]:
        """
        Retorna estatísticas de performance.
        
        Returns:
            Dict com p50, p95, p99, max, avg, min, count, total_recorded
        """
        with self._lock:
            if not self.times:
                return {
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'avg': 0.0,
                    'count': 0,
                    'total_recorded': self._total_recorded,
                }
            
            sorted_times = sorted(self.times)
            n = len(sorted_times)
            
            return {
                'p50': sorted_times[int(n * 0.5)],
                'p95': sorted_times[min(int(n * 0.95), n - 1)],
                'p99': sorted_times[min(int(n * 0.99), n - 1)],
                'max': sorted_times[-1],
                'min': sorted_times[0],
                'avg': sum(sorted_times) / n,
                'count': n,
                'total_recorded': self._total_recorded,
            }
    
    def reset(self) -> None:
        """Reseta histórico de tempos."""
        with self._lock:
            self.times.clear()
            # Mantém total_recorded para histórico


# ==============================================================================
# CIRCUIT BREAKER
# ==============================================================================

class CircuitState(Enum):
    """Estados do circuit breaker."""
    CLOSED = "CLOSED"      # Normal, operando
    OPEN = "OPEN"          # Falhas demais, bloqueando
    HALF_OPEN = "HALF_OPEN"  # Tentando recuperar


@dataclass
class CircuitBreaker:
    """
    Circuit breaker para proteção contra falhas em cascata.
    
    Estados:
    - CLOSED: Operação normal
    - OPEN: Muitas falhas, rejeita operações
    - HALF_OPEN: Permite uma operação de teste
    
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, recovery_time_ms=5000)
        >>> if breaker.can_execute():
        ...     try:
        ...         do_operation()
        ...         breaker.record_success()
        ...     except:
        ...         breaker.record_failure()
    """
    
    failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD
    recovery_time_ms: int = CIRCUIT_BREAKER_RECOVERY_TIME_MS
    
    # Estado interno
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _last_failure_time: int = field(default=0, init=False)
    _last_state_change: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def __post_init__(self):
        self._last_state_change = get_current_time_ms()
    
    @property
    def state(self) -> str:
        """Estado atual como string."""
        return self._state.value
    
    @property
    def failures(self) -> int:
        """Número de falhas consecutivas."""
        return self._failures
    
    def can_execute(self) -> bool:
        """
        Verifica se pode executar operação.
        
        Returns:
            True se permitido, False se circuit aberto
        """
        with self._lock:
            now = get_current_time_ms()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Verifica se pode tentar recuperar
                time_since_failure = now - self._last_failure_time
                if time_since_failure >= self.recovery_time_ms:
                    self._transition_to(CircuitState.HALF_OPEN, now)
                    return True
                return False
            
            # HALF_OPEN: permite uma tentativa
            return True
    
    def record_success(self) -> None:
        """Registra operação bem-sucedida."""
        with self._lock:
            now = get_current_time_ms()
            self._successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Recuperado!
                self._transition_to(CircuitState.CLOSED, now)
                self._failures = 0
    
    def record_failure(self) -> None:
        """Registra falha."""
        with self._lock:
            now = get_current_time_ms()
            self._failures += 1
            self._last_failure_time = now
            
            if self._state == CircuitState.HALF_OPEN:
                # Falhou durante recuperação
                self._transition_to(CircuitState.OPEN, now)
            elif self._state == CircuitState.CLOSED:
                if self._failures >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN, now)
    
    def _transition_to(self, new_state: CircuitState, now: int) -> None:
        """Transiciona para novo estado."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = now
        
        if lazy_log.should_log(f"circuit_breaker_{new_state.value}"):
            logging.info(
                f"🔌 CircuitBreaker: {old_state.value} → {new_state.value} "
                f"(failures={self._failures})"
            )
    
    def reset(self) -> None:
        """Reseta circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0
            self._last_failure_time = 0
            self._last_state_change = get_current_time_ms()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do circuit breaker."""
        with self._lock:
            now = get_current_time_ms()
            time_in_state = now - self._last_state_change
            
            recovery_remaining = 0
            if self._state == CircuitState.OPEN:
                recovery_remaining = max(
                    0, 
                    self.recovery_time_ms - (now - self._last_failure_time)
                )
            
            return {
                'state': self._state.value,
                'failures': self._failures,
                'successes': self._successes,
                'time_in_state_ms': time_in_state,
                'recovery_remaining_ms': recovery_remaining,
                'threshold': self.failure_threshold,
            }


# ==============================================================================
# HEALTH CHECKER
# ==============================================================================

@dataclass
class HealthStatus:
    """Status de saúde do sistema."""
    status: str  # HEALTHY, DEGRADED, UNHEALTHY
    issues: List[str]
    metrics: Dict[str, Any]
    timestamp: str
    
    def is_healthy(self) -> bool:
        return self.status == "HEALTHY"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'issues': self.issues,
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'is_healthy': self.is_healthy(),
        }


class HealthChecker:
    """
    Verificador de saúde do FlowAnalyzer.
    
    Monitora:
    - Taxa de erros
    - Latência de processamento
    - Uso de memória
    - Estado do circuit breaker
    """
    
    def __init__(
        self,
        valid_rate_threshold: float = 95.0,
        latency_p99_threshold_ms: float = HIGH_LATENCY_P99_MS,
        memory_usage_threshold: float = MEMORY_USAGE_WARNING_RATIO,
    ):
        self.valid_rate_threshold = valid_rate_threshold
        self.latency_p99_threshold_ms = latency_p99_threshold_ms
        self.memory_usage_threshold = memory_usage_threshold
    
    def check(
        self,
        stats: Dict[str, Any],
        perf_stats: Dict[str, Any],
        circuit_breaker: Optional[CircuitBreaker] = None,
        timestamp_formatter=None,
    ) -> HealthStatus:
        """
        Executa verificação de saúde.
        
        Args:
            stats: Estatísticas do analyzer
            perf_stats: Estatísticas de performance
            circuit_breaker: Circuit breaker (opcional)
            timestamp_formatter: Função para formatar timestamp
            
        Returns:
            HealthStatus com resultado da verificação
        """
        issues = []
        status = "HEALTHY"
        
        # 1. Taxa de validação
        valid_rate = stats.get("valid_rate_pct", 100)
        if valid_rate < self.valid_rate_threshold:
            issues.append(f"Low valid rate: {valid_rate:.1f}%")
            status = "DEGRADED"
        
        # 2. Latência P99
        p99 = perf_stats.get("p99", 0)
        if p99 > self.latency_p99_threshold_ms:
            issues.append(f"High processing time P99: {p99:.1f}ms")
            status = "DEGRADED"
        
        # 3. Late trades
        late_trades = stats.get("late_trades", 0)
        if late_trades > 100:
            issues.append(f"High late trades: {late_trades}")
        
        # 4. Memory usage
        capacity = stats.get("flow_trades_capacity", 0)
        count = stats.get("flow_trades_count", 0)
        if capacity > 0:
            usage = count / capacity
            if usage > self.memory_usage_threshold:
                issues.append(f"High memory usage: {count}/{capacity} ({usage:.1%})")
        
        # 5. Circuit breaker
        if circuit_breaker:
            cb_stats = circuit_breaker.get_stats()
            if cb_stats['state'] == "OPEN":
                issues.append(f"Circuit breaker OPEN (failures={cb_stats['failures']})")
                status = "UNHEALTHY"
            elif cb_stats['state'] == "HALF_OPEN":
                issues.append("Circuit breaker recovering")
        
        # 6. Error counts
        error_counts = stats.get("error_counts", {})
        total_errors = sum(error_counts.values())
        if total_errors > 1000:
            issues.append(f"High error count: {total_errors}")
        
        # Timestamp
        ts_str = "unknown"
        if timestamp_formatter:
            try:
                ts_str = timestamp_formatter(get_current_time_ms())
            except Exception:
                pass
        
        return HealthStatus(
            status=status,
            issues=issues,
            metrics={
                'valid_rate_pct': valid_rate,
                'latency_p99_ms': p99,
                'late_trades': late_trades,
                'memory_usage': count / capacity if capacity > 0 else 0,
                'total_errors': total_errors,
            },
            timestamp=ts_str,
        )


# ==============================================================================
# BUY/SELL RATIO CALCULATOR
# ==============================================================================

def calculate_buy_sell_ratios(flow_data: dict) -> dict:
    """
    Calcula Buy/Sell Ratios em múltiplas janelas temporais.
    
    Ratio > 1.0 = mais compra que venda (bullish pressure)
    Ratio < 1.0 = mais venda que compra (bearish pressure)
    Ratio = 1.0 = equilibrado
    
    Também detecta tendência do ratio (aceleração/desaceleração).
    
    Args:
        flow_data: Dict com dados de fluxo. Espera chaves como:
            - buy_volume ou buy_volume_btc
            - sell_volume ou sell_volume_btc
            - Opcionalmente: net_flow_1m, net_flow_5m, net_flow_15m
            - Opcionalmente: sector_flow com retail/mid/whale
            
    Returns:
        Dict com ratios por janela e análise de tendência.
    """
    # Extrair volumes de compra/venda
    buy_vol = (
        flow_data.get("buy_volume_btc")
        or flow_data.get("buy_volume")
        or 0
    )
    sell_vol = (
        flow_data.get("sell_volume_btc")
        or flow_data.get("sell_volume")
        or 0
    )

    # Ratio principal
    if sell_vol > 0:
        main_ratio = round(buy_vol / sell_vol, 4)
    else:
        main_ratio = 1.0 if buy_vol == 0 else 99.0

    # Extrair flows de múltiplas janelas
    net_flow_1m = flow_data.get("net_flow_1m", 0)
    net_flow_5m = flow_data.get("net_flow_5m", 0)
    net_flow_15m = flow_data.get("net_flow_15m", 0)
    total_volume = flow_data.get("total_volume", 0) or flow_data.get("total_volume_btc", 0)

    # Calcular ratios por janela usando net_flow
    # net_flow > 0 = mais compra, net_flow < 0 = mais venda
    ratios = {
        "current": main_ratio,
    }

    # Imbalance por janela (normalizado)
    if total_volume and total_volume > 0:
        ratios["imbalance_1m"] = round(net_flow_1m / total_volume, 4) if net_flow_1m else 0
        ratios["imbalance_5m"] = round(net_flow_5m / total_volume, 4) if net_flow_5m else 0
        ratios["imbalance_15m"] = round(net_flow_15m / total_volume, 4) if net_flow_15m else 0

    # Sector ratios (se disponível)
    sector_flow = flow_data.get("sector_flow", {})
    sector_ratios = {}
    for sector_name, sector_data in sector_flow.items():
        if isinstance(sector_data, dict):
            s_buy = sector_data.get("buy", 0)
            s_sell = sector_data.get("sell", 0)
            if s_sell > 0:
                # Cap ratio em 10.0 para evitar valores extremos
                sector_ratios[sector_name] = round(min(s_buy / s_sell, 10.0), 4)
            elif s_buy > 0:
                sector_ratios[sector_name] = 10.0  # buy-only: cap máximo
            else:
                sector_ratios[sector_name] = 1.0

    # Detecção de tendência do fluxo
    if net_flow_1m != 0 and net_flow_5m != 0 and net_flow_15m != 0:
        # Todos na mesma direção = tendência forte
        all_positive = net_flow_1m > 0 and net_flow_5m > 0 and net_flow_15m > 0
        all_negative = net_flow_1m < 0 and net_flow_5m < 0 and net_flow_15m < 0

        if all_positive:
            # Verificar se está acelerando (1m > 5m/5 > 15m/15)
            norm_1m = abs(net_flow_1m)
            norm_5m = abs(net_flow_5m) / 5
            norm_15m = abs(net_flow_15m) / 15
            if norm_1m > norm_5m > norm_15m:
                trend = "accelerating_buying"
            elif norm_1m > norm_5m:
                trend = "increasing_buying"
            else:
                trend = "consistent_buying"
        elif all_negative:
            norm_1m = abs(net_flow_1m)
            norm_5m = abs(net_flow_5m) / 5
            norm_15m = abs(net_flow_15m) / 15
            if norm_1m > norm_5m > norm_15m:
                trend = "accelerating_selling"
            elif norm_1m > norm_5m:
                trend = "increasing_selling"
            else:
                trend = "consistent_selling"
        else:
            # Direções mistas
            if net_flow_1m > 0 and net_flow_5m < 0:
                trend = "short_term_reversal_to_buy"
            elif net_flow_1m < 0 and net_flow_5m > 0:
                trend = "short_term_reversal_to_sell"
            else:
                trend = "mixed"
    else:
        trend = "insufficient_data"

    # Classificação do pressure (baseada em ratio + flow_trend para consistência)
    if main_ratio > 2.0:
        pressure = "STRONG_BUY"
    elif main_ratio > 1.3:
        pressure = "MODERATE_BUY"
    elif main_ratio > 1.05:
        pressure = "SLIGHT_BUY"
    elif main_ratio > 0.95:
        # Zona neutra: usar flow_trend para desambiguar
        if trend in ("accelerating_buying", "increasing_buying"):
            pressure = "SLIGHT_BUY"
        elif trend in ("accelerating_selling", "increasing_selling"):
            pressure = "SLIGHT_SELL"
        else:
            pressure = "NEUTRAL"
    elif main_ratio > 0.7:
        pressure = "SLIGHT_SELL"
    elif main_ratio > 0.5:
        pressure = "MODERATE_SELL"
    else:
        pressure = "STRONG_SELL"

    return {
        "buy_sell_ratio": main_ratio,
        "ratios": ratios,
        "sector_ratios": sector_ratios,
        "pressure": pressure,
        "flow_trend": trend,
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
    }