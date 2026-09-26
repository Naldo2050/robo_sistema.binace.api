# flow_analyzer/metrics.py
"""
Métricas e observabilidade do FlowAnalyzer.

Inclui:
- PerformanceMonitor: Tracking de latência com percentis
- CircuitBreaker: Proteção contra falhas em cascata
- HealthChecker: Verificação de saúde
"""

import logging
import math
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

def _finite_volume(value):
    """Coage volume de entrada: float finito ou None (ausente/inválido).

    Alinhado à política canônica (NaN/±Inf = ausência, nunca 0).
    0.0 presente continua 0.0.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _present_volume(flow_data: dict, *keys):
    """Primeiro valor presente e finito (0.0 presente conta como presente)."""
    for key in keys:
        v = _finite_volume(flow_data.get(key))
        if v is not None:
            return v
    return None


# ==============================================================================
# P0-B1: INTEGRIDADE DO PRODUTOR + METADATA TEMPORAL (fail-closed, aditivo)
# ==============================================================================
#
# Canônico e de fonte única: aggregates.py reutiliza estas funções
# (mesma direção de import já existente para _finite_volume/_present_volume;
# metrics.py nunca importa aggregates — sem ciclo).
#
# Contrato:
# - Valores parciais 1m/5m/15m SEMPRE preservados para observabilidade.
# - Cada janela tentada recebe temporal_validity:
#     VALID   <=> flow_window_integrity[tf].is_temporal_coverage_valid is True
#               (prova positiva de cobertura FULL; sem ela, nunca VALID).
#     PARTIAL <=> aritmética válida mas cobertura não-FULL (WARMING_UP,
#               CAPACITY_TRUNCATED, TRUNCATED, ...) ou integridade ausente
#               (reason=NO_INTEGRITY_INFO — fail-safe, nunca VALID assumido).
#     INVALID <=> aritmética inválida (non-finite, denominador ausente/
#               não-finito/não-positivo, ou |imb|>1 além do epsilon):
#               imbalance_value=null (chave numérica omitida, como antes),
#               reason=INVARIANT_VIOLATION. Nunca clamp silencioso para 1.0.
# - Nenhum threshold novo: o único epsilon é _IMBALANCE_FP_EPS = 1e-9,
#   literal já existente no clamp de fronteira abaixo (reutilizado, não criado).
# - Janelas cuja chave net_flow_Xm sequer existe em flow_data são "ausentes"
#   (sem entrada em validity): ausência de input ≠ violação.

_IMBALANCE_FP_EPS = 1e-9  # reutiliza tolerância preexistente (ver loop abaixo)

_IMBALANCE_WINDOWS = (
    (1, "net_flow_1m", "imbalance_1m", "1m"),
    (5, "net_flow_5m", "imbalance_5m", "5m"),
    (15, "net_flow_15m", "imbalance_15m", "15m"),
)


def _window_total_for_imbalance(flow_data: dict, window_min: int):
    """Denominador da PRÓPRIA janela (paridade P01 com o código anterior).

    Fallback legado total_volume/total_volume_btc vale SOMENTE para 1m.
    Retorna float finito ou None (ausente/inválido). Ao contrário do código
    anterior, o fallback passa por _finite_volume: lixo não-finito vira
    INVALID explícito em vez de omitir silenciosamente ou levantar TypeError.
    """
    for key in (f"total_volume_{window_min}m", f"total_{window_min}m"):
        v = _finite_volume(flow_data.get(key))
        if v is not None:
            return v
    if window_min == 1:
        fallback = flow_data.get("total_volume", 0) or flow_data.get("total_volume_btc", 0)
        return _finite_volume(fallback)
    return None


def _temporal_validity_for_window(integrity: Any, window_key: str):
    """(validity, reason) a partir EXCLUSIVAMENTE de flow_window_integrity.

    Sem threshold novo: só lê is_temporal_coverage_valid (bool) e status (str).
    """
    entry = integrity.get(window_key) if isinstance(integrity, dict) else None
    if isinstance(entry, dict) and entry.get("is_temporal_coverage_valid") is True:
        return "VALID", None
    if isinstance(entry, dict):
        return "PARTIAL", str(entry.get("status", "UNKNOWN_STATUS"))
    return "PARTIAL", "NO_INTEGRITY_INFO"


def is_temporal_confirmation_valid(integrity: Any, validity_map: Any, window_key: str) -> bool:
    """Gate canônico P0-B2: janela conta como confirmação temporal?

    Fail-closed duplo (contrato P0-B2 §1): True SOMENTE se
      imbalance_validity[tf].validity == "VALID"
      E flow_window_integrity[tf].is_temporal_coverage_valid is True.
    Qualquer divergência, ausência, PARTIAL, INVALID ou NO_INTEGRITY_INFO
    => False (observável, nunca confirmatório). Sem threshold novo.

    Localização: metrics.py (folha: só stdlib+constants+utils). Reutilizado por
    aggregates.py (importa daqui), enricher, flow_summary e payload builders
    (lazy import — sem ciclo: flow_analyzer nunca importa esses pacotes).
    """
    entry = validity_map.get(window_key) if isinstance(validity_map, dict) else None
    if not isinstance(entry, dict) or entry.get("validity") != "VALID":
        return False
    info = integrity.get(window_key) if isinstance(integrity, dict) else None
    if not isinstance(info, dict) or info.get("is_temporal_coverage_valid") is not True:
        return False
    return True


def compute_window_imbalances(flow_data: dict):
    """Núcleo canônico P0-B1: (ratios_numéricos, validity_map).

    ratios_numéricos: {imbalance_Xm: float} só para aritmética válida em
      [-1, +1] (snap de fronteira ±1e-9 preservado). Bit-equivalente ao código
      anterior para todos os inputs válidos.
    validity_map: {"1m"|"5m"|"15m": {"validity": VALID|PARTIAL|INVALID,
      "reason": None|status|NO_INTEGRITY_INFO|INVARIANT_VIOLATION}} somente
      para janelas cujo net_flow_Xm foi fornecido (tentadas). Janela sem chave
      net => ausente em ambos (nunca fabricada).
    """
    ratios: Dict[str, float] = {}
    validity: Dict[str, Dict[str, Any]] = {}
    integrity = flow_data.get("flow_window_integrity")

    for window_min, net_key, imb_key, window_label in _IMBALANCE_WINDOWS:
        if net_key not in flow_data:
            continue  # input ausente => janela ausente (não é violação)
        net_flow = _finite_volume(flow_data.get(net_key))
        if net_flow is None:
            # Chave presente mas non-finite (NaN/Inf/str): E => null/INVALID.
            validity[window_label] = {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
            continue
        window_total = _window_total_for_imbalance(flow_data, window_min)
        if window_total is None or not window_total > 0:
            # Denominador ausente/não-finito/não-positivo => null/INVALID.
            validity[window_label] = {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
            continue
        raw_imbalance = net_flow / window_total  # sem arredondar antes da divisão
        if not math.isfinite(raw_imbalance):
            validity[window_label] = {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
            continue
        # Proteção preexistente só contra epsilon numérico (não clamp de erro):
        # ramos separados para +1 e -1, como no código anterior.
        if raw_imbalance > 1.0 and raw_imbalance <= 1.0 + _IMBALANCE_FP_EPS:
            raw_imbalance = 1.0
        elif raw_imbalance < -1.0 and raw_imbalance >= -1.0 - _IMBALANCE_FP_EPS:
            raw_imbalance = -1.0
        elif raw_imbalance > 1.0 or raw_imbalance < -1.0:
            # Fora de [-1,+1] além da tolerância: NUNCA clamp para ±1.0,
            # NUNCA publica o número; null + INVALID explícito.
            validity[window_label] = {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
            continue
        ratios[imb_key] = round(raw_imbalance, 4)
        temporal, reason = _temporal_validity_for_window(integrity, window_label)
        validity[window_label] = {"validity": temporal, "reason": reason}

    return ratios, validity


def calculate_buy_sell_ratios(flow_data: dict) -> dict:
    """
    Calcula Buy/Sell Ratios em múltiplas janelas temporais.

    Ratio > 1.0 = mais compra que venda (bullish pressure)
    Ratio < 1.0 = mais venda que compra (bearish pressure)
    Ratio = 1.0 = equilibrado (somente com buy>0 E sell>0 observados)

    Contrato serializável (B-P0-1): ratio nunca é NaN/Inf/sentinela.
      - sell_only (buy=0, sell>0): ratio 0.0 (limite matemático válido)
      - buy_only (buy>0, sell=0): ratio None (infinito não serializa);
        direção preservada nos volumes/pcts/status
      - no_volume (0/0) ou invalid (missing/None/NaN/Inf): ratio None
      - ratio_state: two_sided | buy_only | sell_only | no_volume | invalid

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
    # Extrair volumes de compra/venda (finito ou None; sem coagir p/ 0)
    buy_vol = _present_volume(flow_data, "buy_volume_btc", "buy_volume")
    sell_vol = _present_volume(flow_data, "sell_volume_btc", "sell_volume")

    # Ratio principal (serializável)
    if (buy_vol is None or sell_vol is None
            or buy_vol < 0 or sell_vol < 0):
        main_ratio, ratio_state = None, "invalid"
    elif buy_vol == 0 and sell_vol == 0:
        main_ratio, ratio_state = None, "no_volume"
    elif buy_vol > 0 and sell_vol > 0:
        main_ratio, ratio_state = round(buy_vol / sell_vol, 4), "two_sided"
    elif sell_vol > 0:
        # buy == 0: limite matemático 0.0 (válido)
        main_ratio, ratio_state = 0.0, "sell_only"
    else:
        # buy > 0, sell == 0: infinito não serializa; direção nos volumes
        main_ratio, ratio_state = None, "buy_only"

    # P0-B1: núcleo canônico (valores + temporal_validity). O cálculo vive em
    # compute_window_imbalances (fonte única com aggregates.py); net/total por
    # janela são lidos lá, sempre da PRÓPRIA janela (P01, sem fallback cruzado).
    window_ratios, imbalance_validity = compute_window_imbalances(flow_data)

    # Calcular ratios por janela usando net_flow da própria janela
    # net_flow > 0 = mais compra, net_flow < 0 = mais venda
    ratios = {
        "current": main_ratio,
    }
    ratios.update(window_ratios)

    # Sector ratios (se disponível; mesma regra: só com ambos finitos)
    sector_flow = flow_data.get("sector_flow", {})
    sector_ratios = {}
    for sector_name, sector_data in sector_flow.items():
        if isinstance(sector_data, dict):
            s_buy = _present_volume(sector_data, "buy")
            s_sell = _present_volume(sector_data, "sell")
            if s_buy is None or s_sell is None or (s_buy == 0 and s_sell == 0):
                sector_ratios[sector_name] = None
            elif s_sell > 0:
                # Cap ratio em 10.0 para evitar valores extremos
                sector_ratios[sector_name] = round(min(s_buy / s_sell, 10.0), 4)
            elif s_buy > 0:
                sector_ratios[sector_name] = None  # buy-only: direção nos volumes
            else:
                sector_ratios[sector_name] = 0.0  # sell-only: limite válido

    # Detecção de tendência do fluxo (imbalance normalizado por janela).
    # P0-B2: tendência multi-TF exige 1m E 5m confirmatórios (dual VALID);
    # PARTIAL é observável mas NÃO confirma. Sem os horizontes exigidos pela
    # fórmula => "insufficient_data" (nunca "stable" por ausência). Matemática
    # com horizontes VALID inalterada (mesmo THRESHOLD 0.05).
    integrity = flow_data.get("flow_window_integrity")
    _trend_windows_ok = (
        is_temporal_confirmation_valid(integrity, imbalance_validity, "1m")
        and is_temporal_confirmation_valid(integrity, imbalance_validity, "5m")
    )
    imbalance_1m = ratios.get("imbalance_1m", 0)
    imbalance_5m = ratios.get("imbalance_5m", 0)

    if not _trend_windows_ok:
        trend = "insufficient_data"
    elif not imbalance_1m and not imbalance_5m:
        trend = "insufficient_data"
    else:
        recent = abs(imbalance_1m)
        medium = abs(imbalance_5m)
        THRESHOLD = 0.05
        if recent > medium + THRESHOLD:
            flow_trend = "accelerating"
        elif recent < medium - THRESHOLD:
            flow_trend = "decelerating"
        else:
            flow_trend = "stable"
        direction = "selling" if imbalance_1m < 0 else "buying"
        trend = f"{flow_trend}_{direction}"

    # Classificação do pressure (baseada em ratio + flow_trend para consistência)
    # Ratio ausente => pressure ausente (nunca NEUTRAL fabricado).
    if main_ratio is None:
        pressure = None
    elif main_ratio > 2.0:
        pressure = "STRONG_BUY"
    elif main_ratio > 1.3:
        pressure = "MODERATE_BUY"
    elif main_ratio > 1.05:
        pressure = "SLIGHT_BUY"
    elif main_ratio > 0.95:
        # Zona neutra: usar flow_trend para desambiguar
        if trend in ("accelerating_buying",):
            pressure = "SLIGHT_BUY"
        elif trend in ("accelerating_selling",):
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
        "ratio_state": ratio_state,
        "ratios": ratios,
        # P0-B1 (aditivo): temporal_validity por janela tentada
        # {"1m"|"5m"|"15m": {"validity": VALID|PARTIAL|INVALID, "reason": ...}}.
        # Ausente quando nenhuma janela foi tentada. Nenhum consumer é obrigado
        # a lê-lo nesta etapa (P0-B2 cuidará dos gates).
        "imbalance_validity": imbalance_validity,
        "sector_ratios": sector_ratios,
        "pressure": pressure,
        "flow_trend": trend,
        "buy_volume": round(buy_vol, 4) if buy_vol is not None else None,
        "sell_volume": round(sell_vol, 4) if sell_vol is not None else None,
    }