# circuit_breaker.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
import time
import threading
import logging
import random


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5          # falhas consecutivas para abrir
    success_threshold: int = 2          # sucessos no HALF_OPEN para fechar
    timeout_seconds: float = 30.0       # tempo em OPEN antes de ir para HALF_OPEN
    half_open_max_calls: int = 3        # máximo de tentativas no HALF_OPEN
    fallback_enabled: bool = True       # habilita fallback para REST API
    max_retry_attempts: int = 3         # máximo de tentativas de retry
    base_retry_delay: float = 1.0       # delay base para retry
    max_retry_delay: float = 10.0       # delay máximo para retry


class CircuitBreaker:
    """
    Circuit Breaker thread-safe.

    Contabiliza falhas por "operação" (ex.: uma chamada completa de fetch),
    não por tentativa interna de retry (isso é responsabilidade do caller).
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED

        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time_mono: Optional[float] = None

        self._half_open_calls = 0

    def _transition_to(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time_mono = None

        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            self._half_open_calls = 0
            # last_failure_time_mono já deve estar setado no record_failure()

        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0

        logging.warning(f"CircuitBreaker[{self.name}] {old.value} -> {new_state.value}")

    def _maybe_move_open_to_half_open(self, now_mono: float) -> None:
        if self._state != CircuitState.OPEN:
            return
        if self._last_failure_time_mono is None:
            return
        if (now_mono - self._last_failure_time_mono) >= self.config.timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)

    def state(self) -> CircuitState:
        with self._lock:
            now = time.monotonic()
            self._maybe_move_open_to_half_open(now)
            return self._state

    def allow_request(self) -> bool:
        """
        CLOSED: permite
        OPEN: bloqueia até timeout_seconds
        HALF_OPEN: permite até half_open_max_calls
        """
        with self._lock:
            now = time.monotonic()
            self._maybe_move_open_to_half_open(now)

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True

            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                return

            if self._state == CircuitState.CLOSED:
                # sucesso zera sequência de falhas
                self._failure_count = 0
                return

            # se estiver OPEN, sucesso não deveria ocorrer (pois estaria bloqueando),
            # mas não quebra se acontecer:
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time_mono = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # qualquer falha em HALF_OPEN abre imediatamente
                self._transition_to(CircuitState.OPEN)
                return

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._maybe_move_open_to_half_open(now)
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "timeout_seconds": self.config.timeout_seconds,
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "half_open_max_calls": self.config.half_open_max_calls,
                "fallback_enabled": self.config.fallback_enabled,
                "max_retry_attempts": self.config.max_retry_attempts,
            }

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calcula delay com backoff exponencial e jitter para evitar thundering herd.
        
        Args:
            attempt: Número da tentativa (0-indexed)
            
        Returns:
            Delay em segundos com jitter
        """
        delay = min(
            self.config.max_retry_delay,
            self.config.base_retry_delay * (2 ** attempt)
        )
        jitter = random.uniform(0, delay * 0.25)
        return delay + jitter

    def should_fallback_to_rest(self) -> bool:
        """
        Determina se deve usar fallback para REST API quando WebSocket falhar.
        
        Returns:
            True se deve usar fallback, False caso contrário
        """
        with self._lock:
            return (
                self.config.fallback_enabled and
                self._state == CircuitState.OPEN and
                self._failure_count >= self.config.failure_threshold
            )