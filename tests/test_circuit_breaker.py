# tests/test_circuit_breaker.py
from __future__ import annotations

import time

from orderbook_core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


def test_circuit_breaker_opens_and_blocks_then_recovers():
    cb = CircuitBreaker(
        name="t",
        config=CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.2,
            half_open_max_calls=2,
        ),
    )

    # Inicialmente permite
    assert cb.allow_request() is True

    # 2 falhas -> OPEN
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is False  # OPEN bloqueia

    # Espera timeout -> HALF_OPEN
    time.sleep(0.25)
    assert cb.allow_request() is True  # HALF_OPEN permite chamadas limitadas
    assert cb.allow_request() is True
    assert cb.allow_request() is False  # excedeu half_open_max_calls

    # Sucessos suficientes em HALF_OPEN -> CLOSED
    cb.record_success()
    cb.record_success()
    assert cb.allow_request() is True  # CLOSED volta a permitir


def test_half_open_failure_goes_back_to_open():
    cb = CircuitBreaker(
        name="t2",
        config=CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            timeout_seconds=0.1,
            half_open_max_calls=1,
        ),
    )

    cb.record_failure()
    assert cb.allow_request() is False  # OPEN

    time.sleep(0.12)
    assert cb.allow_request() is True  # HALF_OPEN

    # Falhou em HALF_OPEN -> OPEN imediatamente
    cb.record_failure()
    assert cb.allow_request() is False