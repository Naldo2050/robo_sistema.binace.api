# tests/test_rate_limiter.py
from __future__ import annotations

import market_orchestrator.connection.robust_connection as rc


def test_rate_limiter_blocks_when_exceeding_calls(monkeypatch):
    """
    RateLimiter.acquire deve bloquear (via sleep) quando o número de chamadas
    excede max_calls dentro de period_seconds.
    Utilizamos monkeypatch em time.monotonic e time.sleep para não atrasar o teste.
    """
    current = {"t": 0.0}

    def fake_monotonic():
        return current["t"]

    def fake_sleep(dt: float):
        current["t"] += dt

    # Patching time usado dentro de robust_connection
    monkeypatch.setattr(rc.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(rc.time, "sleep", fake_sleep)

    rl = rc.RateLimiter(max_calls=2, period_seconds=1.0)

    # Primeira chamada: t=0, sem bloqueio
    rl.acquire()
    assert current["t"] == 0.0

    # Segunda chamada: ainda dentro da janela, sem bloqueio
    rl.acquire()
    assert current["t"] == 0.0

    # Terceira chamada: deve "dormir" até 1.0s
    rl.acquire()

    # Nosso fake_sleep incrementa o tempo em sleep_time.
    # Esperamos que o tempo tenha avançado cerca de 1.0s.
    assert current["t"] >= 1.0 - 1e-6