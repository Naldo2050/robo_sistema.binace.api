# tests/unit/test_volume_spike_dual_gate.py
# -*- coding: utf-8 -*-
"""
Testes unitários para o gate duplo de detecção de VOLUME_SPIKE em trading/alert_engine.py.
"""

import pytest
from trading.alert_engine import detect_volume_spike, _get_volume_baseline_p95


def test_volume_spike_blocked_when_volume_below_p95():
    """Mesmo com ratio alto (ex: 5x a média), volume abaixo do p95 não deve disparar alerta."""
    # Hora 14 UTC: p95 é 769.82
    # current_volume = 100, average_volume = 20 -> ratio = 5.0 (alto!), mas 100 < 769.82
    alert = detect_volume_spike(
        current_volume=100.0,
        average_volume=20.0,
        threshold_factor=3.0,
        hour_utc=14,
    )
    assert alert is None, "Deveria ser bloqueado pelo gate de volume mínimo p95"


def test_volume_spike_triggers_when_both_gates_pass():
    """Quando ratio >= threshold E current_volume >= p95, alerta deve ser disparado."""
    # Hora 14 UTC: p95 é 769.82
    # current_volume = 1000.0, average_volume = 200.0 -> ratio = 5.0, 1000.0 >= 769.82
    alert = detect_volume_spike(
        current_volume=1000.0,
        average_volume=200.0,
        threshold_factor=3.0,
        hour_utc=14,
    )
    assert alert is not None
    assert alert["type"] == "VOLUME_SPIKE"
    assert alert["current_volume"] == 1000.0
    assert alert["p95_threshold"] == pytest.approx(769.82, abs=0.1)
    assert alert["threshold_exceeded"] == 5.0


def test_volume_spike_fallback_global_p95():
    """Quando hora não é encontrada ou fallback é acionado, usa p95 global de 367.6."""
    p95_val = _get_volume_baseline_p95(hour_utc=999)
    assert p95_val == pytest.approx(367.6, abs=0.1)

    # 350.0 < 367.6 -> bloqueado
    alert_blocked = detect_volume_spike(
        current_volume=350.0,
        average_volume=50.0,
        threshold_factor=3.0,
        hour_utc=999,
    )
    assert alert_blocked is None

    # 400.0 >= 367.6 e 400/50 = 8.0 >= 3.0 -> aprovado
    alert_passed = detect_volume_spike(
        current_volume=400.0,
        average_volume=50.0,
        threshold_factor=3.0,
        hour_utc=999,
    )
    assert alert_passed is not None
    assert alert_passed["p95_threshold"] == pytest.approx(367.6, abs=0.1)
