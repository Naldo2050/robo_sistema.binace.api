# tests/unit/test_shadow_hardening_c1.py
# -*- coding: utf-8 -*-
"""
Testes de Cloud Hardening C1 para Shadow Transport:
1. Drenagem segura e idempotente no shutdown.
2. Zero instâncias quando feature OFF (get_if_initialized).
3. Recuperação de horizontes PENDING no startup sem uso de wall clock.
4. Respeito a EFFORT_RESPONSE_SHADOW_FILEPATH.
"""

import json
import os
import time
from pathlib import Path
import pytest

from flow_analyzer.effort_response_transport import (
    EffortResponseSnapshotDTO,
    ShadowAsyncTransport,
)
from flow_analyzer.effort_response_dataset import (
    EffortResponseShadowRecord,
    EffortResponseShadowStorage,
    build_shadow_record,
)


@pytest.fixture(autouse=True)
def clean_shadow_singleton():
    ShadowAsyncTransport.reset_instance_for_testing()
    yield
    ShadowAsyncTransport.reset_instance_for_testing()


def test_shadow_get_if_initialized_returns_none_when_uninitialized():
    """Quando feature OFF ou nunca instanciada, get_if_initialized retorna None sem criar singleton."""
    assert ShadowAsyncTransport.get_if_initialized() is None


def test_shadow_filepath_env_var(tmp_path, monkeypatch):
    """Verifica se EFFORT_RESPONSE_SHADOW_FILEPATH é respeitado."""
    custom_path = tmp_path / "custom_shadow.jsonl"
    monkeypatch.setenv("EFFORT_RESPONSE_SHADOW_FILEPATH", str(custom_path))
    monkeypatch.setenv("EFFORT_RESPONSE_SHADOW_ENABLED", "1")

    transport = ShadowAsyncTransport()
    assert transport.filepath == custom_path
    transport.close()


def test_shadow_shutdown_drains_and_prevents_new_submissions(tmp_path):
    """Valida drain na parada e rejeição de novas submissões após close."""
    jsonl_file = tmp_path / "shadow_drain.jsonl"
    transport = ShadowAsyncTransport(filepath=jsonl_file, enabled=True, queue_capacity=100)

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=1700000060000,
        observation_open_ms=1700000000000,
        observation_close_ms=1700000059999,
        buy_notional_usd=50000.0,
        sell_notional_usd=30000.0,
        open=65000.0,
        high=65100.0,
        low=64900.0,
        close=65050.0,
        window_duration_ms=60000,
    )

    # Submete item válido
    assert transport.submit_nowait(dto) is True

    # Executa close
    assert transport.close(timeout=5.0) is True

    # Novas submissões devem ser rejeitadas
    assert transport.submit_nowait(dto) is False

    # Confirma que o item foi gravado no disco
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["provenance"]["symbol"] == "BTCUSDT"


def test_shadow_pending_recovery_without_wall_clock(tmp_path):
    """
    Testa a recuperação no startup:
    - Sem watermark inicial: entra em RECOVERY_PENDING.
    - Na chegada do primeiro event-time real:
      * Horizontes anteriores a (event_time - tolerância) viram INSUFFICIENT_DATA.
      * Horizontes futuros em relação ao watermark são reinseridos no heap ativo.
    """
    jsonl_file = tmp_path / "shadow_recovery.jsonl"

    # 1. Cria registro anterior com causal_anchor_ms = 1000s
    t_anchor = 1_000_000
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=t_anchor - 60_000,
        window_close_ms=t_anchor - 1,
        window_data={"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0, "buy_notional_usd": 10.0, "sell_notional_usd": 10.0, "window_duration_ms": 60000},
        causal_anchor_ms=t_anchor,
        observation_open_ms=t_anchor - 60_000,
        observation_close_ms=t_anchor - 1,
    )

    storage = EffortResponseShadowStorage(jsonl_file)
    storage.append_record(rec)

    # 2. Inicializa transporte sem watermark (startup_event_time_ms=None)
    transport = ShadowAsyncTransport(filepath=jsonl_file, enabled=True, start_worker=False)

    stats = transport.get_stats()
    assert stats["recovery_pending_state"] == "RECOVERY_PENDING"
    assert stats["recovery_pending_count"] == 1
    # Heap ainda vazio até a chegada do primeiro watermark
    assert stats["pending_heap_size"] == 0

    # 3. Primeiro event-time real chega da exchange: 1_000_000 + 120_000 ms (+2 min)
    # 1m (target = 1000s + 60s) já venceu durante o downtime -> deve virar INSUFFICIENT_DATA
    # 5m (target = 1000s + 300s) ainda é futuro -> deve ser reinserido no heap ativo
    # 15m (target = 1000s + 900s) ainda é futuro -> deve ser reinserido no heap ativo
    first_event_time = t_anchor + 120_000
    transport.resolve_startup_recovery(first_event_time)

    stats_after = transport.get_stats()
    assert stats_after["recovery_pending_state"] == "RESOLVED"
    assert stats_after["recovery_pending_count"] == 0
    # 5m e 15m foram reinseridos no heap ativo (2 itens)
    assert stats_after["pending_heap_size"] == 2

    # Verifica o JSONL: 1m deve ter sido atualizado com INSUFFICIENT_DATA
    updated_records = storage.read_records()
    assert len(updated_records) == 1
    outcomes = updated_records[0].outcomes_future["horizons"]
    assert outcomes["1m"]["status"] == "INSUFFICIENT_DATA"
    assert outcomes["5m"]["status"] == "PENDING"
    assert outcomes["15m"]["status"] == "PENDING"

    transport.close()
