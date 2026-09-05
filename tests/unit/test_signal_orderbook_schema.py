# tests/unit/test_signal_orderbook_schema.py
# -*- coding: utf-8 -*-
"""
Teste unitário para validação do schema de orderbook_data em eventos de sinal (Absorção / Exaustão)
comparado ao ANALYSIS_TRIGGER:
Garante que sinais carreguem o bloco orderbook_data completo com timestamps, source e snapshot_offset_ms.
"""
from __future__ import annotations

import time
from typing import Dict, Any

import pytest
from data_processing.data_handler import (
    create_absorption_event,
    create_exhaustion_event,
)
from data_processing.enrichment_integrator import build_analysis_trigger_event


@pytest.fixture
def fake_trades_fixture():
    """Fixture com trades mínimos válidos para gerar janelas."""
    base_time = 1757000000000
    trades = []
    for i in range(20):
        trades.append({
            "p": 60000.0 + (i % 3) * 10,
            "q": 0.5,
            "T": base_time + i * 1000,
            "m": (i % 2 == 0),
        })
    return trades


@pytest.fixture
def sample_ob_event():
    """Envelope de orderbook idêntico ao produzido por run_orderbook_analyze / fetch_orderbook_with_retry."""
    close_ms = 1757000020000
    exchange_ms = close_ms + 150
    return {
        "is_valid": True,
        "source": "live_sync",
        "source_type": "live_sync",
        "snapshot_offset_ms": 150,
        "timestamps": {
            "exchange_ms": exchange_ms,
            "received_ms": close_ms + 160,
            "delay_ms": 10,
        },
        "orderbook_data": {
            "bids": [[60000.0, 1.5]],
            "asks": [[60001.0, 2.0]],
            "spread_bps": 0.16,
            "source": "live_sync",
            "snapshot_offset_ms": 150,
            "timestamps": {
                "exchange_ms": exchange_ms,
                "received_ms": close_ms + 160,
            },
        },
    }


def test_absorption_signal_carries_orderbook_fields(fake_trades_fixture, sample_ob_event):
    """Garante que evento de Absorção contenha timestamps, source e snapshot_offset_ms em orderbook_data."""
    event = create_absorption_event(
        window_data=fake_trades_fixture,
        symbol="BTCUSDT",
        orderbook_data=sample_ob_event,
    )

    assert "orderbook_data" in event
    ob_data = event["orderbook_data"]
    assert isinstance(ob_data, dict)
    assert ob_data.get("source") == "live_sync"
    assert ob_data.get("snapshot_offset_ms") == 150
    assert "timestamps" in ob_data
    assert ob_data["timestamps"].get("exchange_ms") == 1757000020150

    # Garantir presença no raw_event
    assert "raw_event" in event
    raw_ob = event["raw_event"].get("orderbook_data")
    assert isinstance(raw_ob, dict)
    assert raw_ob.get("source") == "live_sync"
    assert raw_ob.get("snapshot_offset_ms") == 150
    assert "timestamps" in raw_ob


def test_exhaustion_signal_carries_orderbook_fields(fake_trades_fixture, sample_ob_event):
    """Garante que evento de Exaustão contenha timestamps, source e snapshot_offset_ms em orderbook_data."""
    event = create_exhaustion_event(
        window_data=fake_trades_fixture,
        symbol="BTCUSDT",
        history_volumes=[10.0, 12.0, 8.0],
        orderbook_data=sample_ob_event,
    )

    assert "orderbook_data" in event
    ob_data = event["orderbook_data"]
    assert isinstance(ob_data, dict)
    assert ob_data.get("source") == "live_sync"
    assert ob_data.get("snapshot_offset_ms") == 150
    assert "timestamps" in ob_data

    # Garantir presença no raw_event
    assert "raw_event" in event
    raw_ob = event["raw_event"].get("orderbook_data")
    assert isinstance(raw_ob, dict)
    assert raw_ob.get("source") == "live_sync"
    assert raw_ob.get("snapshot_offset_ms") == 150


def test_signal_and_analysis_trigger_orderbook_schema_parity(fake_trades_fixture, sample_ob_event):
    """Compara evento de sinal com ANALYSIS_TRIGGER garantindo paridade do bloco orderbook_data."""
    sig_event = create_absorption_event(
        window_data=fake_trades_fixture,
        symbol="BTCUSDT",
        orderbook_data=sample_ob_event,
    )

    trigger_raw = {
        "delta": 1.0,
        "volume_total": 10.0,
        "preco_fechamento": 60000.0,
        "orderbook_data": sample_ob_event,
    }
    trigger_event = build_analysis_trigger_event("BTCUSDT", trigger_raw)

    sig_ob = sig_event["orderbook_data"]
    trig_ob = trigger_event["raw_event"]["orderbook_data"]

    for key in ("timestamps", "source", "snapshot_offset_ms"):
        assert key in sig_ob, f"Chave '{key}' ausente em sig_event['orderbook_data']"
        assert key in trig_ob, f"Chave '{key}' ausente em trigger_event['orderbook_data']"
        assert sig_ob[key] == trig_ob[key], f"Valor divergente para chave '{key}'"
