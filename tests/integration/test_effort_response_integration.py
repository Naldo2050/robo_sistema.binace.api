# tests/integration/test_effort_response_integration.py
"""
Testes de integração para o hook de coleta Shadow Effort/Response no WindowProcessor (P1-F Commit 2).

Testa:
1. Feature flag OFF por padrão: nenhuma coleta, zero arquivo, zero fila.
2. Feature flag ON: enfileiramento e persistência corretos com DTO minimalista.
3. Resiliência: falha na coleta não interrompe o processador de janelas.
4. Boundary causal: janela com obs_close_ms >= causal_anchor_ms é descartada do shadow.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from flow_analyzer.effort_response_dataset import EffortResponseShadowStorage
from flow_analyzer.effort_response_transport import ShadowAsyncTransport
from market_orchestrator.windows.window_processor import _try_record_effort_response_shadow


@pytest.fixture(autouse=True)
def reset_shadow_transport():
    """Garante isolamento de singleton e variáveis de ambiente."""
    ShadowAsyncTransport.reset_instance_for_testing()
    old_flag = os.environ.get("EFFORT_RESPONSE_SHADOW_ENABLED")
    yield
    ShadowAsyncTransport.reset_instance_for_testing()
    if old_flag is not None:
        os.environ["EFFORT_RESPONSE_SHADOW_ENABLED"] = old_flag
    else:
        os.environ.pop("EFFORT_RESPONSE_SHADOW_ENABLED", None)


@pytest.fixture
def temp_jsonl(tmp_path):
    """Arquivo temporário para persistência de teste."""
    return tmp_path / "shadow_test_integration.jsonl"


def _build_mock_bot():
    bot = MagicMock()
    bot.symbol = "BTCUSDT"
    return bot


def _build_mock_window_data(causal_anchor_ms: int):
    # Trades ocorridos estritamente antes do causal_anchor_ms
    return [
        {"p": 80000.0, "q": 1.5, "T": causal_anchor_ms - 50_000, "m": False},
        {"p": 80010.0, "q": 0.5, "T": causal_anchor_ms - 30_000, "m": True},
        {"p": 80020.0, "q": 2.0, "T": causal_anchor_ms - 1_390, "m": False},  # J2: last trade
    ]


def _build_mock_enriched():
    return {
        "symbol": "BTCUSDT",
        "ohlc": {
            "open": 80000.0,
            "high": 80025.0,
            "low": 79990.0,
            "close": 80020.0,
            "vwap": 80012.5,
        },
        "poc_price": 80015.0,
    }


def test_hook_noop_when_flag_disabled(temp_jsonl):
    """Quando EFFORT_RESPONSE_SHADOW_ENABLED está desligado (0), o hook não faz nada."""
    os.environ["EFFORT_RESPONSE_SHADOW_ENABLED"] = "0"

    bot = _build_mock_bot()
    causal_anchor = 1788702420000
    trades = _build_mock_window_data(causal_anchor)
    enriched = _build_mock_enriched()

    _try_record_effort_response_shadow(
        bot=bot,
        valid_window_data=trades,
        close_ms=causal_anchor,
        enriched=enriched,
        flow_metrics=None,
        ob_event=None,
        macro_context=None,
    )

    # O arquivo nem sequer deve ser criado
    assert not temp_jsonl.exists()


def test_hook_enqueues_and_persists_when_flag_enabled(temp_jsonl):
    """Quando EFFORT_RESPONSE_SHADOW_ENABLED=1, o hook coleta os dados e grava o registro."""
    os.environ["EFFORT_RESPONSE_SHADOW_ENABLED"] = "1"

    # Inicializa transport singleton apontando para o arquivo temporário
    transport = ShadowAsyncTransport(filepath=temp_jsonl, enabled=True, queue_capacity=50)
    ShadowAsyncTransport._singleton = transport

    bot = _build_mock_bot()
    causal_anchor = 1788702420000
    trades = _build_mock_window_data(causal_anchor)
    enriched = _build_mock_enriched()

    flow_metrics = {
        "buy_notional_usdt": 280000.0,
        "sell_notional_usdt": 40005.0,
    }
    ob_event = {
        "spread": 0.5,
        "bid_depth": 15.0,
        "ask_depth": 12.0,
        "imbalance": 0.11,
        "source_type": "L2_SNAPSHOT",
    }
    macro_context = {
        "regime_current_at_t": "TRENDING_EXPANSION",
        "regime_status_at_t": "ACTIVE",
        "regime_calibration_status_at_t": "CALIBRATED",
    }

    _try_record_effort_response_shadow(
        bot=bot,
        valid_window_data=trades,
        close_ms=causal_anchor,
        enriched=enriched,
        flow_metrics=flow_metrics,
        ob_event=ob_event,
        macro_context=macro_context,
    )

    # Aguarda o worker single writer persistir
    flushed = transport.flush(timeout=2.0)
    assert flushed is True

    storage = EffortResponseShadowStorage(temp_jsonl)
    records = storage.read_records()
    assert len(records) == 1

    rec = records[0]
    prov = rec.provenance
    assert prov.symbol == "BTCUSDT"
    assert prov.causal_anchor_ms == causal_anchor
    assert prov.observation_open_ms == trades[0]["T"]
    assert prov.observation_close_ms == trades[-1]["T"]

    # Verifica os horizontes PENDING ancorados no causal_anchor
    horizons = rec.outcomes_future["horizons"]
    assert horizons["1m"]["target_timestamp_ms"] == causal_anchor + 60_000
    assert horizons["5m"]["target_timestamp_ms"] == causal_anchor + 300_000
    assert horizons["15m"]["target_timestamp_ms"] == causal_anchor + 900_000

    transport.close(timeout=1.0)


def test_hook_discards_violating_boundary_for_causal_safety(temp_jsonl):
    """Janela com trade T >= causal_anchor_ms é rejeitada pelo hook para garantir causal safety."""
    os.environ["EFFORT_RESPONSE_SHADOW_ENABLED"] = "1"

    transport = ShadowAsyncTransport(filepath=temp_jsonl, enabled=True, queue_capacity=50)
    ShadowAsyncTransport._singleton = transport

    bot = _build_mock_bot()
    causal_anchor = 1788702420000

    # Trade com T == causal_anchor (violação da exclusão causal estrita)
    trades_with_violation = [
        {"p": 80000.0, "q": 1.0, "T": causal_anchor - 1000, "m": False},
        {"p": 80010.0, "q": 1.0, "T": causal_anchor, "m": True},  # T == anchor proibido
    ]
    enriched = _build_mock_enriched()

    _try_record_effort_response_shadow(
        bot=bot,
        valid_window_data=trades_with_violation,
        close_ms=causal_anchor,
        enriched=enriched,
        flow_metrics=None,
        ob_event=None,
        macro_context=None,
    )

    transport.flush(timeout=1.0)

    # Nenhum registro deve ter sido gravado
    storage = EffortResponseShadowStorage(temp_jsonl)
    records = storage.read_records()
    assert len(records) == 0

    transport.close(timeout=1.0)


def test_hook_failure_isolation(temp_jsonl):
    """Exceção inesperada durante extração de dados não se propaga para o chamador."""
    os.environ["EFFORT_RESPONSE_SHADOW_ENABLED"] = "1"

    # Envia enriched com estrutura corrompida propositalmente
    enriched_corrompido = {
        "ohlc": "invalid_not_a_dict"
    }

    # Deve executar sem levantar exceção
    _try_record_effort_response_shadow(
        bot=_build_mock_bot(),
        valid_window_data=_build_mock_window_data(1788702420000),
        close_ms=1788702420000,
        enriched=enriched_corrompido,  # type: ignore
        flow_metrics=None,
        ob_event=None,
        macro_context=None,
    )
