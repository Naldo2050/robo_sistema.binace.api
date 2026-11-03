# -*- coding: utf-8 -*-
"""Testes herméticos do produtor create_exhaustion_event (H1-B3-D).

Sem dados operacionais. O VPD é substituído por stub que falha de
propósito para forçar o fallback estático sem tocar disco/cache.
"""

import hashlib
import logging

import pytest

import data_processing.data_handler as dh
from common.identity_contract import IDENTITY_VERSION_EXHAUSTION
from data_processing.data_handler import _mk_event_id, create_exhaustion_event

SYMBOL = "BTCUSDT"
BASE_TS = 1700000000000
HISTORY = [1.0] * 5


@pytest.fixture(autouse=True)
def _disable_vpd(monkeypatch):
    class _NoVPD:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("VPD desabilitado em teste hermetico")

    monkeypatch.setattr(dh, "DynamicVolumeProfile", _NoVPD)


def _trade(price, qty, seller_aggressor, ts):
    return {"p": price, "q": qty, "m": seller_aggressor, "T": ts}


def _compra_window():
    return [
        _trade(100.0, 1.0, False, BASE_TS),
        _trade(100.5, 1.0, False, BASE_TS + 1000),
        _trade(101.0, 1.0, False, BASE_TS + 2000),
        _trade(101.2, 0.1, True, BASE_TS + 3000),
    ]


def _venda_window():
    return [
        _trade(101.0, 1.0, True, BASE_TS),
        _trade(100.5, 1.0, True, BASE_TS + 1000),
        _trade(100.0, 1.0, True, BASE_TS + 2000),
        _trade(100.8, 0.1, False, BASE_TS + 3000),
    ]


def _manual_id(window_open_ms, symbol, delta):
    base = f"v2|{window_open_ms}|{symbol}|Exaustão|{delta:.8f}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def test_exhaustion_v2_sinal_compra_delta_positivo():
    event = create_exhaustion_event(_compra_window(), SYMBOL, history_volumes=HISTORY)

    assert event["tipo_evento"] == "Exaustão"
    assert event["is_signal"] is True
    assert event["resultado_da_batalha"] == "Exaustão de Compra"
    assert event["delta"] == event["volume_compra"] - event["volume_venda"]
    assert event["delta"] > 0
    assert event["window_open_ms"] == BASE_TS
    assert event["event_id"] == _manual_id(event["window_open_ms"], SYMBOL, event["delta"])
    assert event["identity_version"] == IDENTITY_VERSION_EXHAUSTION


def test_exhaustion_v2_sinal_venda_delta_negativo():
    event = create_exhaustion_event(_venda_window(), SYMBOL, history_volumes=HISTORY)

    assert event["is_signal"] is True
    assert event["resultado_da_batalha"] == "Exaustão de Venda"
    assert event["delta"] < 0
    assert event["event_id"] == _manual_id(event["window_open_ms"], SYMBOL, event["delta"])


def test_exhaustion_v2_delta_zero_e_distinto_da_v1():
    window = [
        _trade(100.0, 0.5, False, BASE_TS),
        _trade(100.0, 0.5, True, BASE_TS + 1000),
        _trade(100.0, 0.5, False, BASE_TS + 2000),
        _trade(100.0, 0.5, True, BASE_TS + 3000),
    ]
    event = create_exhaustion_event(window, SYMBOL, history_volumes=[0.5] * 5)

    assert event["delta"] == 0.0
    assert event["is_signal"] is False
    assert event["event_id"] == _manual_id(event["window_open_ms"], SYMBOL, 0.0)

    v1_id = _mk_event_id(SYMBOL, "Exaustão", event["window_open_ms"], 0.0)
    assert event["event_id"] != v1_id


def test_exhaustion_v2_delta_minusculo_formata_zero_sem_colisao():
    eps = 1e-9
    window = [
        _trade(100.0, eps, False, BASE_TS),
        _trade(100.5, eps, False, BASE_TS + 1000),
        _trade(101.0, eps, False, BASE_TS + 2000),
        _trade(101.2, 0.1 * eps, True, BASE_TS + 3000),
    ]
    event = create_exhaustion_event(window, SYMBOL, history_volumes=[eps] * 5)

    assert event["is_signal"] is True
    assert event["delta"] != 0.0
    assert f"{event['delta']:.8f}" == "0.00000000"
    assert event["event_id"] == _manual_id(event["window_open_ms"], SYMBOL, event["delta"])


def test_exhaustion_v2_reprocessamento_identico_e_corrigido():
    e1 = create_exhaustion_event(_compra_window(), SYMBOL, history_volumes=HISTORY)
    e2 = create_exhaustion_event(_compra_window(), SYMBOL, history_volumes=HISTORY)
    assert e1["event_id"] == e2["event_id"]
    assert e1["delta"] == e2["delta"]

    corrected = _compra_window()
    corrected[1]["q"] = 1.5
    e3 = create_exhaustion_event(corrected, SYMBOL, history_volumes=HISTORY)
    assert e3["delta"] != e1["delta"]
    assert e3["event_id"] != e1["event_id"]
    assert e3["event_id"] == _manual_id(e3["window_open_ms"], SYMBOL, e3["delta"])


def test_exhaustion_v2_rejeita_delta_nao_finito(caplog):
    huge = 1.7e308
    window = [
        _trade(100.0, huge, False, BASE_TS),
        _trade(100.5, huge, False, BASE_TS + 1000),
    ]

    with caplog.at_level(logging.ERROR):
        event = create_exhaustion_event(window, SYMBOL, history_volumes=HISTORY)

    assert event["tipo_evento"] == "Erro"
    assert event["is_signal"] is False
    assert event["resultado_da_batalha"] == "Erro"
    assert "ativo" in event
    assert "event_id" not in event
    assert any("delta não finito" in rec.getMessage() for rec in caplog.records)


def test_exhaustion_v2_consistencia_com_pipeline_sem_correcao(caplog):
    from data_pipeline.pipeline import DataPipeline

    event = create_exhaustion_event(_compra_window(), SYMBOL, history_volumes=HISTORY)
    delta_antes = event["delta"]

    pipe = DataPipeline.__new__(DataPipeline)
    pipe.symbol = SYMBOL

    with caplog.at_level(logging.WARNING):
        pipe._validate_invariants(event, context="signal")

    assert event["delta"] == delta_antes
    assert not any(
        "INVARIANTE VIOLADA (Delta)" in rec.getMessage() for rec in caplog.records
    )
    assert not any(
        "INVARIANTE VIOLADA (Vol Sum)" in rec.getMessage() for rec in caplog.records
    )
