# -*- coding: utf-8 -*-
"""Testes herméticos da identidade v2 de Zona (H1-B3-D).

Exercita o código real de EnhancedMarketBot._check_zone_touches com
orquestrador mínimo (__new__ + stubs): sem I/O, sem dados operacionais,
sem EventSaver/EventStore reais.
"""

import copy
import logging
from datetime import timezone

import pytest

import market_orchestrator.market_orchestrator as mo_mod
from common.identity_contract import zone_v2_event_id
from market_analysis.levels_registry import LevelZone
from market_orchestrator.market_orchestrator import EnhancedMarketBot

SYMBOL = "BTCUSDT"
WINDOW = "1700000060000"
BASE_TS = 1700000055000
LAST_TOUCHED_1 = "2026-10-24T11:59:59.123+00:00"
LAST_TOUCHED_2 = "2026-10-24T12:10:05.987+00:00"
DET_ID = "a" * 64


class _FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


class _FakeSaver:
    def __init__(self):
        self.saved = []

    def save_event(self, event):
        self.saved.append(event)


class _FakeLevels:
    def __init__(self, zones):
        self._zones = list(zones)

    def check_price(self, price):
        return list(self._zones)


class _FakeTimeManager:
    def now_utc_iso(self, timespec="milliseconds"):
        return "2026-10-24T12:00:00+00:00"


def _zone(
    kind="POC",
    timeframe="daily",
    anchor=100.0,
    low=95.5,
    high=104.5,
    last_touched=LAST_TOUCHED_1,
    touch_count=3,
):
    return LevelZone(
        id="zone-local-uuid",
        kind=kind,
        timeframe=timeframe,
        anchor_price=anchor,
        low=low,
        high=high,
        score=60.0,
        confluence=["DAILY_POC"],
        created_at="2026-10-01T00:00:00+00:00",
        last_touched=last_touched,
        touch_count=touch_count,
        notes="",
    )


def _make_orchestrator(monkeypatch, zones, window_count=7):
    orch = EnhancedMarketBot.__new__(EnhancedMarketBot)
    orch.window_count = window_count
    orch.time_manager = _FakeTimeManager()
    orch.ny_tz = timezone.utc
    orch.event_bus = _FakeBus()
    orch.event_saver = _FakeSaver()
    orch.levels = _FakeLevels(zones)
    monkeypatch.setattr(mo_mod, "adicionar_memoria_evento", lambda event: None)
    monkeypatch.setattr(
        mo_mod, "calcular_probabilidade_historica", lambda event: {"long_prob": 0.5}
    )
    return orch


def _base_absorcao():
    return {
        "tipo_evento": "Absorção",
        "ativo": SYMBOL,
        "is_signal": True,
        "resultado_da_batalha": "Absorção de Compra",
        "window_id": WINDOW,
        "window_open_ms": BASE_TS,
        "preco_fechamento": 100.0,
        "event_id": DET_ID,
        "delta": 2.9,
        "volume_total": 3.1,
        "raw_event": {"nested": {"marker": "intocado"}},
    }


def _base_exaustao():
    base = _base_absorcao()
    base.update(
        {"tipo_evento": "Exaustão", "resultado_da_batalha": "Exaustão de Compra"}
    )
    return base


def _base_trigger():
    return {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": SYMBOL,
        "is_signal": True,
        "features_window_id": WINDOW,
        "epoch_ms": int(WINDOW),
        "preco_fechamento": 100.0,
    }


def _expected_zone_id(zone, window_key=WINDOW):
    return zone_v2_event_id(
        SYMBOL,
        window_key,
        zone.kind,
        zone.timeframe,
        zone.anchor_price,
        zone.low,
        zone.high,
        zone.last_touched,
    )


_ENRICHED = {"ohlc": {"close": 100.0}}


# =============================================================================
# IDENTIDADE PRÓPRIA E BASE
# =============================================================================
@pytest.mark.parametrize("base_factory", [_base_absorcao, _base_exaustao])
def test_zona_recebe_identidade_propria_e_preserva_base_event_id(
    monkeypatch, base_factory
):
    base = base_factory()
    base_snapshot = copy.deepcopy(base)
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    orch._check_zone_touches(_ENRICHED, [base])

    assert len(orch.event_saver.saved) == 1
    event = orch.event_saver.saved[0]
    assert event["event_id"] == _expected_zone_id(zone)
    assert event["event_id"] != base["event_id"]
    assert event["base_event_id"] == base["event_id"]
    assert event["identity_version"] == "zone-v2"
    assert base == base_snapshot
    assert len(orch.event_bus.published) == 1
    assert orch.event_bus.published[0][0] == "zone_touch"


def test_zona_de_trigger_anonimo_sem_base_event_id(monkeypatch):
    base = _base_trigger()
    base_snapshot = copy.deepcopy(base)
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    orch._check_zone_touches(_ENRICHED, [base])

    event = orch.event_saver.saved[0]
    assert event["event_id"] == _expected_zone_id(zone)
    assert "base_event_id" not in event
    assert base == base_snapshot


def test_zona_nao_preserva_uuid8_tecnico_como_base(monkeypatch):
    base = _base_trigger()
    base["event_id"] = "deadbeef"
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    orch._check_zone_touches(_ENRICHED, [base])

    event = orch.event_saver.saved[0]
    assert event["event_id"] == _expected_zone_id(zone)
    assert event["event_id"] != "deadbeef"
    assert "base_event_id" not in event


def test_zona_de_base_sem_event_id(monkeypatch):
    base = _base_absorcao()
    del base["event_id"]
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    orch._check_zone_touches(_ENRICHED, [base])

    event = orch.event_saver.saved[0]
    assert event["event_id"] == _expected_zone_id(zone)
    assert "base_event_id" not in event


def test_zona_window_key_fallback_epoch_ms(monkeypatch):
    base = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": SYMBOL,
        "epoch_ms": int(WINDOW),
    }
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    orch._check_zone_touches(_ENRICHED, [base])

    event = orch.event_saver.saved[0]
    assert event["event_id"] == _expected_zone_id(zone, window_key=WINDOW)


# =============================================================================
# REJEIÇÕES COM DIAGNÓSTICO EXPLÍCITO
# =============================================================================
def test_zona_sem_symbol_rejeitada_com_diagnostico(monkeypatch, caplog):
    base = {"epoch_ms": int(WINDOW)}
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    with caplog.at_level(logging.ERROR):
        orch._check_zone_touches(_ENRICHED, [base])

    assert orch.event_saver.saved == []
    assert orch.event_bus.published == []
    assert any("identidade rejeitada" in rec.getMessage() for rec in caplog.records)
    assert any("symbol" in rec.getMessage() for rec in caplog.records)


def test_zona_sem_window_key_rejeitada_com_diagnostico(monkeypatch, caplog):
    base = {"ativo": SYMBOL}
    zone = _zone()
    orch = _make_orchestrator(monkeypatch, [zone])

    with caplog.at_level(logging.ERROR):
        orch._check_zone_touches(_ENRICHED, [base])

    assert orch.event_saver.saved == []
    assert any("identidade rejeitada" in rec.getMessage() for rec in caplog.records)
    assert any("window_key" in rec.getMessage() for rec in caplog.records)


def test_zona_componente_nao_finito_rejeitado(monkeypatch, caplog):
    base = _base_absorcao()
    zone = _zone(anchor=float("nan"))
    orch = _make_orchestrator(monkeypatch, [zone])

    with caplog.at_level(logging.ERROR):
        orch._check_zone_touches(_ENRICHED, [base])

    assert orch.event_saver.saved == []
    assert orch.event_bus.published == []
    assert any("identidade rejeitada" in rec.getMessage() for rec in caplog.records)


# =============================================================================
# COEXISTÊNCIA, TOQUES REPETIDOS E DETERMINISMO
# =============================================================================
def test_multiplas_zonas_por_janela_coexistem(monkeypatch):
    zones = [
        _zone(kind="POC", timeframe="daily", anchor=100.0, low=95.5, high=104.5),
        _zone(
            kind="VAL",
            timeframe="intraday",
            anchor=100.0,
            low=98.0,
            high=102.0,
            last_touched=LAST_TOUCHED_2,
        ),
    ]
    orch = _make_orchestrator(monkeypatch, zones)

    orch._check_zone_touches(_ENRICHED, [_base_absorcao()])

    assert len(orch.event_saver.saved) == 2
    ids = {event["event_id"] for event in orch.event_saver.saved}
    assert ids == {_expected_zone_id(zones[0]), _expected_zone_id(zones[1])}
    assert len(orch.event_bus.published) == 2


def test_mesmo_snapshot_mesmo_id_e_toque_repetido_id_novo(monkeypatch):
    orch1 = _make_orchestrator(monkeypatch, [_zone(last_touched=LAST_TOUCHED_1)])
    orch1._check_zone_touches(_ENRICHED, [_base_absorcao()])

    orch2 = _make_orchestrator(monkeypatch, [_zone(last_touched=LAST_TOUCHED_1)])
    orch2._check_zone_touches(_ENRICHED, [_base_absorcao()])

    assert (
        orch1.event_saver.saved[0]["event_id"]
        == orch2.event_saver.saved[0]["event_id"]
    )

    orch3 = _make_orchestrator(monkeypatch, [_zone(last_touched=LAST_TOUCHED_2)])
    orch3._check_zone_touches(_ENRICHED, [_base_absorcao()])

    assert (
        orch3.event_saver.saved[0]["event_id"]
        != orch1.event_saver.saved[0]["event_id"]
    )


def test_touch_count_e_janela_numero_fora_da_identidade(monkeypatch):
    orch_a = _make_orchestrator(
        monkeypatch, [_zone(touch_count=3)], window_count=7
    )
    orch_b = _make_orchestrator(
        monkeypatch, [_zone(touch_count=99)], window_count=8
    )

    orch_a._check_zone_touches(_ENRICHED, [_base_absorcao()])
    orch_b._check_zone_touches(_ENRICHED, [_base_absorcao()])

    ev_a = orch_a.event_saver.saved[0]
    ev_b = orch_b.event_saver.saved[0]

    assert ev_a["event_id"] == ev_b["event_id"]
    assert ev_a["janela_numero"] == 7
    assert ev_b["janela_numero"] == 8
    assert ev_a["zone_context"]["touch_count"] == 3
    assert ev_b["zone_context"]["touch_count"] == 99
