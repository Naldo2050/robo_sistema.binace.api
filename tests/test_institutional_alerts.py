# tests/test_institutional_alerts.py
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datetime import datetime, timezone

import pytest

import market_orchestrator.market_orchestrator as mo


# =======================
# FAKES / STUBS
# =======================

class FakeTimeManager:
    def __init__(self):
        self.tz_utc = timezone.utc

    def now_utc_iso(self, timespec: str = "seconds") -> str:
        return datetime.now(self.tz_utc).isoformat(timespec=timespec)


@dataclass
class EventSaverStub:
    saved_events: List[Dict[str, Any]] = field(default_factory=list)

    def save_event(self, evt: Dict[str, Any]) -> None:
        self.saved_events.append(evt)


@dataclass
class FakeBotAlerts:
    symbol: str = "BTCUSDT"
    window_count: int = 7
    time_manager: Any = field(default_factory=FakeTimeManager)

    volume_history: deque = field(default_factory=lambda: deque([100.0, 200.0], maxlen=100))
    volatility_history: deque = field(default_factory=lambda: deque([1.5, 2.0], maxlen=100))

    _alert_cooldown_sec: float = 60.0
    _last_alert_ts: Dict[str, float] = field(default_factory=dict)

    event_saver: Any = field(default_factory=EventSaverStub)

    def _build_institutional_event(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        # encapsula o sinal para permitir inspeção
        return {"wrapped": signal.copy()}


@dataclass
class PipelineStub:
    """
    Stub mínimo de DataPipeline para _process_institutional_alerts:
    precisamos apenas do atributo df["p"] ser indexável.
    """
    df: Any = None

    def __init__(self):
        # Estrutura mínima para pipeline.df["p"]
        class _DF:
            def __getitem__(self, key):
                if key == "p":
                    # sequência fictícia de preços
                    return [100.0, 101.0, 102.0]
                raise KeyError(key)
        self.df = _DF()


# =======================
# FIXTURES
# =======================

@pytest.fixture
def enriched_base() -> Dict[str, Any]:
    return {
        "ohlc": {"close": 101.0},
        "volume_total": 500.0,
    }


# =======================
# TESTES
# =======================

def test_process_institutional_alerts_happy_path(monkeypatch, enriched_base):
    """
    Caminho feliz: generate_alerts retorna um alerta e
    _process_institutional_alerts deve salvar um evento institucional.
    """
    bot = FakeBotAlerts()

    # Stubs de módulos opcionais
    def fake_detect_support_resistance(price_series, num_levels=3):
        return {
            "immediate_support": [99.0],
            "immediate_resistance": [105.0],
        }

    def fake_defense_zones(sr):
        return {"zones": ["zone1"]}

    # generate_alerts retorna uma lista com 1 alerta
    def fake_generate_alerts(
        price,
        support_resistance,
        current_volume,
        average_volume,
        current_volatility,
        recent_volatilities,
        volume_threshold,
        tolerance_pct,
    ):
        return [
            {
                "type": "VOLATILITY_EXPANSION",
                "severity": "HIGH",
                "probability": 0.8,
                "action": "watch",
                "level": price,
                "threshold_exceeded": 3.5,
            }
        ]

    monkeypatch.setattr(mo, "detect_support_resistance", fake_detect_support_resistance)
    monkeypatch.setattr(mo, "defense_zones", fake_defense_zones)
    monkeypatch.setattr(mo, "generate_alerts", fake_generate_alerts)

    pipeline = PipelineStub()

    # chamada estática com FakeBotAlerts como self
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot,
        enriched_base,
        pipeline,
    )

    # Deve ter salvo exatamente 1 evento institucional
    assert len(bot.event_saver.saved_events) == 1
    inst_evt = bot.event_saver.saved_events[0]
    assert "wrapped" in inst_evt
    alert = inst_evt["wrapped"]

    assert alert["tipo_evento"] == "Alerta"
    assert alert["resultado_da_batalha"] == "VOLATILITY_EXPANSION"
    assert alert["context"]["price"] == enriched_base["ohlc"]["close"]
    assert alert["context"]["volume"] == enriched_base["volume_total"]
    assert alert["janela_numero"] == bot.window_count
    assert "support_resistance" in alert
    assert "defense_zones" in alert

    # cooldown atualizado
    assert "VOLATILITY_EXPANSION" in bot._last_alert_ts


def test_process_institutional_alerts_respects_cooldown(monkeypatch, enriched_base):
    """
    Se o mesmo tipo de alerta for gerado novamente dentro do cooldown,
    _process_institutional_alerts não deve salvar um novo evento.
    """
    bot = FakeBotAlerts()
    bot._alert_cooldown_sec = 999.0  # cooldown bem alto

    def fake_detect_support_resistance(price_series, num_levels=3):
        return {"immediate_support": [], "immediate_resistance": []}

    def fake_generate_alerts(
        price,
        support_resistance,
        current_volume,
        average_volume,
        current_volatility,
        recent_volatilities,
        volume_threshold,
        tolerance_pct,
    ):
        return [
            {
                "type": "SUPPLY_EXHAUSTION",
                "severity": "HIGH",
                "probability": 0.9,
                "action": "sell",
            }
        ]

    monkeypatch.setattr(mo, "detect_support_resistance", fake_detect_support_resistance)
    monkeypatch.setattr(mo, "defense_zones", None)  # sem defense_zones
    monkeypatch.setattr(mo, "generate_alerts", fake_generate_alerts)

    pipeline = PipelineStub()

    # Primeira chamada: deve registrar alerta
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot,
        enriched_base,
        pipeline,
    )
    assert len(bot.event_saver.saved_events) == 1

    # Segunda chamada logo em seguida: devido ao cooldown, não deve salvar outro
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot,
        enriched_base,
        pipeline,
    )

    assert len(bot.event_saver.saved_events) == 1  # ainda apenas 1 evento