# tests/test_institutional_alerts.py
from __future__ import annotations
# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing.fix_optimization import clean_event, simplify_historical_vp, remove_enriched_snapshot


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


# =======================
# TESTES COM DETECTOR REAL (alert_engine sem fake)
# =======================

EXPANSION_RECENT_VOLS = [1e-5] * 9 + [1.0, 2.0]  # current = 2.0 -> EXPANSION
SQUEEZE_RECENT_VOLS = [1.0] * 9 + [1e-5]  # current = 1e-5 -> SQUEEZE


def _bot_com_volatilidade(history_vols, history_volumes=None):
    """FakeBotAlerts com histórico de volatilidade e volume controlados."""
    bot = FakeBotAlerts()
    bot.volatility_history = deque(list(history_vols), maxlen=100)
    if history_volumes is not None:
        bot.volume_history = deque(list(history_volumes), maxlen=100)
    return bot


def _stub_alert_deps(monkeypatch, real_generate_alerts=None):
    """Stubs de dependências de _process_institutional_alerts."""
    if real_generate_alerts is not None:
        # DETECTOR REAL (caminho real de trading/alert_engine)
        monkeypatch.setattr(mo, "generate_alerts", real_generate_alerts)
    else:
        monkeypatch.setattr(mo, "generate_alerts", lambda **_: [])
    monkeypatch.setattr(
        mo, "detect_support_resistance",
        lambda price_series, num_levels=3: {
            "immediate_support": [],
            "immediate_resistance": [],
        },
    )
    monkeypatch.setattr(mo, "defense_zones", None)


def test_process_institutional_alerts_expansion_real_detector(
    monkeypatch, enriched_base
):
    """
    Caminho REAL: generate_alerts/trading.alert_engine com volatilidade alta
    deve produzir alerta EXPANDED -> VOLATILITY_EXPANSION salvo no evento.
    """
    from trading.alert_engine import generate_alerts as real_generate_alerts

    bot = _bot_com_volatilidade(
        EXPANSION_RECENT_VOLS, history_volumes=[5000.0, 6000.0]
    )
    _stub_alert_deps(monkeypatch, real_generate_alerts=real_generate_alerts)

    mo.EnhancedMarketBot._process_institutional_alerts(
        bot, enriched_base, PipelineStub()
    )

    assert len(bot.event_saver.saved_events) == 1
    alert = bot.event_saver.saved_events[0]["wrapped"]
    assert alert["tipo_evento"] == "Alerta"
    assert alert["resultado_da_batalha"] == "VOLATILITY_EXPANSION"
    assert alert["context"]["volatility"] == pytest.approx(2.0)

    # cooldown registrado com a chave correta e separada do squeeze
    assert "VOLATILITY_EXPANSION" in bot._last_alert_ts
    assert "VOLATILITY_SQUEEZE" not in bot._last_alert_ts


def test_process_institutional_alerts_squeeze_real_detector(
    monkeypatch, enriched_base
):
    """
    Caminho REAL com volatilidade comprimida -> VOLATILITY_SQUEEZE.
    """
    from trading.alert_engine import generate_alerts as real_generate_alerts

    bot = _bot_com_volatilidade(
        SQUEEZE_RECENT_VOLS, history_volumes=[5000.0, 6000.0]
    )
    _stub_alert_deps(monkeypatch, real_generate_alerts=real_generate_alerts)

    mo.EnhancedMarketBot._process_institutional_alerts(
        bot, enriched_base, PipelineStub()
    )

    assert len(bot.event_saver.saved_events) == 1
    alert = bot.event_saver.saved_events[0]["wrapped"]
    assert alert["resultado_da_batalha"] == "VOLATILITY_SQUEEZE"

    assert "VOLATILITY_SQUEEZE" in bot._last_alert_ts
    assert "VOLATILITY_EXPANSION" not in bot._last_alert_ts


def test_process_institutional_alerts_expansion_and_squeeze_cooldowns_separate(
    monkeypatch, enriched_base
):
    """
    Cooldown separado por tipo: squeeze não bloqueia expansion subsequente
    e vice-versa (chave = alert.get("type") no orquestrador).
    """
    from trading.alert_engine import generate_alerts as real_generate_alerts

    bot = _bot_com_volatilidade(
        EXPANSION_RECENT_VOLS, history_volumes=[5000.0, 6000.0]
    )
    bot._alert_cooldown_sec = 999.0
    _stub_alert_deps(monkeypatch, real_generate_alerts=real_generate_alerts)

    # 1) Squeeze primeiro (grava chave VOLATILITY_SQUEEZE)
    bot.volatility_history = deque(list(SQUEEZE_RECENT_VOLS), maxlen=100)
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot, enriched_base, PipelineStub()
    )
    assert len(bot.event_saver.saved_events) == 1

    # 2) Expansion logo em seguida: chave DIFERENTE -> não é bloqueado
    bot.volatility_history = deque(list(EXPANSION_RECENT_VOLS), maxlen=100)
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot, enriched_base, PipelineStub()
    )
    assert len(bot.event_saver.saved_events) == 2

    # 3) Expansion repetido dentro do cooldown: bloqueado pela própria chave
    mo.EnhancedMarketBot._process_institutional_alerts(
        bot, enriched_base, PipelineStub()
    )
    assert len(bot.event_saver.saved_events) == 2

    results = [
        e["wrapped"]["resultado_da_batalha"]
        for e in bot.event_saver.saved_events
    ]
    assert results == ["VOLATILITY_SQUEEZE", "VOLATILITY_EXPANSION"]
    assert set(bot._last_alert_ts.keys()) == {
        "VOLATILITY_SQUEEZE",
        "VOLATILITY_EXPANSION",
    }