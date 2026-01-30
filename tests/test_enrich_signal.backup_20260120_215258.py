# tests/test_enrich_signal.py
from __future__ import annotations
# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fix_optimization import clean_event, simplify_historical_vp, remove_enriched_snapshot


import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datetime import datetime, timezone

import pytest

import market_orchestrator.market_orchestrator as mo
from data_handler import NY_TZ


# =======================
# FAKES / STUBS
# =======================

class FakeTimeManager:
    def __init__(self):
        self.tz_utc = timezone.utc

    def from_timestamp_ms(self, epoch_ms: int, tz) -> datetime:
        return datetime.fromtimestamp(epoch_ms / 1000.0, tz=tz)

    def now_utc_iso(self, timespec: str = "seconds") -> str:
        return datetime.now(self.tz_utc).isoformat(timespec=timespec)


@dataclass
class LevelsStub:
    last_event: Optional[Dict[str, Any]] = None

    def add_from_event(self, evt: Dict[str, Any]) -> None:
        self.last_event = evt


@dataclass
class EventSaverStub:
    saved_events: List[Dict[str, Any]] = field(default_factory=list)

    def save_event(self, evt: Dict[str, Any]) -> None:
        self.saved_events.append(evt)


@dataclass
# Otimizar ANALYSIS_TRIGGER antes de salvar
        if event.get("tipo_evento") == "ANALYSIS_TRIGGER":
            event = clean_event(event)
            event = simplify_historical_vp(event)
            event = remove_enriched_snapshot(event)
class EventBusStub:
    published: List[Dict[str, Any]] = field(default_factory=list)

    def publish(self, topic: str, evt: Dict[str, Any]) -> None:
        self.published.append({"topic": topic, "event": evt})


@dataclass
class FakeBot:
    symbol: str = "BTCUSDT"
    window_count: int = 5
    time_manager: Any = field(default_factory=FakeTimeManager)
    ny_tz = NY_TZ

    # usados em _enrich_signal
    orderbook_fetch_failures: int = 0
    volume_history: deque = field(default_factory=lambda: deque(maxlen=100))
    volatility_history: deque = field(default_factory=lambda: deque(maxlen=100))
    levels: Any = field(default_factory=LevelsStub)
    event_bus: Any = field(default_factory=EventBusStub)
    event_saver: Any = field(default_factory=EventSaverStub)

    _sent_triggers: set = field(default_factory=set)

    # travas internas
    _ai_pool_lock: threading.Lock = field(default_factory=threading.Lock)

    def _validate_flow_metrics(self, flow_metrics: Dict[str, Any], valid_window_data: List[Dict[str, Any]]) -> bool:
        # para os testes, consideramos sempre válidos
        return True

    def _build_institutional_event(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        # encapsula o sinal para que possamos inspecioná-lo
        return {"wrapped": signal.copy()}

    def _log_event(self, evt: Dict[str, Any]) -> None:
        # não precisamos de saída aqui; apenas evitar exceção
        pass


# =======================
# FIXTURES AUXILIARES
# =======================

@pytest.fixture
def fake_ob_event() -> Dict[str, Any]:
    return {
        "is_valid": True,
        "orderbook_data": {
            "bid_depth_usd": 1000.0,
            "ask_depth_usd": 800.0,
            "imbalance": 0.2,
            "mid": 100.0,
            "spread": 0.5,
            "spread_percent": 0.005,
        },
        "spread_metrics": {
            "mid": 100.0,
            "spread": 0.5,
            "spread_percent": 0.005,
            "bid_depth_usd": 1000.0,
            "ask_depth_usd": 800.0,
        },
        "order_book_depth": {"L5": {"bids": 1000.0, "asks": 800.0, "imbalance": 0.2}},
        "spread_analysis": {"current_spread_bps": 50.0},
        "depth_metrics": {
            "bid_liquidity_top5": 1000.0,
            "ask_liquidity_top5": 800.0,
            "depth_imbalance": 0.2,
        },
        "market_impact_buy": {
            "100k": {"move_usd": 1.0, "bps": 10.0},
            "1M": {"move_usd": 5.0, "bps": 50.0},
        },
        "market_impact_sell": {
            "100k": {"move_usd": 1.5, "bps": 15.0},
            "1M": {"move_usd": 6.0, "bps": 60.0},
        },
        "data_quality": {
            "is_valid": True,
            "data_source": "live",
            "age_seconds": 0.1,
        },
    }


@pytest.fixture
def base_signal() -> Dict[str, Any]:
    return {
        "tipo_evento": "ABSORÇÃO",
        "resultado_da_batalha": "Demanda Forte",
        "descricao": "Evento de teste",
        "delta": 10.0,
        "volume_total": 100.0,
        "volume_compra": 70.0,
        "volume_venda": 30.0,
        "ativo": "BTCUSDT",
    }


# =======================
# TESTES
# =======================

def test_enrich_signal_happy_path(monkeypatch, fake_ob_event, base_signal):
    """
    Caminho feliz: validator aceita o evento, _enrich_signal enriquece,
    publica no EventBus, salva no EventSaver e adiciona nos níveis.
    """
    bot = FakeBot()

    # validator: aceita e "limpa" o sinal
    def fake_validate_and_clean(sig: Dict[str, Any]) -> Dict[str, Any]:
        return {"validated": True}

    monkeypatch.setattr(
        mo.validator,
        "validate_and_clean",
        fake_validate_and_clean,
    )

    # evitar dependências de memória real
    monkeypatch.setattr(mo, "adicionar_memoria_evento", lambda *a, **k: None)
    monkeypatch.setattr(mo, "obter_memoria_eventos", lambda n=4: [])

    derivatives_context = {"dummy": True}
    flow_metrics = {"dummy_flow": True}
    macro_context = {"market_context": {}, "market_environment": {}}
    ml_payload = {"price_features": {}, "volume_features": {}, "microstructure": {}}
    enriched_snapshot = {"ohlc": {"close": 100.0}, "volume_total": 100.0, "delta_fechamento": 10.0}
    contextual_snapshot = {}
    valid_window_data = [{"p": 100.0, "q": 1.0, "T": 1234567890}]

    close_ms = 1_700_000_000_000

    # chama o método de instância com FakeBot como self
    mo.EnhancedMarketBot._enrich_signal(
        bot,
        base_signal,
        derivatives_context,
        flow_metrics,
        total_buy_volume=70.0,
        total_sell_volume=30.0,
        macro_context=macro_context,
        close_ms=close_ms,
        ml_payload=ml_payload,
        enriched_snapshot=enriched_snapshot,
        contextual_snapshot=contextual_snapshot,
        ob_event=fake_ob_event,
        valid_window_data=valid_window_data,
        support_resistance={},
        defense_zones_data={},
    )

    # Deve ter publicado no EventBus
    assert len(bot.event_bus.published) == 1
    pub = bot.event_bus.published[0]
    assert pub["topic"] == "signal"
    evt = pub["event"]
    assert evt["tipo_evento"] == "ABSORÇÃO"
    assert evt["janela_numero"] == bot.window_count
    assert "orderbook_data" in evt
    assert "orderbook_data_quality" in evt

    # Deve ter salvo um evento institucional
    assert len(bot.event_saver.saved_events) == 1
    inst_evt = bot.event_saver.saved_events[0]
    assert "wrapped" in inst_evt
    assert inst_evt["wrapped"]["tipo_evento"] == "ABSORÇÃO"

    # Levels deve ter recebido o sinal
    assert bot.levels.last_event is not None
    assert bot.levels.last_event["tipo_evento"] == "ABSORÇÃO"


def test_enrich_signal_discarded_when_validator_returns_empty(monkeypatch, fake_ob_event, base_signal):
    """
    Quando validator.validate_and_clean retorna dict vazio/falsy,
    o sinal deve ser descartado e nada é publicado/salvo.
    """
    bot = FakeBot()

    def fake_validate_and_clean(sig: Dict[str, Any]) -> Dict[str, Any]:
        return {}  # significa sinal inválido / descartado

    monkeypatch.setattr(
        mo.validator,
        "validate_and_clean",
        fake_validate_and_clean,
    )

    monkeypatch.setattr(mo, "adicionar_memoria_evento", lambda *a, **k: None)
    monkeypatch.setattr(mo, "obter_memoria_eventos", lambda n=4: [])

    derivatives_context = {}
    flow_metrics = {}
    macro_context = {"market_context": {}, "market_environment": {}}
    ml_payload = {}
    enriched_snapshot = {"ohlc": {"close": 100.0}, "volume_total": 100.0, "delta_fechamento": 10.0}
    contextual_snapshot = {}
    valid_window_data = [{"p": 100.0, "q": 1.0, "T": 1234567890}]
    close_ms = 1_700_000_000_000

    mo.EnhancedMarketBot._enrich_signal(
        bot,
        base_signal,
        derivatives_context,
        flow_metrics,
        total_buy_volume=70.0,
        total_sell_volume=30.0,
        macro_context=macro_context,
        close_ms=close_ms,
        ml_payload=ml_payload,
        enriched_snapshot=enriched_snapshot,
        contextual_snapshot=contextual_snapshot,
        ob_event=fake_ob_event,
        valid_window_data=valid_window_data,
        support_resistance={},
        defense_zones_data={},
    )

    # Nada deve ter sido publicado/salvo/registrado em níveis
    assert bot.event_bus.published == []
    assert bot.event_saver.saved_events == []
    assert bot.levels.last_event is None