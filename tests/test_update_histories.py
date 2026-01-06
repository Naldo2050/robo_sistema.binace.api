# tests/test_update_histories.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

import market_orchestrator.market_orchestrator as mo


# =======================
# FAKES / STUBS
# =======================

@dataclass
class FlowAnalyzerStub:
    last_ctx: Dict[str, Any] = field(default_factory=dict)

    def update_volatility_context(self, atr_price, price_volatility):
        self.last_ctx = {
            "atr_price": atr_price,
            "price_volatility": price_volatility,
        }


@dataclass
class FakeBotHist:
    """
    Fake mínimo para testar EnhancedMarketBot._update_histories.
    """
    volume_history: deque = field(default_factory=lambda: deque(maxlen=10))
    delta_history: deque = field(default_factory=lambda: deque(maxlen=10))
    close_price_history: deque = field(default_factory=lambda: deque(maxlen=10))
    volatility_history: deque = field(default_factory=lambda: deque(maxlen=10))
    pattern_ohlc_history: deque = field(default_factory=lambda: deque(maxlen=10))

    flow_analyzer: Any = field(default_factory=FlowAnalyzerStub)


# =======================
# FIXTURES DE DADOS
# =======================

@pytest.fixture
def enriched_base() -> Dict[str, Any]:
    return {
        "volume_total": 123.0,
        "delta_fechamento": 10.5,
        "ohlc": {
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
        },
    }


@pytest.fixture
def ml_payload_full() -> Dict[str, Any]:
    return {
        "price_features": {
            "returns_5": 0.01,
            "volatility_5": 0.02,
        },
        "volume_features": {},
        "microstructure": {},
    }


@pytest.fixture
def ml_payload_vol1_only() -> Dict[str, Any]:
    return {
        "price_features": {
            "volatility_1": 0.03,
        },
        "volume_features": {},
        "microstructure": {},
    }


# =======================
# TESTES
# =======================

def test_update_histories_appends_basic_values(enriched_base, ml_payload_full):
    """
    Garante que _update_histories:
    - adiciona volume e delta nos históricos,
    - adiciona o fechamento em close_price_history,
    - adiciona OHLC em pattern_ohlc_history.
    """
    bot = FakeBotHist()

    assert len(bot.volume_history) == 0
    assert len(bot.delta_history) == 0
    assert len(bot.close_price_history) == 0
    assert len(bot.pattern_ohlc_history) == 0

    mo.EnhancedMarketBot._update_histories(
        bot,
        enriched_base,
        ml_payload_full,
    )

    # volume e delta foram registrados
    assert list(bot.volume_history) == [enriched_base["volume_total"]]
    assert list(bot.delta_history) == [enriched_base["delta_fechamento"]]

    # fechamento foi colocado em close_price_history
    assert list(bot.close_price_history) == [enriched_base["ohlc"]["close"]]

    # pattern_ohlc_history recebeu um dict com high/low/close
    assert len(bot.pattern_ohlc_history) == 1
    ohlc = bot.pattern_ohlc_history[0]
    assert ohlc["high"] == enriched_base["ohlc"]["high"]
    assert ohlc["low"] == enriched_base["ohlc"]["low"]
    assert ohlc["close"] == enriched_base["ohlc"]["close"]


def test_update_histories_uses_volatility_5_first(enriched_base, ml_payload_full):
    """
    Quando price_features tem volatility_5, _update_histories deve:
    - adicionar o valor em volatility_history,
    - chamar flow_analyzer.update_volatility_context com price_volatility = vol * close.
    """
    bot = FakeBotHist()

    mo.EnhancedMarketBot._update_histories(
        bot,
        enriched_base,
        ml_payload_full,
    )

    assert len(bot.volatility_history) == 1
    vol5 = ml_payload_full["price_features"]["volatility_5"]
    assert bot.volatility_history[0] == pytest.approx(vol5)

    close = enriched_base["ohlc"]["close"]
    expected_price_vol = vol5 * close

    assert bot.flow_analyzer.last_ctx["atr_price"] is None
    assert bot.flow_analyzer.last_ctx["price_volatility"] == pytest.approx(expected_price_vol)


def test_update_histories_falls_back_to_volatility_1(enriched_base, ml_payload_vol1_only):
    """
    Se não houver volatility_5, mas houver volatility_1, _update_histories deve usar volatility_1.
    """
    bot = FakeBotHist()

    mo.EnhancedMarketBot._update_histories(
        bot,
        enriched_base,
        ml_payload_vol1_only,
    )

    assert len(bot.volatility_history) == 1
    vol1 = ml_payload_vol1_only["price_features"]["volatility_1"]
    assert bot.volatility_history[0] == pytest.approx(vol1)

    close = enriched_base["ohlc"]["close"]
    expected_price_vol = vol1 * close

    assert bot.flow_analyzer.last_ctx["price_volatility"] == pytest.approx(expected_price_vol)