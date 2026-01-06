# tests/test_orderbook_helpers.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import pytest

import orderbook_analyzer as oba_mod
from orderbook_analyzer import OrderBookAnalyzer


class FakeTimeManager:
    def now_ms(self) -> int:
        return 1_700_000_000_000

    def build_time_index(self, epoch_ms: int, include_local: bool = False, timespec: str = "seconds") -> Dict[str, Any]:
        return {"timestamp_ny": None, "timestamp_utc": None}


# ========================
# Funções modulares
# ========================

def test_to_float_list_basic():
    data = [["100", "1.5"], [101.0, 2], ["bad", "data"], [102, -1]]
    result = oba_mod._to_float_list(data)
    assert result == [(100.0, 1.5), (101.0, 2.0)]  # ignora inválidos/negativos


def test_sum_depth_usd_top_n():
    levels = [(100.0, 1.0), (101.0, 2.0), (102.0, 3.0)]
    v1 = oba_mod._sum_depth_usd(levels, 1)   # 100 * 1
    v2 = oba_mod._sum_depth_usd(levels, 2)   # 100*1 + 101*2
    v5 = oba_mod._sum_depth_usd(levels, 5)   # todos

    assert v1 == pytest.approx(100.0)
    assert v2 == pytest.approx(302.0)
    assert v5 == pytest.approx(100.0 + 101.0 * 2 + 102.0 * 3)


def test_simulate_market_impact_buy_and_sell():
    levels = [(100.0, 1.0), (101.0, 1.0), (102.0, 1.0)]
    mid = 100.0

    # BUY 150 USD → consome 1 nível + metade do segundo
    res_buy = oba_mod._simulate_market_impact(levels, 150.0, "buy", mid)
    assert res_buy["levels"] == 2
    assert res_buy["usd"] == 150.0
    assert res_buy["move_usd"] >= 0.0
    assert res_buy["bps"] >= 0.0

    # SELL 150 USD → usa a mesma lógica mas move para baixo
    res_sell = oba_mod._simulate_market_impact(levels, 150.0, "sell", mid)
    assert res_sell["levels"] == 2
    assert res_sell["usd"] == 150.0
    assert res_sell["move_usd"] >= 0.0
    assert res_sell["bps"] >= 0.0


# ========================
# Métodos da classe
# ========================

@pytest.fixture
def analyzer() -> OrderBookAnalyzer:
    return OrderBookAnalyzer(symbol="BTCUSDT", time_manager=FakeTimeManager())


def test_detect_walls_simple(analyzer: OrderBookAnalyzer):
    """
    Garante que _detect_walls detecta uma parede quando um nível tem
    quantidade muito acima dos demais, com threshold configurado.
    """
    bids: List[Tuple[float, float]] = [
        (100.0, 1.0),
        (99.5, 1.2),
        (99.0, 50.0),   # wall bem maior
        (98.5, 1.1),
        (98.0, 0.9),
    ]

    analyzer.top_n = 5
    # Usa multiplicador 1.0 para não inflar demais o threshold
    analyzer.dynamic_thresholds["wall_threshold_multiplier"] = 1.0

    walls = analyzer._detect_walls(bids, side="bid")

    assert len(walls) >= 1
    assert any(w["price"] == 99.0 for w in walls)


def test_iceberg_reload_detects_increase(analyzer: OrderBookAnalyzer):
    """
    Aumenta significativamente a quantidade em um preço, de forma que
    delta >= 3.0 e qty_now >= tol * qty_prev, gerando score > 0.5.
    """
    prev = {
        "bids": [(100.0, 1.0), (99.5, 1.0)],
        "asks": [(100.5, 1.0), (101.0, 1.0)],
    }
    curr = {
        "bids": [(100.0, 4.5), (99.5, 1.0)],   # delta = 3.5 >= 3.0
        "asks": [(100.5, 1.0), (101.0, 1.0)],
    }

    iceberg, score = analyzer._iceberg_reload(prev, curr, tol=0.75)
    assert iceberg is True
    assert score > 0.5


def test_iceberg_reload_no_prev(analyzer: OrderBookAnalyzer):
    iceberg, score = analyzer._iceberg_reload(None, {"bids": [], "asks": []}, tol=0.75)
    assert iceberg is False
    assert score == 0.0


def test_detect_anomalies_spread_jump_and_depth_drop(analyzer: OrderBookAnalyzer):
    # prev snapshot com spread "normal" e boa liquidez
    prev = {
        "bids": [(100.0, 10.0)],
        "asks": [(100.5, 10.0)],
    }
    # curr com spread bem maior e liquidez muito menor
    curr = {
        "bids": [(100.0, 1.0)],
        "asks": [(101.5, 1.0)],
    }

    anomalies = analyzer._detect_anomalies(
        bids=curr["bids"],
        asks=curr["asks"],
        prev_snapshot=prev,
        spread_jump_bps=30.0,
        depth_drop_pct=60.0,
    )

    # esperamos ao menos um spread_jump e um depth_drop
    assert any("spread_jump" in a for a in anomalies)
    assert any("depth_drop" in a for a in anomalies)