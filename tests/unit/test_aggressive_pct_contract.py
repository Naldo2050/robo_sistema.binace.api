# tests/unit/test_aggressive_pct_contract.py — P1-B tabela verdade A-H.
#
# Contrato: pcts só existem quando observados (+ aggressive_status /
# aggressive_sample_count). 50/50, 100/0 e 0/100 reais preservados;
# missing/insuficiente nunca vira 50/50, balanced ou neutral.
# (Commitado VERMELHO antes do fix; verde depois.)

import pytest

from flow_analyzer.aggregates import analyze_passive_aggressive_flow
from flow_analyzer.core import FlowAnalyzer
from institutional.enricher import _build_passive_flow


class _Clock:
    def __init__(self):
        self._now = 1_700_000_000_000

    def now_ms(self):
        return self._now

    def build_time_index(self, ts, include_local=False, timespec="milliseconds"):
        return {"epoch_ms": ts}

    def format_timestamp(self, ts):
        return str(ts)

    def from_timestamp_ms(self, ts, tz=None):
        return None


def _feed(n_buy, qty_buy, n_sell, qty_sell):
    clock = _Clock()
    flow = FlowAnalyzer(time_manager=clock)
    trades = []
    i = 0
    buys = ["buy"] * n_buy
    sells = ["sell"] * n_sell
    seq = []
    ib = is_ = 0
    while ib < n_buy or is_ < n_sell:
        if ib < n_buy:
            seq.append(("buy", qty_buy))
            ib += 1
        if is_ < n_sell:
            seq.append(("sell", qty_sell))
            is_ += 1
    for side, qty in seq:
        t = {"p": 65000.0, "q": qty, "T": 1_700_000_000_000 + i * 1000,
             "m": side == "sell"}
        trades.append(t)
        clock._now = t["T"]
        flow.process_trade(t)
        i += 1
    if not trades:
        return flow.get_flow_metrics(
            reference_epoch_ms=1_700_000_000_000).get("order_flow", {})
    return flow.get_flow_metrics(
        reference_epoch_ms=trades[-1]["T"]).get("order_flow", {})


def _passive(order_flow):
    return _build_passive_flow({"fluxo_continuo": {"order_flow": order_flow},
                                "orderbook_data": {}})


def test_A_real_50_50_observed():
    of = _feed(10, 0.1, 10, 0.1)
    assert of["aggressive_status"] == "observed"
    assert of["aggressive_sample_count"] == 20
    assert of["aggressive_buy_pct"] == pytest.approx(50.0)
    assert of["aggressive_sell_pct"] == pytest.approx(50.0)
    agg = analyze_passive_aggressive_flow(
        {**of, "buy_volume_btc": 1.0, "sell_volume_btc": 1.0,
         "flow_imbalance": 0.0}, {})
    assert agg["aggressive"]["dominance"] == "balanced"  # legítimo aqui
    assert _passive(of) == {"passive_buy_pct": 50.0, "passive_sell_pct": 50.0}


def test_B_missing_key_is_not_50_50():
    of = {}
    assert of.get("aggressive_status", "unavailable") != "observed"
    assert "aggressive_buy_pct" not in of
    agg = analyze_passive_aggressive_flow(of, {})
    assert agg["aggressive"]["dominance"] != "balanced"
    assert agg["composite"]["signal"] != "neutral_balanced"
    assert _passive(of) == {}  # sem fonte observada => ausente


def test_B2_passive_independent_from_orderbook():
    """Passive via depth do OB independe do agressivo (documentado)."""
    from institutional.enricher import _build_passive_flow as _p
    ev = {"fluxo_continuo": {"order_flow": {}},
          "orderbook_data": {"bid_depth_usd": 300.0, "ask_depth_usd": 100.0}}
    assert _p(ev) == {"passive_buy_pct": 75.0, "passive_sell_pct": 25.0}


def test_C_none_is_not_50_50():
    of = {"aggressive_buy_pct": None, "aggressive_sell_pct": None}
    agg = analyze_passive_aggressive_flow(of, {})
    assert agg["aggressive"]["dominance"] != "balanced"
    assert agg["composite"]["signal"] != "neutral_balanced"
    assert _passive(of) == {}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_D_nonfinite_is_not_50_50(bad):
    of = {"aggressive_buy_pct": bad, "aggressive_sell_pct": bad}
    agg = analyze_passive_aggressive_flow(of, {})
    assert agg["aggressive"]["dominance"] != "balanced"
    assert agg["composite"]["signal"] != "neutral_balanced"
    assert _passive(of) == {}


def test_E_zero_trades():
    of = _feed(0, 0.1, 0, 0.1)
    assert of.get("aggressive_status") == "no_volume"
    assert of.get("aggressive_sample_count") == 0
    assert "aggressive_buy_pct" not in of
    agg = analyze_passive_aggressive_flow(of, {})
    assert agg["aggressive"]["dominance"] != "balanced"
    assert _passive(of) == {}


def test_F_insufficient_trades():
    of = _feed(3, 0.1, 0, 0.1)
    assert of.get("aggressive_status") == "insufficient"
    assert "aggressive_buy_pct" not in of
    agg = analyze_passive_aggressive_flow(of, {})
    assert agg["aggressive"]["dominance"] != "balanced"
    assert agg["composite"]["signal"] != "neutral_balanced"
    assert _passive(of) == {}


def test_G_real_100_0_preserved():
    of = _feed(10, 0.1, 0, 0.1)
    assert of["aggressive_status"] == "observed"
    assert of["aggressive_buy_pct"] == pytest.approx(100.0)
    assert of["aggressive_sell_pct"] == pytest.approx(0.0)
    passive = _passive(of)
    assert passive["passive_buy_pct"] == pytest.approx(0.0)
    assert passive["passive_sell_pct"] == pytest.approx(100.0)


def test_H_real_0_100_preserved():
    of = _feed(0, 0.1, 10, 0.1)
    assert of["aggressive_status"] == "observed"
    assert of["aggressive_buy_pct"] == pytest.approx(0.0)
    assert of["aggressive_sell_pct"] == pytest.approx(100.0)
    passive = _passive(of)
    assert passive["passive_buy_pct"] == pytest.approx(100.0)
    assert passive["passive_sell_pct"] == pytest.approx(0.0)


def test_payload_omits_unobserved():
    from market_orchestrator.ai.payload_builder_compact import (
        build_compact_payload,
    )

    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1_700_000_000_000,
             "ml_features": {},
             "fluxo_continuo": {"order_flow": {"net_flow_1m": 0,
                                               "flow_imbalance": 0.0}},
             "multi_tf": {}}
    compact = build_compact_payload(event)
    assert "ab" not in compact.get("flow", {})
