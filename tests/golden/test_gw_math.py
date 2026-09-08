# tests/golden/test_gw_math.py — L1: FlowAnalyzer + WindowState reais.
#
# Golden math GW1/GW2/GW3 (+ micro sell_only). Sem rede (harness), sem IA.
# P1 em quarentena: RSI/returns/aggressive/prob_up NUNCA congelados aqui.

import pandas as pd
import pytest

from core.window_state import WindowState
from flow_analyzer.core import FlowAnalyzer
from market_orchestrator.windows.window_processor import (
    _populate_window_state,
    _populate_window_state_indicators,
)

from .conftest import (
    FakeClock,
    assert_contract_version,
    assert_no_nonfinite,
    expected_volumes,
    gen_trades,
    load_fixture,
)


def _feed(spec, clock=None):
    clock = clock or FakeClock()
    flow = FlowAnalyzer(time_manager=clock)  # defaults prod (heatmap 2000)
    trades = gen_trades(spec)
    for t in trades:
        clock._now = t["T"]
        flow.process_trade(t)
    return flow, trades, clock


def _metrics(flow, close_ms):
    m = flow.get_flow_metrics(reference_epoch_ms=close_ms)
    assert "order_flow" in m, "shape aninhado order_flow ausente (regressao top-level?)"
    assert "buy_sell_ratio" in m["order_flow"]
    return m


def _window_from_real(spec, flow_metrics, clock):
    ws = WindowState(symbol=spec["symbol"], window_number=1)
    ob_cfg = spec["orderbook"]
    ob = {"bid_depth_usd": ob_cfg["bid_depth_usd"],
          "ask_depth_usd": ob_cfg["ask_depth_usd"],
          "imbalance": ob_cfg["imbalance"],
          "spread_bps": ob_cfg["spread_bps"],
          "is_valid": ob_cfg["is_valid"],
          "data_quality": {"data_source": ob_cfg["data_source"]}}
    deriv = spec.get("derivatives", {})
    macro = {"external": spec.get("macro_external", {}),
             "derivatives": deriv}
    buy, sell = expected_volumes(spec)
    trades = gen_trades(spec)
    enriched = {"ohlc": {"open": trades[0]["p"], "high": trades[0]["p"],
                         "low": trades[0]["p"], "close": trades[-1]["p"],
                         "vwap": trades[0]["p"]},
                "volume_total": buy + sell, "num_trades": len(trades)}
    _populate_window_state(ws, enriched, flow_metrics, ob, macro, buy, sell)
    # plumbing de indicadores (valores de passagem; P1 quarentena: sem asserts)
    _populate_window_state_indicators(
        ws, {"rsi": 60.0, "bb_upper": 1.0, "bb_lower": 1.0,
             "bb_width": 0.001, "atr": 1.0, "realized_vol": 0.03},
        {"1d": {"realized_vol": 0.03}})
    return ws


def test_gw1_balanced_math(frozen_state):
    spec = load_fixture("gw1_balanced.json")
    flow, trades, clock = _feed(spec)
    m = _metrics(flow, trades[-1]["T"])
    of = m["order_flow"]
    buy, sell = expected_volumes(spec)
    assert buy == pytest.approx(1.2) and sell == pytest.approx(1.2)
    assert of["buy_sell_ratio"]["buy_sell_ratio"] == pytest.approx(1.0)
    assert of["buy_sell_ratio"]["ratio_state"] == "two_sided"
    assert of["flow_imbalance"] == pytest.approx(0.0, abs=0.05)
    assert of["buy_sell_ratio"]["pressure"] == "NEUTRAL"
    assert of["buy_sell_ratio"]["buy_sell_ratio"] != 99.0  # sentinela proibida
    # P1-B: 50/50 observado carrega status + amostra (balanced permitido)
    assert of["aggressive_status"] == "observed"
    assert of["aggressive_sample_count"] == 120
    assert of["aggressive_buy_pct"] == pytest.approx(50.0)
    assert_no_nonfinite(of)
    # heatmap: default produtivo rolling_2000, não instância de teste
    hm = m["liquidity_heatmap"]
    assert hm["scope_type"] == "rolling_trades"
    assert hm["scope_size"] == 2000
    assert hm["clusters"]
    # WindowState do shape real
    ws = _window_from_real(spec, m, clock)
    assert ws.flow.buy_sell_ratio == pytest.approx(1.0)
    assert ws.flow.flow_imbalance == pytest.approx(0.0, abs=0.05)
    assert ws.flow.pressure_label == "NEUTRAL"
    assert ws.derivatives.btc_long_short_ratio == pytest.approx(1.02)
    assert ws.flow.validate() == []
    assert ws.derivatives.validate() == []
    assert ws._writers["flow"] and ws._writers["derivatives"]
    assert ws.validate_all() == []


def test_gw1_deterministic_double_run(frozen_state):
    spec = load_fixture("gw1_balanced.json")

    def _once():
        flow, trades, _ = _feed(spec)
        of = _metrics(flow, trades[-1]["T"])["order_flow"]
        hm = flow.get_flow_metrics(reference_epoch_ms=trades[-1]["T"])["liquidity_heatmap"]
        return {"ratio": of["buy_sell_ratio"]["buy_sell_ratio"],
                "imb": of["flow_imbalance"],
                "pressure": of["buy_sell_ratio"]["pressure"],
                "state": of["buy_sell_ratio"]["ratio_state"],
                "clusters": len(hm["clusters"]),
                "scope": hm["scope_size"]}

    a, b = _once(), _once()
    assert a == b


def test_gw2_buy_pressure_math(frozen_state):
    spec = load_fixture("gw2_buy_pressure.json")
    flow, trades, clock = _feed(spec)
    of = _metrics(flow, trades[-1]["T"])["order_flow"]
    assert of["buy_sell_ratio"]["buy_sell_ratio"] == pytest.approx(4.0)
    assert of["buy_sell_ratio"]["ratio_state"] == "two_sided"
    assert of["buy_sell_ratio"]["pressure"] == "STRONG_BUY"
    assert of["flow_imbalance"] == pytest.approx(0.6, abs=0.05)
    assert of["buy_sell_ratio"]["buy_sell_ratio"] != 99.0
    ws = _window_from_real(spec, _metrics(flow, trades[-1]["T"]), clock)
    assert ws.flow.buy_sell_ratio == pytest.approx(4.0)
    assert ws.flow.pressure_label == "STRONG_BUY"


def test_gw4b_flow_missing_contract(frozen_state):
    """P1-B no Golden: sem trades/insuficiente => status explícito, sem pcts.
    Mudança intencional de contrato (não 'ficar verde')."""
    from flow_analyzer.core import FlowAnalyzer as _FA

    clock = FakeClock()
    flow = _FA(time_manager=clock)
    m = flow.get_flow_metrics(reference_epoch_ms=1788880000000)["order_flow"]
    assert m.get("aggressive_status") == "no_volume"
    assert m.get("aggressive_sample_count") == 0
    assert "aggressive_buy_pct" not in m
    assert "aggressive_sell_pct" not in m


def test_gw2b_sell_only_legit_zero(frozen_state):
    """Zero observado (sell_only) != missing: ratio 0.0 legítimo preservado."""
    clock = FakeClock()
    flow = FlowAnalyzer(time_manager=clock)
    base = 1788880000000
    for i in range(5):
        t = {"p": 66875.0, "q": 0.1, "T": base + i * 1000, "m": True}
        clock._now = t["T"]
        flow.process_trade(t)
    of = _metrics(flow, base + 4000)["order_flow"]
    assert of["buy_sell_ratio"]["buy_sell_ratio"] == pytest.approx(0.0)
    assert of["buy_sell_ratio"]["ratio_state"] == "sell_only"
    assert of["flow_imbalance"] == pytest.approx(-1.0)  # extremo OBSERVADO
    # P1-B: 0/100 real também carrega status observed (zero preservado)
    assert of["aggressive_status"] == "observed"
    assert of["aggressive_sample_count"] == 5
    assert of["aggressive_buy_pct"] == pytest.approx(0.0)
    assert of["aggressive_sell_pct"] == pytest.approx(100.0)


def test_gw3_sell_pressure_math(frozen_state):
    spec = load_fixture("gw3_sell_absorption.json")
    flow, trades, clock = _feed(spec)
    of = _metrics(flow, trades[-1]["T"])["order_flow"]
    assert of["buy_sell_ratio"]["buy_sell_ratio"] == pytest.approx(0.25)
    assert of["buy_sell_ratio"]["pressure"] == "STRONG_SELL"
    assert of["flow_imbalance"] == pytest.approx(-0.6, abs=0.05)
    buy, sell = expected_volumes(spec)
    # order_flow usa notional USD (qty * preço); invariante em USD
    px = spec["price_base"]
    assert of["buy_volume"] - of["sell_volume"] == pytest.approx(
        (buy - sell) * px, rel=1e-6)
    ws = _window_from_real(spec, _metrics(flow, trades[-1]["T"]), clock)
    assert ws.flow.buy_sell_ratio == pytest.approx(0.25)
    assert ws.volume.delta == pytest.approx(buy - sell)
