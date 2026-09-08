# tests/golden/test_gw_payload.py — L3: compact -> guardrail -> Groq summary.
#
# Componentes reais, sem LLM. P1 quarentena: nenhum valor P1 congelado.

import pytest

from common.ml_features import _map_correlations_to_features
from flow_analyzer.core import FlowAnalyzer
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer
from market_orchestrator.ai.llm_payload_guardrail import guardrail_rewrap
from market_orchestrator.ai.payload_builder_compact import build_compact_payload

from .conftest import (
    FakeClock,
    assert_contract_version,
    assert_json_strict,
    assert_no_nonfinite,
    gen_trades,
    load_fixture,
)


def _flow_metrics(spec):
    clock = FakeClock()
    flow = FlowAnalyzer(time_manager=clock)
    trades = gen_trades(spec)
    for t in trades:
        clock._now = t["T"]
        flow.process_trade(t)
    return flow.get_flow_metrics(reference_epoch_ms=trades[-1]["T"]), trades


def _ml_cross(spec):
    cx = spec["cross"]
    if cx["mode"] == "warming":
        return {}, "warming_up", None
    values = dict(cx["values"])
    mapped = _map_correlations_to_features(values)
    assert_contract_version(mapped, spec["expected_contract_version"], "gw payload")
    return mapped, "fresh", float(cx["age_s"])


def _event(spec, flow_metrics, trades, ml_cross, status, age):
    ob = spec["orderbook"]
    last = trades[-1]["p"]
    return {
        "symbol": spec["symbol"],
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": last,
        "epoch_ms": trades[-1]["T"],
        "ml_features": {"cross_asset": ml_cross,
                        "cross_asset_status": status,
                        "cross_asset_age_seconds": age},
        "fluxo_continuo": {"order_flow": flow_metrics["order_flow"],
                           "liquidity_heatmap": flow_metrics["liquidity_heatmap"],
                           "cvd": flow_metrics.get("cvd", 0)},
        "derivatives": spec.get("derivatives", {}),
        "orderbook_data": {"bid_depth_usd": ob["bid_depth_usd"],
                           "ask_depth_usd": ob["ask_depth_usd"],
                           "imbalance": ob["imbalance"],
                           "spread_bps": ob["spread_bps"],
                           "is_valid": ob["is_valid"],
                           "data_quality": {"data_source": ob["data_source"]}},
        "multi_tf": {
            "15m": {"tendencia": "Neutra", "rsi_short": 55, "macd": 0,
                    "macd_signal": 0, "adx": 12, "atr": 120, "regime": "Range"},
            "1h": {"tendencia": "Neutra", "rsi_short": 55, "macd": 0,
                   "macd_signal": 0, "adx": 12, "atr": 120, "regime": "Range"},
            "4h": {"tendencia": "Neutra", "rsi_short": 55, "macd": 0,
                   "macd_signal": 0, "adx": 12, "atr": 120, "regime": "Range"},
            "1d": {"tendencia": "Neutra", "rsi_short": 55, "macd": 0,
                   "macd_signal": 0, "adx": 12, "atr": 120, "regime": "Range"},
        },
    }


def _full_chain(event):
    compact = build_compact_payload(event)
    assert_json_strict(compact)
    assert_no_nonfinite(compact)
    rewrapped = guardrail_rewrap(compact)
    final = AIAnalyzer._build_groq_payload_summary(rewrapped["ai_payload"])
    assert_json_strict(final)
    return compact, rewrapped, final


def test_gw1_payload_preserves_critical_metadata(frozen_state):
    spec = load_fixture("gw1_balanced.json")
    fm, trades = _flow_metrics(spec)
    ml_cross, status, age = _ml_cross(spec)
    compact, _, final = _full_chain(_event(spec, fm, trades, ml_cross, status, age))
    assert compact["price"]["c"] == int(trades[-1]["p"])
    assert compact["price"]["fr"] == pytest.approx(0.0001)  # P0.1 funding
    assert compact["flow"]["imb"] == pytest.approx(0.0, abs=0.05)
    # P1-B: observed 50/50 chega como observado (mudança contratual)
    assert compact["flow"]["ab"] == pytest.approx(50.0)
    assert compact["flow"]["ab_s"] == "observed"
    assert compact["flow"]["ab_n"] == 120
    assert compact["cross"]["st"] == "fresh"
    assert compact["cross"]["method"] == "shared_session_returns_v2"
    assert compact["cross"]["n"] == 30
    assert compact["cross"]["inst_ndx"] == "QQQ"
    assert final["cross"]["st"] == "fresh"
    assert final["cross"]["method"] == "shared_session_returns_v2"


def test_gw1_payload_deterministic_double_run(frozen_state):
    spec = load_fixture("gw1_balanced.json")

    def _once():
        fm, trades = _flow_metrics(spec)
        ml_cross, status, age = _ml_cross(spec)
        compact, _, final = _full_chain(_event(spec, fm, trades, ml_cross, status, age))
        return {"c": compact["price"]["c"], "fr": compact["price"].get("fr"),
                "imb": compact["flow"]["imb"], "method": compact["cross"]["method"],
                "n": compact["cross"]["n"], "f_st": final["cross"]["st"],
                "f_method": final["cross"]["method"]}

    assert _once() == _once()


def test_gw4_payload_missing_stays_missing(frozen_state):
    spec = load_fixture("gw4_external_unavailable.json")
    fm, trades = _flow_metrics(spec)
    ml_cross, status, age = _ml_cross(spec)
    assert ml_cross == {}
    compact, _, final = _full_chain(_event(spec, fm, trades, ml_cross, status, age))
    assert "fr" not in compact["price"]  # sem funding => sem fr (A3 echo)
    # P1-B: fluxo da GW4 é observado (120 trades); o missing aqui é externo.
    # Missing de fluxo (E/F) => sem ab; com status não-observed => ab_s presente
    # sem ab (a IA distingue). Prova direta:
    fm2 = dict(fm)
    fm2["order_flow"] = dict(fm["order_flow"])
    fm2["order_flow"].pop("aggressive_buy_pct", None)
    fm2["order_flow"].pop("aggressive_sell_pct", None)
    fm2["order_flow"]["aggressive_status"] = "insufficient"
    fm2["order_flow"]["aggressive_sample_count"] = 3
    compact2, _, _ = _full_chain(_event(spec, fm2, trades, ml_cross, status, age))
    assert "ab" not in compact2.get("flow", {})
    assert compact2["flow"]["ab_s"] == "insufficient"
    assert compact2["flow"]["ab_n"] == 3
    assert compact["cross"]["st"] == "warming_up"
    assert "method" not in compact["cross"]
    assert "n" not in compact["cross"]
    assert "eth7" not in compact.get("ctx", {})
    assert "dxy30" not in compact.get("ctx", {})
    for key in ("btc_dxy_corr_30d", "btc_eth_corr_7d"):
        assert key not in str(compact["cross"])


def test_gw6_payload_cross_metadata(frozen_state):
    import pandas as pd

    from market_analysis.cross_asset_correlations import shared_session_corr

    fx = load_fixture("gw6_weekend_holiday.json")
    s, e = fx["btc_calendar"]["start"], fx["btc_calendar"]["end"]
    excl = set(fx["tradfi_weekdays"]["exclusions"])
    f = fx["closes_formula"]
    cal = pd.date_range(s, e, freq="D")
    biz = [d for d in cal if d.weekday() < 5 and d.date().isoformat() not in excl]
    bmap, tmap = {}, {}
    bv, tv = f["btc_start"], f["tradfi_start"]
    j = 0
    denom = max(len(biz) - 1, 1)
    for d in cal:
        iso = d.date().isoformat()
        if d.weekday() >= 5:
            bmap[iso] = bv
            continue
        drift = f["daily_drift"] * (0.5 + j / denom)
        if iso in excl:
            bv = bv * (1 + drift)
            bmap[iso] = bv
            j += 1
            continue
        bv = bv * (1 + drift)
        tv = tv * (1 + drift)
        bmap[iso] = bv
        tmap[iso] = tv
        j += 1
    btc = pd.Series([bmap[d.date().isoformat()] for d in cal],
                    index=pd.DatetimeIndex(cal, tz="UTC"))
    tra = pd.Series([tmap[d.date().isoformat()] for d in biz],
                    index=pd.DatetimeIndex(biz, tz="UTC"))
    dec = pd.Timestamp(fx["decision_ms"], unit="ms", tz="UTC").date()
    out = shared_session_corr(btc, tra, fx["target_returns"], decision_date=dec)
    values = {"status": "ok", "btc_dxy_corr_30d": out["corr"],
              "btc_dxy_corr_30d_n": out["n"],
              "correlation_method": "shared_session_returns_v2",
              "correlation_contract_version": 2,
              "btc_dxy_instrument": "DX-Y.NYB",
              "nasdaq_instrument": "QQQ", "nasdaq_role": "nasdaq_proxy"}
    mapped = _map_correlations_to_features(values)
    assert_contract_version(mapped, fx["expected_contract_version"], "gw6")
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 66875.0, "epoch_ms": fx["decision_ms"],
             "ml_features": {"cross_asset": mapped,
                             "cross_asset_status": "fresh",
                             "cross_asset_age_seconds": 10.0}}
    compact, _, _ = _full_chain(event)
    assert compact["cross"]["method"] == "shared_session_returns_v2"
    assert compact["cross"]["n"] == 30
