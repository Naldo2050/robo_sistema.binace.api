# tests/unit/test_p0_signoff_cross_contract.py — SIGN-OFF P0 (validação cruzada J2).
#
# Não altera produção. Prova simultânea, num evento J2-like:
# FLOW PARTIAL observável sem confirmação + ABSORÇÃO Compra BEARISH NON_VOTING
# + WHALE flow BUY visível + ORDERBOOK ASK-heavy sem iceberg confirmado +
# REGIME UNKNOWN/INSUFFICIENT + payload sem confluência fabricada.

import json

import pytest

from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
from flow_analyzer.metrics import calculate_buy_sell_ratios as calc
from flow_analyzer.whale_score import WhaleAccumulationCalculator
from institutional.enricher import _build_regime_probabilities as regime
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.payload_sections.flow_summary import build_flow_summary
from market_orchestrator.ai.payload_sections.regime_summary import (
    build_regime_summary)
from orderbook_analyzer import OrderBookAnalyzer
from tests.conftest import make_valid_snapshot


def _j2_flow_data():
    return {
        "buy_volume_btc": 50.0, "sell_volume_btc": 50.0,
        "net_flow_1m": 5667000.0, "net_flow_5m": 8416000.0,
        "net_flow_15m": 8416000.0,
        "total_volume": 7948000.0, "total_volume_5m": 27000000.0,
        "total_volume_15m": 81000000.0,
        "flow_window_integrity": {
            "1m": {"status": "WARMING_UP", "effective_coverage_pct": 95.8,
                   "is_temporal_coverage_valid": False},
            "5m": {"status": "WARMING_UP", "effective_coverage_pct": 31.1,
                   "is_temporal_coverage_valid": False},
            "15m": {"status": "WARMING_UP", "effective_coverage_pct": 10.4,
                    "is_temporal_coverage_valid": False}},
    }


def test_cross_j2_flow_partial_observable_no_confirmation():
    out = calc(_j2_flow_data())
    assert out["ratios"]["imbalance_1m"] == pytest.approx(0.713, abs=0.01)
    assert out["flow_trend"] == "insufficient_data"
    assert all(v["validity"] == "PARTIAL"
               for v in out["imbalance_validity"].values())


def test_cross_j2_absorption_bearish_non_voting():
    calc_whale = WhaleAccumulationCalculator()
    out = calc_whale.calculate(
        sector_flow={"whale": {"delta": 75.171}},
        orderbook_data={},
        absorption_data={"current_absorption": {
            "buyer_strength": 8.6, "seller_exhaustion": 1.4,
            "index": 0.5062, "classification": "STRONG_ABSORPTION",
            "label": "Absorção de Compra"}},
        derivatives_data={}, onchain_data={}, cvd=0.0)
    comp = out["components"]["absorption"]
    assert comp["score"] == 0.0
    assert comp["canonical_direction"] == "BEARISH"
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    # Whale flow BUY continua observável e vota sozinho.
    assert out["components"]["flow"]["score"] == 30.0
    assert out["components"]["flow"]["detail"]["whale_delta"] == 75.171


@pytest.mark.asyncio
async def test_cross_j2_orderbook_ask_heavy_no_iceberg(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    snap = make_valid_snapshot(tm.now_ms())
    snap["asks"] = [[float(p), float(q) * 8.0] for p, q in snap["asks"]]
    evt = await oba.analyze(current_snapshot=snap, event_epoch_ms=tm.now_ms())
    assert evt["is_valid"] is True
    assert evt["flow_imbalance"] < -0.3  # ASK-heavy observável
    assert evt["iceberg_reloaded"] is False
    assert evt["iceberg_score"] == 0.0
    assert evt["iceberg_status"] == "UNSUPPORTED"
    assert evt["iceberg_heuristic"]["validity"] == "UNCONFIRMED"


def test_cross_j2_regime_unknown_without_real_inputs():
    out = regime({"fluxo_continuo": {
        "order_flow": {"buy_sell_ratio": {
            "flow_trend": "accelerating_buying",
            "imbalance_validity": {
                "1m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                "5m": {"validity": "PARTIAL", "reason": "WARMING_UP"}}}},
        "flow_window_integrity": {
            "1m": {"status": "WARMING_UP", "is_temporal_coverage_valid": False},
            "5m": {"status": "WARMING_UP", "is_temporal_coverage_valid": False}}}})
    assert out["status"] == "INSUFFICIENT_DATA"
    assert out["current_regime"] == "UNKNOWN"
    assert out["regime_probabilities"] is None


def test_cross_j2_payload_keeps_observations_no_confluence():
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1700000000000,
             "ml_features": {}, "multi_tf": {},
             "fluxo_continuo": {
                 "order_flow": {"net_flow_1m": 5667000.0, "net_flow_5m": 8416000.0,
                                "net_flow_15m": 8416000.0,
                                "buy_sell_ratio": {"buy_sell_ratio": 1.44}},
                 "flow_window_integrity": {
                     "1m": {"status": "WARMING_UP", "effective_coverage_pct": 95.8},
                     "5m": {"status": "WARMING_UP", "effective_coverage_pct": 31.1},
                     "15m": {"status": "WARMING_UP", "effective_coverage_pct": 10.4}}},
             "regime_analysis": {"status": "INSUFFICIENT_DATA",
                                 "current_regime": "UNKNOWN",
                                 "regime_probabilities": None}}
    payload = build_compact_payload(event)
    assert payload["flow"]["d1"] is not None
    assert payload["flow"]["d5"] is not None
    assert payload["flow"]["d15"] is not None
    assert payload["flow"]["q"]["5m"] == {"s": "warm", "c": 31.1}
    assert payload["regime"]["mode"] == "UNK"
    summary = build_flow_summary(payload)
    assert "reversal_signal" not in summary
    rsummary = build_regime_summary(payload)
    assert rsummary["label"] == "Indeterminado"
    assert rsummary["strategies"] == []
    text = json_dumps_rfc8259(sanitize_json_safe(payload))
    assert "NaN" not in text and "Infinity" not in text


def test_cross_full_regression_supported_data_untouched():
    out = calc({"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
                "net_flow_1m": -450.0, "net_flow_5m": -200.0,
                "total_volume": 1000.0, "total_volume_5m": 1000.0,
                "flow_window_integrity": {
                    "1m": {"status": "FULL", "is_temporal_coverage_valid": True},
                    "5m": {"status": "FULL", "is_temporal_coverage_valid": True}}})
    assert out["flow_trend"] == "accelerating_selling"
    assert out["ratios"]["imbalance_1m"] == -0.45
    assert out["imbalance_validity"]["1m"]["validity"] == "VALID"
    w = WhaleAccumulationCalculator().calculate(
        sector_flow={"whale": {"delta": 3.0}}, orderbook_data={},
        absorption_data=None, derivatives_data={}, onchain_data={}, cvd=0.0)
    assert w["components"]["flow"]["score"] == 30.0
    assert w["components"]["absorption"]["score"] == 0.0
    assert w["score"] == 30
