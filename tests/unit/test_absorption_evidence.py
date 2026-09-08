# tests/unit/test_absorption_evidence.py — P1-B2: 5.0/5.0 só quando observado.
#
# MISSING/INSUFFICIENT => buyer_strength/seller_exhaustion ausentes (None),
# nunca 5.0/5.0 fabricados. index=0 / classification NONE observados seguem
# legítimos. Sem regra global "5/5 inválido".

import pytest

from flow_analyzer.absorption import AbsorptionAnalyzer
from flow_analyzer.core import FlowAnalyzer


def _metrics(order_flow):
    return {"order_flow": dict(order_flow), "tipo_absorcao": "Neutra"}


def _analyzer():
    return FlowAnalyzer.__new__(FlowAnalyzer)


def test_observed_neutral_allows_5_5():
    out = AbsorptionAnalyzer().analyze(
        delta_usd=0.0, total_volume_usd=10000.0, flow_imbalance=0.0,
        buy_pct=50.0, sell_pct=50.0, absorption_label="Neutra", window_min=1)
    assert out is not None
    assert out.buyer_strength == pytest.approx(5.0)
    assert out.seller_exhaustion == pytest.approx(0.0)
    assert out.classification == "NONE"


def test_missing_pcts_yields_no_analysis():
    fa = _analyzer()
    fa._absorption_analyzer = AbsorptionAnalyzer()
    base = {"computation_window_min": 1, "net_flow_1m": 100.0,
            "total_volume": 10000.0, "flow_imbalance": 0.01}
    assert fa._compute_absorption_analysis(_metrics(base)) is None


def test_insufficient_status_yields_no_analysis():
    fa = _analyzer()
    fa._absorption_analyzer = AbsorptionAnalyzer()
    of = {"computation_window_min": 1, "net_flow_1m": 100.0,
          "total_volume": 10000.0, "flow_imbalance": 0.01,
          "aggressive_status": "insufficient", "aggressive_sample_count": 3}
    assert fa._compute_absorption_analysis(_metrics(of)) is None


def test_observed_extreme_yields_real_values():
    fa = _analyzer()
    fa._absorption_analyzer = AbsorptionAnalyzer()
    of = {"computation_window_min": 1, "net_flow_1m": 8000.0,
          "total_volume": 10000.0, "flow_imbalance": 0.8,
          "aggressive_buy_pct": 90.0, "aggressive_sell_pct": 10.0,
          "aggressive_status": "observed", "aggressive_sample_count": 60}
    out = fa._compute_absorption_analysis(_metrics(of))
    assert out is not None
    cur = out["current_absorption"]
    assert cur["buyer_strength"] == pytest.approx(9.0)
    assert cur["classification"] == "MODERATE_ABSORPTION"


def test_payload_distinguishes_absence_from_neutral():
    from market_orchestrator.ai.payload_builder_compact import (
        build_compact_payload,
    )

    base = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 65000.0, "epoch_ms": 1_700_000_000_000,
            "ml_features": {}, "multi_tf": {}}
    ev_neutral = dict(base, fluxo_continuo={
        "order_flow": {"flow_imbalance": 0.0},
        "absorption_analysis": {"current_absorption": {
            "buyer_strength": 5.0, "seller_exhaustion": 0.0,
            "continuation_probability": 0.0}}})
    ev_missing = dict(base, fluxo_continuo={
        "order_flow": {"flow_imbalance": 0.0}})
    c1 = build_compact_payload(ev_neutral)["flow"]
    c2 = build_compact_payload(ev_missing)["flow"]
    assert c1.get("abs_buy_str") == 5.0
    assert "abs_buy_str" not in c2 and "abs_sell_exh" not in c2


def test_whale_provenance_before_aggregation():
    from flow_analyzer.whale_score import WhaleAccumulationCalculator

    calc = WhaleAccumulationCalculator()
    none_out = calc.calculate(sector_flow={}, orderbook_data={},
                              absorption_data=None, derivatives_data={},
                              onchain_data={}, cvd=0.0)
    assert none_out["components"]["absorption"]["score"] == 0.0
    assert none_out["components"]["absorption"]["detail"] == {}
    obs = calc.calculate(
        sector_flow={}, orderbook_data={},
        absorption_data={"current_absorption": {
            "buyer_strength": 5.0, "seller_exhaustion": 5.0,
            "index": 0.0, "classification": "NONE", "label": "Neutra"}},
        derivatives_data={}, onchain_data={}, cvd=0.0)
    assert obs["components"]["absorption"]["score"] == 0.0
    assert obs["components"]["absorption"]["detail"]["buyer_strength"] == 5.0
