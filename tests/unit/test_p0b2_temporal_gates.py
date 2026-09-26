# tests/unit/test_p0b2_temporal_gates.py — P0-B2: consumers obedecem ao contrato temporal.
#
# Princípio: PARTIAL é observável, nunca confirmação. INVALID não vota.
# NO_INTEGRITY_INFO nunca vira VALID. Só VALID participa de comparação
# multi-timeframe/confluência decisória. Sem threshold novo; matemática com
# horizontes VALID inalterada.
#
# Helper canônico: flow_analyzer/metrics.is_temporal_confirmation_valid
# (dual fail-closed: validity==VALID E integrity valid==True).

import pytest

from flow_analyzer.aggregates import (
    calculate_buy_sell_ratios as aggregates_calc,
)
from flow_analyzer.metrics import (
    calculate_buy_sell_ratios as metrics_calc,
    is_temporal_confirmation_valid,
)
from market_orchestrator.ai.payload_sections.flow_summary import build_flow_summary

CALCS = [metrics_calc, aggregates_calc]


def _full_integrity():
    return {
        "1m": {"status": "FULL", "effective_coverage_pct": 100.0,
               "is_temporal_coverage_valid": True},
        "5m": {"status": "FULL", "effective_coverage_pct": 100.0,
               "is_temporal_coverage_valid": True},
        "15m": {"status": "FULL", "effective_coverage_pct": 100.0,
                "is_temporal_coverage_valid": True},
    }


def _j2_integrity():
    # J2: 1m 95.8 / 5m 31.1 / 15m 10.4, todos valid=false.
    return {
        "1m": {"status": "WARMING_UP", "effective_coverage_pct": 95.8,
               "is_temporal_coverage_valid": False},
        "5m": {"status": "WARMING_UP", "effective_coverage_pct": 31.1,
               "is_temporal_coverage_valid": False},
        "15m": {"status": "WARMING_UP", "effective_coverage_pct": 10.4,
                "is_temporal_coverage_valid": False},
    }


def _flow_data_imbalances(net1, tot1, net5, tot5, integrity):
    return {"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
            "net_flow_1m": net1, "net_flow_5m": net5,
            "total_volume": tot1, "total_volume_5m": tot5,
            "flow_window_integrity": integrity}


# ── helper: dual fail-closed ─────────────────────────────────────────────────

def test_helper_requires_both_sides():
    validity = {"1m": {"validity": "VALID", "reason": None}}
    integrity = {"1m": {"status": "FULL", "is_temporal_coverage_valid": True}}
    assert is_temporal_confirmation_valid(integrity, validity, "1m") is True
    # validity VALID mas integrity discorda => False
    assert is_temporal_confirmation_valid(
        {"1m": {"status": "WARMING_UP", "is_temporal_coverage_valid": False}},
        validity, "1m") is False
    # integrity FULL mas validity discorda => False
    assert is_temporal_confirmation_valid(
        integrity, {"1m": {"validity": "PARTIAL", "reason": "WARMING_UP"}},
        "1m") is False
    # ausências => False (nunca VALID assumido)
    assert is_temporal_confirmation_valid(None, validity, "1m") is False
    assert is_temporal_confirmation_valid(integrity, None, "1m") is False
    assert is_temporal_confirmation_valid({}, {}, "5m") is False
    assert is_temporal_confirmation_valid(
        integrity, {"1m": {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}},
        "1m") is False


# ── flow_trend: gate 1m&5m ───────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_trend_gated_without_valid_horizons(calc):
    # J2-like: imbalances fortes mas PARTIAL => insufficient_data (não "stable").
    out = calc(_flow_data_imbalances(-450.0, 1000.0, -200.0, 1000.0, _j2_integrity()))
    assert out["ratios"]["imbalance_1m"] == -0.45
    assert out["ratios"]["imbalance_5m"] == -0.2
    assert out["flow_trend"] == "insufficient_data"


@pytest.mark.parametrize("calc", CALCS)
def test_trend_math_unchanged_when_valid(calc):
    out = calc(_flow_data_imbalances(-450.0, 1000.0, -200.0, 1000.0, _full_integrity()))
    assert out["flow_trend"] == "accelerating_selling"
    out2 = calc(_flow_data_imbalances(100.0, 1000.0, 350.0, 1000.0, _full_integrity()))
    assert out2["flow_trend"] == "decelerating_buying"


@pytest.mark.parametrize("calc", CALCS)
def test_trend_no_integrity_is_non_confirming(calc):
    # NO_INTEGRITY_INFO: valores preservados p/ legacy/display, sem confirmar.
    out = calc({"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
                "net_flow_1m": -450.0, "net_flow_5m": -200.0,
                "total_volume": 1000.0, "total_volume_5m": 1000.0})
    assert out["ratios"]["imbalance_1m"] == -0.45
    assert out["flow_trend"] == "insufficient_data"


# ── A. J2-like: tudo positivo/PARTIAL, zero confirmação ──────────────────────

def _j2_event(flow_trend="accelerating_buying"):
    return {
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 5667000.0, "net_flow_5m": 8416000.0,
                "net_flow_15m": 8416000.0, "flow_imbalance": 0.712,
                "buy_sell_ratio": {
                    "flow_trend": flow_trend,
                    "imbalance_validity": {
                        "1m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                        "5m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                        "15m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                    },
                },
            },
            "flow_window_integrity": _j2_integrity(),
        },
        "institutional_analytics": {},
        "multi_tf": {},
        "market_environment": {},
        "orderbook_data": {},
    }


def test_a_j2_regime_gets_no_multi_tf_bonus():
    from institutional.enricher import _build_regime_probabilities
    probs = _build_regime_probabilities(_j2_event())["regime_probabilities"]
    # Sem bônus accel (+0.15): trending 0; mean_rev = ADX default 0.15 + RANGE
    # 0.25 + RANGE_BOUND 0.20 = 0.60; total 0.60 => mean_reverting 1.0.
    # Comportamento antigo (fail-open) daria trending 0.2.
    assert probs["trending"] == pytest.approx(0.0)
    assert probs["mean_reverting"] == pytest.approx(1.0)
    assert probs["breakout"] == pytest.approx(0.0)


def test_a_j2_summary_no_reversal_with_status():
    # Armadilha J2: d1/d5 divergentes mas ambos PARTIAL — sem reversal, com status.
    flow = {"pa": "neutral", "imb": 0.3, "d1": "+5.6M", "d5": "-8.4M",
            "iv": {"1m": "P", "5m": "P", "15m": "P"},
            "q": {"1m": {"s": "warm", "c": 95.8},
                  "5m": {"s": "warm", "c": 31.1},
                  "15m": {"s": "warm", "c": 10.4}}}
    result = build_flow_summary({"flow": flow})
    assert "reversal_signal" not in result
    assert result["temporal_comparison"]["conclusion"] == "INSUFFICIENT_COVERAGE"


def test_a_j2_all_positive_observable_no_reversal():
    # J2 real (tudo positivo): valores observáveis, nenhum reversal de qualquer
    # forma — o gate impede que a coincidência vire "3 timeframes confirmam".
    flow = {"pa": "neutral", "imb": 0.3, "d1": "+5.6M", "d5": "+8.4M",
            "iv": {"1m": "P", "5m": "P", "15m": "P"},
            "q": {"1m": {"s": "warm", "c": 95.8}}}
    result = build_flow_summary({"flow": flow})
    assert "reversal_signal" not in result


# ── B. triple-window idêntico ────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_b_identical_partial_windows_zero_confirmations(calc):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 8.416, "net_flow_5m": 8.416, "net_flow_15m": 8.416,
                "total_volume_1m": 10.0, "total_volume_5m": 10.0,
                "total_volume_15m": 10.0,
                "flow_window_integrity": _j2_integrity()})
    assert out["flow_trend"] == "insufficient_data"
    assert all(v["validity"] == "PARTIAL"
               for v in out["imbalance_validity"].values())


# ── C. mixed 1m VALID ────────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_c_mixed_no_multi_tf_trend(calc):
    integrity = _j2_integrity()
    integrity["1m"] = {"status": "FULL", "effective_coverage_pct": 100.0,
                       "is_temporal_coverage_valid": True}
    out = calc({"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
                "net_flow_1m": -450.0, "net_flow_5m": -200.0,
                "total_volume": 1000.0, "total_volume_5m": 1000.0,
                "flow_window_integrity": integrity})
    # 1m isolado é VALID/observável...
    assert out["ratios"]["imbalance_1m"] == -0.45
    assert out["imbalance_validity"]["1m"]["validity"] == "VALID"
    # ...mas sem trend multi-TF (5m PARTIAL).
    assert out["flow_trend"] == "insufficient_data"


# ── D. FULL bit-equivalent ───────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_d_full_trend_and_values_unchanged(calc):
    out = calc({"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
                "net_flow_1m": -450.0, "net_flow_5m": -200.0, "net_flow_15m": 0.0,
                "total_volume": 1000.0, "total_volume_5m": 1000.0,
                "total_volume_15m": 1000.0,
                "flow_window_integrity": _full_integrity()})
    assert out["flow_trend"] == "accelerating_selling"
    assert out["ratios"]["imbalance_1m"] == -0.45
    assert out["ratios"]["imbalance_5m"] == -0.2


def test_d_full_regime_bonus_preserved():
    from institutional.enricher import _build_regime_probabilities
    event = _j2_event(flow_trend="accelerating_buying")
    for tf in ("1m", "5m", "15m"):
        event["fluxo_continuo"]["flow_window_integrity"][tf] = {
            "status": "FULL", "effective_coverage_pct": 100.0,
            "is_temporal_coverage_valid": True}
        event["fluxo_continuo"]["order_flow"]["buy_sell_ratio"][
            "imbalance_validity"][tf] = {"validity": "VALID", "reason": None}
    probs = _build_regime_probabilities(event)["regime_probabilities"]
    # trending 0.15 / total 0.75 => 0.2 normalizado (bônus preservado com FULL).
    assert probs["trending"] == pytest.approx(0.2)


# ── E. INVALID nunca entra ───────────────────────────────────────────────────

def test_e_invalid_window_excluded_everywhere():
    out = metrics_calc({"buy_volume_btc": 500.0, "sell_volume_btc": 500.0,
                        "net_flow_1m": -450.0, "net_flow_5m": 8.416,
                        "total_volume": 1000.0, "total_volume_5m": 7.97,
                        "flow_window_integrity": _full_integrity()})
    assert "imbalance_5m" not in out["ratios"]
    assert out["imbalance_validity"]["5m"] == {
        "validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
    assert out["flow_trend"] == "insufficient_data"

    from institutional.enricher import _build_regime_probabilities
    event = _j2_event(flow_trend="accelerating_buying")
    event["fluxo_continuo"]["flow_window_integrity"]["1m"] = {
        "status": "FULL", "effective_coverage_pct": 100.0,
        "is_temporal_coverage_valid": True}
    event["fluxo_continuo"]["order_flow"]["buy_sell_ratio"][
        "imbalance_validity"] = {
            "1m": {"validity": "VALID", "reason": None},
            "5m": {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"}}
    probs = _build_regime_probabilities(event)["regime_probabilities"]
    assert probs["trending"] == pytest.approx(0.0)

    flow = {"pa": "neutral", "imb": 0.3, "d1": "+16K", "d5": "-8K",
            "iv": {"1m": "V", "5m": "I"}}
    result = build_flow_summary({"flow": flow})
    assert "reversal_signal" not in result
    assert result["temporal_comparison"]["conclusion"] == "INSUFFICIENT_COVERAGE"


# ── F. NO_INTEGRITY_INFO legacy ──────────────────────────────────────────────

def test_f_legacy_values_displayable_not_confirming():
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1700000000000,
             "ml_features": {}, "multi_tf": {},
             "fluxo_continuo": {"order_flow": {
                 "net_flow_1m": 100.0, "net_flow_5m": 350.0,
                 "buy_sell_ratio": {"buy_sell_ratio": 1.44}}}}
    payload = build_compact_payload(event)
    # Valores preservados para display...
    assert payload["flow"]["d1"] is not None
    assert payload["flow"]["d5"] is not None
    # ...sem alegar validade (sem integrity não há q nem iv).
    assert "iv" not in payload["flow"]
    assert "q" not in payload["flow"]


# ── G. compressor round-trip ─────────────────────────────────────────────────

def test_g_compressor_preserves_partial_values_and_quality():
    from market_orchestrator.ai.payload_compressor_v3 import _compress_flow
    payload = {"fluxo_continuo": {
        "order_flow": {
            "net_flow_1m": 100.0, "net_flow_5m": 350.0, "net_flow_15m": 700.0,
            "buy_sell_ratio": {
                "buy_sell_ratio": 1.44,
                "imbalance_validity": {
                    "1m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                    "5m": {"validity": "PARTIAL", "reason": "WARMING_UP"},
                    "15m": {"validity": "INVALID", "reason": "INVARIANT_VIOLATION"},
                },
            },
        }}}
    # Descobre assinatura real de _compress_flow por introspecção tolerante.
    import inspect
    try:
        sig = inspect.signature(_compress_flow)
        result = _compress_flow(payload) if len(sig.parameters) == 1 \
            else _compress_flow(payload, {})
    except TypeError:
        result = _compress_flow(payload, {})
    assert result["net_5m"] is not None
    assert result["net_15m"] is not None
    assert result["iv"] == {"1m": "P", "5m": "P", "15m": "I"}


# ── H. prompt snapshot ───────────────────────────────────────────────────────

def test_h_prompt_contains_temporal_validity_instruction():
    from market_orchestrator.ai import analyzer_qwen
    assert "VALID" in analyzer_qwen.SYSTEM_PROMPT
    low = analyzer_qwen.SYSTEM_PROMPT.lower()
    assert "partial" in low and ("nunca" in low or "nunca confirma" in low
                                 or "observacional" in low)
    assert "uma amostra" in low or "contada 3" in low or "3 vezes" in low
    strict = analyzer_qwen.GROQ_STRICT_SYSTEM_PROMPT
    assert "PARCIAIS" in strict or "parcial" in strict.lower()
