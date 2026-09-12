# tests/unit/test_p05_quality_usability.py
"""P05: completeness = features UTILIZÁVEIS (presence != validity).

Escopo estrito P05 (sem CVD/orderbook-math/indicadores/absorção/naming/
participant/payload-dup/dedup). Pesos: soma 10 (REQUIRED w2, CONDITIONAL w1,
OPTIONAL w0 rastreado sem penalizar).
"""
import copy

import pytest

from institutional.enricher import _build_metadata_fields, _score_usability


def _base_event():
    return {
        "preco_fechamento": 80000.0,
        "epoch_ms": 1789158300000,
        "fluxo_continuo": {
            "flow_window_integrity": {
                "1m": {"status": "FULL", "effective_coverage_pct": 99.5,
                       "is_temporal_coverage_valid": True},
                "5m": {"status": "FULL", "effective_coverage_pct": 99.8,
                       "is_temporal_coverage_valid": True},
                "15m": {"status": "FULL", "effective_coverage_pct": 99.1,
                        "is_temporal_coverage_valid": True},
            },
            "order_flow": {"buy_volume_btc": 5.0, "sell_volume_btc": 3.0},
        },
        "orderbook_data": {"is_valid": True, "mid": 80000.0,
                           "bid_depth_usd": 300000.0, "ask_depth_usd": 200000.0,
                           "data_source": "live"},
        "orderbook_quality": "live",
        "institutional_analytics": {
            "session_vwap": {"is_valid": True, "status": "VALID"},
            "positioning": {"is_available": True, "regime": "TREND"},
            "quality": {
                "latency": {"latency_ms": 100, "is_acceptable": True, "is_stale": False},
                "anomalies": {"max_severity": "NONE"},
            },
        },
        "raw_event": {"advanced_analysis": {
            "onchain_metrics": {"mempool_size": 30073.0, "fees_fastest_sat_vb": 5},
            "onchain_status": "fresh",
            "onchain_field_status": {"mempool_size": "VALID",
                                     "fees_fastest_sat_vb": "VALID"},
        }},
        "data_reliability": {"has_options_data": True},
    }


def _meta(ev):
    return _build_metadata_fields(copy.deepcopy(ev), 1789158300000)


# A) tudo required presente, válido e fresh -> máximo contratual 100
def test_a_all_valid_scores_100():
    m = _meta(_base_event())
    assert m["completeness_pct"] == 100.0
    assert m["reliability_score"] == 10.0
    assert m["data_quality_score"] == 10.0
    assert m["quality_reasons"] == []
    assert m["quality_components"]["price"]["usable"] is True


# B) container existe mas value=None -> não conta como válido
def test_b_none_value_not_counted():
    ev = _base_event()
    ev["preco_fechamento"] = None
    ev["contextual_snapshot"] = {"ohlc": {"close": None}}
    m = _meta(ev)
    assert m["quality_components"]["price"]["usable"] is False
    assert m["completeness_pct"] == 80.0  # -20 (w2)
    assert "price_missing_or_invalid" in m["quality_reasons"]


# C) session_vwap STALE/is_valid=0 -> presença sim, validade não (+reliability)
def test_c_stale_session_vwap_invalid():
    ev = _base_event()
    ev["institutional_analytics"]["session_vwap"] = {
        "is_valid": False, "status": "STALE", "age_seconds": 905}
    m = _meta(ev)
    assert m["quality_components"]["session_vwap"]["usable"] is False
    assert m["completeness_pct"] == 90.0
    assert m["reliability_score"] == 9.5
    assert "session_vwap_stale_or_invalid" in m["quality_reasons"]
    assert "reliability:session_vwap_stale_or_invalid" in m["quality_reasons"]


# D) positioning is_available=0 -> rastreado, sem impacto no score
def test_d_positioning_unavailable_no_score_impact():
    ev = _base_event()
    ev["institutional_analytics"]["positioning"] = {"is_available": False,
                                                    "regime": "UNKNOWN"}
    m = _meta(ev)
    assert m["quality_components"]["positioning"]["usable"] is False
    assert m["completeness_pct"] == 100.0
    assert "positioning_unavailable" in m["quality_reasons"]


# E) flow 15m WARMING_UP != FULL
def test_e_flow_15m_warming_up_not_full():
    ev = _base_event()
    ev["fluxo_continuo"]["flow_window_integrity"]["15m"] = {
        "status": "WARMING_UP", "effective_coverage_pct": 67.7,
        "is_temporal_coverage_valid": False}
    m = _meta(ev)
    assert m["quality_components"]["flow_15m"]["usable"] is False
    assert m["completeness_pct"] == 90.0
    assert "flow_15m_warming_up" in m["quality_reasons"]


# F) onchain 1 VALID + 3 API_ERROR -> seção parcial, utilizável, nunca "full silencioso"
def test_f_onchain_partial_usable_with_reason():
    ev = _base_event()
    ev["raw_event"]["advanced_analysis"]["onchain_field_status"] = {
        "mempool_size": "VALID", "mempool_vsize_mb": "API_ERROR",
        "fees_fastest_sat_vb": "API_ERROR", "fees_half_hour_sat_vb": "API_ERROR"}
    m = _meta(ev)
    assert m["quality_components"]["onchain"]["usable"] is True
    assert m["completeness_pct"] == 100.0
    assert "onchain_partial(1/4 valid)" in m["quality_reasons"]


def test_f2_onchain_all_errors_unusable():
    ev = _base_event()
    ev["raw_event"]["advanced_analysis"]["onchain_field_status"] = {
        "mempool_size": "API_ERROR", "fees_fastest_sat_vb": "API_ERROR"}
    ev["raw_event"]["advanced_analysis"]["onchain_metrics"] = {}
    m = _meta(ev)
    assert m["quality_components"]["onchain"]["usable"] is False
    assert m["completeness_pct"] == 90.0


# G) REAL_ZERO conta como válido
def test_g_real_zero_is_valid():
    ev = _base_event()
    ev["raw_event"]["advanced_analysis"]["onchain_field_status"] = {
        "fees_fastest_sat_vb": "REAL_ZERO"}
    m = _meta(ev)
    assert m["quality_components"]["onchain"]["usable"] is True
    assert m["completeness_pct"] == 100.0


# H) cache válido/fresh -> utilizável
def test_h_fresh_cache_usable():
    ev = _base_event()  # status fresh + valores (cache válido indistinguível e válido)
    m = _meta(ev)
    assert m["quality_components"]["onchain"]["usable"] is True


# I) cache STALE com valores reais -> utilizável (freshness vai p/ reliability/coverage)
def test_i_stale_cache_values_still_usable():
    ev = _base_event()
    ev["raw_event"]["advanced_analysis"]["onchain_status"] = "stale"
    m = _meta(ev)
    assert m["quality_components"]["onchain"]["usable"] is True


# J) OPTIONAL ausente não derruba required
def test_j_optional_missing_no_impact():
    ev = _base_event()
    ev["data_reliability"] = {"has_options_data": False}
    m = _meta(ev)
    assert m["completeness_pct"] == 100.0


# K) price REQUIRED ausente -> queda significativa (-20)
def test_k_required_missing_big_drop():
    ev = _base_event()
    del ev["preco_fechamento"]
    m = _meta(ev)
    assert m["completeness_pct"] == 80.0


# L) orderbook is_valid=0 -> não plenamente válido (-20)
def test_l_orderbook_invalid_not_valid():
    ev = _base_event()
    ev["orderbook_data"]["is_valid"] = False
    m = _meta(ev)
    assert m["quality_components"]["orderbook"]["usable"] is False
    assert m["completeness_pct"] == 80.0


# Invariantes de range
def test_invariants_ranges():
    m = _meta(_base_event())
    assert 0 <= m["completeness_pct"] <= 100
    assert 0 <= m["reliability_score"] <= 10
    assert 0 <= m["data_quality_score"] <= 10


# Invariante: inválido->válido nunca reduz; válido->missing nunca aumenta
def test_invariant_monotonicity():
    ev_bad = _base_event()
    ev_bad["institutional_analytics"]["session_vwap"] = {"is_valid": False,
                                                         "status": "STALE"}
    ev_good = _base_event()
    assert _meta(ev_good)["completeness_pct"] >= _meta(ev_bad)["completeness_pct"]
    assert _meta(ev_good)["data_quality_score"] >= _meta(ev_bad)["data_quality_score"]


# Invariante: REAL_ZERO pontua igual a outro valor válido da mesma feature
def test_invariant_real_zero_equals_valid():
    ev_a = _base_event()
    ev_b = _base_event()
    ev_b["raw_event"]["advanced_analysis"]["onchain_field_status"] = {
        "fees_fastest_sat_vb": "REAL_ZERO"}
    ev_b["raw_event"]["advanced_analysis"]["onchain_metrics"] = {
        "fees_fastest_sat_vb": 0}
    assert (_meta(ev_a)["completeness_pct"]
            == _meta(ev_b)["completeness_pct"] == 100.0)
