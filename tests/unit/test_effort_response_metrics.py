# tests/unit/test_effort_response_metrics.py — P1-D: RAW metrics esforço-vs-resposta.
#
# Sem thresholds, classes, sinais, confidence, MFE/MAE ou ratios esforço/preço.
# Contrato de origem: data_handler particiona notionals por m_flags
# (True=SELL, False=BUY); buy/sell_notional JÁ são agressão classificada.

import json
import math
import pathlib

import pytest

from flow_analyzer import effort_response as er
from flow_analyzer.effort_response import compute_effort_response


def _j2():
    return {
        "buy_notional_usd": 6816945.1591,
        "sell_notional_usd": 1149280.5758,
        "open": 79776.9,
        "high": 79810.8,
        "low": 79776.9,
        "close": 79792.7,
        "window_duration_ms": 57453,
        "vwap": 79803.5,
        "poc": 79804.9,
    }


def test_exact_formulas():
    r = compute_effort_response(
        buy_notional_usd=600.0,
        sell_notional_usd=400.0,
        open=100.0,
        high=105.0,
        low=99.0,
        close=102.0,
        window_duration_ms=60000,
        vwap=101.0,
        poc=100.5,
    )
    assert r["validity"] == "VALID" and r["reasons"] == {}
    assert r["total_aggressive_notional_usd"] == 1000.0
    assert r["net_aggressive_notional_usd"] == 200.0
    assert r["buy_share"] == pytest.approx(0.6)
    assert r["sell_share"] == pytest.approx(0.4)
    assert r["price_displacement_usd"] == pytest.approx(2.0)
    assert r["price_displacement_bps"] == pytest.approx(200.0)
    assert r["range_usd"] == pytest.approx(6.0)
    assert r["range_bps"] == pytest.approx(600.0)
    assert r["close_from_high_usd"] == pytest.approx(-3.0)
    assert r["close_from_high_bps"] == pytest.approx(-300.0)
    assert r["close_from_low_usd"] == pytest.approx(3.0)
    assert r["close_from_low_bps"] == pytest.approx(300.0)
    assert r["close_vs_vwap_usd"] == pytest.approx(1.0)
    assert r["close_vs_vwap_bps"] == pytest.approx(100.0)
    assert r["close_vs_poc_usd"] == pytest.approx(1.5)
    assert r["close_vs_poc_bps"] == pytest.approx(150.0)
    assert r["window_duration_ms"] == 60000


def test_j2_numbers():
    r = compute_effort_response(**_j2())
    assert r["validity"] == "VALID"
    assert r["total_aggressive_notional_usd"] == pytest.approx(7966225.7349)
    assert r["net_aggressive_notional_usd"] == pytest.approx(5667664.5833)
    assert r["buy_share"] == pytest.approx(0.8557308, abs=1e-4)
    assert r["sell_share"] == pytest.approx(0.1442692, abs=1e-4)
    assert r["buy_share"] + r["sell_share"] == pytest.approx(1.0)
    assert r["price_displacement_usd"] == pytest.approx(15.8)
    assert r["price_displacement_bps"] == pytest.approx(15.8 / 79776.9 * 10000)
    assert r["range_usd"] == pytest.approx(33.9)
    assert r["range_bps"] == pytest.approx(33.9 / 79776.9 * 10000)
    assert r["close_from_high_usd"] == pytest.approx(-18.1)
    assert r["close_from_high_bps"] == pytest.approx(-18.1 / 79776.9 * 10000)
    assert r["close_from_low_usd"] == pytest.approx(15.8)
    assert r["close_from_low_bps"] == pytest.approx(15.8 / 79776.9 * 10000)
    assert r["close_vs_vwap_usd"] == pytest.approx(-10.8)
    assert r["close_vs_vwap_bps"] == pytest.approx(-10.8 / 79776.9 * 10000)
    assert r["close_vs_poc_usd"] == pytest.approx(-12.2)
    assert r["close_vs_poc_bps"] == pytest.approx(-12.2 / 79776.9 * 10000)


def test_zero_net_allowed():
    base = _j2()
    base.update(buy_notional_usd=100.0, sell_notional_usd=100.0)
    r = compute_effort_response(**base)
    assert r["validity"] == "VALID"
    assert r["net_aggressive_notional_usd"] == 0.0
    assert r["buy_share"] == r["sell_share"] == pytest.approx(0.5)


def test_zero_total_notional_handled_safely():
    base = _j2()
    base.update(buy_notional_usd=0.0, sell_notional_usd=0.0)
    r = compute_effort_response(**base)
    assert r["validity"] == "PARTIAL"
    assert r["total_aggressive_notional_usd"] == 0.0
    assert r["net_aggressive_notional_usd"] == 0.0
    assert r["buy_share"] is None
    assert r["sell_share"] is None
    assert "buy_share" in r["reasons"]
    assert "sell_share" in r["reasons"]


def test_missing_optional_is_partial_not_invalid():
    kw = _j2()
    del kw["vwap"]
    del kw["poc"]
    r = compute_effort_response(**kw)
    assert r["validity"] == "PARTIAL"
    assert r["close_vs_vwap_usd"] is None and r["close_vs_poc_usd"] is None
    assert r["net_aggressive_notional_usd"] == pytest.approx(5667664.5833)


@pytest.mark.parametrize("field,value", [
    ("buy_notional_usd", float("nan")), ("sell_notional_usd", float("inf")),
    ("open", float("-inf")), ("high", float("nan")), ("low", float("nan")),
    ("close", float("inf")), ("window_duration_ms", float("nan")),
    ("buy_notional_usd", None), ("open", None),
])
def test_nonfinite_or_missing_minimum_is_invalid(field, value):
    kw = _j2()
    kw[field] = value
    r = compute_effort_response(**kw)
    assert r["validity"] == "INVALID"
    assert r["price_displacement_usd"] is None
    assert r["reasons"]
    text = json.dumps(r, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_negative_notional_invalid():
    kw = _j2()
    kw["buy_notional_usd"] = -1.0
    assert compute_effort_response(**kw)["validity"] == "INVALID"


@pytest.mark.parametrize("patch", [
    {"high": 79700.0},  # high<low
    {"open": 79900.0},  # high<open e low>close? open acima de high
    {"close": 79900.0},  # high<close
    {"low": 79800.0},  # low>open e low>close
    {"open": 0.0},  # zero open
    {"open": -5.0},
    {"window_duration_ms": 0},
    {"window_duration_ms": -10},
])
def test_invalid_ohlc_or_duration(patch):
    kw = _j2()
    kw.update(patch)
    r = compute_effort_response(**kw)
    assert r["validity"] == "INVALID"
    assert r["reasons"]


def test_optional_nonfinite_does_not_contaminate_minimums():
    kw = _j2()
    kw["vwap"] = float("nan")
    kw["poc"] = float("inf")
    r = compute_effort_response(**kw)
    assert r["validity"] == "PARTIAL"
    assert r["close_vs_vwap_usd"] is None and r["close_vs_poc_usd"] is None
    assert r["net_aggressive_notional_usd"] == pytest.approx(5667664.5833)
    assert "close_vs_vwap_usd" in r["reasons"]
    assert "close_vs_poc_usd" in r["reasons"]


def test_rfc8259_and_no_nan_leak():
    r = compute_effort_response(**_j2())
    text = json.dumps(r, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    for v in r.values():
        if isinstance(v, float):
            assert math.isfinite(v)


def test_no_lookahead_fields():
    r = compute_effort_response(**_j2())
    blob = json.dumps(r).lower()
    assert "mfe" not in blob and "mae" not in blob
    assert "future" not in blob and "lookahead" not in blob
    assert set(r) <= {
        "buy_notional_usd", "sell_notional_usd",
        "total_aggressive_notional_usd", "net_aggressive_notional_usd",
        "buy_share", "sell_share", "price_displacement_usd",
        "price_displacement_bps", "range_usd", "range_bps",
        "close_from_high_usd", "close_from_high_bps", "close_from_low_usd",
        "close_from_low_bps", "close_vs_vwap_usd", "close_vs_vwap_bps",
        "close_vs_poc_usd", "close_vs_poc_bps", "window_duration_ms",
        "core_validity", "optional_completeness",
        "validity", "reasons"}


def test_no_classification_strings():
    # Não afirmar: absorption, bearish, bullish, reversal, breakout, continuation, exhaustion, mfe, mae.
    forbidden = [
        "absor" + "ption",
        "bear" + "ish",
        "bull" + "ish",
        "rever" + "sal",
        "break" + "out",
        "contin" + "uation",
        "exha" + "ustion",
        "mf" + "e",
        "ma" + "e",
    ]
    # 1. Output numérico do cálculo não deve conter nenhuma string classificatória
    r = compute_effort_response(**_j2())
    blob = json.dumps(r).lower()
    for word in forbidden:
        assert word not in blob, f"Output vazou classificação interpretativa: {word}"

    # 2. Código fonte puro de effort_response.py não deve conter termos interpretativos
    src = pathlib.Path("flow_analyzer/effort_response.py").read_text(
        encoding="utf-8").lower()
    for word in forbidden:
        assert word not in src, f"effort_response.py vazou classificação: {word}"


def test_o1_no_heavy_deps():
    src = pathlib.Path("flow_analyzer/effort_response.py").read_text(
        encoding="utf-8")
    for token in ("pandas", "numpy", "polars", "DataFrame", "socket",
                  "requests", "open(", "threading", "asyncio"):
        assert token not in src, token
    assert "import math" in src  # único import além de typing


# ── taxonomy / lineage ───────────────────────────────────────────────────────

def test_taxonomy_effort_fields():
    from institutional import evidence_taxonomy as tx
    from institutional.evidence import EvidenceFamily, EvidenceType
    for fid in ("effort.notional.buy", "effort.notional.sell",
                "effort.notional.total", "effort.notional.net",
                "effort.share.buy", "effort.share.sell"):
        e = tx.FIELDS[fid]
        assert e.family is EvidenceFamily.EXECUTED_FLOW
        assert e.evidence_type is EvidenceType.CONTINUOUS_TRADES
    for fid in ("effort.price.displacement_usd", "effort.price.range_usd",
                "effort.price.close_from_high_usd",
                "effort.price.close_vs_vwap_usd"):
        e = tx.FIELDS[fid]
        assert e.family is EvidenceFamily.PRICE_RESPONSE
    comp = tx.FIELDS["effort.response"]
    assert comp.is_composite is True
    assert comp.calibration.value == "not_applicable"
    assert len(tx.FIELDS) == 103 + 19


def test_effort_shares_ancestry_with_executed_flow():
    from institutional import evidence_taxonomy as tx
    from institutional.evidence_independence import (
        classify_relation, eligible_for_independent_count)
    # 1. Notionals executados compartilham linhagem com executed flow
    assert tx.primary_ancestors("effort.notional.buy") == \
        frozenset({"raw.aggtrade.buy_notional"})
    d = classify_relation("effort.notional.buy", "flow.net.1m")
    assert d["relation"] == "overlapping_lineage"
    assert "raw.aggtrade.buy_notional" in d["shared_ancestors"]

    # 2. Total notional usa exatamente os mesmos primitivos de flow.net.1m
    d_tot = classify_relation("effort.notional.total", "flow.net.1m")
    assert d_tot["relation"] == "identical_lineage"
    assert d_tot["shared_ancestors"] == ["raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional"]

    # 3. OHLC computacional depende apenas de price: computacionalmente disjunto de flow notionals
    d2 = classify_relation("effort.price.displacement_usd", "flow.net.1m")
    assert d2["relation"] == "disjoint_lineage"
    assert d2["shared_ancestors"] == []

    # 4. VWAP computacional depende de price e trades qty: computacionalmente disjunto de notionals
    d_vwap = classify_relation("effort.price.close_vs_vwap_usd", "flow.net.1m")
    assert d_vwap["relation"] == "disjoint_lineage"

    # 5. Composite nunca elegível para contagem de evidência independente
    assert eligible_for_independent_count("effort.response") is False
    d3 = classify_relation("effort.response", "flow.net.1m")
    assert d3["relation"] == "composite"


def test_adapter_evidences_never_vote():
    from flow_analyzer.effort_response import effort_response_to_evidence
    r = compute_effort_response(**_j2())
    evs = effort_response_to_evidence(r, observed_at_ms=1700000000000)
    assert len(evs) > 0
    for e in evs:
        d = e.to_dict()
        assert d["counts_as_vote"] is False
        assert d["direction"] == "unknown"
        assert d["feature_contract_version"] == "1.0.0"
        assert d["validity"] in ("valid", "partial")
    assert {e.to_dict()["source"] for e in evs} == {"effort_response"}
    # INVALID global => nada adaptável.
    bad = compute_effort_response(**{**_j2(), "open": 0.0})
    assert bad["validity"] == "INVALID"
    assert effort_response_to_evidence(bad) == []
    text = json.dumps([e.to_dict() for e in evs], allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_p1c_interaction_full():
    from institutional.evidence_independence import analyze_evidence_set, classify_relation

    # 1. Horizon e Family diferentes nunca forçam independência se houvesse overlap
    # Para métricas de preço computacionais, são disjuntas por ausência de ancestrais compartilhados
    d = classify_relation("effort.price.displacement_usd", "flow.net.1m")
    assert d["relation"] == "disjoint_lineage"

    # 2. Total notional compartilha exatamente os mesmos ancestrais do net flow
    d_total = classify_relation("effort.notional.total", "flow.net.1m")
    assert d_total["relation"] == "identical_lineage"

    # 3. analyze_evidence_set com composite effort.response
    fields = ["effort.response", "flow.net.1m", "effort.price.displacement_usd", "effort.notional.buy"]
    rep = analyze_evidence_set(fields)
    # composite effort.response não é elegível para contagem de independentes
    assert "effort.response" in rep["ineligible_fields"]
    assert "effort.response" in rep["composites"]
    assert any(e["a"] in ("flow.net.1m", "effort.notional.buy") and e["b"] in ("flow.net.1m", "effort.notional.buy")
               for e in rep["overlap_edges"])


def test_adapter_option_b_vwap_absent():
    from flow_analyzer.effort_response import effort_response_to_evidence
    kw = _j2()
    del kw["vwap"]
    r = compute_effort_response(**kw)
    assert r["core_validity"] == "VALID"
    assert r["optional_completeness"] == "PARTIAL"
    assert r["validity"] == "PARTIAL"
    evs = effort_response_to_evidence(r)
    by_metric = {e.to_dict()["metadata"]["metric"]: e for e in evs}
    assert by_metric["price_displacement_usd"].validity.value == "valid"
    assert by_metric["range_usd"].validity.value == "valid"
    assert by_metric["buy_notional_usd"].validity.value == "valid"
    assert "close_vs_vwap_usd" not in by_metric
    assert "close_vs_vwap_bps" not in by_metric
    assert "close_vs_poc_usd" in by_metric
    for e in evs:
        assert e.counts_as_vote is False


def test_adapter_option_b_poc_absent():
    from flow_analyzer.effort_response import effort_response_to_evidence
    kw = _j2()
    del kw["poc"]
    r = compute_effort_response(**kw)
    assert r["core_validity"] == "VALID"
    assert r["optional_completeness"] == "PARTIAL"
    assert r["validity"] == "PARTIAL"
    evs = effort_response_to_evidence(r)
    by_metric = {e.to_dict()["metadata"]["metric"]: e for e in evs}
    assert by_metric["price_displacement_usd"].validity.value == "valid"
    assert "close_vs_poc_usd" not in by_metric
    assert "close_vs_poc_bps" not in by_metric
    assert "close_vs_vwap_usd" in by_metric
    for e in evs:
        assert e.counts_as_vote is False


def test_adapter_option_b_both_optionals_absent():
    from flow_analyzer.effort_response import effort_response_to_evidence
    kw = _j2()
    del kw["vwap"]
    del kw["poc"]
    r = compute_effort_response(**kw)
    assert r["core_validity"] == "VALID"
    assert r["optional_completeness"] == "PARTIAL"
    evs = effort_response_to_evidence(r)
    by_metric = {e.to_dict()["metadata"]["metric"]: e for e in evs}
    assert len(evs) == 14  # 6 flow + 8 price (sem os 4 opcionais)
    for name in ("price_displacement_usd", "range_usd", "buy_notional_usd", "buy_share"):
        assert by_metric[name].validity.value == "valid"
    assert "close_vs_vwap_usd" not in by_metric
    assert "close_vs_poc_usd" not in by_metric


def test_adapter_option_b_zero_total_notional():
    from flow_analyzer.effort_response import effort_response_to_evidence
    kw = _j2()
    kw.update(buy_notional_usd=0.0, sell_notional_usd=0.0)
    r = compute_effort_response(**kw)
    assert r["core_validity"] == "PARTIAL"
    assert r["validity"] == "PARTIAL"
    assert r["buy_share"] is None
    assert r["sell_share"] is None
    evs = effort_response_to_evidence(r)
    by_metric = {e.to_dict()["metadata"]["metric"]: e for e in evs}
    # Sem shares fantasmas
    assert "buy_share" not in by_metric
    assert "sell_share" not in by_metric
    # Preço íntegro emitido como VALID
    assert by_metric["price_displacement_usd"].validity.value == "valid"
    assert by_metric["range_usd"].validity.value == "valid"
    # Notionals zero contêm metadata de cenário não produtivo
    assert by_metric["buy_notional_usd"].metadata.get("zero_total_notional") is True
    assert by_metric["buy_notional_usd"].metadata.get("non_productive_scenario") is True


def test_adapter_invalid_core_emits_no_valid_evidence():
    from flow_analyzer.effort_response import effort_response_to_evidence
    for bad_patch in ({"open": 0.0}, {"high": 79700.0}, {"window_duration_ms": -10}):
        kw = _j2()
        kw.update(bad_patch)
        r = compute_effort_response(**kw)
        assert r["core_validity"] == "INVALID"
        assert r["validity"] == "INVALID"
        evs = effort_response_to_evidence(r)
        assert evs == []  # Nenhuma métrica contaminada vira VALID


def test_j2_complete_bundle_and_adapter():
    from flow_analyzer.effort_response import effort_response_to_evidence
    r = compute_effort_response(**_j2())
    assert r["core_validity"] == "VALID"
    assert r["optional_completeness"] == "COMPLETE"
    assert r["validity"] == "VALID"
    evs = effort_response_to_evidence(r)
    assert len(evs) == 18  # Todas as 18 métricas escalares
    for e in evs:
        assert e.validity.value == "valid"
        assert e.counts_as_vote is False
        assert e.direction.value == "unknown"



