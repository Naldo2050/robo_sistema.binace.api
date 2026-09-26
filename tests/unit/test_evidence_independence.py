# tests/unit/test_evidence_independence.py — P1-C: Independence Policy v1.
#
# Camada PURA de análise (sem decisão/score/peso). DISJOINT_LINEAGE !=
# estatisticamente independente. Sem "INDEPENDENT" em lugar algum.

import pytest

from institutional import evidence_independence as pol
from institutional.evidence_independence import (
    LineageBasis,
    Redundancy,
    Relation,
    analyze_evidence_set,
    classify_relation,
    eligible_for_independent_count,
    redundancy,
)


def test_same_field_is_identical():
    d = classify_relation("flow.imbalance.1m", "flow.imbalance.1m")
    assert d["relation"] == Relation.IDENTICAL_LINEAGE.value
    assert "INDEPENDENT" not in Relation.__members__


def test_imbalance_vs_bsr_proven_identical():
    d = classify_relation("flow.imbalance.1m", "flow.buy_sell_ratio")
    assert d["relation"] == Relation.IDENTICAL_LINEAGE.value
    assert d["shared_ancestors"] == ["raw.aggtrade.buy_notional",
                                     "raw.aggtrade.sell_notional"]
    assert redundancy("flow.imbalance.1m", "flow.buy_sell_ratio") == \
        Redundancy.REDUNDANT_EXACT.value


def test_pressure_vs_imbalance_overlap_not_identical():
    d = classify_relation("orderbook.snapshot.pressure",
                          "orderbook.snapshot.imbalance")
    # pressure depende de imbalance + raws: overlap parcial, não identidade.
    assert d["relation"] == Relation.OVERLAPPING_LINEAGE.value
    assert "raw.l2.bid_depth_usd" in d["shared_ancestors"]
    assert redundancy("orderbook.snapshot.pressure",
                      "orderbook.snapshot.imbalance") == \
        Redundancy.NOT_REDUNDANT.value


def test_partial_overlap():
    # aggressive pcts dependem de qty breakdown além dos notionals:
    # overlap parcial (não identidade) com imbalance, ambos não-composites.
    d = classify_relation("flow.aggressive_buy_pct.1m", "flow.imbalance.1m")
    assert d["relation"] == Relation.OVERLAPPING_LINEAGE.value
    assert "raw.aggtrade.buy_notional" in d["shared_ancestors"]
    assert d["ancestors_a"] != d["ancestors_b"]


def test_disjoint_primary_ancestors():
    d = classify_relation("flow.imbalance.1m", "orderbook.snapshot.imbalance")
    assert d["relation"] == Relation.DISJOINT_LINEAGE.value
    assert d["shared_ancestors"] == []
    # DISJOINT não afirma independência estatística (só lineage disjunta).
    assert "independent" not in d["relation"]


def test_unknown_lineage():
    d = classify_relation("ml.prob_up", "flow.imbalance.1m")
    assert d["relation"] == Relation.UNKNOWN.value
    d2 = classify_relation("campo_inexistente", "flow.imbalance.1m")
    assert d2["relation"] == Relation.UNKNOWN.value
    assert redundancy("ml.prob_up", "flow.imbalance.1m") == \
        Redundancy.UNKNOWN.value
    assert eligible_for_independent_count("ml.prob_up") is False
    assert eligible_for_independent_count("campo_inexistente") is False


def test_composite_involved():
    d = classify_relation("whale.score", "flow.imbalance.1m")
    assert d["relation"] == Relation.COMPOSITE.value
    assert d["is_composite_a"] is True
    assert d["is_composite_b"] is False
    # Detalhes do overlap continuam disponíveis mesmo em COMPOSITE.
    assert "raw.aggtrade.buy_notional" in d["shared_ancestors"]
    d2 = classify_relation("whale.score", "regime.current")
    assert d2["relation"] == Relation.COMPOSITE.value


def test_composite_eligible_false():
    for fid in ("whale.score", "regime.current", "regime.distribution",
                "orderbook.bias_score", "alerts.active", "ml.prob_up"):
        assert eligible_for_independent_count(fid) is False, fid
    assert eligible_for_independent_count("flow.imbalance.1m") is True


def test_exact_redundancy_distinct_fields():
    assert redundancy("flow.imbalance.1m", "flow.net.1m") == \
        Redundancy.REDUNDANT_EXACT.value
    assert redundancy("flow.imbalance.1m", "flow.imbalance.1m") == \
        Redundancy.NOT_REDUNDANT.value  # identidade não precisa do rótulo


def test_overlap_never_redundant():
    assert redundancy("orderbook.snapshot.pressure",
                      "orderbook.snapshot.imbalance") == \
        Redundancy.NOT_REDUNDANT.value
    assert redundancy("flow.trend", "flow.imbalance.1m") == \
        Redundancy.NOT_REDUNDANT.value


def test_family_never_overrides_lineage():
    # Mesma family, lineage disjunta -> DISJOINT (family não força identidade).
    d = classify_relation("flow.net.1m", "derivatives.funding")
    assert d["relation"] == Relation.DISJOINT_LINEAGE.value
    # Mesma family, overlap parcial -> OVERLAPPING (não IDENTICAL).
    d3 = classify_relation("orderbook.snapshot.pressure",
                           "orderbook.snapshot.imbalance")
    assert d3["relation"] == Relation.OVERLAPPING_LINEAGE.value
    # Families diferentes, mesma lineage via runtime -> IDENTICAL/overlap:
    # family não força disjoint.
    d2 = classify_relation("flow.imbalance.1m", "derivatives.funding",
                           {"flow.imbalance.1m": ["x"],
                            "derivatives.funding": ["x"]})
    assert d2["relation"] == Relation.IDENTICAL_LINEAGE.value


def test_horizon_never_forces_independence():
    d = classify_relation("flow.net.1m", "flow.net.5m")
    assert d["relation"] == Relation.IDENTICAL_LINEAGE.value
    assert redundancy("flow.net.1m", "flow.net.5m") == \
        Redundancy.REDUNDANT_EXACT.value
    rep = analyze_evidence_set(["flow.net.1m", "flow.net.5m"])
    assert rep["horizons"] == {"flow.net.1m": 60000, "flow.net.5m": 300000}
    assert rep["lineage_basis"] == LineageBasis.STATIC_POTENTIAL.value


def test_runtime_contributors_change_relation():
    static = classify_relation("whale.score", "flow.imbalance.1m")
    assert static["relation"] == Relation.COMPOSITE.value
    assert static["lineage_basis"] == LineageBasis.STATIC_POTENTIAL.value
    # Runtime restrito: whale votou só via depth nesta observação.
    rt = {"whale.score": ["orderbook.snapshot.bid_depth",
                          "orderbook.snapshot.ask_depth"],
          "orderbook.snapshot.imbalance": ["orderbook.snapshot.bid_depth",
                                           "orderbook.snapshot.ask_depth"]}
    d = classify_relation("whale.score", "orderbook.snapshot.imbalance", rt)
    assert d["lineage_basis"] == LineageBasis.RUNTIME_OBSERVED.value
    assert d["relation"] == Relation.COMPOSITE.value  # composite domina
    assert set(d["shared_ancestors"]) == {
        "orderbook.snapshot.bid_depth", "orderbook.snapshot.ask_depth"}
    # Sem runtime p/ um lado -> STATIC_POTENTIAL (nunca inventado).
    d2 = classify_relation("whale.score", "orderbook.snapshot.imbalance",
                           {"whale.score": ["x"]})
    assert d2["lineage_basis"] == LineageBasis.STATIC_POTENTIAL.value


def test_unknown_never_eligible():
    rep = analyze_evidence_set(["ml.prob_up", "flow.imbalance.1m",
                                "campo_inexistente"])
    assert "ml.prob_up" in rep["unknown_fields"]
    assert "campo_inexistente" in rep["unknown_fields"]
    assert "ml.prob_up" in rep["ineligible_fields"]
    assert "campo_inexistente" in rep["ineligible_fields"]
    assert "flow.imbalance.1m" not in rep["ineligible_fields"]


def test_report_deterministic_and_decision_free():
    fields = ["flow.imbalance.1m", "flow.buy_sell_ratio", "whale.score",
              "orderbook.snapshot.imbalance", "regime.current", "ml.prob_up",
              "flow.imbalance.1m"]
    r1 = analyze_evidence_set(fields)
    r2 = analyze_evidence_set(list(fields))
    assert r1 == r2  # mesma entrada => mesma saída
    assert r1["unique_fields"] == ["flow.imbalance.1m", "flow.buy_sell_ratio",
                                  "whale.score", "orderbook.snapshot.imbalance",
                                  "regime.current", "ml.prob_up"]  # dedup estável
    assert set(r1["composites"]) == {"whale.score", "regime.current",
                                      "ml.prob_up"}
    assert "ml.prob_up" in r1["unknown_fields"]
    for forbidden in ("vote_count", "confidence", "score", "weight",
                      "family_cap", "cap", "recommendation"):
        assert forbidden not in r1, forbidden
    assert "executed_flow" in r1["families_present"]
    assert r1["lineage_basis"] == LineageBasis.STATIC_POTENTIAL.value
    # overlap_edges e disjoint_pairs presentes e ordenados.
    assert any(e["a"] == "flow.imbalance.1m" for e in r1["overlap_edges"]) \
        or r1["overlap_edges"] == []
    assert ["flow.imbalance.1m",
            "orderbook.snapshot.imbalance"] in r1["disjoint_pairs"]
