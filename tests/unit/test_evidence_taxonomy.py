# tests/unit/test_evidence_taxonomy.py — P1-B: taxonomy + lineage (só declaração).
#
# Sem pesos/TTL/half-life/caps/dedup aqui. Nenhum counts_as_vote=true.
# Nenhuma fiação produtiva (confluence/payload/prompt intocados).

import pytest

from institutional import evidence_taxonomy as tx
from institutional.evidence import (
    EvidenceCalibration,
    EvidenceFamily,
    EvidenceType,
)


# ── unicidade / resolução ────────────────────────────────────────────────────

def test_field_ids_unique():
    assert len(tx.FIELDS) > 50  # cobertura mínima do inventário P1-B
    assert len({e.field_id for e in tx.FIELDS.values()}) == len(tx.FIELDS)


def test_aliases_unique_and_resolve():
    assert len(set(tx.ALIASES)) == len(tx.ALIASES)
    assert tx.resolve_field_id("d1") == "flow.net.1m"
    assert tx.resolve_field_id("d5") == "flow.net.5m"
    assert tx.resolve_field_id("d15") == "flow.net.15m"
    assert tx.resolve_field_id("trade_imb") == "flow.imbalance.1m"
    assert tx.resolve_field_id("bsr") == "flow.buy_sell_ratio"
    assert tx.resolve_field_id("w.s") == "whale.score"
    assert tx.resolve_field_id("ob.imb") == "orderbook.snapshot.imbalance"
    assert tx.resolve_field_id("flow.net.1m") == "flow.net.1m"  # id direto
    for alias, target in tx.ALIASES.items():
        assert target in tx.FIELDS  # alias nunca cria evidência nova


def test_unknown_alias_never_guessed():
    assert tx.resolve_field_id("d1d5") is None
    assert tx.resolve_field_id("") is None
    assert tx.resolve_field_id(None) is None
    assert tx.resolve_field_id("imbalance") is None  # sem chute parcial
    with pytest.raises(KeyError):
        tx.primary_ancestors("nao_existe")


# ── grafo: sem ciclos, transitivo ────────────────────────────────────────────

def test_registry_has_no_cycles():
    for fid in tx.FIELDS:  # levanta ValueError se houver ciclo
        tx.primary_ancestors(fid)


def test_primary_ancestors_transitive():
    imb = tx.primary_ancestors("flow.imbalance.1m")
    assert imb == frozenset({"raw.aggtrade.buy_notional",
                             "raw.aggtrade.sell_notional"})
    # BSR compartilha exatamente os mesmos ancestrais (bijection).
    assert tx.primary_ancestors("flow.buy_sell_ratio") == imb
    # Trend atravessa imbalance (não-composite) até os raws.
    trend = tx.primary_ancestors("flow.trend")
    assert {"flow.imbalance.1m", "flow.imbalance.5m"} <= trend
    assert "raw.aggtrade.buy_notional" in trend


def test_pressure_shares_orderbook_ancestors():
    pressure = tx.primary_ancestors("orderbook.snapshot.pressure")
    imb = tx.primary_ancestors("orderbook.snapshot.imbalance")
    assert pressure != imb  # pressure depende de mais (via imbalance)
    assert imb <= pressure  # mas compartilha a base bid/ask
    assert "raw.l2.bid_depth_usd" in pressure


# ── composites ───────────────────────────────────────────────────────────────

def test_whale_score_is_composite():
    e = tx.FIELDS["whale.score"]
    assert e.is_composite is True
    assert e.calibration is not EvidenceCalibration.CALIBRATED
    deps = set(e.derived_from)
    assert {"participants.whale.delta.window",
            "orderbook.snapshot.bid_depth",
            "orderbook.snapshot.ask_depth"} <= deps
    # P0-A2: absorption é NON_VOTING — fora de derived_from, só em notes.
    assert not any("absorption" in d for d in deps)
    assert "NON_VOTING" in e.notes
    # Atravessa o composite sem incluí-lo.
    prim = tx.primary_ancestors("whale.score")
    assert "whale.score" not in prim
    assert "participants.whale.delta.window" in prim
    assert "raw.aggtrade.buy_notional" in prim


def test_regime_current_is_composite_of_voting_inputs_only():
    e = tx.FIELDS["regime.current"]
    assert e.is_composite is True
    assert e.calibration is EvidenceCalibration.UNCALIBRATED_HEURISTIC
    assert set(e.derived_from) == {"flow.trend", "profile.shape",
                                   "orderbook.snapshot.imbalance",
                                   "whale.score"}
    # Defaults removidos P0-D2 não aparecem como dependência.
    prim = tx.primary_ancestors("regime.current")
    assert "whale.score" not in prim  # composite atravessado, não incluído
    assert "raw.aggtrade.buy_notional" in prim
    assert "raw.trades.price_qty" in prim


def test_composites_never_become_primary_ancestors():
    for fid, entry in tx.FIELDS.items():
        if not entry.is_composite:
            continue
        for other in tx.FIELDS:
            assert fid not in tx.primary_ancestors(other), \
                f"{fid} composite vazou como ancestral de {other}"


# ── horizons ─────────────────────────────────────────────────────────────────

def test_horizons_1m_5m_15m():
    assert tx.FIELDS["flow.net.1m"].horizon_ms == 60_000
    assert tx.FIELDS["flow.net.5m"].horizon_ms == 300_000
    assert tx.FIELDS["flow.net.15m"].horizon_ms == 900_000
    assert tx.FIELDS["flow.imbalance.5m"].horizon_ms == 300_000
    assert tx.FIELDS["derivatives.oi_delta.1h"].horizon_ms == 3_600_000
    assert tx.FIELDS["derivatives.oi_delta.4h"].horizon_ms == 14_400_000


def test_l2_and_regime_have_no_horizon():
    assert tx.FIELDS["orderbook.snapshot.imbalance"].horizon_ms is None
    assert tx.FIELDS["orderbook.snapshot.bid_depth"].horizon_ms is None
    assert tx.FIELDS["regime.current"].horizon_ms is None
    assert tx.FIELDS["whale.score"].horizon_ms is None
    assert tx.FIELDS["macro.vix"].horizon_ms is None


# ── proibições P1-B ─────────────────────────────────────────────────────────

def test_no_weight_ttl_half_life_confidence():
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(tx.TaxonomyEntry)}
    assert field_names == {"field_id", "family", "evidence_type",
                           "logical_source", "horizon_ms", "derived_from",
                           "calibration", "is_composite", "validity_source",
                           "notes"}
    for fid, entry in tx.FIELDS.items():
        assert entry.horizon_ms is None or isinstance(entry.horizon_ms, int), fid


def test_no_counts_as_vote_true_introduced():
    import pathlib
    src = pathlib.Path("institutional/evidence_taxonomy.py").read_text(
        encoding="utf-8")
    assert "counts_as_vote" not in src


def test_heuristic_calibration_never_calibrated():
    for fid in ("whale.score", "regime.current", "regime.distribution",
                "regime.change_prob", "market_impact.liquidity.score",
                "ml.prob_up", "alerts.active", "orderbook.bias_score",
                "flow.trend", "absorption.current"):
        cal = tx.FIELDS[fid].calibration
        assert cal is not EvidenceCalibration.CALIBRATED, fid
        assert cal in (EvidenceCalibration.UNCALIBRATED_HEURISTIC,), fid


def test_physical_fields_not_applicable():
    for fid in ("price.close", "orderbook.snapshot.bid_depth",
                "derivatives.funding", "macro.vix", "flow.net.1m",
                "cross.corr.eth_7d"):
        assert tx.FIELDS[fid].calibration is EvidenceCalibration.NOT_APPLICABLE


def test_iceberg_capability_representable():
    e = tx.FIELDS["orderbook.iceberg.heuristic"]
    assert e.evidence_type is EvidenceType.POINT_IN_TIME_L2
    assert e.validity_source == "capabilities.ICEBERG_DETECTION_SUPPORTED"
    assert "UNCONFIRMED" in e.notes


# ── double-count report ──────────────────────────────────────────────────────

def test_shared_groups_reveal_known_overlaps():
    groups = tx.shared_ancestor_groups()
    by_member: dict = {}
    for ancestors, members in groups.items():
        for m in members:
            by_member.setdefault(m, ancestors)
    imb_group = by_member["flow.imbalance.1m"]
    assert "flow.buy_sell_ratio" in groups[imb_group]
    assert "flow.net.1m" in groups[imb_group]
    # aggressive pcts dependem de qty breakdown a mais: mesmo grupo não, mas
    # compartilham a base aggTrades (overlap, não igualdade).
    ab = tx.primary_ancestors("flow.aggressive_buy_pct.1m")
    assert set(imb_group) <= ab
    whale_group = by_member["whale.score"]
    assert "whale.classification" in groups[whale_group]
    regime_group = by_member["regime.current"]
    assert "regime.distribution" in groups[regime_group]
    assert "regime.mode" in groups[regime_group]


def test_distribution_by_family_covers_all_families():
    from collections import Counter
    counts = Counter(e.family for e in tx.FIELDS.values())
    for fam in (EvidenceFamily.EXECUTED_FLOW, EvidenceFamily.ORDERBOOK_SNAPSHOT,
                EvidenceFamily.PRICE_RESPONSE, EvidenceFamily.MARKET_STRUCTURE,
                EvidenceFamily.DERIVATIVES, EvidenceFamily.CROSS_ASSET,
                EvidenceFamily.MACRO):
        assert counts[fam] > 0, fam
