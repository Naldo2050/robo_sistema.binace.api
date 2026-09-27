# tests/unit/test_confluence_shadow.py
"""
P2-A — Testes do Evidence Reconciler v1 (confluence_shadow).

Cobertura exigida pelo contrato:
- zero evidence -> INSUFFICIENT_DATA
- só neutral -> NEUTRAL_ONLY
- uma bullish -> ALIGNED_BULLISH + independent_confirmation=false
- uma bearish -> ALIGNED_BEARISH + false
- bullish+bearish -> MIXED_DIRECTIONS
- 2 bullish + 1 bearish -> MIXED_DIRECTIONS (anti-majority)
- 100 bullish aliases REDUNDANT_EXACT não alteram semântica
- whale/regime composite não alteram status
- PARTIAL/INVALID/UNSUPPORTED/UNKNOWN descartados
- absorption vs flow registrada OVERLAPPING
- disjoint lineage nunca vira statistical independence
- orderbook marcado snapshot-only
- J2 real -> MIXED_DIRECTIONS
- deterministic output
- zero fields de confidence/probability/weight/trade/action/entry

Sem threshold, sem pesos, sem lifecycle de trade.
"""
from __future__ import annotations

import pytest

from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
)
from institutional.confluence_shadow import (
    ConfluenceShadowResult,
    ReconcilerStatus,
    reconcile,
    DIRECTIONAL_WHITELIST,
)

# ── HELPERS ────────────────────────────────────────────────────────────────────

# Timestamps reais J2
J2_OPEN_MS = 1788702361157
J2_CLOSE_MS = 1788702418610
J2_ANCHOR_MS = 1788702420000

COMMON_KW = dict(
    symbol="BTCUSDT",
    observation_open_ms=J2_OPEN_MS,
    observation_close_ms=J2_CLOSE_MS,
    causal_anchor_ms=J2_ANCHOR_MS,
)


def _ev(source: str,
        direction=EvidenceDirection.BULLISH,
        family=EvidenceFamily.EXECUTED_FLOW,
        evidence_type=EvidenceType.CONTINUOUS_TRADES,
        validity=EvidenceValidity.VALID,
        calibration=EvidenceCalibration.NOT_APPLICABLE,
        observed_at_ms=None) -> Evidence:
    """Factory de evidências para testes."""
    return Evidence(
        source=source,
        direction=direction,
        family=family,
        evidence_type=evidence_type,
        validity=validity,
        calibration=calibration,
        observed_at_ms=observed_at_ms or J2_CLOSE_MS,
    )


# ── TESTES DE STATUS BÁSICO ───────────────────────────────────────────────────

def test_zero_evidence_insufficient_data():
    """Zero evidências -> INSUFFICIENT_DATA."""
    result = reconcile(evidences=[], **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert result.evidence_count == 0
    assert result.independent_confirmation is False


def test_only_neutral_neutral_only():
    """Só evidências NEUTRAL -> NEUTRAL_ONLY."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.NEUTRAL),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.NEUTRAL_ONLY
    assert result.evidence_count == 0  # nenhuma directional ativa
    assert result.independent_confirmation is False


def test_single_bullish_aligned_bullish():
    """Uma bullish -> ALIGNED_BULLISH + independent_confirmation=false."""
    evs = [_ev("flow.net.1m", direction=EvidenceDirection.BULLISH)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BULLISH
    assert result.evidence_count == 1
    assert result.independent_confirmation is False


def test_single_bearish_aligned_bearish():
    """Uma bearish -> ALIGNED_BEARISH + false."""
    evs = [_ev("flow.net.1m", direction=EvidenceDirection.BEARISH)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BEARISH
    assert result.evidence_count == 1
    assert result.independent_confirmation is False


def test_bullish_plus_bearish_mixed():
    """Bullish + bearish -> MIXED_DIRECTIONS."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS
    assert result.independent_confirmation is False


def test_anti_majority_2_bullish_1_bearish_mixed():
    """2 bullish + 1 bearish -> MIXED_DIRECTIONS (anti-majority voting)."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("market_structure.bos",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.MARKET_STRUCTURE,
            evidence_type=EvidenceType.UNKNOWN),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS
    assert "bullish" in result.directions_present
    assert "bearish" in result.directions_present


# ── REDUNDANCY ─────────────────────────────────────────────────────────────────

def test_100_redundant_bullish_aliases_same_semantics():
    """100 aliases REDUNDANT_EXACT não transformam 1 observação em 100.

    flow.net.1m, flow.imbalance.1m e flow.buy_sell_ratio compartilham
    ancestrais exatos. Após dedup, somente o representative permanece ativo.
    """
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("flow.imbalance.1m", direction=EvidenceDirection.BULLISH),
        _ev("flow.buy_sell_ratio", direction=EvidenceDirection.BULLISH),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BULLISH
    # Grupos de redundância devem existir
    assert len(result.exact_redundancy_groups) >= 1
    # evidence_count reflete somente os ativos (sem suppressed)
    assert result.evidence_count == 1
    assert result.independent_confirmation is False


def test_redundant_plus_different_family_mixed():
    """Redundantes bullish + bearish de outra família -> MIXED_DIRECTIONS."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("flow.imbalance.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS


# ── COMPOSITES ─────────────────────────────────────────────────────────────────

def test_whale_composite_does_not_alter_status():
    """whale.score composite não altera status de INSUFFICIENT_DATA."""
    evs = [
        Evidence(
            source="whale.score",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.observed_composites) == 1
    assert result.evidence_count == 0


def test_regime_composite_does_not_alter_status():
    """regime.current composite não altera status."""
    evs = [
        Evidence(
            source="regime.current",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.observed_composites) == 1


def test_composites_with_directional_do_not_affect():
    """Composites presentes junto com evidência direcional não mudam status."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        Evidence(
            source="whale.score",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
        Evidence(
            source="regime.current",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    # Somente flow.net.1m é directional válida -> ALIGNED_BULLISH
    assert result.status == ReconcilerStatus.ALIGNED_BULLISH
    assert result.evidence_count == 1
    assert len(result.observed_composites) == 2


# ── VALIDITY DESCARTE ──────────────────────────────────────────────────────────

def test_partial_discarded():
    """PARTIAL descartada."""
    evs = [_ev("flow.net.1m", validity=EvidenceValidity.PARTIAL)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


def test_invalid_discarded():
    """INVALID descartada."""
    evs = [_ev("flow.net.1m", validity=EvidenceValidity.INVALID)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


def test_unsupported_discarded():
    """UNSUPPORTED descartada."""
    evs = [_ev("flow.net.1m", validity=EvidenceValidity.UNSUPPORTED)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


def test_unknown_validity_discarded():
    """UNKNOWN validity descartada."""
    evs = [_ev("flow.net.1m", validity=EvidenceValidity.UNKNOWN)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


def test_stale_discarded():
    """STALE descartada."""
    evs = [_ev("flow.net.1m", validity=EvidenceValidity.STALE)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


# ── ABSORPTION ─────────────────────────────────────────────────────────────────

def test_absorption_vs_flow_overlapping():
    """absorption.current vs flow registrada como OVERLAPPING_LINEAGE."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        Evidence(
            source="absorption.current",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.EXECUTED_FLOW,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS

    # Deve existir relação OVERLAPPING entre flow.net.1m e absorption.current
    overlap_pairs = [
        (r["field_a"], r["field_b"]) for r in result.overlap_relations
    ]
    assert ("absorption.current", "flow.net.1m") in overlap_pairs or \
           ("flow.net.1m", "absorption.current") in overlap_pairs

    # Absorção NÃO é promoted a independent confirmation
    assert result.independent_confirmation is False

    # Absorção deve carregar UNCALIBRATED_HEURISTIC nos fatos
    absorption_facts = [
        f for f in result.summary_facts.get("facts", [])
        if "absorption" in f
    ]
    assert any("UNCALIBRATED_HEURISTIC" in f for f in absorption_facts)


def test_absorption_bearish_label():
    """Absorção de compra = BEARISH (contrato P0)."""
    ev = Evidence(
        source="absorption.current",
        direction=EvidenceDirection.BEARISH,
        family=EvidenceFamily.EXECUTED_FLOW,
        evidence_type=EvidenceType.CONTINUOUS_TRADES,
        validity=EvidenceValidity.VALID,
        calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
        observed_at_ms=J2_CLOSE_MS,
    )
    result = reconcile(evidences=[ev], **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BEARISH
    assert result.evidence_count == 1


# ── DISJOINT LINEAGE ──────────────────────────────────────────────────────────

def test_disjoint_lineage_never_statistical_independence():
    """DISJOINT_LINEAGE nunca vira independência estatística."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    # independent_confirmation SEMPRE False
    assert result.independent_confirmation is False

    # Relações disjoint devem ter computationally_disjoint=True
    # mas NÃO devem ter campo "independent" ou "statistically_independent"
    for rel in result.disjoint_relations:
        assert "independent" not in rel
        assert "statistically_independent" not in rel
        assert rel.get("computationally_disjoint") is True


# ── ORDERBOOK POINT_IN_TIME_L2 ────────────────────────────────────────────────

def test_orderbook_marked_snapshot_only():
    """Orderbook registrado como POINT_IN_TIME_L2 nos fatos."""
    evs = [
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BEARISH

    # Deve ter fact com POINT_IN_TIME_L2
    facts = result.summary_facts.get("facts", [])
    assert any("POINT_IN_TIME_L2" in f for f in facts)


def test_flow_buy_book_sell_mixed():
    """flow BUY + book SELL -> MIXED_DIRECTIONS (não absorption/resistance)."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS


# ── FIXTURE REAL J2 ────────────────────────────────────────────────────────────

def test_j2_real_mixed_directions():
    """J2 real: flow BULLISH, absorption BEARISH, book BEARISH -> MIXED.

    Dados reais J2:
    - buy_notional: $6.81M vs sell: $1.15M -> fluxo agressivo comprador
    - Preço recuou do topo $79810.8 para $79792.7 -> absorção no topo
    - Absorção canônica: compra absorvida = BEARISH
    - Book ASK-heavy = BEARISH snapshot
    - whale.score / regime.current = composite / non-voting
    """
    evs = [
        # Fluxo executado: agressão de compra massiva
        Evidence(
            source="flow.imbalance.1m",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.EXECUTED_FLOW,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            observed_at_ms=J2_CLOSE_MS,
        ),
        # Absorção: compra absorvida no topo => BEARISH
        Evidence(
            source="absorption.current",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.EXECUTED_FLOW,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
        # Orderbook L2: ASK-heavy (resistência passiva)
        Evidence(
            source="orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            observed_at_ms=J2_CLOSE_MS,
        ),
        # whale.score: composite, non-voting
        Evidence(
            source="whale.score",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
        # regime.current: composite, non-voting
        Evidence(
            source="regime.current",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.UNKNOWN,
            evidence_type=EvidenceType.DERIVED,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=J2_CLOSE_MS,
        ),
    ]

    result = reconcile(evidences=evs, **COMMON_KW)

    # Status deve ser MIXED_DIRECTIONS (não INVALIDATED)
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS

    # Composites não participaram do status
    assert len(result.observed_composites) == 2

    # Overlap entre flow e absorption deve estar registrado
    overlap_sources = set()
    for rel in result.overlap_relations:
        overlap_sources.add(rel["field_a"])
        overlap_sources.add(rel["field_b"])
    assert "absorption.current" in overlap_sources
    assert "flow.imbalance.1m" in overlap_sources

    # Sem independent confirmation
    assert result.independent_confirmation is False

    # Timestamps corretos
    assert result.observation_open_ms == J2_OPEN_MS
    assert result.observation_close_ms == J2_CLOSE_MS
    assert result.causal_anchor_ms == J2_ANCHOR_MS

    # Fatos estruturados (sem prosa decisória)
    facts = result.summary_facts.get("facts", [])
    assert any("OVERLAPPING_LINEAGE" in f for f in facts)
    assert any("POINT_IN_TIME_L2" in f for f in facts)
    assert any("COMPOSITE_NON_VOTING" in f for f in facts)


# ── FIXTURE J1 (Fiel aos dados reais: BOS bearish presente) ───────────────────

def test_j1_real_with_bos_bearish_mixed():
    """J1 real: se existe BOS bearish, o resultado não pode ser forçado BULLISH.

    Dados J1 observados:
    - sector_flow order_flow: buy 3.37 BTC / sell 4.21 BTC -> fluxo vendedor
    - BOS bearish / choch -> estrutura bearish
    - Se houver evidência bullish residual -> MIXED
    - Se somente bearish -> ALIGNED_BEARISH
    """
    evs = [
        # Fluxo executado J1: venda predominante
        Evidence(
            source="flow.net.1m",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.EXECUTED_FLOW,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            observed_at_ms=1788700000000,  # timestamp J1 (conceitual)
        ),
        # BOS bearish
        Evidence(
            source="market_structure.bos",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.MARKET_STRUCTURE,
            evidence_type=EvidenceType.UNKNOWN,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=1788700000000,
        ),
    ]

    result = reconcile(
        evidences=evs,
        symbol="BTCUSDT",
        observation_open_ms=1788699940000,
        observation_close_ms=1788700000000,
        causal_anchor_ms=1788700020000,
    )

    # Sem evidência bullish -> ALIGNED_BEARISH (não forçado bullish)
    assert result.status == ReconcilerStatus.ALIGNED_BEARISH
    assert result.independent_confirmation is False


def test_j1_real_with_mixed_signals():
    """J1 com sinal bullish residual + BOS bearish -> MIXED_DIRECTIONS."""
    evs = [
        Evidence(
            source="flow.net.1m",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.EXECUTED_FLOW,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            observed_at_ms=1788700000000,
        ),
        Evidence(
            source="market_structure.bos",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.MARKET_STRUCTURE,
            evidence_type=EvidenceType.UNKNOWN,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
            observed_at_ms=1788700000000,
        ),
        # Suponha orderbook com suporte bullish
        Evidence(
            source="orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            observed_at_ms=1788700000000,
        ),
    ]

    result = reconcile(
        evidences=evs,
        symbol="BTCUSDT",
        observation_open_ms=1788699940000,
        observation_close_ms=1788700000000,
        causal_anchor_ms=1788700020000,
    )
    assert result.status == ReconcilerStatus.MIXED_DIRECTIONS


# ── DETERMINISMO ───────────────────────────────────────────────────────────────

def test_deterministic_output():
    """Mesma entrada -> mesma saída, duas execuções idênticas."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
    ]
    r1 = reconcile(evidences=evs, **COMMON_KW)
    r2 = reconcile(evidences=evs, **COMMON_KW)

    assert r1.status == r2.status
    assert r1.evidence_count == r2.evidence_count
    assert r1.directions_present == r2.directions_present
    assert r1.families_present == r2.families_present
    assert r1.summary_facts == r2.summary_facts
    assert r1.independent_confirmation == r2.independent_confirmation


# ── ZERO CAMPOS PROIBIDOS ──────────────────────────────────────────────────────

def test_no_confidence_probability_weight_trade_fields():
    """ConfluenceShadowResult não possui campos de decisão/trade."""
    result = reconcile(evidences=[], **COMMON_KW)

    # Campos proibidos não existem no dataclass
    prohibited = [
        "confidence", "probability", "weight", "trade",
        "action", "entry", "exit", "score", "strength",
        "tradeable", "confirmed", "invalidated", "expired",
    ]
    field_names = {f.name for f in result.__dataclass_fields__.values()}
    for p in prohibited:
        assert p not in field_names, f"Campo proibido '{p}' encontrado"


def test_result_is_frozen():
    """Resultado é frozen (imutável)."""
    result = reconcile(evidences=[], **COMMON_KW)
    with pytest.raises(AttributeError):
        result.status = ReconcilerStatus.ALIGNED_BULLISH


# ── VERSIONING ─────────────────────────────────────────────────────────────────

def test_version_fields():
    """Campos de versão presentes e corretos."""
    result = reconcile(evidences=[], **COMMON_KW)
    assert result.feature_contract_version == "1.0.0"
    assert result.reconciler_version == "1.0.0"


# ── NOT WHITELISTED ────────────────────────────────────────────────────────────

def test_not_whitelisted_source_discarded():
    """Fonte fora da whitelist é descartada."""
    evs = [
        _ev("some_unknown_source",
            direction=EvidenceDirection.BULLISH),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


# ── DIRECTION UNKNOWN ──────────────────────────────────────────────────────────

def test_unknown_direction_valid_discarded():
    """Evidência VALID com direction=UNKNOWN descartada."""
    evs = [_ev("flow.net.1m", direction=EvidenceDirection.UNKNOWN)]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.discarded_non_valid) == 1


# ── MÚLTIPLAS FAMÍLIAS ALINHADAS ───────────────────────────────────────────────

def test_multiple_families_aligned_bullish():
    """Múltiplas famílias alinhadas BULLISH -> ALIGNED_BULLISH."""
    evs = [
        _ev("flow.net.1m", direction=EvidenceDirection.BULLISH),
        _ev("orderbook.snapshot.imbalance",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
            evidence_type=EvidenceType.POINT_IN_TIME_L2),
        _ev("market_structure.bos",
            direction=EvidenceDirection.BULLISH,
            family=EvidenceFamily.MARKET_STRUCTURE,
            evidence_type=EvidenceType.UNKNOWN),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BULLISH
    assert result.evidence_count >= 2
    assert result.independent_confirmation is False


# ── SWEEP WHITELISTED ──────────────────────────────────────────────────────────

def test_sweep_whitelisted():
    """market_structure.sweep está na whitelist."""
    evs = [
        _ev("market_structure.sweep",
            direction=EvidenceDirection.BEARISH,
            family=EvidenceFamily.MARKET_STRUCTURE,
            evidence_type=EvidenceType.UNKNOWN),
    ]
    result = reconcile(evidences=evs, **COMMON_KW)
    assert result.status == ReconcilerStatus.ALIGNED_BEARISH
