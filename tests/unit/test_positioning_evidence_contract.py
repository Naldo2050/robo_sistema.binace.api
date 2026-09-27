# tests/unit/test_positioning_evidence_contract.py
# -*- coding: utf-8 -*-
"""
Testes unitários para P2-C — OI / Positioning Evidence Contract v1.

Cobre rigorosamente:
1. Valid OI (unidade contracts)
2. Valid OI USD (unidade USD)
3. Missing OI != 0
4. Zero observado de funding (VALID_ZERO_OBSERVED) vs missing funding
5. Valid LSR (accounts e positions)
6. Missing LSR != 0
7. Non-finite / NaN / Inf / malformed (fail-closed)
8. Stale cache (>900s ou flag is_stale) — STALE nunca vira 0
9. Partial snapshot
10. Funding real zero vs funding missing
11. Evidence counts_as_vote=False
12. Evidence direction=UNKNOWN
13. Provenance e linhagem
14. Unidades explícitas e diferenciadas
15. RFC 8259 serialização JSON determinística
16. J2 positioning unavailable honesto (MISSING/NON_VOTING)
17. Whale derivatives ignora stale/missing sem alterar score
18. Integração com BinancePositioningSnapshot e positioning_from_event
"""

import json
import math
import pytest

from fetchers.binance_positioning_fetcher import BinancePositioningSnapshot
from flow_analyzer.whale_score import WhaleAccumulationCalculator
from institutional.evidence import (
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
)
from institutional.positioning_contract import (
    DEFAULT_CACHE_TTL_MS,
    DEFAULT_MAX_STALE_MS,
    POSITIONING_CONTRACT_VERSION,
    PositioningField,
    PositioningFieldValidity,
    PositioningSnapshot,
    PositioningSnapshotStatus,
    build_positioning_snapshot,
    positioning_from_event,
    positioning_snapshot_to_evidence,
)


# ── 1. VALIDAÇÃO DE OI EM CONTRATOS E EM USD ──────────────────────────────────

def test_valid_open_interest_contracts_and_usd():
    """Valida OI em contratos (base asset) e OI em USD com unidades explícitas."""
    pos_data = {
        "open_interest": 85432.125,
        "open_interest_usd": 5_980_250_000.0,
        "global_account_ratio": 1.25,
        "top_position_ratio": 1.40,
        "funding_rate": 0.0001,
        "source_timestamp": 1720000000000,
        "is_available": True,
    }
    now_ms = 1720000060000  # 60s depois (fresco)

    snap = build_positioning_snapshot(pos_data, now_ms=now_ms)

    assert snap.status == PositioningSnapshotStatus.AVAILABLE
    assert snap.symbol == "BTCUSDT"

    # OI em contratos
    assert snap.open_interest.value == 85432.125
    assert snap.open_interest.unit == "contracts"
    assert snap.open_interest.validity == PositioningFieldValidity.VALID
    assert snap.open_interest.age_ms == 60_000
    assert snap.open_interest.observed_at_ms == 1720000000000

    # OI em USD
    assert snap.open_interest_usd.value == 5_980_250_000.0
    assert snap.open_interest_usd.unit == "USD"
    assert snap.open_interest_usd.validity == PositioningFieldValidity.VALID
    assert snap.open_interest_usd.age_ms == 60_000


def test_missing_open_interest_is_not_zero():
    """Missing OI NUNCA deve virar 0.0 numérico silencioso."""
    pos_data = {
        "open_interest": None,
        "open_interest_usd": None,
        "global_account_ratio": 1.20,
        "top_position_ratio": 1.30,
        "funding_rate": 0.0001,
    }
    snap = build_positioning_snapshot(pos_data)

    assert snap.open_interest.value is None
    assert snap.open_interest.validity == PositioningFieldValidity.MISSING
    assert snap.open_interest.reason == "OI_MISSING"

    assert snap.open_interest_usd.value is None
    assert snap.open_interest_usd.validity == PositioningFieldValidity.MISSING
    assert snap.open_interest_usd.reason == "OI_USD_MISSING"

    # Snapshot fica PARTIAL porque outros campos existem
    assert snap.status == PositioningSnapshotStatus.PARTIAL


# ── 2. FUNDING RATE: ZERO OBSERVADO VS MISSING ────────────────────────────────

def test_zero_observed_funding_rate_vs_missing_funding():
    """Funding 0.0 real observado vira VALID_ZERO_OBSERVED; ausente vira MISSING."""
    # Caso A: funding 0.0 explícito
    snap_zero = build_positioning_snapshot(
        funding_data=0.0,
        positioning_data={"is_available": True},
    )
    assert snap_zero.funding_rate.value == 0.0
    assert snap_zero.funding_rate.unit == "rate_8h"
    assert snap_zero.funding_rate.validity == PositioningFieldValidity.VALID_ZERO_OBSERVED
    assert snap_zero.funding_rate.reason == "VALID_ZERO_OBSERVED"

    # Caso B: funding ausente (None)
    snap_missing = build_positioning_snapshot(
        funding_data=None,
        positioning_data={"is_available": True},
    )
    assert snap_missing.funding_rate.value is None
    assert snap_missing.funding_rate.validity == PositioningFieldValidity.MISSING
    assert snap_missing.funding_rate.reason == "FUNDING_MISSING"

    # Caso C: funding positivo real (ex: 0.0001)
    snap_pos = build_positioning_snapshot(
        funding_data=0.0001,
        positioning_data={"is_available": True},
    )
    assert snap_pos.funding_rate.value == 0.0001
    assert snap_pos.funding_rate.validity == PositioningFieldValidity.VALID

    # Caso D: funding vindo como percentual em derivatives_data (ex: 0.01% -> 0.0001)
    snap_pct = build_positioning_snapshot(
        derivatives_data={"funding_rate_percent": 0.01},
        positioning_data={"is_available": True},
    )
    assert snap_pct.funding_rate.value == 0.0001
    assert snap_pct.funding_rate.validity == PositioningFieldValidity.VALID


# ── 3. LONG/SHORT RATIOS: ACCOUNTS E POSITIONS ─────────────────────────────────

def test_valid_and_missing_long_short_ratios():
    """Valida separação semântica e ausência de ratios."""
    # Válidos
    snap = build_positioning_snapshot(
        positioning_data={
            "global_account_ratio": 1.35,
            "top_position_ratio": 1.65,
            "is_available": True,
        }
    )
    assert snap.long_short_accounts_ratio.value == 1.35
    assert snap.long_short_accounts_ratio.unit == "ratio"
    assert snap.long_short_accounts_ratio.validity == PositioningFieldValidity.VALID

    assert snap.long_short_positions_ratio.value == 1.65
    assert snap.long_short_positions_ratio.unit == "ratio"
    assert snap.long_short_positions_ratio.validity == PositioningFieldValidity.VALID

    # Ausentes
    snap_miss = build_positioning_snapshot(
        positioning_data={
            "global_account_ratio": None,
            "top_position_ratio": None,
            "is_available": True,
        }
    )
    assert snap_miss.long_short_accounts_ratio.value is None
    assert snap_miss.long_short_accounts_ratio.validity == PositioningFieldValidity.MISSING
    assert snap_miss.long_short_accounts_ratio.reason == "LSR_ACCOUNTS_MISSING"

    assert snap_miss.long_short_positions_ratio.value is None
    assert snap_miss.long_short_positions_ratio.validity == PositioningFieldValidity.MISSING
    assert snap_miss.long_short_positions_ratio.reason == "LSR_POSITIONS_MISSING"

    # Inválido: ratio negativo ou zero
    snap_inv = build_positioning_snapshot(
        positioning_data={
            "global_account_ratio": -0.5,
            "top_position_ratio": 0.0,
            "is_available": True,
        }
    )
    assert snap_inv.long_short_accounts_ratio.validity == PositioningFieldValidity.INVALID
    assert snap_inv.long_short_accounts_ratio.reason == "NON_POSITIVE_RATIO"
    assert snap_inv.long_short_positions_ratio.validity == PositioningFieldValidity.INVALID
    assert snap_inv.long_short_positions_ratio.reason == "NON_POSITIVE_RATIO"


# ── 4. NON-FINITE (NaN / Inf / MALFORMED) ──────────────────────────────────────

def test_non_finite_nan_and_inf_handling():
    """Valores non-finite são coagidos para None com validity=INVALID."""
    pos_data = {
        "open_interest": float("nan"),
        "open_interest_usd": float("inf"),
        "global_account_ratio": float("-inf"),
        "top_position_ratio": "not_a_number",
        "funding_rate": True,  # boolean é rejeitado
        "is_available": True,
    }
    snap = build_positioning_snapshot(pos_data)

    assert snap.status == PositioningSnapshotStatus.INVALID
    assert snap.open_interest.value is None
    assert snap.open_interest.validity == PositioningFieldValidity.INVALID
    assert snap.open_interest.reason == "OI_NONFINITE_OR_MALFORMED"

    assert snap.open_interest_usd.value is None
    assert snap.open_interest_usd.validity == PositioningFieldValidity.INVALID

    assert snap.long_short_accounts_ratio.value is None
    assert snap.long_short_accounts_ratio.validity == PositioningFieldValidity.INVALID

    assert snap.long_short_positions_ratio.value is None
    assert snap.long_short_positions_ratio.validity == PositioningFieldValidity.INVALID

    assert snap.funding_rate.value is None
    assert snap.funding_rate.validity == PositioningFieldValidity.INVALID


# ── 5. FRESHNESS E STALE CACHE ────────────────────────────────────────────────

def test_stale_cache_and_freshness_age():
    """Dados mais velhos que max_stale_ms (>900s) viram STALE sem zerar valor."""
    t0 = 1720000000000
    pos_data = {
        "open_interest": 50000.0,
        "open_interest_usd": 3_500_000_000.0,
        "global_account_ratio": 1.10,
        "top_position_ratio": 1.15,
        "funding_rate": 0.0001,
        "source_timestamp": t0,
        "is_available": True,
    }

    # 1. 200 segundos depois (dentro do cache de 300s): FRESH / VALID
    snap_fresh = build_positioning_snapshot(pos_data, now_ms=t0 + 200_000)
    assert snap_fresh.status == PositioningSnapshotStatus.AVAILABLE
    assert snap_fresh.open_interest.validity == PositioningFieldValidity.VALID
    assert snap_fresh.open_interest.age_ms == 200_000

    # 2. 1000 segundos depois (> 900s max_stale): STALE
    snap_stale = build_positioning_snapshot(pos_data, now_ms=t0 + 1_000_000)
    assert snap_stale.status == PositioningSnapshotStatus.STALE
    assert "STALE_DATA" in snap_stale.reasons

    # STALE NUNCA vira 0: o valor histórico é rigorosamente mantido
    assert snap_stale.open_interest.value == 50000.0
    assert snap_stale.open_interest.validity == PositioningFieldValidity.STALE
    assert snap_stale.open_interest.reason == "STALE_DATA"
    assert snap_stale.open_interest.age_ms == 1_000_000

    assert snap_stale.funding_rate.value == 0.0001
    assert snap_stale.funding_rate.validity == PositioningFieldValidity.STALE

    # 3. Flag explícito is_stale vindo do fetcher
    pos_data_flag = dict(pos_data, is_stale=True)
    snap_flag_stale = build_positioning_snapshot(pos_data_flag, now_ms=t0 + 50_000)
    assert snap_flag_stale.status == PositioningSnapshotStatus.STALE
    assert snap_flag_stale.open_interest.validity == PositioningFieldValidity.STALE


# ── 6. PARTIAL SNAPSHOT ───────────────────────────────────────────────────────

def test_partial_snapshot():
    """Snapshot com apenas parte dos campos válidos tem status PARTIAL."""
    pos_data = {
        "open_interest": 42000.0,
        # open_interest_usd ausente
        "global_account_ratio": 1.05,
        # top_position_ratio ausente
        "funding_rate": 0.0002,
        "is_available": True,
    }
    snap = build_positioning_snapshot(pos_data)

    assert snap.status == PositioningSnapshotStatus.PARTIAL
    assert snap.open_interest.validity == PositioningFieldValidity.VALID
    assert snap.open_interest_usd.validity == PositioningFieldValidity.MISSING
    assert snap.long_short_accounts_ratio.validity == PositioningFieldValidity.VALID
    assert snap.long_short_positions_ratio.validity == PositioningFieldValidity.MISSING
    assert snap.funding_rate.validity == PositioningFieldValidity.VALID


# ── 7. ADAPTER EVIDENCE V1 ────────────────────────────────────────────────────

def test_evidence_adapters_non_voting_and_direction_unknown():
    """Evidências geradas devem ter counts_as_vote=False, Direction=UNKNOWN e Family=DERIVATIVES."""
    pos_data = {
        "open_interest": 60000.0,
        "open_interest_usd": 4_200_000_000.0,
        "global_account_ratio": 1.40,
        "top_position_ratio": 1.80,
        "funding_rate": 0.0,
        "source_timestamp": 1720000000000,
        "is_available": True,
    }
    snap = build_positioning_snapshot(pos_data, now_ms=1720000010000)
    evidences = positioning_snapshot_to_evidence(snap)

    assert len(evidences) == 5

    field_ids = [ev.metadata["field_id"] for ev in evidences]
    assert field_ids == [
        "derivatives.open_interest",
        "derivatives.open_interest_usd",
        "derivatives.long_short_accounts_ratio",
        "derivatives.long_short_positions_ratio",
        "derivatives.funding_rate",
    ]

    for ev in evidences:
        assert ev.family == EvidenceFamily.DERIVATIVES
        assert ev.evidence_type == EvidenceType.SLOW_CONTEXT
        # Estritamente sem viés nem voto
        assert ev.counts_as_vote is False
        assert ev.direction == EvidenceDirection.UNKNOWN
        assert ev.calibration == EvidenceCalibration.NOT_APPLICABLE
        assert ev.source == "binance_usdm"
        assert ev.provenance["exchange"] == "binance"
        assert ev.provenance["market"] == "usdm_futures"

    # Checar valores específicos
    ev_oi = evidences[0]
    assert ev_oi.value == 60000.0
    assert ev_oi.validity == EvidenceValidity.VALID
    assert ev_oi.metadata["unit"] == "contracts"

    ev_oi_usd = evidences[1]
    assert ev_oi_usd.value == 4_200_000_000.0
    assert ev_oi_usd.validity == EvidenceValidity.VALID
    assert ev_oi_usd.metadata["unit"] == "USD"

    ev_funding = evidences[4]
    assert ev_funding.value == 0.0
    assert ev_funding.validity == EvidenceValidity.VALID
    assert ev_funding.reason == "VALID_ZERO_OBSERVED"
    assert ev_funding.metadata["unit"] == "rate_8h"


def test_evidence_adapters_stale_and_invalid():
    """Campos stale geram EvidenceValidity.STALE; inválidos geram INVALID com value=None."""
    pos_data = {
        "open_interest": 50000.0,
        "open_interest_usd": float("nan"),
        "global_account_ratio": 1.2,
        "is_stale": True,
        "is_available": True,
    }
    snap = build_positioning_snapshot(pos_data)
    evidences = positioning_snapshot_to_evidence(snap)

    ev_oi = next(e for e in evidences if e.metadata["field_id"] == "derivatives.open_interest")
    assert ev_oi.validity == EvidenceValidity.STALE
    assert ev_oi.value == 50000.0

    ev_oi_usd = next(e for e in evidences if e.metadata["field_id"] == "derivatives.open_interest_usd")
    assert ev_oi_usd.validity == EvidenceValidity.INVALID
    assert ev_oi_usd.value is None


# ── 8. UNIDADES EXPLÍCITAS ────────────────────────────────────────────────────

def test_units_are_explicit_and_differentiated():
    """Garante que as unidades estão formalmente tipadas e diferenciadas."""
    snap = build_positioning_snapshot({
        "open_interest": 100.0,
        "open_interest_usd": 7_000_000.0,
        "global_account_ratio": 1.1,
        "top_position_ratio": 1.2,
        "funding_rate": 0.0001,
        "is_available": True,
    })
    assert snap.open_interest.unit == "contracts"
    assert snap.open_interest_usd.unit == "USD"
    assert snap.long_short_accounts_ratio.unit == "ratio"
    assert snap.long_short_positions_ratio.unit == "ratio"
    assert snap.funding_rate.unit == "rate_8h"


# ── 9. SERIALIZAÇÃO DETERMINÍSTICA RFC 8259 ───────────────────────────────────

def test_rfc8259_serialization_json_safe():
    """Valida serialização JSON determinística e segura sem floats anômalos."""
    snap = build_positioning_snapshot({
        "open_interest": float("nan"),
        "open_interest_usd": float("inf"),
        "global_account_ratio": 1.25,
        "is_available": True,
    })
    d = snap.to_dict()

    # json.dumps não deve falhar com ValueError (Out of range float values are not JSON compliant)
    dumped = json.dumps(d)
    parsed = json.loads(dumped)

    assert parsed["open_interest"]["value"] is None
    assert parsed["open_interest_usd"]["value"] is None
    assert parsed["long_short_accounts_ratio"]["value"] == 1.25
    assert parsed["contract_version"] == POSITIONING_CONTRACT_VERSION

    # Testar também to_dict() das evidências
    evidences = positioning_snapshot_to_evidence(snap)
    for ev in evidences:
        ev_dict = ev.to_dict()
        dumped_ev = json.dumps(ev_dict)
        assert json.loads(dumped_ev) is not None


# ── 10. J1/J2 POSITIONING UNAVAILABLE ─────────────────────────────────────────

def test_j2_real_positioning_unavailable_honest_representation():
    """J2 real tem positioning.is_available=0. Deve ser MISSING e NÃO NEUTRAL."""
    j2_pos_data = {
        "is_available": 0,
        "open_interest": 80000.0,  # mesmo se houvesse lixo, is_available=0 invalida
        "global_account_ratio": 1.5,
    }
    snap = build_positioning_snapshot(j2_pos_data)

    assert snap.status == PositioningSnapshotStatus.MISSING
    assert "POSITIONING_UNAVAILABLE" in snap.reasons

    # Todos os campos viram MISSING
    assert snap.open_interest.validity == PositioningFieldValidity.MISSING
    assert snap.open_interest.value is None
    assert snap.open_interest_usd.validity == PositioningFieldValidity.MISSING
    assert snap.long_short_accounts_ratio.validity == PositioningFieldValidity.MISSING
    assert snap.long_short_positions_ratio.validity == PositioningFieldValidity.MISSING
    assert snap.funding_rate.validity == PositioningFieldValidity.MISSING

    # Evidências viram UNKNOWN
    evidences = positioning_snapshot_to_evidence(snap)
    for ev in evidences:
        assert ev.validity == EvidenceValidity.UNKNOWN
        assert ev.value is None
        assert ev.direction == EvidenceDirection.UNKNOWN
        assert ev.counts_as_vote is False


# ── 11. ISOLAMENTO DO WHALE SCORE DERIVATIVES ─────────────────────────────────

def test_whale_derivatives_ignores_stale_and_missing():
    """Garante que derivativos ausentes ou stale não pontuam no WhaleAccumulationCalculator."""
    calc = WhaleAccumulationCalculator()

    # 1. Derivatives ausente (None / vazio): contribuição 0.0
    res_empty = calc.calculate(
        sector_flow={"whale": {"delta": 5.0}},
        orderbook_data={"bid_depth_usd": 1000.0, "ask_depth_usd": 1000.0},
        derivatives_data={},
    )
    deriv_comp = res_empty["components"]["derivatives"]
    assert deriv_comp["score"] == 0.0
    assert deriv_comp["max"] == 25
    assert deriv_comp["status"] == "NON_VOTING_MISSING"

    # 2. LSR e Funding com non-finite: contribuição 0.0
    res_nan = calc.calculate(
        sector_flow={"whale": {"delta": 5.0}},
        orderbook_data={"bid_depth_usd": 1000.0, "ask_depth_usd": 1000.0},
        derivatives_data={"BTCUSDT": {"long_short_ratio": float("nan"), "funding_rate": float("inf")}},
    )
    deriv_nan = res_nan["components"]["derivatives"]
    assert deriv_nan["score"] == 0.0
    assert deriv_nan["status"] == "NON_VOTING_INVALID_INPUT"

    # 3. Open interest no whale score é puramente informativo (sem peso)
    res_oi = calc.calculate(
        sector_flow={"whale": {"delta": 5.0}},
        orderbook_data={"bid_depth_usd": 1000.0, "ask_depth_usd": 1000.0},
        derivatives_data={"BTCUSDT": {"open_interest": 999999.0, "long_short_ratio": 1.0, "funding_rate": 0.0}},
    )
    deriv_oi = res_oi["components"]["derivatives"]
    assert deriv_oi["detail"]["open_interest"] == 999999.0
    # LSR=1.0 -> score 0.0; Funding=0.0 -> score 0.0
    assert deriv_oi["score"] == 0.0


# ── 12. INTEGRAÇÃO COM BINANCE POSITIONING SNAPSHOT ───────────────────────────

def test_binance_positioning_snapshot_integration():
    """Valida que aceita diretamente a dataclass BinancePositioningSnapshot."""
    fetcher_snap = BinancePositioningSnapshot(
        symbol="BTCUSDT",
        period="5m",
        observed_at=1720000000.0,
        source="binance_usdm",
        global_account_ratio=1.18,
        top_account_ratio=1.35,
        top_position_ratio=1.42,
        open_interest=75000.0,
        open_interest_usd=5_250_000_000.0,
        source_timestamp=1720000000000,
        is_available=True,
        is_stale=False,
    )

    contract_snap = build_positioning_snapshot(
        positioning_data=fetcher_snap,
        funding_data=0.0001,
        now_ms=1720000030000,
    )

    assert contract_snap.status == PositioningSnapshotStatus.AVAILABLE
    assert contract_snap.open_interest.value == 75000.0
    assert contract_snap.open_interest.unit == "contracts"
    assert contract_snap.open_interest_usd.value == 5_250_000_000.0
    assert contract_snap.open_interest_usd.unit == "USD"
    assert contract_snap.long_short_accounts_ratio.value == 1.18
    assert contract_snap.long_short_positions_ratio.value == 1.42
    assert contract_snap.funding_rate.value == 0.0001


# ── 13. HELPER POSITIONING FROM EVENT ─────────────────────────────────────────

def test_positioning_from_event_helper():
    """Valida extração completa a partir de um event dict legado/enriquecido."""
    event = {
        "symbol": "BTCUSDT",
        "institutional_analytics": {
            "positioning": {
                "global_account_ratio": 1.22,
                "top_position_ratio": 1.55,
                "open_interest": 82000.0,
                "open_interest_usd": 5_740_000_000.0,
                "is_available": True,
                "source_timestamp": 1720000000000,
            }
        },
        "derivatives": {
            "BTCUSDT": {
                "funding_rate_percent": 0.015,  # 0.015% -> 0.00015
            }
        }
    }

    snap = positioning_from_event(event, now_ms=1720000050000)

    assert snap.status == PositioningSnapshotStatus.AVAILABLE
    assert snap.open_interest.value == 82000.0
    assert snap.open_interest_usd.value == 5_740_000_000.0
    assert snap.long_short_accounts_ratio.value == 1.22
    assert snap.long_short_positions_ratio.value == 1.55
    assert snap.funding_rate.value == 0.00015
    assert snap.funding_rate.unit == "rate_8h"
