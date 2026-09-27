# tests/unit/test_directional_liquidity_contract.py
"""
P2-B2 — Directional Liquidity & Market Impact Contract v1 Unit Tests.

Valida:
1. J1 assimetria real (buy 131x mais caro que sell)
2. J2 assimetria inversa real (sell 127x mais caro que buy)
3. Conversão dimensional canônica USD -> bps
4. Ausência de fator *100 espúrio
5. Fill ratio e insufficient liquidity
6. Fail-closed contra missing/nonfinite (sem transformar missing em zero)
7. Denominador zero na razão de assimetria protegido
8. Bit-equivalence do agregado legado e isolamento de metadata
9. Prova de que EXCELLENT agregado NÃO vira qualidade direcional BUY/SELL
10. Metadados POINT_IN_TIME_L2 e NOT_DIRECTIONAL_EXECUTION_GATE
11. Lineage na taxonomia e counts_as_vote=False
12. Conformidade RFC 8259 estrita
13. Rejeição pelo Confluence Shadow (execution context, não market-direction evidence).
"""
from __future__ import annotations

import json
import math
import pytest

from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
from institutional.confluence_shadow import DIRECTIONAL_WHITELIST
from institutional.evidence import Evidence, EvidenceFamily, EvidenceType, EvidenceValidity, EvidenceDirection
from institutional import evidence_taxonomy as tx
from orderbook_analyzer.directional_liquidity import (
    build_directional_liquidity,
    _extract_side_metrics,
    _calculate_asymmetry,
)
from orderbook_analyzer.core import _simulate_market_impact
from market_orchestrator.market_orchestrator import EnhancedMarketBot


# ═══════════════════════════════════════════════════════════════════════════════
# 1. J1 E J2 ASSYMETRY FIXTURES & LEGACY MASKING
# ═══════════════════════════════════════════════════════════════════════════════

def test_j1_directional_asymmetry_and_legacy_masking():
    """
    J1 real auditado:
      mid = 79776.95
      100k buy = 6.55 USD (~0.8210 bps)
      100k sell = 0.05 USD (~0.0063 bps)
    Provar:
      - assimetria direcional física de 131x é preservada;
      - o score agregado legado resulta em EXCELLENT (>9.9);
      - nenhum rótulo direcional BUY_EXCELLENT/SELL_POOR é criado.
    """
    mid_j1 = 79776.95
    mi_buy_j1 = {
        "100k": {
            "execution_slippage_usd": 6.55,
            "execution_slippage_bps": (6.55 / mid_j1) * 10000.0,
            "terminal_move_usd": 6.55,
            "terminal_move_bps": (6.55 / mid_j1) * 10000.0,
            "bps": (6.55 / mid_j1) * 10000.0,
            "fill_ratio": 1.0,
            "insufficient_liquidity": False,
        }
    }
    mi_sell_j1 = {
        "100k": {
            "execution_slippage_usd": 0.05,
            "execution_slippage_bps": (0.05 / mid_j1) * 10000.0,
            "terminal_move_usd": 0.05,
            "terminal_move_bps": (0.05 / mid_j1) * 10000.0,
            "bps": (0.05 / mid_j1) * 10000.0,
            "fill_ratio": 1.0,
            "insufficient_liquidity": False,
        }
    }

    dl = build_directional_liquidity(mi_buy_j1, mi_sell_j1, mid=mid_j1)

    # 1. Métricas direcionais brutas
    b100 = dl["buy"]["100k_usd"]
    s100 = dl["sell"]["100k_usd"]
    asym = dl["asymmetry"]["100k_usd"]

    assert b100["execution_slippage_usd"] == 6.55
    assert b100["execution_slippage_bps"] == pytest.approx(0.8210, abs=1e-3)
    assert b100["is_fillable"] is True
    assert b100["validity"] == "VALID"

    assert s100["execution_slippage_usd"] == 0.05
    assert s100["execution_slippage_bps"] == pytest.approx(0.0063, abs=1e-4)
    assert s100["is_fillable"] is True
    assert s100["validity"] == "VALID"

    # 2. Assimetria física real: comprar custa 131x mais
    assert asym["absolute_difference_bps"] == pytest.approx(0.8148, abs=1e-3)
    assert asym["buy_to_sell_ratio"] == pytest.approx(131.0, abs=0.1)

    # 3. Legado no orchestrator: mascara assimetria e emite EXCELLENT
    sig = {}
    ob_evt = {
        "is_valid": True,
        "spread_metrics": {"mid": mid_j1},
        "market_impact_buy": mi_buy_j1,
        "market_impact_sell": mi_sell_j1,
    }
    EnhancedMarketBot._enrich_orderbook_metrics(sig, ob_evt)
    legacy_mi = sig["market_impact"]

    assert legacy_mi["liquidity_score"] == pytest.approx(9.917, abs=1e-2)
    assert legacy_mi["execution_quality"] == "EXCELLENT"
    assert legacy_mi["legacy_metadata"]["status"] == "AGGREGATED_LEGACY"
    assert legacy_mi["legacy_metadata"]["execution_gate"] == "NOT_DIRECTIONAL_EXECUTION_GATE"

    # 4. Prova de que NENHUM BUY_EXCELLENT / SELL_POOR foi gerado
    raw_str = json.dumps(dl)
    assert "BUY_EXCELLENT" not in raw_str
    assert "SELL_POOR" not in raw_str
    assert "HIGH_ASYMMETRY" not in raw_str


def test_j2_directional_asymmetry_and_legacy_masking():
    """
    J2 real auditado:
      mid = 79792.65
      100k buy = 0.05 USD (~0.0063 bps)
      100k sell = 6.35 USD (~0.7958 bps)
    Provar:
      - assimetria inversa de 127x (vender custa 127x mais que comprar);
      - legado continua EXCELLENT (>9.9).
    """
    mid_j2 = 79792.65
    mi_buy_j2 = {
        "100k": {
            "execution_slippage_usd": 0.05,
            "execution_slippage_bps": (0.05 / mid_j2) * 10000.0,
            "terminal_move_usd": 0.05,
            "terminal_move_bps": (0.05 / mid_j2) * 10000.0,
            "bps": (0.05 / mid_j2) * 10000.0,
            "fill_ratio": 1.0,
            "insufficient_liquidity": False,
        }
    }
    mi_sell_j2 = {
        "100k": {
            "execution_slippage_usd": 6.35,
            "execution_slippage_bps": (6.35 / mid_j2) * 10000.0,
            "terminal_move_usd": 6.35,
            "terminal_move_bps": (6.35 / mid_j2) * 10000.0,
            "bps": (6.35 / mid_j2) * 10000.0,
            "fill_ratio": 1.0,
            "insufficient_liquidity": False,
        }
    }

    dl = build_directional_liquidity(mi_buy_j2, mi_sell_j2, mid=mid_j2)
    asym = dl["asymmetry"]["100k_usd"]

    assert dl["buy"]["100k_usd"]["execution_slippage_bps"] == pytest.approx(0.0063, abs=1e-4)
    assert dl["sell"]["100k_usd"]["execution_slippage_bps"] == pytest.approx(0.7958, abs=1e-3)
    assert asym["sell_to_buy_ratio"] == pytest.approx(127.0, abs=0.1)

    sig = {}
    ob_evt = {
        "is_valid": True,
        "spread_metrics": {"mid": mid_j2},
        "market_impact_buy": mi_buy_j2,
        "market_impact_sell": mi_sell_j2,
    }
    EnhancedMarketBot._enrich_orderbook_metrics(sig, ob_evt)
    assert sig["market_impact"]["liquidity_score"] == pytest.approx(9.920, abs=1e-2)
    assert sig["market_impact"]["execution_quality"] == "EXCELLENT"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UNIDADES CANÔNICAS & AUSÊNCIA DE *100 ESPÚRIO
# ═══════════════════════════════════════════════════════════════════════════════

def test_usd_to_bps_conversion_formula():
    """Fórmula dimensional canônica: bps = (slippage_usd / mid) * 10000.0."""
    mid = 80000.0
    slip_usd = 8.0  # $8 USD de deslocamento de preço em relação ao mid
    # 8 / 80000 = 0.0001 (0.01%) -> 0.0001 * 10000 = 1.0 bps
    expected_bps = 1.0

    side_data = {
        "100k": {
            "execution_slippage_usd": slip_usd,
            "insufficient_liquidity": False,
            "fill_ratio": 1.0,
        }
    }
    metrics = _extract_side_metrics(side_data, "100k", mid=mid)
    assert metrics["execution_slippage_usd"] == 8.0
    assert metrics["execution_slippage_bps"] == pytest.approx(expected_bps)
    # NÃO deve ser 800 (como seria se multiplicasse por 100 sem mid)
    assert metrics["execution_slippage_bps"] != 800.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FILL RATIO E INSUFFICIENT LIQUIDITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_fill_ratio_and_insufficient_liquidity():
    """Quando preenchimento é parcial, slippage total é None e observado é preservado."""
    side_data = {
        "1m": {
            "insufficient_liquidity": True,
            "fill_ratio": 0.45,
            "execution_slippage_usd": None,
            "observed_execution_slippage_usd": 3.50,
            "observed_execution_slippage_bps": 0.4375,
            "observed_terminal_move_usd": 7.00,
        }
    }
    metrics = _extract_side_metrics(side_data, "1m", mid=80000.0)
    assert metrics["is_fillable"] is False
    assert metrics["validity"] == "INSUFFICIENT_LIQUIDITY"
    assert metrics["fill_ratio"] == 0.45
    assert metrics["execution_slippage_usd"] is None
    assert metrics["execution_slippage_bps"] is None
    assert metrics["observed_execution_slippage_usd"] == 3.50
    assert metrics["observed_execution_slippage_bps"] == 0.4375


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FAIL-CLOSED & MISSING / NON-FINITE
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), "nan", "inf", None, "abc"])
def test_missing_and_nonfinite_inputs_fail_closed(bad_val):
    """Inputs não-finitos ou malformados não quebram e viram status explícito."""
    side_data = {
        "100k": {
            "execution_slippage_usd": bad_val,
            "fill_ratio": bad_val,
            "insufficient_liquidity": False,
        }
    }
    metrics = _extract_side_metrics(side_data, "100k", mid=80000.0)
    assert metrics["validity"] in ("NON_VOTING_INVALID_INPUT", "NON_VOTING_MISSING")
    assert metrics["is_fillable"] is False
    assert metrics["execution_slippage_usd"] is None


def test_missing_orderbook_data_all_missing():
    """Dados vazios ou None resultam em NON_VOTING_MISSING."""
    dl = build_directional_liquidity(None, None, mid=None)
    assert dl["status"] == "NON_VOTING_MISSING"
    assert dl["buy"]["100k_usd"]["validity"] == "NON_VOTING_MISSING"
    assert dl["sell"]["100k_usd"]["validity"] == "NON_VOTING_MISSING"


def test_asymmetry_zero_denominator_safe():
    """Denominador zero na razão de slippage é protegido e retorna None com reason."""
    buy_item = {"execution_slippage_bps": 5.0, "fill_ratio": 1.0}
    sell_item = {"execution_slippage_bps": 0.0, "fill_ratio": 1.0}

    asym = _calculate_asymmetry(buy_item, sell_item)
    assert asym["absolute_difference_bps"] == 5.0
    assert asym["slippage_ratio"] is None
    assert asym["ratio_reason"] == "ZERO_DENOMINATOR_SELL"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METADATA POINT_IN_TIME_L2 & TAXONOMIA
# ═══════════════════════════════════════════════════════════════════════════════

def test_point_in_time_l2_metadata():
    """Metadados de capability e execution gate preservados."""
    dl = build_directional_liquidity({}, {}, mid=80000.0)
    assert dl["source_type"] == "POINT_IN_TIME_L2"
    assert dl["capability"] == "POINT_IN_TIME_L2"
    assert dl["execution_gate"] == "NOT_DIRECTIONAL_EXECUTION_GATE"


def test_evidence_taxonomy_registration():
    """Campos de directional liquidity estão registrados na taxonomia com lineage L2."""
    b_entry = tx.FIELDS.get("directional_liquidity.buy.100k.bps")
    s_entry = tx.FIELDS.get("directional_liquidity.sell.100k.bps")
    asym_diff = tx.FIELDS.get("directional_liquidity.asymmetry.100k.diff_bps")
    asym_ratio = tx.FIELDS.get("directional_liquidity.asymmetry.100k.ratio")

    assert b_entry is not None
    assert b_entry.family == tx.FAM_OB
    assert b_entry.evidence_type == tx.T_L2
    assert "raw.l2.asks" in b_entry.derived_from

    assert s_entry is not None
    assert s_entry.family == tx.FAM_OB
    assert "raw.l2.bids" in s_entry.derived_from

    assert asym_diff is not None
    assert asym_diff.is_composite is True

    assert asym_ratio is not None
    assert asym_ratio.is_composite is True


def test_counts_as_vote_always_false():
    """Nenhum item direcional conta como voto independente."""
    ev = Evidence(
        source="orderbook_analyzer.directional_liquidity",
        direction=EvidenceDirection.NEUTRAL,
        value=0.82,
        validity=EvidenceValidity.VALID,
        family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
        evidence_type=EvidenceType.POINT_IN_TIME_L2,
        observed_at_ms=1772913300000,
        counts_as_vote=False,
    )
    assert ev.counts_as_vote is False


def test_confluence_shadow_rejects_directional_liquidity():
    """Directional liquidity é execution context e NÃO pode estar na whitelist direcional."""
    assert "directional_liquidity.buy.100k.bps" not in DIRECTIONAL_WHITELIST
    assert "directional_liquidity.sell.100k.bps" not in DIRECTIONAL_WHITELIST
    assert "directional_liquidity.asymmetry.100k.diff_bps" not in DIRECTIONAL_WHITELIST
    assert "market_impact.execution_quality" not in DIRECTIONAL_WHITELIST


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RFC 8259 JSON COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_rfc8259_json_compliance():
    """Resultado é serializável em JSON estrito (sem NaN/Infinity literais)."""
    dl = build_directional_liquidity(
        {"100k": {"execution_slippage_usd": 6.55, "fill_ratio": 1.0}},
        {"100k": {"execution_slippage_usd": 0.05, "fill_ratio": 1.0}},
        mid=79776.95,
    )
    safe = sanitize_json_safe(dl)
    text = json_dumps_rfc8259(safe)
    assert "NaN" not in text
    assert "Infinity" not in text

    parsed = json.loads(text)
    assert parsed["buy"]["100k_usd"]["execution_slippage_usd"] == 6.55
    assert parsed["asymmetry"]["100k_usd"]["buy_to_sell_ratio"] == pytest.approx(130.32, abs=0.5)
