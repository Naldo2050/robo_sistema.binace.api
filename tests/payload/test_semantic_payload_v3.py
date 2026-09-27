# tests/payload/test_semantic_payload_v3.py
# -*- coding: utf-8 -*-
"""
Suíte de testes para P2-F2: Semantic LLM Payload v3 Contract.

Cobre:
- Root separation (5 seções estruturadas isoladas)
- Evidence representatives only (supressão de aliases redundantes exatos)
- Composites non-voting (whale, regime, positioning, macro, liquidations)
- Propagação de validade estrita
- Correção de unidades (slippage sem x100, funding decimal)
- Nomes heurísticos seguros (sem falsas probabilidades)
- Reconciliação forense J1 e J2
- Serialização determinística RFC 8259
- Regressão do payload compacto v2 legado.
"""

import json
import pytest

from institutional.confluence_shadow import ReconcilerStatus
from market_orchestrator.ai import payload_builder_compact as bcp
from market_orchestrator.ai.semantic_payload_builder_v3 import (
    SEMANTIC_PAYLOAD_VERSION,
    build_semantic_payload_v3,
    extract_evidence_from_event,
)


def make_full_test_event() -> dict:
    """Fixture completa de evento com todas as fontes contemporâneas."""
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "preco_fechamento": 66875.0,
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 16000.0,
                "net_flow_5m": 45000.0,
                "net_flow_15m": 78000.0,
                "flow_imbalance": 0.45,
                "buy_sell_ratio": {
                    "buy_sell_ratio": 2.63,
                },
            },
            "absorption_analysis": {
                "current_absorption": {
                    "label": "Forte Vendedora",  # Absorção de venda -> viés BULLISH
                    "index": 0.82,
                    "buyer_strength": 7.5,
                    "seller_exhaustion": 3.0,
                }
            },
            "flow_window_integrity": {
                "1m": {"status": "FULL", "effective_coverage_pct": 100.0},
                "5m": {"status": "FULL", "effective_coverage_pct": 100.0},
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 303000.0,
            "ask_depth_usd": 170000.0,
            "imbalance": 0.28,
            "spread_percent": 0.0015,
        },
        "market_impact": {
            "execution_quality": "EXCELLENT",
            "liquidity_score": 9.2,
            "slippage_matrix": {
                "100k_usd": {
                    "buy": 2.50,   # $2.50 reais de slippage (NÃO escalonado)
                    "sell": 1.80,  # $1.80 reais de slippage
                }
            },
            "fill_ratio_matrix": {
                "100k_usd": {
                    "buy": 1.0,
                    "sell": 1.0,
                }
            },
        },
        "institutional_analytics": {
            "flow_analysis": {
                "whale_accumulation": {
                    "score": 35,
                    "classification": "MILD_ACCUMULATION",
                    "smart_money_divergence": "smart_money_accumulation",
                }
            }
        },
        "regime_analysis": {
            "current_regime": "TRENDING",
            "regime_probabilities": {
                "trending": 0.70,
                "mean_reverting": 0.20,
                "breakout": 0.10,
            },
            "regime_change_probability": 0.15,
            "status": "VALID",
        },
        "ml_prediction": {
            "prob_up": 0.62,
            "confidence": 0.58,
            "ml_stale": False,
        },
        "derivatives": {
            "BTCUSDT": {
                "open_interest": 85000.0,
                "open_interest_usd": 5684375000.0,
                "long_short_ratio": 1.45,
                "funding_rate": 0.0001,
            }
        },
        "liquidation_telemetry": {
            "observed_notional_usd_long": 150000.0,
            "observed_notional_usd_short": 50000.0,
            "observed_notional_usd_total": 200000.0,
            "estimated_notional_usd_total": 0.0,
            "event_count": 8,
        },
        "macro_calendar_snapshot": {
            "provider_status": "AVAILABLE",
            "reference_time_ms": 1775173800000,
            "nearest_upcoming_event": {
                "event_type": "US_CPI",
                "scheduled_at_ms": 1775173800000 + 3600000,
                "scheduled_at_utc": "2026-04-02T14:30:00Z",
                "importance": "HIGH_IMPACT_TARGET_SET",
            },
        },
        "market_structure": {
            "bos": "BULL_66900",
            "sw": "SELL_66700",
        },
    }


# ==============================================================================
# 1. ROOT SEPARATION & VERSIONING
# ==============================================================================

def test_semantic_payload_v3_root_separation():
    """Valida o shape raiz e a separação semântica estrita das 5 seções."""
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    assert payload["semantic_payload_version"] == "3.0.0"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["market"] == "futures_perpetual"
    assert "as_of_ms" in payload

    # 5 seções estruturadas obrigatórias
    expected_sections = {
        "directional_evidence",
        "confluence_reconciler",
        "execution_context",
        "non_voting_context",
        "data_quality",
    }
    for sec in expected_sections:
        assert sec in payload, f"Seção raiz obrigatória ausente: {sec}"


# ==============================================================================
# 2. EVIDENCE REPRESENTATIVES ONLY & REDUNDANCY SUPPRESSION
# ==============================================================================

def test_directional_evidence_suppresses_redundant_aliases():
    """
    Garante que bsr (buy_sell_ratio) é suprimido na lista de evidências
    quando flow.imbalance.1m está presente, evitando double counting.
    """
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    dir_evidence = payload["directional_evidence"]
    field_ids = [item["field_id"] for item in dir_evidence["items"]]

    # flow.net.1m é o representante mantido do grupo _AGG
    assert "flow.net.1m" in field_ids

    # flow.imbalance.1m e flow.buy_sell_ratio devem ser suprimidos como redundâncias exatas
    assert "flow.imbalance.1m" not in field_ids
    assert "flow.buy_sell_ratio" not in field_ids
    assert "flow.imbalance.1m" in dir_evidence["suppressed_redundant_aliases"]
    assert "flow.buy_sell_ratio" in dir_evidence["suppressed_redundant_aliases"]


# ==============================================================================
# 3. COMPOSITES & NON-VOTING CONTEXT
# ==============================================================================

def test_composites_are_non_voting_and_explicitly_flagged():
    """
    Verifica que whale, regime, positioning, ML, macro e liquidations
    possuem counts_as_vote=False e rótulos de calibração explícitos.
    """
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    nv = payload["non_voting_context"]

    # Whale
    assert nv["whale"]["counts_as_vote"] is False
    assert nv["whale"]["role"] == "COMPOSITE_CONTEXT"
    assert nv["whale"]["calibration"] == "UNCALIBRATED_HEURISTIC"

    # Regime
    assert nv["regime"]["counts_as_vote"] is False
    assert nv["regime"]["role"] == "COMPOSITE_CONTEXT"
    assert nv["regime"]["calibration"] == "UNCALIBRATED_HEURISTIC"

    # ML
    assert nv["ml"]["counts_as_vote"] is False
    assert nv["ml"]["role"] == "ML_MODEL_OUTPUT"
    assert nv["ml"]["calibration"] == "UNCALIBRATED_HEURISTIC"

    # Derivatives
    assert nv["derivatives"]["counts_as_vote"] is False

    # Liquidations
    assert nv["liquidations"]["counts_as_vote"] is False
    assert nv["liquidations"]["capability"] == "OBSERVED_FORCE_ORDER_SNAPSHOT"

    # Macro
    assert nv["macro_calendar"]["counts_as_vote"] is False


# ==============================================================================
# 4. EXECUTION CONTEXT & CANONICAL SLIPPAGE (NO SCALE *100)
# ==============================================================================

def test_execution_context_canonical_units():
    """
    Garante que o slippage é exposto em USD real ($2.50) e bps,
    sem scaling legado x100, e que chaves perigosas foram banidas.
    """
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    exec_ctx = payload["execution_context"]
    assert exec_ctx["source_type"] == "POINT_IN_TIME_L2"
    assert exec_ctx["buy"]["execution_slippage_usd"] == 2.50  # USD real, não 250.0!
    assert exec_ctx["sell"]["execution_slippage_usd"] == 1.80
    assert exec_ctx["buy"]["is_fillable"] is True

    # Não deve conter campos legados perigosos
    assert "slip_b" not in exec_ctx
    assert "slip_s" not in exec_ctx


# ==============================================================================
# 5. REGIME HEURISTIC NAMING (NO PROB_TREND)
# ==============================================================================

def test_regime_heuristic_naming_no_false_probabilities():
    """Valida renomeação semântica de regime_probabilities para heuristic_distribution."""
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    reg = payload["non_voting_context"]["regime"]
    dist = reg["heuristic_distribution"]

    assert dist["trending_share"] == 0.70
    assert dist["mean_reverting_share"] == 0.20
    assert dist["breakout_share"] == 0.10

    # Chaves enganosas não devem existir
    assert "prob_trend" not in reg
    assert "prob_rev" not in reg
    assert "prob_break" not in reg


# ==============================================================================
# 6. ML SCORE NAMING (NO PU AS PROBABILITY)
# ==============================================================================

def test_ml_score_naming_no_probability_claim():
    """Valida que pu é exposto como ml_up_score com disclaimer uncalibrated."""
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    ml_sec = payload["non_voting_context"]["ml"]
    assert ml_sec["ml_up_score"] == 0.62
    assert "pu" not in ml_sec
    assert "prob_up" not in ml_sec
    assert ml_sec["calibration"] == "UNCALIBRATED_HEURISTIC"


# ==============================================================================
# 7. FORENSE J1 REAL (MIXED_DIRECTIONS)
# ==============================================================================

def test_j1_forensic_payload_v3_mixed_directions():
    """
    Cenário J1 real auditado:
    - Fluxo executado: BULLISH (compra agressiva)
    - Orderbook snapshot: BULLISH (imbalance bid)
    - Estrutura de mercado: BEARISH (BOS breakdown)
    - Whale/Regime: NON_VOTING
    O reconciliador v3 DEVE emitir status MIXED_DIRECTIONS,
    sem permitir que os múltiplos campos de fluxo mascarem o rompimento vendedor.
    """
    event_j1 = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "preco_fechamento": 66875.0,
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 34458.0,
                "flow_imbalance": 0.8013,
                "aggressive_buy_pct": 90.06,
                "buy_sell_ratio": {"buy_sell_ratio": 9.06},
            }
        },
        "orderbook_data": {
            "imbalance": 0.864,
            "bid_depth_usd": 500000.0,
            "ask_depth_usd": 100000.0,
        },
        "market_structure": {
            "bos": "BEAR_BREAKDOWN_CONFIRMED",
        },
        "institutional_analytics": {
            "flow_analysis": {
                "whale_accumulation": {"score": 45}  # Composite
            }
        },
        "regime_analysis": {
            "current_regime": "TRENDING",  # Composite
        },
    }

    payload = build_semantic_payload_v3(event_j1, symbol="BTCUSDT")
    rec = payload["confluence_reconciler"]

    # Deve ser rigorosamente MIXED_DIRECTIONS
    assert rec["status"] == ReconcilerStatus.MIXED_DIRECTIONS.value
    assert "bullish" in rec["directions_present"]
    assert "bearish" in rec["directions_present"]

    # Whale e Regime NÃO votaram (estão apenas em non_voting_context)
    assert payload["non_voting_context"]["whale"]["counts_as_vote"] is False
    assert payload["non_voting_context"]["regime"]["counts_as_vote"] is False


# ==============================================================================
# 8. FORENSE J2 REAL (MISSING POSITIONING NO FABRICATION)
# ==============================================================================

def test_j2_forensic_payload_v3_missing_data_honesty():
    """
    Cenário J2 real auditado:
    - Derivativos/Posicionamento indisponíveis no domingo.
    - O payload v3 NÃO fabrica zero nem neutralidade; relata MISSING/null.
    """
    event_j2 = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775177400000,
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 0.0,
                "flow_imbalance": 0.0,
            }
        },
        "orderbook_data": {
            "imbalance": 0.0,
        },
        # Sem derivatives, sem macro
    }

    payload = build_semantic_payload_v3(event_j2, symbol="BTCUSDT")
    nv = payload["non_voting_context"]

    # Open interest e funding rate devem ser None/null, sem invenção de dados
    assert nv["derivatives"]["open_interest_contracts"] is None
    assert nv["derivatives"]["funding_rate_decimal"] is None
    assert nv["macro_calendar"]["provider_status"] == "MISSING"


# ==============================================================================
# 9. RFC 8259 SERIALIZATION
# ==============================================================================

def test_semantic_payload_v3_rfc8259():
    """Garante serialização estrita para JSON válido RFC 8259."""
    event = make_full_test_event()
    payload = build_semantic_payload_v3(event, symbol="BTCUSDT")

    dumped = json.dumps(payload)
    loaded = json.loads(dumped)

    assert loaded["semantic_payload_version"] == "3.0.0"
    assert loaded["symbol"] == "BTCUSDT"
    assert isinstance(loaded["directional_evidence"]["items"], list)


# ==============================================================================
# 10. BACKWARD COMPATIBILITY: PAYLOAD V2 REMAINS OPERATIONAL
# ==============================================================================

def test_legacy_payload_v2_regression_preserved():
    """Valida que build_compact_payload v2 continua funcionando sem alterações."""
    event = make_full_test_event()
    payload_v2 = bcp.build_compact_payload(event)

    # v2 continua contendo suas chaves clássicas
    assert "mkt" in payload_v2 and payload_v2["mkt"] == "fut_perp"
    assert "price" in payload_v2
    assert "flow" in payload_v2
    assert "ob" in payload_v2
    assert "regime" in payload_v2
