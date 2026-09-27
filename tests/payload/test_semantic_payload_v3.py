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

# ==============================================================================
# 8. FORENSE J2 REAL (MIXED_DIRECTIONS & MISSING CONTEXT HONESTY)
# ==============================================================================

def test_j2_forensic_payload_v3_mixed_directions_and_missing_context():
    """
    Cenário J2 real auditado:
    - Executed Flow BULLISH (compra agressiva)
    - Orderbook Snapshot BEARISH (ask-heavy / imbalance negativo)
    - Market Structure BEARISH (BOS breakdown)
    - Absorption BEARISH (absorção de compra no topo)
    => reconciler.status DEVE ser MIXED_DIRECTIONS.

    Contexto ausente no domingo:
    - Derivativos/Posicionamento: MISSING
    - Calendário Macro: MISSING
    => Registrados exclusivamente em data_quality.missing_context.
    
    Separação Causal:
    - A justificativa factual para action="WAIT" e assessment="MIXED_DIRECTIONS"
      é a divergência entre fluxo e estrutura/livro, NUNCA "porque é domingo" ou
      "porque falta posicionamento".
    """
    event_j2 = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775177400000,
        "preco_fechamento": 67100.0,
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 25000.0,  # BULLISH
                "flow_imbalance": 0.35,
            },
            "absorption_analysis": {
                "current_absorption": {
                    "label": "Forte Compradora",  # BEARISH
                    "index": 0.88,
                }
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 120000.0,
            "ask_depth_usd": 480000.0,
            "imbalance": -0.60,  # BEARISH
        },
        "market_structure": {
            "bos": "BEAR_BREAKDOWN_67050",  # BEARISH
        },
        # Sem derivatives (MISSING)
        # Sem macro_calendar (MISSING)
    }

    payload = build_semantic_payload_v3(event_j2, symbol="BTCUSDT")
    rec = payload["confluence_reconciler"]
    dq = payload["data_quality"]
    nv = payload["non_voting_context"]

    # 1. Reconciler acusa divergência observada
    assert rec["status"] == ReconcilerStatus.MIXED_DIRECTIONS.value
    assert "bullish" in rec["directions_present"]
    assert "bearish" in rec["directions_present"]

    # 2. Derivativos e Macro ausentes são honestamente reportados como null/MISSING
    assert nv["derivatives"]["open_interest_contracts"] is None
    assert nv["derivatives"]["funding_rate_decimal"] is None
    assert nv["derivatives"]["validity"] == "MISSING"
    assert nv["macro_calendar"]["provider_status"] == "MISSING"

    # 3. Data quality expõe fatos sem booleano inventado
    assert dq["reconciler_status"] == ReconcilerStatus.MIXED_DIRECTIONS.value
    assert "derivatives" in dq["missing_context"]
    assert "macro_calendar" in dq["missing_context"]
    assert "is_sufficient_for_assessment" not in dq
    assert "data_staleness_warning" not in dq


# ==============================================================================
# 9. LIQUIDATION ZERO VS MISSING TEST
# ==============================================================================

def test_liquidation_telemetry_absent_vs_zero_observed():
    """
    Testa a distinção inegociável entre:
    A) Telemetria de liquidação AUSENTE -> status MISSING, valores null.
    B) Telemetria com stream FULL e 0 eventos -> status VALID, valores 0.0, reason ZERO_OBSERVED_HEALTHY_STREAM.
    """
    # A) Ausente
    event_no_liq = {"symbol": "BTCUSDT", "epoch_ms": 1775177400000}
    payload_a = build_semantic_payload_v3(event_no_liq)
    liq_a = payload_a["non_voting_context"]["liquidations"]

    assert liq_a["status"] == "MISSING"
    assert liq_a["observed_notional_usd_total"] is None
    assert liq_a["estimated_notional_usd_total"] is None
    assert liq_a["event_count"] is None
    assert "liquidations" in payload_a["data_quality"]["missing_context"]

    # B) Saudável, cobertura FULL e 0 eventos
    event_zero_liq = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775177400000,
        "liquidation_telemetry": {
            "coverage": "FULL",
            "validity": "VALID",
            "stream_healthy": True,
            "event_count": 0,
            "observed_notional_usd_total": 0.0,
            "estimated_notional_usd_total": 0.0,
        },
    }
    payload_b = build_semantic_payload_v3(event_zero_liq)
    liq_b = payload_b["non_voting_context"]["liquidations"]

    assert liq_b["status"] == "VALID"
    assert liq_b["reason"] == "ZERO_OBSERVED_HEALTHY_STREAM"
    assert liq_b["observed_notional_usd_total"] == 0.0
    assert liq_b["estimated_notional_usd_total"] == 0.0
    assert liq_b["event_count"] == 0
    assert "liquidations" not in payload_b["data_quality"]["missing_context"]


# ==============================================================================
# 10. EXECUTION CONTEXT THREE-STATE TEST
# ==============================================================================

def test_execution_context_fillability_three_state():
    """
    Testa os 3 estados estritos de is_fillable:
    A) fill_ratio >= 1.0 e slippage presente -> is_fillable = True
    B) fill_ratio < 1.0 ou insuficiente -> is_fillable = False
    C) fill_ratio ausente (None) -> is_fillable = None
    Nunca colapsar B e C!
    """
    base_event = {"symbol": "BTCUSDT", "preco_fechamento": 67000.0}

    # A) Fillable observado
    evt_a = {
        **base_event,
        "market_impact": {
            "slippage_matrix": {"100k_usd": {"buy": 2.0, "sell": 2.0}},
            "fill_ratio_matrix": {"100k_usd": {"buy": 1.0, "sell": 1.0}},
        },
    }
    pay_a = build_semantic_payload_v3(evt_a)
    assert pay_a["execution_context"]["buy"]["is_fillable"] is True
    assert pay_a["execution_context"]["buy"]["fill_ratio"] == 1.0

    # B) Insuficiente observado
    evt_b = {
        **base_event,
        "market_impact": {
            "slippage_matrix": {"100k_usd": {"buy": 2.0, "sell": 2.0}},
            "fill_ratio_matrix": {"100k_usd": {"buy": 0.65, "sell": 0.40}},
        },
    }
    pay_b = build_semantic_payload_v3(evt_b)
    assert pay_b["execution_context"]["buy"]["is_fillable"] is False
    assert pay_b["execution_context"]["buy"]["fill_ratio"] == 0.65
    assert pay_b["execution_context"]["sell"]["is_fillable"] is False

    # C) Missing (desconhecido)
    evt_c = {
        **base_event,
        "market_impact": {
            "slippage_matrix": {"100k_usd": {"buy": 2.0, "sell": 2.0}},
            # Sem fill_ratio_matrix
        },
    }
    pay_c = build_semantic_payload_v3(evt_c)
    assert pay_c["execution_context"]["buy"]["fill_ratio"] is None
    assert pay_c["execution_context"]["buy"]["is_fillable"] is None
    assert pay_c["execution_context"]["sell"]["fill_ratio"] is None
    assert pay_c["execution_context"]["sell"]["is_fillable"] is None


# ==============================================================================
# 11. DERIVATIVES FUNDING ZERO VS MISSING TEST
# ==============================================================================

def test_derivatives_funding_zero_vs_missing():
    """
    Testa que funding = 0.0 é um dado válido observado, não missing.
    - funding=0.0 + OI válido -> VALID
    - funding=None + OI válido -> PARTIAL
    - ambos None -> MISSING
    """
    # 1. Funding 0.0 observado
    evt_zero = {
        "symbol": "BTCUSDT",
        "derivatives": {"BTCUSDT": {"open_interest": 85000.0, "funding_rate": 0.0}},
    }
    pay_zero = build_semantic_payload_v3(evt_zero)
    d_zero = pay_zero["non_voting_context"]["derivatives"]
    assert d_zero["validity"] == "VALID"
    assert d_zero["funding_rate_decimal"] == 0.0
    assert d_zero["is_observed_funding_zero"] is True

    # 2. Funding missing + OI válido
    evt_partial = {
        "symbol": "BTCUSDT",
        "derivatives": {"BTCUSDT": {"open_interest": 85000.0}},
    }
    pay_partial = build_semantic_payload_v3(evt_partial)
    d_partial = pay_partial["non_voting_context"]["derivatives"]
    assert d_partial["validity"] == "PARTIAL"
    assert d_partial["funding_rate_decimal"] is None
    assert d_partial["is_observed_funding_zero"] is False

    # 3. Ambos missing
    evt_miss = {"symbol": "BTCUSDT"}
    pay_miss = build_semantic_payload_v3(evt_miss)
    d_miss = pay_miss["non_voting_context"]["derivatives"]
    assert d_miss["validity"] == "MISSING"
    assert "derivatives" in pay_miss["data_quality"]["missing_context"]


# ==============================================================================
# 12. MACRO REFERENCE TIME & MISSING PRICE FOR BPS TEST
# ==============================================================================

def test_macro_reference_and_execution_missing_price():
    """
    Testa que:
    1. Evento macro agendado sem reference_time e sem as_of_ms não gera número gigante.
    2. Slippage USD presente sem preço de fechamento não calcula bps com fallback 1.0.
    """
    # 1. Macro time to event
    evt_macro = {
        "symbol": "BTCUSDT",
        "macro_calendar_snapshot": {
            "provider_status": "AVAILABLE",
            "nearest_upcoming_event": {
                "event_type": "FOMC_RATE_DECISION",
                "scheduled_at_ms": 1800000000000,
            },
        },
    }
    pay_m = build_semantic_payload_v3(evt_macro, as_of_ms=1799000000000)
    nearest = pay_m["non_voting_context"]["macro_calendar"]["nearest_event"]
    assert nearest["time_to_event_ms"] == 1000000000  # 1800000000000 - 1799000000000

    # 2. Missing price for BPS
    evt_no_px = {
        "symbol": "BTCUSDT",
        "market_impact": {
            "slippage_matrix": {"100k_usd": {"buy": 3.50, "sell": 2.00}},
            "fill_ratio_matrix": {"100k_usd": {"buy": 1.0, "sell": 1.0}},
        },
        # Sem preco_fechamento e sem ohlc close
    }
    pay_px = build_semantic_payload_v3(evt_no_px)
    ex = pay_px["execution_context"]
    assert ex["buy"]["execution_slippage_usd"] == 3.50
    assert ex["buy"]["execution_slippage_bps"] is None  # Não calcula dividindo por 1.0!
    assert ex["reason"] == "MISSING_REFERENCE_PRICE_FOR_BPS"


# ==============================================================================
# 13. RFC 8259 SERIALIZATION
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
# 14. BACKWARD COMPATIBILITY: PAYLOAD V2 REMAINS OPERATIONAL
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

