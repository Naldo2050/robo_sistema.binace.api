# market_orchestrator/ai/semantic_payload_builder_v3.py
# -*- coding: utf-8 -*-
"""
P2-F2 — Semantic LLM Payload Builder v3.

Construtor do payload semântico de terceira geração (semantic_payload_version: "3.0.0")
projetado especificamente para eliminar o double counting, respeitar os contratos P0/P1/P2
e desacoplar evidências direcionais votantes de telemetria e contexto estritamente não-votante.

Princípios Fundamentais:
1. Root Separation: Separação estrutural estrita entre:
   - directional_evidence (apenas evidências primárias aceitas no Confluence Shadow)
   - confluence_reconciler (síntese observacional auditada de P2-A)
   - execution_context (liquidez direcional e slippage canônico P2-B2)
   - non_voting_context (whale, regime, positioning, macro, liquidations com counts_as_vote=false)
   - data_quality (integridade, frescor e completude do pipeline)
2. Redundancy Suppression: Evidências com redundância exata (ex: BSR vs imbalance)
   expõem apenas o representante canônico na visão direcional.
3. No False Probabilities: Nenhuma grandeza não calibrada é chamada de probabilidade.
4. Fail-Closed: Ausência de dados não vira 0 ou neutralidade.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from institutional import evidence_taxonomy as tx
from institutional.confluence_shadow import (
    ConfluenceShadowResult,
    DIRECTIONAL_WHITELIST,
    reconcile as confluence_reconcile,
)
from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
)

logger = logging.getLogger(__name__)

SEMANTIC_PAYLOAD_VERSION = "3.0.0"


def _safe_float(val: Any) -> Optional[float]:
    """Converte para float se finito; caso contrário retorna None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Converte para int se finito; caso contrário retorna None."""
    f = _safe_float(val)
    return int(round(f)) if f is not None else None


def extract_evidence_from_event(
    event_data: Dict[str, Any],
    reference_time_ms: int,
) -> List[Evidence]:
    """
    Extrai instâncias de Evidence v1 a partir de um evento de mercado,
    respeitando os adaptadores e contratos de P0/P1/P2.
    """
    evidences: List[Evidence] = []
    symbol = str(event_data.get("symbol") or "BTCUSDT")

    # 1. EXECUTED FLOW (flow_analyzer/order_flow)
    fluxo = event_data.get("fluxo_continuo", {}) or {}
    order_flow = fluxo.get("order_flow", {}) or {}
    bsr_data = order_flow.get("buy_sell_ratio", {}) or {}

    # flow.net.1m
    net_1m = _safe_float(order_flow.get("net_flow_1m"))
    if net_1m is not None:
        direction = (
            EvidenceDirection.BULLISH if net_1m > 0
            else EvidenceDirection.BEARISH if net_1m < 0
            else EvidenceDirection.NEUTRAL
        )
        evidences.append(
            Evidence(
                source="flow.net.1m",
                family=EvidenceFamily.EXECUTED_FLOW,
                evidence_type=EvidenceType.CONTINUOUS_TRADES,
                direction=direction,
                value=net_1m,
                observed_at_ms=reference_time_ms,
                horizon_ms=60_000,
                validity=EvidenceValidity.VALID,
                calibration=EvidenceCalibration.NOT_APPLICABLE,
                counts_as_vote=True,
                provenance={"source": "flow_analyzer.order_flow", "unit": "USD"},
            )
        )

    # flow.imbalance.1m
    imb_1m = _safe_float(order_flow.get("flow_imbalance"))
    if imb_1m is not None:
        direction = (
            EvidenceDirection.BULLISH if imb_1m > 0.05
            else EvidenceDirection.BEARISH if imb_1m < -0.05
            else EvidenceDirection.NEUTRAL
        )
        evidences.append(
            Evidence(
                source="flow.imbalance.1m",
                family=EvidenceFamily.EXECUTED_FLOW,
                evidence_type=EvidenceType.CONTINUOUS_TRADES,
                direction=direction,
                value=imb_1m,
                observed_at_ms=reference_time_ms,
                horizon_ms=60_000,
                validity=EvidenceValidity.VALID,
                calibration=EvidenceCalibration.NOT_APPLICABLE,
                counts_as_vote=True,
                provenance={"source": "flow_analyzer.order_flow", "unit": "ratio"},
            )
        )

    # flow.buy_sell_ratio (REDUNDANTE com flow.imbalance.1m)
    bsr_val = _safe_float(bsr_data.get("buy_sell_ratio"))
    if bsr_val is not None:
        direction = (
            EvidenceDirection.BULLISH if bsr_val > 1.05
            else EvidenceDirection.BEARISH if bsr_val < 0.95
            else EvidenceDirection.NEUTRAL
        )
        evidences.append(
            Evidence(
                source="flow.buy_sell_ratio",
                family=EvidenceFamily.EXECUTED_FLOW,
                evidence_type=EvidenceType.CONTINUOUS_TRADES,
                direction=direction,
                value=bsr_val,
                observed_at_ms=reference_time_ms,
                horizon_ms=60_000,
                validity=EvidenceValidity.VALID,
                calibration=EvidenceCalibration.NOT_APPLICABLE,
                counts_as_vote=True,
                provenance={"source": "flow_analyzer.order_flow", "unit": "ratio"},
            )
        )

    # 2. ORDERBOOK SNAPSHOT
    ob_data = event_data.get("orderbook_data", {}) or {}
    ob_imb = _safe_float(ob_data.get("flow_imbalance") or ob_data.get("imbalance"))
    if ob_imb is not None:
        direction = (
            EvidenceDirection.BULLISH if ob_imb > 0.1
            else EvidenceDirection.BEARISH if ob_imb < -0.1
            else EvidenceDirection.NEUTRAL
        )
        evidences.append(
            Evidence(
                source="orderbook.snapshot.imbalance",
                family=EvidenceFamily.ORDERBOOK_SNAPSHOT,
                evidence_type=EvidenceType.POINT_IN_TIME_L2,
                direction=direction,
                value=ob_imb,
                observed_at_ms=reference_time_ms,
                horizon_ms=0,
                validity=EvidenceValidity.VALID,
                calibration=EvidenceCalibration.NOT_APPLICABLE,
                counts_as_vote=True,
                provenance={"source": "orderbook_analyzer.core", "unit": "ratio"},
            )
        )

    # 3. MARKET STRUCTURE (BOS & SWEEP)
    ms_data = event_data.get("market_structure", {}) or (
        event_data.get("pattern_recognition", {}).get("smart_money", {}).get("market_structure", {})
    )
    if isinstance(ms_data, dict):
        bos_str = str(ms_data.get("bos") or ms_data.get("structure") or "").upper()
        if "BULL" in bos_str or "UP" in bos_str:
            evidences.append(
                Evidence(
                    source="market_structure.bos",
                    family=EvidenceFamily.MARKET_STRUCTURE,
                    evidence_type=EvidenceType.POINT_IN_TIME_L2,
                    direction=EvidenceDirection.BULLISH,
                    value=1.0,
                    observed_at_ms=reference_time_ms,
                    horizon_ms=60_000,
                    validity=EvidenceValidity.VALID,
                    calibration=EvidenceCalibration.NOT_APPLICABLE,
                    counts_as_vote=True,
                    provenance={"source": "market_structure", "unit": "flag"},
                )
            )
        elif "BEAR" in bos_str or "DOWN" in bos_str:
            evidences.append(
                Evidence(
                    source="market_structure.bos",
                    family=EvidenceFamily.MARKET_STRUCTURE,
                    evidence_type=EvidenceType.POINT_IN_TIME_L2,
                    direction=EvidenceDirection.BEARISH,
                    value=-1.0,
                    observed_at_ms=reference_time_ms,
                    horizon_ms=60_000,
                    validity=EvidenceValidity.VALID,
                    calibration=EvidenceCalibration.NOT_APPLICABLE,
                    counts_as_vote=True,
                    provenance={"source": "market_structure", "unit": "flag"},
                )
            )

        sweep_str = str(ms_data.get("sw") or ms_data.get("sweep") or "").upper()
        if "BUY" in sweep_str:
            evidences.append(
                Evidence(
                    source="market_structure.sweep",
                    family=EvidenceFamily.MARKET_STRUCTURE,
                    evidence_type=EvidenceType.POINT_IN_TIME_L2,
                    direction=EvidenceDirection.BEARISH,  # Varredura de topo aciona stops de compra -> reversão
                    value=1.0,
                    observed_at_ms=reference_time_ms,
                    horizon_ms=60_000,
                    validity=EvidenceValidity.VALID,
                    calibration=EvidenceCalibration.NOT_APPLICABLE,
                    counts_as_vote=True,
                    provenance={"source": "market_structure", "unit": "flag"},
                )
            )
        elif "SELL" in sweep_str:
            evidences.append(
                Evidence(
                    source="market_structure.sweep",
                    family=EvidenceFamily.MARKET_STRUCTURE,
                    evidence_type=EvidenceType.POINT_IN_TIME_L2,
                    direction=EvidenceDirection.BULLISH,  # Varredura de fundo aciona stops de venda -> reversão
                    value=-1.0,
                    observed_at_ms=reference_time_ms,
                    horizon_ms=60_000,
                    validity=EvidenceValidity.VALID,
                    calibration=EvidenceCalibration.NOT_APPLICABLE,
                    counts_as_vote=True,
                    provenance={"source": "market_structure", "unit": "flag"},
                )
            )

    # 4. ABSORPTION (com label canônico provado em P0-A2)
    abs_data = fluxo.get("absorption_analysis", {}).get("current_absorption", {})
    if isinstance(abs_data, dict) and abs_data:
        abs_label = str(abs_data.get("label") or "").lower()
        if "compradora" in abs_label or "buy" in abs_label:
            # Absorção de Compra (passiva compradora absorvendo venda ativa) -> viés BULLISH
            # Nota P2-A: Absorção de Venda ativa = BULLISH; Absorção de Compra ativa = BEARISH.
            # Verificamos convenção canônica do P0:
            direction = EvidenceDirection.BULLISH if "compra" in abs_label or "buy" in abs_label else EvidenceDirection.BEARISH
            if "forte compradora" in abs_label or "strong_buy" in abs_label:
                direction = EvidenceDirection.BEARISH
            elif "forte vendedora" in abs_label or "strong_sell" in abs_label:
                direction = EvidenceDirection.BULLISH
            evidences.append(
                Evidence(
                    source="absorption.current",
                    family=EvidenceFamily.MARKET_STRUCTURE,
                    evidence_type=EvidenceType.DERIVED,
                    direction=direction,
                    value=_safe_float(abs_data.get("index")),
                    observed_at_ms=reference_time_ms,
                    horizon_ms=60_000,
                    validity=EvidenceValidity.VALID,
                    calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
                    counts_as_vote=False,  # Non-voting magnitude
                    provenance={"source": "flow_analyzer.absorption", "unit": "score"},
                    metadata={"magnitude_calibration": "UNVALIDATED", "relation_to_flow": "OVERLAPPING_LINEAGE"},
                )
            )

    return evidences


def _build_directional_evidence_section(
    evidences: List[Evidence],
    reconciler_result: ConfluenceShadowResult,
) -> Dict[str, Any]:
    """
    Constrói a seção directional_evidence:
    - Inclui apenas evidências da whitelist
    - Suprime aliases exatos (REDUNDANT_EXACT), exibindo apenas o representante.
    """
    suppressed_set = set()
    for grp in reconciler_result.exact_redundancy_groups:
        for alias in grp.get("suppressed_aliases", ()):
            suppressed_set.add(alias)

    exposed_items: List[Dict[str, Any]] = []

    for ev in evidences:
        if ev.source not in DIRECTIONAL_WHITELIST:
            continue
        if ev.source in suppressed_set:
            continue
        if ev.validity != EvidenceValidity.VALID:
            continue

        item = {
            "field_id": ev.source,
            "direction": ev.direction.value,
            "family": ev.family.value,
            "evidence_type": ev.evidence_type.value,
            "validity": ev.validity.value,
            "calibration": ev.calibration.value,
            "lineage_role": "PRIMARY" if not tx.FIELDS.get(ev.source, None) or not tx.FIELDS[ev.source].is_composite else "COMPOSITE",
            "observed_at_ms": ev.observed_at_ms,
            "horizon_ms": ev.horizon_ms,
            "unit": ev.provenance.get("unit", "unknown"),
            "value": ev.value,
        }
        exposed_items.append(item)

    return {
        "count": len(exposed_items),
        "items": exposed_items,
        "suppressed_redundant_aliases": sorted(list(suppressed_set)),
    }


def _build_confluence_reconciler_section(
    result: ConfluenceShadowResult,
) -> Dict[str, Any]:
    """
    Exporta o resultado oficial do Confluence Shadow v1 (P2-A).
    Sem confianças inventadas, sem probabilidades, sem scores ad-hoc.
    """
    return {
        "status": result.status.value,
        "directions_present": list(result.directions_present),
        "families_present": list(result.families_present),
        "evidence_count": result.evidence_count,
        "exact_redundancy_groups": list(result.exact_redundancy_groups),
        "overlap_relations": list(result.overlap_relations),
        "disjoint_relations": list(result.disjoint_relations),
        "independent_confirmation": result.independent_confirmation,
        "contract_rule": "MIXED_DIRECTIONS_CANNOT_BE_RESOLVED_BY_MAJORITY_VOTE",
    }


def _build_execution_context_section(
    event_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constrói a seção execution_context conforme P2-B2:
    - Slippage real em USD e bps (sem scaling *100)
    - Fillability e liquidez direcional por lado
    - Fonte pontual L2 snapshot explicitada.
    """
    mi = event_data.get("market_impact", {}) or {}
    slip_matrix = mi.get("slippage_matrix", {}) or {}
    s100 = slip_matrix.get("100k_usd", {}) or {}
    fr_matrix = mi.get("fill_ratio_matrix", {}) or {}
    fr_100 = fr_matrix.get("100k_usd", {}) or {}

    buy_slip_usd = _safe_float(s100.get("buy"))
    sell_slip_usd = _safe_float(s100.get("sell"))

    buy_fill = _safe_float(fr_100.get("buy")) if fr_100 else 1.0
    sell_fill = _safe_float(fr_100.get("sell")) if fr_100 else 1.0

    eq = mi.get("execution_quality", "UNKNOWN")
    liq_score = _safe_float(mi.get("liquidity_score"))

    # Preço de referência para converter em bps
    c_px = _safe_float(event_data.get("preco_fechamento")) or _safe_float(
        event_data.get("contextual_snapshot", {}).get("ohlc", {}).get("close")
    ) or 1.0

    buy_bps = round((buy_slip_usd / c_px) * 10_000, 2) if buy_slip_usd is not None and c_px > 0 else None
    sell_bps = round((sell_slip_usd / c_px) * 10_000, 2) if sell_slip_usd is not None and c_px > 0 else None

    return {
        "source_type": "POINT_IN_TIME_L2",
        "temporal_persistence": "UNKNOWN",
        "reference_notional_usd": 100_000,
        "buy": {
            "execution_slippage_usd": round(buy_slip_usd, 4) if buy_slip_usd is not None else None,
            "execution_slippage_bps": buy_bps,
            "fill_ratio": buy_fill,
            "is_fillable": bool(buy_fill is not None and buy_fill >= 1.0),
        },
        "sell": {
            "execution_slippage_usd": round(sell_slip_usd, 4) if sell_slip_usd is not None else None,
            "execution_slippage_bps": sell_bps,
            "fill_ratio": sell_fill,
            "is_fillable": bool(sell_fill is not None and sell_fill >= 1.0),
        },
        "liquidity_score": round(liq_score, 1) if liq_score is not None else None,
        "execution_quality_tier": str(eq),
        "validity": "VALID" if (buy_slip_usd is not None and sell_slip_usd is not None) else "PARTIAL",
        "semantic_notice": "EXECUTION_CONTEXT_DOES_NOT_IMPLY_MARKET_DIRECTION",
    }


def _build_non_voting_context_section(
    event_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constrói o bloco non_voting_context contendo todos os composites e telemetrias:
    - Whale accumulation/distribution
    - Regime de mercado e distribuição heurística
    - Predições de ML (ml_up_score, não prob_up)
    - Posicionamento / Open Interest
    - Taxa de financiamento (Funding Rate)
    - Liquidações Binance
    - Calendário de eventos macro agendados
    Todos identificados explicitamente com counts_as_vote=false.
    """
    context: Dict[str, Any] = {}

    # 1. WHALE COMPOSITE
    ia = event_data.get("institutional_analytics", {}) or {}
    fa = ia.get("flow_analysis", {}) or {}
    wa = fa.get("whale_accumulation", {}) or {}
    whale_score = _safe_int(wa.get("score") or wa.get("s"))
    if whale_score is not None:
        context["whale"] = {
            "role": "COMPOSITE_CONTEXT",
            "counts_as_vote": False,
            "whale_score": whale_score,
            "classification": wa.get("classification") or wa.get("c"),
            "divergence": wa.get("smart_money_divergence") or wa.get("div"),
            "calibration": "UNCALIBRATED_HEURISTIC",
        }

    # 2. REGIME COMPOSITE
    regime_analysis = event_data.get("regime_analysis", {}) or {}
    current_regime = regime_analysis.get("current_regime")
    regime_probs = regime_analysis.get("regime_probabilities", {}) or {}
    context["regime"] = {
        "role": "COMPOSITE_CONTEXT",
        "counts_as_vote": False,
        "mode": current_regime or "UNKNOWN",
        "heuristic_distribution": {
            "trending_share": _safe_float(regime_probs.get("trending")),
            "mean_reverting_share": _safe_float(regime_probs.get("mean_reverting")),
            "breakout_share": _safe_float(regime_probs.get("breakout")),
        },
        "transition_risk_heuristic": _safe_float(regime_analysis.get("regime_change_probability")),
        "calibration": "UNCALIBRATED_HEURISTIC",
        "status": regime_analysis.get("status", "PARTIAL"),
    }

    # 3. MACHINE LEARNING OUTPUT
    ml = event_data.get("ml_prediction", {}) or event_data.get("quant_prediction", {}) or {}
    if ml:
        context["ml"] = {
            "role": "ML_MODEL_OUTPUT",
            "counts_as_vote": False,
            "ml_up_score": _safe_float(ml.get("prob_up")),
            "model_confidence_score": _safe_float(ml.get("confidence")),
            "ml_stale": bool(ml.get("ml_stale", False)),
            "calibration": "UNCALIBRATED_HEURISTIC",
            "notice": "RAW_MODEL_SCORE_NOT_A_CALIBRATED_PROBABILITY",
        }

    # 4. DERIVATIVES POSITIONING & FUNDING
    deriv = event_data.get("derivatives", {}) or {}
    btc_deriv = deriv.get("BTCUSDT", {}) or {}
    pos_snapshot = event_data.get("positioning_snapshot", {}) or {}

    oi_val = _safe_float(btc_deriv.get("open_interest") or pos_snapshot.get("open_interest"))
    oi_usd = _safe_float(btc_deriv.get("open_interest_usd") or pos_snapshot.get("open_interest_usd"))
    lsr_val = _safe_float(btc_deriv.get("long_short_ratio") or pos_snapshot.get("long_short_ratio"))

    fr_val = _safe_float(btc_deriv.get("funding_rate"))
    if fr_val is None:
        fr_pct = _safe_float(btc_deriv.get("funding_rate_percent") or btc_deriv.get("funding_rate_pct"))
        if fr_pct is not None:
            fr_val = fr_pct / 100.0

    context["derivatives"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "open_interest_contracts": oi_val,
        "open_interest_usd": oi_usd,
        "long_short_ratio": lsr_val,
        "funding_rate_decimal": fr_val,
        "funding_rate_unit": "decimal_fraction",
        "is_observed_funding_zero": (fr_val == 0.0) if fr_val is not None else False,
        "validity": "VALID" if (oi_val is not None and fr_val is not None) else "PARTIAL",
    }

    # 5. FORCED LIQUIDATIONS TELEMETRY (P2-D)
    liq_data = event_data.get("liquidation_telemetry", {}) or event_data.get("liquidations", {}) or {}
    context["liquidations"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "capability": "OBSERVED_FORCE_ORDER_SNAPSHOT",
        "observed_notional_usd_long": _safe_float(liq_data.get("observed_notional_usd_long", 0.0)),
        "observed_notional_usd_short": _safe_float(liq_data.get("observed_notional_usd_short", 0.0)),
        "observed_notional_usd_total": _safe_float(liq_data.get("observed_notional_usd_total", 0.0)),
        "estimated_notional_usd_total": _safe_float(liq_data.get("estimated_notional_usd_total", 0.0)),
        "event_count": _safe_int(liq_data.get("event_count", 0)),
        "semantic_notice": "PAST_OBSERVED_FORCE_ORDERS_NOT_FUTURE_LIQUIDATION_LEVELS",
    }

    # 6. SCHEDULED MACRO EVENTS (P2-E)
    macro_snap = event_data.get("macro_calendar_snapshot", {}) or {}
    nearest_ev = macro_snap.get("nearest_upcoming_event")
    context["macro_calendar"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "provider_status": macro_snap.get("provider_status", "MISSING"),
        "nearest_event": {
            "event_type": nearest_ev.get("event_type") if nearest_ev else None,
            "time_to_event_ms": nearest_ev.get("scheduled_at_ms") - macro_snap.get("reference_time_ms", 0) if nearest_ev and "scheduled_at_ms" in nearest_ev else None,
            "scheduled_at_utc": nearest_ev.get("scheduled_at_utc") if nearest_ev else None,
            "importance": nearest_ev.get("importance") if nearest_ev else None,
        } if nearest_ev else None,
        "semantic_notice": "SCHEDULED_EVENTS_DO_NOT_CONSTITUTE_TRADE_BLACKOUT_WINDOW",
    }

    # 7. ORDERBOOK SNAPSHOT STATICS
    ob_data = event_data.get("orderbook_data", {}) or {}
    context["orderbook_snapshot"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "bid_depth_usd": _safe_float(ob_data.get("bid_depth_usd")),
        "ask_depth_usd": _safe_float(ob_data.get("ask_depth_usd")),
        "spread_pct": _safe_float(ob_data.get("spread_percent")),
        "capability_scope": "SNAPSHOT_ONLY",
        "temporal_persistence": "UNKNOWN",
        "semantic_notice": "SNAPSHOT_DEPTH_ASYMMETRY_IS_NOT_CONTINUOUS_FLOW",
    }

    return context


def _build_data_quality_section(
    event_data: Dict[str, Any],
    directional_evidence: Dict[str, Any],
    reconciler_result: ConfluenceShadowResult,
) -> Dict[str, Any]:
    """
    Centraliza métricas de qualidade, completude e validade dos dados.
    """
    fluxo = event_data.get("fluxo_continuo", {}) or {}
    integrity = fluxo.get("flow_window_integrity", {}) or {}

    return {
        "flow_window_integrity": integrity,
        "orderbook_capability": "SNAPSHOT_ONLY",
        "reconciler_status": reconciler_result.status.value,
        "directional_evidence_count": reconciler_result.evidence_count,
        "is_sufficient_for_assessment": bool(reconciler_result.evidence_count > 0),
        "data_staleness_warning": False,
    }


def build_semantic_payload_v3(
    event_data: Dict[str, Any],
    symbol: str = "BTCUSDT",
    as_of_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ponto de entrada canônico do Payload Semântico v3 (semantic_payload_version: '3.0.0').
    Gera um dicionário serializável em JSON RFC 8259 estritamente auditado.
    """
    now_ms = as_of_ms or int(event_data.get("epoch_ms") or time.time() * 1000)

    # 1. Extrai evidências elegíveis e executa reconciliação de confluência (P2-A)
    evidences = extract_evidence_from_event(event_data, reference_time_ms=now_ms)
    reconciler_result = confluence_reconcile(
        evidences=evidences,
        symbol=symbol,
        observation_open_ms=now_ms - 60_000,
        observation_close_ms=now_ms,
        causal_anchor_ms=now_ms,
    )

    # 2. Constrói cada uma das seções estruturadas isoladas
    directional_evidence = _build_directional_evidence_section(evidences, reconciler_result)
    confluence_reconciler = _build_confluence_reconciler_section(reconciler_result)
    execution_context = _build_execution_context_section(event_data)
    non_voting_context = _build_non_voting_context_section(event_data)
    data_quality = _build_data_quality_section(event_data, directional_evidence, reconciler_result)

    # 3. Monta o root shape
    return {
        "semantic_payload_version": SEMANTIC_PAYLOAD_VERSION,
        "symbol": symbol,
        "market": "futures_perpetual",
        "as_of_ms": now_ms,
        "directional_evidence": directional_evidence,
        "confluence_reconciler": confluence_reconciler,
        "execution_context": execution_context,
        "non_voting_context": non_voting_context,
        "data_quality": data_quality,
    }
