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
    - Fillability e liquidez direcional por lado (tri-state: True, False, None)
    - Fonte pontual L2 snapshot explicitada.
    """
    mi = event_data.get("market_impact", {}) or {}
    dir_liq = event_data.get("directional_liquidity") or mi.get("directional_liquidity") or {}

    buy_can = dir_liq.get("buy", {}).get("100k_usd") or dir_liq.get("buy", {}).get("100k") or {}
    sell_can = dir_liq.get("sell", {}).get("100k_usd") or dir_liq.get("sell", {}).get("100k") or {}

    slip_matrix = mi.get("slippage_matrix", {}) or {}
    s100 = slip_matrix.get("100k_usd", {}) or slip_matrix.get("100k", {}) or {}
    fr_matrix = mi.get("fill_ratio_matrix", {}) or {}
    fr_100 = fr_matrix.get("100k_usd", {}) or fr_matrix.get("100k", {}) or {}

    # 1. Slippage em USD
    buy_slip_usd = _safe_float(buy_can.get("execution_slippage_usd"))
    if buy_slip_usd is None:
        buy_slip_usd = _safe_float(s100.get("buy"))

    sell_slip_usd = _safe_float(sell_can.get("execution_slippage_usd"))
    if sell_slip_usd is None:
        sell_slip_usd = _safe_float(s100.get("sell"))

    # 2. Fill Ratio (sem fallback 1.0)
    buy_fill = _safe_float(buy_can.get("fill_ratio")) if buy_can.get("validity") != "NON_VOTING_MISSING" else None
    if buy_fill is None and fr_100 and "buy" in fr_100 and fr_100.get("buy") is not None:
        buy_fill = _safe_float(fr_100.get("buy"))

    sell_fill = _safe_float(sell_can.get("fill_ratio")) if sell_can.get("validity") != "NON_VOTING_MISSING" else None
    if sell_fill is None and fr_100 and "sell" in fr_100 and fr_100.get("sell") is not None:
        sell_fill = _safe_float(fr_100.get("sell"))

    # 3. is_fillable (tri-state estrito: True = observado fillable, False = observado insuficiente, None = missing)
    buy_insufficient = (
        buy_can.get("validity") == "INSUFFICIENT_LIQUIDITY"
        or bool(s100.get("insufficient_liquidity", False))
        or bool(fr_100.get("insufficient_liquidity", False))
    )
    if buy_fill is None:
        buy_is_fillable = None
    elif buy_insufficient or buy_fill < 1.0 - 1e-4 or buy_slip_usd is None:
        buy_is_fillable = False
    else:
        buy_is_fillable = True

    sell_insufficient = (
        sell_can.get("validity") == "INSUFFICIENT_LIQUIDITY"
        or bool(s100.get("insufficient_liquidity", False))
        or bool(fr_100.get("insufficient_liquidity", False))
    )
    if sell_fill is None:
        sell_is_fillable = None
    elif sell_insufficient or sell_fill < 1.0 - 1e-4 or sell_slip_usd is None:
        sell_is_fillable = False
    else:
        sell_is_fillable = True

    # 4. Slippage em BPS (preferir canônico de P2-B2; se ausente, derivar com preço de referência real SEM fallback 1.0)
    buy_bps = _safe_float(buy_can.get("execution_slippage_bps"))
    sell_bps = _safe_float(sell_can.get("execution_slippage_bps"))

    c_px = _safe_float(event_data.get("preco_fechamento"))
    if c_px is None:
        c_px = _safe_float(event_data.get("contextual_snapshot", {}).get("ohlc", {}).get("close"))

    execution_reason: Optional[str] = None
    if c_px is not None and c_px > 0:
        if buy_bps is None and buy_slip_usd is not None:
            buy_bps = round((buy_slip_usd / c_px) * 10_000, 2)
        if sell_bps is None and sell_slip_usd is not None:
            sell_bps = round((sell_slip_usd / c_px) * 10_000, 2)
    else:
        if (buy_slip_usd is not None and buy_bps is None) or (sell_slip_usd is not None and sell_bps is None):
            execution_reason = "MISSING_REFERENCE_PRICE_FOR_BPS"

    eq = mi.get("execution_quality") or "UNKNOWN"
    liq_score = _safe_float(mi.get("liquidity_score"))

    # Validade factual do contexto de execução
    if buy_slip_usd is None and sell_slip_usd is None and buy_fill is None and sell_fill is None:
        validity = "MISSING"
    elif (
        buy_slip_usd is not None and sell_slip_usd is not None
        and buy_fill is not None and sell_fill is not None
    ):
        validity = "VALID"
    else:
        validity = "PARTIAL"

    res = {
        "source_type": "POINT_IN_TIME_L2",
        "temporal_persistence": "UNKNOWN",
        "reference_notional_usd": 100_000,
        "buy": {
            "execution_slippage_usd": round(buy_slip_usd, 4) if buy_slip_usd is not None else None,
            "execution_slippage_bps": buy_bps,
            "fill_ratio": buy_fill,
            "is_fillable": buy_is_fillable,
        },
        "sell": {
            "execution_slippage_usd": round(sell_slip_usd, 4) if sell_slip_usd is not None else None,
            "execution_slippage_bps": sell_bps,
            "fill_ratio": sell_fill,
            "is_fillable": sell_is_fillable,
        },
        "liquidity_score": round(liq_score, 1) if liq_score is not None else None,
        "execution_quality_tier": str(eq),
        "validity": validity,
        "semantic_notice": "EXECUTION_CONTEXT_DOES_NOT_IMPLY_MARKET_DIRECTION",
    }
    if execution_reason:
        res["reason"] = execution_reason
    return res


def _build_liquidations_context(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói a telemetria de liquidação forçada P2-D/P2-D1 com semântica estrita:
    - Se ausente ou vazia: status MISSING, valores null.
    - Se FULL coverage + 0 eventos com stream saudável: status VALID, valores 0.0, reason ZERO_OBSERVED_HEALTHY_STREAM.
    - Se PARTIAL coverage: status PARTIAL, preserva valores observados reais.
    """
    liq_data = event_data.get("liquidation_telemetry")
    if liq_data is None:
        liq_data = event_data.get("liquidations")

    if not isinstance(liq_data, dict) or not liq_data:
        return {
            "role": "CONTEXT_ONLY",
            "counts_as_vote": False,
            "status": "MISSING",
            "capability": "OBSERVED_FORCE_ORDER_SNAPSHOT",
            "observed_notional_usd_long": None,
            "observed_notional_usd_short": None,
            "observed_notional_usd_total": None,
            "estimated_notional_usd_total": None,
            "event_count": None,
            "semantic_notice": "PAST_OBSERVED_FORCE_ORDERS_NOT_FUTURE_LIQUIDATION_LEVELS",
        }

    raw_status = str(liq_data.get("status") or "").upper()
    cov_str = str(liq_data.get("coverage") or liq_data.get("connection_coverage_status") or "").upper()
    val_str = str(liq_data.get("validity") or "").upper()
    is_healthy = liq_data.get("stream_healthy")

    obs_long = _safe_float(liq_data.get("observed_notional_usd_long"))
    obs_short = _safe_float(liq_data.get("observed_notional_usd_short"))
    obs_total = _safe_float(liq_data.get("observed_notional_usd_total"))
    est_total = _safe_float(liq_data.get("estimated_notional_usd_total"))
    ev_count = _safe_int(liq_data.get("event_count"))
    raw_reason = liq_data.get("reason")

    # Caso B: Stream saudável com cobertura FULL e 0 eventos observados
    is_full_healthy_zero = (
        ("FULL" in cov_str)
        and (val_str in ("VALID", "") or is_healthy is not False)
        and ev_count == 0
        and (obs_total == 0.0 or obs_total is None)
    )

    if is_full_healthy_zero:
        status = "VALID"
        reason = "ZERO_OBSERVED_HEALTHY_STREAM"
        obs_long = 0.0 if obs_long is None or obs_long == 0.0 else obs_long
        obs_short = 0.0 if obs_short is None or obs_short == 0.0 else obs_short
        obs_total = 0.0
        est_total = 0.0 if est_total is None or est_total == 0.0 else est_total
        ev_count = 0
    elif cov_str == "PARTIAL" or val_str == "PARTIAL" or raw_status == "PARTIAL":
        status = "PARTIAL"
        reason = raw_reason or "PARTIAL_CONNECTION_COVERAGE"
    elif (
        cov_str in ("NONE", "DISCONNECTED")
        or is_healthy is False
        or val_str in ("UNKNOWN", "INVALID")
        or raw_status in ("MISSING", "UNKNOWN", "INVALID")
    ):
        status = "MISSING"
        reason = raw_reason or "STREAM_UNHEALTHY_OR_NO_COVERAGE"
        obs_long = None
        obs_short = None
        obs_total = None
        est_total = None
        ev_count = None
    elif obs_total is not None or ev_count is not None:
        status = raw_status if raw_status in ("VALID", "PARTIAL", "MISSING") else ("VALID" if val_str == "VALID" else "VALID")
        reason = raw_reason
    else:
        status = "MISSING"
        reason = "DATA_MISSING"

    result: Dict[str, Any] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "status": status,
        "capability": "OBSERVED_FORCE_ORDER_SNAPSHOT",
        "observed_notional_usd_long": obs_long,
        "observed_notional_usd_short": obs_short,
        "observed_notional_usd_total": obs_total,
        "estimated_notional_usd_total": est_total,
        "event_count": ev_count,
        "semantic_notice": "PAST_OBSERVED_FORCE_ORDERS_NOT_FUTURE_LIQUIDATION_LEVELS",
    }
    if reason:
        result["reason"] = reason
    return result


def _build_non_voting_context_section(
    event_data: Dict[str, Any],
    as_of_ms: Optional[int] = None,
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
    whale_score = _safe_int(wa.get("score") if wa.get("score") is not None else wa.get("s"))
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

    # 4. DERIVATIVES POSITIONING & FUNDING (sem truthiness em 0.0)
    deriv = event_data.get("derivatives", {}) or {}
    btc_deriv = deriv.get("BTCUSDT", {}) or {}
    pos_snapshot = event_data.get("positioning_snapshot", {}) or {}

    raw_oi = btc_deriv.get("open_interest")
    if raw_oi is None:
        raw_oi = pos_snapshot.get("open_interest")
    oi_val = _safe_float(raw_oi)

    raw_oi_usd = btc_deriv.get("open_interest_usd")
    if raw_oi_usd is None:
        raw_oi_usd = pos_snapshot.get("open_interest_usd")
    oi_usd = _safe_float(raw_oi_usd)

    raw_lsr = btc_deriv.get("long_short_ratio")
    if raw_lsr is None:
        raw_lsr = pos_snapshot.get("long_short_ratio")
    lsr_val = _safe_float(raw_lsr)

    fr_val = _safe_float(btc_deriv.get("funding_rate"))
    if fr_val is None:
        raw_fr_pct = btc_deriv.get("funding_rate_percent")
        if raw_fr_pct is None:
            raw_fr_pct = btc_deriv.get("funding_rate_pct")
        fr_pct = _safe_float(raw_fr_pct)
        if fr_pct is not None:
            fr_val = fr_pct / 100.0

    oi_present = (oi_val is not None)
    fr_present = (fr_val is not None)

    pos_status = str(pos_snapshot.get("status") or btc_deriv.get("status") or "").upper()
    if pos_status in ("STALE", "INVALID"):
        deriv_validity = pos_status
    elif oi_present and fr_present:
        deriv_validity = "VALID"
    elif not oi_present and not fr_present:
        deriv_validity = "MISSING"
    else:
        deriv_validity = "PARTIAL"

    context["derivatives"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "open_interest_contracts": oi_val,
        "open_interest_usd": oi_usd,
        "long_short_ratio": lsr_val,
        "funding_rate_decimal": fr_val,
        "funding_rate_unit": "decimal_fraction",
        "is_observed_funding_zero": (fr_val == 0.0) if fr_val is not None else False,
        "validity": deriv_validity,
    }

    # 5. FORCED LIQUIDATIONS TELEMETRY (P2-D / P2-D1 / P2-F2.1)
    context["liquidations"] = _build_liquidations_context(event_data)

    # 6. SCHEDULED MACRO EVENTS (P2-E / P2-F2.1)
    macro_snap = event_data.get("macro_calendar_snapshot", {}) or {}
    nearest_ev = macro_snap.get("nearest_upcoming_event")
    macro_provider_status = macro_snap.get("provider_status", "MISSING")

    time_to_event: Optional[int] = None
    if nearest_ev and isinstance(nearest_ev, dict):
        if "time_to_event_ms" in nearest_ev and nearest_ev["time_to_event_ms"] is not None:
            time_to_event = _safe_int(nearest_ev["time_to_event_ms"])
        elif "scheduled_at_ms" in nearest_ev and nearest_ev["scheduled_at_ms"] is not None:
            sched_ms = _safe_int(nearest_ev["scheduled_at_ms"])
            ref_ms = _safe_int(macro_snap.get("reference_time_ms"))
            if ref_ms is None and as_of_ms is not None:
                ref_ms = as_of_ms
            if sched_ms is not None and ref_ms is not None:
                time_to_event = sched_ms - ref_ms

    nearest_event_dict = None
    if nearest_ev and isinstance(nearest_ev, dict):
        nearest_event_dict = {
            "event_type": nearest_ev.get("event_type"),
            "time_to_event_ms": time_to_event,
            "scheduled_at_utc": nearest_ev.get("scheduled_at_utc"),
            "importance": nearest_ev.get("importance"),
        }

    context["macro_calendar"] = {
        "role": "CONTEXT_ONLY",
        "counts_as_vote": False,
        "provider_status": macro_provider_status,
        "nearest_event": nearest_event_dict,
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
    execution_context: Dict[str, Any],
    non_voting_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Centraliza métricas de qualidade, completude e validade dos dados.
    Expõe exclusivamente fatos observáveis sem thresholds inventados nem booleanos de decisão.
    """
    fluxo = event_data.get("fluxo_continuo", {}) or {}
    integrity = fluxo.get("flow_window_integrity", {}) or {}

    missing_context: List[str] = []
    degraded_context: List[str] = []

    # 1. Derivatives
    deriv = non_voting_context.get("derivatives", {})
    deriv_val = deriv.get("validity")
    if deriv_val == "MISSING":
        missing_context.append("derivatives")
    elif deriv_val in ("PARTIAL", "STALE", "INVALID"):
        degraded_context.append(f"derivatives_{deriv_val.lower()}")

    # 2. Macro calendar
    macro = non_voting_context.get("macro_calendar", {})
    macro_status = macro.get("provider_status")
    if macro_status == "MISSING":
        missing_context.append("macro_calendar")
    elif macro_status in ("DEGRADED", "PARTIAL", "STALE", "UNSUPPORTED"):
        degraded_context.append(f"macro_{macro_status.lower()}")

    # 3. Liquidations
    liq = non_voting_context.get("liquidations", {})
    liq_status = liq.get("status")
    if liq_status == "MISSING":
        missing_context.append("liquidations")
    elif liq_status in ("PARTIAL", "DEGRADED", "STALE"):
        degraded_context.append(f"liquidations_{liq_status.lower()}")

    # 4. Execution context
    exec_val = execution_context.get("validity")
    if exec_val == "MISSING":
        missing_context.append("execution_context")
    elif exec_val in ("PARTIAL", "INSUFFICIENT_LIQUIDITY"):
        degraded_context.append(f"execution_{exec_val.lower()}")

    # 5. Flow window integrity
    for win_key, win_data in integrity.items():
        if isinstance(win_data, dict):
            w_status = str(win_data.get("status") or "").upper()
            if w_status and w_status != "FULL":
                degraded_context.append(f"flow_integrity_{win_key}_{w_status.lower()}")

    # 6. Regime
    reg = non_voting_context.get("regime", {})
    reg_status = reg.get("status")
    if reg_status in ("PARTIAL", "INSUFFICIENT", "DEGRADED"):
        degraded_context.append(f"regime_{reg_status.lower()}")
    elif reg.get("mode") == "UNKNOWN" and reg_status == "MISSING":
        missing_context.append("regime")

    return {
        "flow_window_integrity": integrity,
        "orderbook_capability": "SNAPSHOT_ONLY",
        "reconciler_status": reconciler_result.status.value,
        "valid_directional_evidence_count": reconciler_result.evidence_count,
        "missing_context": sorted(missing_context),
        "degraded_context": sorted(degraded_context),
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
    non_voting_context = _build_non_voting_context_section(event_data, as_of_ms=now_ms)
    data_quality = _build_data_quality_section(
        event_data,
        directional_evidence,
        reconciler_result,
        execution_context,
        non_voting_context,
    )

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
