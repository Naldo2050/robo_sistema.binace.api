# flow_analyzer/effort_response.py
"""
P1-D — Effort vs Result RAW Metrics v1 (métricas puras, sem interpretação).

Representa, sem classificar e sem prever:
- "quanto esforço agressor ocorreu" (notionals buy/sell já classificados);
- "qual resposta de preço foi observada" (deslocamento/range/posições).

Contrato de origem (data_processing/data_handler.py:1119-1130):
notional_usdt = prices * qtys particionado por m_flags, onde m True=SELL e
False=BUY (linha 254; compatível com o campo 'm'/buyer_is_maker do aggTrade,
linha 551). Logo buy_notional_usdt JÁ é agressão BUY e sell_notional_usdt JÁ
é agressão SELL: NUNCA multiplicar de novo por aggressive_buy_pct/sell_pct
(que é a mesma partição em forma percentual — faria double counting).
OHLC vem dos preços dos MESMOS aggTrades da janela (linhas 1066-1098): a
linhagem de preço NÃO é disjunta da linhagem dos notionals.

Sem IO, estado, rede ou dependência pesada. O(1). Sem métricas pós-entrada
(futuro outcome/evaluation). Sem ratio esforço/preço (sem contrato de
denominador). Sem thresholds, classes, sinais ou confidence.
"""
from __future__ import annotations

import math
from typing import Any, Optional

VALID = "VALID"
PARTIAL = "PARTIAL"
INVALID = "INVALID"

COMPLETE = "COMPLETE"

_BPS = 10000.0


def _finite_number(value: Any) -> Optional[float]:
    """float finito ou None (bool/None/str inválida/NaN/±Inf => None)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def compute_effort_response(
    *,
    buy_notional_usd: Any,
    sell_notional_usd: Any,
    open: Any,
    high: Any,
    low: Any,
    close: Any,
    window_duration_ms: Any,
    vwap: Any = None,
    poc: Any = None,
) -> dict:
    """Métricas RAW esforço-vs-resposta (O(1), puras).

    Retorna dict plano:
    - métricas numéricas ou None
    - `core_validity`: VALID / PARTIAL (zero total) / INVALID (incoerente)
    - `optional_completeness`: COMPLETE (vwap e poc válidos) / PARTIAL
    - `validity`: global compatibilidade (VALID somente se core VALID + optional COMPLETE)
    - `reasons`: {campo: motivo} só p/ nulos/causas
    - `window_duration_ms` ecoado.
    """
    reasons: dict = {}

    buy = _finite_number(buy_notional_usd)
    sell = _finite_number(sell_notional_usd)
    o = _finite_number(open)
    h = _finite_number(high)
    lo = _finite_number(low)
    c = _finite_number(close)
    duration = _finite_number(window_duration_ms)
    vwap_v = _finite_number(vwap) if vwap is not None else None
    poc_v = _finite_number(poc) if poc is not None else None

    invalid_reasons: list = []
    if buy is None:
        invalid_reasons.append("buy_notional_usd missing/non-finite")
    elif buy < 0:
        invalid_reasons.append("buy_notional_usd negative")
    if sell is None:
        invalid_reasons.append("sell_notional_usd missing/non-finite")
    elif sell < 0:
        invalid_reasons.append("sell_notional_usd negative")
    if o is None or o <= 0:
        invalid_reasons.append("open missing/non-finite/non-positive")
    for _name, _v in (("high", h), ("low", lo), ("close", c)):
        if _v is None:
            invalid_reasons.append(f"{_name} missing/non-finite")
        elif _v <= 0:
            invalid_reasons.append(f"{_name} non-positive")
    if not invalid_reasons:
        # Invariantes matemáticas do OHLC (sem tolerância arbitrária).
        if h < lo:
            invalid_reasons.append("high<low")
        if h < o or h < c:
            invalid_reasons.append("high below open/close")
        if lo > o or lo > c:
            invalid_reasons.append("low above open/close")
    if duration is None:
        invalid_reasons.append("window_duration_ms missing/non-finite")
    elif duration <= 0:
        invalid_reasons.append("window_duration_ms non-positive")

    out: dict = {
        "buy_notional_usd": None,
        "sell_notional_usd": None,
        "total_aggressive_notional_usd": None,
        "net_aggressive_notional_usd": None,
        "buy_share": None,
        "sell_share": None,
        "price_displacement_usd": None,
        "price_displacement_bps": None,
        "range_usd": None,
        "range_bps": None,
        "close_from_high_usd": None,
        "close_from_high_bps": None,
        "close_from_low_usd": None,
        "close_from_low_bps": None,
        "close_vs_vwap_usd": None,
        "close_vs_vwap_bps": None,
        "close_vs_poc_usd": None,
        "close_vs_poc_bps": None,
        "window_duration_ms": duration,
        "core_validity": INVALID,
        "optional_completeness": PARTIAL,
        "validity": INVALID,
        "reasons": {},
    }
    if invalid_reasons:
        for _field in ("buy_notional_usd", "sell_notional_usd",
                       "price_displacement_usd", "price_displacement_bps",
                       "range_usd", "range_bps", "close_from_high_usd",
                       "close_from_high_bps", "close_from_low_usd",
                       "close_from_low_bps"):
            reasons[_field] = ";".join(invalid_reasons)
        out["reasons"] = reasons
        return out

    total = buy + sell
    out["buy_notional_usd"] = buy
    out["sell_notional_usd"] = sell
    out["total_aggressive_notional_usd"] = total
    out["net_aggressive_notional_usd"] = buy - sell  # zero permitido
    if total > 0:
        out["buy_share"] = buy / total
        out["sell_share"] = sell / total
        core_validity = VALID
    else:
        reasons["buy_share"] = "ZERO_TOTAL_NOTIONAL"
        reasons["sell_share"] = "ZERO_TOTAL_NOTIONAL"
        core_validity = PARTIAL

    out["price_displacement_usd"] = c - o
    out["price_displacement_bps"] = (c - o) / o * _BPS
    out["range_usd"] = h - lo
    out["range_bps"] = (h - lo) / o * _BPS
    out["close_from_high_usd"] = c - h
    out["close_from_high_bps"] = (c - h) / o * _BPS
    out["close_from_low_usd"] = c - lo
    out["close_from_low_bps"] = (c - lo) / o * _BPS

    vwap_valid = False
    if vwap_v is not None and vwap_v > 0:
        out["close_vs_vwap_usd"] = c - vwap_v
        out["close_vs_vwap_bps"] = (c - vwap_v) / o * _BPS
        vwap_valid = True
    else:
        if vwap is not None:
            reasons["close_vs_vwap_usd"] = "vwap missing/non-finite/non-positive"
            reasons["close_vs_vwap_bps"] = "vwap missing/non-finite/non-positive"

    poc_valid = False
    if poc_v is not None and poc_v > 0:
        out["close_vs_poc_usd"] = c - poc_v
        out["close_vs_poc_bps"] = (c - poc_v) / o * _BPS
        poc_valid = True
    else:
        if poc is not None:
            reasons["close_vs_poc_usd"] = "poc missing/non-finite/non-positive"
            reasons["close_vs_poc_bps"] = "poc missing/non-finite/non-positive"

    optional_completeness = COMPLETE if (vwap_valid and poc_valid) else PARTIAL

    out["core_validity"] = core_validity
    out["optional_completeness"] = optional_completeness

    if core_validity == VALID and optional_completeness == COMPLETE:
        out["validity"] = VALID
    else:
        out["validity"] = PARTIAL

    out["reasons"] = reasons
    return out


# ── Adapter P1-A (puro, opcional) ────────────────────────────────────────────

_EFFORT_FIELDS = {
    "buy_notional_usd", "sell_notional_usd", "total_aggressive_notional_usd",
    "net_aggressive_notional_usd", "buy_share", "sell_share",
}
_PRICE_FIELDS = {
    "price_displacement_usd", "price_displacement_bps", "range_usd",
    "range_bps", "close_from_high_usd", "close_from_high_bps",
    "close_from_low_usd", "close_from_low_bps", "close_vs_vwap_usd",
    "close_vs_vwap_bps", "close_vs_poc_usd", "close_vs_poc_bps",
}

_EFFORT_DERIVED = {
    "buy_notional_usd": ("raw.aggtrade.buy_notional",),
    "sell_notional_usd": ("raw.aggtrade.sell_notional",),
    "total_aggressive_notional_usd": ("raw.aggtrade.buy_notional",
                                      "raw.aggtrade.sell_notional"),
    "net_aggressive_notional_usd": ("raw.aggtrade.buy_notional",
                                    "raw.aggtrade.sell_notional"),
    "buy_share": ("raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional"),
    "sell_share": ("raw.aggtrade.buy_notional", "raw.aggtrade.sell_notional"),
}
_PRICE_DERIVED = {
    "price_displacement_usd": ("raw.aggtrade.price",),
    "price_displacement_bps": ("raw.aggtrade.price",),
    "range_usd": ("raw.aggtrade.price",),
    "range_bps": ("raw.aggtrade.price",),
    "close_from_high_usd": ("raw.aggtrade.price",),
    "close_from_high_bps": ("raw.aggtrade.price",),
    "close_from_low_usd": ("raw.aggtrade.price",),
    "close_from_low_bps": ("raw.aggtrade.price",),
    "close_vs_vwap_usd": ("raw.aggtrade.price", "raw.trades.price_qty"),
    "close_vs_vwap_bps": ("raw.aggtrade.price", "raw.trades.price_qty"),
    "close_vs_poc_usd": ("raw.aggtrade.price", "raw.trades.price_qty"),
    "close_vs_poc_bps": ("raw.aggtrade.price", "raw.trades.price_qty"),
}


def effort_response_to_evidence(metrics: dict,
                                *,
                                source: str = "effort_response",
                                observed_at_ms=None) -> list:
    """Métricas presentes => lista de Evidence (uma por escalar).

    Opção B (validade atômica por métrica):
    - Se core_validity == INVALID (ou global validity == INVALID): lista vazia.
    - Cada métrica atômica calculada e finita nasce como EvidenceValidity.VALID.
    - Opcionais ausentes (None) NÃO geram Evidence fantasma.
    - Em caso de zero total: shares são omitidos (None), preço íntegro gera
      Evidence VALID, e notionals recebem metadata de cenário não produtivo.
    - Todas com counts_as_vote=false, direction UNKNOWN (o sinal fica no value),
      calibration NOT_APPLICABLE (medição física).
    """
    from institutional.evidence import (Evidence, EvidenceCalibration,
                                        EvidenceDirection, EvidenceFamily,
                                        EvidenceType, EvidenceValidity)
    if not isinstance(metrics, dict):
        return []
    core_validity = metrics.get("core_validity", metrics.get("validity"))
    if core_validity == INVALID:
        return []

    is_zero_total = (metrics.get("total_aggressive_notional_usd") == 0.0)

    out = []
    for name in sorted(_EFFORT_FIELDS | _PRICE_FIELDS):
        value = metrics.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue

        if name in _EFFORT_FIELDS:
            family = EvidenceFamily.EXECUTED_FLOW
            derived = _EFFORT_DERIVED[name]
        else:
            family = EvidenceFamily.PRICE_RESPONSE
            derived = _PRICE_DERIVED[name]

        meta = {"metric": name}
        if name in _PRICE_FIELDS:
            meta["same_tape_as_flow"] = True
        if is_zero_total and name in _EFFORT_FIELDS:
            meta["zero_total_notional"] = True
            meta["non_productive_scenario"] = True

        out.append(Evidence(
            source=source,
            family=family,
            evidence_type=EvidenceType.CONTINUOUS_TRADES,
            direction=EvidenceDirection.UNKNOWN,
            value=v,
            observed_at_ms=observed_at_ms,
            validity=EvidenceValidity.VALID,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            counts_as_vote=False,
            derived_from=derived,
            metadata=meta,
        ))
    return out
