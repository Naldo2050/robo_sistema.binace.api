# fetchers/binance_liquidation_stream.py
# -*- coding: utf-8 -*-
"""
P2-D — Binance Forced Liquidations Telemetry Contract v1.

Consome, normaliza e agrega telemetria de liquidações forçadas da Binance
USD-M Futures (forceOrder / allForceOrders).

Garante:
1. Semântica estrita do Side da liquidação:
   - forceOrder S="SELL"  => liquidated_position_side = "LONG"
   - forceOrder S="BUY"   => liquidated_position_side = "SHORT"
2. Quantidade e Preço observados:
   - Prioriza filled quantity (z / executedQty / l) sobre original quantity (q).
   - Prioriza average execution price (ap / averagePrice) com fallback transparente para limit price (p).
   - observed_notional_usd = filled_qty * average_price.
3. Deduplicação determinística:
   - event_id determinístico derivado dos atributos físicos da exchange.
4. Capability declarada honestamente:
   - OBSERVED_FORCE_ORDER_SNAPSHOT (a Binance publica snapshots das ordens de liquidação;
     NÃO é um complete continuous tape).
5. Agregação contemporânea por janela temporal [window_start_ms, window_end_ms).
6. Distinção estrita:
   - stream saudável + 0 eventos => VALID (event_count=0, notionals=0.0 como ZERO_OBSERVED).
   - stream indisponível/desconectado => UNKNOWN/MISSING (notionals=None).
7. Evidence v1 adapter:
   - Family: DERIVATIVES
   - Type: SLOW_CONTEXT (conforme capability snapshot)
   - counts_as_vote: False (NÃO vota)
   - direction: UNKNOWN (NÃO infere direção/sinal)
   - calibration: NOT_APPLICABLE
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
    _json_safe,
)

LIQUIDATION_CONTRACT_VERSION = "1.0.0"

# Capability formal: a Binance publica a última ordem de liquidação ou lotes amostrados.
# NÃO afirmar COMPLETE_LIQUIDATION_TAPE.
STREAM_CAPABILITY = "OBSERVED_FORCE_ORDER_SNAPSHOT"


class OrderSide(str, Enum):
    """Lado da ordem forçada emitida no mercado."""
    SELL = "SELL"
    BUY = "BUY"
    UNKNOWN = "UNKNOWN"


class LiquidatedPositionSide(str, Enum):
    """Lado da posição do trader que foi compulsoriamente liquidada."""
    LONG = "LONG"    # Liquidada por uma ordem de venda forçada (SELL)
    SHORT = "SHORT"  # Liquidada por uma ordem de compra forçada (BUY)
    UNKNOWN = "UNKNOWN"


class LiquidationValidity(str, Enum):
    """Validade do evento ou resumo de liquidação."""
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class LargerObservedSide(str, Enum):
    """Comparação estritamente matemática entre lados observados na janela (sem conotação direcional)."""
    LONG = "LONG"
    SHORT = "SHORT"
    EQUAL = "EQUAL"
    NONE = "NONE"


def _finite_or_none(val: Any) -> Optional[float]:
    """Coage número para float estritamente finito. Rejeita bool, strings e non-finite."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class ForcedLiquidationEvent:
    """Evento individual normalizado de liquidação forçada Binance USD-M."""
    symbol: str
    event_time_ms: int
    trade_time_ms: int
    order_side: str                     # "SELL" ou "BUY"
    liquidated_position_side: str       # "LONG" ou "SHORT"
    original_qty: Optional[float]
    filled_qty: Optional[float]
    average_price: Optional[float]
    observed_notional_usd: Optional[float]
    order_status: str
    source: str = "binance_usdm"
    stream: str = "<symbol>@forceOrder"
    validity: LiquidationValidity = LiquidationValidity.VALID
    reason: Optional[str] = None
    event_id: str = ""
    capability: str = STREAM_CAPABILITY
    contract_version: str = LIQUIDATION_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialização determinística RFC 8259 sem valores não-finitos."""
        return {
            "contract_version": self.contract_version,
            "event_id": self.event_id,
            "symbol": self.symbol,
            "source": self.source,
            "stream": self.stream,
            "capability": self.capability,
            "event_time_ms": self.event_time_ms,
            "trade_time_ms": self.trade_time_ms,
            "order_side": self.order_side,
            "liquidated_position_side": self.liquidated_position_side,
            "original_qty": self.original_qty,
            "filled_qty": self.filled_qty,
            "average_price": self.average_price,
            "observed_notional_usd": self.observed_notional_usd,
            "order_status": self.order_status,
            "validity": (
                self.validity.value
                if isinstance(self.validity, LiquidationValidity)
                else str(self.validity)
            ),
            "reason": self.reason,
        }


def parse_force_order_payload(payload: Any) -> ForcedLiquidationEvent:
    """
    Parser robusto e fail-closed para payloads de liquidação da Binance Futures.

    Suporta:
    1. Stream direto WebSocket: {"e": "forceOrder", "E": ..., "o": {...}}
    2. Envelope combinado: {"stream": "btcusdt@forceOrder", "data": {...}}
    3. Item de REST /fapi/v1/allForceOrders: {"symbol": ..., "price": ..., "executedQty": ..., "side": ...}
    """
    if not isinstance(payload, dict):
        return ForcedLiquidationEvent(
            symbol="UNKNOWN",
            event_time_ms=0,
            trade_time_ms=0,
            order_side=OrderSide.UNKNOWN.value,
            liquidated_position_side=LiquidatedPositionSide.UNKNOWN.value,
            original_qty=None,
            filled_qty=None,
            average_price=None,
            observed_notional_usd=None,
            order_status="UNKNOWN",
            validity=LiquidationValidity.INVALID,
            reason="PAYLOAD_NOT_A_DICT",
        )

    # Unwrap envelope combinado se presente
    data = payload.get("data", payload) if "data" in payload and isinstance(payload.get("data"), dict) else payload

    # Identificar formato (WebSocket 'o' order object vs REST flat)
    order_dict = data.get("o", data) if isinstance(data.get("o"), dict) else data

    # 1. Símbolo
    raw_symbol = order_dict.get("s") or order_dict.get("symbol") or "UNKNOWN"
    symbol = str(raw_symbol).upper().strip()

    # 2. Timestamps
    event_time_raw = data.get("E") or data.get("eventTime") or order_dict.get("T") or order_dict.get("time") or 0
    trade_time_raw = order_dict.get("T") or order_dict.get("time") or data.get("E") or 0

    try:
        event_time_ms = int(event_time_raw)
        trade_time_ms = int(trade_time_raw)
    except (ValueError, TypeError):
        event_time_ms = 0
        trade_time_ms = 0

    # 3. Side Semântica: SELL => LONG liquidado; BUY => SHORT liquidado
    raw_side = str(order_dict.get("S") or order_dict.get("side") or "").upper().strip()
    order_side: str
    liquidated_position_side: str
    side_valid = True

    if raw_side == "SELL":
        order_side = OrderSide.SELL.value
        liquidated_position_side = LiquidatedPositionSide.LONG.value
    elif raw_side == "BUY":
        order_side = OrderSide.BUY.value
        liquidated_position_side = LiquidatedPositionSide.SHORT.value
    else:
        order_side = OrderSide.UNKNOWN.value
        liquidated_position_side = LiquidatedPositionSide.UNKNOWN.value
        side_valid = False

    # 4. Quantidade e Preço
    raw_orig_qty = order_dict.get("q") or order_dict.get("origQty")
    raw_exec_qty = order_dict.get("z") or order_dict.get("executedQty")
    raw_last_qty = order_dict.get("l") or order_dict.get("lastFilledQty")
    raw_avg_price = order_dict.get("ap") or order_dict.get("averagePrice")
    raw_limit_price = order_dict.get("p") or order_dict.get("price")
    order_status = str(order_dict.get("X") or order_dict.get("status") or "UNKNOWN").upper().strip()

    orig_qty = _finite_or_none(raw_orig_qty)
    exec_qty = _finite_or_none(raw_exec_qty)
    last_qty = _finite_or_none(raw_last_qty)
    avg_price = _finite_or_none(raw_avg_price)
    limit_price = _finite_or_none(raw_limit_price)

    # Determinar filled_qty
    filled_qty: Optional[float] = None
    qty_reasons: List[str] = []

    if exec_qty is not None and exec_qty > 0.0:
        filled_qty = exec_qty
    elif last_qty is not None and last_qty > 0.0:
        filled_qty = last_qty
        qty_reasons.append("LAST_QTY_USED")
    elif orig_qty is not None and orig_qty > 0.0 and order_status in ("FILLED", "NEW"):
        # Em alguns snapshots iniciais, z pode ser 0 ou omitido enquanto q está preenchido
        filled_qty = orig_qty
        qty_reasons.append("ORIG_QTY_FALLBACK")

    # Determinar execution price
    execution_price: Optional[float] = None
    price_reasons: List[str] = []

    if avg_price is not None and avg_price > 0.0:
        execution_price = avg_price
    elif limit_price is not None and limit_price > 0.0:
        execution_price = limit_price
        price_reasons.append("LIMIT_PRICE_FALLBACK")

    # 5. Cálculo do Notional
    observed_notional_usd: Optional[float] = None
    if filled_qty is not None and execution_price is not None:
        if filled_qty > 0 and execution_price > 0:
            observed_notional_usd = round(filled_qty * execution_price, 2)

    # 6. Avaliação de Validade
    validity = LiquidationValidity.VALID
    reasons: List[str] = []
    reasons.extend(qty_reasons)
    reasons.extend(price_reasons)

    # Checagem estrita de nonfinite ou valores estritamente negativos (< 0)
    has_nonfinite = any(
        raw is not None and _finite_or_none(raw) is None
        for raw in (raw_orig_qty, raw_exec_qty, raw_last_qty, raw_avg_price, raw_limit_price)
    )
    has_strictly_negative = any(
        v is not None and v < 0.0
        for v in (orig_qty, exec_qty, last_qty, avg_price, limit_price)
    )

    if not side_valid:
        validity = LiquidationValidity.INVALID
        reasons.append("INVALID_SIDE")
    elif symbol in ("UNKNOWN", ""):
        validity = LiquidationValidity.INVALID
        reasons.append("INVALID_SYMBOL")
    elif trade_time_ms <= 0 or event_time_ms <= 0:
        validity = LiquidationValidity.INVALID
        reasons.append("INVALID_TIMESTAMPS")
    elif has_nonfinite:
        validity = LiquidationValidity.INVALID
        reasons.append("NONFINITE_INPUT")
    elif has_strictly_negative:
        validity = LiquidationValidity.INVALID
        reasons.append("NEGATIVE_PRICE_OR_QTY")
    elif (filled_qty is not None and filled_qty <= 0) or (execution_price is not None and execution_price <= 0):
        validity = LiquidationValidity.INVALID
        reasons.append("NON_POSITIVE_PRICE_OR_QTY")
    elif filled_qty is None or execution_price is None or observed_notional_usd is None:
        validity = LiquidationValidity.PARTIAL
        reasons.append("NOTIONAL_CANNOT_BE_CALCULATED")

    # 7. Gerar event_id determinístico para deduplicação
    # symbol + trade_time + order_side + filled_qty + exec_price + status
    fq_str = f"{filled_qty:.8f}" if filled_qty is not None else "none"
    pr_str = f"{execution_price:.2f}" if execution_price is not None else "none"
    event_id = f"{symbol}_{trade_time_ms}_{order_side}_{fq_str}_{pr_str}_{order_status}"

    return ForcedLiquidationEvent(
        symbol=symbol,
        event_time_ms=event_time_ms,
        trade_time_ms=trade_time_ms,
        order_side=order_side,
        liquidated_position_side=liquidated_position_side,
        original_qty=orig_qty,
        filled_qty=filled_qty,
        average_price=execution_price,
        observed_notional_usd=observed_notional_usd,
        order_status=order_status,
        source="binance_usdm",
        stream=f"{symbol.lower()}@forceOrder",
        validity=validity,
        reason=";".join(reasons) if reasons else None,
        event_id=event_id,
        capability=STREAM_CAPABILITY,
    )


@dataclass(frozen=True)
class LiquidationWindowSummary:
    """Resumo contemporâneo de liquidações agregadas em uma janela de tempo."""
    window_start_ms: int
    window_end_ms: int
    event_count: Optional[int]
    long_liquidated_qty: Optional[float]
    short_liquidated_qty: Optional[float]
    long_liquidated_notional_usd: Optional[float]
    short_liquidated_notional_usd: Optional[float]
    total_liquidated_notional_usd: Optional[float]
    latest_event_ms: Optional[int] = None
    source: str = "binance_usdm"
    coverage: str = STREAM_CAPABILITY
    validity: LiquidationValidity = LiquidationValidity.VALID
    reason: Optional[str] = None
    larger_observed_side: str = LargerObservedSide.NONE.value
    stream_healthy: bool = True
    contract_version: str = LIQUIDATION_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialização determinística RFC 8259."""
        return {
            "contract_version": self.contract_version,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "event_count": self.event_count,
            "long_liquidated_qty": self.long_liquidated_qty,
            "short_liquidated_qty": self.short_liquidated_qty,
            "long_liquidated_notional_usd": self.long_liquidated_notional_usd,
            "short_liquidated_notional_usd": self.short_liquidated_notional_usd,
            "total_liquidated_notional_usd": self.total_liquidated_notional_usd,
            "latest_event_ms": self.latest_event_ms,
            "source": self.source,
            "coverage": self.coverage,
            "validity": (
                self.validity.value
                if isinstance(self.validity, LiquidationValidity)
                else str(self.validity)
            ),
            "reason": self.reason,
            "larger_observed_side": self.larger_observed_side,
            "stream_healthy": self.stream_healthy,
        }


class LiquidationWindowAggregator:
    """
    Agregador puro contemporâneo de liquidações com deduplicação determinística.
    Opera estritamente por janela temporal sem alterar trading nem criar previsões.
    """

    def __init__(self, symbol: str = "BTCUSDT", max_seen_events: int = 10_000):
        self.symbol = symbol.upper().strip()
        self.max_seen_events = max_seen_events
        self._events: List[ForcedLiquidationEvent] = []
        self._seen_event_ids: Set[str] = set()

    def add_event(self, raw_or_parsed: Union[ForcedLiquidationEvent, Dict[str, Any]]) -> bool:
        """
        Adiciona evento com deduplicação estrita.
        Retorna True se aceito, False se duplicado ou descartado.
        """
        event: ForcedLiquidationEvent
        if isinstance(raw_or_parsed, ForcedLiquidationEvent):
            event = raw_or_parsed
        else:
            event = parse_force_order_payload(raw_or_parsed)

        if not event.event_id or event.event_id in self._seen_event_ids:
            return False

        if len(self._seen_event_ids) >= self.max_seen_events:
            # Poda metade dos mais antigos para evitar vazamento de memória em execuções longas
            half = self.max_seen_events // 2
            evs_to_keep = self._events[-half:]
            self._seen_event_ids = {e.event_id for e in evs_to_keep if e.event_id}
            self._events = evs_to_keep

        self._seen_event_ids.add(event.event_id)
        self._events.append(event)
        return True

    def summarize_window(
        self,
        window_start_ms: int,
        window_end_ms: int,
        stream_healthy: bool = True,
    ) -> LiquidationWindowSummary:
        """
        Agrega eventos observados dentro do intervalo semi-aberto [window_start_ms, window_end_ms).

        Regra estrita:
        - Stream saudável + 0 eventos: validade VALID, event_count=0, notionals 0.0 (ZERO_OBSERVED).
        - Stream não-saudável / desconectado: validade UNKNOWN, notionals=None.
        """
        if not stream_healthy:
            return LiquidationWindowSummary(
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                event_count=None,
                long_liquidated_qty=None,
                short_liquidated_qty=None,
                long_liquidated_notional_usd=None,
                short_liquidated_notional_usd=None,
                total_liquidated_notional_usd=None,
                latest_event_ms=None,
                source="binance_usdm",
                coverage=STREAM_CAPABILITY,
                validity=LiquidationValidity.UNKNOWN,
                reason="STREAM_UNHEALTHY_OR_DISCONNECTED",
                larger_observed_side=LargerObservedSide.NONE.value,
                stream_healthy=False,
            )

        # Filtrar eventos da janela pelo trade_time_ms (ou event_time_ms se trade_time=0)
        window_events = [
            e for e in self._events
            if e.symbol == self.symbol
            and window_start_ms <= (e.trade_time_ms or e.event_time_ms) < window_end_ms
        ]

        if not window_events:
            return LiquidationWindowSummary(
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                event_count=0,
                long_liquidated_qty=0.0,
                short_liquidated_qty=0.0,
                long_liquidated_notional_usd=0.0,
                short_liquidated_notional_usd=0.0,
                total_liquidated_notional_usd=0.0,
                latest_event_ms=None,
                source="binance_usdm",
                coverage=STREAM_CAPABILITY,
                validity=LiquidationValidity.VALID,
                reason="ZERO_OBSERVED_HEALTHY_STREAM",
                larger_observed_side=LargerObservedSide.NONE.value,
                stream_healthy=True,
            )

        long_qty = 0.0
        short_qty = 0.0
        long_notional = 0.0
        short_notional = 0.0
        has_partial = False
        has_invalid = False
        latest_ts = 0

        valid_events_count = 0

        for e in window_events:
            ts = e.trade_time_ms or e.event_time_ms
            if ts > latest_ts:
                latest_ts = ts

            if e.validity == LiquidationValidity.INVALID:
                has_invalid = True
                continue

            if e.validity == LiquidationValidity.PARTIAL:
                has_partial = True

            valid_events_count += 1

            if e.liquidated_position_side == LiquidatedPositionSide.LONG.value:
                if e.filled_qty is not None:
                    long_qty += e.filled_qty
                if e.observed_notional_usd is not None:
                    long_notional += e.observed_notional_usd
            elif e.liquidated_position_side == LiquidatedPositionSide.SHORT.value:
                if e.filled_qty is not None:
                    short_qty += e.filled_qty
                if e.observed_notional_usd is not None:
                    short_notional += e.observed_notional_usd

        total_notional = long_notional + short_notional

        # Comparação matemática descritiva (sem viés de sinal)
        larger_side = LargerObservedSide.NONE.value
        if long_notional > short_notional:
            larger_side = LargerObservedSide.LONG.value
        elif short_notional > long_notional:
            larger_side = LargerObservedSide.SHORT.value
        elif total_notional > 0:
            larger_side = LargerObservedSide.EQUAL.value

        validity = LiquidationValidity.VALID
        reasons = []
        if has_invalid:
            reasons.append("WINDOW_CONTAINS_INVALID_EVENTS")
        if has_partial:
            validity = LiquidationValidity.PARTIAL
            reasons.append("WINDOW_CONTAINS_PARTIAL_EVENTS")

        return LiquidationWindowSummary(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            event_count=valid_events_count,
            long_liquidated_qty=round(long_qty, 4),
            short_liquidated_qty=round(short_qty, 4),
            long_liquidated_notional_usd=round(long_notional, 2),
            short_liquidated_notional_usd=round(short_notional, 2),
            total_liquidated_notional_usd=round(total_notional, 2),
            latest_event_ms=latest_ts if latest_ts > 0 else None,
            source="binance_usdm",
            coverage=STREAM_CAPABILITY,
            validity=validity,
            reason=";".join(reasons) if reasons else None,
            larger_observed_side=larger_side,
            stream_healthy=True,
        )


def liquidation_summary_to_evidence(
    summary: LiquidationWindowSummary,
    symbol: str = "BTCUSDT",
) -> List[Evidence]:
    """
    Adapter que converte um LiquidationWindowSummary nas 4 instâncias canônicas
    de Evidence v1 registradas na taxonomia institucional.

    Contrato estrito:
    - Family: DERIVATIVES
    - Type: SLOW_CONTEXT (conforme capability snapshot throttled)
    - counts_as_vote: False (estritamente não-votante)
    - direction: UNKNOWN (estritamente sem inferência de sinal)
    - calibration: NOT_APPLICABLE
    """
    horizon_ms = max(0, summary.window_end_ms - summary.window_start_ms)
    observed_at = summary.latest_event_ms or summary.window_end_ms

    base_provenance = {
        "exchange": "binance",
        "market": "usdm_futures",
        "symbol": symbol,
        "source": summary.source,
        "capability": summary.coverage,
        "contract_version": summary.contract_version,
        "window_start_ms": summary.window_start_ms,
        "window_end_ms": summary.window_end_ms,
        "larger_observed_side": summary.larger_observed_side,
    }

    ev_validity: EvidenceValidity
    if summary.validity == LiquidationValidity.VALID:
        ev_validity = EvidenceValidity.VALID
    elif summary.validity == LiquidationValidity.PARTIAL:
        ev_validity = EvidenceValidity.PARTIAL
    elif summary.validity == LiquidationValidity.INVALID:
        ev_validity = EvidenceValidity.INVALID
    else:
        ev_validity = EvidenceValidity.UNKNOWN

    field_specs = [
        (
            "derivatives.liquidations.long_observed_notional",
            summary.long_liquidated_notional_usd,
            "USD",
            {"liquidated_position_side": "LONG"},
        ),
        (
            "derivatives.liquidations.short_observed_notional",
            summary.short_liquidated_notional_usd,
            "USD",
            {"liquidated_position_side": "SHORT"},
        ),
        (
            "derivatives.liquidations.total_observed_notional",
            summary.total_liquidated_notional_usd,
            "USD",
            {"is_composite": True},
        ),
        (
            "derivatives.liquidations.event_count",
            float(summary.event_count) if summary.event_count is not None else None,
            "count",
            {},
        ),
    ]

    evidences: List[Evidence] = []

    for field_id, raw_val, unit, extra_meta in field_specs:
        val: Optional[float] = None
        if ev_validity in (EvidenceValidity.VALID, EvidenceValidity.PARTIAL) and raw_val is not None:
            v_fin = _finite_or_none(raw_val)
            if v_fin is not None:
                val = v_fin

        ev = Evidence(
            source=summary.source,
            family=EvidenceFamily.DERIVATIVES,
            evidence_type=EvidenceType.SLOW_CONTEXT,
            direction=EvidenceDirection.UNKNOWN,
            value=val,
            observed_at_ms=observed_at,
            horizon_ms=horizon_ms,
            provenance={**base_provenance, "field_id": field_id, "unit": unit},
            validity=ev_validity,
            reason=summary.reason,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            counts_as_vote=False,
            derived_from=(),
            metadata={
                "field_id": field_id,
                "unit": unit,
                "capability": summary.coverage,
                **extra_meta,
            },
        )
        evidences.append(ev)

    return evidences
