# fetchers/binance_liquidation_stream.py
# -*- coding: utf-8 -*-
"""
P2-D1.1 — Binance Forced Liquidations Telemetry Contract v1.

Consome, normaliza e agrega telemetria de liquidações forçadas da Binance
USD-M Futures (forceOrder / allForceOrders) com endurecimento semântico estrito.

Garante:
1. Semântica estrita do Side da liquidação:
   - forceOrder S="SELL"  => liquidated_position_side = "LONG"
   - forceOrder S="BUY"   => liquidated_position_side = "SHORT"
2. Distinção Rígida entre Preço Observado vs Estimado (Price Fallback):
   - ap (average execution price) > 0:
     * average_price = ap
     * observed_notional_usd = filled_qty * ap
     * notional_quality = OBSERVED_EXECUTION
     * validity = VALID
   - ap ausente/zero mas p (limit price) > 0:
     * average_price = None (não mente que houve execução observada a esse preço)
     * limit_price = p
     * observed_notional_usd = None (NÃO preenchido como se fosse observado)
     * estimated_notional_usd = filled_qty * p
     * notional_quality = ESTIMATED_LIMIT_PRICE
     * validity = PARTIAL
     * reason inclui LIMIT_PRICE_FALLBACK
3. Agregação Semântica:
   - observed totals somam EXCLUSIVAMENTE observed_notional_usd.
   - estimated totals são mantidos em campos separados.
   - NUNCA misturar observado + estimado em total chamado 'observed'.
4. Health da Conexão vs Natureza Event-Sparse:
   - forceOrder é inerentemente event-sparse (poucos eventos por hora em regime calmo).
   - "silêncio por N segundos" NÃO é interpretado como unhealthy.
   - stream_healthy é determinado pela camada de conexão (socket/ping-pong):
     CONNECTED / DISCONNECTED / RECONNECTING / DEGRADED.
   - CONNECTED + 0 eventos => ZERO_OBSERVED (validity=VALID, notionals=0.0).
   - DISCONNECTED => UNKNOWN / MISSING (notionals=None).
5. Deduplicação Determinística & Risco Residual:
   - event_id estável gerado a partir de:
     symbol + trade_time_ms + order_side + orig_qty + filled_qty + price + status.
   - Risco residual documentado: caso duas liquidações distintas ocorram no mesmo
     milissegundo T com par, lado, quantidades, preços e status idênticos, a segunda
     é tratada como duplicata. A API pública da Binance não fornece orderId único.
6. Limites de Janela Contemporânea:
   - Intervalo semi-aberto estrito: [window_start_ms, window_end_ms).
   - Âncora causal: trade_time_ms (T). Event time (E) é usado apenas como fallback
     se T estiver ausente/corrompido.
7. Evidence v1 adapter:
   - Family: DERIVATIVES
   - Type: SLOW_CONTEXT (conforme capability snapshot)
   - counts_as_vote: False (NÃO vota)
   - direction: UNKNOWN (NÃO infere direção/sinal)
   - calibration: NOT_APPLICABLE
"""

import asyncio
import json
import logging
import math
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

try:
    from prometheus_client import (
        Counter as _PromCounter,
        Gauge as _PromGauge,
        REGISTRY as _PROM_REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    _PromCounter = None
    _PromGauge = None
    _PROM_REGISTRY = None
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)

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


class NotionalQuality(str, Enum):
    """Qualidade da valoração nocional do evento de liquidação."""
    OBSERVED_EXECUTION = "OBSERVED_EXECUTION"      # ap (average execution price) válido
    ESTIMATED_LIMIT_PRICE = "ESTIMATED_LIMIT_PRICE"  # ap ausente/zero, fallback p (limit price)
    NONE = "NONE"                                  # Preço/quantidade ausentes ou inválidos


class StreamConnectionStatus(str, Enum):
    """Estado de conexão e transporte da stream WebSocket/REST."""
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"


class ConnectionCoverageStatus(str, Enum):
    """Status de cobertura da conexão durante o intervalo da janela."""
    FULL = "FULL"        # Conectado continuamente durante toda a janela
    PARTIAL = "PARTIAL"  # Conexão esteve ativa em apenas parte da janela
    NONE = "NONE"        # Sem conexão (desconectado) durante toda a janela


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


@dataclass
class ConnectionStateInterval:
    """Intervalo contínuo de estado de conexão."""
    status: StreamConnectionStatus
    start_ms: int
    end_ms: Optional[int] = None


class ConnectionIntervalTracker:
    """
    Rastreia transições temporais de status de conexão WebSocket para derivar com
    exatidão a cobertura (FULL, PARTIAL, NONE) em qualquer janela [window_start_ms, window_end_ms).
    """

    def __init__(self, initial_status: StreamConnectionStatus = StreamConnectionStatus.DISCONNECTED):
        self._lock = threading.Lock()
        self._history: List[ConnectionStateInterval] = []
        self._current_status: StreamConnectionStatus = initial_status
        self._current_start_ms: int = int(time.time() * 1000)

    @property
    def current_status(self) -> StreamConnectionStatus:
        with self._lock:
            return self._current_status

    def record_status(
        self,
        status: Union[StreamConnectionStatus, str],
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """Registra transição de estado da conexão."""
        resolved: StreamConnectionStatus
        if isinstance(status, StreamConnectionStatus):
            resolved = status
        else:
            try:
                resolved = StreamConnectionStatus(str(status).upper())
            except ValueError:
                resolved = StreamConnectionStatus.DEGRADED

        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)

        with self._lock:
            if resolved == self._current_status:
                return
            closed_interval = ConnectionStateInterval(
                status=self._current_status,
                start_ms=self._current_start_ms,
                end_ms=ts,
            )
            self._history.append(closed_interval)
            self._current_status = resolved
            self._current_start_ms = ts

    def evaluate_coverage(self, window_start_ms: int, window_end_ms: int) -> ConnectionCoverageStatus:
        """
        Determina determinística e temporalmente a cobertura de CONNECTED na janela:
        - FULL: esteve CONNECTED ininterruptamente durante todo o intervalo [window_start_ms, window_end_ms).
        - NONE: não esteve CONNECTED em nenhum milissegundo da janela.
        - PARTIAL: esteve CONNECTED em parte da janela, ou sofreu desconexões/reconexões.
        """
        with self._lock:
            intervals = list(self._history)
            intervals.append(
                ConnectionStateInterval(
                    status=self._current_status,
                    start_ms=self._current_start_ms,
                    end_ms=None,
                )
            )

        connected_overlaps: List[Tuple[float, float]] = []
        for it in intervals:
            if it.status != StreamConnectionStatus.CONNECTED:
                continue
            s = float(it.start_ms)
            e = float(it.end_ms) if it.end_ms is not None else float("inf")
            overlap_s = max(float(window_start_ms), s)
            overlap_e = min(float(window_end_ms), e)
            if overlap_s < overlap_e:
                connected_overlaps.append((overlap_s, overlap_e))

        if not connected_overlaps:
            return ConnectionCoverageStatus.NONE

        for s, e in connected_overlaps:
            if s <= window_start_ms and e >= window_end_ms:
                return ConnectionCoverageStatus.FULL

        return ConnectionCoverageStatus.PARTIAL

    def prune_older_than(self, older_than_ms: int) -> int:
        """Poda intervalos históricos encerrados antes de older_than_ms."""
        with self._lock:
            initial = len(self._history)
            self._history = [
                it for it in self._history
                if (it.end_ms is None or it.end_ms >= older_than_ms)
            ]
            return initial - len(self._history)


class LiquidationMetrics:
    """
    Singleton thread-safe para métricas Prometheus de liquidações forçadas.
    Evita duplicate registration no REGISTRY.
    Se prometheus_client indisponível, opera em modo no-op.
    """
    _instance: Optional["LiquidationMetrics"] = None
    _lock = threading.Lock()

    def __init__(self, prefix: str = "liquidation"):
        self.enabled = HAS_PROMETHEUS
        self.prefix = prefix
        self.stream_connected = None
        self.events_received_total = None
        self.events_valid_total = None
        self.events_partial_total = None
        self.events_invalid_total = None
        self.events_deduplicated_total = None
        self.stream_reconnects_total = None

        if not self.enabled:
            return

        def _get_or_create_counter(name: str, doc: str, labels: Tuple[str, ...]) -> Any:
            if _PROM_REGISTRY is not None and name in _PROM_REGISTRY._names_to_collectors:
                return _PROM_REGISTRY._names_to_collectors[name]
            try:
                return _PromCounter(name, doc, labels)
            except ValueError:
                if _PROM_REGISTRY is not None and name in _PROM_REGISTRY._names_to_collectors:
                    return _PROM_REGISTRY._names_to_collectors[name]
                return None

        def _get_or_create_gauge(name: str, doc: str, labels: Tuple[str, ...]) -> Any:
            if _PROM_REGISTRY is not None and name in _PROM_REGISTRY._names_to_collectors:
                return _PROM_REGISTRY._names_to_collectors[name]
            try:
                return _PromGauge(name, doc, labels)
            except ValueError:
                if _PROM_REGISTRY is not None and name in _PROM_REGISTRY._names_to_collectors:
                    return _PROM_REGISTRY._names_to_collectors[name]
                return None

        self.stream_connected = _get_or_create_gauge(
            f"{prefix}_stream_connected",
            "1 se a stream WebSocket de liquidações está conectada, 0 caso contrário",
            ("symbol",),
        )
        self.events_received_total = _get_or_create_counter(
            f"{prefix}_events_received_total",
            "Total de eventos de liquidação recebidos no WebSocket",
            ("symbol",),
        )
        self.events_valid_total = _get_or_create_counter(
            f"{prefix}_events_valid_total",
            "Total de eventos de liquidação com validade VALID",
            ("symbol",),
        )
        self.events_partial_total = _get_or_create_counter(
            f"{prefix}_events_partial_total",
            "Total de eventos de liquidação com validade PARTIAL",
            ("symbol",),
        )
        self.events_invalid_total = _get_or_create_counter(
            f"{prefix}_events_invalid_total",
            "Total de eventos de liquidação com validade INVALID",
            ("symbol",),
        )
        self.events_deduplicated_total = _get_or_create_counter(
            f"{prefix}_events_deduplicated_total",
            "Total de eventos descartados por deduplicação de ID",
            ("symbol",),
        )
        self.stream_reconnects_total = _get_or_create_counter(
            f"{prefix}_stream_reconnects_total",
            "Total de reconexões ocorridas no stream de liquidações",
            ("symbol",),
        )

    @classmethod
    def get_instance(cls, prefix: str = "liquidation") -> "LiquidationMetrics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(prefix=prefix)
            return cls._instance

    def set_connected(self, is_connected: bool, symbol: str = "BTCUSDT") -> None:
        if self.stream_connected is not None:
            try:
                self.stream_connected.labels(symbol=symbol.upper()).set(1.0 if is_connected else 0.0)
            except Exception:
                pass

    def record_received(self, symbol: str = "BTCUSDT") -> None:
        if self.events_received_total is not None:
            try:
                self.events_received_total.labels(symbol=symbol.upper()).inc()
            except Exception:
                pass

    def record_validity(self, validity: Union[LiquidationValidity, str], symbol: str = "BTCUSDT") -> None:
        v_str = validity.value if isinstance(validity, LiquidationValidity) else str(validity).upper()
        target = None
        if v_str == "VALID":
            target = self.events_valid_total
        elif v_str == "PARTIAL":
            target = self.events_partial_total
        elif v_str == "INVALID":
            target = self.events_invalid_total

        if target is not None:
            try:
                target.labels(symbol=symbol.upper()).inc()
            except Exception:
                pass

    def record_deduplicated(self, symbol: str = "BTCUSDT") -> None:
        if self.events_deduplicated_total is not None:
            try:
                self.events_deduplicated_total.labels(symbol=symbol.upper()).inc()
            except Exception:
                pass

    def record_reconnect(self, symbol: str = "BTCUSDT") -> None:
        if self.stream_reconnects_total is not None:
            try:
                self.stream_reconnects_total.labels(symbol=symbol.upper()).inc()
            except Exception:
                pass



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
    average_price: Optional[float]      # Preço médio real de execução (ap) se observado
    limit_price: Optional[float]        # Preço limite da ordem (p)
    observed_notional_usd: Optional[float]   # filled_qty * average_price (somente se ap observado)
    estimated_notional_usd: Optional[float]  # filled_qty * limit_price (quando ap ausente)
    notional_quality: str = NotionalQuality.NONE.value
    order_status: str = "UNKNOWN"
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
            "limit_price": self.limit_price,
            "observed_notional_usd": self.observed_notional_usd,
            "estimated_notional_usd": self.estimated_notional_usd,
            "notional_quality": self.notional_quality,
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
            limit_price=None,
            observed_notional_usd=None,
            estimated_notional_usd=None,
            notional_quality=NotionalQuality.NONE.value,
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

    # 2. Timestamps: T (trade_time) é a âncora causal canônica; E é o envelope de evento
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
        # Em alguns snapshots iniciais, z pode ser 0 enquanto q está preenchido
        filled_qty = orig_qty
        qty_reasons.append("ORIG_QTY_FALLBACK")

    # 5. P2-D1.1 Hardening: Distinção estrita entre Preço Médio Observado e Preço Limite Estimado
    observed_average_price: Optional[float] = None
    observed_notional_usd: Optional[float] = None
    estimated_notional_usd: Optional[float] = None
    notional_quality = NotionalQuality.NONE.value
    price_reasons: List[str] = []

    if avg_price is not None and avg_price > 0.0:
        # Preço médio de execução realmente observado pela exchange
        observed_average_price = avg_price
        if filled_qty is not None and filled_qty > 0:
            observed_notional_usd = round(filled_qty * avg_price, 2)
            notional_quality = NotionalQuality.OBSERVED_EXECUTION.value
    elif limit_price is not None and limit_price > 0.0:
        # Fallback de preço limite: NÃO vira observed_notional_usd!
        price_reasons.append("LIMIT_PRICE_FALLBACK")
        if filled_qty is not None and filled_qty > 0:
            estimated_notional_usd = round(filled_qty * limit_price, 2)
            notional_quality = NotionalQuality.ESTIMATED_LIMIT_PRICE.value

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
    elif filled_qty is not None and filled_qty <= 0:
        validity = LiquidationValidity.INVALID
        reasons.append("NON_POSITIVE_QTY")
    elif notional_quality == NotionalQuality.ESTIMATED_LIMIT_PRICE.value:
        # P2-D1.1: Fallback para limit price torna o evento PARTIAL
        validity = LiquidationValidity.PARTIAL
    elif observed_notional_usd is None:
        validity = LiquidationValidity.PARTIAL
        reasons.append("NOTIONAL_CANNOT_BE_CALCULATED")

    # 7. Gerar event_id determinístico para deduplicação
    # Composição: symbol_tradeTime_orderSide_origQty_filledQty_price_status
    # Risco residual de colisão documentado: No evento teórico de duas liquidações
    # distintas no exato mesmo ms T com atributos idênticos, a segunda é deduplicada.
    orig_str = f"{orig_qty:.8f}" if orig_qty is not None else "none"
    fq_str = f"{filled_qty:.8f}" if filled_qty is not None else "none"
    ref_price = observed_average_price or limit_price
    pr_str = f"{ref_price:.2f}" if ref_price is not None else "none"
    event_id = f"{symbol}_{trade_time_ms}_{order_side}_{orig_str}_{fq_str}_{pr_str}_{order_status}"

    return ForcedLiquidationEvent(
        symbol=symbol,
        event_time_ms=event_time_ms,
        trade_time_ms=trade_time_ms,
        order_side=order_side,
        liquidated_position_side=liquidated_position_side,
        original_qty=orig_qty,
        filled_qty=filled_qty,
        average_price=observed_average_price,
        limit_price=limit_price,
        observed_notional_usd=observed_notional_usd,
        estimated_notional_usd=estimated_notional_usd,
        notional_quality=notional_quality,
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
    # Totais observados: somam SOMENTE observed_notional_usd (OBSERVED_EXECUTION)
    long_liquidated_notional_usd: Optional[float]
    short_liquidated_notional_usd: Optional[float]
    total_liquidated_notional_usd: Optional[float]
    # Totais estimados: somam SOMENTE estimated_notional_usd (ESTIMATED_LIMIT_PRICE)
    long_estimated_notional_usd: Optional[float] = None
    short_estimated_notional_usd: Optional[float] = None
    total_estimated_notional_usd: Optional[float] = None
    latest_event_ms: Optional[int] = None
    source: str = "binance_usdm"
    coverage: str = STREAM_CAPABILITY
    validity: LiquidationValidity = LiquidationValidity.VALID
    reason: Optional[str] = None
    larger_observed_side: str = LargerObservedSide.NONE.value
    stream_healthy: bool = True
    connection_status: str = StreamConnectionStatus.CONNECTED.value
    connection_coverage_status: str = ConnectionCoverageStatus.FULL.value
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
            "long_estimated_notional_usd": self.long_estimated_notional_usd,
            "short_estimated_notional_usd": self.short_estimated_notional_usd,
            "total_estimated_notional_usd": self.total_estimated_notional_usd,
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
            "connection_status": self.connection_status,
            "connection_coverage_status": self.connection_coverage_status,
        }


class LiquidationWindowAggregator:
    """
    Agregador puro contemporâneo de liquidações com deduplicação determinística.
    Opera estritamente por janela temporal semi-aberta [window_start_ms, window_end_ms).
    """

    def __init__(self, symbol: str = "BTCUSDT", max_seen_events: int = 10_000):
        self.symbol = symbol.upper().strip()
        self.max_seen_events = max_seen_events
        self._events: List[ForcedLiquidationEvent] = []
        self._seen_event_ids: OrderedDict[str, int] = OrderedDict()

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
            # Poda FIFO bounded para manter teto estrito O(max_seen_events) de memória
            self._seen_event_ids.popitem(last=False)

        self._seen_event_ids[event.event_id] = event.trade_time_ms or event.event_time_ms
        self._events.append(event)
        return True

    def prune_events_older_than(self, older_than_ms: int) -> int:
        """
        Poda eventos da lista em memória cujo timestamp for estritamente anterior a older_than_ms.
        Retorna a quantidade de eventos descartados.
        Preserva os IDs em _seen_event_ids para manter a barreira de deduplicação causal pós-reconexão.
        """
        initial_count = len(self._events)
        self._events = [
            e for e in self._events
            if (e.trade_time_ms or e.event_time_ms) >= older_than_ms
        ]
        return initial_count - len(self._events)

    def summarize_window(
        self,
        window_start_ms: int,
        window_end_ms: int,
        stream_healthy: bool = True,
        connection_status: Optional[Union[StreamConnectionStatus, str]] = None,
        connection_coverage_status: Optional[Union[ConnectionCoverageStatus, str]] = None,
    ) -> LiquidationWindowSummary:
        """
        Agrega eventos observados dentro do intervalo semi-aberto [window_start_ms, window_end_ms).

        Regra estrita P2-D1.1 e P2-D2:
        - trade_time == window_start_ms => ENTRA na janela atual.
        - trade_time == window_end_ms   => EXCLUÍDO (pertence à próxima janela).
        - Âncora causal: trade_time_ms (T). Event time (E) é usado apenas como fallback.
        - FULL coverage + 0 eventos: VALID, event_count=0, notionals 0.0 (ZERO_OBSERVED).
        - PARTIAL coverage + 0 eventos: PARTIAL, event_count=0, notionals=None (evita falso zero).
        - NONE coverage: UNKNOWN, notionals=None.
        - Totais observados somam SOMENTE observed_notional_usd.
        - Totais estimados somam SOMENTE estimated_notional_usd.
        """
        # Resolver status de conexão
        conn_str: str
        is_healthy: bool
        if connection_status is not None:
            conn_str = (
                connection_status.value
                if isinstance(connection_status, StreamConnectionStatus)
                else str(connection_status).upper()
            )
            is_healthy = conn_str == StreamConnectionStatus.CONNECTED.value
        else:
            is_healthy = bool(stream_healthy)
            conn_str = (
                StreamConnectionStatus.CONNECTED.value
                if is_healthy
                else StreamConnectionStatus.DISCONNECTED.value
            )

        # Resolver status de cobertura
        cov_str: str
        if connection_coverage_status is not None:
            cov_str = (
                connection_coverage_status.value
                if isinstance(connection_coverage_status, ConnectionCoverageStatus)
                else str(connection_coverage_status).upper()
            )
        else:
            cov_str = (
                ConnectionCoverageStatus.FULL.value
                if is_healthy
                else ConnectionCoverageStatus.NONE.value
            )

        # 1. Sem cobertura ou conexão inativa durante toda a janela
        if cov_str == ConnectionCoverageStatus.NONE.value or (not is_healthy and cov_str != ConnectionCoverageStatus.PARTIAL.value):
            return LiquidationWindowSummary(
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
                event_count=None,
                long_liquidated_qty=None,
                short_liquidated_qty=None,
                long_liquidated_notional_usd=None,
                short_liquidated_notional_usd=None,
                total_liquidated_notional_usd=None,
                long_estimated_notional_usd=None,
                short_estimated_notional_usd=None,
                total_estimated_notional_usd=None,
                latest_event_ms=None,
                source="binance_usdm",
                coverage=STREAM_CAPABILITY,
                validity=LiquidationValidity.UNKNOWN,
                reason=f"STREAM_{conn_str}" if not is_healthy else "NO_CONNECTION_COVERAGE",
                larger_observed_side=LargerObservedSide.NONE.value,
                stream_healthy=is_healthy,
                connection_status=conn_str,
                connection_coverage_status=cov_str,
            )

        # 2. Filtrar eventos estritamente pelo trade_time_ms em [window_start_ms, window_end_ms)
        window_events = [
            e for e in self._events
            if e.symbol == self.symbol
            and window_start_ms <= (e.trade_time_ms or e.event_time_ms) < window_end_ms
        ]

        # 3. Tratar caso de zero eventos observados
        if not window_events:
            if cov_str == ConnectionCoverageStatus.PARTIAL.value:
                # Conexão caiu durante parte da janela: NÃO declarar ZERO_OBSERVED completo
                return LiquidationWindowSummary(
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    event_count=0,
                    long_liquidated_qty=None,
                    short_liquidated_qty=None,
                    long_liquidated_notional_usd=None,
                    short_liquidated_notional_usd=None,
                    total_liquidated_notional_usd=None,
                    long_estimated_notional_usd=None,
                    short_estimated_notional_usd=None,
                    total_estimated_notional_usd=None,
                    latest_event_ms=None,
                    source="binance_usdm",
                    coverage=STREAM_CAPABILITY,
                    validity=LiquidationValidity.PARTIAL,
                    reason="PARTIAL_CONNECTION_COVERAGE_ZERO_OBSERVED",
                    larger_observed_side=LargerObservedSide.NONE.value,
                    stream_healthy=is_healthy,
                    connection_status=conn_str,
                    connection_coverage_status=cov_str,
                )
            else:
                # FULL coverage + zero eventos => ZERO_OBSERVED válido
                return LiquidationWindowSummary(
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    event_count=0,
                    long_liquidated_qty=0.0,
                    short_liquidated_qty=0.0,
                    long_liquidated_notional_usd=0.0,
                    short_liquidated_notional_usd=0.0,
                    total_liquidated_notional_usd=0.0,
                    long_estimated_notional_usd=0.0,
                    short_estimated_notional_usd=0.0,
                    total_estimated_notional_usd=0.0,
                    latest_event_ms=None,
                    source="binance_usdm",
                    coverage=STREAM_CAPABILITY,
                    validity=LiquidationValidity.VALID,
                    reason="ZERO_OBSERVED_HEALTHY_STREAM",
                    larger_observed_side=LargerObservedSide.NONE.value,
                    stream_healthy=True,
                    connection_status=conn_str,
                    connection_coverage_status=cov_str,
                )

        long_qty = 0.0
        short_qty = 0.0
        # Totais puramente observados
        long_observed_notional = 0.0
        short_observed_notional = 0.0
        # Totais puramente estimados
        long_estimated_notional = 0.0
        short_estimated_notional = 0.0

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

            is_long = e.liquidated_position_side == LiquidatedPositionSide.LONG.value
            is_short = e.liquidated_position_side == LiquidatedPositionSide.SHORT.value

            if e.filled_qty is not None:
                if is_long:
                    long_qty += e.filled_qty
                elif is_short:
                    short_qty += e.filled_qty

            # Somar observed notional SOMENTE onde for OBSERVED_EXECUTION
            if e.observed_notional_usd is not None:
                if is_long:
                    long_observed_notional += e.observed_notional_usd
                elif is_short:
                    short_observed_notional += e.observed_notional_usd

            # Somar estimated notional SOMENTE onde for ESTIMATED_LIMIT_PRICE
            if e.estimated_notional_usd is not None:
                if is_long:
                    long_estimated_notional += e.estimated_notional_usd
                elif is_short:
                    short_estimated_notional += e.estimated_notional_usd

        total_observed_notional = long_observed_notional + short_observed_notional
        total_estimated_notional = long_estimated_notional + short_estimated_notional

        # Comparação matemática descritiva sobre os volumes observados (se existirem) ou estimados
        larger_side = LargerObservedSide.NONE.value
        comp_long = long_observed_notional if total_observed_notional > 0 else long_estimated_notional
        comp_short = short_observed_notional if total_observed_notional > 0 else short_estimated_notional
        if comp_long > comp_short:
            larger_side = LargerObservedSide.LONG.value
        elif comp_short > comp_long:
            larger_side = LargerObservedSide.SHORT.value
        elif (comp_long + comp_short) > 0:
            larger_side = LargerObservedSide.EQUAL.value

        validity = LiquidationValidity.VALID
        reasons = []
        if has_invalid:
            reasons.append("WINDOW_CONTAINS_INVALID_EVENTS")
        if has_partial:
            validity = LiquidationValidity.PARTIAL
            reasons.append("WINDOW_CONTAINS_PARTIAL_EVENTS")
        if cov_str == ConnectionCoverageStatus.PARTIAL.value:
            validity = LiquidationValidity.PARTIAL
            reasons.append("PARTIAL_CONNECTION_COVERAGE")

        return LiquidationWindowSummary(
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            event_count=valid_events_count,
            long_liquidated_qty=round(long_qty, 4),
            short_liquidated_qty=round(short_qty, 4),
            long_liquidated_notional_usd=round(long_observed_notional, 2),
            short_liquidated_notional_usd=round(short_observed_notional, 2),
            total_liquidated_notional_usd=round(total_observed_notional, 2),
            long_estimated_notional_usd=round(long_estimated_notional, 2),
            short_estimated_notional_usd=round(short_estimated_notional, 2),
            total_estimated_notional_usd=round(total_estimated_notional, 2),
            latest_event_ms=latest_ts if latest_ts > 0 else None,
            source="binance_usdm",
            coverage=STREAM_CAPABILITY,
            validity=validity,
            reason=";".join(reasons) if reasons else None,
            larger_observed_side=larger_side,
            stream_healthy=is_healthy,
            connection_status=conn_str,
            connection_coverage_status=cov_str,
        )


def liquidation_summary_to_evidence(
    summary: LiquidationWindowSummary,
    symbol: str = "BTCUSDT",
) -> List[Evidence]:
    """
    Adapter que converte um LiquidationWindowSummary nas 4 instâncias canônicas
    de Evidence v1 registradas na taxonomia institucional.

    Contrato estrito P2-D1.1:
    - derivatives.liquidations.*_observed_notional reflete EXCLUSIVAMENTE
      os valores observados de execução (long_liquidated_notional_usd, etc.),
      nunca misturando com estimativas de preço limite.
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
        "connection_status": summary.connection_status,
        "connection_coverage_status": summary.connection_coverage_status,
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
            {
                "liquidated_position_side": "LONG",
                "estimated_notional_usd": summary.long_estimated_notional_usd,
            },
        ),
        (
            "derivatives.liquidations.short_observed_notional",
            summary.short_liquidated_notional_usd,
            "USD",
            {
                "liquidated_position_side": "SHORT",
                "estimated_notional_usd": summary.short_estimated_notional_usd,
            },
        ),
        (
            "derivatives.liquidations.total_observed_notional",
            summary.total_liquidated_notional_usd,
            "USD",
            {
                "is_composite": True,
                "estimated_notional_usd": summary.total_estimated_notional_usd,
            },
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
                "connection_coverage_status": summary.connection_coverage_status,
                **extra_meta,
            },
        )
        evidences.append(ev)

    return evidences


# =====================================================================
# P2-D2: ROUTING GUARDS
# =====================================================================

def is_force_order_message(data: Any) -> bool:
    """
    Identifica com precisão se a mensagem é um payload de liquidação forçada Binance Futures.
    Retorna False para aggTrade, depthUpdate, klines ou outros eventos.
    """
    if not isinstance(data, dict):
        return False
    inner = data.get("data", data) if isinstance(data.get("data"), dict) else data
    if not isinstance(inner, dict):
        return False
    e = inner.get("e")
    if e == "forceOrder":
        return True
    if e in ("aggTrade", "trade", "depthUpdate", "kline", "bookTicker"):
        return False
    # Checa estrutura de ordem de liquidação se 'e' for omitido (ex: REST format ou wrapper alternativo)
    order = inner.get("o")
    if isinstance(order, dict) and "s" in order and "S" in order and ("ap" in order or "p" in order) and "q" in order:
        return True
    return False


def is_agg_trade_message(data: Any) -> bool:
    """
    Identifica se a mensagem é um trade ou aggTrade da Binance.
    """
    if not isinstance(data, dict):
        return False
    inner = data.get("data", data) if isinstance(data.get("data"), dict) else data
    if not isinstance(inner, dict):
        return False
    e = inner.get("e")
    if e == "aggTrade" or "a" in inner:
        return True
    if e == "trade" or ("t" in inner and "p" in inner and "q" in inner):
        return True
    return False


# =====================================================================
# P2-D2: WEBSOCKET DEDICATED LISTENER
# =====================================================================

class BinanceLiquidationListener:
    """
    Listener WebSocket dedicado e assíncrono para telemetria de liquidações Binance USD-M.

    Garante:
    - Conexão separada dedicada (Opção B): isolamento total do failure-domain de aggTrade.
    - Zero chamadas pesadas no callback (apenas parse, add_event em memória e métricas).
    - Roteamento cruzado estrito: aggTrade é descartado sem processamento.
    - Rastreamento fino de cobertura temporal via ConnectionIntervalTracker.
    - Reconexão resiliente com backoff exponencial e jitter.
    - Heartbeat e keepalive com timeout.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        stream_url: Optional[str] = None,
        aggregator: Optional[LiquidationWindowAggregator] = None,
        tracker: Optional[ConnectionIntervalTracker] = None,
        metrics: Optional[LiquidationMetrics] = None,
        max_reconnect_attempts: int = 50,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        on_event_callback: Optional[Callable[[ForcedLiquidationEvent], None]] = None,
    ):
        self.symbol = symbol.upper().strip()
        self.stream_url = stream_url or f"wss://fstream.binance.com/ws/{self.symbol.lower()}@forceOrder"
        self.aggregator = aggregator or LiquidationWindowAggregator(symbol=self.symbol)
        self.tracker = tracker or ConnectionIntervalTracker()
        self.metrics = metrics or LiquidationMetrics.get_instance()
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.on_event_callback = on_event_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._session: Optional[Any] = None
        self._ws: Optional[Any] = None
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self.tracker.current_status == StreamConnectionStatus.CONNECTED

    def handle_raw_message(self, raw_message: Union[str, bytes, Dict[str, Any]]) -> Optional[ForcedLiquidationEvent]:
        """
        Processa mensagem bruta da rede no hot-path de liquidação.
        Leve, não-bloqueante e com guards de roteamento cruzado.
        """
        data: Dict[str, Any]
        if isinstance(raw_message, (str, bytes)):
            try:
                data = json.loads(raw_message)
            except Exception:
                return None
        elif isinstance(raw_message, dict):
            data = raw_message
        else:
            return None

        # Cross-routing guard: nunca passar aggTrade pelo parser de liquidação
        if is_agg_trade_message(data):
            return None

        # Validação de payload forceOrder
        if not is_force_order_message(data):
            return None

        self.metrics.record_received(symbol=self.symbol)
        event = parse_force_order_payload(data)
        self.metrics.record_validity(event.validity, symbol=self.symbol)

        is_new = self.aggregator.add_event(event)
        if not is_new:
            self.metrics.record_deduplicated(symbol=self.symbol)

        if self.on_event_callback is not None:
            try:
                self.on_event_callback(event)
            except Exception as e:
                logger.warning(f"Erro no on_event_callback de liquidação: {e}")

        return event

    async def run(self) -> None:
        """Loop de conexão contínua com auto-reconnect e backoff exponencial."""
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp é obrigatório para execução do BinanceLiquidationListener.")

        self._running = True
        self.tracker.record_status(StreamConnectionStatus.DISCONNECTED)
        self.metrics.set_connected(False, symbol=self.symbol)

        while self._running:
            try:
                self.tracker.record_status(StreamConnectionStatus.RECONNECTING)
                timeout = aiohttp.ClientTimeout(total=self.ping_timeout + 20)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    self._session = session
                    async with session.ws_connect(
                        self.stream_url,
                        heartbeat=self.ping_interval,
                        autoping=True,
                    ) as ws:
                        self._ws = ws
                        self._reconnect_count = 0
                        self.tracker.record_status(StreamConnectionStatus.CONNECTED)
                        self.metrics.set_connected(True, symbol=self.symbol)
                        logger.info(f"Conexão ativa ao stream de liquidações ({self.stream_url})")

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self.handle_raw_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Exceção no stream WebSocket de liquidações: {exc}")
            finally:
                self.tracker.record_status(StreamConnectionStatus.DISCONNECTED)
                self.metrics.set_connected(False, symbol=self.symbol)

            if not self._running:
                break

            self._reconnect_count += 1
            if self._reconnect_count > self.max_reconnect_attempts:
                logger.error("Máximo de tentativas de reconexão atingido no stream de liquidações.")
                break

            self.metrics.record_reconnect(symbol=self.symbol)
            delay = min(self.initial_delay * (2 ** (self._reconnect_count - 1)), self.max_delay)
            jitter = delay * random.uniform(-0.15, 0.15)
            wait_time = max(0.2, delay + jitter)
            logger.info(f"Reconectando stream de liquidações em {wait_time:.2f}s...")
            try:
                await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                break

        self.tracker.record_status(StreamConnectionStatus.DISCONNECTED)
        self.metrics.set_connected(False, symbol=self.symbol)

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.Task:
        """Inicia o listener em background asyncio task."""
        self._running = True
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        self._task = loop.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        """Encerra graciosamente a conexão e o loop assíncrono."""
        self._running = False
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._session is not None and not self._session.closed:
            await self._session.close()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.tracker.record_status(StreamConnectionStatus.DISCONNECTED)
        self.metrics.set_connected(False, symbol=self.symbol)


def get_liquidation_window_context(summary: LiquidationWindowSummary) -> Dict[str, Any]:
    """
    Retorna dicionário aditivo com telemetria da janela para contexto observacional interno.
    NÃO altera payload de IA/LLM, NEM confluência, NEM sinais, NEM whale score.
    """
    return {
        "liquidation_summary": summary.to_dict(),
        "is_telemetry_only": True,
        "counts_as_vote": False,
        "direction": "UNKNOWN",
    }

