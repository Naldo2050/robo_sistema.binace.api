# paper_trading/execution_sink.py
"""
Hermetic Execution Sink for Paper Trading.

Single serialized boundary between normalized market trades, risk-approved PaperOrders,
and the PaperExecutor. Guarantees temporal causality, total local ordering via ingest_seq,
bounded O(1) deduplication, and failure isolation under a single threading.RLock.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple

from paper_trading.contracts import PaperOrder, Rejection
from paper_trading.executor import PaperExecutor, TickEvents


class TradeStatus(str, Enum):
    """Outcome status for on_market_trade."""
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    INVALID_TRADE = "INVALID_TRADE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    DISABLED = "DISABLED"
    EXECUTOR_ERROR = "EXECUTOR_ERROR"


class SubmitStatus(str, Enum):
    """Outcome status for submit_order."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    DISABLED = "DISABLED"
    EXECUTOR_ERROR = "EXECUTOR_ERROR"


@dataclass(frozen=True)
class ExecutionTick:
    """
    Immutable representation of an ingested market trade tick ready for paper execution.
    """
    symbol: str
    event_timestamp: int
    raw_timestamp: int
    trade_id: int | str | None
    price: float
    quantity: float
    is_buyer_maker: bool
    ingest_seq: int
    source: str
    received_at_ms: int | None = None
    ooo_clamped: bool = False
    continuity_suspect: bool = False


@dataclass(frozen=True)
class TradeResult:
    """Result of processing a normalized market trade through the sink."""
    status: TradeStatus
    tick: Optional[ExecutionTick] = None
    events: Optional[TickEvents] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class SubmitResult:
    """Result of submitting a PaperOrder through the sink."""
    status: SubmitStatus
    order: Optional[PaperOrder] = None
    rejection: Optional[Rejection] = None
    error_message: Optional[str] = None


class ExecutionSink:
    """
    Thread-safe, hermetic execution sink serializing order submissions and market ticks.

    Invariants enforced:
    - Single RLock boundary across submit_order, on_market_trade, and executor calls.
    - Strictly monotonic ingest_seq incremented only for accepted, non-duplicate ticks.
    - Temporal causality: ticks observed before order submission never fill that order.
    - Bounded O(1) deduplication via deque + set.
    - Out-of-order fail-closed: event_timestamp cannot regress below last accepted timestamp.
    - Zero I/O hot path: no disk, no DB, no logging per tick, no network access.
    - Exception isolation: circuit breaker disables sink after consecutive executor errors.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        executor: Optional[PaperExecutor] = None,
        dedup_capacity: int = 5000,
        consecutive_error_threshold: int = 3,
    ) -> None:
        if not (isinstance(symbol, str) and symbol.strip()):
            raise ValueError("symbol must be a non-empty string")
        if isinstance(dedup_capacity, bool) or not isinstance(dedup_capacity, int) or dedup_capacity <= 0:
            raise ValueError("dedup_capacity must be a positive integer")
        if (
            isinstance(consecutive_error_threshold, bool)
            or not isinstance(consecutive_error_threshold, int)
            or consecutive_error_threshold <= 0
        ):
            raise ValueError("consecutive_error_threshold must be a positive integer")

        self.symbol: str = symbol.strip().upper()
        self.executor: PaperExecutor = executor or PaperExecutor()
        self.dedup_capacity: int = dedup_capacity
        self.consecutive_error_threshold: int = consecutive_error_threshold

        # Single serialization boundary lock
        self._lock: threading.RLock = threading.RLock()

        # Ingestion sequence and order tracking
        self._ingest_seq: int = 0
        self._registered_ingest_seq: Dict[str, int] = {}

        # Deduplication cache: deque of keys + set for O(1) membership
        self._dedup_deque: deque[Tuple[str, int | str]] = deque(maxlen=dedup_capacity)
        self._dedup_set: Set[Tuple[str, int | str]] = set()

        # Timestamp and continuity tracking per symbol
        self._last_accepted_event_ts: Optional[int] = None
        self._previous_trade_id: Optional[int] = None

        # Circuit breaker and error metrics
        self._disabled: bool = False
        self._consecutive_errors: int = 0
        self._total_executor_errors: int = 0

        # Telemetry counters
        self._accepted_ticks: int = 0
        self._duplicate_ticks: int = 0
        self._invalid_ticks: int = 0
        self._out_of_order_ticks: int = 0
        self._id_regressions: int = 0
        self._continuity_suspect_ticks: int = 0
        self._accepted_orders: int = 0
        self._rejected_orders: int = 0

    @property
    def is_disabled(self) -> bool:
        """Indicates whether the circuit breaker is currently active."""
        with self._lock:
            return self._disabled

    def reset_circuit_breaker(self) -> None:
        """Administrative reset to re-enable sink after circuit breaker trip."""
        with self._lock:
            self._disabled = False
            self._consecutive_errors = 0

    def submit_order(self, order: PaperOrder) -> SubmitResult:
        """
        Submit a pre-built, risk-approved PaperOrder into execution.

        Acquires the single RLock, calls executor.submit_order(order), and if accepted,
        records registered_ingest_seq[order_id] = current_ingest_seq.
        """
        with self._lock:
            if self._disabled:
                return SubmitResult(
                    status=SubmitStatus.DISABLED,
                    order=None,
                    rejection=None,
                    error_message="ExecutionSink is disabled by circuit breaker",
                )

            try:
                reg_order, rejection = self.executor.submit_order(order)
                self._consecutive_errors = 0
            except Exception as exc:
                self._total_executor_errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.consecutive_error_threshold:
                    self._disabled = True
                return SubmitResult(
                    status=SubmitStatus.EXECUTOR_ERROR,
                    order=None,
                    rejection=None,
                    error_message=f"Executor submit error: {exc}",
                )

            if rejection is not None:
                self._rejected_orders += 1
                return SubmitResult(
                    status=SubmitStatus.REJECTED,
                    order=None,
                    rejection=rejection,
                )

            assert reg_order is not None
            self._accepted_orders += 1
            # Record registration sequence for causality auditing
            self._registered_ingest_seq[reg_order.order_id] = self._ingest_seq

            return SubmitResult(
                status=SubmitStatus.ACCEPTED,
                order=reg_order,
                rejection=None,
            )

    def on_market_trade(self, norm: Dict[str, Any]) -> TradeResult:
        """
        Process a single normalized market trade dictionary.

        Validates payload, enforces monotonic timestamp and deduplication, increments
        ingest_seq, delivers tick to executor, and isolates all execution failures.
        """
        with self._lock:
            if self._disabled:
                return TradeResult(
                    status=TradeStatus.DISABLED,
                    tick=None,
                    events=None,
                    error_message="ExecutionSink is disabled by circuit breaker",
                )

            # 1. Validate normalized trade input defensivamente without mutating norm
            tick_params = self._validate_and_extract_norm(norm)
            if tick_params is None:
                self._invalid_ticks += 1
                return TradeResult(
                    status=TradeStatus.INVALID_TRADE,
                    tick=None,
                    events=None,
                    error_message="Malformed or invalid normalized trade payload",
                )

            p, q, event_ts, raw_ts, m, source, trade_id, received_at = tick_params

            # 2. Check Deduplication by (symbol, trade_id) when trade_id is available
            if trade_id is not None:
                dedup_key = (self.symbol, trade_id)
                if dedup_key in self._dedup_set:
                    self._duplicate_ticks += 1
                    return TradeResult(
                        status=TradeStatus.DUPLICATE,
                        tick=None,
                        events=None,
                        error_message=f"Duplicate trade_id detected for {dedup_key}",
                    )

            # 3. Check Monotonicity of event_timestamp (OOO check)
            if self._last_accepted_event_ts is not None and event_ts < self._last_accepted_event_ts:
                self._out_of_order_ticks += 1
                return TradeResult(
                    status=TradeStatus.OUT_OF_ORDER,
                    tick=None,
                    events=None,
                    error_message=f"Event timestamp {event_ts} regressed below last accepted {self._last_accepted_event_ts}",
                )

            # 4. Check Continuity / ID Regression for integer trade_ids
            continuity_suspect = False
            if isinstance(trade_id, int) and not isinstance(trade_id, bool):
                if self._previous_trade_id is not None:
                    if trade_id > self._previous_trade_id + 1:
                        continuity_suspect = True
                        self._continuity_suspect_ticks += 1
                    elif trade_id < self._previous_trade_id:
                        self._id_regressions += 1
                self._previous_trade_id = trade_id

            # 5. Passed all filters -> Increment monotonic ingest_seq
            self._ingest_seq += 1
            current_seq = self._ingest_seq

            # 6. Record Deduplication key in bounded cache
            if trade_id is not None:
                dedup_key = (self.symbol, trade_id)
                if len(self._dedup_deque) >= self.dedup_capacity:
                    evicted_key = self._dedup_deque.popleft()
                    self._dedup_set.discard(evicted_key)
                self._dedup_deque.append(dedup_key)
                self._dedup_set.add(dedup_key)

            # 7. Update accepted event timestamp watermark
            self._last_accepted_event_ts = event_ts

            # 8. Construct ExecutionTick
            ooo_clamped = (raw_ts != event_ts)
            tick = ExecutionTick(
                symbol=self.symbol,
                event_timestamp=event_ts,
                raw_timestamp=raw_ts,
                trade_id=trade_id,
                price=p,
                quantity=q,
                is_buyer_maker=m,
                ingest_seq=current_seq,
                source=source,
                received_at_ms=received_at,
                ooo_clamped=ooo_clamped,
                continuity_suspect=continuity_suspect,
            )

            # 9. Deliver to PaperExecutor with exception isolation
            try:
                events = self.executor.on_tick(
                    T=tick.event_timestamp,
                    p=tick.price,
                    q=tick.quantity,
                    m=tick.is_buyer_maker,
                    trade_id=tick.trade_id if tick.trade_id is not None else 0,
                )
                self._consecutive_errors = 0
            except Exception as exc:
                self._total_executor_errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.consecutive_error_threshold:
                    self._disabled = True
                return TradeResult(
                    status=TradeStatus.EXECUTOR_ERROR,
                    tick=tick,
                    events=None,
                    error_message=f"Executor on_tick error: {exc}",
                )

            self._accepted_ticks += 1

            # 10. Clean up registered_ingest_seq for orders that exited pending state
            if self._registered_ingest_seq:
                pending_order_ids = self.executor.pending_orders
                exited_order_ids = [
                    oid for oid in self._registered_ingest_seq if oid not in pending_order_ids
                ]
                for oid in exited_order_ids:
                    del self._registered_ingest_seq[oid]

            return TradeResult(
                status=TradeStatus.ACCEPTED,
                tick=tick,
                events=events,
            )

    def _validate_and_extract_norm(
        self, norm: Any
    ) -> Optional[Tuple[float, float, int, int, bool, str, int | str | None, Optional[int]]]:
        """Validates normalized dict input defensivamente without modifying original."""
        if not isinstance(norm, dict):
            return None

        # Price 'p'
        p = norm.get("p")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            return None
        p_float = float(p)

        # Quantity 'q'
        q = norm.get("q")
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or q <= 0:
            return None
        q_float = float(q)

        # Event Timestamp 'T'
        T = norm.get("T")
        if isinstance(T, bool) or not isinstance(T, int) or T <= 0:
            return None

        # Raw Timestamp 'T_raw'
        T_raw = norm.get("T_raw", T)
        if isinstance(T_raw, bool) or not isinstance(T_raw, int) or T_raw <= 0:
            return None

        # Buyer maker 'm' - strictly real bool
        m = norm.get("m")
        if not isinstance(m, bool):
            return None

        # Source
        source = norm.get("source")
        if not (isinstance(source, str) and source.strip()):
            return None
        source_clean = source.strip()

        # Trade ID: int, non-empty str, or None
        trade_id = norm.get("trade_id")
        clean_trade_id: int | str | None = None
        if trade_id is not None:
            if isinstance(trade_id, bool):
                return None
            if isinstance(trade_id, int):
                clean_trade_id = trade_id
            elif isinstance(trade_id, str):
                s_id = trade_id.strip()
                if not s_id:
                    return None
                clean_trade_id = s_id
            else:
                return None

        # Optional received_at_ms (do not invent)
        received_at = norm.get("_received_at_ms")
        clean_received: Optional[int] = None
        if received_at is not None:
            if isinstance(received_at, int) and not isinstance(received_at, bool) and received_at > 0:
                clean_received = received_at

        return (p_float, q_float, T, T_raw, m, source_clean, clean_trade_id, clean_received)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns an immutable snapshot dictionary of internal sink metrics."""
        with self._lock:
            return {
                "accepted_ticks": self._accepted_ticks,
                "duplicate_ticks": self._duplicate_ticks,
                "invalid_ticks": self._invalid_ticks,
                "out_of_order_ticks": self._out_of_order_ticks,
                "id_regressions": self._id_regressions,
                "continuity_suspect_ticks": self._continuity_suspect_ticks,
                "executor_errors": self._total_executor_errors,
                "accepted_orders": self._accepted_orders,
                "rejected_orders": self._rejected_orders,
                "disabled": self._disabled,
                "current_ingest_seq": self._ingest_seq,
                "dedup_cache_size": len(self._dedup_set),
                "registered_orders": len(self._registered_ingest_seq),
            }
