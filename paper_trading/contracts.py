# paper_trading/contracts.py
"""
Data contracts for hermetic paper trading.

All decision inputs and trade representations are strictly typed and immutable
wherever possible to prevent in-place mutation and ensure reproducible replay.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from common.exceptions import BotBaseError
from common.json_safe import sanitize_json_safe
from common.signal_direction import SignalSide, OutcomeResult


class PaperTradingError(BotBaseError):
    """Base exception for all paper trading errors."""
    pass


class InvalidDecisionError(PaperTradingError):
    """Raised when CanonicalDecision violates causal or geometric invariants."""
    pass


@dataclass(frozen=True)
class PaperCostConfig:
    """
    Versionable and auditable cost configuration per cohort.

    Eliminates universal hardcoded fee constants and establishes traceable fee baselines.
    Supports zero-cost configurations for pure mathematical verification.
    """

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    entry_slippage_bps: float = 1.0
    exit_slippage_bps: float = 1.0
    source: str = "default_cost_config"
    effective_at: str = "2026-01-01T00:00:00Z"
    funding_times_utc: Tuple[int, ...] = (0, 8, 16)


@dataclass(frozen=True)
class CanonicalDecision:
    """
    Immutable canonical decision contract.

    Invariants enforced at construction:
    - signal_timestamp <= decision_timestamp <= available_at (strict temporal causality)
    - reference_price finite and > 0
    - Individual SL/TP validated against reference_price:
        LONG:  stop_loss < reference_price, take_profit > reference_price
        SHORT: stop_loss > reference_price, take_profit < reference_price
        (SL and TP are optional; neither is required simultaneously)
    - confidence must be float in [0.0, 1.0] or None (None = uncalibrated/unknown, never coerced to 0.0)
    - UUID5 deterministic identification from:
        cohort_id, symbol, window_id, decision_provider, strategy_version, model_version
    """

    cohort_id: str
    symbol: str
    window_id: str
    decision_provider: str
    strategy_version: str
    signal_timestamp: int
    decision_timestamp: int
    available_at: int
    side: SignalSide
    reference_price: float
    notional_usdt: float
    horizon_s: int
    entry_type: str = "MARKET"
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    funding_rate_at_decision: Optional[float] = None
    funding_rate_source: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    provider_meta: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        # Temporal causality: signal_timestamp <= decision_timestamp <= available_at
        if not (self.signal_timestamp <= self.decision_timestamp <= self.available_at):
            raise InvalidDecisionError(
                f"Causal violation: expected signal_timestamp ({self.signal_timestamp}) "
                f"<= decision_timestamp ({self.decision_timestamp}) "
                f"<= available_at ({self.available_at})"
            )

        # Reference price validation
        if not (math.isfinite(self.reference_price) and self.reference_price > 0):
            raise InvalidDecisionError(
                f"reference_price must be finite and > 0, got {self.reference_price}"
            )

        # Confidence validation: float in [0.0, 1.0] or None. Never convert UNKNOWN to 0.0.
        if self.confidence is not None:
            if not math.isfinite(self.confidence) or self.confidence < 0.0 or self.confidence > 1.0:
                raise InvalidDecisionError(
                    f"Invalid confidence: {self.confidence} (must be in [0.0, 1.0] or None)"
                )

        # Individual SL / TP Geometry validation against reference_price
        sl = self.stop_loss
        tp = self.take_profit
        ref = self.reference_price

        if sl is not None:
            if not (math.isfinite(sl) and sl > 0):
                raise InvalidDecisionError(f"stop_loss must be positive finite number, got {sl}")
            if self.side == "LONG" and not (sl < ref):
                raise InvalidDecisionError(f"LONG stop_loss ({sl}) must be < reference_price ({ref})")
            elif self.side == "SHORT" and not (sl > ref):
                raise InvalidDecisionError(f"SHORT stop_loss ({sl}) must be > reference_price ({ref})")

        if tp is not None:
            if not (math.isfinite(tp) and tp > 0):
                raise InvalidDecisionError(f"take_profit must be positive finite number, got {tp}")
            if self.side == "LONG" and not (tp > ref):
                raise InvalidDecisionError(f"LONG take_profit ({tp}) must be > reference_price ({ref})")
            elif self.side == "SHORT" and not (tp < ref):
                raise InvalidDecisionError(f"SHORT take_profit ({tp}) must be < reference_price ({ref})")

        # Deterministic UUID5 calculation
        model_str = self.model_version or "NONE"
        identity_string = (
            f"{self.cohort_id}:{self.symbol}:{self.window_id}:"
            f"{self.decision_provider}:{self.strategy_version}:{model_str}"
        )
        object.__setattr__(
            self,
            "decision_id",
            str(uuid.uuid5(uuid.NAMESPACE_OID, identity_string)),
        )

        # Ensure context and metadata are sanitized against NaN / Inf
        object.__setattr__(self, "context", sanitize_json_safe(self.context))
        object.__setattr__(self, "provider_meta", sanitize_json_safe(self.provider_meta))


@dataclass(frozen=True)
class Rejection:
    """Recorded when a decision cannot be executed."""

    decision_id: str
    cohort_id: str
    decision_provider: str
    symbol: str
    decision_timestamp: int
    reason: Literal[
        "NO_DIRECTION",
        "POSITION_OPEN",
        "DUPLICATE_DECISION",
        "CIRCUIT_BREAKER",
        "EXPIRED_NO_MARKET_DATA",
        "INVALID_DECISION",
    ]
    details: str
    rejected_at: int


@dataclass(frozen=True)
class PaperOrder:
    """
    Pending simulated order awaiting fill after available_at latency threshold.

    Designed to be compatible with existing production TradeRequest schemas.
    """

    order_id: str
    decision_id: str
    cohort_id: str
    decision_provider: str
    symbol: str
    side: SignalSide
    reference_price: float
    notional_usdt: float
    signal_timestamp: int
    decision_timestamp: int
    available_at: int
    expires_at: int
    horizon_s: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    funding_rate_at_decision: Optional[float] = None
    funding_rate_source: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperFill:
    """Execution receipt for an order filled on market tick."""

    fill_id: str
    order_id: str
    decision_id: str
    cohort_id: str
    symbol: str
    side: SignalSide
    fill_price: float
    raw_price: float
    slippage_bps: float
    quantity: float
    notional_usdt: float
    fill_timestamp: int
    trade_id_used: int | str
    fee_usdt: float
    decision_to_fill_ms: int
    available_to_fill_ms: int


@dataclass
class PaperPosition:
    """Mutable tracking structure for open simulated positions."""

    position_id: str
    cohort_id: str
    decision_provider: str
    symbol: str
    side: SignalSide
    entry_price: float
    reference_price: float
    quantity: float
    notional_usdt: float
    opened_ts_ms: int
    horizon_deadline_ms: int
    decision_id: str
    signal_timestamp: int
    decision_timestamp: int
    available_at: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    funding_rate_at_decision: Optional[float] = None
    funding_rate_source: Optional[str] = None
    entry_fee_usdt: float = 0.0
    entry_slippage_usdt: float = 0.0
    mae_bps: float = 0.0
    mfe_bps: float = 0.0
    ticks_processed: int = 0
    last_tick_ts_ms: int = 0
    data_gap: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClosedTrade:
    """Completed round-trip simulated trade with exhaustive cost and directional breakdown."""

    trade_id: str
    decision_id: str
    cohort_id: str
    decision_provider: str
    symbol: str
    side: SignalSide
    entry_price: float
    exit_price: float
    quantity: float
    notional_usdt: float
    opened_ts_ms: int
    closed_ts_ms: int
    exit_reason: Literal[
        "TAKE_PROFIT",
        "STOP_LOSS",
        "HORIZON_EXPIRY",
        "UNKNOWN_DATA_GAP",
        "KILL_SWITCH",
    ]
    trade_direction_profitable: Optional[bool]
    prediction_direction_correct: Optional[bool]
    trade_win: Optional[bool]
    gross_pnl_bps: Optional[float]
    net_pnl_bps: Optional[float]
    fees_bps: float
    slippage_bps: float
    funding_bps: Optional[float]
    gross_pnl_usdt: Optional[float]
    fees_usdt: float
    slippage_usdt: float
    funding_usdt: Optional[float]
    net_pnl_usdt: Optional[float]
    pnl_R: Optional[float]
    costs_complete: bool
    data_gap: bool
    mae_bps: float
    mfe_bps: float
    ticks_count: int
    direction_correct: Optional[bool] = None  # alias for trade_direction_profitable
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction_correct is None and self.trade_direction_profitable is not None:
            object.__setattr__(self, "direction_correct", self.trade_direction_profitable)
