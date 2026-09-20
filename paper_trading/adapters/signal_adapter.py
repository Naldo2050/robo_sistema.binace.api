# paper_trading/adapters/signal_adapter.py
"""
Hermetic Signal-to-Decision Adapter for Paper Trading (Gate C1-B).

Transforms qualified runtime market events and signals into immutable CanonicalDecision
instances or explicit skip results, decoupling the signal pipeline from trade execution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

from common.signal_direction import (
    SignalSide,
    infer_signal_side,
    normalize_signal_label,
)
from paper_trading.contracts import CanonicalDecision
from paper_trading.decision_providers import DecisionProvider

AdapterMode = Literal["FOLLOW_SIGNAL", "BASELINE"]
AdapterStatus = Literal[
    "DECISION_CREATED",
    "SKIPPED_NON_DIRECTIONAL",
    "SKIPPED_INVALID_EVENT",
]


@dataclass(frozen=True)
class AdapterResult:
    """
    Explicit structured output of the signal decision adapter.

    Attributes:
        decision: The constructed CanonicalDecision if status is DECISION_CREATED, else None.
        status: DECISION_CREATED, SKIPPED_NON_DIRECTIONAL, or SKIPPED_INVALID_EVENT.
        source_side: The original direction inferred directly from the event (LONG/SHORT/NEUTRAL/UNKNOWN).
        reason: Diagnostic explanation when an event is skipped or rejected.
        source_window_id: Canonical window identity in the format '{symbol}:{timeframe}:{close_ms}'.
        source_event_key: Deterministic event identifier within the window.
    """

    decision: Optional[CanonicalDecision]
    status: AdapterStatus
    source_side: SignalSide
    reason: Optional[str]
    source_window_id: str
    source_event_key: str


class SignalDecisionAdapter:
    """
    Adapts market signal dictionaries into CanonicalDecision contracts.

    Operates in two modes:
    1. FOLLOW_SIGNAL: Directly executes the observed market signal direction
       (LONG -> LONG, SHORT -> SHORT). Non-directional signals (NEUTRAL, UNKNOWN)
       are skipped with SKIPPED_NON_DIRECTIONAL.
    2. BASELINE: Evaluates a DecisionProvider (Fixed or Random) to generate
       a null-hypothesis benchmark side, preserving source_side for statistical auditing.
    """

    def __init__(
        self,
        cohort_id: str,
        strategy_version: str = "c1_v1.0.0",
        mode: AdapterMode = "FOLLOW_SIGNAL",
        provider: Optional[DecisionProvider] = None,
        timeframe: str = "1m",
        default_notional_usdt: float = 1000.0,
        default_horizon_s: int = 300,
        configured_latency_ms: int = 0,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        if mode not in ("FOLLOW_SIGNAL", "BASELINE"):
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'FOLLOW_SIGNAL' or 'BASELINE'")
        if mode == "BASELINE" and provider is None:
            raise ValueError("mode='BASELINE' requires an explicit DecisionProvider")

        self.cohort_id = cohort_id
        self.strategy_version = strategy_version
        self.mode = mode
        self.provider = provider
        self.timeframe = timeframe
        self.default_notional_usdt = default_notional_usdt
        self.default_horizon_s = default_horizon_s
        self.configured_latency_ms = max(0, configured_latency_ms)
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))

    def build_source_window_id(self, symbol: str, close_ms: int) -> str:
        """Construct canonical deterministic window identity."""
        return f"{symbol}:{self.timeframe}:{close_ms}"

    def build_source_event_key(self, event: Dict[str, Any], reference_price: float) -> str:
        """
        Construct stable deterministic event key within a window.

        Extracts purely immutable signal properties without relying on random tokens,
        wall-clock times, native object IDs, or dictionary traversal order.
        """
        raw_type = event.get("tipo_evento") or event.get("event_type") or "SIGNAL"
        norm_type = normalize_signal_label(str(raw_type)) or "SIGNAL"

        raw_battle = (
            event.get("resultado_da_batalha")
            or event.get("battle_result")
            or event.get("alert_type")
            or event.get("trigger_type")
            or "DEFAULT"
        )
        norm_battle = normalize_signal_label(str(raw_battle)) or "DEFAULT"

        # Stable price string representation (2 decimal places)
        price_str = f"{reference_price:.2f}" if math.isfinite(reference_price) else "0.00"

        # Sub-identifier if present (e.g. specific zone price level or cluster)
        level = event.get("nivel") or event.get("zone_level") or event.get("level")
        level_part = f":L{level}" if level is not None else ""

        return f"{norm_type}:{norm_battle}:{price_str}{level_part}"

    def process_signal(self, event: Dict[str, Any]) -> AdapterResult:
        """
        Process a raw signal event dictionary into an AdapterResult.

        Invariants enforced:
        - Strict temporal causality (signal_timestamp <= decision_timestamp <= available_at).
        - Fail-closed if clock is behind signal_timestamp.
        - Reference price validated as finite and > 0.
        - Confidence remains strictly None until formal calibration is established.
        - Idempotent decision_id generation via composite canonical window scope.
        """
        if not isinstance(event, dict):
            return AdapterResult(
                decision=None,
                status="SKIPPED_INVALID_EVENT",
                source_side="UNKNOWN",
                reason="Event payload must be a dictionary",
                source_window_id="UNKNOWN:UNKNOWN:0",
                source_event_key="INVALID",
            )

        symbol = str(event.get("symbol") or "BTCUSDT").upper()

        # Extract close_ms / signal_timestamp
        close_ms_raw = event.get("epoch_ms") or event.get("close_ms") or event.get("T")
        if close_ms_raw is None:
            return AdapterResult(
                decision=None,
                status="SKIPPED_INVALID_EVENT",
                source_side="UNKNOWN",
                reason="Missing temporal epoch boundary ('epoch_ms', 'close_ms', or 'T')",
                source_window_id=f"{symbol}:{self.timeframe}:0",
                source_event_key="INVALID",
            )

        try:
            signal_timestamp = int(close_ms_raw)
        except (ValueError, TypeError):
            return AdapterResult(
                decision=None,
                status="SKIPPED_INVALID_EVENT",
                source_side="UNKNOWN",
                reason=f"Invalid signal timestamp: {close_ms_raw}",
                source_window_id=f"{symbol}:{self.timeframe}:0",
                source_event_key="INVALID",
            )

        source_window_id = self.build_source_window_id(symbol, signal_timestamp)

        # Extract reference price
        raw_price = (
            event.get("preco_fechamento")
            or event.get("p")
            or event.get("price")
            or event.get("close")
            or (event.get("ohlc") or {}).get("close")
        )
        try:
            reference_price = float(raw_price) if raw_price is not None else 0.0
        except (ValueError, TypeError):
            reference_price = 0.0

        if not (math.isfinite(reference_price) and reference_price > 0):
            return AdapterResult(
                decision=None,
                status="SKIPPED_INVALID_EVENT",
                source_side="UNKNOWN",
                reason=f"reference_price must be finite and > 0, got {raw_price}",
                source_window_id=source_window_id,
                source_event_key="INVALID_PRICE",
            )

        source_event_key = self.build_source_event_key(event, reference_price)

        # Temporal causality check
        decision_timestamp = self.clock_ms()
        if decision_timestamp < signal_timestamp:
            return AdapterResult(
                decision=None,
                status="SKIPPED_INVALID_EVENT",
                source_side="UNKNOWN",
                reason=(
                    f"Causal violation: decision clock ({decision_timestamp}) "
                    f"< signal_timestamp ({signal_timestamp})"
                ),
                source_window_id=source_window_id,
                source_event_key=source_event_key,
            )

        available_at = decision_timestamp + self.configured_latency_ms

        # Infer source directional side
        source_side = infer_signal_side(
            event_type=event.get("tipo_evento") or event.get("event_type"),
            battle_result=event.get("resultado_da_batalha") or event.get("battle_result"),
            explicit_side=event.get("side") or event.get("absorption_side"),
        )

        # Determine decision side based on adapter mode
        decision_provider_id: str
        decision_side: SignalSide

        if self.mode == "FOLLOW_SIGNAL":
            decision_provider_id = "signal_follow"
            if source_side in ("NEUTRAL", "UNKNOWN"):
                return AdapterResult(
                    decision=None,
                    status="SKIPPED_NON_DIRECTIONAL",
                    source_side=source_side,
                    reason=f"FOLLOW_SIGNAL mode skips non-directional side '{source_side}'",
                    source_window_id=source_window_id,
                    source_event_key=source_event_key,
                )
            decision_side = source_side

        elif self.mode == "BASELINE":
            assert self.provider is not None
            decision_provider_id = self.provider.provider_id
            decision_side = self.provider.get_decision_side(source_side, context=event)
            if decision_side in ("NEUTRAL", "UNKNOWN"):
                return AdapterResult(
                    decision=None,
                    status="SKIPPED_NON_DIRECTIONAL",
                    source_side=source_side,
                    reason=f"BASELINE provider '{decision_provider_id}' returned non-directional '{decision_side}'",
                    source_window_id=source_window_id,
                    source_event_key=source_event_key,
                )
        else:
            raise RuntimeError(f"Unhandled mode: {self.mode}")

        # Qualified composite window identity ensures unique deterministic decision_id
        # for distinct events within the same time window, while guaranteeing
        # identical decision_id for retries of the same event.
        canonical_window_scope = f"{source_window_id}#{source_event_key}"

        decision = CanonicalDecision(
            cohort_id=self.cohort_id,
            symbol=symbol,
            window_id=canonical_window_scope,
            decision_provider=decision_provider_id,
            strategy_version=self.strategy_version,
            signal_timestamp=signal_timestamp,
            decision_timestamp=decision_timestamp,
            available_at=available_at,
            side=decision_side,
            reference_price=reference_price,
            notional_usdt=self.default_notional_usdt,
            horizon_s=self.default_horizon_s,
            entry_type="MARKET",
            model_version=None,
            confidence=None,  # Uncalibrated; never coerced to 0.0
            funding_rate_at_decision=None,
            funding_rate_source=None,
            context={
                "source_window_id": source_window_id,
                "source_event_key": source_event_key,
                "source_side": source_side,
                "mode": self.mode,
            },
            provider_meta={
                "adapter_version": "c1_b",
                "source_event_type": event.get("tipo_evento"),
            },
        )

        return AdapterResult(
            decision=decision,
            status="DECISION_CREATED",
            source_side=source_side,
            reason=None,
            source_window_id=source_window_id,
            source_event_key=source_event_key,
        )
