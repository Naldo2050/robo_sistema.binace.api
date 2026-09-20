# paper_trading/executor.py
"""
Simulated paper order executor.

Enforces execution latency via available_at, strict tick-by-tick fill mechanics,
TTL expiration, single active position constraints, and zero-I/O hot path processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from paper_trading.contracts import (
    CanonicalDecision,
    PaperFill,
    PaperOrder,
    PaperPosition,
    Rejection,
)
from paper_trading.cost_model import CostModel
from paper_trading.positions import PositionManager


@dataclass(frozen=True)
class ExecutorConfig:
    """Configuration parameters for simulated execution latency and order lifecycles."""

    order_ttl_ms: int = 5000


@dataclass
class TickEvents:
    """Batch of execution events produced during a single market tick."""

    fills: List[PaperFill] = field(default_factory=list)
    opened_positions: List[PaperPosition] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)
    closed_trades: List[Any] = field(default_factory=list)

    @property
    def has_events(self) -> bool:
        return bool(self.fills or self.opened_positions or self.rejections or self.closed_trades)


class PaperExecutor:
    """
    Manages pending orders and coordinates fills into active positions.

    Hot-path invariants:
    - on_tick() executes in O(pending_orders + active_positions)
    - Zero per-tick logging and zero synchronous disk/DB I/O
    - Invariant: signal_timestamp <= decision_timestamp <= available_at <= fill_timestamp
    - Eligible tick satisfies tick.timestamp >= available_at
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        position_manager: Optional[PositionManager] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.position_manager = position_manager or PositionManager(cost_model=self.cost_model)
        self.config = config or ExecutorConfig()
        self.pending_orders: Dict[str, PaperOrder] = {}
        # Pending symbol map: (cohort_id, decision_provider, symbol) -> order_id
        self._pending_symbol_map: Dict[Tuple[str, str, str], str] = {}

    def submit_decision(
        self,
        decision: CanonicalDecision,
    ) -> Tuple[Optional[PaperOrder], Optional[Rejection]]:
        """
        Evaluate a canonical decision and transition it into a pending PaperOrder or Rejection.

        Invariants enforced:
        - NEUTRAL or UNKNOWN side -> Rejection(reason=NO_DIRECTION)
        - Active position already exists -> Rejection(reason=POSITION_OPEN)
        - Pending order already exists -> Rejection(reason=POSITION_OPEN)
        - Never fills immediately: must await first eligible tick where tick.timestamp >= available_at
        """
        # Direction check: NEUTRAL and UNKNOWN never become orders
        if decision.side in ("NEUTRAL", "UNKNOWN"):
            rejection = Rejection(
                decision_id=decision.decision_id,
                cohort_id=decision.cohort_id,
                decision_provider=decision.decision_provider,
                symbol=decision.symbol,
                decision_timestamp=decision.decision_timestamp,
                reason="NO_DIRECTION",
                details=f"Decision side '{decision.side}' does not constitute an actionable trade order.",
                rejected_at=decision.decision_timestamp,
            )
            return None, rejection

        # Position / pending order conflict check
        key = (decision.cohort_id, decision.decision_provider, decision.symbol)
        if self.position_manager.has_open_position(
            cohort_id=decision.cohort_id,
            decision_provider=decision.decision_provider,
            symbol=decision.symbol,
        ) or key in self._pending_symbol_map:
            rejection = Rejection(
                decision_id=decision.decision_id,
                cohort_id=decision.cohort_id,
                decision_provider=decision.decision_provider,
                symbol=decision.symbol,
                decision_timestamp=decision.decision_timestamp,
                reason="POSITION_OPEN",
                details=f"An active position or pending order already exists for {key}.",
                rejected_at=decision.decision_timestamp,
            )
            return None, rejection

        # Construct pending PaperOrder
        expires_at = decision.available_at + self.config.order_ttl_ms
        order_id = f"ord_{decision.decision_id[:12]}_{decision.decision_timestamp}"

        order = PaperOrder(
            order_id=order_id,
            decision_id=decision.decision_id,
            cohort_id=decision.cohort_id,
            decision_provider=decision.decision_provider,
            symbol=decision.symbol,
            side=decision.side,
            reference_price=decision.reference_price,
            notional_usdt=decision.notional_usdt,
            signal_timestamp=decision.signal_timestamp,
            decision_timestamp=decision.decision_timestamp,
            available_at=decision.available_at,
            expires_at=expires_at,
            horizon_s=decision.horizon_s,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            funding_rate_at_decision=decision.funding_rate_at_decision,
            funding_rate_source=decision.funding_rate_source,
            context=decision.context,
        )

        self.pending_orders[order.order_id] = order
        self._pending_symbol_map[key] = order.order_id

        return order, None

    def on_tick(
        self,
        T: int,
        p: float,
        q: float,
        m: bool,
        trade_id: int | str = 0,
    ) -> TickEvents:
        """
        Process incoming tick across all pending orders and active positions.

        Strict invariants:
        - Order can only be filled by the first tick where T >= order.available_at.
        - Fills record decision_to_fill_ms = T - decision_timestamp >= 0
          and available_to_fill_ms = T - available_at >= 0.
        - No tick received before expires_at -> EXPIRED_NO_MARKET_DATA rejection.
        - Fills execute at raw tick price + adverse slippage.
        """
        events = TickEvents()

        # 1. Process existing active positions against incoming tick
        closed_trades = self.position_manager.on_tick(
            T=T,
            p=p,
            q=q,
            m=m,
            trade_id=trade_id,
        )
        events.closed_trades.extend(closed_trades)

        # 2. Process pending orders eligible for fill on incoming tick
        expired_order_ids: List[str] = []
        filled_order_ids: List[str] = []

        for order_id, order in list(self.pending_orders.items()):
            # Temporal eligibility: tick timestamp must be >= available_at
            if T < order.available_at:
                continue

            if T > order.expires_at:
                # Expired without market data fill
                rejection = Rejection(
                    decision_id=order.decision_id,
                    cohort_id=order.cohort_id,
                    decision_provider=order.decision_provider,
                    symbol=order.symbol,
                    decision_timestamp=order.decision_timestamp,
                    reason="EXPIRED_NO_MARKET_DATA",
                    details=f"No eligible tick received before order expired at {order.expires_at}ms.",
                    rejected_at=T,
                )
                events.rejections.append(rejection)
                expired_order_ids.append(order_id)
                continue

            # Eligible for fill: order.available_at <= T <= order.expires_at
            (
                fill_price,
                quantity,
                entry_fee_usdt,
                entry_slippage_usdt,
            ) = self.cost_model.calculate_entry_fill(
                side=order.side,
                raw_price=p,
                notional_usdt=order.notional_usdt,
                is_maker=False,
            )

            decision_to_fill_ms = T - order.decision_timestamp
            available_to_fill_ms = T - order.available_at

            assert decision_to_fill_ms >= 0, f"decision_to_fill_ms must be >= 0, got {decision_to_fill_ms}"
            assert available_to_fill_ms >= 0, f"available_to_fill_ms must be >= 0, got {available_to_fill_ms}"

            fill = PaperFill(
                fill_id=f"fill_{order.order_id}",
                order_id=order.order_id,
                decision_id=order.decision_id,
                cohort_id=order.cohort_id,
                symbol=order.symbol,
                side=order.side,
                fill_price=fill_price,
                raw_price=p,
                slippage_bps=self.cost_model.config.entry_slippage_bps,
                quantity=quantity,
                notional_usdt=order.notional_usdt,
                fill_timestamp=T,
                trade_id_used=trade_id,
                fee_usdt=entry_fee_usdt,
                decision_to_fill_ms=decision_to_fill_ms,
                available_to_fill_ms=available_to_fill_ms,
            )
            events.fills.append(fill)

            horizon_deadline = order.decision_timestamp + (order.horizon_s * 1000)
            position = PaperPosition(
                position_id=f"pos_{order.order_id}",
                cohort_id=order.cohort_id,
                decision_provider=order.decision_provider,
                symbol=order.symbol,
                side=order.side,
                entry_price=fill_price,
                reference_price=order.reference_price,
                quantity=quantity,
                notional_usdt=order.notional_usdt,
                opened_ts_ms=T,
                horizon_deadline_ms=horizon_deadline,
                decision_id=order.decision_id,
                signal_timestamp=order.signal_timestamp,
                decision_timestamp=order.decision_timestamp,
                available_at=order.available_at,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                funding_rate_at_decision=order.funding_rate_at_decision,
                funding_rate_source=order.funding_rate_source,
                entry_fee_usdt=entry_fee_usdt,
                entry_slippage_usdt=entry_slippage_usdt,
                mae_bps=0.0,
                mfe_bps=0.0,
                ticks_processed=0,
                last_tick_ts_ms=T,
                data_gap=False,
                context=order.context,
            )
            self.position_manager.add_position(position)
            events.opened_positions.append(position)
            filled_order_ids.append(order_id)

        # Cleanup processed orders
        for o_id in expired_order_ids + filled_order_ids:
            popped_order = self.pending_orders.get(o_id)
            if popped_order is not None:
                del self.pending_orders[o_id]
                key = (popped_order.cohort_id, popped_order.decision_provider, popped_order.symbol)
                self._pending_symbol_map.pop(key, None)

        return events
