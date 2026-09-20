# paper_trading/replay.py
"""
Deterministic replay runner for paper trading simulations.

Merges an ordered stream of canonical decisions with historical or synthetic market ticks,
coordinating executor, position tracking, cost evaluation, and ledger recording entirely in-memory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from paper_trading.contracts import (
    CanonicalDecision,
    ClosedTrade,
    PaperCostConfig,
    PaperFill,
    Rejection,
)
from paper_trading.cost_model import CostModel, DEFAULT_PAPER_COST_CONFIG
from paper_trading.executor import ExecutorConfig, PaperExecutor
from paper_trading.ledger import PaperLedger
from paper_trading.positions import PositionConfig, PositionManager
from paper_trading.tape import TapeTick


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a complete deterministic replay run."""

    closed_trades: List[ClosedTrade]
    rejections: List[Rejection]
    ledger_hash: str
    fills: List[PaperFill] = field(default_factory=list)


class ReplayRunner:
    """
    Executes a deterministic simulation over a tick tape and decision set.
    """

    def __init__(
        self,
        decisions: Sequence[CanonicalDecision],
        tape: Iterable[TapeTick],
        cost_config: Optional[PaperCostConfig] = None,
        executor_config: Optional[ExecutorConfig] = None,
        position_config: Optional[PositionConfig] = None,
        db_path: str = ":memory:",
    ) -> None:
        self.decisions = sorted(decisions, key=lambda d: d.decision_timestamp)
        self.tape = tape
        self.cost_model = CostModel(cost_config or DEFAULT_PAPER_COST_CONFIG)
        self.position_manager = PositionManager(
            cost_model=self.cost_model,
            config=position_config or PositionConfig(),
        )
        self.executor = PaperExecutor(
            cost_model=self.cost_model,
            position_manager=self.position_manager,
            config=executor_config or ExecutorConfig(),
        )
        self.ledger = PaperLedger(db_path=db_path)

    def run(self) -> ReplayResult:
        """
        Execute simulation loop.

        Steps:
        1. As ticks arrive with timestamp T, submit all decisions where decision_timestamp <= T.
        2. Deliver tick to executor and position manager.
        3. Persist all generated transitions into the ledger.
        4. Flush remaining expired orders/positions at end of tape.
        """
        decision_idx = 0
        total_decisions = len(self.decisions)
        last_tick_t = 0

        for tick in self.tape:
            last_tick_t = tick.T

            # Submit all decisions produced at or before this tick's timestamp
            while decision_idx < total_decisions:
                d = self.decisions[decision_idx]
                if d.decision_timestamp <= tick.T:
                    self.ledger.record_decision(d)
                    order, rejection = self.executor.submit_decision(d)
                    if order is not None:
                        self.ledger.record_order(order)
                    if rejection is not None:
                        self.ledger.record_rejection(rejection)
                    decision_idx += 1
                else:
                    break

            # Dispatch tick to executor
            events = self.executor.on_tick(
                T=tick.T,
                p=tick.p,
                q=tick.q,
                m=tick.m,
                trade_id=tick.trade_id,
            )

            # Record event transitions
            for fill in events.fills:
                self.ledger.record_fill(fill)
            for pos in events.opened_positions:
                self.ledger.record_position(pos)
            for rej in events.rejections:
                self.ledger.record_rejection(rej)
            for trade in events.closed_trades:
                self.ledger.record_closed_trade(trade)

        # Flush any decisions that occurred after the last tape tick
        while decision_idx < total_decisions:
            d = self.decisions[decision_idx]
            self.ledger.record_decision(d)
            order, rejection = self.executor.submit_decision(d)
            if order is not None:
                self.ledger.record_order(order)
                rej = Rejection(
                    decision_id=order.decision_id,
                    cohort_id=order.cohort_id,
                    decision_provider=order.decision_provider,
                    symbol=order.symbol,
                    decision_timestamp=order.decision_timestamp,
                    reason="EXPIRED_NO_MARKET_DATA",
                    details="Simulation ended with no eligible market data ticks.",
                    rejected_at=order.decision_timestamp,
                )
                self.ledger.record_rejection(rej)
            if rejection is not None:
                self.ledger.record_rejection(rejection)
            decision_idx += 1

        # Check remaining open positions for expiry at end of tape
        final_expired = self.position_manager.check_expired_positions(
            last_tick_t + self.position_manager.config.horizon_grace_ms + 1000
        )
        for trade in final_expired:
            self.ledger.record_closed_trade(trade)

        # Drain and flush ledger
        self.ledger.flush()

        closed_trades = self.ledger.get_closed_trades()
        rejections = self.ledger.get_rejections()
        fills = self.ledger.get_fills()
        ledger_hash = self._compute_ledger_hash()

        # Clean up worker thread
        self.ledger.close()

        return ReplayResult(
            closed_trades=closed_trades,
            rejections=rejections,
            ledger_hash=ledger_hash,
            fills=fills,
        )

    def _compute_ledger_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint across all primary ledger tables."""
        hasher = hashlib.sha256()
        cur = self.ledger._conn.cursor()

        for query in [
            "SELECT decision_id, cohort_id, decision_provider, symbol, side, reference_price, notional_usdt, confidence FROM decisions ORDER BY decision_id",
            "SELECT decision_id, reason, details FROM rejections ORDER BY decision_id, reason",
            "SELECT order_id, decision_id, side, reference_price, notional_usdt FROM orders ORDER BY order_id",
            "SELECT fill_id, order_id, fill_price, quantity, fee_usdt, decision_to_fill_ms, available_to_fill_ms FROM fills ORDER BY fill_id",
            "SELECT trade_id, exit_reason, round(entry_price, 4), round(exit_price, 4), round(COALESCE(net_pnl_bps, 0.0), 4), trade_win, trade_direction_profitable FROM closed_trades ORDER BY trade_id",
        ]:
            cur.execute(query)
            for row in cur.fetchall():
                hasher.update(str(tuple(row)).encode("utf-8"))

        return hasher.hexdigest()
