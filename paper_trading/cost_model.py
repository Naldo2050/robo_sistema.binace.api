# paper_trading/cost_model.py
"""
Cost model for hermetic paper trading.

Handles taker/maker fees, adverse slippage for entries and exits,
and funding rate settlements based on discrete UTC funding boundary crossings.
Supports zero-cost mode for mathematical invariants validation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from common.signal_direction import SignalSide
from paper_trading.contracts import PaperCostConfig


DEFAULT_PAPER_COST_CONFIG = PaperCostConfig(
    maker_fee_bps=2.0,
    taker_fee_bps=5.0,
    entry_slippage_bps=1.0,
    exit_slippage_bps=1.0,
    source="simulated_default_2026",
    effective_at="2026-01-01T00:00:00Z",
    funding_times_utc=(0, 8, 16),
)


def count_funding_crossings(
    opened_ts_ms: int,
    closed_ts_ms: int,
    funding_hours: Sequence[int] = (0, 8, 16),
) -> int:
    """
    Count how many funding timestamp events occurred strictly in (opened_ts_ms, closed_ts_ms].

    Funding events occur discretely at the top of the hour for the configured UTC funding hours
    (e.g., 00:00:00, 08:00:00, 16:00:00 UTC). No continuous linear pro-rata is applied.
    """
    if closed_ts_ms <= opened_ts_ms:
        return 0

    dt_open = datetime.datetime.fromtimestamp(opened_ts_ms / 1000.0, tz=datetime.timezone.utc)
    dt_close = datetime.datetime.fromtimestamp(closed_ts_ms / 1000.0, tz=datetime.timezone.utc)

    start_date = dt_open.date()
    end_date = dt_close.date()

    crossings = 0
    current_date = start_date
    one_day = datetime.timedelta(days=1)

    while current_date <= end_date:
        for hour in funding_hours:
            funding_dt = datetime.datetime(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                hour=hour,
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=datetime.timezone.utc,
            )
            funding_ts_ms = int(funding_dt.timestamp() * 1000)
            if opened_ts_ms < funding_ts_ms <= closed_ts_ms:
                crossings += 1
        current_date += one_day

    return crossings


@dataclass(frozen=True)
class PnLBreakdown:
    """Consolidated PnL result separating raw market movement from friction."""

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
    direction_correct: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.direction_correct is None and self.trade_direction_profitable is not None:
            object.__setattr__(self, "direction_correct", self.trade_direction_profitable)


class CostModel:
    """Calculates entry/exit fills, fees, slippage, and net PnL breakdown."""

    def __init__(self, config: Optional[PaperCostConfig] = None) -> None:
        self.config = config or DEFAULT_PAPER_COST_CONFIG

    def calculate_entry_fill(
        self,
        side: SignalSide,
        raw_price: float,
        notional_usdt: float,
        is_maker: bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate execution price and initial cost for an entry order.

        Adverse slippage:
        - LONG buys higher: raw_price * (1 + slippage_bps)
        - SHORT sells lower: raw_price * (1 - slippage_bps)

        When slippage_bps == 0, fill_price == raw_price.
        """
        slip_rate = self.config.entry_slippage_bps * 1e-4
        if slip_rate == 0.0 or side not in ("LONG", "SHORT"):
            fill_price = raw_price
        elif side == "LONG":
            fill_price = raw_price * (1.0 + slip_rate)
        else:  # SHORT
            fill_price = raw_price * (1.0 - slip_rate)

        quantity = notional_usdt / fill_price if fill_price > 0 else 0.0
        fee_rate = (self.config.maker_fee_bps if is_maker else self.config.taker_fee_bps) * 1e-4
        entry_fee_usdt = notional_usdt * fee_rate
        entry_slippage_usdt = abs(fill_price - raw_price) * quantity

        return fill_price, quantity, entry_fee_usdt, entry_slippage_usdt

    def calculate_exit_fill(
        self,
        side: SignalSide,
        exit_reason: str,
        raw_price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate exit execution price and costs.

        When exit_slippage_bps == 0, exit_price == raw_price.
        """
        if exit_reason == "TAKE_PROFIT" and is_maker:
            slippage_bps = 0.0
            fee_bps = self.config.maker_fee_bps
        else:
            slippage_bps = self.config.exit_slippage_bps
            fee_bps = self.config.taker_fee_bps

        slip_rate = slippage_bps * 1e-4
        if slip_rate == 0.0 or side not in ("LONG", "SHORT"):
            exit_price = raw_price
        elif side == "LONG":
            exit_price = raw_price * (1.0 - slip_rate)
        else:  # SHORT
            exit_price = raw_price * (1.0 + slip_rate)

        exit_notional_usdt = exit_price * quantity
        exit_fee_usdt = exit_notional_usdt * (fee_bps * 1e-4)
        exit_slippage_usdt = abs(raw_price - exit_price) * quantity

        return exit_price, exit_notional_usdt, exit_fee_usdt, exit_slippage_usdt

    def calculate_trade_pnl(
        self,
        side: SignalSide,
        raw_entry_price: float,
        raw_exit_price: float,
        quantity: float,
        notional_usdt: float,
        entry_fee_usdt: float,
        exit_fee_usdt: float,
        entry_slippage_usdt: float,
        exit_slippage_usdt: float,
        opened_ts_ms: int,
        closed_ts_ms: int,
        stop_loss: Optional[float] = None,
        funding_rate: Optional[float] = None,
    ) -> PnLBreakdown:
        """
        Calculate full PnL breakdown with costs, discrete funding, and trade_direction_profitable vs trade_win.
        """
        # Gross PnL from raw market price movement
        if side == "LONG":
            gross_pnl_usdt = (raw_exit_price - raw_entry_price) * quantity
        elif side == "SHORT":
            gross_pnl_usdt = (raw_entry_price - raw_exit_price) * quantity
        else:
            gross_pnl_usdt = 0.0

        total_fees_usdt = entry_fee_usdt + exit_fee_usdt
        total_slippage_usdt = entry_slippage_usdt + exit_slippage_usdt

        # Gross PnL in bps
        gross_pnl_bps = (gross_pnl_usdt / notional_usdt * 10_000.0) if notional_usdt > 0 else 0.0
        trade_direction_profitable: Optional[bool] = (gross_pnl_bps > 0.0) if notional_usdt > 0 else None
        # Prediction direction is reserved for canonical horizon evaluation (Gate C/D)
        prediction_direction_correct: Optional[bool] = None

        fees_bps = (total_fees_usdt / notional_usdt * 10_000.0) if notional_usdt > 0 else 0.0
        slippage_bps = (total_slippage_usdt / notional_usdt * 10_000.0) if notional_usdt > 0 else 0.0

        # Discrete funding settlement
        crossings = count_funding_crossings(
            opened_ts_ms=opened_ts_ms,
            closed_ts_ms=closed_ts_ms,
            funding_hours=self.config.funding_times_utc,
        )

        costs_complete = True
        funding_usdt: Optional[float] = 0.0
        funding_bps: Optional[float] = 0.0

        if crossings > 0:
            if funding_rate is None:
                funding_usdt = None
                funding_bps = None
                costs_complete = False
            else:
                side_mult = 1.0 if side == "LONG" else -1.0
                funding_usdt = -side_mult * funding_rate * notional_usdt * crossings
                funding_bps = (funding_usdt / notional_usdt * 10_000.0) if notional_usdt > 0 else 0.0

        # Net PnL calculation
        if funding_usdt is None:
            net_pnl_usdt = None
            net_pnl_bps = None
            trade_win = None
            pnl_R = None
        else:
            total_costs = total_fees_usdt + total_slippage_usdt
            if total_costs == 0.0 and funding_usdt == 0.0:
                # Zero cost mode: exact equality
                net_pnl_usdt = gross_pnl_usdt
                net_pnl_bps = gross_pnl_bps
            else:
                net_pnl_usdt = gross_pnl_usdt - total_costs + funding_usdt
                net_pnl_bps = (net_pnl_usdt / notional_usdt * 10_000.0) if notional_usdt > 0 else 0.0

            trade_win = net_pnl_usdt > 0.0

            # pnl_R: net / initial risk
            if stop_loss is not None and stop_loss > 0:
                risk_usdt = abs(raw_entry_price - stop_loss) * quantity
                if risk_usdt > 0:
                    pnl_R = net_pnl_usdt / risk_usdt
                else:
                    pnl_R = None
            else:
                pnl_R = None

        return PnLBreakdown(
            trade_direction_profitable=trade_direction_profitable,
            prediction_direction_correct=prediction_direction_correct,
            trade_win=trade_win,
            gross_pnl_bps=gross_pnl_bps,
            net_pnl_bps=net_pnl_bps,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            gross_pnl_usdt=gross_pnl_usdt,
            fees_usdt=total_fees_usdt,
            slippage_usdt=total_slippage_usdt,
            funding_usdt=funding_usdt,
            net_pnl_usdt=net_pnl_usdt,
            pnl_R=pnl_R,
            costs_complete=costs_complete,
        )
