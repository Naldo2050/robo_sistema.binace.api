# tests/unit/paper_trading/test_position_exits.py
"""Unit tests for position tracking, exit mechanics, inclusive TP/SL, and gap resolutions."""

import pytest

from paper_trading.contracts import PaperCostConfig, PaperPosition
from paper_trading.cost_model import CostModel
from paper_trading.positions import PositionConfig, PositionManager


def create_sample_position(side="LONG", entry_price=100.0, sl=95.0, tp=110.0, horizon_s=60):
    return PaperPosition(
        position_id="pos_1",
        cohort_id="c1",
        decision_provider="flow_v1",
        symbol="BTCUSDT",
        side=side,
        entry_price=entry_price,
        reference_price=100.0,
        quantity=1.0,
        notional_usdt=100.0,
        opened_ts_ms=1000,
        horizon_deadline_ms=1000 + (horizon_s * 1000),
        decision_id="dec_1",
        signal_timestamp=980,
        decision_timestamp=1000,
        available_at=1020,
        stop_loss=sl,
        take_profit=tp,
        entry_fee_usdt=0.05,
        entry_slippage_usdt=0.01,
    )


def test_inclusive_take_profit_and_stop_loss():
    """Item 3: Level exactly traded counts as reached (inclusive TP/SL)."""
    pm = PositionManager()
    # LONG: p == 110.0 reaches TP
    pos_long = create_sample_position(side="LONG", entry_price=100.0, sl=90.0, tp=110.0)
    pm.add_position(pos_long)

    closed_tp = pm.on_tick(T=2000, p=110.0, q=1.0, m=False)
    assert len(closed_tp) == 1
    assert closed_tp[0].exit_reason == "TAKE_PROFIT"
    assert closed_tp[0].trade_win is True

    # SHORT: p == 90.0 reaches TP
    pos_short = create_sample_position(side="SHORT", entry_price=100.0, sl=110.0, tp=90.0)
    pm.add_position(pos_short)

    closed_tp_short = pm.on_tick(T=3000, p=90.0, q=1.0, m=False)
    assert len(closed_tp_short) == 1
    assert closed_tp_short[0].exit_reason == "TAKE_PROFIT"
    assert closed_tp_short[0].trade_win is True


def test_gap_executes_on_observable_price_without_invented_level():
    """Item 4: Jumps beyond SL/TP execute on first observable price with adverse slippage and data_gap=True."""
    cfg = PaperCostConfig(
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        source="test",
        effective_at="2026-01-01T00:00:00Z",
    )
    cm = CostModel(cfg)
    pm = PositionManager(cost_model=cm)

    # LONG position with SL at 95.0, TP at 110.0
    pos_long = create_sample_position(side="LONG", entry_price=100.0, sl=95.0, tp=110.0)
    pm.add_position(pos_long)

    # Market jumps directly to 92.0 (below 95.0 SL)
    closed = pm.on_tick(T=2000, p=92.0, q=1.0, m=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP_LOSS"
    assert closed[0].data_gap is True
    # Executed on observable price 92.0 with 1 bps adverse slippage: 92.0 * (1 - 0.0001) = 91.9908
    assert pytest.approx(closed[0].exit_price, rel=1e-5) == 91.9908
    assert closed[0].exit_price != 95.0  # NOT invented fill at exact SL


def test_sl_tp_chronological_order():
    """Invariant 10: TP/SL follow strict chronological order of ticks without OHLC aggregation."""
    pm = PositionManager()
    pos = create_sample_position(side="LONG", entry_price=100.0, sl=95.0, tp=110.0)
    pm.add_position(pos)

    # Tick 1 at T=1500 drops to 94.0 -> SL is touched first!
    closed = pm.on_tick(T=1500, p=94.0, q=1.0, m=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP_LOSS"
    assert closed[0].trade_win is False
    assert pm.has_open_position("c1", "flow_v1", "BTCUSDT") is False

    # Subsequent tick rising to 112.0 cannot trigger TP because position is already closed
    closed_subsequent = pm.on_tick(T=1600, p=112.0, q=1.0, m=False)
    assert len(closed_subsequent) == 0


def test_horizon_expiry_on_first_tick_at_or_after_deadline():
    """Horizon closes position at market on the very first tick where T >= deadline."""
    pm = PositionManager()
    pos = create_sample_position(horizon_s=10)
    pm.add_position(pos)

    assert len(pm.on_tick(T=10_999, p=102.0, q=1.0, m=False)) == 0

    closed = pm.on_tick(T=11_000, p=102.0, q=1.0, m=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "HORIZON_EXPIRY"


def test_unknown_data_gap_when_no_ticks_past_grace():
    """If no ticks arrive before horizon + grace period, position closes as UNKNOWN_DATA_GAP with None PnL."""
    cfg = PositionConfig(horizon_grace_ms=5000)
    pm = PositionManager(config=cfg)
    pos = create_sample_position(horizon_s=10)
    pm.add_position(pos)

    closed = pm.check_expired_positions(current_ts_ms=16_001)
    assert len(closed) == 1
    assert closed[0].exit_reason == "UNKNOWN_DATA_GAP"
    assert closed[0].gross_pnl_usdt is None
    assert closed[0].net_pnl_usdt is None
    assert closed[0].trade_win is None
    assert closed[0].trade_direction_profitable is None
    assert closed[0].costs_complete is False
    assert closed[0].data_gap is True


def test_mae_and_mfe_tracking():
    """MAE and MFE track worst adverse and best favorable excursion in bps."""
    pm = PositionManager()
    pos = create_sample_position(side="LONG", entry_price=100.0, sl=80.0, tp=120.0)
    pm.add_position(pos)

    pm.on_tick(T=2000, p=98.0, q=1.0, m=False)
    assert pytest.approx(pos.mae_bps, rel=1e-3) == 200.0
    assert pos.mfe_bps == 0.0

    pm.on_tick(T=3000, p=105.0, q=1.0, m=False)
    assert pytest.approx(pos.mae_bps, rel=1e-3) == 200.0
    assert pytest.approx(pos.mfe_bps, rel=1e-3) == 500.0


def test_close_all_kill_switch():
    """close_all() terminates all positions at market on the next tick with exit_reason=KILL_SWITCH."""
    pm = PositionManager()
    pos = create_sample_position()
    pm.add_position(pos)

    pm.close_all(reason="MANUAL_KILL_SWITCH")

    closed = pm.on_tick(T=2000, p=101.0, q=1.0, m=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "KILL_SWITCH"
    assert pm.has_open_position("c1", "flow_v1", "BTCUSDT") is False
