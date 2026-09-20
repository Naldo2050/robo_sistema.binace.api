# tests/unit/paper_trading/test_cost_model.py
"""Unit tests for the paper trading cost model."""

import pytest

from paper_trading.contracts import PaperCostConfig
from paper_trading.cost_model import CostModel, count_funding_crossings, PnLBreakdown


def test_round_trip_fees():
    """Taker fees applied on entry and exit."""
    cfg = PaperCostConfig(
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        source="test_vip0",
        effective_at="2026-01-01T00:00:00Z",
    )
    cm = CostModel(cfg)

    # Entry: notional 1000 USDT -> fee = 1000 * 0.0005 = 0.50 USDT
    fill_p, qty, entry_fee, entry_slip = cm.calculate_entry_fill("LONG", raw_price=100.0, notional_usdt=1000.0)
    assert pytest.approx(entry_fee, rel=1e-5) == 0.50

    # Exit at 110.0 (TAKE_PROFIT taker) -> notional = 110.0 * qty -> fee taker
    exit_p, exit_notional, exit_fee, exit_slip = cm.calculate_exit_fill("LONG", "TAKE_PROFIT", raw_price=110.0, quantity=qty, is_maker=False)
    assert pytest.approx(exit_fee, rel=1e-5) == exit_notional * 0.0005

    # Exit maker fee test
    _, exit_maker_notional, exit_maker_fee, _ = cm.calculate_exit_fill("LONG", "TAKE_PROFIT", raw_price=110.0, quantity=qty, is_maker=True)
    assert pytest.approx(exit_maker_fee, rel=1e-5) == exit_maker_notional * 0.0002


def test_slippage_is_always_adverse():
    """Invariant 7: Adverse slippage makes LONG buy higher and SHORT sell lower on entry, and mirrored on exit."""
    cfg = PaperCostConfig(
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
        entry_slippage_bps=1.0,
        exit_slippage_bps=1.0,
        source="test",
        effective_at="2026-01-01T00:00:00Z",
    )
    cm = CostModel(cfg)

    raw = 100.0
    long_p, _, _, _ = cm.calculate_entry_fill("LONG", raw_price=raw, notional_usdt=100.0)
    short_p, _, _, _ = cm.calculate_entry_fill("SHORT", raw_price=raw, notional_usdt=100.0)

    # LONG fills higher than raw price
    assert long_p > raw
    assert pytest.approx(long_p, rel=1e-5) == 100.01

    # SHORT fills lower than raw price
    assert short_p < raw
    assert pytest.approx(short_p, rel=1e-5) == 99.99

    # Exit with adverse slippage:
    # LONG exits lower (sells lower)
    long_exit_p, _, _, _ = cm.calculate_exit_fill("LONG", "STOP_LOSS", raw_price=raw, quantity=1.0)
    assert long_exit_p < raw
    assert pytest.approx(long_exit_p, rel=1e-5) == 99.99

    # SHORT exits higher (buys higher)
    short_exit_p, _, _, _ = cm.calculate_exit_fill("SHORT", "STOP_LOSS", raw_price=raw, quantity=1.0)
    assert short_exit_p > raw
    assert pytest.approx(short_exit_p, rel=1e-5) == 100.01


def test_fees_strictly_reduce_pnl():
    """Invariant 8: Fees and costs always strictly reduce net PnL below gross PnL when total_cost_bps > 0."""
    cm = CostModel()
    pnl: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.5,
        exit_fee_usdt=0.5,
        entry_slippage_usdt=0.1,
        exit_slippage_usdt=0.1,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )
    assert pnl.gross_pnl_usdt == 50.0
    assert pnl.net_pnl_usdt is not None
    assert pnl.net_pnl_usdt < pnl.gross_pnl_usdt
    assert pnl.net_pnl_bps is not None
    assert pnl.gross_pnl_bps is not None
    assert pnl.net_pnl_bps < pnl.gross_pnl_bps
    assert pnl.fees_usdt > 0.0


def test_zero_costs_mode_invariants():
    """Item 7: When total_cost_bps == 0, net_pnl_bps == gross_pnl_bps exactly."""
    zero_cfg = PaperCostConfig(
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        entry_slippage_bps=0.0,
        exit_slippage_bps=0.0,
        source="zero_cost",
        effective_at="2026-01-01T00:00:00Z",
    )
    cm = CostModel(zero_cfg)

    # Check fills have zero slippage and zero fee
    entry_p, qty, fee_in, slip_in = cm.calculate_entry_fill("LONG", raw_price=100.0, notional_usdt=1000.0)
    assert entry_p == 100.0
    assert fee_in == 0.0
    assert slip_in == 0.0

    exit_p, notional_out, fee_out, slip_out = cm.calculate_exit_fill("LONG", "STOP_LOSS", raw_price=95.0, quantity=qty)
    assert exit_p == 95.0
    assert fee_out == 0.0
    assert slip_out == 0.0

    pnl: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=qty,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )
    assert pnl.net_pnl_usdt == pnl.gross_pnl_usdt
    assert pnl.net_pnl_bps == pnl.gross_pnl_bps
    assert pnl.fees_bps == 0.0
    assert pnl.slippage_bps == 0.0


def test_funding_only_at_boundaries():
    """Invariant 9: Funding is applied strictly only when crossing real boundaries (no linear pro-rata)."""
    t_open = 1_789_804_200_000
    t_close_no_cross = 1_789_804_740_000
    assert count_funding_crossings(t_open, t_close_no_cross) == 0

    t_close_cross = 1_789_805_100_000
    assert count_funding_crossings(t_open, t_close_cross) == 1

    cm = CostModel()
    pnl_long: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=100.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.5,
        exit_fee_usdt=0.5,
        entry_slippage_usdt=0.1,
        exit_slippage_usdt=0.1,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close_cross,
        funding_rate=0.0001,
    )
    assert pytest.approx(pnl_long.funding_usdt, rel=1e-5) == -0.10
    assert pnl_long.costs_complete is True

    pnl_short: PnLBreakdown = cm.calculate_trade_pnl(
        side="SHORT",
        raw_entry_price=100.0,
        raw_exit_price=100.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.5,
        exit_fee_usdt=0.5,
        entry_slippage_usdt=0.1,
        exit_slippage_usdt=0.1,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close_cross,
        funding_rate=0.0001,
    )
    assert pytest.approx(pnl_short.funding_usdt, rel=1e-5) == 0.10
    assert pnl_short.costs_complete is True


def test_costs_complete_flag_consistency():
    """Invariant 15: If position crosses funding boundary but rate is None, funding is None and costs_complete is False."""
    t_open = 1_789_804_200_000
    t_close_cross = 1_789_805_100_000

    cm = CostModel()
    pnl: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.5,
        exit_fee_usdt=0.5,
        entry_slippage_usdt=0.1,
        exit_slippage_usdt=0.1,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close_cross,
        funding_rate=None,
    )
    assert pnl.funding_usdt is None
    assert pnl.funding_bps is None
    assert pnl.net_pnl_usdt is None
    assert pnl.net_pnl_bps is None
    assert pnl.trade_win is None
    assert pnl.pnl_R is None
    assert pnl.costs_complete is False
    assert pnl.trade_direction_profitable is True


def test_long_short_pre_cost_symmetry():
    """Invariant 11: LONG and SHORT gross PnL are mathematically symmetric before costs."""
    cm = CostModel()
    pnl_long: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )

    pnl_short: PnLBreakdown = cm.calculate_trade_pnl(
        side="SHORT",
        raw_entry_price=105.0,
        raw_exit_price=100.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )

    assert pnl_long.gross_pnl_usdt == 50.0
    assert pnl_short.gross_pnl_usdt == 50.0
    assert pnl_long.gross_pnl_bps == pnl_short.gross_pnl_bps == 500.0


def test_long_short_pre_cost_symmetry_same_p0():
    """
    GAP-2: LONG and SHORT with identical initial price P0 = 100.0 and mirrored percentage moves.

    Formulas and Semantics:
    - P0 = 100.0, quantity = 10.0, notional_usdt = 1000.0 (entry notional: P0 * quantity)
    - fees = 0, entry_slippage = 0, exit_slippage = 0, funding = 0
    - LONG:
        entry = 100.0, exit = 105.0 (+5% price move)
        gross_pnl_usdt = (exit - entry) * quantity = (105 - 100) * 10 = +50.0 USDT
        gross_pnl_bps  = (gross_pnl_usdt / notional_usdt) * 10,000 = +500.0 bps
    - SHORT (favorable):
        entry = 100.0, exit = 95.0 (-5% price move)
        gross_pnl_usdt = (entry - exit) * quantity = (100 - 95) * 10 = +50.0 USDT
        gross_pnl_bps  = (gross_pnl_usdt / notional_usdt) * 10,000 = +500.0 bps
    - SHORT (adverse, detecting incorrect directional sign inversion):
        entry = 100.0, exit = 105.0 (+5% price move, adverse to short)
        gross_pnl_usdt = (entry - exit) * quantity = (100 - 105) * 10 = -50.0 USDT
        gross_pnl_bps  = (gross_pnl_usdt / notional_usdt) * 10,000 = -500.0 bps
    """
    cm = CostModel()

    pnl_long: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )

    pnl_short_favorable: PnLBreakdown = cm.calculate_trade_pnl(
        side="SHORT",
        raw_entry_price=100.0,
        raw_exit_price=95.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )

    pnl_short_adverse: PnLBreakdown = cm.calculate_trade_pnl(
        side="SHORT",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=1000.0,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        funding_rate=None,
    )

    # 1. Exact mathematical symmetry between LONG and SHORT for mirrored move
    assert pnl_long.gross_pnl_usdt == pnl_short_favorable.gross_pnl_usdt == 50.0
    assert pnl_long.gross_pnl_bps == pnl_short_favorable.gross_pnl_bps == 500.0
    assert pnl_long.trade_direction_profitable is True
    assert pnl_short_favorable.trade_direction_profitable is True

    # 2. Strict detection of adverse SHORT move (detects incorrect sign inversions)
    assert pnl_short_adverse.gross_pnl_usdt == -50.0
    assert pnl_short_adverse.gross_pnl_bps == -500.0
    assert pnl_short_adverse.trade_direction_profitable is False


def test_multi_boundary_funding_crossings():
    """
    GAP-3: Position spans multiple consecutive UTC funding boundaries (e.g. 08:00, 16:00, 00:00).
    Verifies discrete accumulation without linear pro-rata, opposite signs for LONG and SHORT,
    and proper invalidation when funding rate is missing.
    """
    import datetime

    # Open at 2026-01-01 07:00:00 UTC (before 08:00 UTC boundary)
    dt_open = datetime.datetime(2026, 1, 1, 7, 0, 0, tzinfo=datetime.timezone.utc)
    # Close at 2026-01-02 01:00:00 UTC (after crossing 08:00, 16:00, and 00:00 UTC boundaries)
    dt_close = datetime.datetime(2026, 1, 2, 1, 0, 0, tzinfo=datetime.timezone.utc)

    t_open = int(dt_open.timestamp() * 1000)
    t_close = int(dt_close.timestamp() * 1000)

    # 1. Verify exact crossings count == 3
    crossings = count_funding_crossings(t_open, t_close, funding_hours=(0, 8, 16))
    assert crossings == 3

    cm = CostModel()
    notional = 1000.0
    rate = 0.0001  # 1 bps per boundary => 0.10 USDT per boundary
    single_boundary_fee = rate * notional  # 0.10 USDT

    # 2. LONG position: pays funding when rate > 0 => 3 * -0.10 = -0.30 USDT
    pnl_long: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=100.0,
        quantity=10.0,
        notional_usdt=notional,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close,
        funding_rate=rate,
    )
    assert pytest.approx(pnl_long.funding_usdt, rel=1e-5) == -(single_boundary_fee * 3)
    assert pytest.approx(pnl_long.funding_usdt, rel=1e-5) == -0.30
    assert pnl_long.costs_complete is True

    # 3. SHORT position: receives funding when rate > 0 => 3 * +0.10 = +0.30 USDT
    pnl_short: PnLBreakdown = cm.calculate_trade_pnl(
        side="SHORT",
        raw_entry_price=100.0,
        raw_exit_price=100.0,
        quantity=10.0,
        notional_usdt=notional,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close,
        funding_rate=rate,
    )
    assert pytest.approx(pnl_short.funding_usdt, rel=1e-5) == (single_boundary_fee * 3)
    assert pytest.approx(pnl_short.funding_usdt, rel=1e-5) == 0.30
    assert pnl_short.costs_complete is True

    # 4. Multi-boundary crossing with missing funding rate => strict invalidation
    pnl_missing: PnLBreakdown = cm.calculate_trade_pnl(
        side="LONG",
        raw_entry_price=100.0,
        raw_exit_price=105.0,
        quantity=10.0,
        notional_usdt=notional,
        entry_fee_usdt=0.0,
        exit_fee_usdt=0.0,
        entry_slippage_usdt=0.0,
        exit_slippage_usdt=0.0,
        opened_ts_ms=t_open,
        closed_ts_ms=t_close,
        funding_rate=None,
    )
    assert pnl_missing.funding_bps is None
    assert pnl_missing.funding_usdt is None
    assert pnl_missing.net_pnl_bps is None
    assert pnl_missing.net_pnl_usdt is None
    assert pnl_missing.trade_win is None
    assert pnl_missing.costs_complete is False
    assert pnl_missing.trade_direction_profitable is True
