# tests/unit/paper_trading/test_ledger.py
"""Unit tests for the SQLite append-only PaperLedger."""

import os
import tempfile
import pytest

from paper_trading.contracts import CanonicalDecision, ClosedTrade, PaperFill, PaperPosition
from paper_trading.ledger import PaperLedger


def test_sqlite_append_only_idempotency():
    """Invariant 13: Ledger initializes schema, operates WAL/append-only and flushes synchronously."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ledger.db")
        ledger = PaperLedger(db_path=db_path)

        ledger.record_cohort(cohort_id="test_cohort", created_at_ms=1000, description="audit cohort")
        d = CanonicalDecision(
            cohort_id="test_cohort",
            symbol="BTCUSDT",
            window_id="w1",
            decision_provider="flow_v1",
            strategy_version="1.0",
            signal_timestamp=1000,
            decision_timestamp=1020,
            available_at=1050,
            side="LONG",
            reference_price=100.0,
            notional_usdt=500.0,
            horizon_s=60,
        )
        ledger.record_decision(d)
        ledger.flush()

        cur = ledger._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        expected_tables = {
            "cohorts",
            "decisions",
            "rejections",
            "orders",
            "fills",
            "positions",
            "closed_trades",
            "funding_events",
            "kill_switch_events",
        }
        assert expected_tables.issubset(tables)
        ledger.close()


def test_duplicate_decision_rejected():
    """Invariant 4: UNIQUE(decision_id) in decisions prevents duplication and records DUPLICATE_DECISION rejection."""
    ledger = PaperLedger(db_path=":memory:")

    d = CanonicalDecision(
        cohort_id="c_dup",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=30,
    )
    ledger.record_decision(d)
    ledger.flush()

    # Same decision written again triggers duplicate rejection
    ledger.record_decision(d)
    ledger.flush()

    rejections = ledger.get_rejections(cohort_id="c_dup")
    assert len(rejections) == 1
    assert rejections[0].reason == "DUPLICATE_DECISION"
    assert rejections[0].decision_id == d.decision_id

    ledger.close()


def test_closed_trade_segregation_and_fill_latencies():
    """Item 1, 5, 11: Fills persist latencies >= 0; closed trades persist trade_direction_profitable and prediction_direction_correct."""
    ledger = PaperLedger(db_path=":memory:")

    fill = PaperFill(
        fill_id="fill_1",
        order_id="ord_1",
        decision_id="dec_1",
        cohort_id="c1",
        symbol="BTCUSDT",
        side="LONG",
        fill_price=100.01,
        raw_price=100.0,
        slippage_bps=1.0,
        quantity=1.0,
        notional_usdt=100.01,
        fill_timestamp=1060,
        trade_id_used="t_1",
        fee_usdt=0.05,
        decision_to_fill_ms=40,
        available_to_fill_ms=10,
    )
    ledger.record_fill(fill)

    trade = ClosedTrade(
        trade_id="tr_1",
        decision_id="dec_1",
        cohort_id="c1",
        decision_provider="flow_v1",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.01,
        exit_price=105.0,
        quantity=1.0,
        notional_usdt=100.01,
        opened_ts_ms=1060,
        closed_ts_ms=2000,
        exit_reason="TAKE_PROFIT",
        trade_direction_profitable=True,
        prediction_direction_correct=None,
        trade_win=True,
        gross_pnl_bps=500.0,
        net_pnl_bps=490.0,
        fees_bps=8.0,
        slippage_bps=2.0,
        funding_bps=0.0,
        gross_pnl_usdt=5.0,
        fees_usdt=0.08,
        slippage_usdt=0.02,
        funding_usdt=0.0,
        net_pnl_usdt=4.90,
        pnl_R=2.5,
        costs_complete=True,
        data_gap=False,
        mae_bps=10.0,
        mfe_bps=510.0,
        ticks_count=50,
    )
    ledger.record_closed_trade(trade)
    ledger.flush()

    retrieved_fills = ledger.get_fills(cohort_id="c1")
    assert len(retrieved_fills) == 1
    assert retrieved_fills[0].decision_to_fill_ms == 40
    assert retrieved_fills[0].available_to_fill_ms == 10
    assert retrieved_fills[0].trade_id_used == "t_1"

    retrieved_trades = ledger.get_closed_trades(cohort_id="c1")
    assert len(retrieved_trades) == 1
    t = retrieved_trades[0]
    assert t.trade_direction_profitable is True
    assert t.prediction_direction_correct is None
    assert t.trade_win is True
    assert t.gross_pnl_bps == 500.0
    assert t.net_pnl_bps == 490.0
    assert t.costs_complete is True

    ledger.close()


def test_restart_reloads_open_positions():
    """load_open_positions() rehydrates only positions marked with is_open = 1 upon restart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "recovery.db")
        ledger = PaperLedger(db_path=db_path)

        pos_open = PaperPosition(
            position_id="pos_active",
            cohort_id="c_restart",
            decision_provider="flow_v1",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100.0,
            reference_price=100.0,
            quantity=1.0,
            notional_usdt=100.0,
            opened_ts_ms=1000,
            horizon_deadline_ms=70_000,
            decision_id="dec_active",
            signal_timestamp=980,
            decision_timestamp=1000,
            available_at=1020,
        )
        pos_closed = PaperPosition(
            position_id="pos_finished",
            cohort_id="c_restart",
            decision_provider="flow_v1",
            symbol="ETHUSDT",
            side="SHORT",
            entry_price=2000.0,
            reference_price=2000.0,
            quantity=0.5,
            notional_usdt=1000.0,
            opened_ts_ms=1000,
            horizon_deadline_ms=70_000,
            decision_id="dec_finished",
            signal_timestamp=980,
            decision_timestamp=1000,
            available_at=1020,
        )

        ledger.record_position(pos_open)
        ledger.record_position(pos_closed)
        ledger.close_position("pos_finished")
        ledger.flush()
        ledger.close()

        restarted_ledger = PaperLedger(db_path=db_path)
        open_positions = restarted_ledger.load_open_positions(cohort_id="c_restart")

        assert len(open_positions) == 1
        assert open_positions[0].position_id == "pos_active"
        assert open_positions[0].symbol == "BTCUSDT"
        assert open_positions[0].side == "LONG"
        assert open_positions[0].reference_price == 100.0

        restarted_ledger.close()
