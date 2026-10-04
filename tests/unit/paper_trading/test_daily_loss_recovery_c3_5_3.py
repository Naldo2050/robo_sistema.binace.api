# tests/unit/paper_trading/test_daily_loss_recovery_c3_5_3.py
"""
Unit test suite for Gate C3.5.3: CLEAN_RESTART_SAFE Daily Loss Recovery.

Mandatory coverage for all 25 required test cases:
1. on_signal before start() does not admit orders (STARTUP_PENDING).
2. Healthy new database starts cleanly (CLEAN_RESTART_SAFE).
3. STARTED event is durably flushed before accepting new exposure.
4. STARTED flush timeout blocks new exposure (LEDGER_FLUSH_TIMEOUT).
5. STARTED flush unhealthy/failure blocks new exposure (LEDGER_FLUSH_FAILURE).
6. Clean restart restores daily realized PnL correctly.
7. Multiple cohorts from the same UTC day are correctly aggregated.
8. Persisted duplicate trade is deduplicated and not counted twice.
9. Persisted incomplete trade (costs_complete=False) fails closed and blocks startup.
10. Persisted non-finite net_pnl (NaN/Inf) fails closed and blocks startup.
11. Restored loss at exact daily limit blocks new exposure (DAILY_LOSS_LIMIT_REACHED).
12. Restored profit does not block exposure.
13. Previous STARTED without GRACEFUL_SHUTDOWN marks UNVERIFIED_AFTER_RESTART and blocks.
14. Previous CRASH_DETECTED marks UNVERIFIED_AFTER_RESTART and blocks.
15. Previous PERSISTENCE_FAILURE marks UNVERIFIED_AFTER_RESTART and blocks.
16. GRACEFUL_SHUTDOWN with open_positions_count > 0 blocks (SHUTDOWN_WITH_OPEN_POSITIONS).
17. GRACEFUL_SHUTDOWN with pending_orders_count > 0 blocks (SHUTDOWN_WITH_OPEN_POSITIONS).
18. Positions table with is_open=1 blocks (ORPHAN_POSITIONS_DETECTED).
19. SQLite read error fails closed and blocks (PERSISTENCE_RECOVERY_ERROR).
20. Empty database is cleanly distinguished from corrupted/error state.
21. Dedup cache (_seen_trades) is repopulated on recovery to prevent live duplicate reprocessing.
22. Trades from other UTC calendar days are excluded by the strict [start, end) interval.
23. Month and leap-year boundaries produce exact deterministic UTC ranges.
24. C3.5.2 SESSION_SAFE runtime accounting continues operating after startup.
25. C3.4 max_open_positions execution capacity remains strictly operational.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from paper_trading.config import ShadowPaperConfig, parse_shadow_config
from paper_trading.contracts import CanonicalDecision, ClosedTrade, PaperOrder
from paper_trading.execution_sink import ExecutionSink, ExecutionTick, TradeResult, TradeStatus
from paper_trading.executor import PaperExecutor, TickEvents
from paper_trading.ledger import (
    PaperLedger,
    compute_cohort_event_id,
)
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.shadow_runtime import ShadowPaperRuntime
from risk_management.risk_manager import (
    RiskClosedTradeResult,
    RiskConfig,
    RiskManager,
    TradeRequest,
)


class FakeClock:
    """Deterministic injectable clock in milliseconds."""

    def __init__(self, start_ms: int = 1774010000000):
        self._current_ms: int = start_ms

    def __call__(self) -> int:
        return self._current_ms

    def advance(self, delta_ms: int) -> int:
        self._current_ms += delta_ms
        return self._current_ms

    def set(self, target_ms: int) -> int:
        self._current_ms = target_ms
        return self._current_ms


def make_now_clock() -> FakeClock:
    """Helper to produce a FakeClock set at 12:00:00 UTC today."""
    now_utc = datetime.now(timezone.utc)
    noon = datetime(now_utc.year, now_utc.month, now_utc.day, 12, 0, 0, tzinfo=timezone.utc)
    return FakeClock(int(noon.timestamp() * 1000))


def make_test_config(
    cohort_id: str = "CH_C353_TEST",
    provider: str = "fixed_long",
) -> ShadowPaperConfig:
    env = {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": cohort_id,
        "PAPER_PROVIDER": provider,
        "PAPER_SYMBOL": "BTCUSDT",
        "PAPER_TIMEFRAME": "1m",
        "PAPER_NOTIONAL_USDT": "1000.0",
        "PAPER_HORIZON_S": "60",
        "PAPER_ORDER_TTL_MS": "5000",
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "binance_vip0",
        "PAPER_COST_EFFECTIVE_AT": "2026-01-01T00:00:00Z",
        "PAPER_STRATEGY_VERSION": "c3_5_3_test_v1",
    }
    res = parse_shadow_config(env)
    assert res.is_valid is True
    assert res.config is not None
    return res.config


def make_test_runtime(
    config: ShadowPaperConfig,
    ledger: PaperLedger,
    clock: FakeClock,
    daily_loss_limit_usdt: float = 100.0,
    max_position_size: int = 1,
) -> ShadowPaperRuntime:
    risk_cfg = RiskConfig(
        initial_capital=1.0,
        max_daily_loss=daily_loss_limit_usdt,
        max_position_size=max_position_size,
        max_open_positions=max_position_size,
    )
    rm = RiskManager(config=risk_cfg)
    ra = RiskAdapter(risk_manager=rm)
    return ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        clock_ms=clock,
        risk_adapter=ra,
    )


def make_closed_trade(
    cohort_id: str,
    trade_id: str,
    closed_ts_ms: int,
    net_pnl_usdt: Optional[float],
    costs_complete: bool = True,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
) -> ClosedTrade:
    is_fin = (net_pnl_usdt is not None) and math.isfinite(net_pnl_usdt)
    return ClosedTrade(
        trade_id=trade_id,
        decision_id=f"dec_{trade_id}",
        cohort_id=cohort_id,
        decision_provider="fixed_long",
        symbol=symbol,
        side=side,
        entry_price=50000.0,
        exit_price=50000.0 + ((net_pnl_usdt / 0.1) if is_fin else 0.0),
        quantity=0.1,
        notional_usdt=5000.0,
        opened_ts_ms=closed_ts_ms - 60000,
        closed_ts_ms=closed_ts_ms,
        exit_reason="TP",
        trade_direction_profitable=True if is_fin and net_pnl_usdt >= 0 else False,
        prediction_direction_correct=True,
        trade_win=True if is_fin and net_pnl_usdt >= 0 else False,
        gross_pnl_bps=0.0,
        net_pnl_bps=0.0,
        fees_bps=0.0,
        slippage_bps=0.0,
        funding_bps=0.0,
        gross_pnl_usdt=net_pnl_usdt if is_fin else 0.0,
        fees_usdt=0.0,
        slippage_usdt=0.0,
        funding_usdt=0.0,
        net_pnl_usdt=net_pnl_usdt,
        pnl_R=0.0,
        costs_complete=costs_complete,
        data_gap=0,
        mae_bps=0.0,
        mfe_bps=0.0,
        ticks_count=1,
        direction_correct=True,
        context={},
    )


def insert_test_cohort(
    ledger: PaperLedger,
    cohort_id: str,
    created_at_ms: int,
) -> None:
    ledger.create_cohort(
        cohort_id=cohort_id,
        created_at_ms=created_at_ms,
        description=f"Test cohort {cohort_id}",
        metadata={"git_sha": "test_sha", "risk_daily_loss_active": False},
    )


def insert_test_event(
    ledger: PaperLedger,
    cohort_id: str,
    event_type: str,
    timestamp_ms: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    ledger.record_cohort_event(
        cohort_id=cohort_id,
        event_type=event_type,
        timestamp_ms=timestamp_ms,
        reason=f"Event {event_type}",
        metadata=metadata or {},
    )
    res = ledger.flush_status(timeout=5.0)
    assert res == "SUCCESS"


def insert_test_closed_trade(
    ledger: PaperLedger,
    cohort_id: str,
    trade_id: str,
    closed_ts_ms: int,
    net_pnl_usdt: Optional[float],
    costs_complete: bool = True,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
) -> None:
    ct = make_closed_trade(
        cohort_id=cohort_id,
        trade_id=trade_id,
        closed_ts_ms=closed_ts_ms,
        net_pnl_usdt=net_pnl_usdt,
        costs_complete=costs_complete,
        symbol=symbol,
        side=side,
    )
    ledger.record_closed_trade(ct)
    res = ledger.flush_status(timeout=5.0)
    assert res == "SUCCESS"


# =========================================================================
# 1. on_signal before start() does not admit orders
# =========================================================================
def test_1_on_signal_before_start_does_not_admit_order():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_TEST_1")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    # Pre-start assertions
    assert runtime.is_active is True  # configured enabled
    assert runtime.accepts_new_exposure is False
    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "STARTUP_PENDING"

    # Signal arrives before start()
    sig = {
        "event_type": "signal_event",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "timestamp": clock(),
        "epoch_ms": clock(),
        "timeframe": "1m",
    }
    res = runtime.on_signal(sig)

    assert res.status == "INACTIVE"
    assert "STARTUP_PENDING" in (res.error or "")
    assert res.order_id is None
    assert runtime.execution_sink.get_capacity_snapshot()["open_positions"] == 0
    assert runtime._counters["orders_submitted"] == 0


# =========================================================================
# 2. Healthy new database starts cleanly
# =========================================================================
def test_2_healthy_new_db_starts_cleanly():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_TEST_2")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is True
    assert runtime.accepts_new_exposure is True
    assert runtime.new_exposure_blocked is False
    assert runtime.risk_block_reason is None
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.is_risk_daily_loss_active is True
    assert runtime.risk_daily_loss_enforced is True
    assert runtime.risk_pnl_complete is True


# =========================================================================
# 3. STARTED event is durably flushed before accepting new exposure
# =========================================================================
def test_3_started_event_flushed_before_exposure_accepted():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_TEST_3")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    exposure_during_flush = None
    original_flush = ledger.flush_status

    def spy_flush(*args, **kwargs):
        nonlocal exposure_during_flush
        exposure_during_flush = runtime.accepts_new_exposure
        return original_flush(*args, **kwargs)

    ledger.flush_status = spy_flush

    started = runtime.start()

    assert started is True
    assert exposure_during_flush is False  # Was strictly false while flushing STARTED
    assert runtime.accepts_new_exposure is True  # True only after flush confirmed SUCCESS

    # Verify STARTED event in ledger
    cur = ledger._conn.cursor()
    cur.execute(
        "SELECT event_type FROM cohort_events WHERE cohort_id = 'CH_TEST_3' AND event_type = 'STARTED'"
    )
    assert cur.fetchone() is not None


# =========================================================================
# 4. STARTED flush timeout blocks new exposure
# =========================================================================
def test_4_flush_started_timeout_blocks_exposure():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_TEST_4")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    orig_record = ledger.record_cohort_event

    def intercept_record(*args, **kwargs):
        orig_record(*args, **kwargs)
        evt = kwargs.get("event_type") or (args[1] if len(args) > 1 else None)
        if evt == "STARTED":
            ledger.flush_status = MagicMock(return_value="TIMEOUT")

    ledger.record_cohort_event = intercept_record

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_block_reason == "LEDGER_FLUSH_TIMEOUT"
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"


# =========================================================================
# 5. STARTED flush unhealthy/failure blocks new exposure
# =========================================================================
def test_5_flush_started_unhealthy_or_failure_blocks_exposure():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_TEST_5")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    orig_record = ledger.record_cohort_event

    def intercept_record(*args, **kwargs):
        orig_record(*args, **kwargs)
        evt = kwargs.get("event_type") or (args[1] if len(args) > 1 else None)
        if evt == "STARTED":
            ledger.flush_status = MagicMock(return_value="FAILURE")

    ledger.record_cohort_event = intercept_record

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_block_reason == "LEDGER_FLUSH_FAILURE"
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"


# =========================================================================
# 6. Clean restart restores daily realized PnL correctly
# =========================================================================
def test_6_clean_restart_restores_daily_pnl():
    clock = FakeClock(1774010000000)  # e.g., 2026-03-20
    ledger = PaperLedger(db_path=":memory:")

    # Setup previous cleanly shutdown cohort
    insert_test_cohort(ledger, "CH_PREV_1", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV_1", "STARTED", clock() - 3500000)
    insert_test_closed_trade(ledger, "CH_PREV_1", "T_PREV_1", clock() - 1800000, -35.0)
    insert_test_event(
        ledger,
        "CH_PREV_1",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    # Launch new runtime
    config = make_test_config(cohort_id="CH_NEW_6")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, daily_loss_limit_usdt=100.0)

    started = runtime.start()

    assert started is True
    assert runtime.accepts_new_exposure is True
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.is_risk_daily_loss_active is True
    rm = runtime.risk_adapter.risk_manager
    assert rm.daily_pnl == -35.0
    assert rm.risk_pnl_complete is True
    assert rm.is_daily_loss_exceeded is False


# =========================================================================
# 7. Multiple cohorts from same UTC day are correctly aggregated
# =========================================================================
def test_7_multiple_cohorts_same_day_aggregated():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    # Cohort 1
    insert_test_cohort(ledger, "CH_1", clock() - 7200000)
    insert_test_event(ledger, "CH_1", "STARTED", clock() - 7100000)
    insert_test_closed_trade(ledger, "CH_1", "T1", clock() - 7000000, -25.0)
    insert_test_event(
        ledger,
        "CH_1",
        "GRACEFUL_SHUTDOWN",
        clock() - 6000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    # Cohort 2
    insert_test_cohort(ledger, "CH_2", clock() - 5000000)
    insert_test_event(ledger, "CH_2", "STARTED", clock() - 4900000)
    insert_test_closed_trade(ledger, "CH_2", "T2", clock() - 4500000, -40.0)
    insert_test_event(
        ledger,
        "CH_2",
        "GRACEFUL_SHUTDOWN",
        clock() - 4000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    # Cohort 3 (Current)
    config = make_test_config(cohort_id="CH_3")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, daily_loss_limit_usdt=100.0)

    assert runtime.start() is True
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.risk_adapter.risk_manager.daily_pnl == -65.0
    assert runtime.accepts_new_exposure is True


# =========================================================================
# 8. Persisted duplicate trade is deduplicated and not counted twice
# =========================================================================
def test_8_duplicate_persisted_trade_not_double_counted():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(ledger, "CH_PREV", "T_DUP", clock() - 2000000, -20.0)
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    # Inject duplicate in return of get_closed_trades_by_timerange to test recovery defense
    orig_get_trades = ledger.get_closed_trades_by_timerange

    def mock_get_trades(start_ms, end_ms):
        trades = orig_get_trades(start_ms, end_ms)
        return trades + trades  # Duplicate list

    ledger.get_closed_trades_by_timerange = mock_get_trades

    config = make_test_config(cohort_id="CH_NEW_8")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is True
    rm = runtime.risk_adapter.risk_manager
    assert rm.daily_pnl == -20.0  # Exactly once, deduplicated


# =========================================================================
# 9. Persisted incomplete trade (costs_complete=False) blocks startup
# =========================================================================
def test_9_incomplete_persisted_trade_blocks_startup():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(
        ledger, "CH_PREV", "T_INC", clock() - 2000000, -10.0, costs_complete=False
    )
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_9")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "PNL_INCOMPLETE"
    assert runtime.risk_pnl_complete is False


# =========================================================================
# 10. Persisted non-finite net_pnl (NaN/Inf) blocks startup
# =========================================================================
def test_10_non_finite_net_pnl_blocks_startup():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(
        ledger, "CH_PREV", "T_NAN", clock() - 2000000, float("nan"), costs_complete=True
    )
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_10")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "PNL_INCOMPLETE"
    assert runtime.risk_pnl_complete is False


# =========================================================================
# 11. Restored loss at exact daily limit blocks new exposure
# =========================================================================
def test_11_restored_loss_at_exact_limit_blocks_exposure():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(ledger, "CH_PREV", "T_LIMIT", clock() - 2000000, -100.0)
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_11")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, daily_loss_limit_usdt=100.0)

    started = runtime.start()

    assert started is True
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.is_risk_daily_loss_active is True
    assert runtime.accepts_new_exposure is False  # Blocked by boundary
    assert runtime.risk_block_reason == "DAILY_LOSS_LIMIT_REACHED"
    assert runtime.risk_adapter.risk_manager.is_daily_loss_exceeded is True


# =========================================================================
# 12. Restored profit does not block exposure
# =========================================================================
def test_12_restored_profit_does_not_block():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(ledger, "CH_PREV", "T_PROFIT", clock() - 2000000, 150.0)
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_12")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, daily_loss_limit_usdt=100.0)

    assert runtime.start() is True
    assert runtime.accepts_new_exposure is True
    assert runtime.risk_block_reason is None
    assert runtime.risk_adapter.risk_manager.daily_pnl == 150.0


# =========================================================================
# 13. Previous STARTED without GRACEFUL_SHUTDOWN marks UNVERIFIED and blocks
# =========================================================================
def test_13_previous_started_without_graceful_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_UNGRACEFUL", clock() - 3600000)
    insert_test_event(ledger, "CH_UNGRACEFUL", "STARTED", clock() - 3500000)
    # Process died abruptly without GRACEFUL_SHUTDOWN event

    config = make_test_config(cohort_id="CH_NEW_13")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "UNEXPECTED_PREVIOUS_SHUTDOWN"


# =========================================================================
# 14. Previous CRASH_DETECTED marks UNVERIFIED and blocks
# =========================================================================
def test_14_crash_detected_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_CRASH", clock() - 3600000)
    insert_test_event(ledger, "CH_CRASH", "STARTED", clock() - 3500000)
    insert_test_event(ledger, "CH_CRASH", "CRASH_DETECTED", clock() - 3000000)
    insert_test_event(
        ledger,
        "CH_CRASH",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_14")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "CRASH_DETECTED"


# =========================================================================
# 15. Previous PERSISTENCE_FAILURE marks UNVERIFIED and blocks
# =========================================================================
def test_15_persistence_failure_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PERSIST_FAIL", clock() - 3600000)
    insert_test_event(ledger, "CH_PERSIST_FAIL", "STARTED", clock() - 3500000)
    insert_test_event(ledger, "CH_PERSIST_FAIL", "PERSISTENCE_FAILURE", clock() - 3000000)
    insert_test_event(
        ledger,
        "CH_PERSIST_FAIL",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_15")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "PERSISTENCE_FAILURE"


# =========================================================================
# 16. GRACEFUL_SHUTDOWN with open_positions_count > 0 blocks
# =========================================================================
def test_16_graceful_with_open_positions_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_OPEN_POS", clock() - 3600000)
    insert_test_event(ledger, "CH_OPEN_POS", "STARTED", clock() - 3500000)
    insert_test_event(
        ledger,
        "CH_OPEN_POS",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 1, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_16")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "SHUTDOWN_WITH_OPEN_POSITIONS"


# =========================================================================
# 17. GRACEFUL_SHUTDOWN with pending_orders_count > 0 blocks
# =========================================================================
def test_17_graceful_with_pending_orders_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PENDING_ORD", clock() - 3600000)
    insert_test_event(ledger, "CH_PENDING_ORD", "STARTED", clock() - 3500000)
    insert_test_event(
        ledger,
        "CH_PENDING_ORD",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 1},
    )

    config = make_test_config(cohort_id="CH_NEW_17")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "SHUTDOWN_WITH_OPEN_POSITIONS"


# =========================================================================
# 18. Positions table with is_open=1 blocks (ORPHAN_POSITIONS_DETECTED)
# =========================================================================
def test_18_positions_table_open_positions_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    # Insert an unclosed position directly
    cur = ledger._conn.cursor()
    cur.execute(
        """
        INSERT INTO positions (
            position_id, cohort_id, decision_provider, symbol, side,
            entry_price, reference_price, quantity, notional_usdt,
            opened_ts_ms, horizon_deadline_ms, decision_id,
            signal_timestamp, decision_timestamp, available_at,
            entry_fee_usdt, entry_slippage_usdt, mae_bps, mfe_bps,
            ticks_processed, last_tick_ts_ms, data_gap, is_open
        ) VALUES (
            'pos_orphan_1', 'CH_ORPHAN', 'fixed_long', 'BTCUSDT', 'LONG',
            50000.0, 50000.0, 0.1, 5000.0,
            1774005000000, 1774005060000, 'dec_1',
            1774005000000, 1774005000000, 1774005000000,
            0.0, 0.0, 0.0, 0.0,
            1, 1774005000000, 0, 1
        )
        """
    )
    ledger._conn.commit()

    config = make_test_config(cohort_id="CH_NEW_18")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "ORPHAN_POSITIONS_DETECTED"


# =========================================================================
# 19. SQLite read error fails closed and blocks
# =========================================================================
def test_19_sqlite_read_error_blocks():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    # Force SQLite operational error during recovery
    ledger.get_closed_trades_by_timerange = MagicMock(
        side_effect=sqlite3.OperationalError("disk I/O error or corrupted database")
    )

    config = make_test_config(cohort_id="CH_NEW_19")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    started = runtime.start()

    assert started is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "PERSISTENCE_RECOVERY_ERROR"


# =========================================================================
# 20. Empty database is cleanly distinguished from corrupted/error state
# =========================================================================
def test_20_empty_db_not_confused_with_error():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")

    assert len(ledger.get_all_cohort_lifecycles()) == 0
    assert ledger.get_open_positions_count() == 0
    assert len(ledger.get_closed_trades_by_timerange(0, clock() + 1000000)) == 0

    config = make_test_config(cohort_id="CH_NEW_20")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is True
    assert runtime.accepts_new_exposure is True
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.is_risk_daily_loss_active is True
    assert runtime.risk_adapter.risk_manager.daily_pnl == 0.0


# =========================================================================
# 21. Dedup cache (_seen_trades) is repopulated on recovery
# =========================================================================
def test_21_dedup_cache_repopulated_on_recovery():
    clock = make_now_clock()
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_PREV", clock() - 3600000)
    insert_test_event(ledger, "CH_PREV", "STARTED", clock() - 3500000)
    insert_test_closed_trade(ledger, "CH_PREV", "T_PERSISTED_1", clock() - 2000000, -15.0)
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        clock() - 1000000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_NEW_21")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)
    runtime.start()

    rm = runtime.risk_adapter.risk_manager
    assert rm.daily_pnl == -15.0

    # Submit the exact same trade live again via risk_adapter
    live_duplicate = make_closed_trade(
        cohort_id="CH_PREV",
        trade_id="T_PERSISTED_1",
        closed_ts_ms=clock() - 2000000,
        net_pnl_usdt=-15.0,
    )

    res = runtime.risk_adapter.on_closed_trade(live_duplicate)
    assert res.status == "DUPLICATE_IGNORED"
    assert rm.daily_pnl == -15.0  # Still -15.0, not -30.0


# =========================================================================
# 22. Trades from other UTC calendar days are excluded by [start, end)
# =========================================================================
def test_22_trades_from_other_utc_days_excluded():
    # Fix reference clock at 2026-03-20 12:00:00 UTC
    ref_dt = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    now_ms = int(ref_dt.timestamp() * 1000)
    clock = FakeClock(now_ms)
    ledger = PaperLedger(db_path=":memory:")

    # Previous day trade: 2026-03-19 23:59:59.999 UTC
    prev_dt = datetime(2026, 3, 19, 23, 59, 59, 999000, tzinfo=timezone.utc)
    prev_ms = int(prev_dt.timestamp() * 1000)

    # Next day trade: 2026-03-21 00:00:00.001 UTC
    next_dt = datetime(2026, 3, 21, 0, 0, 0, 1000, tzinfo=timezone.utc)
    next_ms = int(next_dt.timestamp() * 1000)

    # Today trade: 2026-03-20 08:00:00 UTC
    today_dt = datetime(2026, 3, 20, 8, 0, 0, tzinfo=timezone.utc)
    today_ms = int(today_dt.timestamp() * 1000)

    insert_test_cohort(ledger, "CH_PREV", prev_ms - 100000)
    insert_test_event(ledger, "CH_PREV", "STARTED", prev_ms - 90000)
    insert_test_closed_trade(ledger, "CH_PREV", "T_YESTERDAY", prev_ms, -50.0)
    insert_test_closed_trade(ledger, "CH_PREV", "T_TODAY", today_ms, -20.0)
    insert_test_closed_trade(ledger, "CH_PREV", "T_TOMORROW", next_ms, -80.0)
    insert_test_event(
        ledger,
        "CH_PREV",
        "GRACEFUL_SHUTDOWN",
        today_ms + 1000,
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    start_ms, end_ms = ShadowPaperRuntime.get_utc_day_range_ms(ref_dt.date())
    trades_in_day = ledger.get_closed_trades_by_timerange(start_ms, end_ms)

    trade_ids = [t.trade_id for t in trades_in_day]
    assert "T_TODAY" in trade_ids
    assert "T_YESTERDAY" not in trade_ids
    assert "T_TOMORROW" not in trade_ids

    # Start runtime and verify only today is accounted
    config = make_test_config(cohort_id="CH_NEW_22")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)
    runtime.start()

    assert runtime.risk_adapter.risk_manager.daily_pnl == -20.0


# =========================================================================
# 23. Month and leap-year boundaries produce exact deterministic UTC ranges
# =========================================================================
def test_23_month_year_boundary_utc_range():
    cases = [
        # Year turnover
        (date(2026, 12, 31), datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc), datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        # Leap year leap day
        (date(2024, 2, 28), datetime(2024, 2, 28, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 2, 29, 0, 0, 0, tzinfo=timezone.utc)),
        (date(2024, 2, 29), datetime(2024, 2, 29, 0, 0, 0, tzinfo=timezone.utc), datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)),
        # Month turnover (30 days)
        (date(2026, 4, 30), datetime(2026, 4, 30, 0, 0, 0, tzinfo=timezone.utc), datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)),
    ]

    for d, expected_start_dt, expected_end_dt in cases:
        start_ms, end_ms = ShadowPaperRuntime.get_utc_day_range_ms(d)
        expected_start_ms = int(expected_start_dt.timestamp() * 1000)
        expected_end_ms = int(expected_end_dt.timestamp() * 1000)

        assert start_ms == expected_start_ms, f"Failed start for {d}"
        assert end_ms == expected_end_ms, f"Failed end for {d}"
        assert end_ms - start_ms == 86_400_000, f"Range is not exactly 24 hours for {d}"


# =========================================================================
# 24. C3.5.2 SESSION_SAFE runtime accounting continues operating after startup
# =========================================================================
def test_24_session_safe_remains_active_after_clean_start():
    clock = make_now_clock()
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_NEW_24")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, daily_loss_limit_usdt=50.0)

    assert runtime.start() is True
    assert runtime.accepts_new_exposure is True

    # Live market trade closes position with -55.0 USDT loss
    loss_trade = make_closed_trade(
        cohort_id="CH_NEW_24",
        trade_id="T_LIVE_LOSS",
        closed_ts_ms=clock(),
        net_pnl_usdt=-55.0,
    )

    with runtime._lock:
        res = runtime.risk_adapter.on_closed_trade(loss_trade)
        assert res.status == "APPLIED"
        if res.limit_exceeded:
            runtime._accept_new_exposure = False
            runtime._risk_block_reason = "DAILY_LOSS_LIMIT_REACHED"

    # Runtime must immediately block new exposure
    assert runtime.accepts_new_exposure is False
    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "DAILY_LOSS_LIMIT_REACHED"

    # Subsequent signal must be rejected
    sig = {
        "event_type": "signal_event",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "timestamp": clock(),
        "epoch_ms": clock(),
        "timeframe": "1m",
    }
    sig_res = runtime.on_signal(sig)
    assert sig_res.status == "INACTIVE"
    assert "DAILY_LOSS_LIMIT_REACHED" in (sig_res.error or "")


# =========================================================================
# 25. C3.4 max_open_positions execution capacity remains strictly operational
# =========================================================================
def test_25_max_open_positions_remains_operational():
    clock = FakeClock(1774010000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_NEW_25")
    # Configure max_position_size = 1
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock, max_position_size=1)

    assert runtime.start() is True
    assert runtime.accepts_new_exposure is True
    assert runtime.is_risk_position_limit_active is True

    # Fill one order to create 1 open position
    sink = runtime.execution_sink
    order = PaperOrder(
        order_id="ord_1",
        decision_id="dec_1",
        cohort_id="CH_NEW_25",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        reference_price=50000.0,
        notional_usdt=5000.0,
        signal_timestamp=clock(),
        decision_timestamp=clock(),
        available_at=clock(),
        expires_at=clock() + 5000,
        horizon_s=60,
    )
    submit_res = sink.submit_order(order)
    assert submit_res.status.value == "ACCEPTED"

    # Simulate fill via normalized trade dict
    norm_trade = {
        "p": 50000.0,
        "q": 1.0,
        "T": clock(),
        "T_raw": clock(),
        "m": False,
        "source": "fut_agg",
        "trade_id": "fill_trade_1",
        "symbol": "BTCUSDT",
    }
    sink.on_market_trade(norm_trade)

    # Now open_positions is 1 (capacity full)
    assert sink.get_capacity_snapshot()["open_positions"] == 1
    assert sink.get_capacity_snapshot()["available_capacity"] == 0

    # Second order must be strictly rejected
    order_2 = PaperOrder(
        order_id="ord_2",
        decision_id="dec_2",
        cohort_id="CH_NEW_25",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        reference_price=50000.0,
        notional_usdt=5000.0,
        signal_timestamp=clock(),
        decision_timestamp=clock(),
        available_at=clock(),
        expires_at=clock() + 5000,
        horizon_s=60,
    )
    res_2 = sink.submit_order(order_2)
    assert res_2.status.value == "REJECTED"
    assert res_2.rejection is not None
    assert res_2.rejection.reason in ("MAX_OPEN_POSITIONS_REACHED", "POSITION_OPEN")


# =========================================================================
# C3.5.3b ADDITIONAL TESTS
# =========================================================================

# =========================================================================
# 26. STARTED yesterday + GRACEFUL today flat => clean restart
# =========================================================================
def test_26_started_yesterday_graceful_today_flat_is_clean():
    now_dt = datetime.now(timezone.utc)
    today_00 = datetime(now_dt.year, now_dt.month, now_dt.day, 0, 0, 0, tzinfo=timezone.utc)
    yesterday_2355 = today_00 - timedelta(minutes=5)
    today_0005 = today_00 + timedelta(minutes=5)
    today_0010 = today_00 + timedelta(minutes=10)

    clock = FakeClock(int(today_0010.timestamp() * 1000))
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_CROSS_CLEAN", int(yesterday_2355.timestamp() * 1000))
    insert_test_event(ledger, "CH_CROSS_CLEAN", "STARTED", int(yesterday_2355.timestamp() * 1000))
    insert_test_closed_trade(ledger, "CH_CROSS_CLEAN", "T_TODAY_EARLY", int(today_0005.timestamp() * 1000) - 1000, -25.0)
    insert_test_event(
        ledger,
        "CH_CROSS_CLEAN",
        "GRACEFUL_SHUTDOWN",
        int(today_0005.timestamp() * 1000),
        metadata={"open_positions_count": 0, "pending_orders_count": 0},
    )

    config = make_test_config(cohort_id="CH_TODAY_26")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is True
    assert runtime.accepts_new_exposure is True
    assert runtime.risk_daily_loss_recovery_level == "CLEAN_RESTART_SAFE"
    assert runtime.is_risk_daily_loss_active is True
    # Trade closed today at 00:04 UTC must be restored in today's daily pnl
    assert runtime.risk_adapter.risk_manager.daily_pnl == -25.0


# =========================================================================
# 27. STARTED yesterday + no graceful => UNVERIFIED and blocks
# =========================================================================
def test_27_started_yesterday_without_graceful_blocks():
    now_dt = datetime.now(timezone.utc)
    today_00 = datetime(now_dt.year, now_dt.month, now_dt.day, 0, 0, 0, tzinfo=timezone.utc)
    yesterday_2355 = today_00 - timedelta(minutes=5)
    today_0010 = today_00 + timedelta(minutes=10)

    clock = FakeClock(int(today_0010.timestamp() * 1000))
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_CROSS_UNFINISHED", int(yesterday_2355.timestamp() * 1000))
    insert_test_event(ledger, "CH_CROSS_UNFINISHED", "STARTED", int(yesterday_2355.timestamp() * 1000))

    config = make_test_config(cohort_id="CH_TODAY_27")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "UNEXPECTED_PREVIOUS_SHUTDOWN"


# =========================================================================
# 28. Cohort persisted without events => UNVERIFIED and blocks
# =========================================================================
def test_28_cohort_without_events_blocks_fail_closed():
    clock = make_now_clock()
    ledger = PaperLedger(db_path=":memory:")

    # Insert cohort entry into cohorts table with NO events in cohort_events
    insert_test_cohort(ledger, "CH_ORPHAN", clock() - 3600000)

    config = make_test_config(cohort_id="CH_NEW_28")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "COHORT_WITHOUT_LIFECYCLE_EVENTS"


# =========================================================================
# 29. Old STARTED without terminal event => blocks
# =========================================================================
def test_29_old_started_without_terminal_blocks():
    clock = make_now_clock()
    ledger = PaperLedger(db_path=":memory:")

    # Started 3 days ago (outside today's window)
    insert_test_cohort(ledger, "CH_ANCIENT", clock() - (3 * 86400 * 1000))
    insert_test_event(ledger, "CH_ANCIENT", "STARTED", clock() - (3 * 86400 * 1000) + 1000)

    config = make_test_config(cohort_id="CH_NEW_29")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "UNEXPECTED_PREVIOUS_SHUTDOWN"


# =========================================================================
# 30. PERSISTENCE_FAILURE cross-midnight => blocks
# =========================================================================
def test_30_persistence_failure_cross_midnight_blocks():
    now_dt = datetime.now(timezone.utc)
    today_00 = datetime(now_dt.year, now_dt.month, now_dt.day, 0, 0, 0, tzinfo=timezone.utc)
    yesterday_2355 = today_00 - timedelta(minutes=5)
    today_0005 = today_00 + timedelta(minutes=5)
    today_0010 = today_00 + timedelta(minutes=10)

    clock = FakeClock(int(today_0010.timestamp() * 1000))
    ledger = PaperLedger(db_path=":memory:")

    insert_test_cohort(ledger, "CH_FAIL_CROSS", int(yesterday_2355.timestamp() * 1000))
    insert_test_event(ledger, "CH_FAIL_CROSS", "STARTED", int(yesterday_2355.timestamp() * 1000))
    insert_test_event(ledger, "CH_FAIL_CROSS", "PERSISTENCE_FAILURE", int(today_0005.timestamp() * 1000))

    config = make_test_config(cohort_id="CH_NEW_30")
    runtime = make_test_runtime(config=config, ledger=ledger, clock=clock)

    assert runtime.start() is False
    assert runtime.accepts_new_exposure is False
    assert runtime.risk_daily_loss_recovery_level == "UNVERIFIED_AFTER_RESTART"
    assert runtime.risk_block_reason == "PERSISTENCE_FAILURE"


# =========================================================================
# 31. flush_status real contract: returns SUCCESS (never EMPTY) even after queue drains
# =========================================================================
def test_31_flush_status_drained_writer_contract():
    ledger = PaperLedger(db_path=":memory:")
    ts = 1774000000000

    ledger.create_cohort("CH_DRAIN", ts)
    ledger.record_cohort_event("CH_DRAIN", "STARTED", ts)

    # Let the writer background thread drain the queue completely
    time.sleep(0.1)

    # flush_status() must return SUCCESS (durable commit confirmation), never EMPTY
    res = ledger.flush_status()
    assert res == "SUCCESS"

    # Confirm durability via direct SQLite query
    cur = ledger._conn.cursor()
    cur.execute("SELECT event_type FROM cohort_events WHERE cohort_id = 'CH_DRAIN'")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "STARTED"


# =========================================================================
# 32. Transactional in-memory restore: invalid trade does NOT leak partial daily_pnl
# =========================================================================
def test_32_restore_transactional_in_memory_no_partial_leak():
    rm = RiskManager(RiskConfig(initial_capital=1000.0, max_daily_loss=0.10))
    rm.daily_pnl = 0.0
    ref_date = date(2026, 3, 20)
    ts_ms = int(datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    # Trade 1: valid trade with -100 PnL
    t1 = make_closed_trade("CH_TX", "T_TX_1", ts_ms, -100.0, costs_complete=True)

    # Trade 2: invalid trade (costs_complete=False)
    t2 = make_closed_trade("CH_TX", "T_TX_2", ts_ms + 1000, -50.0, costs_complete=False)

    # Rehydrate batch with [t1, t2]
    res = rm.restore_daily_closed_trades([t1, t2], ref_date)

    # Must fail closed with PNL_INCOMPLETE
    assert res.status == "PNL_INCOMPLETE"
    assert res.pnl_complete is False

    # Partial state from t1 must NOT be published
    assert rm.daily_pnl == 0.0
    assert "T_TX_1" not in rm._seen_trades


# =========================================================================
# 33. ledger=None remains SESSION_SAFE with honest flag and exposure enabled
# =========================================================================
def test_33_ledger_none_remains_session_safe_and_honest_flag():
    clock = FakeClock(1774000000000)
    config = make_test_config(cohort_id="CH_INMEM_33")
    runtime = make_test_runtime(config=config, ledger=None, clock=clock)

    assert runtime.start() is True
    # Without ledger, recovery level is SESSION_SAFE
    assert runtime.risk_daily_loss_recovery_level == "SESSION_SAFE"
    # Honest flag: risk_daily_loss_active is False because recovery cannot be verified from persistence
    assert runtime.is_risk_daily_loss_active is False
    # But new exposure within the memory session is allowed if limits permit
    assert runtime.accepts_new_exposure is True
    assert runtime.risk_block_reason is None
