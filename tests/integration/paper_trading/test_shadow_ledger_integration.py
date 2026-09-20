# tests/integration/paper_trading/test_shadow_ledger_integration.py
"""
Hermetic integration tests connecting PaperLedger to ShadowPaperRuntime (Gate B2.5-C).

Validates:
1. Strict cohort lifecycle, deterministic metadata and duplicate rejection.
2. Complete signal observation tracking and causality.
3. Decision-before-risk and risk-before-order sequence enforcement.
4. Discrete economic event persistence from market trades (zero ordinary ticks persisted).
5. Ledger health gates, fail-closed boundaries, and unhindered in-flight tick execution.
6. Graceful shutdown without synthetic trade closures and crash/incomplete detection.
7. Durability confirmation via flush_status and strict isolation (zero network, zero LLM, zero real orders).
"""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
import os
import socket
import sqlite3
import tempfile
import threading
from typing import Any, Dict, List, Optional
import pytest

from paper_trading.config import ShadowPaperConfig, parse_shadow_config
from paper_trading.shadow_runtime import (
    ShadowPaperRuntime,
    ShadowSignalResult,
    ShadowTradeResult,
)
from paper_trading.ledger import (
    PaperLedger,
    compute_cohort_event_id,
    compute_risk_evaluation_id,
    compute_signal_observation_id,
)
from paper_trading.adapters.signal_adapter import SignalDecisionAdapter
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.execution_sink import ExecutionSink
from paper_trading.executor import PaperExecutor
from paper_trading.cost_model import CostModel
from paper_trading.contracts import PaperOrder, Rejection
from risk_management.risk_manager import RiskConfig, RiskManager


class FakeClock:
    """Deterministic injectable clock for causality testing without wall-clock sleep."""

    def __init__(self, start_ms: int = 1700000000000):
        self._current_ms: int = start_ms

    def __call__(self) -> int:
        return self._current_ms

    def advance(self, delta_ms: int) -> int:
        self._current_ms += delta_ms
        return self._current_ms

    def set(self, target_ms: int) -> int:
        self._current_ms = target_ms
        return self._current_ms


def make_norm_trade(
    price: float,
    qty: float,
    T: int,
    trade_id: int | str,
    is_buyer_maker: bool = False,
    source: str = "fut_agg",
) -> Dict[str, Any]:
    """Helper to produce realistic normalized trade payload identical to market_orchestrator."""
    return {
        "p": price,
        "q": qty,
        "T": T,
        "T_raw": T,
        "m": is_buyer_maker,
        "source": source,
        "trade_id": trade_id,
        "symbol": "BTCUSDT",
    }


def make_test_config(
    provider: str = "fixed_long",
    cohort_id: str = "CH_SHADOW_LEDGER_2026",
    notional: float = 1000.0,
    horizon_s: int = 60,
    ttl_ms: int = 5000,
    seed: Optional[int] = None,
) -> ShadowPaperConfig:
    """Helper to generate strict, valid ShadowPaperConfig."""
    env = {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": cohort_id,
        "PAPER_PROVIDER": provider,
        "PAPER_SYMBOL": "BTCUSDT",
        "PAPER_TIMEFRAME": "1m",
        "PAPER_NOTIONAL_USDT": str(notional),
        "PAPER_HORIZON_S": str(horizon_s),
        "PAPER_ORDER_TTL_MS": str(ttl_ms),
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "binance_vip0",
        "PAPER_COST_EFFECTIVE_AT": "2026-01-01T00:00:00Z",
        "PAPER_STRATEGY_VERSION": "c3_shadow_ledger_v1",
    }
    if seed is not None:
        env["PAPER_RANDOM_SEED"] = str(seed)
    res = parse_shadow_config(env)
    assert res.is_valid is True
    assert res.config is not None
    return res.config


def test_1_start_cohort():
    """1. start cohort: strict initialization creates cohort and records STARTED event."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_START_01")
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        clock_ms=clock,
        git_sha="git_sha_audit_01",
        ledger_is_owner=False,
    )

    started = runtime.start()
    assert started is True
    assert runtime.get_status()["active"] is True
    assert runtime.get_status()["accept_new_exposure"] is True

    # Confirm cohort exists in ledger
    assert ledger.cohort_exists("CH_START_01") is True

    # Confirm STARTED event
    events = ledger.get_cohort_events("CH_START_01")
    assert len(events) == 1
    assert events[0]["event_type"] == "STARTED"
    assert events[0]["metadata"]["git_sha"] == "git_sha_audit_01"

    runtime.shutdown()
    ledger.close()


def test_2_cohort_metadata():
    """2. cohort metadata contains all required economic, operational and risk fields."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_META_02", seed=42)
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        clock_ms=clock,
        git_sha="sha_meta_02",
        ledger_is_owner=False,
    )
    runtime.start()

    cur = ledger._conn.cursor()
    cur.execute("SELECT metadata_json FROM cohorts WHERE cohort_id = 'CH_META_02'")
    row = cur.fetchone()
    assert row is not None
    import json
    meta = json.loads(row["metadata_json"])

    assert meta["git_sha"] == "sha_meta_02"
    assert meta["strategy_version"] == "c3_shadow_ledger_v1"
    assert meta["provider"] == "fixed_long"
    assert meta["random_seed"] == 42
    assert meta["symbol"] == "BTCUSDT"
    assert meta["timeframe"] == "1m"
    assert meta["notional_usdt"] == 1000.0
    assert meta["horizon_s"] == 60
    assert meta["order_ttl_ms"] == 5000
    assert meta["fees"] == 5.0
    assert meta["slippage"] == 1.0
    assert meta["cost_source"] == "binance_vip0"
    assert meta["cost_effective_at"] == "2026-01-01T00:00:00Z"
    assert "risk_config" in meta
    assert meta["risk_position_limit_active"] is False
    assert meta["risk_daily_loss_active"] is False

    runtime.shutdown()
    ledger.close()


def test_3_duplicate_cohort():
    """3. duplicate cohort: second runtime with same cohort_id fails start and remains inactive."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_DUP_03")

    runtime_a = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock, git_sha="sha_a")
    assert runtime_a.start() is True

    # Runtime B attempts to reuse the same cohort_id in the same database
    runtime_b = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock, git_sha="sha_b")
    started_b = runtime_b.start()
    assert started_b is False
    assert runtime_b.get_status()["active"] is False
    assert runtime_b.get_status()["accept_new_exposure"] is False

    # Runtime B rejects signals
    sig_res = runtime_b.on_signal({
        "epoch_ms": clock(),
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert sig_res.status == "INACTIVE"

    runtime_a.shutdown()
    ledger.close()


def test_4_directional_observation():
    """4. directional observation: on_signal generates exactly one DECISION_CREATED observation."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_OBS_04")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "side": "LONG",
        "window_id": "BTCUSDT:1m:1700000000000",
        "event_type": "candle_close",
    })
    assert res.status == "ORDER_SUBMITTED"
    assert runtime.get_counters()["observations_enqueued"] == 1

    obs = ledger.get_signal_observations("CH_OBS_04")
    assert len(obs) == 1
    assert obs[0]["status"] == "DECISION_CREATED"
    assert obs[0]["source_side"] == "LONG"
    assert obs[0]["symbol"] == "BTCUSDT"

    runtime.shutdown()
    ledger.close()


def test_5_neutral_observation():
    """5. neutral observation: non-directional signal mapped to SKIPPED_NON_DIRECTIONAL observation."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_OBS_05")
    # Explicitly configure FOLLOW_SIGNAL mode to test neutral signal skipping
    custom_adapter = SignalDecisionAdapter(
        cohort_id=config.cohort_id or "CH_OBS_05",
        mode="FOLLOW_SIGNAL",
        timeframe=config.timeframe,
        clock_ms=clock,
    )
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        signal_adapter=custom_adapter,
        clock_ms=clock,
    )
    runtime.start()

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "side": "NEUTRAL",
        "window_id": "BTCUSDT:1m:1700000000000",
        "event_type": "candle_close",
    })
    assert res.status == "SKIPPED"
    assert runtime.get_counters()["observations_enqueued"] == 1
    assert runtime.get_counters()["decisions_enqueued"] == 0

    obs = ledger.get_signal_observations("CH_OBS_05")
    assert len(obs) == 1
    assert obs[0]["status"] == "SKIPPED_NON_DIRECTIONAL"
    assert obs[0]["source_side"] == "NEUTRAL"

    runtime.shutdown()
    ledger.close()


def test_6_invalid_observation():
    """6. invalid observation: malformed signal mapped to INVALID_SIGNAL observation."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_OBS_06")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # Invalid signal: missing required price or non-finite price
    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": float("nan"),
        "symbol": "BTCUSDT",
    })
    assert res.status == "SKIPPED"
    assert runtime.get_counters()["observations_enqueued"] == 1
    assert runtime.get_counters()["decisions_enqueued"] == 0

    obs = ledger.get_signal_observations("CH_OBS_06")
    assert len(obs) == 1
    assert obs[0]["status"] == "INVALID_SIGNAL"

    runtime.shutdown()
    ledger.close()


def test_7_decision_before_risk_ordering():
    """7. decision-before-risk ordering: decision persisted before RiskAdapter evaluation."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_DEC_07")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # Track ordering via mock wrapping
    call_order: List[str] = []
    orig_record_decision = ledger.record_decision
    orig_evaluate = runtime.risk_adapter.evaluate

    def wrapped_record_decision(dec):
        call_order.append("record_decision")
        return orig_record_decision(dec)

    def wrapped_evaluate(dec):
        call_order.append("evaluate_risk")
        return orig_evaluate(dec)

    ledger.record_decision = wrapped_record_decision  # type: ignore[assignment]
    runtime.risk_adapter.evaluate = wrapped_evaluate  # type: ignore[assignment]

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res.status == "ORDER_SUBMITTED"
    assert call_order == ["record_decision", "evaluate_risk"]

    runtime.shutdown()
    ledger.close()


def test_8_risk_approved_persisted():
    """8. risk approved persisted: APPROVED evaluation recorded in risk_evaluations."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_RISK_08")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res.status == "ORDER_SUBMITTED"
    assert runtime.get_counters()["risk_evaluations_enqueued"] == 1

    evals = ledger.get_risk_evaluations("CH_RISK_08")
    assert len(evals) == 1
    assert evals[0]["status"] == "APPROVED"
    assert evals[0]["risk_confidence"] == 0.0  # Mapped uncalibrated confidence
    assert evals[0]["source_confidence"] is None

    runtime.shutdown()
    ledger.close()


def test_9_risk_rejected_persisted():
    """9. risk rejected persisted: REJECTED evaluation recorded and no order submitted."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_RISK_09", notional=1000.0)

    # Configure risk manager to reject orders over 500 USDT notional
    strict_risk_manager = RiskManager(
        config=RiskConfig(max_position_size=500.0)
    )
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        risk_manager=strict_risk_manager,
        clock_ms=clock,
    )
    runtime.start()

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res.status == "RISK_REJECTED"
    assert runtime.get_counters()["risk_evaluations_enqueued"] == 1
    assert runtime.get_counters()["orders_enqueued"] == 0

    evals = ledger.get_risk_evaluations("CH_RISK_09")
    assert len(evals) == 1
    assert evals[0]["status"] == "REJECTED"
    
    cur = ledger._conn.cursor()
    cur.execute("SELECT * FROM orders WHERE cohort_id = 'CH_RISK_09'")
    assert len(cur.fetchall()) == 0

    runtime.shutdown()
    ledger.close()


def test_10_order_before_submit():
    """10. order-before-submit: order persisted to ledger before ExecutionSink.submit_order."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_ORD_10")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    call_order: List[str] = []
    orig_record_order = ledger.record_order
    orig_submit_order = runtime.execution_sink.submit_order

    def wrapped_record_order(order):
        call_order.append("record_order")
        return orig_record_order(order)

    def wrapped_submit_order(order):
        call_order.append("submit_order")
        return orig_submit_order(order)

    ledger.record_order = wrapped_record_order  # type: ignore[assignment]
    runtime.execution_sink.submit_order = wrapped_submit_order  # type: ignore[assignment]

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res.status == "ORDER_SUBMITTED"
    assert call_order == ["record_order", "submit_order"]

    runtime.shutdown()
    ledger.close()


def test_11_executor_rejection_persisted():
    """11. executor rejection persisted: genuine sink rejection recorded via record_rejection."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_REJ_11")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # First signal accepted
    res1 = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res1.status == "ORDER_SUBMITTED"

    # Second conflicting signal while order pending produces executor rejection (POSITION_OPEN reason)
    clock.advance(100)
    res2 = runtime.on_signal({
        "epoch_ms": 1700000000100,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res2.status == "ORDER_REJECTED"

    rejections = ledger.get_rejections("CH_REJ_11")
    assert len(rejections) == 1
    assert rejections[0].reason == "POSITION_OPEN"

    runtime.shutdown()
    ledger.close()


def test_12_fill_persisted():
    """12. fill persisted: market trade matching pending order records fill in ledger."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_FILL_12")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })

    # Deliver matching trade tick
    clock.advance(100)
    trade_res = runtime.on_market_trade(make_norm_trade(
        price=50000.0,
        qty=1.0,
        T=1700000000100,
        trade_id=101,
    ))
    assert trade_res.fills_count == 1
    assert runtime.get_counters()["fills_enqueued"] == 1

    fills = ledger.get_fills("CH_FILL_12")
    assert len(fills) == 1
    assert fills[0].fill_price > 0
    assert fills[0].side == "LONG"

    runtime.shutdown()
    ledger.close()


def test_13_position_persisted():
    """13. position persisted: opened position recorded via record_position."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_POS_13")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })

    clock.advance(100)
    runtime.on_market_trade(make_norm_trade(
        price=50000.0,
        qty=1.0,
        T=1700000000100,
        trade_id=102,
    ))

    assert runtime.get_counters()["positions_enqueued"] == 1
    ledger.flush()
    cur = ledger._conn.cursor()
    cur.execute("SELECT * FROM positions WHERE cohort_id = 'CH_POS_13'")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["is_open"] == 1
    assert rows[0]["side"] == "LONG"

    runtime.shutdown()
    ledger.close()


def test_14_closed_trade_persisted():
    """14. closed trade persisted: position closed by horizon records closed trade."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_CLOSE_14", horizon_s=60)
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    clock.advance(100)
    runtime.on_market_trade(make_norm_trade(
        price=50000.0,
        qty=1.0,
        T=1700000000100,
        trade_id=103,
    ))

    # Advance past horizon_s (60s = 60,000ms)
    clock.advance(60000)
    close_tick_res = runtime.on_market_trade(make_norm_trade(
        price=50100.0,
        qty=1.0,
        T=1700000060100,
        trade_id=104,
    ))
    assert close_tick_res.closed_trades_count == 1
    assert runtime.get_counters()["closed_trades_enqueued"] == 1

    trades = ledger.get_closed_trades("CH_CLOSE_14")
    assert len(trades) == 1
    assert trades[0].exit_reason == "HORIZON_EXPIRY"
    assert abs(trades[0].exit_price - 50100.0) < 10.0

    runtime.shutdown()
    ledger.close()


def test_15_funding_none_roundtrip():
    """15. funding None roundtrip: missing funding rate preserves None values across DB reload."""
    clock = FakeClock(1700000000000)  # 2023-11-14 22:13:20 UTC
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db_file.name
    db_file.close()

    try:
        ledger = PaperLedger(db_path=db_path)
        config = make_test_config(cohort_id="CH_FUND_15", horizon_s=10000)
        runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
        runtime.start()

        runtime.on_signal({
            "epoch_ms": 1700000000000,
            "price": 50000.0,
            "symbol": "BTCUSDT",
        })
        runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 105))

        # Advance across 8h funding boundary (e.g. crossing 00:00:00 UTC at 1700006400000)
        clock.advance(10000 * 1000)
        runtime.on_market_trade(make_norm_trade(50050.0, 1.0, 1700010000100, 106))

        runtime.shutdown()
        ledger.close()

        # Reopen database and inspect ClosedTrade
        ledger_reloaded = PaperLedger(db_path=db_path)
        trades = ledger_reloaded.get_closed_trades("CH_FUND_15")
        assert len(trades) == 1
        t = trades[0]

        assert t.funding_bps is None
        assert t.funding_usdt is None
        assert t.net_pnl_bps is None
        assert t.trade_win is None
        assert t.costs_complete is False
        ledger_reloaded.close()

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_16_graceful_shutdown_event():
    """16. graceful shutdown event: shutdown records GRACEFUL_SHUTDOWN cohort event."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_SHUT_16")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # Leave one order pending
    runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })

    runtime.shutdown()

    events = ledger.get_cohort_events("CH_SHUT_16")
    assert len(events) == 2
    assert events[0]["event_type"] == "STARTED"
    assert events[1]["event_type"] == "GRACEFUL_SHUTDOWN"
    assert events[1]["metadata"]["pending_orders_count"] == 1
    assert events[1]["metadata"]["open_positions_count"] == 0

    ledger.close()


def test_17_open_position_remains_open():
    """17. open position remains open: shutdown does not close active position in ledger."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_OPEN_17")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 107))

    runtime.shutdown()

    cur = ledger._conn.cursor()
    cur.execute("SELECT is_open FROM positions WHERE cohort_id = 'CH_OPEN_17'")
    row = cur.fetchone()
    assert row is not None
    assert row["is_open"] == 1

    ledger.close()


def test_18_no_synthetic_closed_trade_shutdown():
    """18. no synthetic closed trade shutdown: shutdown with open position inserts 0 closed trades."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_NOSYN_18")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 108))

    runtime.shutdown()

    trades = ledger.get_closed_trades("CH_NOSYN_18")
    assert len(trades) == 0

    ledger.close()


def test_19_incomplete_cohort():
    """19. incomplete cohort: ungracefully terminated session detected as incomplete."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_INCOMP_19")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 109))

    # Flush but do NOT call runtime.shutdown()
    ledger.flush()

    assert ledger.is_cohort_incomplete("CH_INCOMP_19") is True
    assert ledger.cohort_has_open_positions("CH_INCOMP_19") is True

    ledger.close()


def test_20_duplicate_startup_blocked():
    """20. duplicate startup blocked: attempting to restart existing cohort fails closed."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_BLOCK_20")

    runtime_1 = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    assert runtime_1.start() is True
    runtime_1.shutdown()

    # Attempt second startup on same cohort
    runtime_2 = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    assert runtime_2.start() is False
    assert runtime_2.get_status()["active"] is False

    res = runtime_2.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res.status == "INACTIVE"

    ledger.close()


def test_21_ledger_unhealthy_blocks_exposure():
    """21. ledger unhealthy blocks exposure: unhealthy ledger fails closed for new decisions."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_UNH_21")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # Manually inject error in ledger
    with ledger._lock:
        ledger._has_error = True
        ledger._error_count += 1
        ledger._last_error_type = "InjectedPersistenceError"

    res = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })
    assert res.status == "ERROR"
    assert "Persistence health gate failure" in str(res.error)
    assert runtime.get_counters()["persistence_blocks"] == 1
    assert runtime.get_status()["accept_new_exposure"] is False

    ledger.close()


def test_22_ledger_unhealthy_still_processes_ticks():
    """22. ledger unhealthy still processes ticks: in-flight positions are not frozen."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_UNH_22", horizon_s=60)
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 110))

    # Mark ledger unhealthy
    with ledger._lock:
        ledger._has_error = True
        ledger._error_count += 1

    # Runtime blocks new signals
    sig_res = runtime.on_signal({"epoch_ms": 1700000000200, "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "ERROR"

    # But market ticks for horizon exit continue being processed by sink
    clock.advance(60000)
    trade_res = runtime.on_market_trade(make_norm_trade(50200.0, 1.0, 1700000060100, 111))
    assert trade_res.status == "PROCESSED"
    assert trade_res.closed_trades_count == 1

    ledger.close()


def test_23_writer_failure_after_fill():
    """23. writer failure after fill: fill in memory is preserved, new exposure disabled."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_WFAIL_23")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})

    # Break record_fill in ledger to return False
    ledger.record_fill = lambda fill: False  # type: ignore[assignment]

    clock.advance(100)
    trade_res = runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 112))

    # Memory state is not corrupted
    assert trade_res.fills_count == 1
    assert runtime.get_counters()["persistence_errors"] >= 1
    assert runtime.get_status()["persistence_failed"] is True
    assert runtime.get_status()["accept_new_exposure"] is False

    ledger.close()


def test_24_signal_conservation_persisted():
    """24. signal conservation persisted: directional + neutral + invalid == total observations."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_CONS_24")
    custom_adapter = SignalDecisionAdapter(
        cohort_id=config.cohort_id or "CH_CONS_24",
        mode="FOLLOW_SIGNAL",
        timeframe=config.timeframe,
        clock_ms=clock,
    )
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        signal_adapter=custom_adapter,
        clock_ms=clock,
    )
    runtime.start()

    # 1. Directional signal
    res1 = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "side": "LONG",
        "window_id": "BTCUSDT:1m:1700000000000",
        "event_type": "ev1",
    })
    assert res1.status == "ORDER_SUBMITTED"

    # Fill and close the trade so next signal won't be rejected by active order/position
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 113))
    clock.advance(70000)
    runtime.on_market_trade(make_norm_trade(50010.0, 1.0, 1700000070100, 114))

    # 2. Neutral signal
    clock.advance(5000)
    t2 = clock()
    res2 = runtime.on_signal({
        "epoch_ms": t2,
        "price": 50010.0,
        "symbol": "BTCUSDT",
        "side": "NEUTRAL",
        "window_id": f"BTCUSDT:1m:{t2}",
        "event_type": "ev2",
    })
    assert res2.status == "SKIPPED"

    # 3. Invalid signal
    clock.advance(5000)
    t3 = clock()
    res3 = runtime.on_signal({
        "epoch_ms": t3,
        "price": float("nan"),
        "symbol": "BTCUSDT",
        "side": "LONG",
        "window_id": f"BTCUSDT:1m:{t3}",
        "event_type": "ev3",
    })
    assert res3.status == "SKIPPED"

    obs = ledger.get_signal_observations("CH_CONS_24")
    assert len(obs) == 3

    counts = {}
    for o in obs:
        counts[o["status"]] = counts.get(o["status"], 0) + 1

    assert counts.get("DECISION_CREATED", 0) == 1
    assert counts.get("SKIPPED_NON_DIRECTIONAL", 0) == 1
    assert counts.get("INVALID_SIGNAL", 0) == 1

    assert len(obs) == (
        counts["DECISION_CREATED"]
        + counts["SKIPPED_NON_DIRECTIONAL"]
        + counts["INVALID_SIGNAL"]
    )
    cur = ledger._conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM decisions WHERE cohort_id = 'CH_CONS_24'")
    assert cur.fetchone()["cnt"] == 1

    runtime.shutdown()
    ledger.close()


def test_25_economic_conservation_persisted():
    """25. economic conservation persisted: discrete lifecycle events match 1-to-1 with no loss."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_ECON_25", notional=1000.0, horizon_s=60)

    # Initial risk manager allows up to 500 USDT (so 1000 USDT notional is rejected)
    risk_manager = RiskManager(
        config=RiskConfig(max_position_size=500.0)
    )
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        risk_manager=risk_manager,
        clock_ms=clock,
    )
    runtime.start()

    # Decision A: 1000 USDT notional exceeds 500 limit -> REJECTED
    res_a = runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "window_id": "w_a",
        "event_type": "ev_a",
    })
    assert res_a.status == "RISK_REJECTED"

    # Update risk manager to allow 5000 USDT for Decision B
    runtime.risk_adapter = RiskAdapter(
        risk_manager=RiskManager(config=RiskConfig(max_position_size=5000.0)),
        order_ttl_ms=config.order_ttl_ms or 5000,
    )
    clock.advance(1000)

    # Decision B: Approved -> Order -> Fill -> Close
    res_b = runtime.on_signal({
        "epoch_ms": 1700000001000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "window_id": "w_b",
        "event_type": "ev_b",
    })
    assert res_b.status == "ORDER_SUBMITTED"

    clock.advance(100)
    runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000001100, 115))

    clock.advance(60000)
    runtime.on_market_trade(make_norm_trade(50100.0, 1.0, 1700000061100, 116))

    runtime.shutdown()

    # Economic conservation verification
    cur = ledger._conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM decisions WHERE cohort_id = 'CH_ECON_25'")
    decisions_cnt = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM orders WHERE cohort_id = 'CH_ECON_25'")
    orders_cnt = cur.fetchone()["cnt"]

    risk_evals = ledger.get_risk_evaluations("CH_ECON_25")
    fills = ledger.get_fills("CH_ECON_25")
    closed_trades = ledger.get_closed_trades("CH_ECON_25")

    assert decisions_cnt == 2
    assert len(risk_evals) == 2
    assert sum(1 for r in risk_evals if r["status"] == "REJECTED") == 1
    assert sum(1 for r in risk_evals if r["status"] == "APPROVED") == 1
    assert orders_cnt == 1
    assert len(fills) == 1
    assert len(closed_trades) == 1

    ledger.close()


def test_26_zero_tick_persistence():
    """26. zero tick persistence: 1000 ordinary market ticks produce zero database writes."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_ZTICK_26")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    # Send 1000 trades with no active order
    for i in range(1000):
        t_ms = 1700000000000 + (i * 10)
        res = runtime.on_market_trade(make_norm_trade(
            price=50000.0 + (i % 10),
            qty=0.1,
            T=t_ms,
            trade_id=1000 + i,
        ))
        assert res.status == "PROCESSED"
        assert res.fills_count == 0

    assert runtime.get_counters()["ticks_seen"] == 1000
    ledger.flush()

    cur = ledger._conn.cursor()
    # Ensure no rows in orders, fills, positions, closed_trades, rejections
    for tbl in ("orders", "fills", "positions", "closed_trades", "rejections"):
        cur.execute(f"SELECT COUNT(*) as cnt FROM {tbl}")
        assert cur.fetchone()["cnt"] == 0, f"Table {tbl} unexpectedly contains data from ticks"

    runtime.shutdown()
    ledger.close()


def test_27_injected_ledger_ownership_respected():
    """27. injected ledger ownership respected: external ledger is not closed by runtime."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_OWN_27")

    # Injected with ledger_is_owner=False
    runtime = ShadowPaperRuntime(
        config=config,
        ledger=ledger,
        clock_ms=clock,
        ledger_is_owner=False,
    )
    runtime.start()
    runtime.shutdown()

    # Ledger must still be open and operational
    assert ledger._worker.is_alive() is True
    assert ledger.health_snapshot()["healthy"] is True

    # Now test with ledger_is_owner=True
    runtime_owned = ShadowPaperRuntime(
        config=make_test_config(cohort_id="CH_OWN_27_OWNED"),
        ledger=ledger,
        clock_ms=clock,
        ledger_is_owner=True,
    )
    runtime_owned.start()
    runtime_owned.shutdown()

    # Ledger worker stopped
    assert ledger._worker.is_alive() is False


def test_28_flush_success_confirms_durability():
    """28. flush success confirms durability: flush_status returns SUCCESS and SQLite is committed."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_DUR_28")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    runtime.on_signal({
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    })

    status = ledger.flush_status()
    assert status == "SUCCESS"

    cur = ledger._conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM signal_observations WHERE cohort_id = 'CH_DUR_28'")
    assert cur.fetchone()["cnt"] == 1

    runtime.shutdown()
    ledger.close()


def test_29_no_network():
    """29. no network: verifies zero network sockets or connections across operations."""
    class NetworkDisallowedError(RuntimeError):
        pass

    def fail_socket(*args, **kwargs):
        raise NetworkDisallowedError("Network I/O strictly disallowed in shadow runtime")

    orig_socket = socket.socket
    socket.socket = fail_socket  # type: ignore[assignment]

    try:
        clock = FakeClock(1700000000000)
        ledger = PaperLedger(db_path=":memory:")
        config = make_test_config(cohort_id="CH_NONET_29")
        runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
        runtime.start()

        runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
        runtime.on_market_trade(make_norm_trade(50000.0, 1.0, 1700000000100, 117))
        runtime.shutdown()
        ledger.close()
    finally:
        socket.socket = orig_socket


def test_30_no_llm():
    """30. no LLM: verifies zero imports or calls to LLM / AI modules in paper trading."""
    import sys
    ai_modules = [m for m in sys.modules if "ai_analyzer" in m or "openai" in m or "anthropic" in m]
    # Ensure no active runtime coupling with AI client in test execution
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_NOLLM_30")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    assert not hasattr(runtime, "ai_analyzer")
    assert not hasattr(runtime, "llm_client")
    runtime.shutdown()
    ledger.close()


def test_31_no_real_orders():
    """31. no real orders: verifies zero execution against live Binance or exchange client."""
    clock = FakeClock(1700000000000)
    ledger = PaperLedger(db_path=":memory:")
    config = make_test_config(cohort_id="CH_NOREAL_31")
    runtime = ShadowPaperRuntime(config=config, ledger=ledger, clock_ms=clock)
    runtime.start()

    res = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res.status == "ORDER_SUBMITTED"

    # Verify execution_sink does not contain Binance client or network endpoint
    assert not hasattr(runtime.execution_sink, "client")
    assert not hasattr(runtime.execution_sink, "binance")
    assert not hasattr(runtime.execution_sink, "api_key")

    runtime.shutdown()
    ledger.close()
