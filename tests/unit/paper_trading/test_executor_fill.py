# tests/unit/paper_trading/test_executor_fill.py
"""Unit tests for the simulated paper executor fill and lifecycle rules."""

import pytest

from paper_trading.contracts import CanonicalDecision
from paper_trading.executor import ExecutorConfig, PaperExecutor
from paper_trading.positions import PositionManager


def test_no_fill_without_eligible_tick():
    """Invariant 6 & Item 1: No fill before available_at; first tick >= available_at fills with >= 0 latencies."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=5000))

    d = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )

    order, rej = executor.submit_decision(d)
    assert order is not None
    assert rej is None
    assert order.available_at == 1050

    # 1. Tick at T == 1049: strictly before available_at -> must NOT fill
    ev_early = executor.on_tick(T=1049, p=100.0, q=1.0, m=False)
    assert len(ev_early.fills) == 0
    assert len(executor.pending_orders) == 1

    # 2. First tick at T == 1050: exact boundary equality satisfies tick.timestamp >= available_at -> fills!
    ev_fill = executor.on_tick(T=1050, p=100.0, q=1.0, m=False, trade_id="tr_42")
    assert len(ev_fill.fills) == 1
    fill = ev_fill.fills[0]
    assert fill.fill_timestamp == 1050
    assert fill.trade_id_used == "tr_42"

    # Latency metrics
    assert fill.decision_to_fill_ms == (1050 - 1020) == 30
    assert fill.available_to_fill_ms == (1050 - 1050) == 0
    assert fill.decision_to_fill_ms >= 0
    assert fill.available_to_fill_ms >= 0
    assert len(executor.pending_orders) == 0


def test_order_expires_when_no_ticks_within_ttl():
    """If no ticks arrive before expires_at, order transitions to EXPIRED_NO_MARKET_DATA without fill."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=5000))

    d = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    executor.submit_decision(d)
    # expires_at = 1050 + 5000 = 6050

    # Tick arrives at T=6051 (past TTL)
    events = executor.on_tick(T=6051, p=100.0, q=1.0, m=False)
    assert len(events.fills) == 0
    assert len(events.rejections) == 1
    assert events.rejections[0].reason == "EXPIRED_NO_MARKET_DATA"
    assert len(executor.pending_orders) == 0


def test_position_open_rejection():
    """Submitting a decision for a symbol with an existing open position is rejected."""
    pm = PositionManager()
    executor = PaperExecutor(position_manager=pm)

    d1 = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    executor.submit_decision(d1)
    # Fill d1 at T=1100
    executor.on_tick(T=1100, p=100.0, q=1.0, m=False)
    assert pm.has_open_position("c1", "flow_v1", "BTCUSDT") is True

    # Now submit d2 for the same cohort, provider and symbol
    d2 = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w2",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=2000,
        decision_timestamp=2020,
        available_at=2050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    order, rej = executor.submit_decision(d2)
    assert order is None
    assert rej is not None
    assert rej.reason == "POSITION_OPEN"
