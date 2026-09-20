# tests/unit/paper_trading/test_contracts.py
"""Unit tests for paper trading contracts and invariants."""

import math
import pytest

from paper_trading.contracts import CanonicalDecision, InvalidDecisionError
from paper_trading.executor import PaperExecutor


def test_deterministic_uuid5():
    """Invariant 3: Identical decision inputs generate identical deterministic UUID5 decision_id."""
    d1 = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="win_1000",
        decision_provider="flow_v1",
        strategy_version="1.0.0",
        model_version="xgboost_v2",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    d2 = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="win_1000",
        decision_provider="flow_v1",
        strategy_version="1.0.0",
        model_version="xgboost_v2",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    assert d1.decision_id == d2.decision_id
    assert len(d1.decision_id) == 36

    d3 = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="win_1000",
        decision_provider="flow_v1",
        strategy_version="1.0.1",
        model_version="xgboost_v2",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    assert d1.decision_id != d3.decision_id


def test_temporal_causality_chain():
    """Invariant 5: signal_timestamp <= decision_timestamp <= available_at must hold strictly."""
    d_valid = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1000,
        available_at=1000,
        side="LONG",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    assert d_valid.signal_timestamp <= d_valid.decision_timestamp <= d_valid.available_at

    with pytest.raises(InvalidDecisionError, match="Causal violation"):
        CanonicalDecision(
            cohort_id="c1",
            symbol="BTCUSDT",
            window_id="w1",
            decision_provider="flow_v1",
            strategy_version="1.0",
            signal_timestamp=1005,
            decision_timestamp=1000,
            available_at=1050,
            side="LONG",
            reference_price=100.0,
            notional_usdt=100.0,
            horizon_s=60,
        )

    with pytest.raises(InvalidDecisionError, match="Causal violation"):
        CanonicalDecision(
            cohort_id="c1",
            symbol="BTCUSDT",
            window_id="w1",
            decision_provider="flow_v1",
            strategy_version="1.0",
            signal_timestamp=1000,
            decision_timestamp=1050,
            available_at=1040,
            side="LONG",
            reference_price=100.0,
            notional_usdt=100.0,
            horizon_s=60,
        )


def test_reference_price_and_individual_sl_tp_validation():
    """Item 2: Individual SL/TP validation against finite reference_price > 0."""
    # reference_price must be > 0 and finite
    with pytest.raises(InvalidDecisionError, match="reference_price"):
        CanonicalDecision(
            cohort_id="c1",
            symbol="BTCUSDT",
            window_id="w1",
            decision_provider="flow_v1",
            strategy_version="1.0",
            signal_timestamp=1000,
            decision_timestamp=1020,
            available_at=1050,
            side="LONG",
            reference_price=-10.0,
            notional_usdt=100.0,
            horizon_s=60,
        )

    # Valid: only stop_loss without take_profit
    d_only_sl = CanonicalDecision(
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
        notional_usdt=100.0,
        horizon_s=60,
        stop_loss=95.0,
        take_profit=None,
    )
    assert d_only_sl.stop_loss == 95.0
    assert d_only_sl.take_profit is None

    # Valid: only take_profit without stop_loss
    d_only_tp = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="SHORT",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
        stop_loss=None,
        take_profit=90.0,
    )
    assert d_only_tp.take_profit == 90.0

    # Invalid LONG stop_loss >= reference_price
    with pytest.raises(InvalidDecisionError, match="LONG stop_loss .* must be < reference_price"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            stop_loss=100.0,
        )

    # Invalid SHORT take_profit >= reference_price
    with pytest.raises(InvalidDecisionError, match="SHORT take_profit .* must be < reference_price"):
        CanonicalDecision(
            cohort_id="c1",
            symbol="BTCUSDT",
            window_id="w1",
            decision_provider="flow_v1",
            strategy_version="1.0",
            signal_timestamp=1000,
            decision_timestamp=1020,
            available_at=1050,
            side="SHORT",
            reference_price=100.0,
            notional_usdt=100.0,
            horizon_s=60,
            take_profit=105.0,
        )


def test_neutral_and_unknown_generate_rejection():
    """NEUTRAL and UNKNOWN decisions must be rejected with NO_DIRECTION reason."""
    executor = PaperExecutor()

    d_neutral = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="NEUTRAL",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    order, rej = executor.submit_decision(d_neutral)
    assert order is None
    assert rej is not None
    assert rej.reason == "NO_DIRECTION"

    d_unknown = CanonicalDecision(
        cohort_id="c1",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="UNKNOWN",
        reference_price=100.0,
        notional_usdt=100.0,
        horizon_s=60,
    )
    order, rej = executor.submit_decision(d_unknown)
    assert order is None
    assert rej is not None
    assert rej.reason == "NO_DIRECTION"


def test_confidence_none_preserved():
    """GAP-1: confidence=None is valid and preserved as None (uncalibrated/unknown)."""
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
        notional_usdt=100.0,
        horizon_s=60,
        confidence=None,
    )
    assert d.confidence is None


def test_confidence_zero_preserved():
    """GAP-1: confidence=0.0 is valid and strictly preserved as 0.0 (not converted to None)."""
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
        notional_usdt=100.0,
        horizon_s=60,
        confidence=0.0,
    )
    assert d.confidence == 0.0
    assert d.confidence is not None


def test_confidence_one_preserved():
    """GAP-1: confidence=1.0 is valid and strictly preserved as 1.0."""
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
        notional_usdt=100.0,
        horizon_s=60,
        confidence=1.0,
    )
    assert d.confidence == 1.0


def test_confidence_negative_rejected():
    """GAP-1: confidence=-0.0001 is invalid and raises InvalidDecisionError."""
    with pytest.raises(InvalidDecisionError, match="Invalid confidence"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            confidence=-0.0001,
        )


def test_confidence_greater_than_one_rejected():
    """GAP-1: confidence=1.0001 is invalid and raises InvalidDecisionError."""
    with pytest.raises(InvalidDecisionError, match="Invalid confidence"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            confidence=1.0001,
        )


def test_confidence_nan_rejected():
    """GAP-1: confidence=NaN is invalid and raises InvalidDecisionError."""
    with pytest.raises(InvalidDecisionError, match="Invalid confidence"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            confidence=float("nan"),
        )


def test_confidence_positive_inf_rejected():
    """GAP-1: confidence=+Inf is invalid and raises InvalidDecisionError."""
    with pytest.raises(InvalidDecisionError, match="Invalid confidence"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            confidence=float("inf"),
        )


def test_confidence_negative_inf_rejected():
    """GAP-1: confidence=-Inf is invalid and raises InvalidDecisionError."""
    with pytest.raises(InvalidDecisionError, match="Invalid confidence"):
        CanonicalDecision(
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
            notional_usdt=100.0,
            horizon_s=60,
            confidence=float("-inf"),
        )


def test_nan_inf_sanitization_rejected():
    """Invariant 14: Context with NaN/Inf values is properly sanitized to None."""
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
        notional_usdt=100.0,
        horizon_s=60,
        context={"val_nan": float("nan"), "val_inf": float("inf"), "safe_val": 42},
    )
    assert d.context["val_nan"] is None
    assert d.context["val_inf"] is None
    assert d.context["safe_val"] == 42
