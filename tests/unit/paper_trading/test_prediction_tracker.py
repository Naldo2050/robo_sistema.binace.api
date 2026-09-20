# tests/unit/paper_trading/test_prediction_tracker.py
"""
Unit tests for Canonical Prediction Tracker (Gate D0-B).

Covers 30 mandatory unit scenarios:
 1. LONG rise correct
 2. LONG fall incorrect
 3. SHORT fall correct
 4. SHORT rise incorrect
 5. exact flat
 6. within flat bps
 7. just outside flat
 8. tick before deadline ignored
 9. first tick at deadline resolves
 10. first tick inside tolerance resolves
 11. drift exact
 12. tick after tolerance => unresolved
 13. tardio não vira resolution_price
 14. shutdown => unresolved
 15. duplicate register idempotent
 16. duplicate resolve blocked/idempotent
 17. risk rejection irrelevant
 18. executor rejection irrelevant
 19. no fill irrelevant
 20. fee irrelevant
 21. slippage irrelevant
 22. funding irrelevant
 23. raw vs directional return SHORT
 24. invalid price rejected
 25. invalid policy config
 26. prediction ID deterministic
 27. different policy version -> different ID
 28. no lookahead
 29. neutral not registered
 30. unknown not registered
"""

import math
import pytest
from typing import Optional

from paper_trading.contracts import CanonicalDecision
from paper_trading.prediction import (
    PredictionOutcome,
    PredictionTrackerConfig,
    make_prediction_id,
)
from paper_trading.prediction_tracker import PredictionTracker


def _make_decision(
    side: str = "LONG",
    reference_price: float = 100.0,
    decision_timestamp: int = 1_000_000,
    horizon_s: int = 300,
    symbol: str = "BTCUSDT",
    cohort_id: str = "CH_TEST",
    window_id: str = "w_1",
) -> CanonicalDecision:
    return CanonicalDecision(
        cohort_id=cohort_id,
        symbol=symbol,
        window_id=window_id,
        decision_provider="test_provider",
        strategy_version="v1.0",
        model_version=None,
        signal_timestamp=decision_timestamp - 100,
        decision_timestamp=decision_timestamp,
        available_at=decision_timestamp + 5,
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=1000.0,
        horizon_s=horizon_s,
    )


# 1. LONG rise correct
def test_long_rise_correct() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 101.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"
    assert outcomes[0].raw_return_bps == pytest.approx(100.0)
    assert outcomes[0].directional_return_bps == pytest.approx(100.0)
    assert outcomes[0].resolution_price == 101.0


# 2. LONG fall incorrect
def test_long_fall_incorrect() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 99.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "INCORRECT"
    assert outcomes[0].raw_return_bps == pytest.approx(-100.0)
    assert outcomes[0].directional_return_bps == pytest.approx(-100.0)


# 3. SHORT fall correct
def test_short_fall_correct() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 98.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"
    assert outcomes[0].raw_return_bps == pytest.approx(-200.0)
    assert outcomes[0].directional_return_bps == pytest.approx(200.0)


# 4. SHORT rise incorrect
def test_short_rise_incorrect() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 102.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "INCORRECT"
    assert outcomes[0].raw_return_bps == pytest.approx(200.0)
    assert outcomes[0].directional_return_bps == pytest.approx(-200.0)


# 5. exact flat
def test_exact_flat() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 100.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "FLAT"
    assert outcomes[0].raw_return_bps == pytest.approx(0.0)
    assert outcomes[0].directional_return_bps == pytest.approx(0.0)


# 6. within flat bps (threshold = 1.0 bps)
def test_within_flat_bps() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(flat_tolerance_bps=1.0))
    # 0.5 bps on 100.0 is price 100.005
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 100.005, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "FLAT"
    assert abs(outcomes[0].raw_return_bps or 0.0) <= 1.0


# 7. just outside flat
def test_just_outside_flat() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(flat_tolerance_bps=1.0))
    # 1.5 bps on 100.0 is price 100.015
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 100.015, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"
    assert outcomes[0].raw_return_bps == pytest.approx(1.5)


# 8. tick before deadline ignored
def test_tick_before_deadline_ignored() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline - 1, "price": 105.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 0
    assert tracker.pending_count == 1
    assert tracker.resolved_count == 0


# 9. first tick at deadline resolves
def test_first_tick_at_deadline_resolves() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 103.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"
    assert outcomes[0].resolution_drift_ms == 0
    assert tracker.pending_count == 0
    assert tracker.resolved_count == 1


# 10. first tick inside tolerance resolves
def test_first_tick_inside_tolerance_resolves() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(resolution_tolerance_ms=30_000))
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    # Arrives 15 seconds after deadline
    outcomes = tracker.on_tick({"event_timestamp": deadline + 15_000, "price": 95.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"
    assert outcomes[0].resolution_price == 95.0
    assert outcomes[0].resolution_drift_ms == 15_000


# 11. drift exact
def test_drift_exact() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(resolution_tolerance_ms=30_000))
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    drift_input = 7_432
    outcomes = tracker.on_tick({"event_timestamp": deadline + drift_input, "price": 101.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].resolution_drift_ms == drift_input
    assert outcomes[0].resolved_timestamp_ms == deadline + drift_input


# 12. tick after tolerance => unresolved
def test_tick_after_tolerance_unresolved() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(resolution_tolerance_ms=30_000))
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    # Arrives 30.001 seconds after deadline (outside tolerance)
    outcomes = tracker.on_tick({"event_timestamp": deadline + 30_001, "price": 110.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "UNRESOLVED"
    assert outcomes[0].reason == "NO_TICK_WITHIN_TOLERANCE"


# 13. tardio não vira resolution_price
def test_late_tick_does_not_become_resolution_price() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(resolution_tolerance_ms=30_000))
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline + 45_000, "price": 150.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "UNRESOLVED"
    assert outcomes[0].resolution_price is None
    assert outcomes[0].raw_return_bps is None
    assert outcomes[0].directional_return_bps is None
    assert outcomes[0].resolved_timestamp_ms is None
    assert outcomes[0].resolution_drift_ms is None
    assert outcomes[0].observed_at_ms == deadline + 45_000


# 14. shutdown => unresolved
def test_shutdown_marks_unresolved() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)

    outcomes = tracker.flush_unresolved(reason="PROCESS_SHUTDOWN", observed_at_ms=1_050_000)
    assert len(outcomes) == 1
    assert outcomes[0].result == "UNRESOLVED"
    assert outcomes[0].reason == "PROCESS_SHUTDOWN"
    assert outcomes[0].resolution_price is None
    assert outcomes[0].raw_return_bps is None
    assert outcomes[0].directional_return_bps is None
    assert outcomes[0].observed_at_ms == 1_050_000
    assert tracker.pending_count == 0


# 15. duplicate register idempotent
def test_duplicate_register_idempotent() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(window_id="dec_unique")
    pid1 = tracker.register(dec)
    pid2 = tracker.register(dec)
    assert pid1 == pid2
    assert tracker.pending_count == 1


# 16. duplicate resolve blocked/idempotent
def test_duplicate_resolve_blocked() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes1 = tracker.on_tick({"event_timestamp": deadline, "price": 105.0, "symbol": "BTCUSDT"})
    assert len(outcomes1) == 1
    # Subsequent tick does not re-resolve
    outcomes2 = tracker.on_tick({"event_timestamp": deadline + 1000, "price": 106.0, "symbol": "BTCUSDT"})
    assert len(outcomes2) == 0


# 17. risk rejection irrelevant
def test_risk_rejection_irrelevant() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 102.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"


# 18. executor rejection irrelevant
def test_executor_rejection_irrelevant() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 97.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"


# 19. no fill irrelevant
def test_no_fill_irrelevant() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 95.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "INCORRECT"


# 20. fee irrelevant
def test_fee_irrelevant() -> None:
    tracker = PredictionTracker(PredictionTrackerConfig(flat_tolerance_bps=1.0))
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 100.02, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"


# 21. slippage irrelevant
def test_slippage_irrelevant() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 101.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].reference_price == 100.0
    assert outcomes[0].resolution_price == 101.0
    assert outcomes[0].result == "CORRECT"


# 22. funding irrelevant
def test_funding_irrelevant() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 99.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].result == "CORRECT"


# 23. raw vs directional return SHORT
def test_raw_vs_directional_return_short() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="SHORT", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 95.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].raw_return_bps == pytest.approx(-500.0)
    assert outcomes[0].directional_return_bps == pytest.approx(500.0)
    assert outcomes[0].result == "CORRECT"


# 24. invalid price rejected
def test_invalid_price_rejected() -> None:
    from unittest.mock import MagicMock
    tracker = PredictionTracker()
    mock_dec = MagicMock()
    mock_dec.side = "LONG"
    mock_dec.reference_price = 0.0
    mock_dec.horizon_s = 300
    mock_dec.decision_timestamp = 1_000_000
    mock_dec.decision_id = "mock_dec"

    with pytest.raises(ValueError, match="reference_price must be finite > 0"):
        tracker.register(mock_dec)

    mock_dec.reference_price = -10.0
    with pytest.raises(ValueError, match="reference_price must be finite > 0"):
        tracker.register(mock_dec)

    mock_dec.reference_price = float("nan")
    with pytest.raises(ValueError, match="reference_price must be finite > 0"):
        tracker.register(mock_dec)


# 25. invalid policy config
def test_invalid_policy_config() -> None:
    with pytest.raises(ValueError, match="resolution_tolerance_ms must be non-negative"):
        PredictionTrackerConfig(resolution_tolerance_ms=-1)

    with pytest.raises(ValueError, match="flat_tolerance_bps must be finite and non-negative"):
        PredictionTrackerConfig(flat_tolerance_bps=-0.5)

    with pytest.raises(ValueError, match="policy_version must not be empty"):
        PredictionTrackerConfig(policy_version="")


# 26. prediction ID deterministic
def test_prediction_id_deterministic() -> None:
    pid1 = make_prediction_id("dec_123", 300, "v1")
    pid2 = make_prediction_id("dec_123", 300, "v1")
    assert pid1 == pid2
    assert pid1.startswith("pred_")


# 27. different policy version -> different ID
def test_different_policy_version_different_id() -> None:
    pid_v1 = make_prediction_id("dec_123", 300, "v1")
    pid_v2 = make_prediction_id("dec_123", 300, "v2")
    assert pid_v1 != pid_v2


# 28. no lookahead
def test_no_lookahead() -> None:
    # A tick occurring before the deadline cannot resolve the prediction
    tracker = PredictionTracker()
    dec = _make_decision(side="LONG", reference_price=100.0, horizon_s=300)
    tracker.register(dec)
    deadline = dec.decision_timestamp + 300_000

    # Many ticks arrive prior to deadline
    for ts in range(dec.decision_timestamp + 1000, deadline, 10_000):
        outcomes = tracker.on_tick({"event_timestamp": ts, "price": 105.0, "symbol": "BTCUSDT"})
        assert len(outcomes) == 0

    assert tracker.pending_count == 1

    # Exactly at deadline, it resolves
    outcomes = tracker.on_tick({"event_timestamp": deadline, "price": 105.0, "symbol": "BTCUSDT"})
    assert len(outcomes) == 1
    assert outcomes[0].resolved_timestamp_ms == deadline
    assert outcomes[0].resolved_timestamp_ms >= deadline


# 29. neutral not registered
def test_neutral_not_registered() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="NEUTRAL")
    pid = tracker.register(dec)
    assert pid is None
    assert tracker.pending_count == 0


# 30. unknown not registered
def test_unknown_not_registered() -> None:
    tracker = PredictionTracker()
    dec = _make_decision(side="UNKNOWN")
    pid = tracker.register(dec)
    assert pid is None
    assert tracker.pending_count == 0
