# tests/unit/paper_trading/test_crash_safe_scorecard.py
"""
Unit tests for Crash-Safe Prediction Accounting & Lifecycle Classification (Gate D0-B.1).

Covers all 10 mandatory scenarios:
  1. Graceful zero missing
  2. Graceful missing => integrity gap fail
  3. Crash missing => interrupted
  4. Crash conservation passes
  5. Missing excluded from accuracy
  6. Missing excluded from flat rate
  7. Unresolved distinct from interrupted
  8. D0-C style 2 decisions / 0 outcomes => interrupted=2
  9. No decisions
  10. Unknown lifecycle

Plus end-to-end pipeline fixtures (graceful, crash, integrity gap)
and forensic verification of the preserved D0-C real database.
"""

from __future__ import annotations

import hashlib
import os
import pytest
from typing import List, Optional

from paper_trading.contracts import CanonicalDecision
from paper_trading.ledger import PaperLedger
from paper_trading.prediction import (
    PredictionOutcome,
    PredictionResult,
    PredictionTrackerConfig,
    make_prediction_id,
)
from paper_trading.prediction_tracker import PredictionTracker
from paper_trading.scorecard import (
    classify_cohort_lifecycle,
    prediction_scorecard,
)


def _make_pred_outcome(
    pred_id: str,
    result: PredictionResult,
    cohort_id: str = "c_test",
    decision_timestamp: int = 1_000_000,
    horizon_s: int = 300,
    side: str = "LONG",
    reference_price: float = 100.0,
    reason: Optional[str] = None,
) -> PredictionOutcome:
    deadline = decision_timestamp + (horizon_s * 1000)
    is_terminal_resolved = result in ("CORRECT", "INCORRECT", "FLAT")
    res_price = 101.0 if result == "CORRECT" else (99.0 if result == "INCORRECT" else (100.0 if result == "FLAT" else None))
    raw_ret = 100.0 if result == "CORRECT" else (-100.0 if result == "INCORRECT" else (0.0 if result == "FLAT" else None))

    return PredictionOutcome(
        prediction_id=pred_id,
        decision_id=f"dec_{pred_id}",
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        side=side,
        reference_price=reference_price,
        decision_timestamp=decision_timestamp,
        horizon_s=horizon_s,
        deadline_ms=deadline,
        result=result,
        reason=reason or ("HORIZON_RESOLVED" if is_terminal_resolved else "NO_TICK_WITHIN_TOLERANCE"),
        resolution_price=res_price,
        raw_return_bps=raw_ret,
        directional_return_bps=raw_ret,
        resolved_timestamp_ms=deadline if is_terminal_resolved else None,
        resolution_drift_ms=0 if is_terminal_resolved else None,
        observed_at_ms=deadline,
        flat_tolerance_bps=1.0,
        resolution_tolerance_ms=30_000,
        policy_version="v1",
        created_at_ms=deadline,
    )


def _make_decision(
    decision_id: str,
    side: str = "LONG",
    cohort_id: str = "c_test",
    reference_price: float = 100.0,
    decision_timestamp: int = 1_000_000,
    horizon_s: int = 300,
) -> CanonicalDecision:
    return CanonicalDecision(
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        window_id=f"win_{decision_id}",
        decision_provider="test_prov",
        strategy_version="v1",
        model_version=None,
        signal_timestamp=decision_timestamp - 100,
        decision_timestamp=decision_timestamp,
        available_at=decision_timestamp + 5,
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=100.0,
        horizon_s=horizon_s,
    )


# 1. Graceful zero missing
def test_1_graceful_zero_missing() -> None:
    cohort_events = [
        {"event_type": "STARTED", "timestamp_ms": 1_000},
        {"event_type": "GRACEFUL_SHUTDOWN", "timestamp_ms": 2_000},
    ]
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(5)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(3)]
        + [_make_pred_outcome(f"u_{i}", "UNRESOLVED", reason="PROCESS_SHUTDOWN") for i in range(2)]
    )

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
        cohort_events=cohort_events,
    )

    assert score.total_directional_decisions == 10
    assert score.terminal_prediction_outcomes == 10
    assert score.missing_prediction_outcomes == 0
    assert score.interrupted == 0
    assert score.cohort_lifecycle == "GRACEFUL"
    assert score.prediction_accounting == "PASS"
    assert score.prediction_integrity == "PASS"
    assert score.conservation_passed is True
    assert score.interrupted_rate == 0.0
    assert score.terminal_outcome_coverage == 1.0


# 2. Graceful missing => integrity fail
def test_2_graceful_missing_integrity_fail() -> None:
    cohort_events = [
        {"event_type": "STARTED", "timestamp_ms": 1_000},
        {"event_type": "GRACEFUL_SHUTDOWN", "timestamp_ms": 2_000},
    ]
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(5)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(3)]
        + [_make_pred_outcome(f"u_{i}", "UNRESOLVED", reason="PROCESS_SHUTDOWN") for i in range(1)]
    )  # 9 outcomes recorded, but 10 directional decisions

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
        cohort_events=cohort_events,
    )

    assert score.total_directional_decisions == 10
    assert score.terminal_prediction_outcomes == 9
    assert score.missing_prediction_outcomes == 1
    assert score.cohort_lifecycle == "GRACEFUL"
    assert score.prediction_integrity == "FAIL"  # INTEGRITY_GAP detected!
    assert score.conservation_passed is True  # 10 = 9 + 1 still holds
    assert score.prediction_accounting == "PASS"


# 3. Crash missing => interrupted
def test_3_crash_missing_interrupted() -> None:
    cohort_events = [
        {"event_type": "STARTED", "timestamp_ms": 1_000},
    ]  # GRACEFUL_SHUTDOWN absent
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(4)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(2)]
        + [_make_pred_outcome(f"f_{i}", "FLAT") for i in range(1)]
    )  # 7 outcomes for 10 decisions

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
        cohort_events=cohort_events,
    )

    assert score.total_directional_decisions == 10
    assert score.terminal_prediction_outcomes == 7
    assert score.missing_prediction_outcomes == 3
    assert score.interrupted == 3
    assert score.cohort_lifecycle == "INCOMPLETE"
    assert score.prediction_integrity == "PASS"
    assert score.prediction_accounting == "PASS"
    assert score.interrupted_rate == pytest.approx(0.3)


# 4. Crash conservation passes
def test_4_crash_conservation_passes() -> None:
    cohort_events = [{"event_type": "STARTED", "timestamp_ms": 1_000}]
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(4)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(2)]
        + [_make_pred_outcome("f_0", "FLAT")]
    )  # 7 terminal

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
        cohort_events=cohort_events,
    )

    # D = C + I + F + U + missing: 10 == 4 + 2 + 1 + 0 + 3
    assert score.conservation_passed is True
    assert score.total_directional_decisions == (
        score.correct + score.incorrect + score.flat + score.unresolved + score.missing_prediction_outcomes
    )
    assert score.prediction_accounting == "PASS"


# 5. Missing excluded from accuracy
def test_5_missing_excluded_accuracy() -> None:
    # 10 decisions: 4 CORRECT, 2 INCORRECT, 1 FLAT, 1 UNRESOLVED, 2 MISSING
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(4)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(2)]
        + [_make_pred_outcome("f_0", "FLAT")]
        + [_make_pred_outcome("u_0", "UNRESOLVED")]
    )

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
    )

    assert score.prediction_directional_n == 6  # 4 + 2 (FLAT, UNRESOLVED, and MISSING excluded)
    assert score.directional_accuracy == pytest.approx(4.0 / 6.0)
    assert score.directional_accuracy == pytest.approx(0.66666667)


# 6. Missing excluded from flat rate
def test_6_missing_excluded_flat_rate() -> None:
    # 10 decisions: 4 CORRECT, 2 INCORRECT, 1 FLAT, 1 UNRESOLVED, 2 MISSING
    outcomes = (
        [_make_pred_outcome(f"c_{i}", "CORRECT") for i in range(4)]
        + [_make_pred_outcome(f"i_{i}", "INCORRECT") for i in range(2)]
        + [_make_pred_outcome("f_0", "FLAT")]
        + [_make_pred_outcome("u_0", "UNRESOLVED")]
    )

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=10,
    )

    assert score.observed_n == 7  # 4 + 2 + 1
    assert score.flat_rate == pytest.approx(1.0 / 7.0)


# 7. Unresolved distinct from interrupted
def test_7_unresolved_distinct_from_interrupted() -> None:
    # 5 decisions: 2 CORRECT, 1 UNRESOLVED, 2 MISSING (no outcome records)
    outcomes = [
        _make_pred_outcome("c_0", "CORRECT"),
        _make_pred_outcome("c_1", "CORRECT"),
        _make_pred_outcome("u_0", "UNRESOLVED", reason="NO_TICK_WITHIN_TOLERANCE"),
    ]

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=5,
    )

    assert score.unresolved == 1
    assert score.interrupted == 2
    assert score.missing_prediction_outcomes == 2
    assert score.unresolved != score.interrupted
    # Missing decisions are NEVER automatically labeled as UNRESOLVED
    assert score.terminal_prediction_outcomes == 3


# 8. D0-C style 2 decisions / 0 outcomes => interrupted=2
def test_8_d0c_style_reconstruction() -> None:
    cohort_events = [{"event_type": "STARTED", "timestamp_ms": 1789947441809}]
    outcomes: List[PredictionOutcome] = []

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=2,
        cohort_events=cohort_events,
    )

    assert score.total_directional_decisions == 2
    assert score.terminal_prediction_outcomes == 0
    assert score.missing_prediction_outcomes == 2
    assert score.interrupted == 2
    assert score.correct == 0
    assert score.incorrect == 0
    assert score.flat == 0
    assert score.unresolved == 0
    assert score.conservation_passed is True  # 2 = 0 + 0 + 0 + 0 + 2
    assert score.cohort_lifecycle == "INCOMPLETE"
    assert score.prediction_accounting == "PASS"
    assert score.prediction_integrity == "PASS"
    assert score.directional_accuracy is None
    assert score.terminal_outcome_coverage == 0.0
    assert score.interrupted_rate == 1.0


# 9. No decisions
def test_9_no_decisions() -> None:
    score = prediction_scorecard(
        prediction_outcomes=[],
        total_directional_decisions=0,
    )

    assert score.total_directional_decisions == 0
    assert score.terminal_prediction_outcomes == 0
    assert score.missing_prediction_outcomes == 0
    assert score.interrupted == 0
    assert score.conservation_passed is True
    assert score.directional_accuracy is None
    assert score.horizon_observation_coverage is None
    assert score.terminal_outcome_coverage is None
    assert score.prediction_accounting_coverage is None
    assert score.interrupted_rate is None


# 10. Unknown lifecycle
def test_10_unknown_lifecycle() -> None:
    outcomes = [
        _make_pred_outcome("c_0", "CORRECT"),
        _make_pred_outcome("i_0", "INCORRECT"),
        _make_pred_outcome("f_0", "FLAT"),
    ]

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=5,
        cohort_events=[],  # No events
    )

    assert score.cohort_lifecycle == "UNKNOWN"
    assert score.missing_prediction_outcomes == 2
    assert score.interrupted == 2
    assert score.conservation_passed is True
    assert score.prediction_accounting == "PASS"


# 11. Pipeline Graceful Fixture (Section 7)
def test_11_fixture_graceful_pipeline_e2e() -> None:
    ledger = PaperLedger(db_path=":memory:")
    tracker = PredictionTracker()
    cohort_id = "CH_GRACEFUL_10"

    ledger.record_cohort_event(cohort_id, "STARTED", 1_000_000)

    # 10 decisions
    for i in range(10):
        dec = _make_decision(f"dec_{i}", cohort_id=cohort_id, decision_timestamp=1_000_000 + (i * 1000))
        ledger.record_decision(dec)
        tracker.register(dec)

    # 8 resolved by ticks (deadline = 1_000_000 + i*1000 + 300_000)
    for i in range(8):
        deadline = 1_000_000 + (i * 1000) + 300_000
        price = 102.0 if i % 2 == 0 else 98.0
        outs = tracker.on_tick({"event_timestamp": deadline, "price": price, "symbol": "BTCUSDT"})
        for o in outs:
            ledger.record_prediction_outcome(o)

    assert tracker.pending_count == 2
    assert tracker.resolved_count == 8

    # Graceful shutdown flushes remaining 2 pendings as UNRESOLVED (PROCESS_SHUTDOWN)
    unresolved_outs = tracker.flush_unresolved(reason="PROCESS_SHUTDOWN", observed_at_ms=1_500_000)
    for u in unresolved_outs:
        ledger.record_prediction_outcome(u)

    ledger.record_cohort_event(cohort_id, "GRACEFUL_SHUTDOWN", 1_500_000)
    ledger.flush()

    # Query ledger and evaluate scorecard
    decisions = ledger.get_decisions(cohort_id)
    outcomes = ledger.get_prediction_outcomes(cohort_id)
    events = ledger.get_cohort_events(cohort_id)
    ledger.close()

    assert len(decisions) == 10
    assert len(outcomes) == 10

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=len(decisions),
        cohort_events=events,
        cohort_id=cohort_id,
    )

    assert score.total_directional_decisions == 10
    assert score.terminal_prediction_outcomes == 10
    assert score.missing_prediction_outcomes == 0
    assert score.unresolved == 2
    assert score.cohort_lifecycle == "GRACEFUL"
    assert score.prediction_integrity == "PASS"
    assert score.conservation_passed is True


# 12. Pipeline Crash Fixture (Section 8)
def test_12_fixture_crash_pipeline_e2e() -> None:
    ledger = PaperLedger(db_path=":memory:")
    tracker = PredictionTracker()
    cohort_id = "CH_CRASH_10"

    ledger.record_cohort_event(cohort_id, "STARTED", 1_000_000)

    # 10 decisions
    for i in range(10):
        dec = _make_decision(f"dec_{i}", cohort_id=cohort_id, decision_timestamp=1_000_000 + (i * 1000))
        ledger.record_decision(dec)
        tracker.register(dec)

    # 7 resolved by ticks
    for i in range(7):
        deadline = 1_000_000 + (i * 1000) + 300_000
        price = 101.0
        outs = tracker.on_tick({"event_timestamp": deadline, "price": price, "symbol": "BTCUSDT"})
        for o in outs:
            ledger.record_prediction_outcome(o)

    # Crash! No flush_unresolved, no GRACEFUL_SHUTDOWN
    ledger.flush()

    decisions = ledger.get_decisions(cohort_id)
    outcomes = ledger.get_prediction_outcomes(cohort_id)
    events = ledger.get_cohort_events(cohort_id)
    ledger.close()

    assert len(decisions) == 10
    assert len(outcomes) == 7

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=len(decisions),
        cohort_events=events,
        cohort_id=cohort_id,
    )

    assert score.terminal_prediction_outcomes == 7
    assert score.interrupted == 3
    assert score.missing_prediction_outcomes == 3
    assert score.conservation_passed is True
    assert score.cohort_lifecycle == "INCOMPLETE"
    assert score.prediction_accounting == "PASS"


# 13. Pipeline Integrity Gap Fixture (Section 9)
def test_13_fixture_integrity_gap_pipeline_e2e() -> None:
    ledger = PaperLedger(db_path=":memory:")
    cohort_id = "CH_GAP_10"

    ledger.record_cohort_event(cohort_id, "STARTED", 1_000_000)

    # 10 decisions
    for i in range(10):
        dec = _make_decision(f"dec_{i}", cohort_id=cohort_id, decision_timestamp=1_000_000 + (i * 1000))
        ledger.record_decision(dec)

    # 9 outcomes recorded (1 accidentally omitted)
    for i in range(9):
        outcome = _make_pred_outcome(f"p_{i}", "CORRECT", cohort_id=cohort_id)
        ledger.record_prediction_outcome(outcome)

    # GRACEFUL_SHUTDOWN recorded, but missing == 1!
    ledger.record_cohort_event(cohort_id, "GRACEFUL_SHUTDOWN", 1_500_000)
    ledger.flush()

    decisions = ledger.get_decisions(cohort_id)
    outcomes = ledger.get_prediction_outcomes(cohort_id)
    events = ledger.get_cohort_events(cohort_id)
    ledger.close()

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=len(decisions),
        cohort_events=events,
        cohort_id=cohort_id,
    )

    assert score.missing_prediction_outcomes == 1
    assert score.cohort_lifecycle == "GRACEFUL"
    assert score.prediction_integrity == "FAIL"  # INTEGRITY_GAP!
    assert score.conservation_passed is True


# 14. Preserved Real DB D0-C Forensic Replay (Section 1, 6, 13)
def test_14_d0c_real_preserved_db_replay() -> None:
    real_db_path = "dados/paper_live_d0c_20260920T233712Z.db"
    assert os.path.exists(real_db_path), f"Preserved DB file {real_db_path} not found"

    # Compute SHA-256 before reading
    with open(real_db_path, "rb") as f:
        hash_before = hashlib.sha256(f.read()).hexdigest().upper()

    assert hash_before == "BE9BB42E5F9312811C6EA7A738B21D113B06DC9CB67E5FB2B4BF093706B0D21B"

    ledger = PaperLedger(db_path=real_db_path, read_only=True)
    decisions = ledger.get_decisions()
    outcomes = ledger.get_prediction_outcomes()
    events = ledger.get_cohort_events()
    ledger.close()

    # Directional decisions = 2 (both SHORT)
    dir_decisions = [d for d in decisions if d.side in ("LONG", "SHORT")]
    assert len(dir_decisions) == 2
    assert len(outcomes) == 0

    score = prediction_scorecard(
        prediction_outcomes=outcomes,
        total_directional_decisions=len(dir_decisions),
        cohort_events=events,
    )

    assert score.total_directional_decisions == 2
    assert score.terminal_prediction_outcomes == 0
    assert score.missing_prediction_outcomes == 2
    assert score.interrupted == 2
    assert score.conservation_passed is True  # 2 = 0 + 0 + 0 + 0 + 2
    assert score.cohort_lifecycle == "INCOMPLETE"
    assert score.prediction_accounting == "PASS"
    assert score.prediction_integrity == "PASS"
    assert score.directional_accuracy is None

    # Compute SHA-256 after reading to ensure strict byte-level preservation
    with open(real_db_path, "rb") as f:
        hash_after = hashlib.sha256(f.read()).hexdigest().upper()

    assert hash_after == hash_before
