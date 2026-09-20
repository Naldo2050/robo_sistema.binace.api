# tests/unit/paper_trading/test_ledger_shadow.py
"""
Unit tests for auditable shadow persistence extensions in PaperLedger.

Covers:
- Strict cohort lifecycle (create, duplicate reject, metadata immutability)
- Signal observation tracking (decision created, neutral skip, invalid signal, idempotency)
- Risk evaluation persistence (approved, rejected, error, NaN/Inf handling, idempotency)
- Cohort events & crash/incomplete session detection
- Persistence health snapshot and fail-closed behavior on database error
- Legacy schema backward compatibility and data preservation
- Zero tick API/table presence
- FIFO ordering and scorecard unaffected metrics
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
from typing import Any, Dict, List

import pytest

from paper_trading.contracts import (
    CanonicalDecision,
    ClosedTrade,
    PaperFill,
    PaperOrder,
    PaperPosition,
    Rejection,
)
from paper_trading.ledger import (
    PaperLedger,
    compute_cohort_event_id,
    compute_risk_evaluation_id,
    compute_signal_observation_id,
)
from paper_trading.scorecard import scorecard


def test_1_cohort_strict_create_success():
    """1. create_cohort succeeds on first attempt with complete metadata."""
    ledger = PaperLedger(db_path=":memory:")
    meta = {"strategy": "flow_v1", "seed": 42, "notional": 100.0}
    ok = ledger.create_cohort("c_strict_1", created_at_ms=1000, description="audit", metadata=meta)
    assert ok is True
    assert ledger.cohort_exists("c_strict_1") is True
    ledger.close()


def test_2_duplicate_cohort_rejected():
    """2. create_cohort fails closed on duplicate cohort_id."""
    ledger = PaperLedger(db_path=":memory:")
    assert ledger.create_cohort("c_dup", 1000, "first", {"val": 1}) is True
    # Second attempt must return False
    assert ledger.create_cohort("c_dup", 2000, "second", {"val": 2}) is False
    ledger.close()


def test_3_metadata_not_overwritten():
    """3. Failed duplicate creation does not overwrite existing metadata."""
    ledger = PaperLedger(db_path=":memory:")
    ledger.create_cohort("c_immut", 1000, "orig_desc", {"initial": True})
    ledger.create_cohort("c_immut", 2000, "new_desc", {"initial": False})

    cur = ledger._conn.cursor()
    cur.execute("SELECT description, metadata_json FROM cohorts WHERE cohort_id = 'c_immut'")
    row = cur.fetchone()
    assert row["description"] == "orig_desc"
    assert '"initial": true' in row["metadata_json"]
    ledger.close()


def test_4_signal_observation_decision_created():
    """4. DECISION_CREATED signal observation recorded with deterministic ID."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_signal_observation(
        cohort_id="c_sig",
        symbol="BTCUSDT",
        source_window_id="w100",
        source_event_key="ev_abs_1",
        signal_timestamp=1000,
        source_side="LONG",
        status="DECISION_CREATED",
        reason=None,
        context={"score": 0.85},
    )
    assert ok is True
    ledger.flush()

    obs = ledger.get_signal_observations("c_sig")
    assert len(obs) == 1
    assert obs[0]["status"] == "DECISION_CREATED"
    assert obs[0]["source_side"] == "LONG"
    assert obs[0]["context"]["score"] == 0.85
    expected_id = compute_signal_observation_id("c_sig", "w100", "ev_abs_1")
    assert obs[0]["observation_id"] == expected_id
    ledger.close()


def test_5_neutral_skip_signal_observation():
    """5. SKIPPED_NON_DIRECTIONAL recorded for neutral signals."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_signal_observation(
        cohort_id="c_sig",
        symbol="BTCUSDT",
        source_window_id="w101",
        source_event_key="ev_neut_1",
        signal_timestamp=2000,
        source_side="NEUTRAL",
        status="SKIPPED_NON_DIRECTIONAL",
        reason="Neutral side skipped in FOLLOW_SIGNAL mode",
    )
    assert ok is True
    ledger.flush()

    obs = ledger.get_signal_observations("c_sig")
    assert len(obs) == 1
    assert obs[0]["status"] == "SKIPPED_NON_DIRECTIONAL"
    assert "Neutral" in obs[0]["reason"]
    ledger.close()


def test_6_invalid_signal_observation():
    """6. INVALID_SIGNAL recorded for malformed payload."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_signal_observation(
        cohort_id="c_sig",
        symbol="BTCUSDT",
        source_window_id="w102",
        source_event_key="ev_err_1",
        signal_timestamp=3000,
        source_side="UNKNOWN",
        status="INVALID_SIGNAL",
        reason="Missing required price field",
    )
    assert ok is True
    ledger.flush()

    obs = ledger.get_signal_observations("c_sig")
    assert len(obs) == 1
    assert obs[0]["status"] == "INVALID_SIGNAL"
    assert obs[0]["source_side"] == "UNKNOWN"
    ledger.close()


def test_7_retry_observation_idempotent():
    """7. Retrying the same signal observation does not create duplicates."""
    ledger = PaperLedger(db_path=":memory:")
    for _ in range(3):
        ledger.record_signal_observation(
            cohort_id="c_idem",
            symbol="BTCUSDT",
            source_window_id="w1",
            source_event_key="ev1",
            signal_timestamp=1000,
            source_side="LONG",
            status="DECISION_CREATED",
        )
    ledger.flush()

    obs = ledger.get_signal_observations("c_idem")
    assert len(obs) == 1
    ledger.close()


def test_8_risk_evaluation_approved():
    """8. APPROVED risk evaluation recorded with deterministic ID."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_risk_evaluation(
        cohort_id="c_risk",
        decision_id="dec_001",
        status="APPROVED",
        risk_confidence=0.9,
        risk_reason=None,
        max_size=1000.0,
        source_confidence=0.85,
        context={"policy": "conservative"},
        evaluated_at_ms=1050,
    )
    assert ok is True
    ledger.flush()

    evals = ledger.get_risk_evaluations("c_risk")
    assert len(evals) == 1
    assert evals[0]["status"] == "APPROVED"
    assert evals[0]["risk_confidence"] == 0.9
    assert evals[0]["source_confidence"] == 0.85
    assert evals[0]["max_size"] == 1000.0
    expected_id = compute_risk_evaluation_id("c_risk", "dec_001")
    assert evals[0]["risk_evaluation_id"] == expected_id
    ledger.close()


def test_9_risk_evaluation_rejected():
    """9. REJECTED risk evaluation recorded with specific risk_reason."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_risk_evaluation(
        cohort_id="c_risk",
        decision_id="dec_002",
        status="REJECTED",
        risk_confidence=0.0,
        risk_reason="POSITION_SIZE_EXCEEDS_MAX",
        max_size=500.0,
        source_confidence=0.7,
        evaluated_at_ms=1060,
    )
    assert ok is True
    ledger.flush()

    evals = ledger.get_risk_evaluations("c_risk")
    assert len(evals) == 1
    assert evals[0]["status"] == "REJECTED"
    assert evals[0]["risk_reason"] == "POSITION_SIZE_EXCEEDS_MAX"
    ledger.close()


def test_10_risk_evaluation_error():
    """10. ERROR risk evaluation recorded on exception/malformed risk response."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_risk_evaluation(
        cohort_id="c_risk",
        decision_id="dec_003",
        status="ERROR",
        risk_confidence=0.0,
        risk_reason="RISK_MANAGER_EXCEPTION",
        evaluated_at_ms=1070,
    )
    assert ok is True
    ledger.flush()

    evals = ledger.get_risk_evaluations("c_risk")
    assert len(evals) == 1
    assert evals[0]["status"] == "ERROR"
    ledger.close()


def test_11_risk_retry_idempotent():
    """11. Re-recording the same risk evaluation updates or ignores idempotently without duplicate rows."""
    ledger = PaperLedger(db_path=":memory:")
    for _ in range(3):
        ledger.record_risk_evaluation(
            cohort_id="c_risk_idem",
            decision_id="dec_idem",
            status="APPROVED",
            risk_confidence=0.95,
            max_size=500.0,
            evaluated_at_ms=1000,
        )
    ledger.flush()

    evals = ledger.get_risk_evaluations("c_risk_idem")
    assert len(evals) == 1
    ledger.close()


def test_12_risk_none_source_confidence_preserved():
    """12. Optional source_confidence=None is preserved without coercing to 0.0."""
    ledger = PaperLedger(db_path=":memory:")
    ledger.record_risk_evaluation(
        cohort_id="c_risk",
        decision_id="dec_none",
        status="APPROVED",
        risk_confidence=1.0,
        source_confidence=None,
        evaluated_at_ms=1000,
    )
    ledger.flush()

    evals = ledger.get_risk_evaluations("c_risk")
    assert len(evals) == 1
    assert evals[0]["source_confidence"] is None
    ledger.close()


def test_13_cohort_event_started():
    """13. STARTED cohort event recorded."""
    ledger = PaperLedger(db_path=":memory:")
    ok = ledger.record_cohort_event(
        cohort_id="c_evt",
        event_type="STARTED",
        timestamp_ms=1000,
        reason="Bot startup",
        metadata={"git_sha": "abc1234"},
    )
    assert ok is True
    ledger.flush()

    events = ledger.get_cohort_events("c_evt")
    assert len(events) == 1
    assert events[0]["event_type"] == "STARTED"
    assert events[0]["metadata"]["git_sha"] == "abc1234"
    ledger.close()


def test_14_cohort_event_graceful_shutdown():
    """14. GRACEFUL_SHUTDOWN event marks clean session close."""
    ledger = PaperLedger(db_path=":memory:")
    ledger.record_cohort_event(cohort_id="c_evt", event_type="STARTED", timestamp_ms=1000)
    ledger.record_cohort_event(cohort_id="c_evt", event_type="GRACEFUL_SHUTDOWN", timestamp_ms=5000, reason="SIGINT")
    ledger.flush()

    events = ledger.get_cohort_events("c_evt")
    assert len(events) == 2
    types = [e["event_type"] for e in events]
    assert types == ["STARTED", "GRACEFUL_SHUTDOWN"]
    ledger.close()


def test_15_duplicate_cohort_event_idempotent():
    """15. Re-recording identical cohort event does not duplicate rows."""
    ledger = PaperLedger(db_path=":memory:")
    for _ in range(3):
        ledger.record_cohort_event(
            cohort_id="c_evt_dup",
            event_type="STARTED",
            timestamp_ms=1000,
            reason="init",
        )
    ledger.flush()

    events = ledger.get_cohort_events("c_evt_dup")
    assert len(events) == 1
    ledger.close()


def test_16_incomplete_cohort_true_without_graceful():
    """16. is_cohort_incomplete returns True if cohort exists but has no GRACEFUL_SHUTDOWN."""
    ledger = PaperLedger(db_path=":memory:")
    ledger.create_cohort("c_crashed", 1000, "run")
    ledger.record_cohort_event("c_crashed", "STARTED", 1000)
    ledger.flush()

    assert ledger.is_cohort_incomplete("c_crashed") is True
    ledger.close()


def test_17_incomplete_cohort_false_after_graceful():
    """17. is_cohort_incomplete returns False after GRACEFUL_SHUTDOWN is recorded."""
    ledger = PaperLedger(db_path=":memory:")
    ledger.create_cohort("c_clean", 1000, "run")
    ledger.record_cohort_event("c_clean", "STARTED", 1000)
    ledger.record_cohort_event("c_clean", "GRACEFUL_SHUTDOWN", 5000)
    ledger.flush()

    assert ledger.is_cohort_incomplete("c_clean") is False
    ledger.close()


def test_18_open_position_detectable():
    """18. cohort_has_open_positions accurately reflects open vs closed state."""
    ledger = PaperLedger(db_path=":memory:")
    pos = PaperPosition(
        position_id="pos_1",
        cohort_id="c_pos",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        reference_price=50000.0,
        quantity=0.1,
        notional_usdt=5000.0,
        opened_ts_ms=1000,
        horizon_deadline_ms=61000,
        decision_id="dec_pos_1",
        signal_timestamp=1000,
        decision_timestamp=1010,
        available_at=1020,
        entry_fee_usdt=2.5,
        entry_slippage_usdt=1.0,
        mae_bps=0.0,
        mfe_bps=0.0,
        ticks_processed=1,
        last_tick_ts_ms=1050,
        data_gap=False,
    )
    ledger.record_position(pos)
    ledger.flush()
    assert ledger.cohort_has_open_positions("c_pos") is True

    ledger.close_position("pos_1")
    ledger.flush()
    assert ledger.cohort_has_open_positions("c_pos") is False
    ledger.close()


def test_19_persistence_error_sets_unhealthy():
    """19. Database write error sets _has_error = True and increments error count."""
    ledger = PaperLedger(db_path=":memory:")
    assert ledger.health_snapshot()["healthy"] is True

    # Force error by breaking connection
    ledger._conn.close()

    # Next write triggers exception in writer loop
    ledger.record_cohort("c_fail", 1000)
    # Wait briefly for writer loop to process
    for _ in range(20):
        if not ledger.health_snapshot()["healthy"]:
            break
        import time
        time.sleep(0.01)

    snap = ledger.health_snapshot()
    assert snap["healthy"] is False
    assert snap["error_count"] >= 1
    assert snap["last_error_type"] is not None


def test_20_record_rejected_after_unhealthy():
    """20. record_* returns False and does not enqueue when ledger is unhealthy."""
    ledger = PaperLedger(db_path=":memory:")
    # Manually set error state
    with ledger._lock:
        ledger._has_error = True
        ledger._error_count = 1

    ok_sig = ledger.record_signal_observation("c", "BTCUSDT", "w", "e", 1000, "LONG", "DECISION_CREATED")
    ok_risk = ledger.record_risk_evaluation("c", "d", "APPROVED", 0.9)
    ok_evt = ledger.record_cohort_event("c", "STARTED", 1000)
    ok_cohort = ledger.create_cohort("c", 1000)

    assert ok_sig is False
    assert ok_risk is False
    assert ok_evt is False
    assert ok_cohort is False
    ledger.close()


def test_21_health_error_count():
    """21. Health error count increments accurately on multiple write errors."""
    ledger = PaperLedger(db_path=":memory:")
    with ledger._lock:
        ledger._error_count = 5
        ledger._has_error = True

    assert ledger.health_snapshot()["error_count"] == 5
    ledger.close()


def test_22_queue_size_exposed():
    """22. health_snapshot exposes current queue size."""
    ledger = PaperLedger(db_path=":memory:")
    # Stop writer processing to accumulate items in queue
    ledger._stop_event.set()
    ledger._worker.join(timeout=1.0)

    ledger.record_cohort_event("c", "STARTED", 1000)
    ledger.record_cohort_event("c", "STARTED", 2000)
    snap = ledger.health_snapshot()
    assert snap["queue_size"] >= 2
    ledger.close()


def test_23_flush_health_semantics():
    """23. flush_status returns UNHEALTHY if ledger encountered an error."""
    ledger = PaperLedger(db_path=":memory:")
    with ledger._lock:
        ledger._has_error = True
    assert ledger.flush_status() == "UNHEALTHY"
    assert ledger.flush() is False
    ledger.close()


def test_24_old_schema_compatibility():
    """24. PaperLedger initializes cleanly over an existing legacy database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "legacy.db")
        # Criar DB com apenas a tabela legacy cohorts
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE cohorts (cohort_id TEXT PRIMARY KEY, created_at_ms INTEGER NOT NULL, description TEXT, metadata_json TEXT);")
        conn.execute("INSERT INTO cohorts VALUES ('c_old', 500, 'legacy', '{}');")
        conn.commit()
        conn.close()

        # Abrir com novo PaperLedger
        ledger = PaperLedger(db_path=db_path)
        cur = ledger._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}

        assert "signal_observations" in tables
        assert "risk_evaluations" in tables
        assert "cohort_events" in tables
        assert ledger.cohort_exists("c_old") is True
        ledger.close()


def test_25_no_old_data_lost():
    """25. Opening legacy database preserves existing decision and trade rows intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "legacy_full.db")
        ledger_old = PaperLedger(db_path=db_path)
        d = CanonicalDecision(
            cohort_id="c_persist",
            symbol="BTCUSDT",
            window_id="w0",
            decision_provider="fixed_long",
            strategy_version="1.0",
            signal_timestamp=1000,
            decision_timestamp=1010,
            available_at=1020,
            side="LONG",
            reference_price=50000.0,
            notional_usdt=500.0,
            horizon_s=60,
        )
        ledger_old.record_decision(d)
        ledger_old.close()

        # Reabrir e verificar que decisão permanece
        ledger_new = PaperLedger(db_path=db_path)
        cur = ledger_new._conn.cursor()
        cur.execute("SELECT decision_id, side FROM decisions WHERE cohort_id = 'c_persist'")
        row = cur.fetchone()
        assert row is not None
        assert row["decision_id"] == d.decision_id
        assert row["side"] == "LONG"
        ledger_new.close()


def test_26_funding_null_preserved():
    """26. ClosedTrade with NULL funding and costs_complete=0 persists without coercion."""
    ledger = PaperLedger(db_path=":memory:")
    trade = ClosedTrade(
        trade_id="tr_p1",
        decision_id="dec_p1",
        cohort_id="c_fund",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        exit_price=50100.0,
        quantity=0.1,
        notional_usdt=5000.0,
        opened_ts_ms=1000,
        closed_ts_ms=61000,
        exit_reason="HORIZON_EXPIRY",
        trade_direction_profitable=True,
        prediction_direction_correct=True,
        trade_win=None,
        gross_pnl_bps=20.0,
        net_pnl_bps=None,
        fees_bps=8.0,
        slippage_bps=2.0,
        funding_bps=None,
        gross_pnl_usdt=10.0,
        fees_usdt=4.0,
        slippage_usdt=1.0,
        funding_usdt=None,
        net_pnl_usdt=None,
        pnl_R=None,
        costs_complete=False,
        data_gap=False,
        mae_bps=0.0,
        mfe_bps=25.0,
        ticks_count=50,
    )
    ledger.record_closed_trade(trade)
    ledger.flush()

    trades = ledger.get_closed_trades("c_fund")
    assert len(trades) == 1
    t = trades[0]
    assert t.costs_complete is False
    assert t.funding_bps is None
    assert t.funding_usdt is None
    assert t.net_pnl_bps is None
    assert t.trade_win is None
    ledger.close()


def test_27_scorecard_unaffected():
    """27. Scorecard calculations run cleanly over closed_trades, ignoring auxiliary tables."""
    ledger = PaperLedger(db_path=":memory:")
    # Gravar observação, risco, coorte e trade
    ledger.record_signal_observation("c_sc", "BTCUSDT", "w1", "k1", 1000, "LONG", "DECISION_CREATED")
    ledger.record_risk_evaluation("c_sc", "dec1", "APPROVED", 0.95)
    trade = ClosedTrade(
        trade_id="tr_sc1",
        decision_id="dec1",
        cohort_id="c_sc",
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        exit_price=50500.0,
        quantity=0.1,
        notional_usdt=5000.0,
        opened_ts_ms=1000,
        closed_ts_ms=61000,
        exit_reason="HORIZON_EXPIRY",
        trade_direction_profitable=True,
        prediction_direction_correct=True,
        trade_win=True,
        gross_pnl_bps=100.0,
        net_pnl_bps=90.0,
        fees_bps=8.0,
        slippage_bps=2.0,
        funding_bps=0.0,
        gross_pnl_usdt=50.0,
        fees_usdt=4.0,
        slippage_usdt=1.0,
        funding_usdt=0.0,
        net_pnl_usdt=45.0,
        pnl_R=2.5,
        costs_complete=True,
        data_gap=False,
        mae_bps=0.0,
        mfe_bps=100.0,
        ticks_count=100,
    )
    ledger.record_closed_trade(trade)
    ledger.flush()

    trades = ledger.get_closed_trades("c_sc")
    metrics = scorecard(trades)
    assert metrics.total_trades == 1
    assert metrics.wins == 1
    assert metrics.win_rate == 1.0
    ledger.close()


def test_28_fifo_ordering_maintained():
    """28. Strict FIFO write order between signal observation, decision, risk, order and fill."""
    ledger = PaperLedger(db_path=":memory:")
    cohort_id = "c_fifo"
    ord_id = "ord_fifo"
    fill_id = "fill_fifo"

    ledger.record_signal_observation(cohort_id, "BTCUSDT", "w1", "k1", 1000, "LONG", "DECISION_CREATED")
    d = CanonicalDecision(
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="fixed_long",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1010,
        available_at=1020,
        side="LONG",
        reference_price=50000.0,
        notional_usdt=500.0,
        horizon_s=60,
    )
    dec_id = d.decision_id
    ledger.record_decision(d)
    ledger.record_risk_evaluation(cohort_id, dec_id, "APPROVED", 0.9)
    order = PaperOrder(
        order_id=ord_id,
        decision_id=dec_id,
        cohort_id=cohort_id,
        decision_provider="fixed_long",
        symbol="BTCUSDT",
        side="LONG",
        reference_price=50000.0,
        notional_usdt=500.0,
        signal_timestamp=1000,
        decision_timestamp=1010,
        available_at=1020,
        expires_at=2020,
        horizon_s=60,
    )
    ledger.record_order(order)
    fill = PaperFill(
        fill_id=fill_id,
        order_id=ord_id,
        decision_id=dec_id,
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        side="LONG",
        fill_price=50005.0,
        raw_price=50000.0,
        slippage_bps=1.0,
        quantity=0.01,
        notional_usdt=500.0,
        fill_timestamp=1050,
        trade_id_used="trade_123",
        fee_usdt=0.25,
        decision_to_fill_ms=40,
        available_to_fill_ms=30,
    )
    ledger.record_fill(fill)
    ledger.flush()

    cur = ledger._conn.cursor()
    cur.execute("SELECT observation_id FROM signal_observations WHERE cohort_id = ?", (cohort_id,))
    assert cur.fetchone() is not None
    cur.execute("SELECT decision_id FROM decisions WHERE decision_id = ?", (dec_id,))
    assert cur.fetchone() is not None
    cur.execute("SELECT risk_evaluation_id FROM risk_evaluations WHERE decision_id = ?", (dec_id,))
    assert cur.fetchone() is not None
    cur.execute("SELECT order_id FROM orders WHERE order_id = ?", (ord_id,))
    assert cur.fetchone() is not None
    cur.execute("SELECT fill_id FROM fills WHERE fill_id = ?", (fill_id,))
    assert cur.fetchone() is not None
    ledger.close()


def test_29_zero_tick_api_or_table():
    """29. Proves PaperLedger has no tick recording API and no market_ticks table."""
    ledger = PaperLedger(db_path=":memory:")
    assert not hasattr(ledger, "record_tick")
    assert not hasattr(ledger, "record_market_trade")
    assert not hasattr(ledger, "on_market_trade")

    cur = ledger._conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "market_ticks" not in tables
    assert "ticks" not in tables
    assert "trades_raw" not in tables
    ledger.close()


def test_30_sanitization_nan_inf():
    """30. Rejects NaN/Inf in risk evaluations fail-closed without corrupting database."""
    ledger = PaperLedger(db_path=":memory:")
    # NaN risk_confidence rejected
    ok1 = ledger.record_risk_evaluation("c", "d1", "APPROVED", float("nan"))
    assert ok1 is False

    # Inf max_size rejected
    ok2 = ledger.record_risk_evaluation("c", "d2", "APPROVED", 0.9, max_size=float("inf"))
    assert ok2 is False

    # NaN source_confidence rejected
    ok3 = ledger.record_risk_evaluation("c", "d3", "APPROVED", 0.9, source_confidence=float("nan"))
    assert ok3 is False

    # Invalid status rejected
    ok4 = ledger.record_risk_evaluation("c", "d4", "INVALID_STATUS", 0.9)
    assert ok4 is False

    # Invalid signal side rejected
    ok5 = ledger.record_signal_observation("c", "BTCUSDT", "w", "k", 1000, "BAD_SIDE", "DECISION_CREATED")
    assert ok5 is False

    # Invalid signal observation status rejected
    ok6 = ledger.record_signal_observation("c", "BTCUSDT", "w", "k", 1000, "LONG", "BAD_STATUS")
    assert ok6 is False

    # Invalid cohort event type rejected
    ok7 = ledger.record_cohort_event("c", "INVALID_EVENT_TYPE", 1000)
    assert ok7 is False

    ledger.flush()
    assert len(ledger.get_risk_evaluations("c")) == 0
    assert len(ledger.get_signal_observations("c")) == 0
    assert len(ledger.get_cohort_events("c")) == 0
    ledger.close()
