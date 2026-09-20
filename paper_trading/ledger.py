# paper_trading/ledger.py
"""
Append-only SQLite Ledger for hermetic paper trading.

Records cohorts, decisions, rejections, orders, fills, positions,
closed trades, and funding events asynchronously via a background queue.
Provides synchronous flush() for deterministic replays and tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from common.json_safe import sanitize_json_safe
from paper_trading.contracts import (
    CanonicalDecision,
    ClosedTrade,
    PaperFill,
    PaperOrder,
    PaperPosition,
    Rejection,
)
from paper_trading.prediction import PredictionOutcome

VALID_SIGNAL_SIDES = {"LONG", "SHORT", "NEUTRAL", "UNKNOWN"}
VALID_SIGNAL_OBSERVATION_STATUSES = {
    "DECISION_CREATED",
    "SKIPPED_NON_DIRECTIONAL",
    "INVALID_SIGNAL",
}
VALID_RISK_STATUSES = {"APPROVED", "REJECTED", "ERROR"}
VALID_COHORT_EVENT_TYPES = {
    "STARTED",
    "GRACEFUL_SHUTDOWN",
    "CRASH_DETECTED",
    "PERSISTENCE_FAILURE",
}


def _is_finite_number(val: Any) -> bool:
    if val is None:
        return True
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False
    return not (math.isnan(val) or math.isinf(val))


def compute_signal_observation_id(cohort_id: str, source_window_id: str, source_event_key: str) -> str:
    raw = f"{cohort_id}:{source_window_id}:{source_event_key}"
    return f"obs_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def compute_risk_evaluation_id(cohort_id: str, decision_id: str) -> str:
    raw = f"{cohort_id}:{decision_id}"
    return f"reval_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def compute_cohort_event_id(cohort_id: str, event_type: str, timestamp_ms: int) -> str:
    raw = f"{cohort_id}:{event_type}:{timestamp_ms}"
    return f"cevt_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def json_dumps_safe(value: Any) -> str:
    return json.dumps(sanitize_json_safe(value))


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    description TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    window_id TEXT NOT NULL,
    decision_provider TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model_version TEXT,
    signal_timestamp INTEGER NOT NULL,
    decision_timestamp INTEGER NOT NULL,
    available_at INTEGER NOT NULL,
    side TEXT NOT NULL,
    reference_price REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    horizon_s INTEGER NOT NULL,
    entry_type TEXT NOT NULL,
    confidence REAL,
    stop_loss REAL,
    take_profit REAL,
    funding_rate_at_decision REAL,
    funding_rate_source TEXT,
    context_json TEXT,
    provider_meta_json TEXT,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    rejection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    decision_provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decision_timestamp INTEGER NOT NULL,
    reason TEXT NOT NULL,
    details TEXT,
    rejected_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    decision_provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    reference_price REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    signal_timestamp INTEGER NOT NULL,
    decision_timestamp INTEGER NOT NULL,
    available_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    horizon_s INTEGER NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    funding_rate_at_decision REAL,
    funding_rate_source TEXT,
    context_json TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    fill_price REAL NOT NULL,
    raw_price REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    quantity REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    fill_timestamp INTEGER NOT NULL,
    trade_id_used TEXT,
    fee_usdt REAL NOT NULL,
    decision_to_fill_ms INTEGER NOT NULL,
    available_to_fill_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    decision_provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    reference_price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    opened_ts_ms INTEGER NOT NULL,
    horizon_deadline_ms INTEGER NOT NULL,
    decision_id TEXT NOT NULL,
    signal_timestamp INTEGER NOT NULL,
    decision_timestamp INTEGER NOT NULL,
    available_at INTEGER NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    funding_rate_at_decision REAL,
    funding_rate_source TEXT,
    entry_fee_usdt REAL NOT NULL,
    entry_slippage_usdt REAL NOT NULL,
    mae_bps REAL NOT NULL,
    mfe_bps REAL NOT NULL,
    ticks_processed INTEGER NOT NULL,
    last_tick_ts_ms INTEGER NOT NULL,
    data_gap INTEGER NOT NULL,
    context_json TEXT,
    is_open INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS closed_trades (
    trade_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    decision_provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    opened_ts_ms INTEGER NOT NULL,
    closed_ts_ms INTEGER NOT NULL,
    exit_reason TEXT NOT NULL,
    trade_direction_profitable INTEGER,
    prediction_direction_correct INTEGER,
    trade_win INTEGER,
    gross_pnl_bps REAL,
    net_pnl_bps REAL,
    fees_bps REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    funding_bps REAL,
    gross_pnl_usdt REAL,
    fees_usdt REAL NOT NULL,
    slippage_usdt REAL NOT NULL,
    funding_usdt REAL,
    net_pnl_usdt REAL,
    pnl_R REAL,
    costs_complete INTEGER NOT NULL,
    data_gap INTEGER NOT NULL,
    mae_bps REAL NOT NULL,
    mfe_bps REAL NOT NULL,
    ticks_count INTEGER NOT NULL,
    context_json TEXT
);

CREATE TABLE IF NOT EXISTS funding_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    notional_usdt REAL NOT NULL,
    amount_usdt REAL NOT NULL,
    side TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kill_switch_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort_id TEXT NOT NULL,
    triggered_at_ms INTEGER NOT NULL,
    reason TEXT NOT NULL,
    positions_closed_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_observations (
    observation_id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source_window_id TEXT NOT NULL,
    source_event_key TEXT NOT NULL,
    signal_timestamp INTEGER NOT NULL,
    source_side TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    context_json TEXT,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_evaluations (
    risk_evaluation_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    status TEXT NOT NULL,
    risk_reason TEXT,
    max_size REAL,
    source_confidence REAL,
    risk_confidence REAL NOT NULL,
    context_json TEXT,
    evaluated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cohort_events (
    event_id TEXT PRIMARY KEY,
    cohort_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    reason TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    reference_price REAL NOT NULL,
    decision_timestamp INTEGER NOT NULL,
    horizon_s INTEGER NOT NULL,
    deadline_ms INTEGER NOT NULL,
    result TEXT NOT NULL,
    reason TEXT,
    resolution_price REAL,
    raw_return_bps REAL,
    directional_return_bps REAL,
    resolved_timestamp_ms INTEGER,
    resolution_drift_ms INTEGER,
    observed_at_ms INTEGER,
    flat_tolerance_bps REAL NOT NULL,
    resolution_tolerance_ms INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_cohort ON decisions(cohort_id);
CREATE INDEX IF NOT EXISTS idx_rejections_cohort ON rejections(cohort_id);
CREATE INDEX IF NOT EXISTS idx_closed_trades_cohort ON closed_trades(cohort_id);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(cohort_id, is_open);
CREATE INDEX IF NOT EXISTS idx_sig_obs_cohort ON signal_observations(cohort_id);
CREATE INDEX IF NOT EXISTS idx_risk_eval_cohort ON risk_evaluations(cohort_id);
CREATE INDEX IF NOT EXISTS idx_cohort_events_cohort ON cohort_events(cohort_id);
CREATE INDEX IF NOT EXISTS idx_pred_outcomes_cohort ON prediction_outcomes(cohort_id);
CREATE INDEX IF NOT EXISTS idx_pred_outcomes_decision ON prediction_outcomes(decision_id);
CREATE INDEX IF NOT EXISTS idx_pred_outcomes_result ON prediction_outcomes(result);
"""


class PaperLedger:
    """
    Asynchronous append-only SQLite store with synchronous test flush.
    """

    def __init__(self, db_path: str = "dados/paper_trading.db") -> None:
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:" or "mode=memory" in db_path)

        if not self._is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._has_error: bool = False
        self._last_error_type: Optional[str] = None
        self._error_count: int = 0
        self._lock = threading.Lock()

        uri_flag = True if "?" in db_path or db_path.startswith("file:") else False
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            uri=uri_flag,
        )
        self._conn.row_factory = sqlite3.Row

        if not self._is_memory:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")

        self._init_schema()

        self._worker = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="PaperLedgerWriter",
        )
        self._worker.start()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            action, payload, sync_event = task
            try:
                if action == "FLUSH":
                    self._conn.commit()
                elif action == "RECORD_COHORT":
                    self._write_cohort(payload)
                elif action == "RECORD_DECISION":
                    self._write_decision(payload)
                elif action == "RECORD_REJECTION":
                    self._write_rejection(payload)
                elif action == "RECORD_ORDER":
                    self._write_order(payload)
                elif action == "RECORD_FILL":
                    self._write_fill(payload)
                elif action == "RECORD_POSITION":
                    self._write_position(payload)
                elif action == "CLOSE_POSITION":
                    self._update_close_position(payload)
                elif action == "RECORD_CLOSED_TRADE":
                    self._write_closed_trade(payload)
                elif action == "RECORD_FUNDING":
                    self._write_funding(payload)
                elif action == "RECORD_KILL_SWITCH":
                    self._write_kill_switch(payload)
                elif action == "RECORD_SIGNAL_OBSERVATION":
                    self._write_signal_observation(payload)
                elif action == "RECORD_RISK_EVALUATION":
                    self._write_risk_evaluation(payload)
                elif action == "RECORD_COHORT_EVENT":
                    self._write_cohort_event(payload)
                elif action == "RECORD_PREDICTION_OUTCOME":
                    self._write_prediction_outcome(payload)
                self._conn.commit()
            except Exception as exc:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                with self._lock:
                    self._has_error = True
                    self._error_count += 1
                    self._last_error_type = type(exc).__name__
            finally:
                if sync_event is not None:
                    sync_event.set()
                self._queue.task_done()

    def _write_cohort(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO cohorts (cohort_id, created_at_ms, description, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                payload["cohort_id"],
                payload["created_at_ms"],
                payload.get("description", ""),
                json_dumps_safe(payload.get("metadata", {})),
            ),
        )

    def _write_decision(self, decision: CanonicalDecision) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO decisions (
                    decision_id, cohort_id, symbol, window_id, decision_provider,
                    strategy_version, model_version, signal_timestamp,
                    decision_timestamp, available_at, side, reference_price,
                    notional_usdt, horizon_s, entry_type, confidence, stop_loss,
                    take_profit, funding_rate_at_decision, funding_rate_source,
                    context_json, provider_meta_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.cohort_id,
                    decision.symbol,
                    decision.window_id,
                    decision.decision_provider,
                    decision.strategy_version,
                    decision.model_version,
                    decision.signal_timestamp,
                    decision.decision_timestamp,
                    decision.available_at,
                    decision.side,
                    decision.reference_price,
                    decision.notional_usdt,
                    decision.horizon_s,
                    decision.entry_type,
                    decision.confidence,
                    decision.stop_loss,
                    decision.take_profit,
                    decision.funding_rate_at_decision,
                    decision.funding_rate_source,
                    json_dumps_safe(decision.context),
                    json_dumps_safe(decision.provider_meta),
                    decision.decision_timestamp,
                ),
            )
        except sqlite3.IntegrityError:
            self._write_rejection(
                Rejection(
                    decision_id=decision.decision_id,
                    cohort_id=decision.cohort_id,
                    decision_provider=decision.decision_provider,
                    symbol=decision.symbol,
                    decision_timestamp=decision.decision_timestamp,
                    reason="DUPLICATE_DECISION",
                    details=f"Decision with id '{decision.decision_id}' already registered in ledger.",
                    rejected_at=decision.decision_timestamp,
                )
            )

    def _write_rejection(self, rejection: Rejection) -> None:
        self._conn.execute(
            """
            INSERT INTO rejections (
                decision_id, cohort_id, decision_provider, symbol, decision_timestamp,
                reason, details, rejected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rejection.decision_id,
                rejection.cohort_id,
                rejection.decision_provider,
                rejection.symbol,
                rejection.decision_timestamp,
                rejection.reason,
                rejection.details,
                rejection.rejected_at,
            ),
        )

    def _write_order(self, order: PaperOrder) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO orders (
                order_id, decision_id, cohort_id, decision_provider, symbol,
                side, reference_price, notional_usdt, signal_timestamp,
                decision_timestamp, available_at, expires_at, horizon_s,
                stop_loss, take_profit, funding_rate_at_decision,
                funding_rate_source, context_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.decision_id,
                order.cohort_id,
                order.decision_provider,
                order.symbol,
                order.side,
                order.reference_price,
                order.notional_usdt,
                order.signal_timestamp,
                order.decision_timestamp,
                order.available_at,
                order.expires_at,
                order.horizon_s,
                order.stop_loss,
                order.take_profit,
                order.funding_rate_at_decision,
                order.funding_rate_source,
                json_dumps_safe(order.context),
            ),
        )

    def _write_fill(self, fill: PaperFill) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fills (
                fill_id, order_id, decision_id, cohort_id, symbol,
                side, fill_price, raw_price, slippage_bps, quantity,
                notional_usdt, fill_timestamp, trade_id_used, fee_usdt,
                decision_to_fill_ms, available_to_fill_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.decision_id,
                fill.cohort_id,
                fill.symbol,
                fill.side,
                fill.fill_price,
                fill.raw_price,
                fill.slippage_bps,
                fill.quantity,
                fill.notional_usdt,
                fill.fill_timestamp,
                str(fill.trade_id_used),
                fill.fee_usdt,
                fill.decision_to_fill_ms,
                fill.available_to_fill_ms,
            ),
        )

    def _write_position(self, position: PaperPosition) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO positions (
                position_id, cohort_id, decision_provider, symbol, side,
                entry_price, reference_price, quantity, notional_usdt, opened_ts_ms,
                horizon_deadline_ms, decision_id, signal_timestamp,
                decision_timestamp, available_at, stop_loss, take_profit,
                funding_rate_at_decision, funding_rate_source,
                entry_fee_usdt, entry_slippage_usdt, mae_bps, mfe_bps,
                ticks_processed, last_tick_ts_ms, data_gap, context_json, is_open
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                position.position_id,
                position.cohort_id,
                position.decision_provider,
                position.symbol,
                position.side,
                position.entry_price,
                position.reference_price,
                position.quantity,
                position.notional_usdt,
                position.opened_ts_ms,
                position.horizon_deadline_ms,
                position.decision_id,
                position.signal_timestamp,
                position.decision_timestamp,
                position.available_at,
                position.stop_loss,
                position.take_profit,
                position.funding_rate_at_decision,
                position.funding_rate_source,
                position.entry_fee_usdt,
                position.entry_slippage_usdt,
                position.mae_bps,
                position.mfe_bps,
                position.ticks_processed,
                position.last_tick_ts_ms,
                1 if position.data_gap else 0,
                json_dumps_safe(position.context),
            ),
        )

    def _update_close_position(self, position_id: str) -> None:
        self._conn.execute(
            "UPDATE positions SET is_open = 0 WHERE position_id = ?",
            (position_id,),
        )

    def _write_closed_trade(self, trade: ClosedTrade) -> None:
        dir_prof = 1 if trade.trade_direction_profitable is True else (0 if trade.trade_direction_profitable is False else None)
        pred_corr = 1 if trade.prediction_direction_correct is True else (0 if trade.prediction_direction_correct is False else None)
        win_val = 1 if trade.trade_win is True else (0 if trade.trade_win is False else None)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO closed_trades (
                trade_id, decision_id, cohort_id, decision_provider, symbol, side,
                entry_price, exit_price, quantity, notional_usdt,
                opened_ts_ms, closed_ts_ms, exit_reason, trade_direction_profitable,
                prediction_direction_correct, trade_win, gross_pnl_bps, net_pnl_bps,
                fees_bps, slippage_bps, funding_bps, gross_pnl_usdt, fees_usdt,
                slippage_usdt, funding_usdt, net_pnl_usdt, pnl_R, costs_complete,
                data_gap, mae_bps, mfe_bps, ticks_count, context_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.trade_id,
                trade.decision_id,
                trade.cohort_id,
                trade.decision_provider,
                trade.symbol,
                trade.side,
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.notional_usdt,
                trade.opened_ts_ms,
                trade.closed_ts_ms,
                trade.exit_reason,
                dir_prof,
                pred_corr,
                win_val,
                trade.gross_pnl_bps,
                trade.net_pnl_bps,
                trade.fees_bps,
                trade.slippage_bps,
                trade.funding_bps,
                trade.gross_pnl_usdt,
                trade.fees_usdt,
                trade.slippage_usdt,
                trade.funding_usdt,
                trade.net_pnl_usdt,
                trade.pnl_R,
                1 if trade.costs_complete else 0,
                1 if trade.data_gap else 0,
                trade.mae_bps,
                trade.mfe_bps,
                trade.ticks_count,
                json_dumps_safe(trade.context),
            ),
        )

    def _write_funding(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO funding_events (
                cohort_id, symbol, timestamp_ms, funding_rate, notional_usdt, amount_usdt, side
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["cohort_id"],
                payload["symbol"],
                payload["timestamp_ms"],
                payload["funding_rate"],
                payload["notional_usdt"],
                payload["amount_usdt"],
                payload["side"],
            ),
        )

    def _write_kill_switch(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO kill_switch_events (
                cohort_id, triggered_at_ms, reason, positions_closed_count
            ) VALUES (?, ?, ?, ?)
            """,
            (
                payload["cohort_id"],
                payload["triggered_at_ms"],
                payload["reason"],
                payload["positions_closed_count"],
            ),
        )

    def _write_signal_observation(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO signal_observations (
                observation_id, cohort_id, symbol, source_window_id, source_event_key,
                signal_timestamp, source_side, status, reason, context_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["observation_id"],
                payload["cohort_id"],
                payload["symbol"],
                payload["source_window_id"],
                payload["source_event_key"],
                payload["signal_timestamp"],
                payload["source_side"],
                payload["status"],
                payload.get("reason"),
                json_dumps_safe(payload.get("context", {})),
                payload["created_at_ms"],
            ),
        )

    def _write_risk_evaluation(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO risk_evaluations (
                risk_evaluation_id, decision_id, cohort_id, status, risk_reason,
                max_size, source_confidence, risk_confidence, context_json, evaluated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["risk_evaluation_id"],
                payload["decision_id"],
                payload["cohort_id"],
                payload["status"],
                payload.get("risk_reason"),
                payload.get("max_size"),
                payload.get("source_confidence"),
                payload["risk_confidence"],
                json_dumps_safe(payload.get("context", {})),
                payload["evaluated_at_ms"],
            ),
        )

    def _write_cohort_event(self, payload: Dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cohort_events (
                event_id, cohort_id, event_type, timestamp_ms, reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["event_id"],
                payload["cohort_id"],
                payload["event_type"],
                payload["timestamp_ms"],
                payload.get("reason"),
                json_dumps_safe(payload.get("metadata", {})),
            ),
        )

    def _write_prediction_outcome(self, outcome: PredictionOutcome) -> None:
        """
        Record terminal prediction outcome.
        Fail-closed: one terminal outcome per prediction_id.
        Identical retry is idempotent.
        Conflicting outcome raises ValueError to trigger persistence health alert.
        Never uses INSERT OR REPLACE.
        """
        cur = self._conn.execute(
            "SELECT result, reason, resolution_price, directional_return_bps FROM prediction_outcomes WHERE prediction_id = ?",
            (outcome.prediction_id,),
        )
        row = cur.fetchone()
        if row is not None:
            existing_res = row[0]
            existing_reason = row[1]
            existing_price = row[2]
            existing_ret = row[3]
            price_match = (existing_price == outcome.resolution_price or (existing_price is None and outcome.resolution_price is None))
            ret_match = (existing_ret == outcome.directional_return_bps or (existing_ret is None and outcome.directional_return_bps is None))
            if existing_res == outcome.result and existing_reason == outcome.reason and price_match and ret_match:
                return  # Idempotent retry

            raise ValueError(
                f"Conflicting terminal outcome for prediction_id {outcome.prediction_id}: "
                f"existing=({existing_res}, {existing_reason}) vs new=({outcome.result}, {outcome.reason})"
            )

        self._conn.execute(
            """
            INSERT INTO prediction_outcomes (
                prediction_id, decision_id, cohort_id, symbol, side,
                reference_price, decision_timestamp, horizon_s, deadline_ms,
                result, reason, resolution_price, raw_return_bps,
                directional_return_bps, resolved_timestamp_ms, resolution_drift_ms,
                observed_at_ms, flat_tolerance_bps, resolution_tolerance_ms,
                policy_version, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.prediction_id,
                outcome.decision_id,
                outcome.cohort_id,
                outcome.symbol,
                outcome.side,
                outcome.reference_price,
                outcome.decision_timestamp,
                outcome.horizon_s,
                outcome.deadline_ms,
                outcome.result,
                outcome.reason,
                outcome.resolution_price,
                outcome.raw_return_bps,
                outcome.directional_return_bps,
                outcome.resolved_timestamp_ms,
                outcome.resolution_drift_ms,
                outcome.observed_at_ms,
                outcome.flat_tolerance_bps,
                outcome.resolution_tolerance_ms,
                outcome.policy_version,
                outcome.created_at_ms,
            ),
        )

    # Public Enqueue APIs (Thread-safe, Non-blocking, Fail-closed)

    def create_cohort(
        self,
        cohort_id: str,
        created_at_ms: int,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Strict fail-closed cohort creation.
        Returns True if successfully created.
        Returns False if cohort_id already exists or ledger is unhealthy.
        Never overwrites existing metadata.
        """
        with self._lock:
            if self._has_error:
                return False

        self.flush()
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO cohorts (cohort_id, created_at_ms, description, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        cohort_id,
                        created_at_ms,
                        description,
                        json_dumps_safe(metadata or {}),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as exc:
            with self._lock:
                self._has_error = True
                self._error_count += 1
                self._last_error_type = type(exc).__name__
            return False

    def record_cohort(self, cohort_id: str, created_at_ms: int, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_COHORT", {"cohort_id": cohort_id, "created_at_ms": created_at_ms, "description": description, "metadata": metadata or {}}, None))
        return True

    def record_decision(self, decision: CanonicalDecision) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_DECISION", decision, None))
        return True

    def record_rejection(self, rejection: Rejection) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_REJECTION", rejection, None))
        return True

    def record_order(self, order: PaperOrder) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_ORDER", order, None))
        return True

    def record_fill(self, fill: PaperFill) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_FILL", fill, None))
        return True

    def record_position(self, position: PaperPosition) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_POSITION", position, None))
        return True

    def close_position(self, position_id: str) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("CLOSE_POSITION", position_id, None))
        return True

    def record_closed_trade(self, trade: ClosedTrade) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_CLOSED_TRADE", trade, None))
        pos_id = trade.trade_id.replace("tr_", "")
        self._queue.put(("CLOSE_POSITION", pos_id, None))
        return True

    def record_funding(self, cohort_id: str, symbol: str, timestamp_ms: int, funding_rate: float, notional_usdt: float, amount_usdt: float, side: str) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_FUNDING", {"cohort_id": cohort_id, "symbol": symbol, "timestamp_ms": timestamp_ms, "funding_rate": funding_rate, "notional_usdt": notional_usdt, "amount_usdt": amount_usdt, "side": side}, None))
        return True

    def record_kill_switch(self, cohort_id: str, triggered_at_ms: int, reason: str, count: int) -> bool:
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_KILL_SWITCH", {"cohort_id": cohort_id, "triggered_at_ms": triggered_at_ms, "reason": reason, "positions_closed_count": count}, None))
        return True

    def record_signal_observation(
        self,
        cohort_id: str,
        symbol: str,
        source_window_id: str,
        source_event_key: str,
        signal_timestamp: int,
        source_side: str,
        status: str,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        created_at_ms: Optional[int] = None,
    ) -> bool:
        with self._lock:
            if self._has_error:
                return False

        if source_side not in VALID_SIGNAL_SIDES:
            return False
        if status not in VALID_SIGNAL_OBSERVATION_STATUSES:
            return False

        now_ms = created_at_ms if created_at_ms is not None else signal_timestamp
        obs_id = compute_signal_observation_id(cohort_id, source_window_id, source_event_key)
        payload = {
            "observation_id": obs_id,
            "cohort_id": cohort_id,
            "symbol": symbol,
            "source_window_id": source_window_id,
            "source_event_key": source_event_key,
            "signal_timestamp": signal_timestamp,
            "source_side": source_side,
            "status": status,
            "reason": reason,
            "context": context or {},
            "created_at_ms": now_ms,
        }
        self._queue.put(("RECORD_SIGNAL_OBSERVATION", payload, None))
        return True

    def record_risk_evaluation(
        self,
        cohort_id: str,
        decision_id: str,
        status: str,
        risk_confidence: float,
        risk_reason: Optional[str] = None,
        max_size: Optional[float] = None,
        source_confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        evaluated_at_ms: Optional[int] = None,
    ) -> bool:
        with self._lock:
            if self._has_error:
                return False

        if status not in VALID_RISK_STATUSES:
            return False
        if not _is_finite_number(risk_confidence):
            return False
        if max_size is not None and not _is_finite_number(max_size):
            return False
        if source_confidence is not None and not _is_finite_number(source_confidence):
            return False

        now_ms = evaluated_at_ms if evaluated_at_ms is not None else 0
        eval_id = compute_risk_evaluation_id(cohort_id, decision_id)
        payload = {
            "risk_evaluation_id": eval_id,
            "decision_id": decision_id,
            "cohort_id": cohort_id,
            "status": status,
            "risk_reason": risk_reason,
            "max_size": max_size,
            "source_confidence": source_confidence,
            "risk_confidence": float(risk_confidence),
            "context": context or {},
            "evaluated_at_ms": now_ms,
        }
        self._queue.put(("RECORD_RISK_EVALUATION", payload, None))
        return True

    def record_cohort_event(
        self,
        cohort_id: str,
        event_type: str,
        timestamp_ms: int,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            if self._has_error:
                return False

        if event_type not in VALID_COHORT_EVENT_TYPES:
            return False

        evt_id = compute_cohort_event_id(cohort_id, event_type, timestamp_ms)
        payload = {
            "event_id": evt_id,
            "cohort_id": cohort_id,
            "event_type": event_type,
            "timestamp_ms": timestamp_ms,
            "reason": reason,
            "metadata": metadata or {},
        }
        self._queue.put(("RECORD_COHORT_EVENT", payload, None))
        return True

    def record_prediction_outcome(self, outcome: PredictionOutcome) -> bool:
        """Enqueue terminal prediction outcome for asynchronous persistence."""
        with self._lock:
            if self._has_error:
                return False
        self._queue.put(("RECORD_PREDICTION_OUTCOME", outcome, None))
        return True

    def flush_status(self, timeout: float = 10.0) -> str:
        """Block until all queued writes have been committed or timeout/unhealthy."""
        with self._lock:
            if self._has_error:
                return "UNHEALTHY"

        event = threading.Event()
        self._queue.put(("FLUSH", None, event))
        signaled = event.wait(timeout=timeout)

        with self._lock:
            if self._has_error:
                return "UNHEALTHY"
        if not signaled:
            return "TIMEOUT"
        return "SUCCESS"

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until all queued writes have been committed to SQLite. Returns True on SUCCESS, False otherwise."""
        return self.flush_status(timeout=timeout) == "SUCCESS"

    def health_snapshot(self) -> Dict[str, Any]:
        """Thread-safe snapshot of ledger persistence health."""
        with self._lock:
            return {
                "healthy": not self._has_error,
                "error_count": self._error_count,
                "last_error_type": self._last_error_type,
                "queue_size": self._queue.qsize(),
                "worker_alive": self._worker.is_alive(),
            }

    def close(self) -> None:
        """Flush remaining tasks and close background worker."""
        self.flush()
        self._stop_event.set()
        self._worker.join(timeout=2.0)
        try:
            self._conn.close()
        except Exception:
            pass

    def load_open_positions(self, cohort_id: str) -> List[PaperPosition]:
        """Synchronously query all open positions for a given cohort."""
        self.flush()
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT position_id, cohort_id, decision_provider, symbol, side,
                   entry_price, reference_price, quantity, notional_usdt, opened_ts_ms,
                   horizon_deadline_ms, decision_id, signal_timestamp,
                   decision_timestamp, available_at, stop_loss, take_profit,
                   funding_rate_at_decision, funding_rate_source,
                   entry_fee_usdt, entry_slippage_usdt, mae_bps, mfe_bps,
                   ticks_processed, last_tick_ts_ms, data_gap, context_json
            FROM positions
            WHERE cohort_id = ? AND is_open = 1
            """,
            (cohort_id,),
        )
        rows = cur.fetchall()
        positions = []
        for r in rows:
            ctx = json.loads(r["context_json"]) if r["context_json"] else {}
            pos = PaperPosition(
                position_id=r["position_id"],
                cohort_id=r["cohort_id"],
                decision_provider=r["decision_provider"],
                symbol=r["symbol"],
                side=r["side"],
                entry_price=r["entry_price"],
                reference_price=r["reference_price"],
                quantity=r["quantity"],
                notional_usdt=r["notional_usdt"],
                opened_ts_ms=r["opened_ts_ms"],
                horizon_deadline_ms=r["horizon_deadline_ms"],
                decision_id=r["decision_id"],
                signal_timestamp=r["signal_timestamp"],
                decision_timestamp=r["decision_timestamp"],
                available_at=r["available_at"],
                stop_loss=r["stop_loss"],
                take_profit=r["take_profit"],
                funding_rate_at_decision=r["funding_rate_at_decision"],
                funding_rate_source=r["funding_rate_source"],
                entry_fee_usdt=r["entry_fee_usdt"],
                entry_slippage_usdt=r["entry_slippage_usdt"],
                mae_bps=r["mae_bps"],
                mfe_bps=r["mfe_bps"],
                ticks_processed=r["ticks_processed"],
                last_tick_ts_ms=r["last_tick_ts_ms"],
                data_gap=bool(r["data_gap"]),
                context=ctx,
            )
            positions.append(pos)
        return positions

    def get_closed_trades(self, cohort_id: Optional[str] = None) -> List[ClosedTrade]:
        """Synchronously fetch closed trades from the ledger."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM closed_trades WHERE cohort_id = ?", (cohort_id,))
        else:
            cur.execute("SELECT * FROM closed_trades")
        rows = cur.fetchall()

        trades = []
        for r in rows:
            ctx = json.loads(r["context_json"]) if r["context_json"] else {}
            tw = None
            if r["trade_win"] is not None:
                tw = bool(r["trade_win"])
            dir_prof = None
            if r["trade_direction_profitable"] is not None:
                dir_prof = bool(r["trade_direction_profitable"])
            pred_corr = None
            if r["prediction_direction_correct"] is not None:
                pred_corr = bool(r["prediction_direction_correct"])

            tr = ClosedTrade(
                trade_id=r["trade_id"],
                decision_id=r["decision_id"],
                cohort_id=r["cohort_id"],
                decision_provider=r["decision_provider"],
                symbol=r["symbol"],
                side=r["side"],
                entry_price=r["entry_price"],
                exit_price=r["exit_price"],
                quantity=r["quantity"],
                notional_usdt=r["notional_usdt"],
                opened_ts_ms=r["opened_ts_ms"],
                closed_ts_ms=r["closed_ts_ms"],
                exit_reason=r["exit_reason"],
                trade_direction_profitable=dir_prof,
                prediction_direction_correct=pred_corr,
                trade_win=tw,
                gross_pnl_bps=r["gross_pnl_bps"],
                net_pnl_bps=r["net_pnl_bps"],
                fees_bps=r["fees_bps"],
                slippage_bps=r["slippage_bps"],
                funding_bps=r["funding_bps"],
                gross_pnl_usdt=r["gross_pnl_usdt"],
                fees_usdt=r["fees_usdt"],
                slippage_usdt=r["slippage_usdt"],
                funding_usdt=r["funding_usdt"],
                net_pnl_usdt=r["net_pnl_usdt"],
                pnl_R=r["pnl_R"],
                costs_complete=bool(r["costs_complete"]),
                data_gap=bool(r["data_gap"]),
                mae_bps=r["mae_bps"],
                mfe_bps=r["mfe_bps"],
                ticks_count=r["ticks_count"],
                direction_correct=dir_prof,
                context=ctx,
            )
            trades.append(tr)
        return trades

    def get_decisions(self, cohort_id: Optional[str] = None) -> List[CanonicalDecision]:
        """Synchronously fetch decisions from the ledger."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM decisions WHERE cohort_id = ? ORDER BY decision_timestamp ASC", (cohort_id,))
        else:
            cur.execute("SELECT * FROM decisions ORDER BY decision_timestamp ASC")
        rows = cur.fetchall()
        decisions = []
        for r in rows:
            ctx = json.loads(r["context_json"]) if r["context_json"] else {}
            pm = json.loads(r["provider_meta_json"]) if r["provider_meta_json"] else {}
            dec = CanonicalDecision(
                cohort_id=r["cohort_id"],
                symbol=r["symbol"],
                window_id=r["window_id"],
                decision_provider=r["decision_provider"],
                strategy_version=r["strategy_version"],
                model_version=r["model_version"],
                signal_timestamp=r["signal_timestamp"],
                decision_timestamp=r["decision_timestamp"],
                available_at=r["available_at"],
                side=r["side"],
                reference_price=r["reference_price"],
                notional_usdt=r["notional_usdt"],
                horizon_s=r["horizon_s"],
                entry_type=r["entry_type"] or "MARKET",
                confidence=r["confidence"],
                stop_loss=r["stop_loss"],
                take_profit=r["take_profit"],
                funding_rate_at_decision=r["funding_rate_at_decision"],
                funding_rate_source=r["funding_rate_source"],
                context=ctx,
                provider_meta=pm,
            )
            decisions.append(dec)
        return decisions

    def get_rejections(self, cohort_id: Optional[str] = None) -> List[Rejection]:
        """Synchronously fetch rejections from the ledger."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM rejections WHERE cohort_id = ?", (cohort_id,))
        else:
            cur.execute("SELECT * FROM rejections")
        rows = cur.fetchall()
        return [
            Rejection(
                decision_id=r["decision_id"],
                cohort_id=r["cohort_id"],
                decision_provider=r["decision_provider"],
                symbol=r["symbol"],
                decision_timestamp=r["decision_timestamp"],
                reason=r["reason"],
                details=r["details"] or "",
                rejected_at=r["rejected_at"],
            )
            for r in rows
        ]

    def get_fills(self, cohort_id: Optional[str] = None) -> List[PaperFill]:
        """Synchronously fetch fills from the ledger."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM fills WHERE cohort_id = ?", (cohort_id,))
        else:
            cur.execute("SELECT * FROM fills")
        rows = cur.fetchall()
        return [
            PaperFill(
                fill_id=r["fill_id"],
                order_id=r["order_id"],
                decision_id=r["decision_id"],
                cohort_id=r["cohort_id"],
                symbol=r["symbol"],
                side=r["side"],
                fill_price=r["fill_price"],
                raw_price=r["raw_price"],
                slippage_bps=r["slippage_bps"],
                quantity=r["quantity"],
                notional_usdt=r["notional_usdt"],
                fill_timestamp=r["fill_timestamp"],
                trade_id_used=r["trade_id_used"],
                fee_usdt=r["fee_usdt"],
                decision_to_fill_ms=r["decision_to_fill_ms"],
                available_to_fill_ms=r["available_to_fill_ms"],
            )
            for r in rows
        ]

    def get_signal_observations(self, cohort_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Synchronously query signal observations."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM signal_observations WHERE cohort_id = ? ORDER BY signal_timestamp ASC", (cohort_id,))
        else:
            cur.execute("SELECT * FROM signal_observations ORDER BY signal_timestamp ASC")
        rows = cur.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d["context_json"]) if d.get("context_json") else {}
            results.append(d)
        return results

    def get_risk_evaluations(self, cohort_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Synchronously query risk evaluations."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM risk_evaluations WHERE cohort_id = ? ORDER BY evaluated_at_ms ASC", (cohort_id,))
        else:
            cur.execute("SELECT * FROM risk_evaluations ORDER BY evaluated_at_ms ASC")
        rows = cur.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["context"] = json.loads(d["context_json"]) if d.get("context_json") else {}
            results.append(d)
        return results

    def get_cohort_events(self, cohort_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Synchronously query cohort events."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute("SELECT * FROM cohort_events WHERE cohort_id = ? ORDER BY timestamp_ms ASC", (cohort_id,))
        else:
            cur.execute("SELECT * FROM cohort_events ORDER BY timestamp_ms ASC")
        rows = cur.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata_json"]) if d.get("metadata_json") else {}
            results.append(d)
        return results

    def cohort_exists(self, cohort_id: str) -> bool:
        """Check if cohort_id exists in cohorts table."""
        self.flush()
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM cohorts WHERE cohort_id = ? LIMIT 1", (cohort_id,))
        return cur.fetchone() is not None

    def cohort_has_open_positions(self, cohort_id: str) -> bool:
        """Check if cohort has active open positions."""
        self.flush()
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM positions WHERE cohort_id = ? AND is_open = 1 LIMIT 1", (cohort_id,))
        return cur.fetchone() is not None

    def is_cohort_incomplete(self, cohort_id: str) -> bool:
        """
        Check if cohort exists and has not performed a GRACEFUL_SHUTDOWN.
        Returns False if cohort does not exist.
        """
        if not self.cohort_exists(cohort_id):
            return False
        events = self.get_cohort_events(cohort_id)
        has_graceful = any(e.get("event_type") == "GRACEFUL_SHUTDOWN" for e in events)
        return not has_graceful

    def get_prediction_outcomes(self, cohort_id: Optional[str] = None) -> List[PredictionOutcome]:
        """Synchronously query all persisted prediction outcomes."""
        self.flush()
        cur = self._conn.cursor()
        if cohort_id:
            cur.execute(
                "SELECT * FROM prediction_outcomes WHERE cohort_id = ? ORDER BY decision_timestamp ASC",
                (cohort_id,),
            )
        else:
            cur.execute("SELECT * FROM prediction_outcomes ORDER BY decision_timestamp ASC")
        rows = cur.fetchall()
        outcomes: List[PredictionOutcome] = []
        for r in rows:
            outcomes.append(
                PredictionOutcome(
                    prediction_id=r["prediction_id"],
                    decision_id=r["decision_id"],
                    cohort_id=r["cohort_id"],
                    symbol=r["symbol"],
                    side=r["side"],
                    reference_price=float(r["reference_price"]),
                    decision_timestamp=int(r["decision_timestamp"]),
                    horizon_s=int(r["horizon_s"]),
                    deadline_ms=int(r["deadline_ms"]),
                    result=r["result"],
                    reason=r["reason"],
                    resolution_price=float(r["resolution_price"]) if r["resolution_price"] is not None else None,
                    raw_return_bps=float(r["raw_return_bps"]) if r["raw_return_bps"] is not None else None,
                    directional_return_bps=float(r["directional_return_bps"]) if r["directional_return_bps"] is not None else None,
                    resolved_timestamp_ms=int(r["resolved_timestamp_ms"]) if r["resolved_timestamp_ms"] is not None else None,
                    resolution_drift_ms=int(r["resolution_drift_ms"]) if r["resolution_drift_ms"] is not None else None,
                    observed_at_ms=int(r["observed_at_ms"]) if r["observed_at_ms"] is not None else None,
                    flat_tolerance_bps=float(r["flat_tolerance_bps"]),
                    resolution_tolerance_ms=int(r["resolution_tolerance_ms"]),
                    policy_version=r["policy_version"],
                    created_at_ms=int(r["created_at_ms"]),
                )
            )
        return outcomes
