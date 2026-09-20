# paper_trading/ledger.py
"""
Append-only SQLite Ledger for hermetic paper trading.

Records cohorts, decisions, rejections, orders, fills, positions,
closed trades, and funding events asynchronously via a background queue.
Provides synchronous flush() for deterministic replays and tests.
"""

from __future__ import annotations

import json
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

CREATE INDEX IF NOT EXISTS idx_decisions_cohort ON decisions(cohort_id);
CREATE INDEX IF NOT EXISTS idx_rejections_cohort ON rejections(cohort_id);
CREATE INDEX IF NOT EXISTS idx_closed_trades_cohort ON closed_trades(cohort_id);
CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(cohort_id, is_open);
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
                self._conn.commit()
            except Exception:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
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

    # Public Enqueue APIs (Thread-safe, Non-blocking)

    def record_cohort(self, cohort_id: str, created_at_ms: int, description: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._queue.put(("RECORD_COHORT", {"cohort_id": cohort_id, "created_at_ms": created_at_ms, "description": description, "metadata": metadata or {}}, None))

    def record_decision(self, decision: CanonicalDecision) -> None:
        self._queue.put(("RECORD_DECISION", decision, None))

    def record_rejection(self, rejection: Rejection) -> None:
        self._queue.put(("RECORD_REJECTION", rejection, None))

    def record_order(self, order: PaperOrder) -> None:
        self._queue.put(("RECORD_ORDER", order, None))

    def record_fill(self, fill: PaperFill) -> None:
        self._queue.put(("RECORD_FILL", fill, None))

    def record_position(self, position: PaperPosition) -> None:
        self._queue.put(("RECORD_POSITION", position, None))

    def close_position(self, position_id: str) -> None:
        self._queue.put(("CLOSE_POSITION", position_id, None))

    def record_closed_trade(self, trade: ClosedTrade) -> None:
        self._queue.put(("RECORD_CLOSED_TRADE", trade, None))
        pos_id = trade.trade_id.replace("tr_", "")
        self._queue.put(("CLOSE_POSITION", pos_id, None))

    def record_funding(self, cohort_id: str, symbol: str, timestamp_ms: int, funding_rate: float, notional_usdt: float, amount_usdt: float, side: str) -> None:
        self._queue.put(("RECORD_FUNDING", {"cohort_id": cohort_id, "symbol": symbol, "timestamp_ms": timestamp_ms, "funding_rate": funding_rate, "notional_usdt": notional_usdt, "amount_usdt": amount_usdt, "side": side}, None))

    def record_kill_switch(self, cohort_id: str, triggered_at_ms: int, reason: str, count: int) -> None:
        self._queue.put(("RECORD_KILL_SWITCH", {"cohort_id": cohort_id, "triggered_at_ms": triggered_at_ms, "reason": reason, "positions_closed_count": count}, None))

    def flush(self) -> None:
        """Block until all queued writes have been committed to SQLite."""
        event = threading.Event()
        self._queue.put(("FLUSH", None, event))
        event.wait(timeout=10.0)

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
