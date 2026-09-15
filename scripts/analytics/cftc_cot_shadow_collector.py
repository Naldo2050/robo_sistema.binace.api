# scripts/analytics/cftc_cot_shadow_collector.py
# -*- coding: utf-8 -*-
"""
Coletor shadow CFTC COT (P4). NÃO emite sinal, NÃO toca payload/execução.

Coleta o último relatório TFF futures-only de cada contrato CME mapeado,
versiona no raw cache do fetcher e persiste observação em SQLite
(`cftc_cot_shadow_dataset`) com provenance completa para P4/P5/P7.

Uso:
    python scripts/analytics/cftc_cot_shadow_collector.py [--db PATH] [--symbols BTCUSDT,ETHUSDT]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from fetchers.cftc_cot_fetcher import CftcCotFetcher, SYMBOL_TO_CONTRACT
from institutional.cftc_cot import CftcCot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcCotShadowCollector")

DEFAULT_DB_PATH = "dados/trading_bot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cftc_cot_shadow_dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collected_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    contract_code TEXT,
    report_family TEXT NOT NULL,
    report_scope TEXT NOT NULL,
    report_as_of_date TEXT,
    source_row_id TEXT,
    content_hash TEXT,
    revision INTEGER NOT NULL,
    first_seen_at TEXT,
    retrieved_at TEXT,
    status TEXT NOT NULL,
    is_available INTEGER NOT NULL,
    is_stale INTEGER NOT NULL,
    open_interest INTEGER,
    oi_change_wow INTEGER,
    error_code TEXT,
    missing_fields_json TEXT,
    snapshot_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def shadow_health(db_path: str = DEFAULT_DB_PATH,
                  cache_metrics: dict | None = None) -> dict:
    """Métricas operacionais mínimas do shadow contínuo (sem Prometheus).

    cache_metrics: {"cache_write_errors": int} do fetcher quando disponível;
    caso contrário null (erros de escrita vivem em memória + logs).
    """
    health = {
        "last_successful_fetch": None,
        "last_first_seen": None,
        "current_report_as_of": None,
        "current_revision": None,
        "fetch_errors": 0,
        "schema_errors": 0,
        "cache_write_errors": None,
        "shadow_observation_count": 0,
    }
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM cftc_cot_shadow_dataset")
            health["shadow_observation_count"] = int(cur.fetchone()[0])
            cur.execute(
                "SELECT MAX(collected_at) FROM cftc_cot_shadow_dataset "
                "WHERE status IN ('AVAILABLE','PARTIAL','STALE')")
            health["last_successful_fetch"] = cur.fetchone()[0]
            cur.execute(
                "SELECT MAX(first_seen_at) FROM cftc_cot_shadow_dataset "
                "WHERE first_seen_at IS NOT NULL")
            health["last_first_seen"] = cur.fetchone()[0]
            cur.execute(
                "SELECT report_as_of_date, revision FROM cftc_cot_shadow_dataset "
                "WHERE status IN ('AVAILABLE','PARTIAL','STALE') "
                "ORDER BY report_as_of_date DESC, revision DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                health["current_report_as_of"] = row[0]
                health["current_revision"] = row[1]
            cur.execute(
                "SELECT COUNT(*) FROM cftc_cot_shadow_dataset "
                "WHERE status IN ('UNAVAILABLE','ERROR')")
            health["fetch_errors"] = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM cftc_cot_shadow_dataset "
                "WHERE status = 'INVALID'")
            health["schema_errors"] = int(cur.fetchone()[0])
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 - saúde nunca quebra o bot
        health["error"] = str(e)[:200]
    if cache_metrics is not None:
        health["cache_write_errors"] = cache_metrics.get("cache_write_errors")
    return health


async def collect(symbols, db_path: str) -> dict:
    fetcher = CftcCotFetcher()
    cot = CftcCot()
    conn = init_db(db_path)
    summary = {"collected_at": datetime.now(timezone.utc).isoformat(),
               "symbols": {}, "errors": []}
    for symbol in symbols:
        entry = {"status": "PENDING"}
        try:
            code, _ = SYMBOL_TO_CONTRACT[symbol]
        except KeyError:
            snap = cot.unsupported(symbol)
            entry = {"status": "UNSUPPORTED", "error_code": "unsupported_symbol"}
            conn.execute(
                """INSERT INTO cftc_cot_shadow_dataset
                (collected_at, symbol, contract_code, report_family, report_scope,
                 status, is_available, is_stale, revision, error_code,
                 missing_fields_json, snapshot_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (summary["collected_at"], symbol, None, "TFF", "futures_only",
                 "UNSUPPORTED", 0, 0, 0, "unsupported_symbol", "[]",
                 json.dumps(snap.to_dict(), ensure_ascii=False)),
            )
            summary["symbols"][symbol] = entry
            continue
        t0 = time.monotonic()
        try:
            row, err = await fetcher.fetch_latest(code)
            latency_s = round(time.monotonic() - t0, 2)
            now_iso = datetime.now(timezone.utc).isoformat()
            if row is None:
                entry = {"status": "UNAVAILABLE", "error_code": err,
                         "fetch_latency_s": latency_s}
                conn.execute(
                    """INSERT INTO cftc_cot_shadow_dataset
                    (collected_at, symbol, contract_code, report_family, report_scope,
                     status, is_available, is_stale, revision, error_code,
                     missing_fields_json, snapshot_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (summary["collected_at"], symbol, code, "TFF", "futures_only",
                     "UNAVAILABLE", 0, 0, 0, err, "[]",
                     json.dumps({"fetch_latency_s": latency_s}, ensure_ascii=False)),
                )
                summary["symbols"][symbol] = entry
                continue
            record, ing_err = fetcher.ingest_row(code, row, now_iso)
            if ing_err or record is None:
                entry = {"status": "INVALID", "error_code": ing_err,
                         "fetch_latency_s": latency_s}
                summary["symbols"][symbol] = entry
                continue
            prev = fetcher.history(code)
            prev_row = prev[-2].raw if len(prev) >= 2 else None
            snap = cot.analyze(record.raw, symbol=symbol, contract_code=code,
                               first_seen_at=record.first_seen_at,
                               retrieved_at=record.retrieved_at, prev_row=prev_row)
            snap.provenance["content_hash"] = record.content_hash
            snap.provenance["revision"] = record.revision
            snap.quality["revision"] = record.revision
            snap.quality["is_revision"] = record.revision > 0
            d = snap.to_dict()
            conn.execute(
                """INSERT INTO cftc_cot_shadow_dataset
                (collected_at, symbol, contract_code, report_family, report_scope,
                 report_as_of_date, source_row_id, content_hash, revision,
                 first_seen_at, retrieved_at, status, is_available, is_stale,
                 open_interest, oi_change_wow, error_code, missing_fields_json,
                 snapshot_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (summary["collected_at"], symbol, code, "TFF", "futures_only",
                 d.get("report_as_of_date"), record.source_row_id,
                 record.content_hash, record.revision, record.first_seen_at,
                 record.retrieved_at, d["status"], int(d["is_available"]),
                 int(d["is_stale"]),
                 (d.get("open_interest") or {}).get("total"),
                 (d.get("open_interest") or {}).get("change_wow"),
                 d.get("error_code"), json.dumps(d.get("quality", {}).get("missing_fields", [])),
                 json.dumps(d, ensure_ascii=False)),
            )
            entry = {"status": d["status"], "asof": d.get("report_as_of_date"),
                     "oi": (d.get("open_interest") or {}).get("total"),
                     "revision": record.revision,
                     "fetch_latency_s": latency_s, "error_code": d.get("error_code")}
        except Exception as e:  # noqa: BLE001 - shadow nunca quebra o bot
            entry = {"status": "ERROR", "error_code": str(e)[:200]}
            summary["errors"].append(f"{symbol}: {e}")
        summary["symbols"][symbol] = entry
    conn.commit()
    conn.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB_PATH)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summary = asyncio.run(collect(symbols, args.db))
    summary["health"] = shadow_health(args.db)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
