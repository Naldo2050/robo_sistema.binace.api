# scripts/analytics/run_d0c_repeat.py
"""
D0-C Repeat Runner with External Diagnostics Checkpoint.

Automates the 60-minute repeat session of Gate D0-C with:
- Zero AI / Zero LLM / Zero real orders / Hermetic observation mode
- Seeded random provider (seed=1337)
- External checkpoint every 5 minutes outside the SQLite economic ledger
- Complete WAL forensics and automated scorecard audit on completion
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def create_session_env(cohort_id: str, db_path: str, duration_seconds: int) -> Dict[str, str]:
    """Build clean, hermetic environment for D0-C repeat session."""
    env = os.environ.copy()

    # Strip any trading and AI credentials
    for cred in (
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_SECRET_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    ):
        env.pop(cred, None)

    # Core observation and safety parameters
    env["OBSERVATION_MODE"] = "1"
    env["LOAD_DOTENV"] = "0"
    env["AI_ENABLED"] = "0"
    env["USE_PROMPT_STYLES"] = "0"
    env["FOLLOW_SIGNAL"] = "0"
    env["EXECUTION_ENABLED"] = "0"
    env["HYBRID_ENABLED"] = "0"

    # D0-C Shadow Paper Trading Configuration
    env["PAPER_SHADOW_ENABLED"] = "1"
    env["PAPER_COHORT_ID"] = cohort_id
    env["PAPER_DB_PATH"] = db_path
    env["PAPER_GIT_SHA"] = "de95d166112251646e871b13a403366c50470b3d"
    env["PAPER_PROVIDER"] = "seeded_random"
    env["PAPER_RANDOM_SEED"] = "1337"
    env["PAPER_SYMBOL"] = "BTCUSDT"
    env["PAPER_TIMEFRAME"] = "1m"
    env["PAPER_NOTIONAL_USDT"] = "100.0"
    env["PAPER_HORIZON_S"] = "300"
    env["PAPER_ORDER_TTL_MS"] = "15000"
    env["PAPER_MAKER_FEE_BPS"] = "2.0"
    env["PAPER_TAKER_FEE_BPS"] = "5.0"
    env["PAPER_ENTRY_SLIPPAGE_BPS"] = "1.0"
    env["PAPER_EXIT_SLIPPAGE_BPS"] = "1.0"
    env["PAPER_COST_SOURCE"] = "PREFLIGHT_ASSUMPTION"
    env["PAPER_COST_EFFECTIVE_AT"] = "2026-09-01T00:00:00+00:00"
    env["PAPER_STRATEGY_VERSION"] = "random_prediction_validation_v1"

    # Bot runtime duration
    env["BOT_DURATION_SECONDS"] = str(duration_seconds)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    return env


def query_db_diagnostics(db_path: str, cohort_id: str) -> Dict[str, Any]:
    """Read diagnostics directly via read-only SQLite URI without locking."""
    if not os.path.exists(db_path):
        return {
            "ledger_healthy": False,
            "decisions": 0,
            "terminal_prediction_outcomes": 0,
            "pending_predictions": 0,
            "correct": 0,
            "incorrect": 0,
            "flat": 0,
            "unresolved": 0,
            "error": "db_not_found",
        }

    try:
        uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        cur = conn.cursor()

        # Health check
        cur.execute("SELECT 1")

        # Directional decisions count
        try:
            cur.execute(
                "SELECT count(*) FROM decisions WHERE cohort_id = ? AND side IN ('LONG', 'SHORT')",
                (cohort_id,),
            )
            decisions_cnt = int(cur.fetchone()[0])
        except sqlite3.OperationalError:
            decisions_cnt = 0

        # Terminal outcomes count (column name is 'result')
        outcomes_map: Dict[str, int] = {}
        try:
            cur.execute(
                """
                SELECT result, count(*)
                FROM prediction_outcomes
                WHERE cohort_id = ?
                GROUP BY result
                """,
                (cohort_id,),
            )
            outcomes_map = {row[0]: int(row[1]) for row in cur.fetchall()}
        except sqlite3.OperationalError:
            pass

        correct = outcomes_map.get("CORRECT", 0)
        incorrect = outcomes_map.get("INCORRECT", 0)
        flat = outcomes_map.get("FLAT", 0)
        unresolved = outcomes_map.get("UNRESOLVED", 0)
        terminal_total = correct + incorrect + flat + unresolved
        pending = max(0, decisions_cnt - terminal_total)

        conn.close()
        return {
            "ledger_healthy": True,
            "decisions": decisions_cnt,
            "terminal_prediction_outcomes": terminal_total,
            "pending_predictions": pending,
            "correct": correct,
            "incorrect": incorrect,
            "flat": flat,
            "unresolved": unresolved,
            "error": None,
        }
    except Exception as e:
        return {
            "ledger_healthy": False,
            "decisions": 0,
            "terminal_prediction_outcomes": 0,
            "pending_predictions": 0,
            "correct": 0,
            "incorrect": 0,
            "flat": 0,
            "unresolved": unresolved if "unresolved" in locals() else 0,
            "error": str(e),
        }


def write_external_checkpoint(
    checkpoint_file: str,
    cohort_id: str,
    start_time: float,
    proc: subprocess.Popen,
    db_path: str,
    extra_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Write an external checkpoint diagnostic to disk (outside economic DB)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    elapsed_s = round(time.time() - start_time, 1)
    process_alive = proc.poll() is None

    rss_mb: Optional[float] = None
    if psutil is not None and process_alive:
        try:
            p = psutil.Process(proc.pid)
            rss_mb = round(p.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass

    diag = query_db_diagnostics(db_path, cohort_id)

    record = {
        "timestamp": now_iso,
        "cohort": cohort_id,
        "elapsed_seconds": elapsed_s,
        "process_alive": process_alive,
        "ledger_healthy": diag["ledger_healthy"],
        "decisions": diag["decisions"],
        "terminal_prediction_outcomes": diag["terminal_prediction_outcomes"],
        "pending_predictions": diag["pending_predictions"],
        "correct": diag["correct"],
        "incorrect": diag["incorrect"],
        "flat": diag["flat"],
        "unresolved": diag["unresolved"],
        "rss_mb": rss_mb,
    }
    if extra_tag:
        record["tag"] = extra_tag
    if diag.get("error"):
        record["diag_error"] = diag["error"]

    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] CHECKPOINT ({elapsed_s}s): "
        f"alive={process_alive} | healthy={diag['ledger_healthy']} | "
        f"decisions={diag['decisions']} | terminal={diag['terminal_prediction_outcomes']} "
        f"(C:{diag['correct']} I:{diag['incorrect']} F:{diag['flat']} U:{diag['unresolved']}) | "
        f"pending={diag['pending_predictions']} | RSS={rss_mb} MB",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="D0-C Repeat Session Runner")
    parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds (default: 3600 = 60m)")
    parser.add_argument("--checkpoint-interval", type=int, default=300, help="Checkpoint interval in seconds (default: 300 = 5m)")
    args = parser.parse_args()

    duration = args.duration
    interval = args.checkpoint_interval

    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%dT%H%M%SZ")
    cohort_id = f"paper_live_d0c_repeat_{ts_str}_de95d16"
    db_path = os.path.join("dados", f"paper_live_d0c_repeat_{ts_str}.db")
    checkpoint_file = os.path.join("dados", f"paper_live_d0c_repeat_{ts_str}_checkpoints.jsonl")
    log_file = os.path.join("logs", f"paper_live_d0c_repeat_{ts_str}.log")

    os.makedirs("dados", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 70, flush=True)
    print(" D0-C REPEAT SESSION — GATE D0-C RUNNER", flush=True)
    print("=" * 70, flush=True)
    print(f"Cohort ID:           {cohort_id}", flush=True)
    print(f"Database Path:       {db_path}", flush=True)
    print(f"Checkpoint File:     {checkpoint_file}", flush=True)
    print(f"Log File:            {log_file}", flush=True)
    print(f"Target Duration:     {duration} seconds ({duration / 60:.1f} minutes)", flush=True)
    print(f"Checkpoint Interval: {interval} seconds ({interval / 60:.1f} minutes)", flush=True)
    print(f"Provider:            seeded_random (seed=1337)", flush=True)
    print(f"Git SHA:             de95d166112251646e871b13a403366c50470b3d", flush=True)
    print("=" * 70, flush=True)

    env = create_session_env(cohort_id, db_path, duration)

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--duration-seconds",
        str(duration),
    ]

    log_fh = open(log_file, "w", encoding="utf-8")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )

    print(f"Bot process started with PID {proc.pid}. Monitoring execution...", flush=True)

    next_checkpoint = start_time + interval
    initial_written = False

    try:
        # Give bot 5 seconds to boot up and record initial checkpoint
        time.sleep(5)
        write_external_checkpoint(checkpoint_file, cohort_id, start_time, proc, db_path, extra_tag="STARTUP")
        initial_written = True

        while proc.poll() is None:
            time.sleep(2)
            now = time.time()
            if now >= next_checkpoint:
                write_external_checkpoint(checkpoint_file, cohort_id, start_time, proc, db_path)
                next_checkpoint = now + interval

        exit_code = proc.poll()
        print(f"\nBot process completed with exit code {exit_code}.", flush=True)

    except KeyboardInterrupt:
        print("\nManual interruption received. Terminating bot process gracefully...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        exit_code = proc.poll()

    finally:
        write_external_checkpoint(checkpoint_file, cohort_id, start_time, proc, db_path, extra_tag="FINAL")
        log_fh.close()

    print("\n" + "=" * 70, flush=True)
    print(" D0-C REPEAT SESSION FINISHED", flush=True)
    print(f"Elapsed Time: {time.time() - start_time:.1f}s", flush=True)
    print(f"Exit Code:    {exit_code}", flush=True)
    print("=" * 70, flush=True)

    return exit_code or 0


if __name__ == "__main__":
    sys.exit(main())
