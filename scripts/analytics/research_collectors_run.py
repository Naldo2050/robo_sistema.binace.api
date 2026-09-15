# scripts/analytics/research_collectors_run.py
# -*- coding: utf-8 -*-
"""
B3 — Runner idempotente dos coletores de pesquisa (fora do hot path do bot).

Garante por execução:
  - single-flight por job (lock file + heartbeat + stale release seguro);
  - exit code 0=sucesso, 1=lock ocupado, 2=timeout, 3=job falhou;
  - timeout total por job; sem prompt; sem terminal (stdin DEVNULL);
  - logs por dia em logs/research_collectors/;
  - estado em dados/research/.state/{job}.json (last_success/last_attempt/
    consecutive_failures) — base do alerta de falha silenciosa.

Jobs (frequência definida no scheduler, não aqui):
  positioning   binance_positioning_b2_collector.py --symbol BTCUSDT
  cftc-collect  cftc_r4_prospective_collector.py
  cftc-outcomes cftc_r4_outcome_updater.py
  status        binance_positioning_b2_status.py + cftc_r4_status.py

Uso:
  python scripts/analytics/research_collectors_run.py --job positioning [--timeout 600]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = REPO_ROOT / "dados" / "research" / ".locks"
STATE_DIR = REPO_ROOT / "dados" / "research" / ".state"
LOG_DIR = REPO_ROOT / "logs" / "research_collectors"

JOBS = {
    "positioning": {"cmd": [sys.executable, "scripts/analytics/binance_positioning_b2_collector.py",
                            "--symbol", "BTCUSDT"], "timeout": 900},
    "cftc-collect": {"cmd": [sys.executable, "scripts/analytics/cftc_r4_prospective_collector.py"],
                     "timeout": 600},
    "cftc-outcomes": {"cmd": [sys.executable, "scripts/analytics/cftc_r4_outcome_updater.py"],
                      "timeout": 600},
    "status": {"cmd": None, "timeout": 300},  # agregado especial (abaixo)
}
STALE_AFTER_MULT = 2.0  # lock stale após 2x timeout sem heartbeat

EXIT_OK, EXIT_LOCKED, EXIT_TIMEOUT, EXIT_FAILED = 0, 1, 2, 3


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError, OverflowError):
        return False
    return True


def acquire_lock(job: str, timeout_s: float) -> dict | None:
    """Retorna dict do lock adquirido ou None (ocupado). Stale liberado com segurança."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / f"{job}.lock"
    now = time.time()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        hb = float(data.get("heartbeat", 0) or 0)
        pid = int(data.get("pid", 0) or 0)
        stale = (now - hb) > timeout_s * STALE_AFTER_MULT
        if not stale or (pid and _pid_alive(pid)):
            return None  # ocupado (ou stale aparente mas dono vivo)
        # stale + dono morto: remove e prossegue (registrado acima)
        logging.warning("RUNNER_STALE_LOCK job=%s pid=%s age=%.0fs (liberado)",
                        job, pid, now - hb)
        try:
            path.unlink()
        except OSError:
            return None
    mine = {"pid": os.getpid(), "job": job, "started_at": _utcnow_iso(),
            "heartbeat": now}
    try:
        # escrita exclusiva: falha se outro criou entre o check e agora
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(mine))
    except FileExistsError:
        return None
    # revalida conteúdo (corrida residual): se não é o meu, recua
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    if cur.get("pid") != mine["pid"]:
        return None
    return mine


def release_lock(job: str) -> None:
    try:
        data = json.loads((LOCK_DIR / f"{job}.lock").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("pid") == os.getpid():
        try:
            (LOCK_DIR / f"{job}.lock").unlink()
        except OSError:
            pass


def heartbeat(job: str) -> None:
    path = LOCK_DIR / f"{job}.lock"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if data.get("pid") == os.getpid():
        data["heartbeat"] = time.time()
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass


def load_state(job: str) -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = STATE_DIR / f"{job}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"job": job, "last_success": None, "last_attempt": None,
                "consecutive_failures": 0}


def save_state(job: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{job}.json").write_text(json.dumps(state, ensure_ascii=False,
                                                      indent=2), encoding="utf-8")


def run_job(job: str, timeout_s: float, log_fh) -> int:
    spec = JOBS[job]
    proc = subprocess.Popen(spec["cmd"], cwd=str(REPO_ROOT),
                            stdout=log_fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True)
    start = time.time()
    while True:
        try:
            rc = proc.wait(timeout=5)
            return EXIT_OK if rc == 0 else EXIT_FAILED
        except subprocess.TimeoutExpired:
            heartbeat(job)
            if time.time() - start > timeout_s:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                return EXIT_TIMEOUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args(argv)
    job = args.job
    timeout_s = args.timeout or float(JOBS[job]["timeout"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = LOG_DIR / f"{job}-{day}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    lock = acquire_lock(job, timeout_s)
    if lock is None:
        logging.warning("RUNNER_LOCK_BUSY job=%s", job)
        return EXIT_LOCKED
    state = load_state(job)
    state["last_attempt"] = _utcnow_iso()
    save_state(job, state)
    code = EXIT_OK
    try:
        with open(log_path, "a", encoding="utf-8") as log_fh:
            log_fh.write(f"\n===== RUN job={job} at={_utcnow_iso()} timeout={timeout_s}s =====\n")
            log_fh.flush()
            if job == "status":
                for script in ("scripts/analytics/binance_positioning_b2_status.py",
                               "scripts/analytics/cftc_r4_status.py"):
                    proc = subprocess.Popen(
                        [sys.executable, script], cwd=str(REPO_ROOT),
                        stdout=log_fh, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, text=True)
                    try:
                        rc = proc.wait(timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        rc = 124
                    if rc != 0:
                        code = EXIT_FAILED
            else:
                code = run_job(job, timeout_s, log_fh)
    finally:
        release_lock(job)
    state = load_state(job)
    if code == EXIT_OK:
        state["last_success"] = _utcnow_iso()
        state["consecutive_failures"] = 0
    else:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        state["last_error_code"] = code
    save_state(job, state)
    logging.info("RUNNER_DONE job=%s exit=%d", job, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
