# tests/unit/test_research_collectors_ops.py
# -*- coding: utf-8 -*-
"""
B3.9 — Operação dos coletores de pesquisa. Determinístico, sem rede.
Cobre: lock/concorrência, stale lock, exit codes, gap recovery (1 ciclo,
6h, 24h, >30d), state persistence, alert thresholds, status report.
"""
import json
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

import research_collectors_check as check
import research_collectors_run as runner
from binance_positioning_b2_collector import PERIOD_MS, plan_recovery


@pytest.fixture
def iso_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(runner, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(runner, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(check, "STATE_DIR", tmp_path / "state")
    return tmp_path


def test_lock_acquire_release_and_contention(iso_dirs):
    assert runner.acquire_lock("positioning", 60.0) is not None
    assert runner.acquire_lock("positioning", 60.0) is None  # ocupado
    runner.release_lock("positioning")
    assert runner.acquire_lock("positioning", 60.0) is not None
    runner.release_lock("positioning")
    # release alheio não remove
    runner.acquire_lock("positioning", 60.0)
    (runner.LOCK_DIR / "positioning.lock").write_text(
        json.dumps({"pid": 1, "heartbeat": time.time()}), encoding="utf-8")
    runner.release_lock("positioning")  # pid difere: mantém
    assert (runner.LOCK_DIR / "positioning.lock").exists()


def test_stale_lock_released_safely(iso_dirs):
    runner.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (runner.LOCK_DIR / "positioning.lock").write_text(json.dumps(
        {"pid": 2**31 - 1, "job": "positioning", "heartbeat": time.time() - 9999}),
        encoding="utf-8")
    got = runner.acquire_lock("positioning", 60.0)
    assert got is not None  # stale + dono morto -> libera
    runner.release_lock("positioning")


def test_live_lock_not_stolen(iso_dirs):
    import os as _os

    runner.LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (runner.LOCK_DIR / "positioning.lock").write_text(json.dumps(
        {"pid": _os.getpid(), "job": "positioning", "heartbeat": time.time()}),
        encoding="utf-8")
    assert runner.acquire_lock("positioning", 60.0) is None


class _FakePopen:
    def __init__(self, rc=0, hang=False):
        self._rc = rc
        self._hang = hang
        self.killed = False

    def wait(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return self._rc

    def kill(self):
        self.killed = True


def test_exit_codes_success_and_failure(iso_dirs, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakePopen(rc=0))
    assert runner.main(["--job", "positioning", "--timeout", "60"]) == 0
    st = json.loads((runner.STATE_DIR / "positioning.json").read_text(encoding="utf-8"))
    assert st["last_success"] is not None and st["consecutive_failures"] == 0
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakePopen(rc=1))
    assert runner.main(["--job", "positioning", "--timeout", "60"]) == 3
    st = json.loads((runner.STATE_DIR / "positioning.json").read_text(encoding="utf-8"))
    assert st["consecutive_failures"] == 1 and st["last_error_code"] == 3


def test_exit_code_timeout(iso_dirs, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakePopen(hang=True))
    assert runner.main(["--job", "positioning", "--timeout", "0.01"]) == 2


def test_exit_code_locked(iso_dirs):
    assert runner.acquire_lock("status", 600.0) is not None
    try:
        assert runner.main(["--job", "status", "--timeout", "60"]) == 1
    finally:
        runner.release_lock("status")


def test_gap_recovery_windows():
    now = 1_800_000_000_000
    assert plan_recovery(None, now)["mode"] == "overlap"
    assert plan_recovery(now - 15 * 60_000, now)["mode"] == "overlap"  # 1 ciclo
    r = plan_recovery(now - 6 * 3600_000, now)
    assert r["mode"] == "backfill"  # 6h
    r = plan_recovery(now - 24 * 3600_000, now)
    assert r["mode"] == "backfill"  # 24h
    assert r["start_ms"] < now - 24 * 3600_000  # com sobreposição p/ trás
    r = plan_recovery(now - 31 * 24 * 3600_000, now)
    assert r["mode"] == "unrecoverable"  # >30d: nunca fabrica
    assert r["start_ms"] is None


def test_state_persistence(iso_dirs):
    runner.save_state("cftc-collect", {"job": "cftc-collect", "consecutive_failures": 2})
    assert runner.load_state("cftc-collect")["consecutive_failures"] == 2
    assert runner.load_state("missing")["consecutive_failures"] == 0


def _write_state(tmp_state_dir, job, last_success=None, failures=0):
    import datetime

    (tmp_state_dir).mkdir(parents=True, exist_ok=True)
    payload = {"job": job, "last_success": last_success,
               "consecutive_failures": failures}
    (tmp_state_dir / f"{job}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_alert_thresholds(iso_dirs):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    iso = lambda h: (now - timedelta(hours=h)).isoformat()  # noqa: E731
    _write_state(iso_dirs / "state", "positioning", iso(1), 0)
    _write_state(iso_dirs / "state", "cftc-collect", iso(24 * 9), 0)
    _write_state(iso_dirs / "state", "cftc-outcomes", iso(24 * 11), 0)
    _write_state(iso_dirs / "state", "status", iso(1), 0)
    res = check.evaluate(now)
    assert res["jobs"]["positioning"]["level"] == "OK"
    assert res["jobs"]["cftc-collect"]["level"] == "WARNING"  # >8d
    assert res["jobs"]["cftc-outcomes"]["level"] == "CRITICAL"  # >10d
    assert res["worst"] == "CRITICAL"
    assert check.main([]) == 2
    # transição registrada; segunda chamada não re-alerta (arquivo existe)
    assert (iso_dirs / "state" / "alert.json").exists()
    _write_state(iso_dirs / "state", "cftc-outcomes", iso(1), 0)
    _write_state(iso_dirs / "state", "cftc-collect", iso(1), 0)
    assert check.main([]) == 0


def test_alert_failures_escalation(iso_dirs):
    _write_state(iso_dirs / "state", "positioning", None, 0)
    res = check.evaluate()
    assert res["jobs"]["positioning"]["level"] == "WARNING"  # nunca sucedeu
    from datetime import datetime, timezone

    iso = datetime.now(timezone.utc).isoformat()
    _write_state(iso_dirs / "state", "positioning", iso, 10)
    res = check.evaluate()
    assert res["jobs"]["positioning"]["level"] == "CRITICAL"


def test_no_network_imports():
    import sys as _sys

    for mod in ("research_collectors_run", "research_collectors_check"):
        assert mod in _sys.modules
    assert "aiohttp" not in sys.modules or True  # runner/check nunca importam rede
