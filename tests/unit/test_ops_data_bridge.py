# tests/unit/test_ops_data_bridge.py
# -*- coding: utf-8 -*-
"""
OPS-DATA-BRIDGE — Watchdog local + backup. Determinístico, sem rede.
Cobre: níveis OK/WARN/CRIT/UNKNOWN + exit codes, backup/retention/
restore-test, prune, checksum mismatch. Sem produção, sem scheduler.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

import research_backup as backup
import research_local_watchdog as watchdog
from binance_positioning_store import B2Store


def _seed_store(root, now, age_min, n=5):
    store = B2Store(root)
    base_ts = int(now.timestamp() * 1000) - int(age_min * 60000)
    for i in range(n):
        ts = base_ts - (n - 1 - i) * 5 * 60 * 1000
        store.append_raw("BTCUSDT", "global", ts,
                         {"longShortRatio": "1.2",
                          "longAccount": "0.5455", "shortAccount": "0.4545"},
                         retrieved_at=now.isoformat(), mode="LIVE_OBSERVED")
    return store


def test_watchdog_ok_warn_crit_unknown(tmp_path):
    now = datetime.now(timezone.utc)
    assert watchdog.check("BTCUSDT", str(tmp_path / "empty"), now)["exit"] == 3
    _seed_store(tmp_path / "s_ok", now, age_min=5)
    assert watchdog.check("BTCUSDT", str(tmp_path / "s_ok"), now)["exit"] == 0
    _seed_store(tmp_path / "s_warn", now, age_min=45)
    r = watchdog.check("BTCUSDT", str(tmp_path / "s_warn"), now)
    assert r["exit"] == 1 and r["level"] == "WARN"
    _seed_store(tmp_path / "s_crit", now, age_min=150)
    r = watchdog.check("BTCUSDT", str(tmp_path / "s_crit"), now)
    assert r["exit"] == 2 and r["level"] == "CRIT"


def test_backup_roundtrip_retention_restore(tmp_path, monkeypatch):
    src = tmp_path / "research" / "binance_positioning"
    (src / "raw" / "BTCUSDT").mkdir(parents=True)
    (src / "raw" / "BTCUSDT" / "global.jsonl").write_text(
        '{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (src / "normalized").mkdir(parents=True)
    (src / "normalized" / "BTCUSDT.jsonl").write_text('{"b": 1}\n', encoding="utf-8")
    monkeypatch.setattr(backup, "SOURCES", [src])
    dest = tmp_path / "dest"
    res = backup.create_backup(dest, retention_days=35)
    assert res["files"] == 2 and res["pruned"] == []
    assert res["zip"].endswith(".zip")
    # restore test em temp (nunca no live)
    rt = backup.restore_test(res["zip"])
    assert rt["ok"] is True and rt["files_checked"] == 2
    # .env/secrets/db nunca entram
    (src / "x.env").write_text("K=V", encoding="utf-8")
    (src / "trading_bot.db").write_text("x", encoding="utf-8")
    res2 = backup.create_backup(dest, retention_days=35)
    names = json.loads((dest / (res2["zip"].split("/")[-1].replace(
        ".zip", ".manifest.json"))).read_text(encoding="utf-8"))
    assert all("x.env" not in f["path"] and "trading_bot.db" not in f["path"]
               for f in names["files"])
    # prune além da retenção
    old = dest / "research-backup-20200101T000000Z.zip"
    old.write_text("old", encoding="utf-8")
    (dest / "research-backup-20200101T000000Z.manifest.json").write_text(
        "{}", encoding="utf-8")
    (dest / "research-backup-20200101T000000Z.sha256").write_text(
        "x research-backup-20200101T000000Z.zip", encoding="utf-8")
    pruned = backup.prune(dest, 35)
    assert "research-backup-20200101T000000Z.zip" in pruned
    assert not old.exists()


def test_restore_checksum_mismatch(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"data")
    (tmp_path / "x.sha256").write_text("0" * 64 + "  x.zip", encoding="utf-8")
    rt = backup.restore_test(str(tmp_path / "x.zip"))
    assert rt == {"ok": False, "reason": "checksum_mismatch"}
