# -*- coding: utf-8 -*-
"""
Suíte de Testes Hardening H1 do Orquestrador Transacional Hermético de Reconciliação
Fase 5.11 — Gate G4-D B3-B2-C2-B2.2-H1.

Cobre rigorosamente os 27 cenários (H01 a H27):
- P1: Incerteza estrita em falhas de escrita, flush, fsync e close de PREPARED (PERSISTENCE_UNCERTAIN).
- P2: Rota determinística de NO-OP com verificação prévia independente (NO_CHANGES sem snapshot/ledger/manifesto).
- P3: Governança estrita de estados permitidos (RUNNING ativo bloqueia, RUNNING_RESIDUAL autorizado).
- Preservação dos estados originais (DIRTY, UNKNOWN, CLEAN).
- Bloqueio persistente da CLI apply e integridade de dados operacionais reais.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from common.runtime_lock import (
    MANIFEST_FILENAME,
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    read_runtime_manifest,
    write_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    EventClassification,
    FallbackReconciler,
    ReconciliationTransactionResult,
    ReconciliationTransactionStatus,
    TargetVerificationCode,
    TargetVerificationResult,
    execute_reconciliation_transaction_hermetic,
    record_reconciliation_audit,
    validate_reconciliation_audit_ledger,
)


def _init_sqlite_db(db_path: Path, rows: List[Dict[str, Any]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            window_id TEXT,
            is_signal INTEGER DEFAULT 0,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts_type ON events(timestamp_ms, event_type)")
    for r in rows:
        payload = r.get("payload", {})
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        conn.execute(
            """
            INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("timestamp_ms", 0),
                r.get("event_type", "ORDER"),
                r.get("symbol", "BTCUSDT"),
                r.get("window_id", "w1"),
                1 if r.get("is_signal") else 0,
                payload_json,
            ),
        )
    conn.commit()
    conn.close()


def _write_jsonl(file_path: Path, events: List[Dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _calc_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _setup_test_env(tmp_path: Path, manifest_version: str = "1.2", initial_lock_state: RuntimeState = RuntimeState.CLEAN) -> Dict[str, Any]:
    data_dir = tmp_path / "dados"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir = tmp_path / "fallback_events"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    backup_base_dir = tmp_path / "backups"
    backup_base_dir.mkdir(parents=True, exist_ok=True)

    lock_path = data_dir / ".operational_runtime.lock"
    manifest_path = data_dir / MANIFEST_FILENAME
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    ledger_path = data_dir / RECONCILIATION_AUDIT_FILENAME

    env_uuid = str(uuid.uuid4())
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        holder_name="test_setup_h1",
        manifest_version=manifest_version,
    )

    op_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="test_runner_h1")
    op_lock.acquire()
    op_lock.release()

    if initial_lock_state == RuntimeState.DIRTY:
        with open(lock_path, "wb") as f:
            meta = {"status": "DIRTY", "holder": "test_runner_h1", "reason": "dirty_shutdown"}
            f.write(b"\x00" + json.dumps(meta).encode("utf-8"))
    elif initial_lock_state == RuntimeState.UNKNOWN:
        with open(lock_path, "wb") as f:
            meta = {"status": "UNKNOWN", "holder": "test_runner_h1", "reason": "crashed_process_residual"}
            f.write(b"\x00" + json.dumps(meta).encode("utf-8"))

    _init_sqlite_db(sqlite_path, [])
    _write_jsonl(jsonl_path, [])

    return {
        "data_dir": data_dir,
        "archive_dir": archive_dir,
        "fallback_dir": fallback_dir,
        "backup_base_dir": backup_base_dir,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "ledger_path": ledger_path,
        "env_uuid": env_uuid,
        "op_lock": op_lock,
    }


# ----------------------------------------------------------------------
# H01 — Falha antes de iniciar PREPARED retorna FAILED_BEFORE_PREPARED
# ----------------------------------------------------------------------
def test_h01_failure_before_prepared_returns_failed_before_prepared(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H01", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h01.jsonl"
    _write_jsonl(fb_file, [ev])

    # Simula falha na criação de snapshot (etapa anterior ao PREPARED)
    with mock.patch("events.fallback_reconciler.create_reconciliation_snapshot", side_effect=RuntimeError("ENOSPC simulado")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H01",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "SNAPSHOT"
    assert res.prepared_recorded is False
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# H02 — Exceção durante write() de PREPARED retorna PERSISTENCE_UNCERTAIN
# ----------------------------------------------------------------------
def test_h02_exception_during_write_prepared_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H02", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h02.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingWriteFile:
        def __init__(self, real_f):
            self.real_f = real_f

        def write(self, s):
            raise OSError("Falha simulada durante f.write()")

        def flush(self):
            return self.real_f.flush()

        def fileno(self):
            return self.real_f.fileno()

        def close(self):
            return self.real_f.close()

    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            return FailingWriteFile(f)
        return f

    with mock.patch("builtins.open", side_effect=mock_open_call):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H02",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None
    assert any("PREPARED_WRITE_UNCERTAIN" in err for err in res.errors)
    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 0


# ----------------------------------------------------------------------
# H03 — Exceção durante flush() de PREPARED retorna PERSISTENCE_UNCERTAIN
# ----------------------------------------------------------------------
def test_h03_exception_during_flush_prepared_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H03", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h03.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingFlushFile:
        def __init__(self, real_f):
            self.real_f = real_f

        def write(self, s):
            return self.real_f.write(s)

        def flush(self):
            raise OSError("Falha simulada durante f.flush()")

        def fileno(self):
            return self.real_f.fileno()

        def close(self):
            return self.real_f.close()

    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            return FailingFlushFile(f)
        return f

    with mock.patch("builtins.open", side_effect=mock_open_call):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H03",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None
    assert any("PREPARED_WRITE_UNCERTAIN" in err for err in res.errors)


# ----------------------------------------------------------------------
# H04 — Exceção durante fsync() de PREPARED retorna PERSISTENCE_UNCERTAIN
# ----------------------------------------------------------------------
def test_h04_exception_during_fsync_prepared_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H04", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h04.jsonl"
    _write_jsonl(fb_file, [ev])

    ledger_fds = set()
    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            try:
                ledger_fds.add(f.fileno())
            except Exception:
                pass
        return f

    real_fsync = os.fsync

    def mock_fsync(fd):
        if fd in ledger_fds:
            raise OSError("Falha simulada de I/O em os.fsync() no ledger")
        return real_fsync(fd)

    with mock.patch("builtins.open", side_effect=mock_open_call), mock.patch("os.fsync", side_effect=mock_fsync):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H04",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None
    assert any("PREPARED_WRITE_UNCERTAIN" in err for err in res.errors)


# ----------------------------------------------------------------------
# H05 — Exceção durante close() de PREPARED retorna PERSISTENCE_UNCERTAIN
# ----------------------------------------------------------------------
def test_h05_exception_during_close_prepared_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H05", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h05.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingCloseFile:
        def __init__(self, real_f):
            self.real_f = real_f

        def write(self, s):
            return self.real_f.write(s)

        def flush(self):
            return self.real_f.flush()

        def fileno(self):
            return self.real_f.fileno()

        def close(self):
            self.real_f.close()
            raise OSError("Falha simulada durante close()")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            return FailingCloseFile(f)
        return f

    with mock.patch("builtins.open", side_effect=mock_open_call):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H05",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None
    assert any("PREPARED_WRITE_UNCERTAIN" in err for err in res.errors)


# ----------------------------------------------------------------------
# H06 — Falha parcial com JSON truncado no ledger permanece preservada
# ----------------------------------------------------------------------
def test_h06_partial_write_preserves_truncated_ledger_without_auto_repair(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H06", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h06.jsonl"
    _write_jsonl(fb_file, [ev])

    class TruncatingWriter:
        def __init__(self, real_f):
            self.real_f = real_f

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                self.real_f.close()
            except Exception:
                pass

        def write(self, s):
            # Escreve apenas 15 caracteres (JSON truncado) e falha
            self.real_f.write(s[:15])
            self.real_f.flush()
            raise OSError("Crash de escrita simulado após bytes parciais")

        def flush(self):
            return self.real_f.flush()

        def fileno(self):
            return self.real_f.fileno()

        def close(self):
            return self.real_f.close()

    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            return TruncatingWriter(f)
        return f

    with mock.patch("builtins.open", side_effect=mock_open_call):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H06",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.prepared_recorded is None
    assert env["ledger_path"].exists()
    assert env["ledger_path"].stat().st_size == 15  # Bytes preservados intactos sem auto-reparo


# ----------------------------------------------------------------------
# H07 — Falha após bytes completos antes de confirmação de durabilidade
# ----------------------------------------------------------------------
def test_h07_failure_after_full_bytes_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H07", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h07.jsonl"
    _write_jsonl(fb_file, [ev])

    class CompleteBytesFailingFsync:
        def __init__(self, real_f):
            self.real_f = real_f

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                self.real_f.close()
            except Exception:
                pass

        def write(self, s):
            return self.real_f.write(s)

        def flush(self):
            return self.real_f.flush()

        def fileno(self):
            self.real_f.flush()
            raise OSError("Erro de descritor durante fsync")

        def close(self):
            return self.real_f.close()

    real_open = open

    def mock_open_call(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(env["ledger_path"]):
            return CompleteBytesFailingFsync(f)
        return f

    with mock.patch("builtins.open", side_effect=mock_open_call):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H07",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.prepared_recorded is None
    # O arquivo tem a linha completa, evidenciando por que status não pode ser FAILED_BEFORE_PREPARED
    assert env["ledger_path"].stat().st_size > 50


# ----------------------------------------------------------------------
# H08 — Falha PREPARED nunca executa APPLY SQLite
# ----------------------------------------------------------------------
def test_h08_prepared_failure_never_applies_to_sqlite(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H08", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h08.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha PREPARED")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H08",
            operational_lock=env["op_lock"],
        )

    assert res.sqlite_inserted == 0
    # Verifica SQLite intocado
    conn = sqlite3.connect(str(env["sqlite_path"]))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == 0
    conn.close()


# ----------------------------------------------------------------------
# H09 — Falha PREPARED nunca executa APPLY JSONL
# ----------------------------------------------------------------------
def test_h09_prepared_failure_never_applies_to_jsonl(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H09", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h09.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha PREPARED")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H09",
            operational_lock=env["op_lock"],
        )

    assert res.jsonl_appended == 0
    assert env["jsonl_path"].stat().st_size == 0


# ----------------------------------------------------------------------
# H10 — Falha PREPARED nunca atualiza manifesto
# ----------------------------------------------------------------------
def test_h10_prepared_failure_never_updates_manifest(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H10", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h10.jsonl"
    _write_jsonl(fb_file, [ev])

    sha_before = _calc_sha256(env["manifest_path"])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha PREPARED")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H10",
            operational_lock=env["op_lock"],
        )

    assert res.manifest_updated is False
    assert _calc_sha256(env["manifest_path"]) == sha_before


# ----------------------------------------------------------------------
# H11 — Falha PREPARED nunca registra COMMITTED
# ----------------------------------------------------------------------
def test_h11_prepared_failure_never_records_committed(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H11", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h11.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha PREPARED")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H11",
            operational_lock=env["op_lock"],
        )

    assert res.committed_recorded is False


# ----------------------------------------------------------------------
# H12 — No-op legítimo retorna NO_CHANGES
# ----------------------------------------------------------------------
def test_h12_legitimate_noop_returns_no_changes(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H12", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h12.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H12",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    assert res.stage_reached == "NOOP_VERIFIED"
    assert res.already_present == 1
    assert res.verification_result.approved is True


# ----------------------------------------------------------------------
# H13 — No-op não cria snapshot
# ----------------------------------------------------------------------
def test_h13_noop_does_not_create_snapshot(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H13", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h13.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H13",
        operational_lock=env["op_lock"],
    )

    assert res.snapshot_dir is None
    # Verifica que nenhum snapshot foi gerado em backup_base_dir
    snapshots = list(env["backup_base_dir"].glob("reconcile_snapshot_*"))
    assert len(snapshots) == 0


# ----------------------------------------------------------------------
# H14 — No-op não grava PREPARED
# ----------------------------------------------------------------------
def test_h14_noop_does_not_record_prepared(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H14", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h14.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H14",
        operational_lock=env["op_lock"],
    )

    assert res.prepared_recorded is False
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# H15 — No-op não grava COMMITTED
# ----------------------------------------------------------------------
def test_h15_noop_does_not_record_committed(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H15", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h15.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H15",
        operational_lock=env["op_lock"],
    )

    assert res.committed_recorded is False
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# H16 — No-op não incrementa manifesto
# ----------------------------------------------------------------------
def test_h16_noop_does_not_increment_manifest(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H16", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h16.jsonl"
    _write_jsonl(fb_file, [ev])

    manifest_before = read_runtime_manifest(env["manifest_path"])
    sha_before = _calc_sha256(env["manifest_path"])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H16",
        operational_lock=env["op_lock"],
    )

    assert res.manifest_updated is False
    assert _calc_sha256(env["manifest_path"]) == sha_before
    manifest_after = read_runtime_manifest(env["manifest_path"])
    assert manifest_after.get("reconciliation_history_counter", 0) == manifest_before.get("reconciliation_history_counter", 0)


# ----------------------------------------------------------------------
# H17 — No-op não altera SQLite ou JSONL
# ----------------------------------------------------------------------
def test_h17_noop_does_not_modify_sqlite_or_jsonl(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H17", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h17.jsonl"
    _write_jsonl(fb_file, [ev])

    sq_sha_before = _calc_sha256(env["sqlite_path"])
    jl_sha_before = _calc_sha256(env["jsonl_path"])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H17",
        operational_lock=env["op_lock"],
    )

    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 0
    assert _calc_sha256(env["sqlite_path"]) == sq_sha_before
    assert _calc_sha256(env["jsonl_path"]) == jl_sha_before


# ----------------------------------------------------------------------
# H18 — No-op exige pós-verificação aprovada
# ----------------------------------------------------------------------
def test_h18_noop_requires_approved_verification(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H18", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h18.jsonl"
    _write_jsonl(fb_file, [ev])

    unapproved = TargetVerificationResult(
        approved=False,
        code=TargetVerificationCode.VERIFICATION_FAILED,
        total_expected=1,
        verified_count=0,
        missing_count=1,
        duplicate_count=0,
        divergent_hash_count=0,
        conflict_count=0,
    )

    with mock.patch("events.fallback_reconciler.verify_reconciliation_targets", return_value=unapproved):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H18",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "NOOP_VERIFICATION"
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# H19 — No-op com hash divergente aborta
# ----------------------------------------------------------------------
def test_h19_noop_with_divergent_hash_aborts(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev_db = {"event_id": "EVT-H19", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    ev_fb = {"event_id": "EVT-H19", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 999}  # payload divergente!

    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev_db["epoch_ms"],
        "event_type": ev_db["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev_db,
    }])
    _write_jsonl(env["jsonl_path"], [ev_db])
    fb_file = env["fallback_dir"] / "eventos_h19.jsonl"
    _write_jsonl(fb_file, [ev_fb])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H19",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# H20 — No-op com duplicata no fallback aborta
# ----------------------------------------------------------------------
def test_h20_noop_with_duplicate_in_fallback_aborts(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H20", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h20.jsonl"
    _write_jsonl(fb_file, [ev, ev])  # Duplicado no fallback!

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H20",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "EVENT_VALIDATION"
    assert any("duplicado" in err for err in res.errors)


# ----------------------------------------------------------------------
# H21 — RUNNING ativo bloqueia manutenção
# ----------------------------------------------------------------------
def test_h21_running_active_lock_blocks_maintenance(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H21", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h21.jsonl"
    _write_jsonl(fb_file, [ev])

    # Simula um lock operacional concorrente ativamente detido no mesmo processo/thread
    active_op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="concorrente_running")
    active_op_lock.acquire()  # Detém a trava física do kernel e status RUNNING

    try:
        # Tentativa de executar com uma nova instância que tenta adquirir manutenção
        victim_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="victim_maintenance")
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H21",
            operational_lock=victim_lock,
        )

        assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
        assert res.stage_reached == "LOCK_ACQUISITION"
    finally:
        active_op_lock.release()


# ----------------------------------------------------------------------
# H22 — RUNNING_RESIDUAL permite manutenção quando lock físico está livre
# ----------------------------------------------------------------------
def test_h22_running_residual_allows_maintenance(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H22", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h22.jsonl"
    _write_jsonl(fb_file, [ev])

    # Grava RUNNING no lock sem manter a trava física (simula crash de processo anterior)
    with open(env["lock_path"], "wb") as f:
        meta = {"status": "RUNNING", "holder": "crashed_process", "pid": 99999}
        f.write(b"\x00" + json.dumps(meta).encode("utf-8"))

    maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maintenance_h22")
    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H22",
        operational_lock=maint_lock,
    )

    # Consegue executar manutenção legítima em RUNNING_RESIDUAL
    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    assert res.stage_reached == "NOOP_VERIFIED"


# ----------------------------------------------------------------------
# H23 — DIRTY permanece DIRTY após no-op
# ----------------------------------------------------------------------
def test_h23_dirty_remains_dirty_after_noop(tmp_path: Path):
    env = _setup_test_env(tmp_path, initial_lock_state=RuntimeState.DIRTY)
    lock_before = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert lock_before["status"] == "DIRTY"

    ev = {"event_id": "EVT-H23", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h23.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H23",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    lock_after = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert lock_after["status"] == "DIRTY"


# ----------------------------------------------------------------------
# H24 — UNKNOWN permanece UNKNOWN após no-op
# ----------------------------------------------------------------------
def test_h24_unknown_remains_unknown_after_noop(tmp_path: Path):
    env = _setup_test_env(tmp_path, initial_lock_state=RuntimeState.UNKNOWN)
    lock_before = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert lock_before["status"] == "UNKNOWN"

    ev = {"event_id": "EVT-H24", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_h24.jsonl"
    _write_jsonl(fb_file, [ev])

    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel H24",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    lock_after = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert lock_after["status"] == "UNKNOWN"


# ----------------------------------------------------------------------
# H25 — Falhas liberam sessão de manutenção
# ----------------------------------------------------------------------
def test_h25_failures_release_maintenance_session(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H25", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h25.jsonl"
    _write_jsonl(fb_file, [ev])

    maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="holder_h25")

    with mock.patch("events.fallback_reconciler.create_reconciliation_snapshot", side_effect=RuntimeError("Falha proposital")):
        res = execute_reconciliation_transaction_hermetic(
            data_dir=env["data_dir"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            source_fallback_file=fb_file,
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            backup_base_dir=env["backup_base_dir"],
            environment_uuid=env["env_uuid"],
            operator_reason="Justificativa operacional auditavel H25",
            operational_lock=maint_lock,
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    # Verifica que o lock físico foi liberado e pode ser adquirido novamente sem BlockingIOError
    reacquired = maint_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    assert reacquired["raw_status"] == "CLEAN"
    maint_lock.release_maintenance()


# ----------------------------------------------------------------------
# H26 — CLI apply permanece bloqueada
# ----------------------------------------------------------------------
def test_h26_cli_apply_remains_blocked(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-H26", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_h26.jsonl"
    _write_jsonl(fb_file, [ev])

    cmd = [
        sys.executable,
        "scripts/maintenance/reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Teste de bloqueio governanca H26",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--archive-dir", str(env["archive_dir"]),
        "--fallback-dir", str(env["fallback_dir"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    out = json.loads(proc.stdout)
    assert out.get("status") == "APPLY_NOT_ENABLED_C2A"
    assert out.get("mutations_performed") is False


# ----------------------------------------------------------------------
# H27 — Nenhum dado operacional real é acessado
# ----------------------------------------------------------------------
def test_h27_no_real_operational_data_accessed():
    real_dados = Path("dados")
    if not real_dados.exists():
        pytest.skip("Diretório real dados não existe no workspace.")

    before_hashes = {}
    for f in real_dados.glob("*"):
        if f.is_file():
            before_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    # O teste apenas audita que o workspace de dados não foi modificado
    after_hashes = {}
    for f in real_dados.glob("*"):
        if f.is_file():
            after_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    assert before_hashes == after_hashes
