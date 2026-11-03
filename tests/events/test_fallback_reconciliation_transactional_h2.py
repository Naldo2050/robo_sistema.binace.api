"""
Suíte de Testes Herméticos de Hardening Complementar (Gate G4-D B3-B2-C2-B2.2-H2).

Cobre os achados de reauditoria:
- P1-CORRUPT-LOCK-APPLY-ADMISSION-01: Distinção estrita entre CASO A (metadados válidos com status UNKNOWN)
  e CASO B (metadados ausentes, vazios, truncados, ilegíveis ou estruturalmente inválidos),
  com rejeição fail-closed no preflight (CORRUPT_RUNTIME_LOCK_METADATA) impedindo snapshot, PREPARED e APPLY.
- P2-AMBIGUOUS-PREPARED-BOOLEAN-02: Semântica tripartite estrita de prepared_recorded (True, False, None),
  semântica de committed_recorded, serialização fiel e proibição de replay automático.
- Regressão e invariantes de governança financeira.

Total de 32 testes divididos nos Grupos A, B e C.
"""

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from common.runtime_lock import (
    MANIFEST_FILENAME,
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    write_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    ReconciliationTransactionResult,
    ReconciliationTransactionStatus,
    append_event_to_jsonl,
    compute_canonical_payload_hash,
    execute_reconciliation_transaction_hermetic,
    insert_event_to_sqlite,
    record_reconciliation_audit,
    validate_reconciliation_audit_ledger,
    verify_reconciliation_targets,
)


def _init_sqlite_db(db_path: Path, rows: Optional[List[Dict[str, Any]]] = None) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            window_id TEXT,
            is_signal BOOLEAN DEFAULT 0,
            payload JSON NOT NULL,
            canonical_hash TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_ms)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)")

    if rows:
        for r in rows:
            p_json = json.dumps(r.get("payload", {}))
            c_hash = r.get("canonical_hash") or compute_canonical_payload_hash(r.get("payload", {}))
            cur.execute("""
                INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload, canonical_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                r["timestamp_ms"],
                r["event_type"],
                r.get("symbol", "BTCUSDT"),
                r.get("window_id", "w1"),
                1 if r.get("is_signal") else 0,
                p_json,
                c_hash,
            ))
    conn.commit()
    conn.close()


def _write_jsonl(file_path: Path, items: List[Dict[str, Any]], trailing_newline: bool = True) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        for it in items:
            raw = json.dumps(it, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            f.write(raw + b"\n")
        if not trailing_newline and items:
            f.seek(-1, os.SEEK_END)
            f.truncate()


def _setup_test_env(
    tmp_path: Path,
    manifest_version: str = "1.2",
    lock_raw_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
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
        holder_name="test_setup_h2",
        manifest_version=manifest_version,
    )

    op_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="test_runner_h2")
    op_lock.acquire()
    op_lock.release()

    if lock_raw_bytes is not None:
        with open(lock_path, "wb") as f:
            f.write(lock_raw_bytes)

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


# ======================================================================
# GRUPO A — INTEGRIDADE DOS METADADOS DO LOCK (A01 a A12)
# ======================================================================

def test_a01_lock_valid_json_with_explicit_unknown_status_proceeds(tmp_path: Path):
    """A01 — CASO A: Lock com JSON estruturalmente válido e status UNKNOWN procede normalmente."""
    meta = {"status": "UNKNOWN", "holder": "crashed_worker", "reason": "unclean_shutdown", "timestamp_utc": "2026-10-17T00:00:00Z"}
    lock_bytes = b"\x00" + json.dumps(meta).encode("utf-8")
    env = _setup_test_env(tmp_path, lock_raw_bytes=lock_bytes)

    ev = {"event_id": "EVT-A01", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_a01.jsonl"
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
        operator_reason="Justificativa operacional auditavel A01",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert res.stage_reached == "COMMITTED"
    assert res.prepared_recorded is True
    assert res.committed_recorded is True
    assert res.sqlite_inserted == 1
    assert res.jsonl_appended == 1


def test_a02_empty_lock_file_blocks_apply_fail_closed(tmp_path: Path):
    """A02 — CASO B: Lock vazio (0 bytes) bloqueia APPLY no preflight com CORRUPT_RUNTIME_LOCK_METADATA."""
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"")

    ev = {"event_id": "EVT-A02", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a02.jsonl"
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
        operator_reason="Justificativa operacional auditavel A02",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert res.prepared_recorded is False
    assert any("CORRUPT_RUNTIME_LOCK_METADATA" in err for err in res.errors)
    assert any("empty_lock_file" in err for err in res.errors)


def test_a03_truncated_json_lock_blocks_apply_fail_closed(tmp_path: Path):
    """A03 — CASO B: Lock com JSON truncado bloqueia APPLY com CORRUPT_RUNTIME_LOCK_METADATA."""
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00{\"status\": \"UNKN")

    ev = {"event_id": "EVT-A03", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a03.jsonl"
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
        operator_reason="Justificativa operacional auditavel A03",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert any("CORRUPT_RUNTIME_LOCK_METADATA" in err for err in res.errors)


def test_a04_corrupted_binary_lock_blocks_apply_fail_closed(tmp_path: Path):
    """A04 — CASO B: Lock com lixo binário bloqueia APPLY com CORRUPT_RUNTIME_LOCK_METADATA."""
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00\xff\xfe\x00\x12\x34\x56\x78")

    ev = {"event_id": "EVT-A04", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a04.jsonl"
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
        operator_reason="Justificativa operacional auditavel A04",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert any("CORRUPT_RUNTIME_LOCK_METADATA" in err for err in res.errors)


def test_a05_lock_without_status_key_blocks_apply(tmp_path: Path):
    """A05 — CASO B: Lock com JSON válido mas sem a chave 'status' bloqueia APPLY."""
    meta = {"holder": "test_runner", "pid": 1234, "reason": "no_status_key"}
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00" + json.dumps(meta).encode("utf-8"))

    ev = {"event_id": "EVT-A05", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a05.jsonl"
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
        operator_reason="Justificativa operacional auditavel A05",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert any("missing_status_key" in err for err in res.errors)


def test_a06_lock_with_invalid_status_type_blocks_apply(tmp_path: Path):
    """A06 — CASO B: Lock com campo 'status' de tipo numérico ou lista bloqueia APPLY."""
    meta = {"status": 12345, "holder": "test_runner"}
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00" + json.dumps(meta).encode("utf-8"))

    ev = {"event_id": "EVT-A06", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a06.jsonl"
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
        operator_reason="Justificativa operacional auditavel A06",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached in ("PREFLIGHT", "LOCK_ACQUISITION")
    assert any("invalid_status_type_or_empty" in err or "não é elegível" in err for err in res.errors)


def test_a07_lock_with_empty_status_string_blocks_apply(tmp_path: Path):
    """A07 — CASO B: Lock com status string vazia bloqueia APPLY."""
    meta = {"status": "   ", "holder": "test_runner"}
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00" + json.dumps(meta).encode("utf-8"))

    ev = {"event_id": "EVT-A07", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a07.jsonl"
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
        operator_reason="Justificativa operacional auditavel A07",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached in ("PREFLIGHT", "LOCK_ACQUISITION")
    assert any("invalid_status_type_or_empty" in err or "não é elegível" in err for err in res.errors)


def test_a08_running_residual_lock_allows_maintenance_and_retains_residual_state(tmp_path: Path):
    """A08 — RUNNING com lock físico livre é detectado como RUNNING_RESIDUAL e tem metadados válidos."""
    meta = {"status": "RUNNING", "holder": "crashed_process", "pid": 9999}
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00" + json.dumps(meta).encode("utf-8"))

    ev = {"event_id": "EVT-A08", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_a08.jsonl"
    _write_jsonl(fb_file, [ev])

    maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maint_a08")
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
        operator_reason="Justificativa operacional auditavel A08",
        operational_lock=maint_lock,
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    raw_after = env["op_lock"].read_persisted_state(env["lock_path"])
    assert raw_after["raw_status"] == "RUNNING"


def test_a09_running_active_concurrency_blocks_before_metadata_check(tmp_path: Path):
    """A09 — Lock retido fisicamente por outro processo ativo rejeita em LOCK_ACQUISITION."""
    env = _setup_test_env(tmp_path)
    active_process_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="active_daemon")
    active_process_lock.acquire()

    try:
        ev = {"event_id": "EVT-A09", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
        fb_file = env["fallback_dir"] / "eventos_a09.jsonl"
        _write_jsonl(fb_file, [ev])

        concurrent_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="concurrent_operator")
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
            operator_reason="Justificativa operacional auditavel A09",
            operational_lock=concurrent_lock,
        )

        assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
        assert res.stage_reached == "LOCK_ACQUISITION"
        assert any("Conflito de exclusividade" in err for err in res.errors)
    finally:
        active_process_lock.release()


def test_a10_corrupt_lock_rejection_proves_zero_side_effects(tmp_path: Path):
    """A10 — Rejeição por lock corrompido não gera snapshots, PREPARED, inserts, appends, manifesto ou COMMITTED."""
    corrupt_bytes = b"\x00{corrupted_json"
    env = _setup_test_env(tmp_path, lock_raw_bytes=corrupt_bytes)

    ev = {"event_id": "EVT-A10", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 999}
    fb_file = env["fallback_dir"] / "eventos_a10.jsonl"
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
        operator_reason="Justificativa operacional auditavel A10",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False
    assert res.committed_recorded is False
    assert res.manifest_updated is False

    # 1. Zero snapshots criados
    assert len(list(env["backup_base_dir"].glob("*"))) == 0

    # 2. Zero registros de PREPARED / COMMITTED (ledger não criado)
    assert not env["ledger_path"].exists()

    # 3. Zero inserções no SQLite
    conn = sqlite3.connect(str(env["sqlite_path"]))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == 0
    conn.close()

    # 4. Zero appends no JSONL
    assert env["jsonl_path"].stat().st_size == 0

    # 5. Lock corrompido preservado exatamente com os bytes originais (sem auto-reparo)
    assert env["lock_path"].read_bytes() == corrupt_bytes


def test_a11_corrupt_lock_rejection_releases_physical_lock(tmp_path: Path):
    """A11 — Após rejeição por lock corrompido, a trava física é liberada adequadamente no finally."""
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"")

    ev = {"event_id": "EVT-A11", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_a11.jsonl"
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
        operator_reason="Justificativa operacional auditavel A11",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED

    # Comprova que o descritor físico no kernel está livre e pode ser travado por outra ferramenta
    another_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="recovery_tool")
    maint_info = another_lock.acquire_maintenance(allowed_states=[RuntimeState.UNKNOWN])
    assert maint_info["detected_status"] == "UNKNOWN"
    another_lock.release_maintenance()


def test_a12_noop_under_corrupt_lock_is_rejected_fail_closed(tmp_path: Path):
    """A12 — Rota NO_CHANGES sob lock corrompido é rejeitada no preflight: ambiente corrompido veda certificação."""
    corrupt_bytes = b"\x00{bad_json_in_lock"
    env = _setup_test_env(tmp_path, lock_raw_bytes=corrupt_bytes)

    ev = {"event_id": "EVT-A12", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_a12.jsonl"
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
        operator_reason="Justificativa operacional auditavel A12",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert any("CORRUPT_RUNTIME_LOCK_METADATA" in err for err in res.errors)
    assert env["lock_path"].read_bytes() == corrupt_bytes


# ======================================================================
# GRUPO B — SEMÂNTICA DE PREPARED E COMMITTED (B01 a B10)
# ======================================================================

def test_b01_failure_before_prepared_yields_prepared_recorded_false(tmp_path: Path):
    """B01 — Falha anterior a PREPARED (ex: snapshot) define prepared_recorded is False."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B01", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b01.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.create_reconciliation_snapshot", side_effect=RuntimeError("Snapshot fail")):
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
            operator_reason="Justificativa operacional auditavel B01",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False
    assert res.committed_recorded is False


def test_b02_write_exception_in_prepared_yields_prepared_recorded_none(tmp_path: Path):
    """B02 — Falha em write() de PREPARED define prepared_recorded is None."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B02", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b02.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingWriteFile:
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
            raise OSError("Falha simulada em f.write()")
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
            operator_reason="Justificativa operacional auditavel B02",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None
    assert res.committed_recorded is False


def test_b03_flush_exception_in_prepared_yields_prepared_recorded_none(tmp_path: Path):
    """B03 — Falha em flush() de PREPARED define prepared_recorded is None."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B03", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b03.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingFlushFile:
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
            raise OSError("Falha simulada em f.flush()")
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
            operator_reason="Justificativa operacional auditavel B03",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None


def test_b04_fsync_exception_in_prepared_yields_prepared_recorded_none(tmp_path: Path):
    """B04 — Falha em fsync() de PREPARED define prepared_recorded is None."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B04", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b04.jsonl"
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
            raise OSError("Falha simulada em fsync() do ledger")
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
            operator_reason="Justificativa operacional auditavel B04",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None


def test_b05_close_exception_in_prepared_yields_prepared_recorded_none(tmp_path: Path):
    """B05 — Falha em close() de PREPARED define prepared_recorded is None."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B05", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b05.jsonl"
    _write_jsonl(fb_file, [ev])

    class FailingCloseFile:
        def __init__(self, real_f):
            self.real_f = real_f
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
        def write(self, s):
            return self.real_f.write(s)
        def flush(self):
            return self.real_f.flush()
        def fileno(self):
            return self.real_f.fileno()
        def close(self):
            self.real_f.close()
            raise OSError("Falha simulada durante close()")

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
            operator_reason="Justificativa operacional auditavel B05",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert res.prepared_recorded is None


def test_b06_full_bytes_unconfirmed_durability_yields_none(tmp_path: Path):
    """B06 — Bytes completos gravados no arquivo mas sem confirmação física retornam prepared_recorded is None."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B06", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b06.jsonl"
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
            operator_reason="Justificativa operacional auditavel B06",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.prepared_recorded is None
    assert env["ledger_path"].stat().st_size > 50


def test_b07_confirmed_prepared_with_subsequent_sqlite_failure_yields_true(tmp_path: Path):
    """B07 — PREPARED confirmado com durabilidade e falha posterior no SQLite preserva prepared_recorded is True."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B07", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_b07.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Disk I/O error")):
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
            operator_reason="Justificativa operacional auditavel B07",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.stage_reached == "APPLY_SQLITE"
    assert res.prepared_recorded is True
    assert res.committed_recorded is False


def test_b08_noop_route_yields_prepared_and_committed_recorded_false(tmp_path: Path):
    """B08 — Rota NO_CHANGES define estritamente prepared_recorded=False e committed_recorded=False."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B08", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_b08.jsonl"
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
        operator_reason="Justificativa operacional auditavel B08",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    assert res.stage_reached == "NOOP_VERIFIED"
    assert res.prepared_recorded is False
    assert res.committed_recorded is False
    assert res.manifest_updated is False


def test_b09_to_dict_serialization_preserves_three_state_semantics():
    """B09 — Serialização to_dict() preserva True, False e None (null em JSON) fielmente."""
    res_none = ReconciliationTransactionResult(
        audit_id=str(uuid.uuid4()),
        environment_uuid=str(uuid.uuid4()),
        status=ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN,
        stage_reached="PREPARED",
        prepared_recorded=None,
        committed_recorded=None,
    )
    d_none = res_none.to_dict()
    assert d_none["prepared_recorded"] is None
    assert d_none["committed_recorded"] is None

    json_str = json.dumps(d_none)
    parsed = json.loads(json_str)
    assert parsed["prepared_recorded"] is None
    assert parsed["committed_recorded"] is None

    res_bool = ReconciliationTransactionResult(
        audit_id=str(uuid.uuid4()),
        environment_uuid=str(uuid.uuid4()),
        status=ReconciliationTransactionStatus.SUCCESS,
        stage_reached="COMMITTED",
        prepared_recorded=True,
        committed_recorded=True,
    )
    d_bool = res_bool.to_dict()
    assert d_bool["prepared_recorded"] is True
    assert d_bool["committed_recorded"] is True


def test_b10_committed_uncertainty_yields_committed_recorded_none(tmp_path: Path):
    """B10 — Falha na escrita de COMMITTED define committed_recorded is None e preserva prepared_recorded is True."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-B10", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_b10.jsonl"
    _write_jsonl(fb_file, [ev])

    real_record = record_reconciliation_audit
    def mock_record(path, entry):
        if entry.get("reconciliation_phase") == "COMMITTED":
            raise OSError("Falha simulada durante gravação de COMMITTED")
        return real_record(path, entry)

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=mock_record):
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
            operator_reason="Justificativa operacional auditavel B10",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "COMMITTED"
    assert res.prepared_recorded is True
    assert res.committed_recorded is None
    assert res.manifest_updated is True


# ======================================================================
# GRUPO C — REGRESSÃO E INVARIANTES DE GOVERNANÇA (C01 a C10)
# ======================================================================

def test_c01_prep_uncertain_prevents_subsequent_phases(tmp_path: Path):
    """C01 — P1-PREP-UNCERTAIN-01: Incerteza no PREPARED veda avanço para SQLite, JSONL, manifesto e COMMITTED."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-C01", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_c01.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Erro simulado")):
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
            operator_reason="Justificativa operacional auditavel C01",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 0
    assert res.manifest_updated is False
    assert res.committed_recorded is False


def test_c02_noop_does_not_increment_manifest_counter(tmp_path: Path):
    """C02 — P2-NOOP-COUNTER-02: Execução de no-op não incrementa contador no manifesto 1.2."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-C02", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_c02.jsonl"
    _write_jsonl(fb_file, [ev])

    manifest_before = json.loads(env["manifest_path"].read_text(encoding="utf-8"))
    counter_before = manifest_before.get("reconciliation_history_counter", 0)

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
        operator_reason="Justificativa operacional auditavel C02",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    manifest_after = json.loads(env["manifest_path"].read_text(encoding="utf-8"))
    assert manifest_after.get("reconciliation_history_counter", 0) == counter_before


def test_c03_running_active_concurrency_blocks_maintenance(tmp_path: Path):
    """C03 — P3-ALLOWED-STATES-WIDENING-03: Processo concorrente com trava física ativa em RUNNING bloqueia."""
    env = _setup_test_env(tmp_path)
    daemon_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="active_bot")
    daemon_lock.acquire()

    try:
        ev = {"event_id": "EVT-C03", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
        fb_file = env["fallback_dir"] / "eventos_c03.jsonl"
        _write_jsonl(fb_file, [ev])

        maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maint_c03")
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
            operator_reason="Justificativa operacional auditavel C03",
            operational_lock=maint_lock,
        )

        assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
        assert res.stage_reached == "LOCK_ACQUISITION"
    finally:
        daemon_lock.release()


def test_c04_no_committed_without_matching_prepared_in_ledger(tmp_path: Path):
    """C04 — Toda transação concluída possui par exato PREPARED -> COMMITTED com mesmo audit_id."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-C04", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_c04.jsonl"
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
        operator_reason="Justificativa operacional auditavel C04",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["is_valid"] is True
    assert ledger_val["completed_transactions_count"] == 1
    assert ledger_val["has_pending_transaction"] is False


def test_c05_apply_order_strictly_sqlite_before_jsonl(tmp_path: Path):
    """C05 — A ordem sequencial obrigatória é sempre 1. SQLite e 2. JSONL."""
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-C05", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_c05.jsonl"
    _write_jsonl(fb_file, [ev])

    call_order = []
    real_insert = insert_event_to_sqlite
    real_append = append_event_to_jsonl

    def wrapped_insert(p, e):
        call_order.append("SQLITE")
        return real_insert(p, e)

    def wrapped_append(p, e, prefix_separator=""):
        call_order.append("JSONL")
        return real_append(p, e, prefix_separator=prefix_separator)

    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=wrapped_insert), \
         mock.patch("events.fallback_reconciler.append_event_to_jsonl", side_effect=wrapped_append):
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
            operator_reason="Justificativa operacional auditavel C05",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert call_order == ["SQLITE", "JSONL"]


def test_c06_startup_consistency_blocks_on_orphaned_prepared(tmp_path: Path):
    """C06 — Startup e preflight bloqueiam quando houver PREPARED pendente no ledger."""
    env = _setup_test_env(tmp_path)
    orphan_aid = str(uuid.uuid4())
    record_reconciliation_audit(env["ledger_path"], {
        "audit_id": orphan_aid,
        "environment_uuid": env["env_uuid"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reconciliation_phase": "PREPARED",
    })

    ev = {"event_id": "EVT-C06", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_c06.jsonl"
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
        operator_reason="Justificativa operacional auditavel C06",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"
    assert any("PREPARED" in err for err in res.errors)


def test_c07_recovery_mode_is_diagnostic_fail_closed(tmp_path: Path):
    """C07 — Validação do ledger não executa reparo cego nem truncamento automático."""
    env = _setup_test_env(tmp_path)
    with open(env["ledger_path"], "wb") as f:
        f.write(b"{\"audit_id\": \"incomplete_line\"")

    size_before = env["ledger_path"].stat().st_size
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["is_valid"] is False
    assert env["ledger_path"].stat().st_size == size_before


def test_c08_cli_apply_remains_strictly_blocked_by_governance(tmp_path: Path):
    """C08 — CLI de reconciliação permanece 100% bloqueada por APPLY_NOT_ENABLED_C2A."""
    lock_file = tmp_path / ".runtime.lock"
    lock = OperationalRuntimeLock(lock_file)
    lock.acquire()
    lock.release()

    cmd = [
        sys.executable,
        "scripts/maintenance/reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Auditoria de governanca C08",
        "--lock-path", str(lock_file),
        "--json-output",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "APPLY_NOT_ENABLED_C2A" in (proc.stdout + proc.stderr)


def test_c09_persisted_state_is_never_converted_to_clean_after_reconciliation(tmp_path: Path):
    """C09 — Estados DIRTY ou UNKNOWN persistidos nunca são convertidos em CLEAN pelo orquestrador."""
    meta = {"status": "DIRTY", "holder": "crashed_worker", "reason": "dirty_shutdown"}
    env = _setup_test_env(tmp_path, lock_raw_bytes=b"\x00" + json.dumps(meta).encode("utf-8"))

    ev = {"event_id": "EVT-C09", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_c09.jsonl"
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
        operator_reason="Justificativa operacional auditavel C09",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    persisted_after = env["op_lock"].read_persisted_state(env["lock_path"])
    assert persisted_after["raw_status"] == "DIRTY"


def test_c10_authorization_p0_requires_physical_maintenance_session(tmp_path: Path):
    """C10 — Lock operacional convencional (RUNNING) sem sessão de manutenção P0 é rejeitado."""
    env = _setup_test_env(tmp_path)
    conventional_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="conventional_runner")
    conventional_lock.acquire()

    try:
        ev = {"event_id": "EVT-C10", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
        fb_file = env["fallback_dir"] / "eventos_c10.jsonl"
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
            operator_reason="Justificativa operacional auditavel C10",
            operational_lock=conventional_lock,
        )

        assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
        assert res.stage_reached == "AUTHORIZATION"
        assert any("não autoriza manutenção" in err for err in res.errors)
    finally:
        conventional_lock.release()
