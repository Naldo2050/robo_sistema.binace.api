"""
Suíte de Testes do Orquestrador Transacional Hermético de Reconciliação Financeira.
FASE 5.11 — G4-D B3-B2-C2-B2.2

Cobre os 35 cenários de teste obrigatórios T01 a T35:
- T01 — Transação legítima com evento ausente em ambos os destinos.
- T02 — Evento presente em SQLite e ausente em JSONL.
- T03 — Evento presente em JSONL e ausente em SQLite.
- T04 — Evento já presente em ambos.
- T05 — Lote vazio rejeitado.
- T06 — Evento inválido rejeitado.
- T07 — Evento com ID ambíguo rejeitado.
- T08 — Colisão de ID rejeitada.
- T09 — Manifesto inválido bloqueia.
- T10 — Ledger inconsistente bloqueia.
- T11 — PREPARED pendente bloqueia.
- T12 — Sessão sem autorização bloqueia.
- T13 — Lock operacional RUNNING bloqueia.
- T14 — Ambiente divergente bloqueia.
- T15 — Snapshot inválido bloqueia.
- T16 — Falha de snapshot aborta antes de PREPARED.
- T17 — Falha de gravação PREPARED impede APPLY.
- T18 — Falha SQLite após PREPARED preserva evidências.
- T19 — Falha JSONL após SQLite preserva PREPARED.
- T20 — Falha de pós-verificação impede manifesto.
- T21 — Falha de pós-verificação impede COMMITTED.
- T22 — Falha do manifesto impede COMMITTED.
- T23 — Falha COMMITTED retorna resultado incerto (PERSISTENCE_UNCERTAIN).
- T24 — Sessão de manutenção é liberada após sucesso.
- T25 — Sessão de manutenção é liberada após falha.
- T26 — Estado persistido original é preservado.
- T27 — Nenhum fallback é removido ou movido.
- T28 — Nenhum Archive é alterado.
- T29 — CLI apply permanece bloqueada (APPLY_NOT_ENABLED_C2A).
- T30 — Nenhum dado operacional real é acessado.
- T31 — Campos H1/H2 preservados.
- T32 — UUID do ambiente preservado.
- T33 — Transação COMMITTED possui PREPARED correspondente.
- T34 — Nenhuma duplicação financeira ocorre.
- T35 — Verificação final utiliza destinos completos.
"""

from __future__ import annotations

import hashlib
import json
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
    MaintenanceAuthorizationToken,
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    read_runtime_manifest,
    write_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    ReconciliationTransactionResult,
    ReconciliationTransactionStatus,
    TargetVerificationCode,
    TargetVerificationResult,
    compute_canonical_payload_hash,
    execute_reconciliation_transaction_hermetic,
    record_reconciliation_audit,
    validate_reconciliation_audit_ledger,
)


def _init_sqlite_db(db_path: Path, rows: List[Dict[str, Any]]) -> None:
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    for r in rows:
        p = r.get("payload", {})
        payload_json = json.dumps(p)
        cur.execute(
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


def _setup_test_env(tmp_path: Path, manifest_version: str = "1.2") -> Dict[str, Any]:
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
        holder_name="test_setup",
        manifest_version=manifest_version,
    )

    op_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="test_runner")
    op_lock.acquire()
    op_lock.release()

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
# T01 — Transação legítima com evento ausente em ambos os destinos
# ----------------------------------------------------------------------
def test_t01_legitimate_transaction_both_missing(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T01", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 100}
    fb_file = env["fallback_dir"] / "eventos_20260301.jsonl"
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
        operator_reason="Justificativa operacional auditavel T01",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert res.stage_reached == "COMMITTED"
    assert res.total_events == 1
    assert res.sqlite_inserted == 1
    assert res.jsonl_appended == 1
    assert res.already_present == 0
    assert res.prepared_recorded is True
    assert res.committed_recorded is True
    assert res.manifest_updated is True
    assert res.verification_result.approved is True


# ----------------------------------------------------------------------
# T02 — Evento presente em SQLite e ausente em JSONL
# ----------------------------------------------------------------------
def test_t02_event_present_in_sqlite_missing_in_jsonl(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T02", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 200}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    fb_file = env["fallback_dir"] / "eventos_20260302.jsonl"
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
        operator_reason="Justificativa operacional auditavel T02",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 1
    assert res.verification_result.approved is True


# ----------------------------------------------------------------------
# T03 — Evento presente em JSONL e ausente em SQLite
# ----------------------------------------------------------------------
def test_t03_event_present_in_jsonl_missing_in_sqlite(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T03", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 300}
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_20260303.jsonl"
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
        operator_reason="Justificativa operacional auditavel T03",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert res.sqlite_inserted == 1
    assert res.jsonl_appended == 0
    assert res.verification_result.approved is True


# ----------------------------------------------------------------------
# T04 — Evento já presente em ambos os destinos
# ----------------------------------------------------------------------
def test_t04_event_already_present_in_both(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T04", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 400}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(env["jsonl_path"], [ev])
    fb_file = env["fallback_dir"] / "eventos_20260304.jsonl"
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
        operator_reason="Justificativa operacional auditavel T04",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.NO_CHANGES
    assert res.stage_reached == "NOOP_VERIFIED"
    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 0
    assert res.already_present == 1
    assert res.snapshot_dir is None
    assert res.prepared_recorded is False
    assert res.manifest_updated is False
    assert res.committed_recorded is False
    assert res.verification_result.approved is True


# ----------------------------------------------------------------------
# T05 — Lote vazio rejeitado
# ----------------------------------------------------------------------
def test_t05_empty_batch_rejected(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    fb_file = env["fallback_dir"] / "eventos_vazio.jsonl"
    _write_jsonl(fb_file, [])

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
        operator_reason="Justificativa operacional auditavel T05",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False
    assert any("vazio" in err for err in res.errors)


# ----------------------------------------------------------------------
# T06 — Evento inválido rejeitado
# ----------------------------------------------------------------------
def test_t06_invalid_event_rejected(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    fb_file = env["fallback_dir"] / "eventos_invalido.jsonl"
    fb_file.write_text("INVALID_JSON_CORROMPIDO\n", encoding="utf-8")

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
        operator_reason="Justificativa operacional auditavel T06",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False


# ----------------------------------------------------------------------
# T07 — Evento com ID ambíguo / vazio rejeitado
# ----------------------------------------------------------------------
def test_t07_event_missing_id_rejected(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 1}  # sem event_id
    fb_file = env["fallback_dir"] / "eventos_sem_id.jsonl"
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
        operator_reason="Justificativa operacional auditavel T07",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False


# ----------------------------------------------------------------------
# T08 — Colisão de ID rejeitada
# ----------------------------------------------------------------------
def test_t08_id_collision_rejected(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev_persisted = {"event_id": "EVT-T08", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 1}
    _init_sqlite_db(env["sqlite_path"], [{
        "timestamp_ms": ev_persisted["epoch_ms"],
        "event_type": ev_persisted["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev_persisted,
    }])
    ev_divergent = {"event_id": "EVT-T08", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 999}
    fb_file = env["fallback_dir"] / "eventos_colisao.jsonl"
    _write_jsonl(fb_file, [ev_divergent])

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
        operator_reason="Justificativa operacional auditavel T08",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.prepared_recorded is False
    assert any("EVENTO_ID_COLISAO" in err for err in res.errors)


# ----------------------------------------------------------------------
# T09 — Manifesto inválido bloqueia
# ----------------------------------------------------------------------
def test_t09_invalid_manifest_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    env["manifest_path"].write_text("CORROMPIDO_MANIFESTO\n", encoding="utf-8")
    ev = {"event_id": "EVT-T09", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t09.jsonl"
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
        operator_reason="Justificativa operacional auditavel T09",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"


# ----------------------------------------------------------------------
# T10 — Ledger inconsistente bloqueia
# ----------------------------------------------------------------------
def test_t10_inconsistent_ledger_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    # Linha sem newline simula truncamento
    env["ledger_path"].write_bytes(b'{"action":"PREPARED","audit_id":"bad"}')
    ev = {"event_id": "EVT-T10", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t10.jsonl"
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
        operator_reason="Justificativa operacional auditavel T10",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "PREFLIGHT"


# ----------------------------------------------------------------------
# T11 — PREPARED pendente bloqueia
# ----------------------------------------------------------------------
def test_t11_pending_prepared_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    # Grava um PREPARED sem COMMITTED
    record_reconciliation_audit(env["ledger_path"], {
        "audit_id": str(uuid.uuid4()),
        "environment_uuid": env["env_uuid"],
        "timestamp_utc": "2026-10-15T20:00:00+00:00",
        "reconciliation_phase": "PREPARED",
        "action": "PREPARED",
        "events_count": 1,
    })
    ev = {"event_id": "EVT-T11", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t11.jsonl"
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
        operator_reason="Justificativa operacional auditavel T11",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert any("PREPARED pendente" in err or "PREPARED sem COMMITTED" in err for err in res.errors)


# ----------------------------------------------------------------------
# T12 — Sessão sem autorização bloqueia
# ----------------------------------------------------------------------
def test_t12_no_authorization_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T12", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t12.jsonl"
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
        operator_reason="Justificativa operacional auditavel T12",
        lock_authorization=None,
        operational_lock=None,
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "AUTHORIZATION"


# ----------------------------------------------------------------------
# T13 — Lock operacional RUNNING bloqueia
# ----------------------------------------------------------------------
def test_t13_running_lock_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    # Adquire lock como operacional RUNNING convencional (não manutenção)
    env["op_lock"].acquire()
    assert env["op_lock"].is_locked is True

    ev = {"event_id": "EVT-T13", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t13.jsonl"
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
        operator_reason="Justificativa operacional auditavel T13",
        operational_lock=env["op_lock"],
    )

    env["op_lock"].release()
    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "AUTHORIZATION"
    assert any("RUNNING convencional" in err for err in res.errors)


# ----------------------------------------------------------------------
# T14 — Ambiente divergente bloqueia
# ----------------------------------------------------------------------
def test_t14_divergent_environment_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T14", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t14.jsonl"
    _write_jsonl(fb_file, [ev])

    other_uuid = str(uuid.uuid4())
    res = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=other_uuid,  # Divergente do manifesto
        operator_reason="Justificativa operacional auditavel T14",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert any("Divergência de UUID" in err for err in res.errors)


# ----------------------------------------------------------------------
# T15 — Snapshot inválido / symlink bloqueia
# ----------------------------------------------------------------------
def test_t15_symlink_in_target_blocks(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T15", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t15.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch.object(Path, "is_symlink", return_value=True):
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
            operator_reason="Justificativa operacional auditavel T15",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "SECURITY_VALIDATION"


# ----------------------------------------------------------------------
# T16 — Falha de snapshot aborta antes de PREPARED
# ----------------------------------------------------------------------
def test_t16_snapshot_failure_aborts_before_prepared(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T16", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t16.jsonl"
    _write_jsonl(fb_file, [ev])

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
            operator_reason="Justificativa operacional auditavel T16",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert res.stage_reached == "SNAPSHOT"
    assert res.prepared_recorded is False
    assert not env["ledger_path"].exists()


# ----------------------------------------------------------------------
# T17 — Falha de gravação PREPARED impede APPLY
# ----------------------------------------------------------------------
def test_t17_prepared_write_failure_prevents_apply(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T17", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t17.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha I/O PREPARED")):
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
            operator_reason="Justificativa operacional auditavel T17",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.stage_reached == "PREPARED"
    assert any("PREPARED_WRITE_UNCERTAIN" in err for err in res.errors)
    assert res.sqlite_inserted == 0
    assert res.jsonl_appended == 0


# ----------------------------------------------------------------------
# T18 — Falha SQLite após PREPARED preserva evidências
# ----------------------------------------------------------------------
def test_t18_sqlite_failure_preserves_evidence(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T18", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t18.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Simulando disk I/O")):
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
            operator_reason="Justificativa operacional auditavel T18",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.stage_reached == "APPLY_SQLITE"
    assert res.prepared_recorded is True
    assert res.snapshot_dir is not None
    assert env["ledger_path"].exists()


# ----------------------------------------------------------------------
# T19 — Falha JSONL após SQLite preserva PREPARED
# ----------------------------------------------------------------------
def test_t19_jsonl_failure_preserves_prepared(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T19", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t19.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("events.fallback_reconciler.append_event_to_jsonl", side_effect=IOError("Simulando falha de escrita JSONL")):
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
            operator_reason="Justificativa operacional auditavel T19",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.stage_reached == "APPLY_JSONL"
    assert res.prepared_recorded is True
    assert res.sqlite_inserted == 1
    assert res.jsonl_appended == 0


# ----------------------------------------------------------------------
# T20 — Falha de pós-verificação impede manifesto
# ----------------------------------------------------------------------
def test_t20_post_verification_failure_prevents_manifest(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T20", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t20.jsonl"
    _write_jsonl(fb_file, [ev])

    unapproved_res = TargetVerificationResult(
        approved=False,
        code=TargetVerificationCode.VERIFICATION_FAILED,
        total_expected=1,
        verified_count=0,
        missing_count=1,
        duplicate_count=0,
        divergent_hash_count=0,
        conflict_count=0,
    )

    with mock.patch("events.fallback_reconciler.verify_reconciliation_targets", return_value=unapproved_res):
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
            operator_reason="Justificativa operacional auditavel T20",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.stage_reached == "POST_VERIFICATION"
    assert res.manifest_updated is False


# ----------------------------------------------------------------------
# T21 — Falha de pós-verificação impede COMMITTED
# ----------------------------------------------------------------------
def test_t21_post_verification_failure_prevents_committed(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T21", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t21.jsonl"
    _write_jsonl(fb_file, [ev])

    unapproved_res = TargetVerificationResult(
        approved=False,
        code=TargetVerificationCode.VERIFICATION_FAILED,
        total_expected=1,
        verified_count=0,
        missing_count=1,
        duplicate_count=0,
        divergent_hash_count=0,
        conflict_count=0,
    )

    with mock.patch("events.fallback_reconciler.verify_reconciliation_targets", return_value=unapproved_res):
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
            operator_reason="Justificativa operacional auditavel T21",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.committed_recorded is False


# ----------------------------------------------------------------------
# T22 — Falha do manifesto impede COMMITTED
# ----------------------------------------------------------------------
def test_t22_manifest_failure_prevents_committed(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T22", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t22.jsonl"
    _write_jsonl(fb_file, [ev])

    with mock.patch("common.runtime_lock.update_runtime_manifest_reconciliation_state", side_effect=RuntimeLockError("Simulando falha de manifesto")):
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
            operator_reason="Justificativa operacional auditavel T22",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PREPARED_INCOMPLETE
    assert res.stage_reached == "MANIFEST_UPDATE"
    assert res.manifest_updated is False
    assert res.committed_recorded is False


# ----------------------------------------------------------------------
# T23 — Falha COMMITTED retorna resultado incerto (PERSISTENCE_UNCERTAIN)
# ----------------------------------------------------------------------
def test_t23_committed_failure_returns_uncertain(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T23", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t23.jsonl"
    _write_jsonl(fb_file, [ev])

    original_record = record_reconciliation_audit
    calls = []

    def mock_record(audit_file, entry):
        calls.append(entry.get("action"))
        if entry.get("action") == "COMMITTED":
            raise IOError("Simulando falha ao gravar COMMITTED")
        original_record(audit_file, entry)

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
            operator_reason="Justificativa operacional auditavel T23",
            operational_lock=env["op_lock"],
        )

    assert res.status == ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN
    assert res.manifest_updated is True
    assert res.prepared_recorded is True
    assert res.committed_recorded is None
    assert "COMMITTED" in calls


# ----------------------------------------------------------------------
# T24 — Sessão de manutenção é liberada após sucesso
# ----------------------------------------------------------------------
def test_t24_maintenance_session_released_on_success(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T24", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t24.jsonl"
    _write_jsonl(fb_file, [ev])

    # Cria lock não adquirido
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="t24_holder")
    assert lock.is_locked is False

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
        operator_reason="Justificativa operacional auditavel T24",
        operational_lock=lock,
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert lock.is_locked is False
    assert lock._file_obj is None


# ----------------------------------------------------------------------
# T25 — Sessão de manutenção é liberada após falha
# ----------------------------------------------------------------------
def test_t25_maintenance_session_released_on_failure(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    fb_file = env["fallback_dir"] / "eventos_vazio.jsonl"
    _write_jsonl(fb_file, [])  # Vazio causa falha

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="t25_holder")
    assert lock.is_locked is False

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
        operator_reason="Justificativa operacional auditavel T25",
        operational_lock=lock,
    )

    assert res.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert lock.is_locked is False
    assert lock._file_obj is None


# ----------------------------------------------------------------------
# T26 — Estado persistido original é preservado
# ----------------------------------------------------------------------
def test_t26_persisted_lock_state_preserved(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    # Garante que o lock está persistido como CLEAN
    lock_meta = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert lock_meta["status"] == "CLEAN"

    ev = {"event_id": "EVT-T26", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t26.jsonl"
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
        operator_reason="Justificativa operacional auditavel T26",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    # O lock continua persistido como CLEAN, não virou RUNNING nem DIRTY
    post_meta = env["op_lock"].read_persisted_state(env["op_lock"].lock_path)
    assert post_meta["status"] == "CLEAN"


# ----------------------------------------------------------------------
# T27 — Nenhum fallback é removido ou movido
# ----------------------------------------------------------------------
def test_t27_fallback_file_not_removed_or_moved(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T27", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t27.jsonl"
    _write_jsonl(fb_file, [ev])
    sha_before = _calc_sha256(fb_file)

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
        operator_reason="Justificativa operacional auditavel T27",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert fb_file.exists()
    assert _calc_sha256(fb_file) == sha_before


# ----------------------------------------------------------------------
# T28 — Nenhum Archive é alterado
# ----------------------------------------------------------------------
def test_t28_archive_not_modified(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    arch_file = env["archive_dir"] / "eventos_fluxo_20260301_001.jsonl"
    _write_jsonl(arch_file, [{"event_id": "EVT-ARCH", "epoch_ms": 1699999999000, "tipo_evento": "SIG"}])
    sha_arch_before = _calc_sha256(arch_file)

    ev = {"event_id": "EVT-T28", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t28.jsonl"
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
        operator_reason="Justificativa operacional auditavel T28",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert _calc_sha256(arch_file) == sha_arch_before


# ----------------------------------------------------------------------
# T29 — CLI apply permanece bloqueada (APPLY_NOT_ENABLED_C2A)
# ----------------------------------------------------------------------
def test_t29_cli_apply_remains_blocked(tmp_path: Path):
    lock_file = tmp_path / ".runtime.lock"
    lock = OperationalRuntimeLock(lock_file)
    lock.acquire()
    lock.release()

    cmd = [
        sys.executable,
        "scripts/maintenance/reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason",
        "Validacao de seguranca da fase C2-B2.2",
        "--lock-path",
        str(lock_file),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode != 0
    assert "APPLY_NOT_ENABLED_C2A" in (proc.stdout + proc.stderr)


# ----------------------------------------------------------------------
# T30 — Nenhum dado operacional real é acessado
# ----------------------------------------------------------------------
def test_t30_no_real_operational_data_accessed(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T30", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t30.jsonl"
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
        operator_reason="Justificativa operacional auditavel T30",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    # Confirma que operou exclusivamente sob tmp_path
    assert str(tmp_path) in str(res.snapshot_dir)


# ----------------------------------------------------------------------
# T31 — Campos H1/H2 preservados
# ----------------------------------------------------------------------
def test_t31_h1_h2_fields_preserved(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    # Configura manifesto com histórico H1 pré-existente
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["recovery_history_counter"] = 3
    m_data["last_recovery_audit_id"] = str(uuid.uuid4())
    m_data["last_recovery_utc"] = "2026-10-15T10:00:00+00:00"
    m_data["manifest_version"] = "1.2"
    m_data["reconciliation_history_counter"] = 0
    with open(env["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(m_data, f)

    ev = {"event_id": "EVT-T31", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t31.jsonl"
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
        operator_reason="Justificativa operacional auditavel T31",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    post_m = read_runtime_manifest(env["manifest_path"])
    assert post_m["recovery_history_counter"] == 3
    assert post_m["last_recovery_audit_id"] == m_data["last_recovery_audit_id"]
    assert post_m["reconciliation_history_counter"] == 1
    assert post_m["last_reconciliation_audit_id"] == res.audit_id


# ----------------------------------------------------------------------
# T32 — UUID do ambiente preservado
# ----------------------------------------------------------------------
def test_t32_environment_uuid_preserved(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T32", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t32.jsonl"
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
        operator_reason="Justificativa operacional auditavel T32",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    post_m = read_runtime_manifest(env["manifest_path"])
    assert post_m["environment_uuid"] == env["env_uuid"]


# ----------------------------------------------------------------------
# T33 — Transação COMMITTED possui PREPARED correspondente
# ----------------------------------------------------------------------
def test_t33_committed_has_matching_prepared(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T33", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t33.jsonl"
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
        operator_reason="Justificativa operacional auditavel T33",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["is_valid"] is True
    assert ledger_val["completed_transactions_count"] == 1
    assert ledger_val["has_pending_transaction"] is False


# ----------------------------------------------------------------------
# T34 — Nenhuma duplicação financeira ocorre em múltiplas transações
# ----------------------------------------------------------------------
def test_t34_no_financial_duplication_on_repeated_apply(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T34", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t34.jsonl"
    _write_jsonl(fb_file, [ev])

    # 1ª Execução: insere
    res1 = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel T34-1",
        operational_lock=env["op_lock"],
    )
    assert res1.status == ReconciliationTransactionStatus.SUCCESS
    assert res1.sqlite_inserted == 1
    assert res1.jsonl_appended == 1

    # 2ª Execução do mesmo lote: detecta already_present sem duplicar
    res2 = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=fb_file,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_base_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Justificativa operacional auditavel T34-2",
        operational_lock=env["op_lock"],
    )
    assert res2.status == ReconciliationTransactionStatus.NO_CHANGES
    assert res2.stage_reached == "NOOP_VERIFIED"
    assert res2.sqlite_inserted == 0
    assert res2.jsonl_appended == 0
    assert res2.already_present == 1


# ----------------------------------------------------------------------
# T35 — Verificação final utiliza destinos completos
# ----------------------------------------------------------------------
def test_t35_verification_uses_complete_destinations(tmp_path: Path):
    env = _setup_test_env(tmp_path)
    ev = {"event_id": "EVT-T35", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    fb_file = env["fallback_dir"] / "eventos_t35.jsonl"
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
        operator_reason="Justificativa operacional auditavel T35",
        operational_lock=env["op_lock"],
    )

    assert res.status == ReconciliationTransactionStatus.SUCCESS
    assert set(res.verification_result.targets_checked) == {"sqlite", "jsonl"}
