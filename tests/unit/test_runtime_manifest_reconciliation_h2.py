# -*- coding: utf-8 -*-
"""
Suíte de Testes do Gate G4-D B3-B2-C2-B1 (H2):
Manifesto Operacional 1.2 e Histórico Persistente de Reconciliação.

Cobre os 40 cenários obrigatórios de schema, tipagem estrita, consistência bidirecional,
preservação mútua H1/H2, atomicidade com fsync/os.replace, cenários de crash,
compatibilidade com legados 1.0 e 1.1, e integração fail-closed no startup e recovery.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from common.runtime_lock import (
    MANIFEST_FILENAME,
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    read_runtime_manifest,
    validate_manifest_structure,
    verify_startup_preflight,
    write_runtime_manifest,
    verify_manifest_recovery_consistency,
    verify_manifest_reconciliation_consistency,
    update_runtime_manifest_recovery_state,
    update_runtime_manifest_reconciliation_state,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    record_reconciliation_audit,
    validate_reconciliation_audit_ledger,
)
from scripts.maintenance.bootstrap_runtime_lock import bootstrap_runtime_environment
from scripts.maintenance.recover_runtime_state import diagnose_runtime_recovery
from scripts.maintenance.reconcile_fallback_events import main as cli_main


def _setup_test_env(
    tmp_path: Path,
    state: RuntimeState = RuntimeState.CLEAN,
    manifest_version: str = "1.2",
    reconciliation_counter: int = 0,
    recovery_counter: int = 0,
) -> Dict[str, Any]:
    """Configura ambiente temporário padronizado com manifesto e dados operacionais válidos."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir = tmp_path / "fallback_events"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    lock_path = data_dir / ".operational_runtime.lock"
    manifest_path = data_dir / MANIFEST_FILENAME
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    rec_audit_path = data_dir / RECONCILIATION_AUDIT_FILENAME
    recov_audit_path = data_dir / ".recovery_audit.jsonl"

    env_uuid = str(uuid.uuid4())

    manifest_payload: Dict[str, Any] = {
        "manifest_version": manifest_version,
        "environment_uuid": env_uuid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "holder": "test_setup",
        "data_dir_name": data_dir.name,
    }

    if manifest_version in ("1.1", "1.2"):
        manifest_payload["recovery_history_counter"] = recovery_counter
        manifest_payload["last_recovery_audit_id"] = None if recovery_counter == 0 else str(uuid.uuid4())
        manifest_payload["last_recovery_utc"] = None if recovery_counter == 0 else datetime.now(timezone.utc).isoformat()

    if manifest_version == "1.2":
        manifest_payload["reconciliation_history_counter"] = reconciliation_counter
        manifest_payload["last_reconciliation_audit_id"] = None if reconciliation_counter == 0 else str(uuid.uuid4())
        manifest_payload["last_reconciliation_utc"] = None if reconciliation_counter == 0 else datetime.now(timezone.utc).isoformat()

    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    # Inicia lock operacional
    lock_data = {
        "status": state.value,
        "holder": "test_setup",
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "reason": "setup",
        "environment_uuid": env_uuid,
    }
    lock_path.write_bytes(b"#" + json.dumps(lock_data, indent=2).encode("utf-8"))

    # SQLite consistente
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            window_id TEXT,
            is_signal BOOLEAN DEFAULT 0,
            payload JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    # JSONL consistente
    sample_evt = {
        "event_id": "ev_001",
        "timestamp_ms": 1710000000000,
        "event_type": "trade",
        "symbol": "BTCUSDT",
    }
    jsonl_path.write_text(json.dumps(sample_evt) + "\n", encoding="utf-8")

    return {
        "data_dir": data_dir,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "fallback_dir": fallback_dir,
        "rec_audit_path": rec_audit_path,
        "recov_audit_path": recov_audit_path,
        "env_uuid": env_uuid,
        "manifest_payload": manifest_payload,
    }


def _record_recovery_audit(audit_file: Path, audit_id: str, ts: str, env_uuid: str) -> None:
    prep = {
        "audit_id": audit_id,
        "timestamp_utc": ts,
        "environment_uuid": env_uuid,
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "test_dir",
        "snapshot_manifest_sha256": "test_sha256",
        "previous_runtime_state": "DIRTY",
    }
    comm = {
        "audit_id": audit_id,
        "timestamp_utc": ts,
        "environment_uuid": env_uuid,
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "test_dir",
        "snapshot_manifest_sha256": "test_sha256",
        "final_state": "CLEAN",
    }
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prep) + "\n")
        f.write(json.dumps(comm) + "\n")


def _acquire_maint_lock(env: Dict[str, Any]) -> OperationalRuntimeLock:
    lock = OperationalRuntimeLock(
        lock_path=env["lock_path"],
        holder_name="test_maint",
    )
    lock.acquire_maintenance()
    return lock


# ==============================================================================
# 1. TESTES DE SCHEMA E VALIDAÇÃO ESTRUTURAL DO MANIFESTO 1.2
# ==============================================================================

def test_01_schema_12_valido():
    """1. validate_manifest_structure aprova manifesto 1.2 com todos os campos válidos."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 1,
        "last_recovery_audit_id": str(uuid.uuid4()),
        "last_recovery_utc": datetime.now(timezone.utc).isoformat(),
        "reconciliation_history_counter": 1,
        "last_reconciliation_audit_id": str(uuid.uuid4()),
        "last_reconciliation_utc": datetime.now(timezone.utc).isoformat(),
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is True
    assert err is None


def test_02_contador_zero():
    """2. Contador zero com last_id e last_utc nulos é válido em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": 0,
        "last_reconciliation_audit_id": None,
        "last_reconciliation_utc": None,
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is True


def test_03_contador_positivo():
    """3. Contador > 0 sem audit_id obrigatório é rejeitado em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": 2,
        "last_reconciliation_audit_id": None, # Inválido: deveria ter ID
        "last_reconciliation_utc": datetime.now(timezone.utc).isoformat(),
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "last_reconciliation_audit_id" in err


def test_04_contador_booleano_rejeitado():
    """4. Rejeita booleano como contador em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": True, # Inválido: bool
        "last_reconciliation_audit_id": None,
        "last_reconciliation_utc": None,
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "reconciliation_history_counter" in err


def test_05_contador_negativo_rejeitado():
    """5. Rejeita inteiro negativo como contador em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": -1, # Inválido: negativo
        "last_reconciliation_audit_id": None,
        "last_reconciliation_utc": None,
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "reconciliation_history_counter" in err


def test_06_contador_string_rejeitado():
    """6. Rejeita string numérica como contador em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": "0", # Inválido: string
        "last_reconciliation_audit_id": None,
        "last_reconciliation_utc": None,
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "reconciliation_history_counter" in err


def test_07_uuid_invalido_rejeitado():
    """7. Rejeita last_reconciliation_audit_id mal formatado em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": 1,
        "last_reconciliation_audit_id": "not-a-valid-uuid",
        "last_reconciliation_utc": datetime.now(timezone.utc).isoformat(),
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "não é um UUID válido" in err


def test_08_timestamp_invalido_rejeitado():
    """8. Rejeita last_reconciliation_utc mal formatado em 1.2."""
    data = {
        "manifest_version": "1.2",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
        "reconciliation_history_counter": 1,
        "last_reconciliation_audit_id": str(uuid.uuid4()),
        "last_reconciliation_utc": "2026-10-14 20:00:00", # Inválido: sem timezone
    }
    is_ok, err = validate_manifest_structure(data)
    assert is_ok is False
    assert "não é um timestamp ISO8601 válido" in err


# ==============================================================================
# 2. TESTES DE CONSISTÊNCIA MANIFESTO vs LIVRO-RAZÃO (H2)
# ==============================================================================

def test_09_ledger_ausente_com_contador_positivo_bloqueia(tmp_path: Path):
    """9. Manifesto 1.2 com contador > 0 sem ledger retorna MISSING_RECONCILIATION_LEDGER e is_valid=False."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=1)
    # rec_audit_path não existe
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "MISSING_RECONCILIATION_LEDGER"
    assert "anti-tamper fail-closed" in res["error"]


def test_10_ledger_corrompido_bloqueia(tmp_path: Path):
    """10. Ledger com JSON corrompido retorna CORRUPTED_LEDGER e is_valid=False."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=1)
    env["rec_audit_path"].write_text('{"audit_id": "123", "reconciliation_phase": "PREP\n', encoding="utf-8")
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "CORRUPTED_LEDGER"


def test_11_prepared_pendente_bloqueia(tmp_path: Path):
    """11. PREPARED sem COMMITTED retorna PENDING_TRANSACTION e is_valid=False."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "PENDING_TRANSACTION"
    assert res["ledger_has_pending"] is True


def test_12_committed_orfao_bloqueia(tmp_path: Path):
    """12. COMMITTED sem PREPARED prévio é rejeitado como inválido."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=1)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "CORRUPTED_LEDGER"


def test_13_contador_divergente_bloqueia(tmp_path: Path):
    """13. Contador do manifesto diferente das transações COMMITTED retorna DIVERGENT_COUNTER."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=2)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    # Ledger tem 1 transação, mas manifesto diz 2
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "DIVERGENT_COUNTER"


def test_14_audit_id_divergente_bloqueia(tmp_path: Path):
    """14. last_reconciliation_audit_id do manifesto diferente do último COMMITTED retorna DIVERGENT_AUDIT_ID."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=1)
    aid_ledger = str(uuid.uuid4())
    aid_manifest = str(uuid.uuid4()) # ID diferente no manifesto
    ts = datetime.now(timezone.utc).isoformat()

    # Atualiza manifesto com aid_manifest
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["last_reconciliation_audit_id"] = aid_manifest
    env["manifest_path"].write_text(json.dumps(m_data), encoding="utf-8")

    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid_ledger,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid_ledger,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "DIVERGENT_AUDIT_ID"


def test_15_uuid_divergente_bloqueia(tmp_path: Path):
    """15. Registro no ledger com UUID de outro ambiente é rejeitado."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": str(uuid.uuid4()), # UUID estranho
    })
    res = verify_manifest_reconciliation_consistency(
        manifest_data=read_runtime_manifest(env["manifest_path"]),
        ledger_path=env["rec_audit_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert res["is_valid"] is False
    assert res["classification"] == "CORRUPTED_LEDGER"


# ==============================================================================
# 3. TESTES DE PRESERVAÇÃO MÚTUA E CICLOS ALTERNADOS (H1/H2)
# ==============================================================================

def test_16_recovery_apos_reconciliacao(tmp_path: Path):
    """16. Execução de recovery atualiza recovery mantendo intactos os campos de reconciliação."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=3, recovery_counter=1)
    rec_id = env["manifest_payload"]["last_reconciliation_audit_id"]
    rec_utc = env["manifest_payload"]["last_reconciliation_utc"]

    # Cria ledger de recovery consistente com 1 COMMITTED
    recov_id_1 = env["manifest_payload"]["last_recovery_audit_id"]
    _record_recovery_audit(
        env["recov_audit_path"],
        audit_id=recov_id_1,
        ts=env["manifest_payload"]["last_recovery_utc"],
        env_uuid=env["env_uuid"],
    )

    # Agora executa update_runtime_manifest_recovery_state para novo recovery (id=recov_id_2)
    recov_id_2 = str(uuid.uuid4())
    ts_recov = datetime.now(timezone.utc).isoformat()
    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=recov_id_2,
            audit_timestamp_utc=ts_recov,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    assert updated["manifest_version"] == "1.2"
    assert updated["recovery_history_counter"] == 2
    assert updated["last_recovery_audit_id"] == recov_id_2
    # Preservou reconciliação rigorosamente
    assert updated["reconciliation_history_counter"] == 3
    assert updated["last_reconciliation_audit_id"] == rec_id
    assert updated["last_reconciliation_utc"] == rec_utc


def test_17_reconciliacao_apos_recovery(tmp_path: Path):
    """17. Execução de reconciliação atualiza reconciliação mantendo intactos os campos de recovery."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0, recovery_counter=5)
    recov_id = env["manifest_payload"]["last_recovery_audit_id"]
    recov_utc = env["manifest_payload"]["last_recovery_utc"]

    new_rec_id = str(uuid.uuid4())
    new_rec_utc = datetime.now(timezone.utc).isoformat()

    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=new_rec_id,
            audit_timestamp_utc=new_rec_utc,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    assert updated["manifest_version"] == "1.2"
    assert updated["reconciliation_history_counter"] == 1
    assert updated["last_reconciliation_audit_id"] == new_rec_id
    assert updated["last_reconciliation_utc"] == new_rec_utc
    # Preservou recovery rigorosamente
    assert updated["recovery_history_counter"] == 5
    assert updated["last_recovery_audit_id"] == recov_id
    assert updated["last_recovery_utc"] == recov_utc


def test_18_ciclos_alternados(tmp_path: Path):
    """18. Ciclos alternados Apply -> Recovery -> Apply -> Recovery incrementam contadores sem colisão."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0, recovery_counter=0)

    maint_lock = _acquire_maint_lock(env)
    try:
        # 1. Reconciliação 1
        rec_id_1 = str(uuid.uuid4())
        ts1 = datetime.now(timezone.utc).isoformat()
        m1 = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=rec_id_1,
            audit_timestamp_utc=ts1,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
        assert m1["reconciliation_history_counter"] == 1
        assert m1["recovery_history_counter"] == 0

        # Grava COMMITTED no ledger de reconciliação
        record_reconciliation_audit(env["rec_audit_path"], {
            "audit_id": rec_id_1,
            "reconciliation_phase": "PREPARED",
            "timestamp_utc": ts1,
            "environment_uuid": env["env_uuid"],
        })
        record_reconciliation_audit(env["rec_audit_path"], {
            "audit_id": rec_id_1,
            "reconciliation_phase": "COMMITTED",
            "timestamp_utc": ts1,
            "environment_uuid": env["env_uuid"],
        })

        # 2. Recovery 1
        recov_id_1 = str(uuid.uuid4())
        ts2 = datetime.now(timezone.utc).isoformat()
        m2 = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=recov_id_1,
            audit_timestamp_utc=ts2,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=maint_lock,
        )
        assert m2["reconciliation_history_counter"] == 1
        assert m2["recovery_history_counter"] == 1

        # Grava COMMITTED no ledger de recovery
        _record_recovery_audit(
            env["recov_audit_path"],
            audit_id=recov_id_1,
            ts=ts2,
            env_uuid=env["env_uuid"],
        )

        # 3. Reconciliação 2
        rec_id_2 = str(uuid.uuid4())
        ts3 = datetime.now(timezone.utc).isoformat()
        m3 = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=rec_id_2,
            audit_timestamp_utc=ts3,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
        assert m3["reconciliation_history_counter"] == 2
        assert m3["recovery_history_counter"] == 1
    finally:
        maint_lock.release()


def test_19_preservacao_dos_campos_h1(tmp_path: Path):
    """19. update_runtime_manifest_reconciliation_state preserva campos H1 sem alteração."""
    env = _setup_test_env(tmp_path, manifest_version="1.1", recovery_counter=2)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    maint_lock = _acquire_maint_lock(env)
    try:
        m = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()
    assert m["recovery_history_counter"] == 2
    assert m["last_recovery_audit_id"] == env["manifest_payload"]["last_recovery_audit_id"]


def test_20_preservacao_dos_campos_h2(tmp_path: Path):
    """20. update_runtime_manifest_recovery_state preserva campos H2 sem alteração."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=4)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    maint_lock = _acquire_maint_lock(env)
    try:
        m = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()
    assert m["reconciliation_history_counter"] == 4
    assert m["last_reconciliation_audit_id"] == env["manifest_payload"]["last_reconciliation_audit_id"]


# ==============================================================================
# 4. TESTES DE ATOMICIDADE E FALHAS DE I/O
# ==============================================================================

def test_21_falha_de_fsync_aborta_atomicamente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """21. Falha de fsync durante gravação do manifesto aborta sem corromper o original."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    manifest_bytes_before = env["manifest_path"].read_bytes()

    real_fsync = os.fsync
    maint_lock = _acquire_maint_lock(env)

    def failing_fsync(fd):
        raise OSError(5, "Simulated hardware I/O fsync failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=str(uuid.uuid4()),
                audit_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=maint_lock,
            )
        assert "Falha de persistência atômica" in str(exc_info.value)
        assert env["manifest_path"].read_bytes() == manifest_bytes_before
        assert not (env["data_dir"] / f"{MANIFEST_FILENAME}.tmp").exists()
    finally:
        maint_lock.release()


def test_22_falha_de_os_replace_aborta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """22. Falha em os.replace não corrompe o manifesto original e remove .tmp."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    manifest_bytes_before = env["manifest_path"].read_bytes()

    maint_lock = _acquire_maint_lock(env)

    def failing_replace(src, dst):
        raise OSError(13, "Simulated access denied on atomic replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    try:
        with pytest.raises(RuntimeLockError):
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=str(uuid.uuid4()),
                audit_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=maint_lock,
            )
        assert env["manifest_path"].read_bytes() == manifest_bytes_before
    finally:
        maint_lock.release()


# ==============================================================================
# 5. TESTES DE CENÁRIOS DE CRASH (A A J)
# ==============================================================================

def test_23_crash_antes_do_manifesto(tmp_path: Path):
    """23. Crash antes da atualização do manifesto deixa PREPARED órfão no ledger e bloqueia startup."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_24_crash_durante_manifesto(tmp_path: Path):
    """24. Arquivo .tmp residual após crash não altera o preflight e é ignorado se manifesto original estiver válido."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    tmp_file = env["data_dir"] / f"{MANIFEST_FILENAME}.tmp"
    tmp_file.write_text("corrupted partial json {", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_25_crash_apos_manifesto_antes_committed(tmp_path: Path):
    """25. Manifesto incrementado para N+1 mas ledger ainda em PREPARED bloqueia por PENDING_TRANSACTION."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    # Grava PREPARED
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    # Atualiza manifesto para 1 (COMMITTED ainda não foi gravado por crash)
    maint_lock = _acquire_maint_lock(env)
    try:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_26_crash_antes_committed(tmp_path: Path):
    """26. Crash após apply nos destinos antes de gravar COMMITTED bloqueia recovery e startup fail-closed."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0, state=RuntimeState.DIRTY)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })
    diag = diagnose_runtime_recovery(data_dir=env["data_dir"], lock_path=env["lock_path"])
    conflicts_str = " ".join(diag["conflicts"])
    assert "PENDING_TRANSACTION" in conflicts_str or "PREPARED sem COMMITTED" in conflicts_str


def test_27_crash_durante_committed(tmp_path: Path):
    """27. Linha truncada durante COMMITTED bloqueia com CORRUPTED_LEDGER."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    # Linha truncada
    with open(env["rec_audit_path"], "a", encoding="utf-8") as f:
        f.write('{"audit_id": "' + aid + '", "reconciliation_phase": "COMMITTED"') # sem newline e sem fechar dict

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "inconsistente" in msg or "corrompido" in msg or "truncamento" in msg


# ==============================================================================
# 6. TESTES DE COMPATIBILIDADE LEGADA E MIGRAÇÃO CONTROLADA
# ==============================================================================

def test_28_compatibilidade_manifesto_10(tmp_path: Path):
    """28. Manifesto versão 1.0 legado é aprovado no preflight quando não há histórico de anomalias."""
    env = _setup_test_env(tmp_path, manifest_version="1.0")
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_29_compatibilidade_manifesto_11(tmp_path: Path):
    """29. Manifesto versão 1.1 legado é aprovado no preflight quando não há reconciliações pendentes."""
    env = _setup_test_env(tmp_path, manifest_version="1.1", recovery_counter=0)
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_30_migracao_sem_perda_de_historico(tmp_path: Path):
    """30. update_runtime_manifest_reconciliation_state migra manifesto 1.1 para 1.2 preservando recovery."""
    env = _setup_test_env(tmp_path, manifest_version="1.1", recovery_counter=3)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()
    assert updated["manifest_version"] == "1.2"
    assert updated["reconciliation_history_counter"] == 1
    assert updated["recovery_history_counter"] == 3


def test_31_nenhuma_migracao_no_startup(tmp_path: Path):
    """31. verify_startup_preflight opera estritamente em modo read-only e nunca altera o manifesto do disco."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    manifest_bytes_before = env["manifest_path"].read_bytes()
    verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert env["manifest_path"].read_bytes() == manifest_bytes_before


def test_32_nenhuma_migracao_no_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """32. Execução da CLI em dry-run nunca modifica o manifesto no disco."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    manifest_bytes_before = env["manifest_path"].read_bytes()

    test_args = [
        "reconcile_fallback_events.py",
        "--dry-run",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    assert env["manifest_path"].read_bytes() == manifest_bytes_before


def test_33_ledger_removido_apos_shutdown_bloqueia(tmp_path: Path):
    """33. Prova do fechamento da vulnerabilidade: exclusão física de .reconciliation_audit.jsonl bloqueia preflight fail-closed."""
    # 1. Configura ambiente com manifesto 1.2 e reconciliação concluída (counter=1)
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=1)
    aid = env["manifest_payload"]["last_reconciliation_audit_id"]
    ts = env["manifest_payload"]["last_reconciliation_utc"]

    # Cria ledger válido correspondente
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    # Verifica que com o ledger existente e consistente, o boot seria aprovado
    ok_before, msg_before = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok_before is True

    # 2. Simula apagamento do ledger (ataque ou remoção acidental pós-shutdown)
    env["rec_audit_path"].unlink()
    assert not env["rec_audit_path"].exists()

    # 3. O startup DEVE BLOQUEAR fail-closed graças ao manifesto 1.2
    ok_after, msg_after = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok_after is False
    assert "MISSING_RECONCILIATION_LEDGER" in msg_after or "está ausente" in msg_after


def test_34_lock_fisico_obrigatorio(tmp_path: Path):
    """34. Validação de manifesto exige exclusividade em nível de kernel."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    lock1 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="proc_1")
    lock1.acquire()

    lock2 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="proc_2")
    with pytest.raises(RuntimeLockError) as exc_info:
        lock2.acquire(timeout=0.0)
    assert "Conflito de exclusividade" in str(exc_info.value)
    lock1.release()


def test_35_concorrencia_multiprocesso(tmp_path: Path):
    """35. Tentativa de obter lock de manutenção enquanto processo operacional está ativo é barrada."""
    env = _setup_test_env(tmp_path, state=RuntimeState.RUNNING)
    competing = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="active_bot")
    competing.acquire()

    maint = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maint_bot")
    with pytest.raises(RuntimeLockError) as exc_info:
        with maint.maintenance_session(allowed_states=[RuntimeState.RUNNING]):
            pass
    assert "retido por processo ativo" in str(exc_info.value)
    competing.release()


def test_36_preservacao_de_uuid(tmp_path: Path):
    """36. Atualização de reconciliação preserva o environment_uuid original do manifesto."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    orig_uuid = env["env_uuid"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=orig_uuid,
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()
    assert updated["environment_uuid"] == orig_uuid


def test_37_preservacao_de_campos_extras(tmp_path: Path):
    """37. Campos extras/desconhecidos no manifesto JSON são preservados após updates."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", reconciliation_counter=0)
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["custom_future_flag"] = "safe_payload_value"
    env["manifest_path"].write_text(json.dumps(m_data), encoding="utf-8")

    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()
    assert updated.get("custom_future_flag") == "safe_payload_value"


def test_38_recovery_h1_sem_regressao(tmp_path: Path):
    """38. diagnose_runtime_recovery funciona perfeitamente perante manifesto 1.2 consistente."""
    env = _setup_test_env(tmp_path, manifest_version="1.2", state=RuntimeState.DIRTY)
    diag = diagnose_runtime_recovery(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert diag["lock"]["manifest_exists"] is True
    assert diag["audit"]["manifest_version"] == "1.2"


def test_39_cli_c2a_ainda_bloqueada(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """39. Na Fase C2-B1, apply permanece estritamente desabilitado por APPLY_NOT_ENABLED_C2A."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste com mais de 15 chars",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    assert "APPLY_NOT_ENABLED_C2A" in captured.err


def test_40_zero_mutacao_financeira(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """40. Zero mutação nos arquivos SQLite e JSONL operacionais."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    sq_size_before = env["sqlite_path"].stat().st_size
    jl_size_before = env["jsonl_path"].stat().st_size

    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste com mais de 15 chars",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()

    assert env["sqlite_path"].stat().st_size == sq_size_before
    assert env["jsonl_path"].stat().st_size == jl_size_before
