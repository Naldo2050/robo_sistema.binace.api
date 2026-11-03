# -*- coding: utf-8 -*-
"""
Suíte de Testes do Gate G4-D B3-B2-B3-H1:
Persistência e Rastreabilidade Perpétua do Histórico de Recovery Autorizado.

Cobre os 40 cenários obrigatórios de auditoria e crash consistency em tmp_path.
"""

from __future__ import annotations

import json
import os
import platform
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
    update_runtime_manifest_recovery_state,
    validate_manifest_structure,
    validate_recovery_audit_ledger,
    verify_manifest_recovery_consistency,
    verify_startup_preflight,
    write_runtime_manifest,
)
from scripts.maintenance.bootstrap_runtime_lock import bootstrap_runtime_environment
from scripts.maintenance.recover_runtime_state import (
    diagnose_runtime_recovery,
    execute_authorized_recovery,
)


def _setup_base_env(tmp_path: Path, version: str = "1.1", counter: int = 0) -> Dict[str, Path]:
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
    audit_path = data_dir / ".recovery_audit.jsonl"

    sample_event = {
        "event_id": "ev_001",
        "timestamp_ms": 1710000000000,
        "event_type": "trade",
        "symbol": "BTCUSDT",
        "window_id": "w1",
        "is_signal": 0,
        "price": "65000",
    }

    # SQLite com schema real
    import sqlite3
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
    conn.execute(
        """
        INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            sample_event["timestamp_ms"],
            sample_event["event_type"],
            sample_event["symbol"],
            sample_event["window_id"],
            sample_event["is_signal"],
            json.dumps(sample_event),
        ),
    )
    conn.commit()
    conn.close()

    # JSONL consistente
    jsonl_path.write_text(json.dumps(sample_event) + "\n", encoding="utf-8")

    env_uuid = str(uuid.uuid4())

    manifest_payload: Dict[str, Any] = {
        "manifest_version": version,
        "environment_uuid": env_uuid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "holder": "test_setup",
        "data_dir_name": data_dir.name,
    }
    if version == "1.1":
        manifest_payload["recovery_history_counter"] = counter
        manifest_payload["last_recovery_audit_id"] = None if counter == 0 else str(uuid.uuid4())
        manifest_payload["last_recovery_utc"] = None if counter == 0 else datetime.now(timezone.utc).isoformat()

    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    # Lock CLEAN
    lock_data = {
        "status": RuntimeState.CLEAN.value,
        "holder": "test_setup",
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "reason": "setup",
        "environment_uuid": env_uuid,
    }
    lock_path.write_bytes(b"#" + json.dumps(lock_data, indent=2).encode("utf-8"))

    return {
        "data_dir": data_dir,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "audit_path": audit_path,
        "archive_dir": archive_dir,
        "fallback_dir": fallback_dir,
        "env_uuid": env_uuid,
    }


def _set_lock_state(lock_path: Path, status: RuntimeState, holder: str = "test", reason: str = ""):
    lock_data = {
        "status": status.value,
        "holder": holder,
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "reason": reason,
    }
    lock_path.write_bytes(b"#" + json.dumps(lock_data, indent=2).encode("utf-8"))


def _acquire_maint_lock(env: Dict[str, Any]) -> OperationalRuntimeLock:
    lock = OperationalRuntimeLock(
        lock_path=env["lock_path"],
        holder_name="test_maint",
    )
    lock.acquire_maintenance()
    return lock


# =========================================================================
# CENÁRIOS 1 A 10: MANIFESTO 1.1, HISTÓRICO, STARTUP E SHUTDOWN
# =========================================================================

def test_01_manifesto_1_1_valido_sem_recovery(tmp_path: Path):
    """1. Manifesto 1.1 válido sem recovery autoriza startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True
    assert "100% aprovado" in msg


def test_02_manifesto_1_1_com_recovery(tmp_path: Path):
    """2. Manifesto 1.1 com recovery consistente autoriza startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    res = execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Teste de recovery 1.1 valido",
        fallback_dir=env["fallback_dir"],
    )
    assert res["status"] == "RECOVERY_SUCCESS"

    m_data = read_runtime_manifest(env["manifest_path"])
    assert m_data["manifest_version"] == "1.1"
    assert m_data["recovery_history_counter"] == 1
    assert m_data["last_recovery_audit_id"] == res["audit_id"]

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_03_recovery_seguido_de_startup_normal(tmp_path: Path):
    """3. Recovery seguido de startup normal transiciona para RUNNING preservando manifesto."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Teste recovery seguido de startup",
        fallback_dir=env["fallback_dir"],
    )

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="market_bot")
    lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    assert lock.is_locked is True

    # O manifesto mantém counter=1
    m_data = read_runtime_manifest(env["manifest_path"])
    assert m_data["recovery_history_counter"] == 1
    lock.release()


def test_04_recovery_seguido_de_shutdown_normal(tmp_path: Path):
    """4. Recovery seguido de shutdown normal limpa lock reason mas preserva evidência no manifesto."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para teste shutdown",
        fallback_dir=env["fallback_dir"],
    )

    # Simula ciclo completo de startup e shutdown gracioso
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="market_bot")
    lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    lock.release()  # Sobrescreve lock com CLEAN ordinário

    lock_info = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert "Recovery autorizada" not in str(lock_info.get("metadata", {}).get("reason", ""))

    # Startup preflight continua aprovado pois ledger e manifesto continuam consistentes
    ok, _ = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_05_multiplos_ciclos_startup_shutdown(tmp_path: Path):
    """5. Múltiplos ciclos de startup/shutdown preservam o manifesto 1.1 e o histórico."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para multiplos ciclos",
        fallback_dir=env["fallback_dir"],
    )

    for i in range(5):
        lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name=f"bot_cycle_{i}")
        lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
        lock.release()
        ok, _ = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
        assert ok is True

    m_data = read_runtime_manifest(env["manifest_path"])
    assert m_data["recovery_history_counter"] == 1


def test_06_exclusao_do_ledger_apos_shutdown(tmp_path: Path):
    """6. Exclusão do ledger após shutdown bloqueia fail-closed no startup subsequente."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery antes de deletar ledger",
        fallback_dir=env["fallback_dir"],
    )

    # Ciclo normal de shutdown sobrescreve lock
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="market_bot")
    lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    lock.release()

    # Deleta arquivo de auditoria
    env["audit_path"].unlink()

    # O preflight DEVE bloquear porque o manifesto exige counter=1
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "Manifesto registra histórico de 1 recuperação" in msg
    assert "anti-tamper" in msg


def test_07_ledger_truncado_apos_shutdown(tmp_path: Path):
    """7. Ledger truncado após shutdown bloqueia startup fail-closed."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery antes de corromper ledger",
        fallback_dir=env["fallback_dir"],
    )

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="market_bot")
    lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    lock.release()

    # Trunca o ledger
    env["audit_path"].write_text('{"incomplete": true', encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "corrompido ou inconsistente" in msg


def test_08_contador_menor_que_historico(tmp_path: Path):
    """8. Contador menor que histórico no ledger bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para forcar divergencia",
        fallback_dir=env["fallback_dir"],
    )

    # Altera manifesto para counter=0 com 1 transação concluída no ledger
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["recovery_history_counter"] = 0
    m_data["last_recovery_audit_id"] = None
    m_data["last_recovery_utc"] = None
    env["manifest_path"].write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert any(w in msg for w in ["diverge", "registra 0 recuperações", "DIVERGENT_COUNTER"])


def test_09_contador_maior_que_historico(tmp_path: Path):
    """9. Contador maior que histórico no ledger bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para contador inflado",
        fallback_dir=env["fallback_dir"],
    )

    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["recovery_history_counter"] = 5
    env["manifest_path"].write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "diverge das transações COMMITTED" in msg


def test_10_ultimo_audit_id_divergente(tmp_path: Path):
    """10. Último audit_id no manifesto divergente do ledger bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para audit_id divergente",
        fallback_dir=env["fallback_dir"],
    )

    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["last_recovery_audit_id"] = str(uuid.uuid4())
    env["manifest_path"].write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "diverge do último COMMITTED" in msg


# =========================================================================
# CENÁRIOS 11 A 20: COMPATIBILIDADE LEGADA E CRASH CONSISTENCY
# =========================================================================

def test_11_uuid_divergente(tmp_path: Path):
    """11. UUID do manifesto divergente do ambiente esperado bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["environment_uuid"] = str(uuid.uuid4())
    env["manifest_path"].write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "Divergência de identificador de ambiente" in msg


def test_12_manifesto_1_0_sem_historico(tmp_path: Path):
    """12. Manifesto 1.0 legado sem histórico autoriza startup normalmente."""
    env = _setup_base_env(tmp_path, version="1.0")
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True
    assert "100% aprovado" in msg


def test_13_manifesto_1_0_com_ledger_legitimo(tmp_path: Path):
    """13. Manifesto 1.0 com ledger legítimo é aprovado como LEGACY_CONSISTENT."""
    env = _setup_base_env(tmp_path, version="1.0")
    audit_id = str(uuid.uuid4())
    prep = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Motivo legado de 15 caracteres",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    comm = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Motivo legado de 15 caracteres",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "sha",
        "final_state": "CLEAN",
    }
    env["audit_path"].write_text(json.dumps(prep) + "\n" + json.dumps(comm) + "\n", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_14_manifesto_1_0_com_ledger_corrompido(tmp_path: Path):
    """14. Manifesto 1.0 com ledger corrompido é bloqueado fail-closed."""
    env = _setup_base_env(tmp_path, version="1.0")
    env["audit_path"].write_text('corrupt line\n', encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "corrompido ou inconsistente" in msg


def test_15_manifesto_1_0_com_historico_pendente(tmp_path: Path):
    """15. Manifesto 1.0 com transação PREPARED órfã é bloqueado fail-closed."""
    env = _setup_base_env(tmp_path, version="1.0")
    audit_id = str(uuid.uuid4())
    prep = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Motivo legado de 15 caracteres",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(prep) + "\n", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_16_crash_antes_de_prepared(tmp_path: Path):
    """16. Crash antes de PREPARED mantém estado DIRTY e bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="crash_inicial")

    # Nada gravado no ledger
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "DIRTY" in msg


def test_17_crash_apos_prepared(tmp_path: Path):
    """17. Crash após PREPARED mantém lock DIRTY e ledger pendente, bloqueando startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="crash_apos_prep")

    audit_id = str(uuid.uuid4())
    entry = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "dir",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(entry) + "\n", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_18_crash_apos_clean(tmp_path: Path):
    """18. Crash após CLEAN (antes de atualizar manifesto) bloqueia startup por PREPARED órfão."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.CLEAN, reason="Recovery autorizada em andamento")

    audit_id = str(uuid.uuid4())
    entry = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "dir",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(entry) + "\n", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_19_crash_durante_atualizacao_do_manifesto(tmp_path: Path):
    """19. Crash durante atualização do manifesto deixa .tmp residual e bloqueia por PREPARED órfão."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.CLEAN, reason="Recovery autorizada")

    audit_id = str(uuid.uuid4())
    entry = {
        "audit_id": audit_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "dir",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(entry) + "\n", encoding="utf-8")

    # Arquivo temporário incompleto
    (env["data_dir"] / f"{MANIFEST_FILENAME}.tmp").write_text('{"temp": true', encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


def test_20_crash_apos_atualizacao_do_manifesto(tmp_path: Path):
    """20. Crash após atualização do manifesto (antes de COMMITTED) bloqueia por PREPARED órfão."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.CLEAN, reason="Recovery autorizada")

    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "audit_id": audit_id,
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "dir",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(entry) + "\n", encoding="utf-8")

    maint_lock = _acquire_maint_lock(env)
    try:
        update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=audit_id,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "PREPARED sem COMMITTED" in msg


# =========================================================================
# CENÁRIOS 21 A 30: COMMITTED, TMP RESIDUAL E VALIDAÇÃO DE CAMPOS
# =========================================================================

def test_21_crash_durante_committed(tmp_path: Path):
    """21. Crash durante gravação de COMMITTED corrompe última linha e bloqueia startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.CLEAN)

    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    prep = {
        "audit_id": audit_id,
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
        "operator_reason": "Justificativa valida de teste",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "dir",
        "snapshot_manifest_sha256": "sha",
        "previous_runtime_state": "DIRTY",
    }
    env["audit_path"].write_text(json.dumps(prep) + "\n", encoding="utf-8")

    maint_lock = _acquire_maint_lock(env)
    try:
        update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=audit_id,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    # Simula escrita parcial/truncada do COMMITTED
    with open(env["audit_path"], "a", encoding="utf-8") as f:
        f.write('{"audit_id": "' + audit_id + '", "recovery_phase": "COMMI')

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "corrompido ou inconsistente" in msg


def test_22_crash_apos_committed(tmp_path: Path):
    """22. Após COMMITTED concluído, ambiente está íntegro e autoriza startup."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="crash")

    res = execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery concluido integralmente",
        fallback_dir=env["fallback_dir"],
    )
    assert res["status"] == "RECOVERY_SUCCESS"

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_23_manifesto_temporario_residual(tmp_path: Path):
    """23. Arquivo .manifest.tmp residual não afeta leitura do manifesto oficial."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    tmp_m = env["data_dir"] / f"{MANIFEST_FILENAME}.tmp"
    tmp_m.write_text("lixo residual", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True


def test_24_falha_fsync_manifesto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """24. Falha de fsync durante atualização do manifesto aborta com erro atômico."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    real_fsync = os.fsync
    def broken_fsync(fd):
        raise OSError(5, "Simulated I/O error during fsync")

    monkeypatch.setattr(os, "fsync", broken_fsync)

    maint_lock = _acquire_maint_lock(env)
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=audit_id,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["audit_path"],
                lock_authorization=maint_lock,
            )
        assert "Falha de persistência atômica" in str(exc_info.value)
    finally:
        maint_lock.release()


def test_25_falha_os_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """25. Falha de os.replace ao atualizar manifesto aborta com erro atômico."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    def broken_replace(src, dst):
        raise OSError(13, "Permission denied on replace")

    monkeypatch.setattr(os, "replace", broken_replace)

    maint_lock = _acquire_maint_lock(env)
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=audit_id,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["audit_path"],
                lock_authorization=maint_lock,
            )
        assert "Falha de persistência atômica" in str(exc_info.value)
    finally:
        maint_lock.release()


def test_26_contador_negativo(tmp_path: Path):
    """26. recovery_history_counter negativo é rejeitado na validação de estrutura."""
    data = {
        "manifest_version": "1.1",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": -1,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
    }
    is_valid, err = validate_manifest_structure(data)
    assert is_valid is False
    assert "deve ser inteiro >= 0" in err


def test_27_contador_booleano(tmp_path: Path):
    """27. recovery_history_counter booleano (True/False) é estritamente rejeitado."""
    data = {
        "manifest_version": "1.1",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": True,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
    }
    is_valid, err = validate_manifest_structure(data)
    assert is_valid is False
    assert "rejeitando booleanos" in err


def test_28_contador_string(tmp_path: Path):
    """28. recovery_history_counter string ("1") é estritamente rejeitado."""
    data = {
        "manifest_version": "1.1",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": "1",
        "last_recovery_audit_id": str(uuid.uuid4()),
        "last_recovery_utc": datetime.now(timezone.utc).isoformat(),
    }
    is_valid, err = validate_manifest_structure(data)
    assert is_valid is False
    assert "rejeitando booleanos e strings" in err


def test_29_timestamp_invalido(tmp_path: Path):
    """29. last_recovery_utc com timestamp inválido é rejeitado."""
    data = {
        "manifest_version": "1.1",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 1,
        "last_recovery_audit_id": str(uuid.uuid4()),
        "last_recovery_utc": "invalid-timestamp",
    }
    is_valid, err = validate_manifest_structure(data)
    assert is_valid is False
    assert "não é um timestamp ISO8601 válido" in err


def test_30_audit_id_invalido(tmp_path: Path):
    """30. last_recovery_audit_id que não seja UUID válido é rejeitado."""
    data = {
        "manifest_version": "1.1",
        "environment_uuid": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_history_counter": 1,
        "last_recovery_audit_id": "not-a-uuid",
        "last_recovery_utc": datetime.now(timezone.utc).isoformat(),
    }
    is_valid, err = validate_manifest_structure(data)
    assert is_valid is False
    assert "não é um UUID válido" in err


# =========================================================================
# CENÁRIOS 31 A 40: INVARIANTES, SEGURANÇA E AMBIENTES REAIS
# =========================================================================

def test_31_preservacao_de_campos_originais(tmp_path: Path):
    """31. Campos customizados ou desconhecidos do manifesto são estritamente preservados na atualização."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    m_data = read_runtime_manifest(env["manifest_path"])
    m_data["custom_operator_tag"] = "data_center_alpha"
    m_data["custom_cluster_id"] = 42
    env["manifest_path"].write_text(json.dumps(m_data, indent=2), encoding="utf-8")

    audit_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    maint_lock = _acquire_maint_lock(env)
    try:
        updated = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=audit_id,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["audit_path"],
            lock_authorization=maint_lock,
        )
    finally:
        maint_lock.release()

    assert updated["custom_operator_tag"] == "data_center_alpha"
    assert updated["custom_cluster_id"] == 42
    assert updated["recovery_history_counter"] == 1


def test_32_preservacao_do_uuid(tmp_path: Path):
    """32. environment_uuid original permanece inalterado após múltiplos recoveries."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    orig_uuid = env["env_uuid"]

    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash_1")
    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery numero 1 de teste",
        fallback_dir=env["fallback_dir"],
    )

    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash_2")
    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery numero 2 de teste",
        fallback_dir=env["fallback_dir"],
    )

    m_data = read_runtime_manifest(env["manifest_path"])
    assert m_data["environment_uuid"] == orig_uuid
    assert m_data["recovery_history_counter"] == 2


def test_33_imutabilidade_dos_dados_financeiros(tmp_path: Path):
    """33. Tabelas SQLite operacionais permanecem intactas durante atualização do manifesto."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    import sqlite3
    conn = sqlite3.connect(str(env["sqlite_path"]))
    conn.execute(
        """
        INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
        VALUES (1710000002000, 'trade', 'BTCUSDT', 'w1', 0, '{"event_id": "ev_002", "price": "66000"}')
        """
    )
    conn.commit()
    conn.close()

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery com dados financeiros",
        fallback_dir=env["fallback_dir"],
    )

    conn2 = sqlite3.connect(str(env["sqlite_path"]))
    rows = conn2.execute("SELECT * FROM events").fetchall()
    conn2.close()
    assert len(rows) == 2


def test_34_compatibilidade_com_bootstrap(tmp_path: Path):
    """34. Novo bootstrap de ambiente gera manifesto 1.1 com contador 0 e campos nulos."""
    data_dir = tmp_path / "new_dados"
    res = bootstrap_runtime_environment(data_dir=data_dir, holder_name="test_bootstrap")
    m_path = data_dir / MANIFEST_FILENAME
    m_data = read_runtime_manifest(m_path)

    assert m_data["manifest_version"] == "1.1"
    assert m_data["recovery_history_counter"] == 0
    assert m_data["last_recovery_audit_id"] is None
    assert m_data["last_recovery_utc"] is None

    ok, msg = verify_startup_preflight(data_dir=data_dir)
    assert ok is True


def test_35_compatibilidade_com_lock_windows(tmp_path: Path):
    """35. Prefixo de byte 0 b'#' no lock do Windows é mantido íntegro durante recovery."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery no lock do Windows",
        fallback_dir=env["fallback_dir"],
    )

    with open(env["lock_path"], "rb") as f:
        first_byte = f.read(1)
    assert first_byte == b"#"


def test_36_compatibilidade_com_lock_posix(tmp_path: Path):
    """36. Leitura defensiva read_persisted_state interpreta corretamente o estado persistido."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery no formato POSIX",
        fallback_dir=env["fallback_dir"],
    )

    state = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert state["status"] == RuntimeState.CLEAN


def test_37_recovery_concorrente(tmp_path: Path):
    """37. Processo concorrente com lock retido impede segundo recovery de atualizar manifesto."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="test_crash")

    lock1 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="competing_holder")
    lock1.acquire(timeout=0.0)

    with pytest.raises((RuntimeError, RuntimeLockError)) as exc_info:
        execute_authorized_recovery(
            data_dir=env["data_dir"],
            lock_path=env["lock_path"],
            reason="Segundo recovery concorrente",
            fallback_dir=env["fallback_dir"],
        )
    assert any(term in str(exc_info.value) for term in ["recurso retido", "Conflito de exclusividade", "Permission denied", "bloquead"])
    lock1.release()


def test_38_startup_concorrente(tmp_path: Path):
    """38. Startup concorrente é bloqueado mutuamente via lock físico."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)

    lock1 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="primary_bot")
    lock1.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    lock2 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="secondary_bot")
    with pytest.raises(RuntimeLockError) as exc_info:
        lock2.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    assert "Conflito de exclusividade" in str(exc_info.value)
    lock1.release()


def test_39_ausencia_do_ledger_apos_multiplos_shutdowns(tmp_path: Path):
    """39. Deleção do ledger após 10 ciclos de startup/shutdown bloqueia startup sem atenuação."""
    env = _setup_base_env(tmp_path, version="1.1", counter=0)
    _set_lock_state(env["lock_path"], RuntimeState.DIRTY, reason="crash")

    execute_authorized_recovery(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        reason="Recovery para teste de 10 ciclos",
        fallback_dir=env["fallback_dir"],
    )

    for i in range(10):
        lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name=f"bot_run_{i}")
        lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
        lock.release()

    # Deleta ledger após 10 shutdowns normais
    env["audit_path"].unlink()

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "Manifesto registra histórico de 1 recuperação" in msg
    assert "anti-tamper" in msg


def test_40_nenhuma_migracao_automatica_durante_startup(tmp_path: Path):
    """40. Startup preflight é estritamente read-only e não migra manifesto 1.0 para 1.1."""
    env = _setup_base_env(tmp_path, version="1.0")
    m_stat_before = env["manifest_path"].stat()

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is True

    m_data = read_runtime_manifest(env["manifest_path"])
    assert m_data["manifest_version"] == "1.0"
    assert "recovery_history_counter" not in m_data
    assert env["manifest_path"].stat().st_mtime == m_stat_before.st_mtime
