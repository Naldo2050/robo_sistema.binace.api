# -*- coding: utf-8 -*-
"""
Suíte de Testes do Gate G4-D B3-B2-C2-B2-P0:
Hardening da Autorização de Mutação do Manifesto Operacional.

Cobre os 16 cenários obrigatórios de validação de capacidade de autorização,
exclusividade física de lock no SO, rejeição de bypasses, preservação H1/H2 e fail-closed em tmp_path.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from common.runtime_lock import (
    MANIFEST_FILENAME,
    MaintenanceAuthorizationToken,
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    read_runtime_manifest,
    update_runtime_manifest_reconciliation_state,
    verify_maintenance_authorization,
    verify_startup_preflight,
    write_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    record_reconciliation_audit,
)
from scripts.maintenance.reconcile_fallback_events import main as cli_main


def _setup_test_env(tmp_path: Path, manifest_version: str = "1.2") -> Dict[str, Any]:
    """Configura ambiente temporário isolado para testes de autorização P0."""
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

    env_uuid = str(uuid.uuid4())
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        holder_name="test_p0_setup",
        manifest_version=manifest_version,
    )

    # Inicia lock operacional CLEAN
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

    # SQLite
    conn = sqlite3.connect(str(sqlite_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            payload JSON NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    # JSONL
    sample = {"event_id": "ev_001", "timestamp_ms": 1710000000000, "event_type": "trade"}
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    return {
        "data_dir": data_dir,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "fallback_dir": fallback_dir,
        "rec_audit_path": rec_audit_path,
        "env_uuid": env_uuid,
    }


# ==============================================================================
# 1. TESTES DE REJEIÇÃO DE AUTORIZAÇÃO INVÁLIDA / BYPASS
# ==============================================================================

def test_01_mutacao_sem_sessao_rejeitada(tmp_path: Path):
    """1. Chamada sem parâmetro de autorização (lock_authorization=None) é estritamente rejeitada."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=None,
        )
    assert "lock_authorization não pode ser None" in str(exc_info.value)


def test_02_lock_file_obj_none_rejeitado(tmp_path: Path):
    """2. Passar lock_file_obj=None explicitamente é estritamente rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_file_obj=None,
        )
    assert "lock_authorization não pode ser None" in str(exc_info.value)


def test_03_objeto_arbitrario_rejeitado(tmp_path: Path):
    """3. Objetos arbitrários, strings, dicts e booleanos são estritamente rejeitados."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    invalid_authorizations = [
        True,
        False,
        "fake_token_string",
        123456,
        {"is_locked": True, "token": "spoofed"},
        object(),
    ]

    for invalid_auth in invalid_authorizations:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=invalid_auth,
            )
        err = str(exc_info.value)
        assert ("booleano rejeitado" in err) or ("Objeto de autorização inválido" in err)


def test_04_descritor_fechado_rejeitado(tmp_path: Path):
    """4. Autorização cujo descritor de arquivo foi fechado é imediatamente rejeitada."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    # Fecha propositalmente o descritor físico
    op_lock._file_obj.close()

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "descritor físico de lock fechado" in str(exc_info.value)
    op_lock.release_maintenance()


def test_05_sessao_liberada_rejeitada(tmp_path: Path):
    """5. Token de sessão de manutenção que já foi liberada/revogada é estritamente rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    # Libera a sessão antes da tentativa de mutação
    op_lock.release_maintenance()
    assert token.is_active is False

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "sessão de manutenção já foi liberada ou revogada" in str(exc_info.value)


def test_06_sessao_de_outro_ambiente_rejeitada(tmp_path: Path):
    """6. Autorização emitida com environment_uuid divergente é rejeitada fail-closed."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    alien_uuid = str(uuid.uuid4())
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=alien_uuid)

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "token emitido para outro environment_uuid" in str(exc_info.value)
    op_lock.release_maintenance()


def test_07_sessao_de_outro_caminho_rejeitada(tmp_path: Path):
    """7. Autorização cujo lock físico pertença a outro diretório/arquivo é rejeitada."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    alien_dir = tmp_path / "alien_dados"
    alien_dir.mkdir(parents=True, exist_ok=True)
    alien_lock_path = alien_dir / ".operational_runtime.lock"
    alien_lock_path.write_bytes(b"#{\"status\": \"CLEAN\"}")

    alien_lock = OperationalRuntimeLock(lock_path=alien_lock_path, holder_name="alien_p0")
    alien_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = alien_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "lock path do token" in str(exc_info.value)
    alien_lock.release_maintenance()


# ==============================================================================
# 2. TESTES DE AUTORIZAÇÃO VÁLIDA E EXCLUSIVIDADE
# ==============================================================================

def test_08_sessao_valida_aceita(tmp_path: Path):
    """8. Sessão de manutenção legítima permite mutação atômica do manifesto com sucesso."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    with op_lock.maintenance_session(allowed_states=[RuntimeState.CLEAN], environment_uuid=env["env_uuid"]) as maint_info:
        auth_token = maint_info["authorization"]
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=auth_token,
        )
        assert updated["manifest_version"] == "1.2"
        assert updated["reconciliation_history_counter"] == 1
        assert updated["last_reconciliation_audit_id"] == aid

    # Após o context manager, a autorização deve ter sido revogada
    assert auth_token.is_active is False


def test_09_sessao_concorrente_bloqueada(tmp_path: Path):
    """9. Quando uma sessão de manutenção detém a trava no SO, processo concorrente é bloqueado."""
    env = _setup_test_env(tmp_path)
    op_lock_1 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="holder_1")
    op_lock_1.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])

    op_lock_2 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="holder_2")
    with pytest.raises(RuntimeLockError) as exc_info:
        op_lock_2.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    assert "Conflito de exclusividade" in str(exc_info.value)

    op_lock_1.release_maintenance()


# ==============================================================================
# 3. TESTES DE TOLERÂNCIA A FALHAS E INTEGRIDADE
# ==============================================================================

def test_10_falha_fsync_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """10. Falha de fsync sob sessão válida aborta atomicamente sem alterar o manifesto original."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    manifest_bytes_before = env["manifest_path"].read_bytes()

    def failing_fsync(fd):
        raise OSError(5, "Simulated disk failure during fsync")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "Falha de persistência atômica" in str(exc_info.value)
    assert env["manifest_path"].read_bytes() == manifest_bytes_before
    assert not (env["data_dir"] / f"{MANIFEST_FILENAME}.tmp").exists()
    op_lock.release_maintenance()


def test_11_falha_os_replace_bloqueia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """11. Falha em os.replace sob sessão válida aborta e preserva o manifesto intacto."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    manifest_bytes_before = env["manifest_path"].read_bytes()

    def failing_replace(src, dst):
        raise OSError(13, "Simulated access denied during replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=token,
        )
    assert "Falha de persistência atômica" in str(exc_info.value)
    assert env["manifest_path"].read_bytes() == manifest_bytes_before
    op_lock.release_maintenance()


# ==============================================================================
# 4. PRESERVAÇÃO H1 / H2 E GOVERNANÇA FINANCEIRA
# ==============================================================================

def test_12_campos_h1_preservados(tmp_path: Path):
    """12. Mutação autorizada de reconciliação preserva integralmente os campos H1 de recovery."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    # Simula recovery prévio gravado no manifesto 1.1
    m_data = read_runtime_manifest(env["manifest_path"])
    recov_id = str(uuid.uuid4())
    recov_ts = datetime.now(timezone.utc).isoformat()
    m_data["recovery_history_counter"] = 2
    m_data["last_recovery_audit_id"] = recov_id
    m_data["last_recovery_utc"] = recov_ts
    env["manifest_path"].write_text(json.dumps(m_data), encoding="utf-8")

    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    updated = update_runtime_manifest_reconciliation_state(
        manifest_path=env["manifest_path"],
        audit_id=aid,
        audit_timestamp_utc=ts,
        expected_env_uuid=env["env_uuid"],
        ledger_path=env["rec_audit_path"],
        lock_authorization=token,
    )
    assert updated["manifest_version"] == "1.2"
    assert updated["recovery_history_counter"] == 2
    assert updated["last_recovery_audit_id"] == recov_id
    assert updated["last_recovery_utc"] == recov_ts
    assert updated["reconciliation_history_counter"] == 1
    op_lock.release_maintenance()


def test_13_campos_h2_preservados(tmp_path: Path):
    """13. Mutação autorizada atualiza os campos H2 com contadores monotônicos e timestamp válido."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    updated = update_runtime_manifest_reconciliation_state(
        manifest_path=env["manifest_path"],
        audit_id=aid,
        audit_timestamp_utc=ts,
        expected_env_uuid=env["env_uuid"],
        ledger_path=env["rec_audit_path"],
        lock_authorization=token,
    )
    assert updated["reconciliation_history_counter"] == 1
    assert updated["last_reconciliation_audit_id"] == aid
    assert updated["last_reconciliation_utc"] == ts
    op_lock.release_maintenance()


def test_14_uuid_preservado(tmp_path: Path):
    """14. Mutação autorizada preserva exatamente o environment_uuid original."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    orig_uuid = env["env_uuid"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    op_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = op_lock.create_maintenance_authorization(environment_uuid=orig_uuid)

    updated = update_runtime_manifest_reconciliation_state(
        manifest_path=env["manifest_path"],
        audit_id=aid,
        audit_timestamp_utc=ts,
        expected_env_uuid=orig_uuid,
        ledger_path=env["rec_audit_path"],
        lock_authorization=token,
    )
    assert updated["environment_uuid"] == orig_uuid
    op_lock.release_maintenance()


def test_15_cli_apply_continua_desabilitada(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """15. CLI de reconciliação sob --apply encerra com código 1 e APPLY_NOT_ENABLED_C2A."""
    env = _setup_test_env(tmp_path)
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


def test_16_zero_mutacao_financeira(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """16. Nenhuma mutação ocorre nas bases operacionais SQLite e JSONL."""
    env = _setup_test_env(tmp_path)
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
