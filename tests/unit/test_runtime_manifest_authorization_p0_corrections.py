# -*- coding: utf-8 -*-
"""
Suíte de Testes do Gate G4-D B3-B2-C2-B2-P0 (Correções P0):
Hardening Avançado de Autorização, Proteção Anti-TOCTOU e Diferenciação de Sessões.

Cobre os 23 cenários obrigatórios (T01 a T23) para eliminação dos achados:
- P0-AUTH-01 (Recovery sem autorização estritamente bloqueado)
- P1-TOCTOU-01 (Região crítica atômica durante mutação e bloqueio de release concorrente)
- P2-AUTH-02 (Diferenciação formal entre acquire operacional e acquire_maintenance)
- Auditoria de Identidade (Rejeição de tokens forjados, adulterados ou de sessões anteriores)
- Governança e Fail-Closed em tmp_path.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import threading
import time
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
    update_runtime_manifest_recovery_state,
    validate_manifest_structure,
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
    """Configura ambiente temporário isolado para testes de correção P0."""
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
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        holder_name="test_p0_setup",
        manifest_version=manifest_version,
    )

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
        "recov_audit_path": recov_audit_path,
        "env_uuid": env_uuid,
    }


def _record_recovery_audit_entry(audit_file: Path, audit_id: str, ts: str, env_uuid: str) -> None:
    """Grava entrada PREPARED no ledger de recovery."""
    prep = {
        "audit_id": audit_id,
        "timestamp_utc": ts,
        "environment_uuid": env_uuid,
        "operator_reason": "Justificativa valida de teste com mais de 15 caracteres",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "test_dir",
        "snapshot_manifest_sha256": "test_sha",
        "previous_runtime_state": "DIRTY",
    }
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prep) + "\n")


# ==============================================================================
# T01 A T07: CORREÇÃO P0-AUTH-01 (RECOVERY EXIGE AUTORIZAÇÃO OBRIGATÓRIA)
# ==============================================================================

def test_t01_recovery_sem_lock_authorization_rejeitado(tmp_path: Path):
    """T01 — Recovery sem argumento lock_authorization é estritamente rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    with pytest.raises((TypeError, RuntimeLockError)):
        # Chamada omitindo o argumento de autorização
        update_runtime_manifest_recovery_state(  # type: ignore
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
        )


def test_t02_recovery_com_lock_authorization_none_rejeitado(tmp_path: Path):
    """T02 — Recovery com lock_authorization=None é estritamente rejeitado fail-closed."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=None,
        )
    assert "lock_authorization não pode ser None" in str(exc_info.value)


def test_t03_recovery_com_objeto_arbitrario_rejeitado(tmp_path: Path):
    """T03 — Recovery com objeto arbitrário, string ou booleano é rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    invalid_tokens = [True, False, "fake_token", 9999, object()]
    for inv in invalid_tokens:
        with pytest.raises(RuntimeLockError):
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["recov_audit_path"],
                lock_authorization=inv,
            )


def test_t04_recovery_com_sessao_encerrada_rejeitado(tmp_path: Path):
    """T04 — Recovery com token de sessão encerrada/revogada é rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])

    lock.release_maintenance()
    assert token.is_active is False

    with pytest.raises(RuntimeLockError) as exc_info:
        update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=token,
        )
    err_msg = str(exc_info.value)
    assert ("liberada ou revogada" in err_msg) or ("não pertence a uma sessão de manutenção exclusiva" in err_msg)


def test_t05_recovery_com_sessao_legitima_aceito(tmp_path: Path):
    """T05 — Recovery com sessão legítima ativa atualiza o manifesto com sucesso."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=lock,
        )
        assert updated["recovery_history_counter"] == 1
        assert updated["last_recovery_audit_id"] == aid
    finally:
        lock.release_maintenance()


def test_t06_recovery_com_sessao_outro_ambiente_rejeitado(tmp_path: Path):
    """T06 — Recovery com token emitido para outro environment_uuid é rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    alien_uuid = str(uuid.uuid4())
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = lock.create_maintenance_authorization(environment_uuid=alien_uuid)
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["recov_audit_path"],
                lock_authorization=token,
            )
        assert "outro environment_uuid" in str(exc_info.value)
    finally:
        lock.release_maintenance()


def test_t07_recovery_com_sessao_outro_caminho_rejeitado(tmp_path: Path):
    """T07 — Recovery com lock físico de outro diretório é rejeitado fail-closed."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    alien_dir = tmp_path / "alien_dados"
    alien_dir.mkdir(parents=True, exist_ok=True)
    alien_lock_path = alien_dir / ".operational_runtime.lock"
    alien_lock_path.write_bytes(b"#{\"status\": \"CLEAN\"}")

    alien_lock = OperationalRuntimeLock(lock_path=alien_lock_path, holder_name="alien")
    alien_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token = alien_lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["recov_audit_path"],
                lock_authorization=token,
            )
        assert "não corresponde ao diretório do manifesto" in str(exc_info.value)
    finally:
        alien_lock.release_maintenance()


# ==============================================================================
# T08 A T11: CORREÇÃO P2-AUTH-02 (DIFERENCIAÇÃO DE ACQUIRE OPERACIONAL E MANUTENÇÃO)
# ==============================================================================

def test_t08_lock_operacional_nao_autoriza_recovery(tmp_path: Path):
    """T08 — Lock adquirido via acquire() operacional (RUNNING) não autoriza mutação de recovery."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="operational_runner")
    op_lock.acquire() # Operação normal, status=RUNNING
    try:
        assert op_lock.is_locked is True
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_recovery_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["recov_audit_path"],
                lock_authorization=op_lock,
            )
        assert "não está em sessão de manutenção exclusiva" in str(exc_info.value)
    finally:
        op_lock.release()


def test_t09_lock_operacional_nao_autoriza_reconciliacao(tmp_path: Path):
    """T09 — Lock adquirido via acquire() operacional não autoriza mutação de reconciliação."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="operational_runner")
    op_lock.acquire()
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=op_lock,
            )
        assert "não está em sessão de manutenção exclusiva" in str(exc_info.value)
    finally:
        op_lock.release()


def test_t10_sessao_manutencao_legitima_autoriza_recovery(tmp_path: Path):
    """T10 — Sessão de manutenção legítima (acquire_maintenance) autoriza mutação de recovery."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid, ts, env["env_uuid"])

    maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maint_tool")
    maint_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=maint_lock,
        )
        assert updated["recovery_history_counter"] == 1
    finally:
        maint_lock.release_maintenance()


def test_t11_sessao_manutencao_legitima_autoriza_reconciliacao(tmp_path: Path):
    """T11 — Sessão de manutenção legítima (acquire_maintenance) autoriza mutação de reconciliação."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    maint_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="maint_tool")
    maint_lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=maint_lock,
        )
        assert updated["reconciliation_history_counter"] == 1
    finally:
        maint_lock.release_maintenance()


# ==============================================================================
# T12 A T14: AUDITORIA DE IDENTIDADE E CICLO DE VIDA DO TOKEN
# ==============================================================================

def test_t12_token_manualmente_construido_rejeitado(tmp_path: Path):
    """T12 — Token construído manualmente sem emissão pela sessão atual é rejeitado."""
    env = _setup_test_env(tmp_path)
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        # Cria um segundo token paralelo apontando para o lock mas que não é o active_token
        # Primeiro emite o legítimo
        legit_token = lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])
        assert lock.active_maintenance_token is legit_token

        # Cria um token espúrio
        fake_token = MaintenanceAuthorizationToken(
            lock_instance=lock,
            lock_file_obj=lock._file_obj,
            lock_path=lock.lock_path,
            environment_uuid=env["env_uuid"],
        )
        # O active_maintenance_token continua sendo legit_token
        with pytest.raises(RuntimeLockError) as exc_info:
            verify_maintenance_authorization(
                lock_authorization=fake_token,
                manifest_path=env["manifest_path"],
                expected_env_uuid=env["env_uuid"],
            )
        assert "não corresponde à capacidade ativa" in str(exc_info.value)
    finally:
        lock.release_maintenance()


def test_t13_token_de_sessao_anterior_rejeitado(tmp_path: Path):
    """T13 — Token emitido em sessão anterior é estritamente rejeitado em nova sessão."""
    env = _setup_test_env(tmp_path)
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")

    # Sessão 1
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token_1 = lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])
    lock.release_maintenance()

    # Sessão 2
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    token_2 = lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])
    try:
        # Tentativa de usar token_1 na Sessão 2
        with pytest.raises(RuntimeLockError) as exc_info:
            verify_maintenance_authorization(
                lock_authorization=token_1,
                manifest_path=env["manifest_path"],
                expected_env_uuid=env["env_uuid"],
            )
        assert "liberada ou revogada" in str(exc_info.value) or "não corresponde à capacidade ativa" in str(exc_info.value)
    finally:
        lock.release_maintenance()


def test_t14_token_revogado_rejeitado(tmp_path: Path):
    """T14 — Token com revoke() explícito é rejeitado mesmo que descritor continue aberto."""
    env = _setup_test_env(tmp_path)
    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        token = lock.create_maintenance_authorization(environment_uuid=env["env_uuid"])
        token.revoke() # Revogação manual
        with pytest.raises(RuntimeLockError) as exc_info:
            verify_maintenance_authorization(
                lock_authorization=token,
                manifest_path=env["manifest_path"],
                expected_env_uuid=env["env_uuid"],
            )
        assert "liberada ou revogada" in str(exc_info.value)
    finally:
        lock.release_maintenance()


# ==============================================================================
# T15 E T16: CORREÇÃO P1-TOCTOU-01 (REGIÃO CRÍTICA E SINCRONIZAÇÃO MULTI-THREAD)
# ==============================================================================

def test_t15_concorrencia_impede_release_durante_mutacao(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """T15 — Concorrência: Thread B tenta release_maintenance() enquanto Thread A executa mutação; Thread B bloqueia até o término de A."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_toctou")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])

    thread_a_entered = threading.Event()
    release_attempted = threading.Event()
    thread_b_finished = threading.Event()

    real_fsync = os.fsync
    def synchronized_fsync(fd):
        real_fsync(fd)
        thread_a_entered.set()
        # Aguarda a Thread B tentar release antes de prosseguir com os.replace
        release_attempted.wait(timeout=3.0)

    monkeypatch.setattr(os, "fsync", synchronized_fsync)

    b_release_result = {"released": False, "error": None}

    def worker_b():
        # Aguarda Thread A entrar na região crítica da mutação
        if not thread_a_entered.wait(timeout=3.0):
            return
        release_attempted.set()
        try:
            # Tenta executar release_maintenance. Deve ficar bloqueado no _thread_lock!
            lock.release_maintenance()
            b_release_result["released"] = True
        except Exception as e:
            b_release_result["error"] = e
        finally:
            thread_b_finished.set()

    t_b = threading.Thread(target=worker_b, name="Thread-B-Release", daemon=True)
    t_b.start()

    try:
        # Thread A executa a mutação completa
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=lock,
        )
        assert updated["reconciliation_history_counter"] == 1

        # Thread B deve ter completado logo após a liberação da região crítica por A
        assert thread_b_finished.wait(timeout=3.0) is True
        assert b_release_result["released"] is True
        assert lock.is_locked is False
    finally:
        if lock.is_locked:
            lock.release_maintenance()
        t_b.join(timeout=1.0)


def test_t16_liberacao_normal_pos_mutacao(tmp_path: Path):
    """T16 — Após conclusão da mutação protegida, release_maintenance() encerra a sessão com sucesso."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_toctou")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid,
            audit_timestamp_utc=ts,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=lock,
        )
    finally:
        lock.release_maintenance()

    assert lock.is_locked is False
    assert lock.active_maintenance_token is None


# ==============================================================================
# T17 E T18: FAIL-CLOSED EM FALHAS DE I/O SOB REGIÃO CRÍTICA
# ==============================================================================

def test_t17_falha_fsync_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """T17 — Falha de fsync dentro da região crítica preserva o manifesto intacto (fail-closed)."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    manifest_bytes_before = env["manifest_path"].read_bytes()
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    def failing_fsync(fd):
        raise OSError(5, "Hardware write failure during fsync")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=lock,
            )
        assert "Falha de persistência atômica" in str(exc_info.value)
        assert env["manifest_path"].read_bytes() == manifest_bytes_before
        assert not (env["data_dir"] / f"{MANIFEST_FILENAME}.tmp").exists()
    finally:
        lock.release_maintenance()


def test_t18_falha_os_replace_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """T18 — Falha de os.replace preserva manifesto e limpa .tmp de forma segura."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    manifest_bytes_before = env["manifest_path"].read_bytes()
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    def failing_replace(src, dst):
        raise OSError(13, "Access denied during atomic replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        with pytest.raises(RuntimeLockError) as exc_info:
            update_runtime_manifest_reconciliation_state(
                manifest_path=env["manifest_path"],
                audit_id=aid,
                audit_timestamp_utc=ts,
                expected_env_uuid=env["env_uuid"],
                ledger_path=env["rec_audit_path"],
                lock_authorization=lock,
            )
        assert "Falha de persistência atômica" in str(exc_info.value)
        assert env["manifest_path"].read_bytes() == manifest_bytes_before
    finally:
        lock.release_maintenance()


# ==============================================================================
# T19 A T21: PRESERVAÇÃO H1, H2 E IDENTIDADE DE AMBIENTE
# ==============================================================================

def test_t19_campos_h1_e_h2_preservados_apos_recovery(tmp_path: Path):
    """T19 — Campos H1 e H2 são mutuamente preservados após mutação de recovery."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    aid_recov = str(uuid.uuid4())
    ts_recov = datetime.now(timezone.utc).isoformat()
    _record_recovery_audit_entry(env["recov_audit_path"], aid_recov, ts_recov, env["env_uuid"])

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_recovery_state(
            manifest_path=env["manifest_path"],
            audit_id=aid_recov,
            audit_timestamp_utc=ts_recov,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["recov_audit_path"],
            lock_authorization=lock,
        )
        assert updated["manifest_version"] == "1.2"
        assert updated["recovery_history_counter"] == 1
        assert updated["last_recovery_audit_id"] == aid_recov
        assert updated["reconciliation_history_counter"] == 0
    finally:
        lock.release_maintenance()


def test_t20_campos_h1_e_h2_preservados_apos_reconciliacao(tmp_path: Path):
    """T20 — Campos H1 e H2 são mutuamente preservados após mutação de reconciliação."""
    env = _setup_test_env(tmp_path, manifest_version="1.1")
    # Define histórico H1 prévio
    m_data = read_runtime_manifest(env["manifest_path"])
    recov_id = str(uuid.uuid4())
    recov_ts = datetime.now(timezone.utc).isoformat()
    m_data["recovery_history_counter"] = 5
    m_data["last_recovery_audit_id"] = recov_id
    m_data["last_recovery_utc"] = recov_ts
    env["manifest_path"].write_text(json.dumps(m_data), encoding="utf-8")

    aid_rec = str(uuid.uuid4())
    ts_rec = datetime.now(timezone.utc).isoformat()

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=aid_rec,
            audit_timestamp_utc=ts_rec,
            expected_env_uuid=env["env_uuid"],
            ledger_path=env["rec_audit_path"],
            lock_authorization=lock,
        )
        assert updated["manifest_version"] == "1.2"
        assert updated["reconciliation_history_counter"] == 1
        assert updated["recovery_history_counter"] == 5
        assert updated["last_recovery_audit_id"] == recov_id
        assert updated["last_recovery_utc"] == recov_ts
    finally:
        lock.release_maintenance()


def test_t21_environment_uuid_preservado(tmp_path: Path):
    """T21 — environment_uuid permanece imutável após ambas as mutações."""
    env = _setup_test_env(tmp_path, manifest_version="1.2")
    orig_uuid = env["env_uuid"]

    lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="test_p0")
    lock.acquire_maintenance(allowed_states=[RuntimeState.CLEAN])
    try:
        updated = update_runtime_manifest_reconciliation_state(
            manifest_path=env["manifest_path"],
            audit_id=str(uuid.uuid4()),
            audit_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            expected_env_uuid=orig_uuid,
            ledger_path=env["rec_audit_path"],
            lock_authorization=lock,
        )
        assert updated["environment_uuid"] == orig_uuid
    finally:
        lock.release_maintenance()


# ==============================================================================
# T22 E T23: GOVERNANÇA FINANCEIRA E ZERO MUTAÇÃO EM PRODUÇÃO
# ==============================================================================

def test_t22_apply_not_enabled_c2a_continua_ativo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """T22 — CLI apply encerra com código 1 e APPLY_NOT_ENABLED_C2A incondicionalmente."""
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


def test_t23_zero_gravacao_financeira(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """T23 — Nenhuma gravação ocorre nas bases operacionais sob tentativa de apply."""
    env = _setup_test_env(tmp_path)
    sq_size = env["sqlite_path"].stat().st_size
    jl_size = env["jsonl_path"].stat().st_size

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

    assert env["sqlite_path"].stat().st_size == sq_size
    assert env["jsonl_path"].stat().st_size == jl_size
