# -*- coding: utf-8 -*-
"""
Suíte de Testes para o Procedimento Oficial e Autorizado de Recovery de Runtime (Gate G4-D B3-B2-B3).

Cobre exaustivamente os 30 cenários exigidos da Etapa 8:
 1. Recovery DIRTY válido.
 2. Recovery RUNNING residual.
 3. Recovery UNKNOWN.
 4. Reason ausente.
 5. Reason insuficiente (< 15 chars).
 6. Lock ocupado por outro processo.
 7. SQLite corrompido.
 8. JSONL corrompido.
 9. Colisão de payload.
10. Archive com .tmp residual.
11. Falha de snapshot (isolamento de diretório .INCOMPLETE_FAILED).
12. ENOSPC simulado (verificação preventiva de cota).
13. Falha de auditoria antes da transição.
14. Crash antes de PREPARED.
15. Crash após PREPARED (detecção de transação incompleta).
16. Crash durante transição para CLEAN.
17. Crash após CLEAN antes de COMMITTED (fail-closed no startup).
18. Falha durante gravação de COMMITTED.
19. Execução concorrente de recovery.
20. Startup concorrente bloqueado durante manutenção.
21. Estado CLEAN idempotente (zero mutações desnecessárias).
22. Preservação de fixtures Git em fallback.
23. Preservação de WAL/SHM forense no snapshot.
24. Integridade dos hashes SHA256 do snapshot.
25. Imutabilidade dos dados originais (SQLite e JSONL inalterados).
26. Rejeição de manifesto inválido (anti-bypass).
27. Rejeição de UUID divergente entre manifesto e lock.
28. Rejeição de eventos pendentes não reconciliados no fallback.
29. Rejeição de eventos malformados sem ID no fallback.
30. Rejeição de corrupção ou problemas estruturais em archives.

Todos os testes rodam estritamente em tmp_path. Zero escrita ou acesso a dados de produção.
"""

from __future__ import annotations

import hashlib
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
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    MANIFEST_FILENAME,
    write_runtime_manifest,
    bootstrap_runtime_environment,
    verify_startup_preflight,
    validate_recovery_audit_ledger,
)
from scripts.maintenance.recover_runtime_state import (
    diagnose_runtime_recovery,
    create_recovery_snapshot,
    record_recovery_audit,
    execute_authorized_recovery,
)
from events.fallback_reconciler import compute_canonical_payload_hash


def _create_test_sqlite_with_events(db_path: Path, events: List[Dict[str, Any]]) -> None:
    """Cria banco SQLite com schema real da tabela events."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
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
    for ev in events:
        p = ev.get("payload") if ("payload" in ev and isinstance(ev["payload"], dict)) else ev
        cur.execute(
            """
            INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ev.get("timestamp_ms", 1710000000000),
                ev.get("event_type", "trade"),
                ev.get("symbol", "BTCUSDT"),
                ev.get("window_id", "w1"),
                ev.get("is_signal", 0),
                json.dumps(p),
            ),
        )
    conn.commit()
    conn.close()


def _create_test_jsonl(jsonl_path: Path, records: List[Dict[str, Any]], trailing_newline: bool = True) -> None:
    """Cria arquivo JSONL de eventos."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            if "payload" in rec and isinstance(rec["payload"], dict):
                merged = dict(rec)
                merged.update(rec["payload"])
                del merged["payload"]
                line = json.dumps(merged)
            else:
                line = json.dumps(rec)
            if idx == len(records) - 1 and not trailing_newline:
                f.write(line)
            else:
                f.write(line + "\n")


def _setup_test_environment(
    tmp_path: Path,
    lock_status: str = "DIRTY",
    lock_reason: str = "test_shutdown_incomplete",
    events: Optional[List[Dict[str, Any]]] = None,
    env_uuid: Optional[str] = None,
) -> Tuple[Path, Path, Path, Path, str]:
    """Prepara um ambiente completo de teste com manifesto, lock, SQLite e JSONL válidos."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".operational_runtime.lock"
    manifest_path = data_dir / MANIFEST_FILENAME
    backup_dir = tmp_path / "backups" / "recovery_snapshots"
    fallback_dir = tmp_path / "fallback_events"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    target_uuid = env_uuid or str(uuid.uuid4())

    # Grava manifesto
    manifest_meta = {
        "manifest_version": "1.0",
        "environment_uuid": target_uuid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "test_setup",
        "host": platform.node(),
        "pid": os.getpid(),
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_meta, mf, indent=2)

    # Grava lock operacional inicial com byte 0 reservado
    lock_meta = {
        "status": lock_status,
        "holder": "test_holder",
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "reason": lock_reason,
        "environment_uuid": target_uuid,
    }
    payload = b"#" + json.dumps(lock_meta, indent=2).encode("utf-8")
    with open(lock_path, "wb") as lf:
        lf.write(payload)

    # Popula SQLite e JSONL com eventos consistentes
    if events is None:
        events = [
            {
                "event_id": "ev_001",
                "timestamp_ms": 1710000001000,
                "event_type": "trade",
                "symbol": "BTCUSDT",
                "window_id": "w1",
                "price": "65000.0",
                "qty": "0.1",
                "side": "BUY",
            }
        ]

    db_path = data_dir / "trading_bot.db"
    _create_test_sqlite_with_events(db_path, events)

    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_test_jsonl(jsonl_path, events)

    return data_dir, lock_path, backup_dir, fallback_dir, target_uuid


# =============================================================================
# 1. Recovery DIRTY válido
# =============================================================================
def test_01_recovery_dirty_valido(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(
        tmp_path, lock_status="DIRTY", lock_reason="shutdown_interrompido"
    )

    reason = "Autorização formal de recuperação do operador após auditoria"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    assert res["new_state"] == "CLEAN"
    assert res["mutations_performed"] is True
    assert Path(res["snapshot_directory"]).exists()

    # Verifica persistência do lock
    lock_state = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert lock_state["status"] == RuntimeState.CLEAN

    # Verifica livro-razão de auditoria
    audit_file = data_dir / ".recovery_audit.jsonl"
    assert audit_file.exists()
    lines = [json.loads(l) for l in audit_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["recovery_phase"] == "PREPARED"
    assert lines[1]["recovery_phase"] == "COMMITTED"
    assert lines[0]["audit_id"] == res["audit_id"]
    assert lines[1]["audit_id"] == res["audit_id"]


# =============================================================================
# 2. Recovery RUNNING residual (crash de processo anterior)
# =============================================================================
def test_02_recovery_running_residual(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(
        tmp_path, lock_status="RUNNING", lock_reason="processo_anterior_abrupto"
    )

    reason = "Recuperacao controlada de RUNNING residual apos crash do SO"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    assert res["previous_state"] == "RUNNING"
    assert res["new_state"] == "CLEAN"

    lock_state = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert lock_state["status"] == RuntimeState.CLEAN


# =============================================================================
# 3. Recovery UNKNOWN
# =============================================================================
def test_03_recovery_unknown(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(
        tmp_path, lock_status="UNKNOWN", lock_reason="estado_anomalo_anterior"
    )

    reason = "Recuperacao de estado anomalo UNKNOWN autorizada pelo administrador"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    assert res["new_state"] == "CLEAN"
    lock_state = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert lock_state["status"] == RuntimeState.CLEAN


# =============================================================================
# 4. Reason ausente
# =============================================================================
def test_04_reason_ausente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    with pytest.raises(ValueError, match="pelo menos 15 caracteres significativos"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason="",
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Garante que o lock permaneceu intocado
    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 5. Reason insuficiente (< 15 chars)
# =============================================================================
def test_05_reason_insuficiente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    with pytest.raises(ValueError, match="pelo menos 15 caracteres significativos"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason="   motivo curto   ",
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 6. Lock ocupado por outro processo
# =============================================================================
def test_06_lock_ocupado_por_outro_processo(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Retém trava física no kernel
    f_holder = open(lock_path, "r+b")
    try:
        if platform.system() == "Windows":
            import msvcrt
            f_holder.seek(0)
            msvcrt.locking(f_holder.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f_holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        reason = "Tentativa de recuperacao concorrente com lock bloqueado"
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            execute_authorized_recovery(
                data_dir=data_dir,
                lock_path=lock_path,
                reason=reason,
                backup_dir=backup_dir,
                fallback_dir=fallback_dir,
                repo_root=tmp_path,
            )

    finally:
        try:
            if platform.system() == "Windows":
                import msvcrt
                f_holder.seek(0)
                msvcrt.locking(f_holder.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f_holder.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        f_holder.close()

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 7. SQLite corrompido
# =============================================================================
def test_07_sqlite_corrompido(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Corrompe o arquivo do banco
    db_path = data_dir / "trading_bot.db"
    db_path.write_bytes(b"CORRUPTED_BINARY_DATA_NON_SQLITE_HEADER")

    reason = "Tentativa de recovery em banco corrompido"
    with pytest.raises(RuntimeError, match="Falha de integridade SQLite"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 8. JSONL corrompido
# =============================================================================
def test_08_jsonl_corrompido(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write('{"timestamp_ms": 12345, "incompleto": \n')

    reason = "Tentativa de recovery com jsonl corrompido"
    with pytest.raises(RuntimeError, match="linhas corrompidas"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 9. Colisão de payload
# =============================================================================
def test_09_colisao_de_payload(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Modifica o JSONL com um payload diferente para o mesmo evento, gerando divergência de hash canônico
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    divergent_events = [
        {
            "event_id": "ev_001",
            "timestamp_ms": 1710000001000,
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "window_id": "w1",
            "price": "99999.0",
            "qty": "100.0",
            "side": "SELL",
        }
    ]
    _create_test_jsonl(jsonl_path, divergent_events)

    reason = "Tentativa de recovery com colisao detectada entre destinos"
    with pytest.raises(RuntimeError, match="colisões de hash de payload"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 10. Archive com .tmp
# =============================================================================
def test_10_archive_com_tmp(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = archive_dir / "segment_20261012.jsonl.tmp"
    tmp_file.write_text('{"event": "unclosed_write"}\n', encoding="utf-8")

    reason = "Tentativa de recovery com arquivo temporario no archive"
    with pytest.raises(RuntimeError, match="arquivos temporários residuais .tmp"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 11. Falha de snapshot (isolamento de diretório incompleto)
# =============================================================================
def test_11_falha_de_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    def mock_copy2_fail(src, dst):
        raise OSError("Falha simulada de I/O na copia de arquivos do snapshot")

    monkeypatch.setattr(shutil, "copy2", mock_copy2_fail)

    reason = "Tentativa com falha simulada durante a copia do snapshot"
    with pytest.raises(RuntimeError, match="Falha crítica na criação do snapshot forense"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Lock deve permanecer DIRTY e diretório incompleto deve ter sido marcado
    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY
    incompletes = list(backup_dir.glob("*INCOMPLETE_FAILED*"))
    assert len(incompletes) >= 1


# =============================================================================
# 12. ENOSPC simulado
# =============================================================================
def test_12_enospc_simulado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Retorna espaço livre muito baixo (100 bytes)
    def mock_disk_usage(path):
        return (1000000000, 999999900, 100)

    monkeypatch.setattr(shutil, "disk_usage", mock_disk_usage)

    reason = "Tentativa em sistema de arquivos sem espaço livre"
    with pytest.raises(RuntimeError, match="ENOSPC_PREVENTED"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 13. Falha de auditoria
# =============================================================================
def test_13_falha_de_auditoria(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    import scripts.maintenance.recover_runtime_state as rec_module

    def mock_record_fail(audit_file, entry):
        raise IOError("Falha simulada de gravacao no livro-razao de auditoria")

    monkeypatch.setattr(rec_module, "record_recovery_audit", mock_record_fail)

    reason = "Tentativa onde a gravacao de auditoria falha preventivamente"
    with pytest.raises(IOError, match="Falha simulada de gravacao"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Transição para CLEAN não pode ter ocorrido
    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 14. Crash antes de PREPARED
# =============================================================================
def test_14_crash_antes_de_prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    import scripts.maintenance.recover_runtime_state as rec_module
    orig_create_snapshot = rec_module.create_recovery_snapshot

    def crash_after_snapshot(*args, **kwargs):
        res = orig_create_snapshot(*args, **kwargs)
        raise SystemExit("Simulated process kill before PREPARED write")

    monkeypatch.setattr(rec_module, "create_recovery_snapshot", crash_after_snapshot)

    reason = "Processo interrompido antes da gravacao da fase PREPARED"
    with pytest.raises(SystemExit):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Lock permanece DIRTY e auditoria não tem PREPARED
    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY
    audit_file = data_dir / ".recovery_audit.jsonl"
    assert not audit_file.exists() or audit_file.stat().st_size == 0


# =============================================================================
# 15. Crash após PREPARED
# =============================================================================
def test_15_crash_apos_prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Monkeypatch em transition_maintenance_to_clean para falhar, deixando PREPARED pendente
    def crash_transition(*args, **kwargs):
        raise SystemError("Simulated crash right after PREPARED before lock file mutation")

    monkeypatch.setattr(OperationalRuntimeLock, "transition_maintenance_to_clean", crash_transition)

    reason = "Crash abrupto ocorrido imediatamente apos PREPARED"
    with pytest.raises(SystemError):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # O preflight de startup DEVE detectar a transação incompleta e BLOQUEAR o startup (fail-closed)
    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "Transação de recovery incompleta/interrompida detectada" in msg


# =============================================================================
# 16. Crash durante transição para CLEAN
# =============================================================================
def test_16_crash_durante_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    def crash_in_persist(*args, **kwargs):
        raise IOError("Disk write error during lock state persistence")

    monkeypatch.setattr(OperationalRuntimeLock, "_persist_state", crash_in_persist)

    reason = "Falha de IO no momento exato de persistir CLEAN no lock"
    with pytest.raises(IOError, match="Disk write error"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Startup bloqueado fail-closed
    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# =============================================================================
# 17. Crash após CLEAN antes de COMMITTED
# =============================================================================
def test_17_crash_apos_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    import scripts.maintenance.recover_runtime_state as rec_module
    call_count = 0
    orig_record = rec_module.record_recovery_audit

    def record_fail_on_second(af, entry):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            orig_record(af, entry)
        else:
            raise SystemExit("Crash before COMMITTED write")

    monkeypatch.setattr(rec_module, "record_recovery_audit", record_fail_on_second)

    reason = "Simulacao de queda do processo entre escrita do lock e COMMITTED"
    with pytest.raises(SystemExit):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # O lock pode até estar com CLEAN, mas a auditoria possui PREPARED sem COMMITTED.
    # O preflight de startup DEVE RECUSAR o boot!
    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "PREPARED" in msg


# =============================================================================
# 18. Falha durante gravação de COMMITTED
# =============================================================================
def test_18_falha_durante_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    import scripts.maintenance.recover_runtime_state as rec_module
    call_count = 0
    orig_record = rec_module.record_recovery_audit

    def record_error_on_second(af, entry):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            orig_record(af, entry)
        else:
            raise RuntimeError("Excecao de rede/disco durante COMMITTED")

    monkeypatch.setattr(rec_module, "record_recovery_audit", record_error_on_second)

    reason = "Simulacao de erro generico durante gravacao de COMMITTED"
    with pytest.raises(RuntimeError, match="Excecao de rede/disco durante COMMITTED"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Fail-closed no startup
    ok, _ = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# =============================================================================
# 19. Execução concorrente de recovery
# =============================================================================
def test_19_execucao_concorrente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    op1 = OperationalRuntimeLock(lock_path=lock_path, holder_name="worker_1")
    op2 = OperationalRuntimeLock(lock_path=lock_path, holder_name="worker_2")

    # Worker 1 obtém manutenção
    with op1.maintenance_session(allowed_states=[RuntimeState.DIRTY]):
        # Worker 2 tenta adquirir manutenção concorrente
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            op2.acquire_maintenance()


# =============================================================================
# 20. Startup concorrente bloqueado durante manutenção
# =============================================================================
def test_20_startup_concorrente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    maint_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="maint_operator")
    normal_bot = OperationalRuntimeLock(lock_path=lock_path, holder_name="trading_bot_worker")

    with maint_lock.maintenance_session(allowed_states=[RuntimeState.DIRTY]):
        # Robô tenta adquirir lock de produção durante manutenção
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            normal_bot.acquire()


# =============================================================================
# 21. Estado CLEAN idempotente
# =============================================================================
def test_21_estado_clean_idempotente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="CLEAN")

    reason = "Tentativa em ambiente ja no estado CLEAN operacional"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "IDEMPOTENT_ALREADY_CLEAN"
    assert res["mutations_performed"] is False

    # Nenhum snapshot criado
    assert not backup_dir.exists() or len(list(backup_dir.glob("recovery_snapshot_*"))) == 0


# =============================================================================
# 22. Preservação de fixtures Git em fallback
# =============================================================================
def test_22_preservacao_de_fixtures_git(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Cria fixture conhecida de teste no fallback (registrada em KNOWN_TEST_FIXTURE_NAMES)
    fixture_file = fallback_dir / "eventos_20260307.json"
    fixture_file.write_text('{"event_type": "fixture_teste"}\n', encoding="utf-8")

    reason = "Recuperacao com fixture de teste conhecida em fallback"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    # Fixture preservada
    assert fixture_file.exists()


# =============================================================================
# 23. Preservação de WAL forense no snapshot
# =============================================================================
def test_23_preservacao_de_wal(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    wal_file = data_dir / "trading_bot.db-wal"
    wal_file.write_bytes(b"FORENSIC_WAL_DATA_RESIDUAL")
    shm_file = data_dir / "trading_bot.db-shm"
    shm_file.write_bytes(b"FORENSIC_SHM_DATA_RESIDUAL")

    reason = "Recuperacao preservando arquivos residuais de WAL e SHM"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    snap_dir = Path(res["snapshot_directory"])
    assert (snap_dir / "trading_bot.db-wal.raw").exists()
    assert (snap_dir / "trading_bot.db-shm.raw").exists()
    assert (snap_dir / "trading_bot.db-wal.raw").read_bytes() == b"FORENSIC_WAL_DATA_RESIDUAL"


# =============================================================================
# 24. Integridade dos hashes SHA256 do snapshot
# =============================================================================
def test_24_integridade_do_snapshot(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    reason = "Validacao rigorosa de hashes SHA256 do snapshot gravado"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    snap_dir = Path(res["snapshot_directory"])
    manifest_p = snap_dir / "snapshot_manifest.json"
    assert manifest_p.exists()

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    files_map = manifest_data["files"]

    for rel_path, file_info in files_map.items():
        actual_file = snap_dir / rel_path
        assert actual_file.exists()
        calc_sha = hashlib.sha256(actual_file.read_bytes()).hexdigest()
        assert calc_sha == file_info["sha256"]
        assert actual_file.stat().st_size == file_info["size_bytes"]


# =============================================================================
# 25. Imutabilidade dos dados originais (SQLite e JSONL inalterados)
# =============================================================================
def test_25_imutabilidade_dos_dados_originais(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    db_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"

    db_sha_before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    jsonl_sha_before = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    reason = "Verificacao de imutabilidade dos dados operacionais originais"
    res = execute_authorized_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        reason=reason,
        backup_dir=backup_dir,
        fallback_dir=fallback_dir,
        repo_root=tmp_path,
    )

    assert res["status"] == "RECOVERY_SUCCESS"
    db_sha_after = hashlib.sha256(db_path.read_bytes()).hexdigest()
    jsonl_sha_after = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    assert db_sha_after == db_sha_before
    assert jsonl_sha_after == jsonl_sha_before


# =============================================================================
# 26. Rejeição de manifesto inválido
# =============================================================================
def test_26_rejeicao_manifesto_invalido(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.write_text("INVALID_TRUNCATED_JSON{", encoding="utf-8")

    reason = "Tentativa em ambiente com manifesto estruturalmente corrompido"
    with pytest.raises(RuntimeError, match="Violação Anti-Bypass"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 27. Rejeição de UUID divergente
# =============================================================================
def test_27_rejeicao_uuid_divergente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(
        tmp_path, lock_status="DIRTY", env_uuid="UUID_AAA_1111"
    )

    # Altera UUID no arquivo de lock para divergir do manifesto
    lock_meta = {
        "status": "DIRTY",
        "holder": "test_holder",
        "pid": os.getpid(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "reason": "dirty_with_divergent_uuid",
        "environment_uuid": "UUID_BBB_2222",
    }
    payload = b"#" + json.dumps(lock_meta, indent=2).encode("utf-8")
    with open(lock_path, "wb") as lf:
        lf.write(payload)

    reason = "Tentativa com UUID divergente entre manifesto e lock"
    with pytest.raises(RuntimeError, match="Violação Anti-Bypass"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 28. Rejeição de fallback pendente
# =============================================================================
def test_28_rejeicao_fallback_pendente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Arquivo de fallback operacional NÃO fixture com evento não reconciliado
    unreconciled_file = fallback_dir / "unreconciled_events_01.jsonl"
    pending_event = {
        "event_id": "fb_pending_999",
        "timestamp_ms": 1719999999000,
        "event_type": "trade",
        "symbol": "BTCUSDT",
        "window_id": "w_pendente",
        "price": "70000.0",
        "qty": "1.5",
    }
    _create_test_jsonl(unreconciled_file, [pending_event])

    reason = "Tentativa com evento pendente no fallback operacional"
    with pytest.raises(RuntimeError, match="eventos potencialmente pendentes no fallback"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 29. Rejeição de evento sem ID / malformado no fallback
# =============================================================================
def test_29_rejeicao_evento_sem_id(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    # Cria arquivo no fallback com formato corrompido / sem campos identificadores
    bad_fallback = fallback_dir / "bad_fallback_event.jsonl"
    bad_fallback.write_text('{"foo": "bar"}\n', encoding="utf-8")

    reason = "Tentativa com arquivo de fallback contendo evento sem campos chave"
    with pytest.raises(RuntimeError, match="eventos sem identificador"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# 30. Rejeição de problemas estruturais em archives
# =============================================================================
def test_30_rejeicao_colisoes_entre_archives(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    corrupt_archive = archive_dir / "corrupted_archive_seg.jsonl"
    corrupt_archive.write_text('{"bad_json": \n', encoding="utf-8")

    reason = "Tentativa com segmento corrompido no diretorio de archives"
    with pytest.raises(RuntimeError, match="linhas corrompidas"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# =============================================================================
# ETAPA 7: TESTES DE HARDENING DO LIVRO-RAZÃO, SNAPSHOT E CONCORRÊNCIA (31 a 50)
# =============================================================================

# 31. Ledger com JSON inválido
def test_31_ledger_json_invalido(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(tmp_path, lock_status="DIRTY")
    audit_file = data_dir / ".recovery_audit.jsonl"
    audit_file.write_text("{bad_json_malformed\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "JSON inválido ou truncado" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "corrompido ou inconsistente" in msg

    reason = "Tentativa de recovery com ledger contendo json invalido"
    with pytest.raises(RuntimeError, match="corrompido ou inconsistente"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )


# 32. Ledger com última linha truncada
def test_32_ledger_ultima_linha_truncada(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry1 = {
        "audit_id": "a1",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery anterior",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap1",
        "snapshot_manifest_sha256": "h1",
        "previous_runtime_state": "DIRTY",
    }
    audit_file.write_text(json.dumps(entry1) + "\n" + '{"audit_id": "a2", "timestamp_utc":\n', encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "Linha 2" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# 33. Ledger com linha intermediária corrompida
def test_33_ledger_linha_intermediaria_corrompida(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry1 = {
        "audit_id": "a1",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery anterior",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap1",
        "snapshot_manifest_sha256": "h1",
        "previous_runtime_state": "DIRTY",
    }
    entry3 = {
        "audit_id": "a1",
        "timestamp_utc": "2026-03-08T00:01:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery anterior",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snap1",
        "snapshot_manifest_sha256": "h1",
        "final_state": "CLEAN",
    }
    content = json.dumps(entry1) + "\nGARBAGE_LINE_INTERMEDIATE\n" + json.dumps(entry3) + "\n"
    audit_file.write_text(content, encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "Linha 2" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# 34. PREPARED órfão antigo (COMMITTED posterior não mascara o órfão)
def test_34_prepared_orfao_antigo(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(tmp_path, lock_status="DIRTY")
    audit_file = data_dir / ".recovery_audit.jsonl"
    # Transação A ficou em PREPARED (órfão)
    entry_a = {
        "audit_id": "tx_A",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery interrompido",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snapA",
        "snapshot_manifest_sha256": "hA",
        "previous_runtime_state": "DIRTY",
    }
    # Transação B foi PREPARED e COMMITTED
    entry_b_prep = {
        "audit_id": "tx_B",
        "timestamp_utc": "2026-03-08T01:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery subsequente",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snapB",
        "snapshot_manifest_sha256": "hB",
        "previous_runtime_state": "DIRTY",
    }
    entry_b_com = {
        "audit_id": "tx_B",
        "timestamp_utc": "2026-03-08T01:01:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Recovery subsequente",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snapB",
        "snapshot_manifest_sha256": "hB",
        "final_state": "CLEAN",
    }
    audit_file.write_text(
        json.dumps(entry_a) + "\n" + json.dumps(entry_b_prep) + "\n" + json.dumps(entry_b_com) + "\n",
        encoding="utf-8",
    )

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is True
    assert res["has_pending_transaction"] is True
    assert "tx_A" in res["orphaned_prepared"]

    # Preflight de startup deve BLOQUEAR devido ao PREPARED órfão histórico
    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "tx_A" in msg

    # Recovery também deve bloquear
    with pytest.raises(RuntimeError, match="tx_A"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason="Tentativa com orfao historico presente no ledger",
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )


# 35. COMMITTED sem PREPARED
def test_35_committed_sem_prepared(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry_orphan_commit = {
        "audit_id": "tx_standalone",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Forced commit without prepare",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "final_state": "CLEAN",
    }
    audit_file.write_text(json.dumps(entry_orphan_commit) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "sem fase PREPARED prévia" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# 36. PREPARED duplicado
def test_36_prepared_duplicado(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry = {
        "audit_id": "tx_dup",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "previous_runtime_state": "DIRTY",
    }
    audit_file.write_text(json.dumps(entry) + "\n" + json.dumps(entry) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "Transação duplicada ou audit_id reutilizado" in res["error"]


# 37. COMMITTED duplicado
def test_37_committed_duplicado(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    prep = {
        "audit_id": "tx_cdup",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "previous_runtime_state": "DIRTY",
    }
    com = {
        "audit_id": "tx_cdup",
        "timestamp_utc": "2026-03-08T00:01:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "final_state": "CLEAN",
    }
    audit_file.write_text(json.dumps(prep) + "\n" + json.dumps(com) + "\n" + json.dumps(com) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "COMMITTED duplicado" in res["error"]


# 38. audit_id reutilizado
def test_38_audit_id_reutilizado(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    prep1 = {
        "audit_id": "tx_reused",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "previous_runtime_state": "DIRTY",
    }
    com1 = {
        "audit_id": "tx_reused",
        "timestamp_utc": "2026-03-08T00:01:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "COMMITTED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "final_state": "CLEAN",
    }
    prep2 = dict(prep1)
    prep2["timestamp_utc"] = "2026-03-08T00:02:00Z"
    audit_file.write_text(json.dumps(prep1) + "\n" + json.dumps(com1) + "\n" + json.dumps(prep2) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "audit_id reutilizado" in res["error"]


# 39. UUID divergente
def test_39_uuid_divergente(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry = {
        "audit_id": "tx_diff_uuid",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": "OTHER_ENV_UUID_999",
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "PREPARED",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "previous_runtime_state": "DIRTY",
    }
    audit_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "UUID divergente" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# 40. recovery_phase desconhecido
def test_40_recovery_phase_desconhecido(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    entry = {
        "audit_id": "tx_unknown_phase",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "IN_PROGRESS_TEST",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
        "previous_runtime_state": "DIRTY",
    }
    audit_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "desconhecida ou inválida" in res["error"]


# 41. Livro-razão vazio existente
def test_41_livro_razao_vazio_existente(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    audit_file.write_bytes(b"")

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["is_valid"] is False
    assert "vazio (0 bytes)" in res["error"]

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False


# 42. Livro-razão inexistente em ambiente sem histórico
def test_42_livro_razao_inexistente_ambiente_sem_historico(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    if audit_file.exists():
        audit_file.unlink()

    res = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    assert res["exists"] is False
    assert res["is_valid"] is True

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is True


# 43. Falha de leitura do ledger (fail-closed)
def test_43_falha_de_leitura_do_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    audit_file.write_text('{"foo": "bar"}\n', encoding="utf-8")

    orig_open = open

    def mock_broken_open(file, *args, **kwargs):
        if str(file) == str(audit_file):
            raise PermissionError("EACCES: Simulated permission error reading ledger")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_broken_open)

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "corrompido ou inconsistente" in msg


# 44. Falha de fsync do snapshot
def test_44_falha_fsync_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    def mock_failing_fsync(fd):
        raise OSError(5, "Simulated EIO on os.fsync")

    monkeypatch.setattr(os, "fsync", mock_failing_fsync)

    reason = "Tentativa de recovery com falha forçada de fsync no snapshot"
    with pytest.raises(RuntimeError, match="Falha crítica na criação do snapshot"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    incompletes = list(backup_dir.glob("*INCOMPLETE_FAILED*"))
    assert len(incompletes) >= 1
    assert OperationalRuntimeLock.read_persisted_state(lock_path)["status"] == RuntimeState.DIRTY


# 45. Symlink operacional inesperado
def test_45_symlink_operacional_inesperado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    suspicious_file = fallback_dir / "suspicious_fallback_link.jsonl"
    suspicious_file.write_text('{"event_id": "test"}\n', encoding="utf-8")

    orig_is_symlink = Path.is_symlink

    def mock_is_symlink(p):
        if p.name == "suspicious_fallback_link.jsonl":
            return True
        return orig_is_symlink(p)

    monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    reason = "Tentativa com link simbolico malicioso no fallback operacional"
    with pytest.raises(RuntimeError, match="Origem operacional insegura: link simbólico detectado"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    # Verifica também a proteção direta em create_recovery_snapshot
    with pytest.raises(RuntimeError, match="Origem operacional insegura: link simbólico detectado"):
        create_recovery_snapshot(
            data_dir=data_dir,
            lock_path=lock_path,
            backup_base_dir=backup_dir,
            env_uuid="test_uuid",
            operator_reason="Reason valid 15 chars",
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )


# 46. Snapshot incompleto isolado
def test_46_snapshot_incompleto_isolado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    def mock_copy_fail(*args, **kwargs):
        raise IOError("Disk full / medium error during file copy")

    monkeypatch.setattr(shutil, "copy2", mock_copy_fail)

    reason = "Recovery onde a copia de arquivos falha abruptamente"
    with pytest.raises(RuntimeError, match="Falha crítica na criação do snapshot"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )

    incompletes = list(backup_dir.glob("*INCOMPLETE_FAILED*"))
    assert len(incompletes) == 1
    clean_snapshots = [d for d in backup_dir.glob("recovery_snapshot_*") if "INCOMPLETE" not in d.name]
    assert len(clean_snapshots) == 0


# 47. Recovery bloqueado por histórico inconsistente
def test_47_recovery_bloqueado_por_historico_inconsistente(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, env_uuid = _setup_test_environment(tmp_path, lock_status="DIRTY")
    audit_file = data_dir / ".recovery_audit.jsonl"
    bad_entry = {
        "audit_id": "tx_bad",
        "timestamp_utc": "2026-03-08T00:00:00Z",
        "environment_uuid": env_uuid,
        "operator_reason": "Reason valid 15 chars",
        "recovery_phase": "CORRUPT_PHASE",
        "snapshot_directory": "snap",
        "snapshot_manifest_sha256": "h",
    }
    audit_file.write_text(json.dumps(bad_entry) + "\n", encoding="utf-8")

    reason = "Tentativa de recovery em ambiente com historico corrompido"
    with pytest.raises(RuntimeError, match="corrompido ou inconsistente"):
        execute_authorized_recovery(
            data_dir=data_dir,
            lock_path=lock_path,
            reason=reason,
            backup_dir=backup_dir,
            fallback_dir=fallback_dir,
            repo_root=tmp_path,
        )


# 48. Startup bloqueado por histórico inconsistente (mesmo com lock CLEAN)
def test_48_startup_bloqueado_por_historico_inconsistente(tmp_path: Path):
    data_dir, lock_path, _, _, env_uuid = _setup_test_environment(tmp_path, lock_status="CLEAN")
    audit_file = data_dir / ".recovery_audit.jsonl"
    audit_file.write_text("CORRUPTED_JSON_DATA\n", encoding="utf-8")

    ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert ok is False
    assert "corrompido ou inconsistente" in msg


# 49. Recovery concorrente usando subprocessos independentes
def test_49_recovery_concorrente_subprocessos(tmp_path: Path):
    data_dir, lock_path, backup_dir, fallback_dir, _ = _setup_test_environment(tmp_path, lock_status="DIRTY")

    child_code = f"""
import sys, time
from pathlib import Path
from common.runtime_lock import OperationalRuntimeLock, RuntimeState

lock_p = Path(r'{lock_path}')
op = OperationalRuntimeLock(lock_path=lock_p, holder_name="external_worker_proc")
with op.maintenance_session(allowed_states=[RuntimeState.DIRTY]):
    print("LOCK_HELD", flush=True)
    time.sleep(2.0)
print("LOCK_RELEASED", flush=True)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert "LOCK_HELD" in line

        reason = "Tentativa de recovery concorrente sob lock detido por subprocesso"
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            execute_authorized_recovery(
                data_dir=data_dir,
                lock_path=lock_path,
                reason=reason,
                backup_dir=backup_dir,
                fallback_dir=fallback_dir,
                repo_root=tmp_path,
            )
    finally:
        proc.wait(timeout=5)


# 50. Startup concorrente usando subprocessos independentes
def test_50_startup_concorrente_subprocessos(tmp_path: Path):
    data_dir, lock_path, _, _, _ = _setup_test_environment(tmp_path, lock_status="CLEAN")

    child_code = f"""
import sys, time
from pathlib import Path
from common.runtime_lock import OperationalRuntimeLock, RuntimeState

lock_p = Path(r'{lock_path}')
op = OperationalRuntimeLock(lock_path=lock_p, holder_name="maint_subprocess")
with op.maintenance_session(allowed_states=[RuntimeState.CLEAN]):
    print("LOCK_HELD", flush=True)
    time.sleep(2.0)
print("LOCK_RELEASED", flush=True)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert "LOCK_HELD" in line

        bot_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="trading_bot")
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            bot_lock.acquire(timeout=0.0)
    finally:
        proc.wait(timeout=5)

