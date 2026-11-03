# -*- coding: utf-8 -*-
"""
Suíte de Testes do Gate G4-D B3-B2-C2-A:
Governança e Infraestrutura da CLI de Reconciliação de Fallback.

Cobre os 30 cenários obrigatórios de governança, lock de manutenção, snapshot forense,
livro-razão de auditoria e integração com preflight/recovery em tmp_path.
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
    verify_startup_preflight,
    write_runtime_manifest,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    create_reconciliation_snapshot,
    record_reconciliation_audit,
    validate_reconciliation_audit_ledger,
)
from scripts.maintenance.bootstrap_runtime_lock import bootstrap_runtime_environment
from scripts.maintenance.recover_runtime_state import diagnose_runtime_recovery
from scripts.maintenance.reconcile_fallback_events import main as cli_main


def _setup_test_env(tmp_path: Path, state: RuntimeState = RuntimeState.CLEAN) -> Dict[str, Path]:
    """Configura ambiente temporário padronizado com SQLite e JSONL válidos."""
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
    write_runtime_manifest(manifest_path, env_uuid=env_uuid, holder_name="test_setup")

    # Inicia lock com o estado desejado
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

    # Cria tabela events no SQLite
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
    conn.commit()
    conn.close()

    # Cria JSONL ativo com 1 linha válida
    sample_evt = {
        "event_id": "ev_init_001",
        "timestamp_ms": 1710000000000,
        "event_type": "trade",
        "symbol": "BTCUSDT",
    }
    jsonl_path.write_text(json.dumps(sample_evt) + "\n", encoding="utf-8")

    # Arquivo de fallback de teste
    fb_file = fallback_dir / "eventos_20260307_teste.jsonl"
    sample_fb = {
        "event_id": "ev_fb_001",
        "timestamp_ms": 1710000001000,
        "event_type": "trade",
        "symbol": "BTCUSDT",
        "price": "65000",
    }
    fb_file.write_text(json.dumps(sample_fb) + "\n", encoding="utf-8")

    return {
        "data_dir": data_dir,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "fallback_dir": fallback_dir,
        "fallback_file": fb_file,
        "rec_audit_path": rec_audit_path,
        "env_uuid": env_uuid,
    }


# ==============================================================================
# 1. TESTES DE GOVERNANÇA DA CLI
# ==============================================================================

def test_01_dry_run_padrao(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """1. Execução sem flags mutantes roda em dry-run por padrão e sai com código 0."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    test_args = [
        "reconcile_fallback_events.py",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--fallback-dir", str(env["fallback_dir"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 0
    captured = capsys.readouterr()
    assert "RELATÓRIO DE AUDITORIA DE FALLBACK — MODO DRY-RUN" in captured.out


def test_02_apply_sem_confirmacao(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """2. --apply sem --confirm-apply é sumariamente rejeitado com Exit 2."""
    env = _setup_test_env(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--reason", "Justificativa valida de teste com mais de 15 chars",
        "--sqlite-path", str(env["sqlite_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "exige confirmação explícita via --confirm-apply" in captured.err


def test_03_apply_sem_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """3. --apply sem --reason é sumariamente rejeitado com Exit 2."""
    env = _setup_test_env(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--sqlite-path", str(env["sqlite_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "exige justificativa auditável (--reason)" in captured.err


def test_04_reason_insuficiente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """4. Reason com menos de 15 caracteres significativos é rejeitado com Exit 2."""
    env = _setup_test_env(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "curto   ",
        "--sqlite-path", str(env["sqlite_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "pelo menos 15 caracteres significativos" in captured.err


def test_05_flags_contraditorias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """5. --dry-run e --apply simultâneos são rejeitados com Exit 2."""
    env = _setup_test_env(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--dry-run",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste com mais de 15 chars",
        "--sqlite-path", str(env["sqlite_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as exc:
        cli_main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--dry-run e --apply são mutuamente exclusivos" in captured.err


def test_06_apply_explicitamente_desabilitado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """6. Na Fase C2-A, apply autorizado retorna bloqueio explícito APPLY_NOT_ENABLED_C2A e Exit 1."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao autorizada com justificativa longa e formal",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    assert "APPLY_NOT_ENABLED_C2A" in captured.err


# ==============================================================================
# 2. TESTES DE LOCK DE MANUTENÇÃO E ELEGIBILIDADE
# ==============================================================================

def test_07_lock_clean_elegivel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """7. Estado CLEAN é elegível para sessão de manutenção da CLI."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao autorizada em estado CLEAN para teste",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1


def test_08_lock_dirty_elegivel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """8. Estado DIRTY é elegível para manutenção do apply sem alterar DIRTY."""
    env = _setup_test_env(tmp_path, state=RuntimeState.DIRTY)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao de eventos sob lock DIRTY antes do recovery",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["persisted_lock_status"] == "DIRTY"
    assert OperationalRuntimeLock.read_persisted_state(env["lock_path"])["status"] == RuntimeState.DIRTY


def test_09_lock_unknown_elegivel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """9. Estado UNKNOWN é elegível para manutenção do apply preservando o lock."""
    env = _setup_test_env(tmp_path, state=RuntimeState.UNKNOWN)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao de eventos sob lock UNKNOWN para teste",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["persisted_lock_status"] == "UNKNOWN"
    assert OperationalRuntimeLock.read_persisted_state(env["lock_path"])["status"] == RuntimeState.UNKNOWN


def test_10_running_residual_elegivel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """10. Status RUNNING em disco sem processo ativo é detectado como RUNNING_RESIDUAL."""
    env = _setup_test_env(tmp_path)
    # Grava RUNNING no lock sem processo ativo no kernel
    env["lock_path"].write_text(
        json.dumps({
            "status": "RUNNING",
            "holder": "dead_worker",
            "pid": 99999,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
        }),
        encoding="utf-8",
    )
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao de eventos sob RUNNING residual",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "APPLY_NOT_ENABLED_C2A"


def test_11_processo_concorrente_ativo_bloqueia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """11. Processo retendo o lock do kernel bloqueia o apply com Exit 3."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    # Adquire trava ativa concorrente
    competing_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="active_robot_process")
    competing_lock.acquire(timeout=0.0)

    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao concorrente que deve ser bloqueada",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 3
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "CONCURRENCY_LOCK_CONFLICT"
    competing_lock.release()


def test_12_preservacao_dos_bytes_do_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """12. A sessão de manutenção da CLI não modifica nenhum byte do lock original."""
    env = _setup_test_env(tmp_path, state=RuntimeState.DIRTY)
    lock_raw_before = env["lock_path"].read_bytes()

    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao com verificacao de preservacao dos bytes",
        "--sqlite-path", str(env["sqlite_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()

    lock_raw_after = env["lock_path"].read_bytes()
    assert lock_raw_before == lock_raw_after


# ==============================================================================
# 3. TESTES DO SNAPSHOT FORENSE
# ==============================================================================

def test_13_snapshot_integro(tmp_path: Path):
    """13. create_reconciliation_snapshot gera backup SQLite, JSONL e manifesto com SHA256."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    backup_base = tmp_path / "backups" / "reconcile_snapshots"

    snap_dir, snap_hash = create_reconciliation_snapshot(
        data_dir=env["data_dir"],
        lock_path=env["lock_path"],
        backup_base_dir=backup_base,
        env_uuid=env["env_uuid"],
        operator_reason="Teste formal de snapshot de reconciliacao",
        source_fallback_file=env["fallback_file"],
    )

    assert snap_dir.exists()
    assert (snap_dir / "trading_bot.db").exists()
    assert (snap_dir / "eventos_fluxo.jsonl").exists()
    assert (snap_dir / "snapshot_manifest.json").exists()

    manifest_data = json.loads((snap_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["environment_uuid"] == env["env_uuid"]
    assert "trading_bot.db" in manifest_data["files"]
    assert "eventos_fluxo.jsonl" in manifest_data["files"]


def test_14_snapshot_incompleto_isolado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """14. Falha durante criação do snapshot isola diretório com prefixo .INCOMPLETE_FAILED_."""
    env = _setup_test_env(tmp_path)
    backup_base = tmp_path / "backups" / "reconcile_snapshots"

    # Simula falha ao copiar JSONL
    import shutil
    real_copy2 = shutil.copy2
    def broken_copy2(src, dst):
        if "eventos_fluxo.jsonl" in str(src):
            raise OSError(5, "Simulated I/O error on copy")
        return real_copy2(src, dst)

    monkeypatch.setattr(shutil, "copy2", broken_copy2)

    with pytest.raises(RuntimeError) as exc_info:
        create_reconciliation_snapshot(
            data_dir=env["data_dir"],
            lock_path=env["lock_path"],
            backup_base_dir=backup_base,
            env_uuid=env["env_uuid"],
            operator_reason="Teste de falha no snapshot",
            source_fallback_file=env["fallback_file"],
        )
    assert "Falha crítica na criação do snapshot forense" in str(exc_info.value)
    incompletes = list(backup_base.glob("*.INCOMPLETE_FAILED_*"))
    assert len(incompletes) >= 1


def test_15_enospc_simulado_aborta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """15. Falha preventiva de quota de disco (menos de 3x necessário) aborta preventivamente."""
    env = _setup_test_env(tmp_path)
    backup_base = tmp_path / "backups" / "reconcile_snapshots"

    import shutil
    def fake_disk_usage(path):
        return shutil._ntuple_diskusage(1000000000, 999999900, 100) # apenas 100 bytes livres

    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(RuntimeError) as exc_info:
        create_reconciliation_snapshot(
            data_dir=env["data_dir"],
            lock_path=env["lock_path"],
            backup_base_dir=backup_base,
            env_uuid=env["env_uuid"],
            operator_reason="Teste de ENOSPC preventivo",
        )
    assert "ENOSPC_PREVENTED" in str(exc_info.value)


def test_16_falha_de_fsync_no_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """16. Falha em fsync durante snapshot aborta com erro atômico."""
    env = _setup_test_env(tmp_path)
    backup_base = tmp_path / "backups" / "reconcile_snapshots"

    real_fsync = os.fsync
    def broken_fsync(fd):
        raise OSError(5, "Simulated fsync hardware failure")

    monkeypatch.setattr(os, "fsync", broken_fsync)

    with pytest.raises(RuntimeError) as exc_info:
        create_reconciliation_snapshot(
            data_dir=env["data_dir"],
            lock_path=env["lock_path"],
            backup_base_dir=backup_base,
            env_uuid=env["env_uuid"],
            operator_reason="Teste de falha fsync snapshot",
        )
    assert "Falha crítica na criação do snapshot" in str(exc_info.value)


def test_17_symlink_rejeitado_no_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """17. Links simbólicos em arquivos operacionais são rejeitados no snapshot."""
    env = _setup_test_env(tmp_path)
    backup_base = tmp_path / "backups" / "reconcile_snapshots"

    fake_symlink = env["fallback_dir"] / "symlink_fb.jsonl"
    fake_symlink.write_text("{}", encoding="utf-8")

    real_is_symlink = Path.is_symlink
    def mocked_is_symlink(self):
        if self.name == "symlink_fb.jsonl":
            return True
        return real_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", mocked_is_symlink)

    with pytest.raises(RuntimeError) as exc_info:
        create_reconciliation_snapshot(
            data_dir=env["data_dir"],
            lock_path=env["lock_path"],
            backup_base_dir=backup_base,
            env_uuid=env["env_uuid"],
            operator_reason="Teste symlink",
            source_fallback_file=fake_symlink,
        )
    assert "link simbólico detectado" in str(exc_info.value)


# ==============================================================================
# 4. TESTES DO LIVRO-RAZÃO DE RECONCILIAÇÃO
# ==============================================================================

def test_18_ledger_valido(tmp_path: Path):
    """18. validate_reconciliation_audit_ledger aprova ledger com pares PREPARED -> COMMITTED válidos."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
        "planned_events_count": 5,
    })
    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
        "applied_sqlite_count": 5,
    })

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is True
    assert res["completed_transactions_count"] == 1
    assert res["has_pending_transaction"] is False


def test_19_ledger_corrompido(tmp_path: Path):
    """19. validate_reconciliation_audit_ledger reprova ledger com JSON inválido ou truncado."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    audit_file.write_text('{"audit_id": "123", "reconciliation_phase": "PREP\n', encoding="utf-8")

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is False
    assert "JSON inválido ou truncado" in res["error"]


def test_20_prepared_orfao_detectado(tmp_path: Path):
    """20. Registro PREPARED sem COMMITTED é detectado como transação pendente/interrompida."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is False
    assert res["has_pending_transaction"] is True
    assert aid in res["orphaned_prepared"]
    assert "PREPARED sem COMMITTED" in res["error"]


def test_21_committed_orfao_rejeitado(tmp_path: Path):
    """21. Registro COMMITTED sem PREPARED prévio é rejeitado como inválido."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "COMMITTED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is False
    assert "sem respectivo PREPARED prévio" in res["error"]


def test_22_audit_id_duplicado_rejeitado(tmp_path: Path):
    """22. Reutilização de audit_id na mesma fase PREPARED é rejeitada."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })
    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": env["env_uuid"],
    })

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is False
    assert "duplicidade de audit_id" in res["error"]


def test_23_uuid_divergente_rejeitado(tmp_path: Path):
    """23. Registro no ledger com UUID diferente do ambiente é rejeitado."""
    env = _setup_test_env(tmp_path)
    audit_file = env["rec_audit_path"]
    aid = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record_reconciliation_audit(audit_file, {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": ts,
        "environment_uuid": str(uuid.uuid4()), # UUID divergente
    })

    res = validate_reconciliation_audit_ledger(audit_file, expected_environment_uuid=env["env_uuid"])
    assert res["is_valid"] is False
    assert "environment_uuid divergente" in res["error"]


# ==============================================================================
# 5. TESTES DE INTEGRAÇÃO COM STARTUP E RECOVERY
# ==============================================================================

def test_24_startup_bloqueado_por_ledger_inconsistente(tmp_path: Path):
    """24. Startup preflight bloqueia fail-closed se .reconciliation_audit.jsonl tiver PREPARED órfão."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })

    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert ok is False
    assert "livro-razão de reconciliação" in msg
    assert "PREPARED sem COMMITTED" in msg


def test_25_recovery_bloqueado_por_ledger_inconsistente(tmp_path: Path):
    """25. diagnose_runtime_recovery detecta inconsistência no ledger de reconciliação como conflito."""
    env = _setup_test_env(tmp_path, state=RuntimeState.DIRTY)
    aid = str(uuid.uuid4())
    record_reconciliation_audit(env["rec_audit_path"], {
        "audit_id": aid,
        "reconciliation_phase": "PREPARED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env["env_uuid"],
    })

    diag = diagnose_runtime_recovery(data_dir=env["data_dir"], lock_path=env["lock_path"])
    conflicts_str = " ".join(diag["conflicts"])
    assert "livro-razão de reconciliação" in conflicts_str
    assert "PREPARED sem COMMITTED" in conflicts_str


# ==============================================================================
# 6. TESTES DE NÃO-MUTAÇÃO E CONCORRÊNCIA
# ==============================================================================

def test_26_ausencia_de_mutacoes_financeiras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """26. Chamada de --apply em C2-A não adiciona linhas ao SQLite nem ao JSONL."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    sqlite_size_before = env["sqlite_path"].stat().st_size
    jsonl_size_before = env["jsonl_path"].stat().st_size

    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Reconciliacao com verificacao de zero mutacoes",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()

    assert env["sqlite_path"].stat().st_size == sqlite_size_before
    assert env["jsonl_path"].stat().st_size == jsonl_size_before


def test_27_compatibilidade_com_manifesto_h1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """27. Execução da CLI de reconciliação preserva o manifesto versão 1.1 inalterado."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    manifest_before = read_runtime_manifest(env["manifest_path"])

    test_args = [
        "reconcile_fallback_events.py",
        "--dry-run",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()

    manifest_after = read_runtime_manifest(env["manifest_path"])
    assert manifest_before == manifest_after


def test_28_compatibilidade_com_bootstrap(tmp_path: Path):
    """28. Ambiente recém-inicializado com bootstrap passa normalmente pela auditoria de reconciliação."""
    data_dir = tmp_path / "boot_dados"
    bootstrap_runtime_environment(data_dir=data_dir, holder_name="test_boot")
    ok, msg = verify_startup_preflight(data_dir=data_dir)
    assert ok is True


def test_29_concorrencia_multiprocesso(tmp_path: Path):
    """29. Duas instâncias do OperationalRuntimeLock competem sob exclusividade mútua no SO."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    lock1 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="proc_1")
    lock1.acquire(timeout=0.0)

    lock2 = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="proc_2")
    with pytest.raises(RuntimeLockError) as exc_info:
        lock2.acquire(timeout=0.0)
    assert "Conflito de exclusividade" in str(exc_info.value)
    lock1.release()


def test_30_cli_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """30. Flag --json-output emite JSON estruturado válido tanto em dry-run quanto em apply desabilitado."""
    env = _setup_test_env(tmp_path, state=RuntimeState.CLEAN)
    test_args = [
        "reconcile_fallback_events.py",
        "--dry-run",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--fallback-dir", str(env["fallback_dir"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["mode"] == "DRY_RUN_READ_ONLY"
    assert "total_records_read" in data
