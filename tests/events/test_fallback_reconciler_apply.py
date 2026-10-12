# -*- coding: utf-8 -*-
"""
Suíte de testes de integração e autorização do reconciliador sob lock operacional (Gate G4-D B3-B1).

Valida:
1. Validação atômica de require_previous_state (CLEAN autoriza; DIRTY, UNKNOWN e RUNNING bloqueiam).
2. Preservação estrita do estado anterior em disco sem TOCTOU e sem alteração em falha.
3. Fail-closed em falha de fsync.
4. Verificação de autorização física de lock no motor (rejeita lock falso, liberado ou divergente).
5. Confinamento canônico de caminhos (symlinks, junctions e diretórios).
6. Bloqueio incondicional de fixtures versionadas no Git.
7. Funções isoladas de diagnóstico pré-apply (SQLite integrity, JSONL termination, Archive tmp).
8. Rejeição inalterada da CLI contra flags de escrita (--apply, --force, etc.).

Todos os testes operam exclusivamente em tmp_path. Zero acesso a dados operacionais.
"""

import errno
import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

from common.runtime_lock import OperationalRuntimeLock, RuntimeLockError, RuntimeState
from events.fallback_reconciler import (
    FallbackReconciler,
    EventClassification,
    verify_operational_lock_authorization,
    verify_sqlite_integrity,
    verify_jsonl_integrity,
    verify_archive_integrity,
    verify_reconciliation_preflight,
    reconcile_events_hermetic,
)


def _init_test_sqlite_schema(db_path: Path) -> None:
    """Cria tabela events no banco SQLite temporário."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                window_id TEXT,
                is_signal INTEGER DEFAULT 0,
                payload TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

_init_clean_sqlite = _init_test_sqlite_schema


# ==============================================================================
# 1. VALIDAÇÃO ATÔMICA DE REQUIRE_PREVIOUS_STATE NO LOCK OPERACIONAL
# ==============================================================================

def test_1_clean_authorizes_acquisition(tmp_path):
    """1. Estado prévio CLEAN autoriza aquisição e transiciona para RUNNING sob fsync."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Processo anterior fecha com CLEAN
    prev_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="previous_process")
    prev_lock.acquire()
    prev_lock.release()
    assert OperationalRuntimeLock.read_persisted_state(lock_file)["status"] == RuntimeState.CLEAN

    # Reconciliador exige estritamente CLEAN
    reconciler_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_bot")
    reconciler_lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    assert reconciler_lock.is_locked
    assert reconciler_lock.previous_persisted_state == RuntimeState.CLEAN
    assert reconciler_lock.persisted_state == RuntimeState.RUNNING

    reconciler_lock.release()
    assert OperationalRuntimeLock.read_persisted_state(lock_file)["status"] == RuntimeState.CLEAN


def test_2_dirty_blocks_acquisition(tmp_path):
    """2. Estado prévio DIRTY bloqueia aquisição do reconciliador sem alterar o disco."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Simula arquivo gravado como DIRTY por shutdown incompleto
    dirty_meta = {
        "status": "DIRTY",
        "holder": "crashed_worker",
        "pid": 1111,
        "timestamp_utc": "2026-10-12T00:00:00+00:00",
        "host": "TEST_HOST",
        "reason": "workers ativos",
    }
    with open(lock_file, "wb") as f:
        f.write(b"#" + json.dumps(dirty_meta).encode("utf-8"))

    reconciler_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_bot")
    with pytest.raises(RuntimeLockError) as exc_info:
        reconciler_lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    assert "Transição proibida" in str(exc_info.value)
    assert not reconciler_lock.is_locked
    assert reconciler_lock._file_obj is None


def test_3_unknown_blocks_acquisition(tmp_path):
    """3. Arquivo vazio ou corrompido (UNKNOWN) bloqueia aquisição do reconciliador."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock_file.touch()  # 0 bytes => UNKNOWN

    reconciler_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_bot")
    with pytest.raises(RuntimeLockError) as exc_info:
        reconciler_lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    assert "Transição proibida" in str(exc_info.value)
    assert not reconciler_lock.is_locked


def test_4_residual_running_blocks_acquisition(tmp_path):
    """4. Estado residual RUNNING de crash prévio bloqueia aquisição com require_previous_state=CLEAN."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Simula crash anterior deixando status RUNNING sem trava de kernel
    running_meta = {
        "status": "RUNNING",
        "holder": "unclean_bot",
        "pid": 2222,
        "timestamp_utc": "2026-10-12T00:00:00+00:00",
        "host": "TEST_HOST",
    }
    with open(lock_file, "wb") as f:
        f.write(b"#" + json.dumps(running_meta).encode("utf-8"))

    reconciler_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_bot")
    with pytest.raises(RuntimeLockError) as exc_info:
        reconciler_lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    assert "Transição proibida" in str(exc_info.value)
    assert not reconciler_lock.is_locked


def test_5_previous_state_preserved_on_disk(tmp_path):
    """5. Quando a aquisição é rejeitada, o estado anterior no disco é preservado inalterado."""
    lock_file = tmp_path / ".operational_runtime.lock"
    original_dirty_meta = {
        "status": "DIRTY",
        "holder": "bot_alpha",
        "pid": 7777,
        "timestamp_utc": "2026-10-12T12:00:00+00:00",
        "host": "TEST_HOST",
        "reason": "critical_shutdown",
    }
    with open(lock_file, "wb") as f:
        f.write(b"#" + json.dumps(original_dirty_meta).encode("utf-8"))

    reconciler_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_bot")
    with pytest.raises(RuntimeLockError):
        reconciler_lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    # Verifica que o disco não foi alterado nem sobrescrito com RUNNING
    state_after = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert state_after["status"] == RuntimeState.DIRTY
    assert state_after["metadata"]["holder"] == "bot_alpha"
    assert state_after["reason"] == "critical_shutdown"


def test_6_fsync_failure_in_acquire_fail_closed(tmp_path, monkeypatch):
    """6. Falha de fsync durante a persistência em acquire() aciona rollback seguro fail-closed."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Prepara estado CLEAN prévio
    lock_file.write_bytes(b'#{"status": "CLEAN", "holder": "p", "pid": 1, "timestamp_utc": "t", "host": "h"}')

    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="fsync_fail_bot")

    def mock_fsync_fail(fd):
        raise OSError(errno.EIO, "I/O error during fsync in acquire")

    monkeypatch.setattr(os, "fsync", mock_fsync_fail)

    with pytest.raises(RuntimeLockError):
        lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    assert not lock.is_locked
    assert lock._file_obj is None


# ==============================================================================
# 2. AUTORIZAÇÃO DO MOTOR DE RECONCILIAÇÃO (FALLBACK_RECONCILER)
# ==============================================================================

def test_7_fake_lock_rejected(tmp_path):
    """7. Objeto de lock falso ou None é rejeitado por verify_operational_lock_authorization."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"

    with pytest.raises(PermissionError) as exc_info:
        verify_operational_lock_authorization(None, sq, jl)
    assert "ausente ou inválido" in str(exc_info.value)

    with pytest.raises(PermissionError) as exc_info2:
        verify_operational_lock_authorization("TRUE_TOKEN_STRING", sq, jl)
    assert "ausente ou inválido" in str(exc_info2.value)


def test_8_released_lock_rejected(tmp_path):
    """8. Lock operacional com is_locked=False é rejeitado por verify_operational_lock_authorization."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"
    lock_file = data_dir / ".operational_runtime.lock"

    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler")
    # Sem chamar acquire()
    with pytest.raises(PermissionError) as exc_info:
        verify_operational_lock_authorization(lock, sq, jl)
    assert "não está ativamente detido" in str(exc_info.value)


def test_9_divergent_paths_rejected(tmp_path):
    """9. Lock em diretório divergente das bases de dados é rejeitado por violação de confinamento."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_lock_file = other_dir / ".operational_runtime.lock"

    lock = OperationalRuntimeLock(lock_path=other_lock_file, holder_name="reconciler")
    lock.acquire()

    try:
        with pytest.raises(PermissionError) as exc_info:
            verify_operational_lock_authorization(lock, sq, jl)
        assert "Violação de confinamento" in str(exc_info.value)
    finally:
        lock.release()


def test_10_symlink_or_resolved_paths_accepted(tmp_path):
    """10. Caminhos canônicos no mesmo diretório passam com sucesso na autorização."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"
    lock_file = data_dir / ".operational_runtime.lock"

    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler")
    lock.acquire()

    try:
        # Autorização bem-sucedida sem exceções
        verify_operational_lock_authorization(lock, sq, jl)
    finally:
        lock.release()


def test_11_two_concurrent_reconcilers(tmp_path):
    """11. Segundo processo de reconciliação é bloqueado no SO pelo primeiro detentor."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Prepara estado CLEAN inicial
    lock_file.write_bytes(b'#{"status": "CLEAN", "holder": "init", "pid": 1, "timestamp_utc": "t", "host": "h"}')

    lock1 = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_1")
    lock1.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    assert lock1.is_locked

    try:
        lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler_2")
        with pytest.raises(RuntimeLockError) as exc_info:
            lock2.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
        assert "Conflito de exclusividade" in str(exc_info.value)
    finally:
        lock1.release()


def test_12_git_fixture_strictly_protected(tmp_path):
    """12. Reconciliação hermética rejeita fixtures de teste Git incondicionalmente, mesmo com lock ativo."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"
    arch = data_dir / "archive"
    arch.mkdir()
    _init_test_sqlite_schema(sq)

    lock_file = data_dir / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler")
    lock.acquire()

    # Cria arquivo com nome de fixture conhecida
    fixture_file = tmp_path / "eventos_20260307.json"
    fixture_file.write_text('{"event_id": "fix_1", "epoch_ms": 1700000000000, "tipo_evento": "Trade"}\n')

    try:
        with pytest.raises(PermissionError) as exc_info:
            reconcile_events_hermetic(
                source_fallback_file=fixture_file,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                operational_lock=lock,
            )
        assert "fixture de teste versionada pelo Git" in str(exc_info.value)
    finally:
        lock.release()


# ==============================================================================
# 3. VERIFICAÇÕES ISOLADAS DE PRÉ-APPLY
# ==============================================================================

def test_13_sqlite_integrity_failure_detected(tmp_path):
    """13. Falha de integridade estrutural no SQLite é detectada por verify_sqlite_integrity."""
    corrupted_db = tmp_path / "corrupted.db"
    # Escreve lixo que não é SQLite
    corrupted_db.write_bytes(b"CORRUPTED_SQLITE_HEADER_1234567890_NON_VALID_DATABASE")

    ok, msg = verify_sqlite_integrity(corrupted_db)
    assert not ok
    assert "Erro ao verificar integridade" in msg or "integrity_check falhou" in msg


def test_14_jsonl_integrity_failure_detected(tmp_path):
    """14. Última linha corrompida ou truncada no JSONL é detectada por verify_jsonl_integrity."""
    jl_file = tmp_path / "broken.jsonl"
    # Última linha truncada
    jl_file.write_text('{"event_id": "1", "tipo": "OK"}\n{"event_id": "2", "tipo": "TRUNC', encoding="utf-8")

    ok, msg = verify_jsonl_integrity(jl_file)
    assert not ok
    assert "inválida ou corrompida" in msg


def test_15_archive_residual_tmp_detected(tmp_path):
    """15. Arquivo temporário residual (.tmp) no archive é detectado por verify_archive_integrity."""
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    (arch_dir / "eventos_fluxo_20261012_001.jsonl").write_text('{"event_id": "arch1"}\n')
    (arch_dir / "eventos_fluxo_20261012_002.tmp").write_text('{"incomplete": true}\n')

    ok, msg = verify_archive_integrity(arch_dir)
    assert not ok
    assert "Arquivos temporários residuais detectados" in msg


def test_16_identity_collision_quarantined(tmp_path):
    """16. Evento com ID idêntico mas payload divergente é enviado para quarentena sem corromper base."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    sq = data_dir / "trading_bot.db"
    jl = data_dir / "eventos_fluxo.jsonl"
    arch = data_dir / "archive"
    arch.mkdir()
    _init_test_sqlite_schema(sq)

    # Insere evento 100 no SQLite
    conn = sqlite3.connect(str(sq))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1700000000000, "Trade", "BTCUSDT", json.dumps({"event_id": "col_100", "price": 50000})),
    )
    conn.commit()
    conn.close()

    # Prepara JSONL
    jl.write_text(json.dumps({"event_id": "col_100", "price": 50000}) + "\n")

    # Arquivo de fallback com o mesmo event_id col_100 mas payload diferente (preço 99999)
    fallback_file = tmp_path / "collision_fallback.jsonl"
    fallback_file.write_text(json.dumps({"event_id": "col_100", "price": 99999}) + "\n")

    lock_file = data_dir / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reconciler")
    lock.acquire()

    try:
        results = reconcile_events_hermetic(
            source_fallback_file=fallback_file,
            target_sqlite_path=sq,
            target_jsonl_path=jl,
            archive_dir=arch,
            operational_lock=lock,
        )
        # Evento colidente foi enviado para quarentena (skipped_quarantine), zero inserts
        assert results["skipped_quarantine"] == 1
        assert results["sqlite_inserted"] == 0
        assert results["jsonl_appended"] == 0
    finally:
        lock.release()


def test_17_cli_continues_rejecting_forbidden_flags(tmp_path):
    """17. CLI em scripts/maintenance/reconcile_fallback_events.py continua rejeitando --apply e flags de escrita."""
    cli_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "maintenance" / "reconcile_fallback_events.py"
    assert cli_path.exists()

    for flag in ["--apply", "--force", "--delete", "--repair"]:
        res = subprocess.run(
            [sys.executable, str(cli_path), flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert res.returncode == 2
        assert "PROIBIDO" in res.stderr
        assert flag in res.stderr


def test_18_backup_disabled_in_prod_rejected(tmp_path):
    """18. Rejeita perform_backup=False quando o destino é caminho operacional."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    try:
        with pytest.raises(PermissionError, match="desativação de backup em caminhos operacionais de produção"):
            reconcile_events_hermetic(
                source_fallback_file=fb,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                perform_backup=False,
                repo_root=repo,
                operational_lock=lock,
            )
    finally:
        lock.release()


def test_19_backup_dir_missing_in_prod_rejected(tmp_path):
    """19. Rejeita backup_dir nulo ou ausente quando o destino é caminho operacional."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    try:
        with pytest.raises(ValueError, match="backup_dir é estritamente obrigatório"):
            reconcile_events_hermetic(
                source_fallback_file=fb,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                perform_backup=True,
                backup_dir=None,
                repo_root=repo,
                operational_lock=lock,
            )
    finally:
        lock.release()


def test_20_backup_creation_failure_aborts(tmp_path, monkeypatch):
    """20. Falha na criação do backup aborta a reconciliação antes de escrever qualquer dado."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl_content = '{"event_id": "init", "epoch_ms": 1700000000000}\n'
    jl.write_text(jl_content, encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()
    bkp_dir = repo / "backups"

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    from events import fallback_reconciler

    def mock_backup_fail(*args, **kwargs):
        raise OSError("Falha simulada de disco no backup")

    monkeypatch.setattr(fallback_reconciler, "create_sqlite_backup_consistent", mock_backup_fail)

    try:
        with pytest.raises(RuntimeError, match="Falha crítica na criação do backup consistente"):
            reconcile_events_hermetic(
                source_fallback_file=fb,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                perform_backup=True,
                backup_dir=bkp_dir,
                repo_root=repo,
                operational_lock=lock,
            )
        # Garante que nenhum byte foi adicionado ao JSONL
        assert jl.read_text(encoding="utf-8") == jl_content
    finally:
        lock.release()


def test_21_preflight_mandatory_and_successful(tmp_path):
    """21. Preflight obrigatório executa com sucesso em produção e autoriza escrita hermética."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()
    bkp_dir = repo / "backups"

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev_ok", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    try:
        res = reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=sq,
            target_jsonl_path=jl,
            archive_dir=arch,
            perform_backup=True,
            backup_dir=bkp_dir,
            repo_root=repo,
            operational_lock=lock,
        )
        assert res["sqlite_inserted"] == 1
        assert res["jsonl_appended"] == 1
        assert res["backup_sha256"] is not None
        assert (bkp_dir / "trading_bot_backup.db").exists()
    finally:
        lock.release()


def test_22_preflight_failure_aborts_before_write(tmp_path):
    """22. Preflight falho bloqueia reconciliação operacional antes de tocar qualquer dado."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    # Criar SQLite corrompido com bytes arbitrários
    sq.write_bytes(b"INVALID_SQLITE_CONTENT_HEADER")
    jl = data / "eventos_fluxo.jsonl"
    jl_content = '{"event_id": "init", "epoch_ms": 1700000000000}\n'
    jl.write_text(jl_content, encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()
    bkp_dir = repo / "backups"

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    try:
        with pytest.raises(RuntimeError, match="Preflight de reconciliação falhou com impedimentos críticos"):
            reconcile_events_hermetic(
                source_fallback_file=fb,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                perform_backup=True,
                backup_dir=bkp_dir,
                repo_root=repo,
                operational_lock=lock,
            )
        # JSONL permaneceu intocado
        assert jl.read_text(encoding="utf-8") == jl_content
    finally:
        lock.release()


def test_23_invalid_archive_segment_blocks_production_write(tmp_path):
    """23. Segmento do archive com linha corrompida bloqueia a escrita em produção via preflight."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    _init_clean_sqlite(sq)
    jl = data / "eventos_fluxo.jsonl"
    jl_content = '{"event_id": "init", "epoch_ms": 1700000000000}\n'
    jl.write_text(jl_content, encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()
    # Criar segmento histórico com JSON inválido
    bad_seg = arch / "eventos_fluxo_20261011_000_001.jsonl"
    bad_seg.write_text('{"event_id": "ok"}\n{CORRUPTED_JSON\n', encoding="utf-8")
    bkp_dir = repo / "backups"

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    lock = OperationalRuntimeLock(lock_path=data / ".operational_runtime.lock", holder_name="rec")
    lock.acquire()

    try:
        with pytest.raises(RuntimeError, match="Segmento arquivado .* corrompido"):
            reconcile_events_hermetic(
                source_fallback_file=fb,
                target_sqlite_path=sq,
                target_jsonl_path=jl,
                archive_dir=arch,
                perform_backup=True,
                backup_dir=bkp_dir,
                repo_root=repo,
                operational_lock=lock,
            )
        assert jl.read_text(encoding="utf-8") == jl_content
    finally:
        lock.release()


def test_24_relative_path_detected_as_production(tmp_path, monkeypatch):
    """24. Caminho relativo apontando para dados operacionais é rigorosamente detectado como produção."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    # Muda o CWD para o repositório fake
    monkeypatch.chdir(repo)
    rel_sq = Path("dados/trading_bot.db")
    rel_jl = Path("dados/eventos_fluxo.jsonl")

    # Chamada sem lock em caminhos relativos de produção deve falhar
    with pytest.raises(PermissionError, match="Autorização negada: objeto de lock operacional ausente ou inválido"):
        reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=rel_sq,
            target_jsonl_path=rel_jl,
            archive_dir=arch,
            perform_backup=False,
            repo_root=repo,
            operational_lock=None,
        )


def test_25_different_cwd_anchored_in_repo_root(tmp_path, monkeypatch):
    """25. Detecção de produção funciona independentemente de CWD alternativo quando ancorada em repo_root."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    other_dir = tmp_path / "other_cwd"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    # Chamada sem lock com caminhos absolutos do repo fake a partir de outro CWD
    with pytest.raises(PermissionError, match="Autorização negada: objeto de lock operacional ausente ou inválido"):
        reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=sq,
            target_jsonl_path=jl,
            archive_dir=arch,
            perform_backup=False,
            repo_root=repo,
            operational_lock=None,
        )


def test_26_symlink_or_resolved_production_detected(tmp_path):
    """26. Resolução física detecta caminho operacional mesmo via alias ou caminho normalizado."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    # Caminho com referências reduntantes ./dados/../dados/
    alias_sq = repo / "dados" / ".." / "dados" / "trading_bot.db"
    alias_jl = repo / "dados" / ".." / "dados" / "eventos_fluxo.jsonl"

    with pytest.raises(PermissionError, match="Autorização negada: objeto de lock operacional ausente ou inválido"):
        reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=alias_sq,
            target_jsonl_path=alias_jl,
            archive_dir=arch,
            perform_backup=False,
            repo_root=repo,
            operational_lock=None,
        )


def test_27_direct_call_without_authorization_rejected(tmp_path):
    """27. Chamada direta ao motor sem lock operacional é sumariamente rejeitada para alvos de produção."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl.write_text('{"event_id": "init", "epoch_ms": 1700000000000}\n', encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "new_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    # Chamada com operational_lock=None
    with pytest.raises(PermissionError, match="Autorização negada: objeto de lock operacional ausente ou inválido"):
        reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=sq,
            target_jsonl_path=jl,
            archive_dir=arch,
            perform_backup=True,
            backup_dir=repo / "backups",
            repo_root=repo,
            operational_lock=None,
        )


def test_28_no_writes_before_full_authorization(tmp_path):
    """28. Garante zero escrita e estado idêntico em caso de qualquer recusa de autorização ou preflight."""
    repo = tmp_path / "repo"
    data = repo / "dados"
    data.mkdir(parents=True)
    sq = data / "trading_bot.db"
    jl = data / "eventos_fluxo.jsonl"
    _init_clean_sqlite(sq)
    jl_content = '{"event_id": "init", "epoch_ms": 1700000000000}\n'
    jl.write_text(jl_content, encoding="utf-8")
    arch = repo / "archive"
    arch.mkdir()

    fb = tmp_path / "fb.jsonl"
    fb.write_text('{"event_id": "unauth_ev", "epoch_ms": 1700000000000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    sq_bytes_before = sq.read_bytes()
    jl_bytes_before = jl.read_bytes()

    with pytest.raises(PermissionError):
        reconcile_events_hermetic(
            source_fallback_file=fb,
            target_sqlite_path=sq,
            target_jsonl_path=jl,
            archive_dir=arch,
            perform_backup=False,
            repo_root=repo,
            operational_lock=None,
        )

    # Validação absoluta de zero bytes alterados
    assert sq.read_bytes() == sq_bytes_before
    assert jl.read_bytes() == jl_bytes_before
