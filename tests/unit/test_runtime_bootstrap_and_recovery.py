# -*- coding: utf-8 -*-
"""
Suíte de Testes para Bootstrap Controlado e Startup Fail-Closed (Gate G4-D B3-B2-A).

Cobre exaustivamente:
1. Bootstrap autorizado.
2. Bootstrap repetido bloqueado.
3. CLEAN autoriza startup.
4. DIRTY bloqueia startup.
5. UNKNOWN bloqueia startup.
6. RUNNING residual bloqueia (crash detectado).
7. Lock deletado bloqueia (anti-bypass).
8. Manifesto deletado bloqueia (anti-bypass).
9. Dados SQLite inválidos bloqueiam.
10. JSONL inválido bloqueia.
11. Falha de fsync tratada (fail-closed).
12. Processos concorrentes bloqueados.
13. Ciclo de vida de instalação nova.
14. Equivalência de caminhos canônicos.

Todos os testes rodam estritamente em tmp_path. Zero escrita ou acesso a dados de produção.
Nenhum recovery com escrita é executado nesta etapa.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    MANIFEST_FILENAME,
    write_runtime_manifest,
    read_runtime_manifest,
    verify_environment_anti_bypass,
    verify_startup_preflight,
    bootstrap_runtime_environment,
)
from scripts.maintenance.bootstrap_runtime_lock import run_diagnostic


def _create_healthy_sqlite(db_path: Path) -> None:
    """Cria um banco SQLite mínimo e 100% íntegro com schema operacional."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            epoch_ms INTEGER,
            tipo_evento TEXT,
            symbol TEXT,
            canonical_hash TEXT,
            payload_json TEXT
        )
        """
    )
    cur.execute(
        "INSERT INTO event_log (event_id, epoch_ms, tipo_evento, symbol, canonical_hash, payload_json) "
        "VALUES ('evt_init_1', 1710000000000, 'trade', 'BTCUSDT', 'hash_init', '{\"price\": 60000.0}')"
    )
    conn.commit()
    conn.close()


def _create_healthy_jsonl(jsonl_path: Path) -> None:
    """Cria um arquivo JSONL mínimo e íntegro com terminação limpa."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    event_data = {
        "event_id": "evt_init_1",
        "timestamp_ms": 1710000000000,
        "type": "trade",
        "symbol": "BTCUSDT",
        "price": 60000.0,
    }
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")


def test_01_authorized_bootstrap(tmp_path: Path):
    """1. Bootstrap autorizado em ambiente com dados íntegros: cria manifesto e lock CLEAN."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_healthy_sqlite(sqlite_path)
    _create_healthy_jsonl(jsonl_path)

    orig_sqlite_size = sqlite_path.stat().st_size
    orig_jsonl_size = jsonl_path.stat().st_size

    res = bootstrap_runtime_environment(data_dir=data_dir)

    assert res["success"] is True
    assert res["status"] == "CLEAN"
    assert "environment_uuid" in res

    # 1. Manifesto criado com UUID e sem credenciais
    manifest_path = data_dir / MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest_data = read_runtime_manifest(manifest_path)
    assert manifest_data is not None
    assert manifest_data["environment_uuid"] == res["environment_uuid"]
    assert "secret" not in manifest_data
    assert "token" not in manifest_data
    assert "api_key" not in manifest_data

    # 2. Lock operacional criado com status CLEAN
    lock_path = data_dir / ".operational_runtime.lock"
    assert lock_path.exists()
    lock_state = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert lock_state["status"] == RuntimeState.CLEAN

    # 3. Nenhuma escrita em SQLite ou JSONL
    assert sqlite_path.stat().st_size == orig_sqlite_size
    assert jsonl_path.stat().st_size == orig_jsonl_size


def test_02_repeated_bootstrap_blocked(tmp_path: Path):
    """2. Bootstrap repetido em ambiente previamente inicializado é estritamente bloqueado."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    # Primeiro bootstrap: autorizado
    res1 = bootstrap_runtime_environment(data_dir=data_dir)
    assert res1["success"] is True

    # Segundo bootstrap: bloqueado fail-closed
    with pytest.raises(RuntimeLockError, match="Bootstrap repetido bloqueado"):
        bootstrap_runtime_environment(data_dir=data_dir)

    # UUID do manifesto permanece inalterado
    manifest_data = read_runtime_manifest(data_dir / MANIFEST_FILENAME)
    assert manifest_data["environment_uuid"] == res1["environment_uuid"]


def test_03_clean_authorizes_startup(tmp_path: Path):
    """3. Lock em estado CLEAN autoriza startup e aquisição para RUNNING."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is True
    assert "Preflight de startup 100% aprovado" in msg

    # Aquisição do lock com transição para RUNNING
    lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="test_runner")
    lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)
    assert lock.is_locked is True

    state_during_run = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert state_during_run["status"] == RuntimeState.RUNNING

    # Liberação graciosa volta para CLEAN
    lock.release()
    assert lock.is_locked is False
    state_after_release = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert state_after_release["status"] == RuntimeState.CLEAN


def test_04_dirty_blocks_startup(tmp_path: Path):
    """4. Lock em estado DIRTY bloqueia startup e preserva evidências intactas."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    # Simula gravação anterior de estado DIRTY
    dirty_meta = {
        "status": "DIRTY",
        "holder": "unclean_shutdown_proc",
        "pid": 99999,
        "timestamp_utc": "2026-10-12T00:00:00Z",
        "reason": "writers_operacionais_ativos_no_shutdown",
    }
    with open(lock_path, "wb") as f:
        f.write(b"#" + json.dumps(dirty_meta, indent=2).encode("utf-8"))

    # Preflight de startup deve falhar
    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is False
    assert "DIRTY" in msg
    assert "exige estritamente estado anterior CLEAN" in msg

    # Tentativa de acquire direto com require_previous_state=CLEAN também deve falhar
    lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="test_runner")
    with pytest.raises(RuntimeLockError, match="diverge do exigido 'CLEAN'"):
        lock.acquire(timeout=0.0, require_previous_state=RuntimeState.CLEAN)

    # Evidências preservadas intactas
    persisted = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert persisted["status"] == RuntimeState.DIRTY
    assert persisted["reason"] == "writers_operacionais_ativos_no_shutdown"


def test_05_unknown_blocks_startup(tmp_path: Path):
    """5. Lock corrompido ou com estado UNKNOWN bloqueia startup sem sobrescrever evidências."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    # Escreve payload corrompido (JSON truncado)
    corrupted_bytes = b"#{\"status\": \"CORRUPTED_JSON_TRUNCATED"
    with open(lock_path, "wb") as f:
        f.write(corrupted_bytes)

    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is False
    assert "UNKNOWN" in msg

    # O conteúdo do disco deve ser preservado exatamente igual
    with open(lock_path, "rb") as f:
        assert f.read() == corrupted_bytes


def test_06_residual_running_blocks_startup(tmp_path: Path):
    """6. RUNNING residual no disco sem lock do kernel (crash detectado) bloqueia startup."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    # Grava RUNNING mas nenhum processo mantém a trava no kernel
    running_meta = {
        "status": "RUNNING",
        "holder": "crashed_process",
        "pid": 88888,
        "timestamp_utc": "2026-10-12T01:00:00Z",
    }
    with open(lock_path, "wb") as f:
        f.write(b"#" + json.dumps(running_meta, indent=2).encode("utf-8"))

    # read_persisted_state deve classificar como UNKNOWN (crash detectado)
    state = OperationalRuntimeLock.read_persisted_state(lock_path)
    assert state["status"] == RuntimeState.UNKNOWN
    assert "crash_detected" in state["reason"]

    # Preflight de startup deve bloquear fail-closed
    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is False
    assert "UNKNOWN" in msg


def test_07_deleted_lock_blocks_startup(tmp_path: Path):
    """7. Lock deletado em ambiente com dados operacionais bloqueia startup (Anti-Bypass)."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    assert lock_path.exists()
    lock_path.unlink()  # Simula exclusão maliciosa ou acidental

    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is False
    assert "Anti-Bypass" in msg
    assert "Lock operacional ausente" in msg


def test_08_deleted_manifest_blocks_startup(tmp_path: Path):
    """8. Manifesto deletado em ambiente com lock bloqueia startup (Anti-Bypass)."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    manifest_path = data_dir / MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest_path.unlink()  # Simula exclusão do manifesto

    is_ok, msg = verify_startup_preflight(data_dir=data_dir)
    assert is_ok is False
    assert "Anti-Bypass" in msg
    assert "Manifesto de inicialização ausente" in msg


def test_09_invalid_sqlite_blocks_startup(tmp_path: Path):
    """9. Dados SQLite inválidos ou corrompidos bloqueiam o startup."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_healthy_sqlite(sqlite_path)
    _create_healthy_jsonl(jsonl_path)

    bootstrap_runtime_environment(data_dir=data_dir)

    # Corrompe o banco SQLite sobrescrevendo os primeiros 100 bytes
    with open(sqlite_path, "r+b") as f:
        f.seek(16)
        f.write(b"CORRUPTED_SQLITE_PAGE_HEADER_DATA_123456789")

    is_ok, msg = verify_startup_preflight(data_dir=data_dir)
    assert is_ok is False
    assert "Falha de integridade SQLite" in msg


def test_10_invalid_jsonl_blocks_startup(tmp_path: Path):
    """10. JSONL com última linha truncada/inválida bloqueia o startup."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_healthy_sqlite(sqlite_path)
    _create_healthy_jsonl(jsonl_path)

    bootstrap_runtime_environment(data_dir=data_dir)

    # Adiciona uma linha corrompida no JSONL
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write("{\"incomplete_json\": true, \"missing_brace\": \n")

    is_ok, msg = verify_startup_preflight(data_dir=data_dir)
    assert is_ok is False
    assert "Falha de integridade JSONL" in msg


def test_11_fsync_failure_handling(tmp_path: Path):
    """11. Falha de fsync durante bootstrap é tratada com segurança fail-closed."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    with patch("os.fsync", side_effect=OSError(errno.EIO, "Simulated Disk I/O error")):
        with pytest.raises((RuntimeLockError, OSError)):
            bootstrap_runtime_environment(data_dir=data_dir)


def test_12_concurrent_processes_blocked(tmp_path: Path):
    """12. Processo concorrente com lock ativo bloqueia startup e bootstrap fail-closed."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"

    # Inicia subprocesso que segura a trava física do kernel
    child_script = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{PROJECT_ROOT}')
from common.runtime_lock import OperationalRuntimeLock
lock = OperationalRuntimeLock(lock_path=r'{lock_path}', holder_name='competing_process')
lock.acquire()
sys.stdout.write('LOCKED\\n')
sys.stdout.flush()
time.sleep(5)
lock.release()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Aguarda confirmação de que o processo filho travou o descritor
        line = proc.stdout.readline()
        assert "LOCKED" in line

        # Tentativa do processo atual em adquirir o lock deve falhar imediatamente
        local_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="local_proc")
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            local_lock.acquire(timeout=0.0)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


def test_13_fresh_install_lifecycle(tmp_path: Path):
    """13. Ciclo de vida de instalação nova: bloqueio inicial, diagnóstico e bootstrap com sucesso."""
    data_dir = tmp_path / "dados_novos"
    # Diretório totalmente novo e vazio
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".operational_runtime.lock"

    # 1. Startup direto sem bootstrap deve bloquear fail-closed
    is_ok, msg = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok is False
    assert "Ambiente não inicializado (instalação nova)" in msg

    # 2. Diagnóstico read-only indica pronto para bootstrap
    diag = run_diagnostic(
        data_dir=data_dir,
        lock_path=lock_path,
        sqlite_path=data_dir / "trading_bot.db",
        jsonl_path=data_dir / "eventos_fluxo.jsonl",
        archive_dir=data_dir / "archive",
    )
    assert diag["mode"] == "DIAGNOSTIC_READ_ONLY"
    assert diag["already_initialized"] is False
    assert diag["ready_for_bootstrap"] is True

    # 3. Execução do bootstrap
    res = bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)
    assert res["success"] is True

    # 4. Agora o startup prévio é 100% autorizado
    is_ok_after, msg_after = verify_startup_preflight(data_dir=data_dir, lock_path=lock_path)
    assert is_ok_after is True
    assert "Preflight de startup 100% aprovado" in msg_after


def test_14_canonical_paths_equivalence(tmp_path: Path):
    """14. Caminhos com sintaxes relativas e redundantes são canonicamente resolvidos."""
    data_dir = tmp_path / "dados"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    bootstrap_runtime_environment(data_dir=data_dir)

    # Caminho com referências relativas ./ e ../
    redundant_data_path = tmp_path / "sub" / ".." / "dados"
    redundant_lock_path = redundant_data_path / "." / ".operational_runtime.lock"

    is_ok, msg = verify_startup_preflight(
        data_dir=redundant_data_path,
        lock_path=redundant_lock_path,
    )
    assert is_ok is True
    assert "Preflight de startup 100% aprovado" in msg


def test_15_main_entrypoint_blocked_fail_closed(tmp_path: Path):
    """15. main() bloqueia fail-closed com exit code 2 se lock estiver ausente ou DIRTY."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_healthy_sqlite(data_dir / "trading_bot.db")
    _create_healthy_jsonl(data_dir / "eventos_fluxo.jsonl")

    runner_code = f"""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, r'{PROJECT_ROOT}')
from main import main
code = asyncio.run(main(data_dir=Path(r'{data_dir}'), lock_path=Path(r'{lock_path}')))
sys.exit(code)
"""
    # A: Lock ausente em ambiente com dados => exit code 2
    res_missing = subprocess.run([sys.executable, "-c", runner_code], capture_output=True, text=True)
    assert res_missing.returncode == 2

    # Inicializa ambiente
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    # B: Simula DIRTY => exit code 2
    dirty_meta = {"status": "DIRTY", "reason": "test_crash"}
    with open(lock_path, "wb") as f:
        f.write(b"#" + json.dumps(dirty_meta).encode("utf-8"))

    res_dirty = subprocess.run([sys.executable, "-c", runner_code], capture_output=True, text=True)
    assert res_dirty.returncode == 2


def test_16_bootstrap_cli_subprocess(tmp_path: Path):
    """16. Execução via CLI do script de bootstrap em subprocesso."""
    data_dir = tmp_path / "dados_cli"
    lock_path = data_dir / ".operational_runtime.lock"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_healthy_sqlite(sqlite_path)
    _create_healthy_jsonl(jsonl_path)

    script_path = PROJECT_ROOT / "scripts" / "maintenance" / "bootstrap_runtime_lock.py"

    # 1. Modo diagnóstico (sem --confirm-bootstrap) com --json-output
    res_diag = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir", str(data_dir),
            "--lock-path", str(lock_path),
            "--sqlite-path", str(sqlite_path),
            "--jsonl-path", str(jsonl_path),
            "--json-output",
        ],
        capture_output=True,
        text=True,
    )
    assert res_diag.returncode == 0
    diag_data = json.loads(res_diag.stdout)
    assert diag_data["ready_for_bootstrap"] is True
    assert diag_data["already_initialized"] is False
    assert not lock_path.exists()

    # 2. Execução com --confirm-bootstrap e --json-output
    res_boot = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--confirm-bootstrap",
            "--data-dir", str(data_dir),
            "--lock-path", str(lock_path),
            "--sqlite-path", str(sqlite_path),
            "--jsonl-path", str(jsonl_path),
            "--json-output",
        ],
        capture_output=True,
        text=True,
    )
    assert res_boot.returncode == 0
    boot_data = json.loads(res_boot.stdout)
    assert boot_data["success"] is True
    assert boot_data["status"] == "CLEAN"
    assert lock_path.exists()

    # 3. Execução repetida via CLI é bloqueada com returncode 2
    res_repeat = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--confirm-bootstrap",
            "--data-dir", str(data_dir),
            "--lock-path", str(lock_path),
            "--sqlite-path", str(sqlite_path),
            "--jsonl-path", str(jsonl_path),
            "--json-output",
        ],
        capture_output=True,
        text=True,
    )
    assert res_repeat.returncode == 2
