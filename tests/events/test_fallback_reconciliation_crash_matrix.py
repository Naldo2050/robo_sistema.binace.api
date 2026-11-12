"""
Suíte de Testes da Crash Matrix C0–C12 — Encerramento Abrupto Real de Subprocessos.
FASE 5.11 — GATE G4-D B3-B2-C2-B2.3

Testa exaustivamente o comportamento do orquestrador transacional hermético de reconciliação
(execute_reconciliation_transaction_hermetic) sob interrupção abrupta real de processo
(proc.kill / TerminateProcess no Windows / SIGKILL no Linux), distinguindo crash real de exceções Python.

Estrutura da Matriz:
- C0:  Antes de adquirir LOCK.
- C1:  Depois de LOCK, antes de PREFLIGHT.
- C2:  Depois de PREFLIGHT, antes de SNAPSHOT.
- C3:  Depois de SNAPSHOT, antes de PREPARED.
- C4:  Depois de PREPARED confirmado, antes de SQLite.
- C5:  Durante escrita SQLite (variantes de lote parcial e integridade PRAGMA).
- C6:  Depois de SQLite, antes de JSONL.
- C7:  Durante append JSONL (variantes de lote parcial, linha truncada sem newline e sem fsync).
- C8:  Depois de APPLY completo, antes de VERIFICAÇÃO.
- C9:  Depois de VERIFICAÇÃO aprovada, antes de MANIFESTO.
- C10: Depois de MANIFESTO atualizado, antes de COMMITTED.
- C11: Durante gravação de COMMITTED (variantes sem bytes, truncado e sem fsync).
- C12: Depois de COMMITTED confirmado com fsync, antes de liberar o lock de manutenção.

Cada teste opera 100% isolado em sandbox temporário (tmp_path).
Zero acesso ou dependência de dados operacionais reais.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime_lock import (
    MANIFEST_FILENAME,
    OperationalRuntimeLock,
    RuntimeState,
    read_runtime_manifest,
    write_runtime_manifest,
    verify_startup_preflight,
    verify_manifest_reconciliation_consistency,
)
from events.fallback_reconciler import (
    validate_reconciliation_audit_ledger,
    verify_sqlite_integrity,
    verify_jsonl_integrity,
    execute_reconciliation_transaction_hermetic,
    ReconciliationTransactionStatus,
)
from scripts.maintenance.recover_runtime_state import diagnose_runtime_recovery


# =============================================================================
# FIXTURES E SETUP HERMÉTICO DO SANDBOX
# =============================================================================

def _init_sqlite_db(db_path: Path) -> None:
    """Cria banco SQLite com a tabela events no schema exato do EventStore."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                window_id TEXT,
                is_signal INTEGER,
                payload TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _setup_sandbox(tmp_path: Path, events_batch: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Inicializa sandbox temporário 100% hermético com diretórios, banco SQLite,
    manifesto versão 1.2, lock operacional em estado CLEAN e arquivo de fallback.
    """
    data_dir = tmp_path / "dados"
    fb_dir = tmp_path / "fallback_events"
    archive_dir = tmp_path / "archive"
    backup_dir = tmp_path / "backups"

    data_dir.mkdir(parents=True, exist_ok=True)
    fb_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    lock_path = data_dir / ".operational_runtime.lock"
    manifest_path = data_dir / MANIFEST_FILENAME
    ledger_path = data_dir / ".reconciliation_audit.jsonl"
    recov_ledger_path = data_dir / ".recovery_audit.jsonl"

    _init_sqlite_db(sqlite_path)
    jsonl_path.touch()

    # Cria lock inicial em estado CLEAN com formato canônico (byte 0 + JSON)
    init_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="sandbox_init")
    init_lock.acquire()
    init_lock.release()

    env_uuid = str(uuid.uuid4())
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        manifest_version="1.2",
    )

    # Lote padrão de fallback (2 eventos distintos) se não for informado
    if events_batch is None:
        events_batch = [
            {
                "event_id": "EVT-CRASH-01",
                "epoch_ms": 1700000001000,
                "tipo_evento": "SIGNAL_BUY",
                "symbol": "BTCUSDT",
                "price": 50000.0,
            },
            {
                "event_id": "EVT-CRASH-02",
                "epoch_ms": 1700000002000,
                "tipo_evento": "SIGNAL_SELL",
                "symbol": "BTCUSDT",
                "price": 50500.0,
            },
        ]

    fb_file = fb_dir / "lote_fallback_teste.jsonl"
    with open(fb_file, "w", encoding="utf-8") as f:
        for ev in events_batch:
            f.write(json.dumps(ev) + "\n")

    return {
        "tmp_path": tmp_path,
        "data_dir": data_dir,
        "fb_dir": fb_dir,
        "archive_dir": archive_dir,
        "backup_dir": backup_dir,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "lock_path": lock_path,
        "manifest_path": manifest_path,
        "ledger_path": ledger_path,
        "recov_ledger_path": recov_ledger_path,
        "fallback_file": fb_file,
        "env_uuid": env_uuid,
        "events_batch": events_batch,
    }


# =============================================================================
# PROTOCOLO DE SINCRONIZAÇÃO E BARREIRA PAI-FILHO
# =============================================================================

def _run_worker_barrier_and_kill(
    env: Dict[str, Any],
    scenario: str,
    timeout: float = 12.0,
) -> Tuple[subprocess.Popen, str]:
    """
    Executa o worker como subprocesso filho, aguarda deterministamente o marcador
    de barreira emitido pelo filho e executa encerramento forçado via proc.kill().
    Retorna o objeto Popen encerrado e o marcador de barreira confirmado.
    """
    execution_id = str(uuid.uuid4())
    cmd = [
        sys.executable,
        "-m",
        "tests.helpers.reconciliation_crash_worker",
        "--scenario",
        scenario,
        "--execution-id",
        execution_id,
        "--data-dir",
        str(env["data_dir"]),
        "--sqlite-path",
        str(env["sqlite_path"]),
        "--jsonl-path",
        str(env["jsonl_path"]),
        "--archive-dir",
        str(env["archive_dir"]),
        "--fallback-file",
        str(env["fallback_file"]),
        "--manifest-path",
        str(env["manifest_path"]),
        "--ledger-path",
        str(env["ledger_path"]),
        "--backup-dir",
        str(env["backup_dir"]),
        "--env-uuid",
        str(env["env_uuid"]),
        "--lock-path",
        str(env["lock_path"]),
        "--operator-reason",
        f"Execucao controlada Crash Matrix cenario {scenario} id {execution_id}",
    ]

    # Ambiente limpo e seguro
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(PROJECT_ROOT)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    # Remove credenciais sensíveis do subprocesso caso existam
    for secret_key in ["BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"]:
        child_env.pop(secret_key, None)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=child_env,
    )

    q: queue.Queue = queue.Queue()

    def stdout_reader():
        try:
            for line in iter(proc.stdout.readline, ""):
                q.put(line)
        except Exception:
            pass
        finally:
            q.put(None)

    reader_thread = threading.Thread(target=stdout_reader, daemon=True)
    reader_thread.start()

    expected_prefix = f"CRASH_BARRIER:REACHED:{scenario}:{execution_id}:"
    reached_marker: Optional[str] = None
    collected_lines: List[str] = []
    start_time = time.monotonic()

    try:
        while time.monotonic() - start_time < timeout:
            try:
                item = q.get(timeout=min(0.5, timeout - (time.monotonic() - start_time)))
                if item is None:
                    break
                item_s = item.strip()
                collected_lines.append(item_s)
                if item_s.startswith(expected_prefix):
                    reached_marker = item_s
                    break
            except queue.Empty:
                if proc.poll() is not None:
                    break

        if not reached_marker:
            stderr_tail = ""
            try:
                if proc.poll() is None:
                    proc.kill()
                _, stderr_tail = proc.communicate(timeout=2.0)
            except Exception:
                pass
            raise TimeoutError(
                f"Timeout aguardando barreira de crash '{scenario}' (id={execution_id}). "
                f"Linhas recebidas: {collected_lines}. Stderr: {stderr_tail}. ExitCode: {proc.poll()}"
            )

        # Confirmação inequívoca: o processo está VIVO no ponto de barreira
        assert proc.poll() is None, f"Worker para {scenario} encerrou antes do sinal de kill do pai!"

        # Encerramento forçado abrupto (TerminateProcess / SIGKILL)
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

        # Confirma que o processo foi liquidado
        assert proc.poll() is not None, f"Worker para {scenario} permaneceu vivo após proc.kill!"

        # Higiene de IPC e sincronização de handles com o subprocesso encerrado
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass
        if reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        assert not reader_thread.is_alive(), (
            f"reader_thread permaneceu ativa após join para o cenário {scenario}"
        )

        return proc, reached_marker

    finally:
        # Higiene rigorosa contra processos zumbis e fechamento de descritores de IPC
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass


def _verify_kernel_lock_released(lock_path: Path) -> None:
    """Valida se o kernel do SO realmente liberou a trava física msvcrt/flock."""
    test_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="post_crash_verifier")
    test_lock.acquire_maintenance(timeout=0.0)
    assert test_lock.is_locked is True, "Falha: trava física não foi liberada pelo SO após crash do subprocesso!"
    test_lock.release_maintenance()


def _get_sqlite_events_count(sqlite_path: Path) -> int:
    """Conta eventos inseridos no SQLite abrindo conexão read-only."""
    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]
    finally:
        conn.close()


def _get_jsonl_events_count(jsonl_path: Path) -> int:
    """Conta linhas válidas no arquivo JSONL."""
    if not jsonl_path.exists():
        return 0
    count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# =============================================================================
# ETAPA A & B: HARNESS E CENÁRIOS C0 A C4
# =============================================================================

def test_harness_subprocess_kill_and_cleanup(tmp_path: Path):
    """ETAPA A — Validação do harness de subprocessos: barreira, kill forçado e liberação de lock."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C0")
    assert "CRASH_BARRIER:REACHED:C0:" in marker
    assert proc.returncode is not None
    _verify_kernel_lock_released(env["lock_path"])


def test_crash_c0_before_lock(tmp_path: Path):
    """C0 — Interrupção antes de LOCK: lock não adquirido, zero artefatos, startup e preflight limpos."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C0")
    assert "BEFORE_LOCK" in marker

    _verify_kernel_lock_released(env["lock_path"])
    lock_state = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert lock_state["status"] == RuntimeState.CLEAN

    # Zero resíduos
    assert not env["ledger_path"].exists() or env["ledger_path"].stat().st_size == 0
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    # Startup aprovado
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True, f"Startup reprovado indevidamente em C0: {st_msg}"

    # Recovery diagnóstico confirma ausência de conflitos
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["recovery_classification"] == "CLEAN_OPERATIONAL"

    # Novo preflight admitido
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="next_tester")
    res_next = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa subsequente autorizada pos C0",
        operational_lock=op_lock,
    )
    assert res_next.status == ReconciliationTransactionStatus.SUCCESS


def test_crash_c1_after_lock_before_preflight(tmp_path: Path):
    """C1 — Interrupção após LOCK, antes de PREFLIGHT: kernel libera trava, lock retém estado CLEAN."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C1")
    assert "AFTER_LOCK_BEFORE_PREFLIGHT" in marker

    _verify_kernel_lock_released(env["lock_path"])
    lock_state = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert lock_state["status"] == RuntimeState.CLEAN

    # Zero registros
    assert not env["ledger_path"].exists() or env["ledger_path"].stat().st_size == 0
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    # Startup e preflight
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["recovery_classification"] == "CLEAN_OPERATIONAL"


def test_crash_c2_after_preflight_before_snapshot(tmp_path: Path):
    """C2 — Interrupção após PREFLIGHT, antes de SNAPSHOT: preflight não teve efeito colateral."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C2")
    assert "AFTER_PREFLIGHT_BEFORE_SNAPSHOT" in marker

    _verify_kernel_lock_released(env["lock_path"])
    snapshots_dir = env["backup_dir"] / "reconciliation_snapshots"
    assert not snapshots_dir.exists() or len(list(snapshots_dir.iterdir())) == 0

    assert not env["ledger_path"].exists() or env["ledger_path"].stat().st_size == 0
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True


def test_crash_c3_after_snapshot_before_prepared(tmp_path: Path):
    """C3 — Interrupção após SNAPSHOT, antes de PREPARED: snapshot órfão preservado, ledger sem PREPARED."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C3")
    assert "AFTER_SNAPSHOT_BEFORE_PREPARED" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Evidência forense: snapshot foi criado fisicamente!
    snap_folders = list(env["backup_dir"].glob("reconcile_snapshot_*"))
    assert len(snap_folders) == 1, "Snapshot forense deveria ter sido preservado em disco!"

    # Porém, o ledger NÃO possui PREPARED gravado
    assert not env["ledger_path"].exists() or env["ledger_path"].stat().st_size == 0

    # Dados operacionais originais intactos
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    # Startup aprovado (pois dados operacionais e ledger estão consistentes; snapshot órfão em backup não veta startup)
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True, f"Startup deveria aprovar em C3: {st_msg}"

    # Recovery diagnóstico constata integridade
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["recovery_classification"] == "CLEAN_OPERATIONAL"

    # Novo preflight admitido: próximo apply gera novo snapshot e completa
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="next_tester_c3")
    res_next = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa subsequente autorizada pos C3",
        operational_lock=op_lock,
    )
    assert res_next.status == ReconciliationTransactionStatus.SUCCESS
    # Agora existem 2 snapshots no diretório de backups (o órfão do crash anterior + o do apply bem-sucedido)
    assert len(list(env["backup_dir"].glob("reconcile_snapshot_*"))) == 2


def test_crash_c4_after_prepared_before_sqlite(tmp_path: Path):
    """C4 — Interrupção após PREPARED confirmado: startup e novo apply BLOQUEADOS FAIL-CLOSED."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C4")
    assert "AFTER_PREPARED_BEFORE_SQLITE" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Evidência persistida no ledger: PREPARED gravado e fsynced
    assert env["ledger_path"].exists()
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["has_pending_transaction"] is True
    assert len(ledger_val["orphaned_prepared"]) == 1
    orphaned_id = ledger_val["orphaned_prepared"][0]

    # Destinos financeiros ainda intocados (0 inserções)
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    # Manifesto inalterado
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["reconciliation_history_counter"] == 0

    # STARTUP BLOQUEADO FAIL-CLOSED
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False, "Startup deveria ter sido BLOQUEADO fail-closed em C4!"
    assert "Transação de reconciliação incompleta/interrompida" in st_msg or "PENDING_TRANSACTION" in st_msg

    # RECOVERY DIAGNÓSTICO detecta transação pendente e emite conflito
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert any("PENDING_TRANSACTION" in str(c) for c in rec_diag["conflicts"])
    assert orphaned_id in str(rec_diag["conflicts"])

    # NOVO APPLY BLOQUEADO FAIL-CLOSED (sem mutações adicionais)
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="blocked_tester_c4")
    res_blocked = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa que deve ser rejeitada fail-closed",
        operational_lock=op_lock,
    )
    assert res_blocked.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED
    assert any("PREPARED sem COMMITTED" in err or "incompleta" in err for err in res_blocked.errors)

    # Imutabilidade confirmada: zero alterações em SQLite e JSONL
    assert _get_sqlite_events_count(env["sqlite_path"]) == 0
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0


# =============================================================================
# ETAPA C: CENÁRIOS C5 A C9 (APPLY PARCIAL E VERIFICAÇÃO)
# =============================================================================

def test_crash_c5_during_sqlite_write(tmp_path: Path):
    """C5 — Interrupção durante escrita SQLite: lote parcialmente persistido em SQLite, JSONL intocado."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C5")
    assert "DURING_SQLITE_PARTIAL_COMMITTED" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Evidência de presença parcial: SQLite tem 1 evento, JSONL tem 0
    assert _get_sqlite_events_count(env["sqlite_path"]) == 1
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    # Integridade física do banco SQLite preservada (PRAGMA integrity_check)
    sq_ok, sq_msg = verify_sqlite_integrity(env["sqlite_path"])
    assert sq_ok is True, f"Integridade do SQLite comprometida após crash: {sq_msg}"

    # Ledger possui PREPARED pendente
    assert env["ledger_path"].exists()
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["has_pending_transaction"] is True

    # STARTUP BLOQUEADO FAIL-CLOSED por transação pendente
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False
    assert "Transação de reconciliação incompleta/interrompida" in st_msg or "PENDING_TRANSACTION" in st_msg

    # RECOVERY DIAGNÓSTICO detecta assimetria entre destinos
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["consistency"]["events_sqlite_only"] == 1
    assert rec_diag["consistency"]["events_jsonl_only"] == 0
    assert any("PENDING_TRANSACTION" in str(c) for c in rec_diag["conflicts"])

    # NOVO APPLY BLOQUEADO FAIL-CLOSED
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="blocked_tester_c5")
    res_blocked = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa subsequente que deve falhar fail-closed",
        operational_lock=op_lock,
    )
    assert res_blocked.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED

    # Imutabilidade estrita pós-bloqueio
    assert _get_sqlite_events_count(env["sqlite_path"]) == 1
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0


def test_crash_c6_after_sqlite_before_jsonl(tmp_path: Path):
    """C6 — Interrupção após SQLite, antes de JSONL: assimetria completa entre destinos detectada."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C6")
    assert "AFTER_SQLITE_BEFORE_JSONL" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Todos os eventos do lote estão em SQLite; zero eventos em JSONL
    assert _get_sqlite_events_count(env["sqlite_path"]) == 2
    assert _get_jsonl_events_count(env["jsonl_path"]) == 0

    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False

    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["consistency"]["events_sqlite_only"] == 2
    assert rec_diag["consistency"]["events_jsonl_only"] == 0
    assert any("PENDING_TRANSACTION" in str(c) for c in rec_diag["conflicts"])

    # Novo apply bloqueado
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="blocked_tester_c6")
    res_blocked = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa subsequente que deve falhar fail-closed C6",
        operational_lock=op_lock,
    )
    assert res_blocked.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED


def test_crash_c7_during_jsonl_partial_batch(tmp_path: Path):
    """C7 — Interrupção durante append JSONL: SQLite completo (2), JSONL parcial (1)."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C7")
    assert "DURING_JSONL_PARTIAL_BATCH" in marker

    _verify_kernel_lock_released(env["lock_path"])

    assert _get_sqlite_events_count(env["sqlite_path"]) == 2
    assert _get_jsonl_events_count(env["jsonl_path"]) == 1

    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False

    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["consistency"]["events_sqlite_only"] == 1
    assert rec_diag["consistency"]["events_in_both_sqlite_and_jsonl"] == 1


def test_crash_c7_variant_truncated_line(tmp_path: Path):
    """C7 Variante — Interrupção com linha JSONL truncada sem newline: verify_jsonl_integrity reprova."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C7_PARTIAL_LINE")
    assert "DURING_JSONL_TRUNCATED_LINE" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Integridade do JSONL reprovada diretamente
    jl_ok, jl_msg = verify_jsonl_integrity(env["jsonl_path"])
    assert jl_ok is False, "JSONL truncado deveria ser reprovado por verify_jsonl_integrity!"
    assert "inválida ou corrompida" in jl_msg or "JSON" in jl_msg

    # Startup bloqueado fail-closed
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False

    # Recovery diagnóstico acusa corrupção de linha
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["jsonl"]["corrupt_lines_count"] > 0
    assert any("corrompidas ou truncadas" in str(c) for c in rec_diag["conflicts"])


def test_crash_c7_variant_complete_line_no_fsync(tmp_path: Path):
    """C7 Variante — Linha JSONL completa gravada sem confirmação de fsync antes do crash."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C7_COMPLETE_NO_FSYNC")
    assert "DURING_JSONL_NO_FSYNC" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # O SO fechou o arquivo na morte do processo; a linha é observável em disco
    assert env["jsonl_path"].stat().st_size > 0
    jl_ok, _ = verify_jsonl_integrity(env["jsonl_path"])
    assert jl_ok is True

    # Porém o ledger permanece em PREPARED (sem COMMITTED); startup é bloqueado
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False


def test_crash_c8_after_apply_before_verification(tmp_path: Path):
    """C8 — Interrupção após APPLY completo, antes da VERIFICAÇÃO: destinos gravados, ledger pendente."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C8")
    assert "AFTER_APPLY_BEFORE_VERIFICATION" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Ambos os destinos possuem os 2 eventos
    assert _get_sqlite_events_count(env["sqlite_path"]) == 2
    assert _get_jsonl_events_count(env["jsonl_path"]) == 2

    # Manifesto permanece intacto no contador 0
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["reconciliation_history_counter"] == 0

    # Startup e preflight bloqueados por ausência de COMMITTED
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False
    assert "Transação de reconciliação incompleta/interrompida" in st_msg or "PENDING_TRANSACTION" in st_msg


def test_crash_c9_after_verification_before_manifest(tmp_path: Path):
    """C9 — Interrupção após VERIFICAÇÃO, antes de MANIFESTO: verificação aprovada descartada."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C9")
    assert "AFTER_VERIFICATION_BEFORE_MANIFEST" in marker

    _verify_kernel_lock_released(env["lock_path"])

    assert _get_sqlite_events_count(env["sqlite_path"]) == 2
    assert _get_jsonl_events_count(env["jsonl_path"]) == 2
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["reconciliation_history_counter"] == 0

    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False


# =============================================================================
# ETAPA D: CENÁRIOS C10 A C12 E VARIANTES DE PERSISTÊNCIA PARCIAL
# =============================================================================

def test_crash_c10_after_manifest_before_committed(tmp_path: Path):
    """C10 — Interrupção após MANIFESTO, antes de COMMITTED: divergência entre manifesto e ledger."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C10")
    assert "AFTER_MANIFEST_BEFORE_COMMITTED" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Manifesto foi atualizado para contador 1!
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["reconciliation_history_counter"] == 1
    assert man_data["last_reconciliation_audit_id"] is not None

    # Porém o ledger tem apenas PREPARED (zero COMMITTED)
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["has_pending_transaction"] is True
    assert ledger_val["completed_transactions_count"] == 0

    # Consistência H2 detecta divergência categórica
    consistency = verify_manifest_reconciliation_consistency(
        manifest_data=man_data,
        ledger_path=env["ledger_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert consistency["is_valid"] is False
    assert consistency["classification"] in ("PENDING_TRANSACTION", "DIVERGENT_COUNTER")

    # STARTUP BLOQUEADO FAIL-CLOSED DUPLO
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False
    assert "Startup bloqueado fail-closed" in st_msg

    # RECOVERY DIAGNÓSTICO acusa divergência
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert any("Inconsistência de histórico de reconciliação" in str(c) for c in rec_diag["conflicts"])

    # NOVO APPLY BLOQUEADO FAIL-CLOSED
    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="blocked_tester_c10")
    res_blocked = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=env["fallback_file"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Tentativa subsequente que deve falhar fail-closed C10",
        operational_lock=op_lock,
    )
    assert res_blocked.status == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED


def test_crash_c11_before_committed_bytes(tmp_path: Path):
    """C11 — Interrupção antes de gravar bytes de COMMITTED: idêntico a C10 com manifesto adiantado."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C11")
    assert "BEFORE_COMMITTED_WRITE" in marker

    _verify_kernel_lock_released(env["lock_path"])
    st_ok, _ = verify_startup_preflight(env["data_dir"])
    assert st_ok is False


def test_crash_c11_variant_truncated_committed_line(tmp_path: Path):
    """C11 Variante — Linha de COMMITTED parcialmente gravada sem newline: CORRUPTED_LEDGER detectado."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C11_PARTIAL_BYTES")
    assert "DURING_COMMITTED_PARTIAL_BYTES" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # Validação do ledger reprova com erro estrutural explícito
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["is_valid"] is False
    assert "newline" in str(ledger_val["error"]) or "truncado" in str(ledger_val["error"])

    # Consistência H2 classifica como CORRUPTED_LEDGER
    man_data = read_runtime_manifest(env["manifest_path"])
    consistency = verify_manifest_reconciliation_consistency(
        manifest_data=man_data,
        ledger_path=env["ledger_path"],
        expected_env_uuid=env["env_uuid"],
    )
    assert consistency["is_valid"] is False
    assert consistency["classification"] == "CORRUPTED_LEDGER"

    # Startup e novo preflight bloqueados
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False
    assert "corrompido ou inconsistente" in st_msg or "CORRUPTED_LEDGER" in st_msg


def test_crash_c11_variant_complete_committed_no_fsync(tmp_path: Path):
    """C11 Variante — Linha COMMITTED gravada sem fsync: avalia observabilidade física do SO."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C11_COMPLETE_NO_FSYNC")
    assert "COMMITTED_LINE_NO_FSYNC" in marker

    _verify_kernel_lock_released(env["lock_path"])

    # A linha foi persistida pelo SO antes do encerramento
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    # Se a linha no arquivo foi gravada integralmente pelo SO, o ledger tem par PREPARED->COMMITTED
    assert ledger_val["is_valid"] is True
    assert ledger_val["completed_transactions_count"] == 1


def test_crash_c12_after_committed_before_release(tmp_path: Path):
    """C12 — Interrupção após COMMITTED com fsync, antes de liberar lock: transação é durável e startup aprova."""
    env = _setup_sandbox(tmp_path)
    proc, marker = _run_worker_barrier_and_kill(env, scenario="C12")
    assert "AFTER_COMMITTED_BEFORE_RELEASE" in marker

    # O processo foi morto abruptamente antes do finally liberar a manutenção
    # Mas o kernel do SO liberou a trava física imediatamente
    _verify_kernel_lock_released(env["lock_path"])

    # Lock em disco preservou status CLEAN original
    lock_state = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert lock_state["status"] == RuntimeState.CLEAN

    # Ambos os destinos possuem os dados do lote
    assert _get_sqlite_events_count(env["sqlite_path"]) == 2
    assert _get_jsonl_events_count(env["jsonl_path"]) == 2

    # Manifesto e ledger sincronizados (contador 1)
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["reconciliation_history_counter"] == 1

    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["is_valid"] is True
    assert ledger_val["completed_transactions_count"] == 1
    assert ledger_val["has_pending_transaction"] is False

    # STARTUP APROVADO: Como a transação foi 100% commitada antes do crash, o sistema está consistente
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True, f"Startup deveria ser APROVADO em C12: {st_msg}"

    # RECOVERY DIAGNÓSTICO confirma CLEAN operacional sem conflitos
    rec_diag = diagnose_runtime_recovery(env["data_dir"])
    assert rec_diag["recovery_classification"] == "CLEAN_OPERATIONAL"
    assert len(rec_diag["conflicts"]) == 0

    # NOVO PREFLIGHT ADMITIDO: Transação anterior concluída com sucesso permite novas reconciliações
    next_batch = [
        {
            "event_id": "EVT-CRASH-03",
            "epoch_ms": 1700000003000,
            "tipo_evento": "SIGNAL_BUY",
            "symbol": "ETHUSDT",
            "price": 3000.0,
        }
    ]
    next_fb = env["fb_dir"] / "next_batch.jsonl"
    with open(next_fb, "w", encoding="utf-8") as f:
        f.write(json.dumps(next_batch[0]) + "\n")

    op_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="next_tester_c12")
    res_next = execute_reconciliation_transaction_hermetic(
        data_dir=env["data_dir"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        source_fallback_file=next_fb,
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        backup_base_dir=env["backup_dir"],
        environment_uuid=env["env_uuid"],
        operator_reason="Lote subsequente legítimo autorizado pos C12",
        operational_lock=op_lock,
    )
    assert res_next.status == ReconciliationTransactionStatus.SUCCESS
    assert res_next.sqlite_inserted == 1
    assert res_next.jsonl_appended == 1

