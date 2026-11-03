# -*- coding: utf-8 -*-
"""
Suíte de Regressão Consolidada e Validação das Interfaces Financeiras (Gate G4-D B3-B2-C2-B2.5-B).

Testes de Integração Consolidada:
INT-01: Reconciliação concluída -> Startup preflight aceito.
INT-02: Crash C8 (pós-apply pré-verificação) -> Startup preflight bloqueado fail-closed.
INT-03: Crash C8 -> Recovery diagnóstico detecta inconsistência sem mutação.
INT-04: Convergência de lote assimétrico cruzado (SQLite parcial + JSONL parcial).
INT-05: Ledger de reconciliação com linha truncada rejeitado fail-closed.
INT-06: Concorrência multiprocesso real no runtime lock com IPC determinístico.
INT-07: Reconciliação concorrente bloqueia startup preflight.
INT-08: Acúmulo sequencial de múltiplos históricos no manifesto v1.2.
INT-09: Integridade e fidelidade do snapshot forense pré-apply.
INT-10: Fallback malformado/corrompido rejeitado no preflight sem mutações financeiras.

Ambiente: 100% Hermético em tmp_path. Zero dependência de rede ou dados operacionais reais.
"""

from __future__ import annotations

import hashlib
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
    RuntimeLockError,
    RuntimeState,
    read_runtime_manifest,
    write_runtime_manifest,
    verify_startup_preflight,
    verify_manifest_reconciliation_consistency,
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    FallbackReconciler,
    ReconciliationTransactionStatus,
    insert_event_to_sqlite,
    append_event_to_jsonl,
    verify_sqlite_integrity,
    verify_jsonl_integrity,
    verify_reconciliation_targets,
    validate_reconciliation_audit_ledger,
    record_reconciliation_audit,
)
from scripts.maintenance.recover_runtime_state import diagnose_runtime_recovery
from tests.helpers.cli_apply_integration_harness import (
    HermeticSandboxContext,
    execute_apply_integration_hermetic,
)


# =============================================================================
# FIXTURES E SETUP HERMÉTICO DO SANDBOX
# =============================================================================

def _init_sqlite_db(db_path: Path) -> None:
    """Cria banco SQLite com tabela events no schema exato do EventStore."""
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


def _setup_sandbox(tmp_path: Path, initial_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Inicializa sandbox temporário 100% hermético com diretórios, banco SQLite,
    manifesto versão 1.2, lock operacional em estado CLEAN e arquivo de fallback inicial.
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
    ledger_path = data_dir / RECONCILIATION_AUDIT_FILENAME

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

    if initial_events is None:
        initial_events = [
            {
                "event_id": "EVT-CONS-01",
                "epoch_ms": 1700000001000,
                "tipo_evento": "SIGNAL_BUY",
                "symbol": "BTCUSDT",
                "price": 50000.0,
            },
            {
                "event_id": "EVT-CONS-02",
                "epoch_ms": 1700000002000,
                "tipo_evento": "SIGNAL_SELL",
                "symbol": "BTCUSDT",
                "price": 50500.0,
            },
        ]

    fb_file = fb_dir / "lote_fallback_teste.jsonl"
    with open(fb_file, "w", encoding="utf-8") as f:
        for ev in initial_events:
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
        "fallback_file": fb_file,
        "env_uuid": env_uuid,
        "initial_events": initial_events,
    }


def _run_crash_worker_c8(env: Dict[str, Any], timeout: float = 12.0) -> Tuple[subprocess.Popen, str]:
    """Executa o worker de crash C8 em subprocesso isolado e aguarda a barreira determinística."""
    execution_id = str(uuid.uuid4())
    cmd = [
        sys.executable,
        "-m",
        "tests.helpers.reconciliation_crash_worker",
        "--scenario",
        "C8",
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
        f"Execucao controlada Crash C8 id {execution_id}",
    ]

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(PROJECT_ROOT)
    child_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
        env=child_env,
        text=True,
        bufsize=1,
    )

    line_queue: queue.Queue[str] = queue.Queue()

    def _reader(pipe):
        try:
            for line in iter(pipe.readline, ""):
                line_queue.put(line)
        except Exception:
            pass

    t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
    t.start()

    expected_prefix = f"CRASH_BARRIER:REACHED:C8:{execution_id}:"
    confirmed_marker = ""
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            line = line_queue.get(timeout=0.2)
            if line.startswith(expected_prefix):
                confirmed_marker = line.strip()
                break
        except queue.Empty:
            if proc.poll() is not None:
                break

    assert confirmed_marker != "", f"Timeout aguardando barreira de crash C8 para id={execution_id}"

    # Encerramento abrupto forçado no ponto C8 exato
    proc.kill()
    proc.wait(timeout=5.0)

    # Aguardar SO liberar o descritor físico de lock no kernel
    time.sleep(0.15)
    return proc, confirmed_marker


# =============================================================================
# TESTES DE INTEGRAÇÃO CONSOLIDADA (INT-01 A INT-10)
# =============================================================================

def test_int_01_reconciliation_success_to_startup_preflight(tmp_path: Path):
    """INT-01: Reconciliação concluída com sucesso produz manifesto v1.2 aceito pelo startup preflight."""
    env = _setup_sandbox(tmp_path)

    # 1. Reconciliação hermética legítima via harness
    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Reconciliacao hermetica legitima para teste INT-01",
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"
    assert payload["sqlite_inserted"] == 2
    assert payload["jsonl_appended"] == 2

    # 2. Manifesto 1.2 devidamente atualizado com histórico de reconciliação
    man_data = read_runtime_manifest(env["manifest_path"])
    assert man_data["manifest_version"] == "1.2"
    assert man_data["reconciliation_history_counter"] == 1
    assert man_data["last_reconciliation_audit_id"] == payload["audit_id"]
    assert man_data["last_reconciliation_utc"] is not None

    # 3. Startup preflight oficial valida a consistência de ponta a ponta
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is True, f"Startup preflight reprovou indevidamente ambiente reconciliado: {st_msg}"
    assert "aprovado" in st_msg.lower()


def test_int_02_crash_c8_blocks_startup(tmp_path: Path):
    """INT-02: Crash C8 (pós-APPLY, pré-VERIFICAÇÃO) bloqueia startup preflight fail-closed."""
    env = _setup_sandbox(tmp_path)

    # Injeção real de subprocesso com encerramento forçado em C8
    proc, marker = _run_crash_worker_c8(env)
    assert "AFTER_APPLY_BEFORE_VERIFICATION" in marker

    # Ledger possui PREPARED gravado, mas COMMITTED ausente
    assert env["ledger_path"].exists()
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["has_pending_transaction"] is True

    # Startup preflight DEVE ser estritamente bloqueado (fail-closed)
    st_ok, st_msg = verify_startup_preflight(env["data_dir"])
    assert st_ok is False, "Startup preflight deveria ter sido BLOQUEADO fail-closed após crash C8!"
    assert any(term in st_msg for term in ("Transação de reconciliação incompleta", "PENDING_TRANSACTION", "PREPARED sem COMMITTED"))


def test_int_03_crash_c8_recovery_diagnostic(tmp_path: Path):
    """INT-03: Crash C8 é identificado pela ferramenta de recovery em modo diagnóstico sem mutações."""
    env = _setup_sandbox(tmp_path)

    # Injeção de crash C8
    _run_crash_worker_c8(env)

    # Hash dos arquivos antes do diagnóstico
    h_sqlite_before = hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest()
    h_jsonl_before = hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest()
    h_ledger_before = hashlib.sha256(env["ledger_path"].read_bytes()).hexdigest()

    # Executa caminho diagnóstico read-only da ferramenta de recovery
    rec_diag = diagnose_runtime_recovery(env["data_dir"])

    # Conflito de transação incompleta/pendente identificado
    assert any("PENDING_TRANSACTION" in str(c) or "reconciliação" in str(c) for c in rec_diag["conflicts"])

    # Zero mutação confirmada nos destinos financeiros e de auditoria
    assert hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest() == h_sqlite_before
    assert hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest() == h_jsonl_before
    assert hashlib.sha256(env["ledger_path"].read_bytes()).hexdigest() == h_ledger_before


def test_int_04_asymmetric_cross_batch_convergence(tmp_path: Path):
    """INT-04: Convergência de lote assimétrico cruzado (SQLite tem A, JSONL tem B, Fallback tem A, B e C)."""
    env = _setup_sandbox(tmp_path)

    ev_A = {
        "event_id": "EVT-ASYMM-A",
        "epoch_ms": 1700000010000,
        "tipo_evento": "ORDER_NEW",
        "symbol": "BTCUSDT",
        "price": 50100.0,
    }
    ev_B = {
        "event_id": "EVT-ASYMM-B",
        "epoch_ms": 1700000020000,
        "tipo_evento": "ORDER_FILLED",
        "symbol": "BTCUSDT",
        "price": 50200.0,
    }
    ev_C = {
        "event_id": "EVT-ASYMM-C",
        "epoch_ms": 1700000030000,
        "tipo_evento": "ORDER_CANCELLED",
        "symbol": "BTCUSDT",
        "price": 50300.0,
    }

    # Pré-população assimétrica
    insert_event_to_sqlite(env["sqlite_path"], ev_A)
    append_event_to_jsonl(env["jsonl_path"], ev_B)

    # Fallback contém lote completo (A, B e C)
    with open(env["fallback_file"], "w", encoding="utf-8") as f:
        for ev in [ev_A, ev_B, ev_C]:
            f.write(json.dumps(ev) + "\n")

    # Executa reconciliação hermética via harness
    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Convergencia hermetica de lote assimetrico cruzado",
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"
    assert payload["sqlite_inserted"] == 2  # Inseriu B e C (A já estava presente)
    assert payload["jsonl_appended"] == 2   # Appendou A e C (B já estava presente)

    # Validação de convergência simétrica de 100% dos eventos
    verif_res = verify_reconciliation_targets(
        expected_events=[ev_A, ev_B, ev_C],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        require_all_destinations_default=True,
    )
    assert verif_res.approved is True, f"Pós-verificação reprovou convergência simétrica: {verif_res.errors}"


def test_int_05_truncated_ledger_line_fails_closed(tmp_path: Path):
    """INT-05: Linha JSON truncada no ledger de reconciliação aciona rejeição fail-closed sem mutações."""
    env = _setup_sandbox(tmp_path)

    # Injeta linha truncada e corrompida no ledger de reconciliação
    with open(env["ledger_path"], "wb") as f:
        f.write(b'{"audit_id": "11111111-2222-3333-4444-555555555555", "reconciliation_phase": "PREPARED", "corrupt_data": ')
        f.flush()

    h_sq_before = hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest()
    h_jl_before = hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest()

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Tentativa sobre ledger corrompido que deve falhar fail-closed",
    )

    assert exit_code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any(
        term in str(e)
        for e in payload["errors"]
        for term in ("Livro-razão de reconciliação", "truncamento", "corrompido", "inconsistente")
    )

    # Destinos financeiros permanecem imutáveis
    assert hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest() == h_sq_before
    assert hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest() == h_jl_before


def test_int_06_real_multiprocess_concurrency(tmp_path: Path):
    """INT-06: Concorrência multiprocesso real comprova exclusividade física de kernel do lock (Exit 3)."""
    env = _setup_sandbox(tmp_path)

    lock_file = str(env["lock_path"])
    signal_file = tmp_path / "proc1_locked.signal"
    release_file = tmp_path / "proc1_release.signal"

    # Script para o Processo 1
    p1_script = f"""
import sys, time
from pathlib import Path
from common.runtime_lock import OperationalRuntimeLock

lock = OperationalRuntimeLock(lock_path=Path(r"{lock_file}"), holder_name="proc_1")
with lock.maintenance_session(allowed_states=["CLEAN"]):
    Path(r"{signal_file}").write_text("LOCKED", encoding="utf-8")
    sys.stdout.write("P1_LOCKED\\n")
    sys.stdout.flush()
    # Espera até receber sinal para liberar
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if Path(r"{release_file}").exists():
            break
        time.sleep(0.05)
sys.stdout.write("P1_DONE\\n")
sys.stdout.flush()
"""

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(PROJECT_ROOT)
    child_env["PYTHONUNBUFFERED"] = "1"

    p1 = subprocess.Popen([sys.executable, "-c", p1_script], cwd=str(PROJECT_ROOT), env=child_env, text=True)

    try:
        # Aguarda Processo 1 adquirir a trava física de kernel
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if signal_file.exists():
                break
            time.sleep(0.05)
        assert signal_file.exists(), "Processo 1 não adquiriu o lock a tempo!"

        # Processo 2 (processo atual) tenta executar apply hermético sobre o mesmo lock
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            sandbox_root=env["tmp_path"],
            operator_reason="Tentativa concorrente que deve receber Exit 3",
        )

        # Colisão física de lock no kernel comprovada
        assert exit_code == 3, f"Esperado Exit 3 por concorrência física, obtido: {exit_code}"
        assert payload["status"] == "FAILED_BEFORE_PREPARED"
        assert payload["stage_reached"] == "LOCK_ACQUISITION"

    finally:
        # Sinaliza Processo 1 para liberar e aguarda encerramento limpo
        release_file.write_text("RELEASE", encoding="utf-8")
        p1.wait(timeout=5.0)

    # Integridade pós-concorrência: estado CLEAN preservado
    st_info = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert st_info["status"] == RuntimeState.CLEAN


def test_int_07_concurrent_reconciliation_blocks_startup_preflight(tmp_path: Path):
    """INT-07: Sessão de reconciliação concorrente ativa bloqueia startup preflight."""
    env = _setup_sandbox(tmp_path)

    lock_file = str(env["lock_path"])
    signal_file = tmp_path / "reconciler_locked.signal"
    release_file = tmp_path / "reconciler_release.signal"

    worker_script = f"""
import sys, time
from pathlib import Path
from common.runtime_lock import OperationalRuntimeLock

lock = OperationalRuntimeLock(lock_path=Path(r"{lock_file}"), holder_name="concurrent_reconciler")
with lock.maintenance_session(allowed_states=["CLEAN"]):
    Path(r"{signal_file}").write_text("LOCKED", encoding="utf-8")
    sys.stdout.write("LOCKED\\n")
    sys.stdout.flush()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if Path(r"{release_file}").exists():
            break
        time.sleep(0.05)
"""

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(PROJECT_ROOT)
    child_env["PYTHONUNBUFFERED"] = "1"

    worker_proc = subprocess.Popen([sys.executable, "-c", worker_script], cwd=str(PROJECT_ROOT), env=child_env, text=True)

    try:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if signal_file.exists():
                break
            time.sleep(0.05)
        # Enquanto o lock está retido pela reconciliação, a tentativa de startup por processo
        # independente DEVE ser estritamente bloqueada por colisão física de exclusividade no kernel
        startup_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="independent_startup_proc")
        with pytest.raises(RuntimeLockError, match="Conflito de exclusividade"):
            startup_lock.acquire(timeout=0.0)

    finally:
        release_file.write_text("RELEASE", encoding="utf-8")
        worker_proc.wait(timeout=5.0)

    # Após liberação da manutenção, startup preflight aprova e o lock está livre para startup legítimo
    st_ok_after, st_msg_after = verify_startup_preflight(env["data_dir"])
    assert st_ok_after is True, f"Startup preflight reprovou após liberação de manutenção: {st_msg_after}"
    final_startup_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="legitimate_startup_proc")
    final_startup_lock.acquire(timeout=0.0)
    final_startup_lock.release()


def test_int_08_sequential_manifest_1_2_history_accumulation(tmp_path: Path):
    """INT-08: Acúmulo sequencial de histórico do manifesto v1.2 por múltiplas reconciliações independentes."""
    env = _setup_sandbox(tmp_path)

    # 1ª Reconciliação
    code_1, payload_1 = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Primeira reconciliacao sequencial INT-08",
    )
    assert code_1 == 0
    man_1 = read_runtime_manifest(env["manifest_path"])
    assert man_1["reconciliation_history_counter"] == 1
    aid_1 = man_1["last_reconciliation_audit_id"]

    # 2ª Reconciliação com novos eventos inéditos
    batch_2 = [
        {
            "event_id": "EVT-SEQ-03",
            "epoch_ms": 1700000030000,
            "tipo_evento": "SIGNAL_BUY",
            "symbol": "ETHUSDT",
            "price": 3000.0,
        },
        {
            "event_id": "EVT-SEQ-04",
            "epoch_ms": 1700000040000,
            "tipo_evento": "SIGNAL_SELL",
            "symbol": "ETHUSDT",
            "price": 3100.0,
        },
    ]
    fb_2 = env["fb_dir"] / "lote_fallback_2.jsonl"
    with open(fb_2, "w", encoding="utf-8") as f:
        for ev in batch_2:
            f.write(json.dumps(ev) + "\n")

    code_2, payload_2 = execute_apply_integration_hermetic(
        fallback_file=fb_2,
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Segunda reconciliacao sequencial INT-08 com incremento",
    )
    assert code_2 == 0
    man_2 = read_runtime_manifest(env["manifest_path"])

    # Contador incrementado estritamente para 2 e audit_id atualizado
    assert man_2["reconciliation_history_counter"] == 2
    assert man_2["last_reconciliation_audit_id"] == payload_2["audit_id"]
    assert man_2["last_reconciliation_audit_id"] != aid_1

    # Ledger contém exatamente 2 transações COMMITTED
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["completed_transactions_count"] == 2

    # Consistência H2 validada
    consistency_h2 = verify_manifest_reconciliation_consistency(
        manifest_data=man_2, ledger_path=env["ledger_path"], expected_env_uuid=env["env_uuid"]
    )
    assert consistency_h2["is_valid"] is True


def test_int_09_pre_apply_forensic_snapshot_integrity(tmp_path: Path):
    """INT-09: Snapshot forense pré-apply captura fielmente os arquivos pré-modificação com manifesto SHA-256."""
    env = _setup_sandbox(tmp_path)

    # Popula 1 evento prévio no SQLite e no JSONL
    ev_initial = {
        "event_id": "EVT-PRE-EXISTING",
        "epoch_ms": 1700000000100,
        "tipo_evento": "INIT_DEPOSIT",
        "symbol": "USDT",
        "amount": 1000.0,
    }
    insert_event_to_sqlite(env["sqlite_path"], ev_initial)
    append_event_to_jsonl(env["jsonl_path"], ev_initial)

    # Registra hashes SHA-256 exatos do estado pré-apply
    h_sqlite_pre = hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest()
    h_jsonl_pre = hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest()

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Validacao de integridade forense do snapshot INT-09",
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"

    # Inspeciona os snapshots criados em backup_dir
    snapshot_dirs = [d for d in env["backup_dir"].iterdir() if d.is_dir() and "reconcile_snapshot" in d.name]
    assert len(snapshot_dirs) >= 1
    snap_dir = snapshot_dirs[0]

    # Valida presença do manifesto de integridade do snapshot
    snap_manifest_file = snap_dir / "snapshot_manifest.json"
    assert snap_manifest_file.exists()
    snap_manifest = json.loads(snap_manifest_file.read_text(encoding="utf-8"))

    # Os hashes gravados no manifesto do snapshot coincidem com o estado pré-apply
    assert snap_manifest["files"]["eventos_fluxo.jsonl"]["sha256"] == h_jsonl_pre

    # O arquivo copiado no snapshot tem exatamente o hash pré-apply
    snap_jsonl_file = snap_dir / "eventos_fluxo.jsonl"
    assert hashlib.sha256(snap_jsonl_file.read_bytes()).hexdigest() == h_jsonl_pre


def test_int_10_malformed_fallback_rejected_without_mutations(tmp_path: Path):
    """INT-10: Fallback malformado/binário é rejeitado no preflight com zero mutação nos destinos."""
    env = _setup_sandbox(tmp_path)

    # Popula evento prévio para atestar imutabilidade
    ev_initial = {
        "event_id": "EVT-STABLE-01",
        "epoch_ms": 1700000000200,
        "tipo_evento": "HEARTBEAT",
    }
    insert_event_to_sqlite(env["sqlite_path"], ev_initial)
    append_event_to_jsonl(env["jsonl_path"], ev_initial)

    h_sqlite_pre = hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest()
    h_jsonl_pre = hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest()

    # Prepara arquivo de fallback com bytes inválidos e corrompidos
    bad_fallback = env["fb_dir"] / "malformed.jsonl"
    with open(bad_fallback, "wb") as f:
        f.write(b'\x00\xff\xfe\x01{"event_id": "EVT-BAD", corrupt_line\n')

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=bad_fallback,
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["tmp_path"],
        operator_reason="Tentativa com fallback malformado que deve ser rejeitada",
    )

    assert exit_code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert payload["stage_reached"] == "EVENT_VALIDATION"

    # Confirmação de ausência absoluta de mutação financeira
    assert hashlib.sha256(env["sqlite_path"].read_bytes()).hexdigest() == h_sqlite_pre
    assert hashlib.sha256(env["jsonl_path"].read_bytes()).hexdigest() == h_jsonl_pre

    # Ledger permanece livre de registros PREPARED ou COMMITTED
    if env["ledger_path"].exists():
        ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
        assert ledger_val["records_count"] == 0
