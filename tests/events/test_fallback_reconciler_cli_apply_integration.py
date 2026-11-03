# -*- coding: utf-8 -*-
"""
Suíte de Testes de Integração Hermética da CLI Apply — Gate G4-D B3-B2-C2-B2.4-B.

Testa formalmente:
1. Bloqueio incondicional da CLI pública via APPLY_NOT_ENABLED_C2A (Camada A).
2. Validação da função interna execute_apply_integration_hermetic (Camada B) em sandboxes temporários.
3. Conexão com o orquestrador transacional hermético (Camada C).
4. Mapeamento de exit codes (0, 1, 2, 3, 4, 5).
5. Barreira técnica anti-produção (rejeição de caminhos em dados operacionais).
6. Resiliência de lock, integridade de manifesto v1.2, anti-bypass e imunidade a retry/rollback automáticos.
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
from typing import Any, Dict, List, Optional
from unittest import mock

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
)
from events.fallback_reconciler import (
    RECONCILIATION_AUDIT_FILENAME,
    ReconciliationTransactionStatus,
    ReconciliationTransactionResult,
    execute_reconciliation_transaction_hermetic,
    validate_reconciliation_audit_ledger,
)
from scripts.maintenance.reconcile_fallback_events import (
    main as cli_main,
)
from tests.helpers.cli_apply_integration_harness import (
    HermeticSandboxContext,
    execute_apply_integration_hermetic,
)


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
    manifesto v1.2, lock operacional em CLEAN e arquivo de fallback.
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

    # Lock inicial CLEAN com formato canônico (byte 0 + JSON)
    init_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="sandbox_init")
    init_lock.acquire()
    init_lock.release()

    env_uuid = str(uuid.uuid4())
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        manifest_version="1.2",
    )

    if events_batch is None:
        events_batch = [
            {
                "event_id": "EVT-CLI-01",
                "epoch_ms": 1700000001000,
                "tipo_evento": "SIGNAL_BUY",
                "symbol": "BTCUSDT",
                "price": 60000.0,
            },
            {
                "event_id": "EVT-CLI-02",
                "epoch_ms": 1700000002000,
                "tipo_evento": "SIGNAL_SELL",
                "symbol": "BTCUSDT",
                "price": 60500.0,
            },
        ]

    fb_file = fb_dir / "lote_cli_teste.jsonl"
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
        "fallback_file": fb_file,
        "env_uuid": env_uuid,
        "events_batch": events_batch,
    }


def _get_sqlite_count(sqlite_path: Path) -> int:
    uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]
    finally:
        conn.close()


def _get_jsonl_count(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


# =============================================================================
# BLOCO 1: TESTES DA CLI PÚBLICA (CAMADA A — BLOQUEIO INCONDICIONAL)
# =============================================================================

def test_01_cli_apply_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """01. Invocação da CLI pública com --apply e confirmação é bloqueada com Exit 1 e APPLY_NOT_ENABLED_C2A."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste com mais de 15 caracteres",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    assert "APPLY_NOT_ENABLED_C2A" in captured.err


def test_02_cli_apply_with_valid_confirm_still_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """02. Mesmo com todos os parâmetros perfeitamente fornecidos e JSON output, CLI pública emite Exit 1."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa auditavel extremamente formal e documentada",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
        "--json-output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "APPLY_NOT_ENABLED_C2A"
    assert payload["mutations_performed"] is False


def test_03_cli_apply_does_not_call_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """03. Comprova que a CLI pública NÃO chama o orquestrador transacional em nenhuma circunstância."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply bloqueada que nao deve chamar orquestrador",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with mock.patch("tests.helpers.cli_apply_integration_harness.execute_reconciliation_transaction_hermetic") as mock_orch:
        code = cli_main()
        assert code == 1
        assert mock_orch.call_count == 0


def test_04_cli_apply_does_not_create_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """04. CLI pública não cria snapshots de reconciliação."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply que nao deve gerar nenhum snapshot em disco",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    snapshots = list(env["backup_dir"].glob("reconcile_snapshot_*"))
    assert len(snapshots) == 0


def test_05_cli_apply_does_not_modify_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """05. CLI pública não realiza mutações no banco SQLite."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply bloqueada garantindo zero mutacoes no SQLite",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    assert _get_sqlite_count(env["sqlite_path"]) == 0


def test_06_cli_apply_does_not_modify_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """06. CLI pública não realiza appends no arquivo JSONL."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply bloqueada garantindo zero mutacoes no JSONL",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    assert _get_jsonl_count(env["jsonl_path"]) == 0


def test_07_cli_apply_does_not_modify_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """07. CLI pública não altera o manifesto operacional."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply bloqueada preservando manifesto inalterado",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    man = read_runtime_manifest(env["manifest_path"])
    assert man["reconciliation_history_counter"] == 0
    assert man["last_reconciliation_audit_id"] is None


def test_08_cli_apply_does_not_write_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """08. CLI pública não grava registros no livro-razão de conciliação."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Tentativa de apply bloqueada garantindo zero escrita de ledger",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    cli_main()
    assert not env["ledger_path"].exists() or env["ledger_path"].stat().st_size == 0


def test_09_dry_run_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """09. Modo dry-run padrão é preservado, roda em somente leitura e sai com Exit 0."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--fallback-dir", str(env["fb_dir"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 0
    captured = capsys.readouterr()
    assert "MODO DRY-RUN (READ-ONLY)" in captured.out


# =============================================================================
# BLOCO 2: TESTES DA FUNÇÃO INTERNA HERMÉTICA (CAMADA B)
# =============================================================================

def test_10_internal_function_in_valid_sandbox(tmp_path: Path):
    """10. Função interna execute_apply_integration_hermetic executa com sucesso em sandbox válido."""
    env = _setup_sandbox(tmp_path)
    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Justificativa formal hermetica para teste 10",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"
    assert payload["sqlite_inserted"] == 2
    assert payload["jsonl_appended"] == 2


def test_11_success_hermetic(tmp_path: Path):
    """11. SUCCESS hermético: Exit 0, inserções confirmadas nos dois destinos, manifesto e ledger atualizados."""
    env = _setup_sandbox(tmp_path)
    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Aplicacao hermetica com sucesso completo",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"
    assert _get_sqlite_count(env["sqlite_path"]) == 2
    assert _get_jsonl_count(env["jsonl_path"]) == 2

    # Manifesto incrementado
    man = read_runtime_manifest(env["manifest_path"])
    assert man["reconciliation_history_counter"] == 1
    assert man["last_reconciliation_audit_id"] == payload["audit_id"]

    # Ledger com par PREPARED -> COMMITTED
    val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert val["is_valid"] is True
    assert val["completed_transactions_count"] == 1


def test_12_no_changes_hermetic(tmp_path: Path):
    """12. NO_CHANGES hermético: Reexecução de lote já presente retorna Exit 0, zero inserções e ledger limpo."""
    env = _setup_sandbox(tmp_path)
    # Primeira aplicação
    code1, _ = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Primeira aplicacao legitima para popular lote",
        sandbox_root=env["tmp_path"]
    )
    assert code1 == 0

    # Segunda aplicação com o mesmo arquivo (idempotência)
    code2, payload2 = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Segunda aplicacao idêntica testando no-changes",
        sandbox_root=env["tmp_path"]
    )
    assert code2 == 0
    assert payload2["status"] == "NO_CHANGES"
    assert payload2["sqlite_inserted"] == 0
    assert payload2["jsonl_appended"] == 0


def test_13_failed_before_prepared(tmp_path: Path):
    """13. FAILED_BEFORE_PREPARED: Rejeição no preflight retorna Exit 1 e sem mutações."""
    env = _setup_sandbox(tmp_path)
    # Simula erro de preflight corrompendo o SQLite antes de iniciar
    with open(env["sqlite_path"], "wb") as f:
        f.write(b"NOT A SQLITE DATABASE CORRUPTED HEADER")

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Preflight com banco corrompido que deve falhar",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert payload["prepared_recorded"] is False


def test_14_prepared_incomplete(tmp_path: Path):
    """14. PREPARED_INCOMPLETE: Falha pós-PREPARED retorna Exit 4 e requer manual recovery."""
    env = _setup_sandbox(tmp_path)
    # Simula falha ao inserir no SQLite durante o apply
    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Simulando disk error")):
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Falha injetada apos PREPARED confirmada",
        sandbox_root=env["tmp_path"]
        )
    assert exit_code == 4
    assert payload["status"] == "PREPARED_INCOMPLETE"
    assert payload["requires_manual_recovery"] is True


def test_15_persistence_uncertain_in_prepared(tmp_path: Path):
    """15. PERSISTENCE_UNCERTAIN em PREPARED: Falha de I/O no primeiro registro de auditoria retorna Exit 5."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Simulando falha de fsync em PREPARED")):
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Incerteza na durabilidade de PREPARED",
        sandbox_root=env["tmp_path"]
        )
    assert exit_code == 5
    assert payload["status"] == "PERSISTENCE_UNCERTAIN"
    assert payload["requires_manual_recovery"] is True


def test_16_persistence_uncertain_in_committed(tmp_path: Path):
    """16. PERSISTENCE_UNCERTAIN em COMMITTED: Falha de I/O na gravação final de COMMITTED retorna Exit 5."""
    env = _setup_sandbox(tmp_path)
    from events.fallback_reconciler import record_reconciliation_audit as orig_audit

    def broken_audit(p, entry):
        if entry.get("reconciliation_phase") == "COMMITTED":
            raise IOError("Simulando falha fisica no fsync de COMMITTED")
        return orig_audit(p, entry)

    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=broken_audit):
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Incerteza na durabilidade de COMMITTED",
        sandbox_root=env["tmp_path"]
        )
    assert exit_code == 5
    assert payload["status"] == "PERSISTENCE_UNCERTAIN"
    assert payload["requires_manual_recovery"] is True


def test_17_tripartite_serialization(tmp_path: Path):
    """17. Serialização tripartite: payload estruturado contém todas as chaves obrigatórias."""
    env = _setup_sandbox(tmp_path)
    _, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Validacao de campos estruturados serializados",
        sandbox_root=env["tmp_path"]
    )
    required_keys = [
        "status", "exit_code", "audit_id", "environment_uuid", "stage_reached",
        "prepared_recorded", "committed_recorded", "manifest_updated",
        "sqlite_inserted", "jsonl_appended", "errors", "details", "requires_manual_recovery"
    ]
    for rk in required_keys:
        assert rk in payload, f"Chave obrigatória ausente no payload: {rk}"


def test_18_exit_codes_mapping(tmp_path: Path):
    """18. Validação rigorosa dos mapeamentos de códigos de saída (0, 1, 2, 3, 4, 5)."""
    env = _setup_sandbox(tmp_path)

    # 0 -> SUCCESS
    c_ok, _ = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Teste de mapeamento exit 0",
        sandbox_root=env["tmp_path"]
    )
    assert c_ok == 0

    # 1 -> FAILED_BEFORE_PREPARED (ex: reason curto)
    c_fail, _ = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="curto",
        sandbox_root=env["tmp_path"]
    )
    assert c_fail == 1


def test_19_lock_busy_concurrency_conflict(tmp_path: Path):
    """19. Lock ocupado por processo concorrente retorna Exit 3 (CONCURRENCY_LOCK_CONFLICT)."""
    env = _setup_sandbox(tmp_path)
    # Adquire lock concorrente ativo no kernel
    competing_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="competing_process")
    competing_lock.acquire()

    try:
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Concorrencia de lock que deve retornar exit 3",
        sandbox_root=env["tmp_path"]
        )
        assert exit_code == 3
        assert payload["status"] == "FAILED_BEFORE_PREPARED"
        assert any("Conflito de exclusividade" in err or "retido por processo ativo" in err for err in payload["errors"])
    finally:
        competing_lock.release()


def test_20_lock_dirty_handled(tmp_path: Path):
    """20. Lock em estado DIRTY é aceito como elegível para manutenção de reconciliação."""
    env = _setup_sandbox(tmp_path)
    # Simula estado DIRTY anterior
    dirty_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="crashed_holder")
    dirty_lock.acquire()
    dirty_lock.update_state(RuntimeState.DIRTY, reason="crash_simulado")
    # Destrava fisicamente fechando o descritor mantendo DIRTY em disco
    dirty_lock._file_obj.close()
    dirty_lock._is_locked = False

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Sessao de manutencao autorizada sob estado DIRTY previo",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"


def test_21_lock_unknown_handled(tmp_path: Path):
    """21. Lock em estado UNKNOWN é aceito como elegível para manutenção de reconciliação."""
    env = _setup_sandbox(tmp_path)
    unk_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="unk_holder")
    unk_lock.acquire()
    unk_lock.update_state(RuntimeState.UNKNOWN, reason="unknown_crash")
    unk_lock._file_obj.close()
    unk_lock._is_locked = False

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Sessao de manutencao autorizada sob estado UNKNOWN previo",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"


def test_22_lock_running_residual_handled(tmp_path: Path):
    """22. Lock em estado RUNNING sem processo ativo é detectado como RUNNING_RESIDUAL e admitido."""
    env = _setup_sandbox(tmp_path)
    run_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="dead_holder")
    run_lock.acquire()
    # Fecha descritor sem chamar release para simular abandono de processo
    run_lock._file_obj.close()
    run_lock._is_locked = False

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Sessao de manutencao autorizada sob RUNNING_RESIDUAL",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"


def test_23_invalid_manifest_blocked(tmp_path: Path):
    """23. Manifesto inexistente, vazio ou corrompido bloqueia execução fail-closed com Exit 1."""
    env = _setup_sandbox(tmp_path)
    env["manifest_path"].unlink()

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Manifesto ausente que deve falhar fail-closed",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any("Manifesto operacional inválido" in err for err in payload["errors"])


def test_24_divergent_uuid_blocked(tmp_path: Path):
    """24. Inconsistência de UUID no manifesto bloqueia execução fail-closed."""
    env = _setup_sandbox(tmp_path)
    # Grava manifesto com UUID inválido
    with open(env["manifest_path"], "w", encoding="utf-8") as f:
        json.dump({"manifest_version": "1.2", "environment_uuid": "not-a-valid-uuid"}, f)

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="UUID malformado no manifesto bloqueia preflight",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"


def test_25_symlink_forbidden(tmp_path: Path):
    """25. Qualquer caminho configurado como symlink é rejeitado fail-closed."""
    env = _setup_sandbox(tmp_path)
    link_file = env["fb_dir"] / "link_fallback.jsonl"
    try:
        link_file.symlink_to(env["fallback_file"])
    except (OSError, NotImplementedError):
        pytest.skip("Sistema operacional não autorizou criação de symlink sem privilégios")

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=link_file,
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Symlink proibido deve ser rejeitado",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 1
    assert any("link simbólico detectado" in err for err in payload["errors"])


def test_26_git_fixture_forbidden(tmp_path: Path):
    """26. Arquivo de fallback que seja fixture rastreada no Git é bloqueado fail-closed."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.FallbackReconciler.is_git_tracked_fixture", return_value=True):
        exit_code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Fixture rastreada no Git deve ser bloqueada",
        sandbox_root=env["tmp_path"]
        )
    assert exit_code == 1
    assert any("fixture versionada no Git" in err for err in payload["errors"])


def test_27_external_path_blocked(tmp_path: Path):
    """27. Barreira técnica anti-produção: caminho apontando para dados operacionais de produção levanta PermissionError."""
    prod_sqlite = PROJECT_ROOT / "dados" / "trading_bot.db"
    prod_jsonl = PROJECT_ROOT / "dados" / "eventos_fluxo.jsonl"
    prod_archive = PROJECT_ROOT / "dados" / "archive"
    fake_fb = PROJECT_ROOT / "fallback_events" / "fake_fb.jsonl"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=fake_fb,
            sqlite_path=prod_sqlite,
            jsonl_path=prod_jsonl,
            archive_dir=prod_archive,
            sandbox_root=tmp_path,
            operator_reason="Tentativa proibida em caminhos de producao",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_28_invalid_fallback_file(tmp_path: Path):
    """28. Arquivo fallback inexistente ou com extensão inválida é rejeitado antes de mutações."""
    env = _setup_sandbox(tmp_path)
    inv_file = env["fb_dir"] / "inexistente.txt"

    exit_code, payload = execute_apply_integration_hermetic(
        fallback_file=inv_file,
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_dir"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        operator_reason="Arquivo inexistente rejeitado de imediato",
        sandbox_root=env["tmp_path"]
    )
    assert exit_code == 1
    assert any("Arquivo de fallback não encontrado" in err for err in payload["errors"])


def test_29_no_retry(tmp_path: Path):
    """29. Falha durante a transação não dispara retentativa automática."""
    env = _setup_sandbox(tmp_path)
    call_count = 0
    orig_execute = execute_reconciliation_transaction_hermetic

    def tracked_exec(*a, **kw):
        nonlocal call_count
        call_count += 1
        return orig_execute(*a, **kw)

    with mock.patch("tests.helpers.cli_apply_integration_harness.execute_reconciliation_transaction_hermetic", side_effect=tracked_exec):
        # Provoca falha no preflight
        with open(env["sqlite_path"], "wb") as f:
            f.write(b"CORRUPTED")

        code, _ = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Verificacao de ausencia de retry automatico",
        sandbox_root=env["tmp_path"]
        )
        assert code == 1
        assert call_count == 1, "Orquestrador foi invocado mais de uma vez (retry indevido detectado)!"


def test_30_no_automatic_rollback(tmp_path: Path):
    """30. Falha após PREPARED preserva estado sujo e evidências intactas sem auto-rollback silencioso."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Simulando falha")):
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Verificacao de ausencia de rollback automatico",
        sandbox_root=env["tmp_path"]
        )

    # O ledger deve preservar a linha PREPARED como evidência física de estado incompleto
    assert env["ledger_path"].exists()
    val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert val["has_pending_transaction"] is True, "PREPARED foi apagado indevidamente (auto-rollback proibido)!"


def test_31_no_synthetic_committed(tmp_path: Path):
    """31. Estado incompleto nunca gera COMMITTED artificial no ledger."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Simulando falha")):
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Verificacao de ausencia de COMMITTED sintetico",
        sandbox_root=env["tmp_path"]
        )
    val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert val["completed_transactions_count"] == 0


def test_32_no_state_converted_to_clean(tmp_path: Path):
    """32. Falhas transacionais nunca convertem indevidamente o estado persistido do lock para CLEAN."""
    env = _setup_sandbox(tmp_path)
    # Inicia lock em UNKNOWN
    unk_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="unk_tester")
    unk_lock.acquire()
    unk_lock.update_state(RuntimeState.UNKNOWN, reason="crash_inicial")
    unk_lock._file_obj.close()
    unk_lock._is_locked = False

    # Falha durante a transação
    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Falha proposital")):
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Verificacao de conservacao de estado de lock",
        sandbox_root=env["tmp_path"]
        )

    # O estado do lock em disco não pode ter sido convertido para CLEAN
    persisted = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert persisted["status"] != RuntimeState.CLEAN
    assert persisted["status"] == RuntimeState.UNKNOWN


# =============================================================================
# BLOCO 3: TESTES DE HARDENING H01 A H22 (Gate C2-B2.4-B-H1)
# =============================================================================

def test_h01_arbitrary_repo_root_does_not_alter_prod_anchor(tmp_path: Path):
    """H01: repo_root arbitrário não altera a âncora imutável de produção; levanta PermissionError."""
    fake_fb = tmp_path / "fallback_events" / "fb.jsonl"
    fake_fb.parent.mkdir(parents=True, exist_ok=True)
    fake_fb.write_text('{"event_id": "EVT-H01", "epoch_ms": 1700000001000, "tipo_evento": "BUY"}\n')

    real_prod_sqlite = PROJECT_ROOT / "dados" / "trading_bot.db"
    tmp_jsonl = tmp_path / "dados" / "eventos_fluxo.jsonl"
    tmp_archive = tmp_path / "dados" / "archive"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=fake_fb,
            sqlite_path=real_prod_sqlite,
            jsonl_path=tmp_jsonl,
            archive_dir=tmp_archive,
            repo_root=tmp_path,  # Tenta deslocar a âncora para tmp_path
            operator_reason="Tentativa de bypass com repo_root arbitrario",
        sandbox_root=tmp_path
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h02_real_sqlite_rejected_before_open(tmp_path: Path):
    """H02: SQLite operacional real é rejeitado antes de qualquer abertura ou conexão."""
    env = _setup_sandbox(tmp_path)
    real_prod_sqlite = PROJECT_ROOT / "dados" / "trading_bot.db"

    with mock.patch("sqlite3.connect") as mock_conn:
        with pytest.raises(PermissionError) as exc_info:
            execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=real_prod_sqlite,
                jsonl_path=env["jsonl_path"],
                archive_dir=env["archive_dir"],
                backup_dir=env["backup_dir"],
                operator_reason="Tentativa de apontar para SQLite operacional real",
        sandbox_root=env["tmp_path"]
            )
        assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)
        assert mock_conn.call_count == 0, "sqlite3.connect foi chamado indevidamente para banco de produção!"


def test_h03_real_jsonl_rejected_before_open(tmp_path: Path):
    """H03: JSONL operacional real é rejeitado imediatamente antes de abertura."""
    env = _setup_sandbox(tmp_path)
    real_prod_jsonl = PROJECT_ROOT / "dados" / "eventos_fluxo.jsonl"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=real_prod_jsonl,
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            operator_reason="Tentativa de apontar para JSONL operacional real",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h04_real_manifest_rejected(tmp_path: Path):
    """H04: Manifesto operacional real é rejeitado pela barreira anti-produção."""
    env = _setup_sandbox(tmp_path)
    real_prod_manifest = PROJECT_ROOT / "dados" / MANIFEST_FILENAME

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            manifest_path=real_prod_manifest,
            operator_reason="Tentativa de apontar para manifesto operacional real",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h05_real_ledger_rejected(tmp_path: Path):
    """H05: Livro-razão (ledger) operacional real é rejeitado pela barreira anti-produção."""
    env = _setup_sandbox(tmp_path)
    real_prod_ledger = PROJECT_ROOT / "dados" / RECONCILIATION_AUDIT_FILENAME

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            ledger_path=real_prod_ledger,
            operator_reason="Tentativa de apontar para ledger operacional real",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h06_real_lock_rejected(tmp_path: Path):
    """H06: Lock operacional real é rejeitado pela barreira anti-produção."""
    env = _setup_sandbox(tmp_path)
    real_prod_lock = PROJECT_ROOT / "dados" / ".operational_runtime.lock"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=real_prod_lock,
            operator_reason="Tentativa de apontar para lock operacional real",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h07_real_backup_rejected(tmp_path: Path):
    """H07: Diretório de backup apontando para dados de produção é rejeitado."""
    env = _setup_sandbox(tmp_path)
    real_prod_backup = PROJECT_ROOT / "dados" / "backups"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=real_prod_backup,
            operator_reason="Tentativa de apontar backup para dentro de dados operacionais",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h08_real_fallback_in_prod_rejected(tmp_path: Path):
    """H08: Arquivo de fallback localizado dentro de dados de produção é rejeitado."""
    env = _setup_sandbox(tmp_path)
    real_prod_fb = PROJECT_ROOT / "dados" / "fake_fb.jsonl"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=real_prod_fb,
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            operator_reason="Tentativa de passar fallback dentro de dados operacionais",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)


def test_h09_mixed_tmp_sqlite_and_real_ledger_rejected(tmp_path: Path):
    """H09: Mistura de SQLite temporário e ledger operacional de produção é rejeitada."""
    env = _setup_sandbox(tmp_path)
    real_prod_ledger = PROJECT_ROOT / "dados" / RECONCILIATION_AUDIT_FILENAME

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            ledger_path=real_prod_ledger,
            operator_reason="Tentativa de misturar SQLite tmp com ledger de producao",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value) or "INCONSISTÊNCIA DE SANDBOX" in str(exc_info.value)


def test_h10_mixed_tmp_manifest_and_real_lock_rejected(tmp_path: Path):
    """H10: Mistura de manifesto temporário e lock operacional de produção é rejeitada."""
    env = _setup_sandbox(tmp_path)
    real_prod_lock = PROJECT_ROOT / "dados" / ".operational_runtime.lock"

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=real_prod_lock,
            operator_reason="Tentativa de misturar manifesto tmp com lock de producao",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value) or "INCONSISTÊNCIA DE SANDBOX" in str(exc_info.value)


def test_h11_incoherent_paths_outside_sandbox_rejected(tmp_path: Path):
    """H11: Caminho de dados externo ou com pai divergente de data_dir é rejeitado."""
    env = _setup_sandbox(tmp_path)
    other_dir = tmp_path / "other_foreign_dir"
    other_dir.mkdir(parents=True, exist_ok=True)
    divergent_jsonl = other_dir / "eventos_fluxo.jsonl"
    divergent_jsonl.touch()

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=divergent_jsonl,
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            operator_reason="Tentativa de passar jsonl com parent divergente de data_dir",
        sandbox_root=env["tmp_path"]
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "INCONSISTÊNCIA DE SANDBOX" in str(exc_info.value)


def test_h12_symlink_final_mock_deterministic(tmp_path: Path):
    """H12: Symlink final em qualquer arquivo é rejeitado determinísticamente via mock sem privilégios de SO."""
    env = _setup_sandbox(tmp_path)
    target_resolved = env["fallback_file"].resolve()

    def fake_is_symlink(self):
        return self == target_resolved

    with mock.patch.object(Path, "is_symlink", side_effect=fake_is_symlink, autospec=True):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Teste deterministico de mock para symlink final",
        sandbox_root=env["tmp_path"]
        )
    assert code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any("link simbólico detectado" in err for err in payload["errors"])


def test_h13_symlink_intermediate_rejected(tmp_path: Path):
    """H13: Symlink em componente intermediário de diretório é rejeitado fail-closed."""
    env = _setup_sandbox(tmp_path)
    target_parent = env["data_dir"].resolve()

    def fake_is_symlink(self):
        return self == target_parent

    with mock.patch.object(Path, "is_symlink", side_effect=fake_is_symlink, autospec=True):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Teste deterministico de mock para symlink intermediario",
        sandbox_root=env["tmp_path"]
        )
    assert code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any("link simbólico" in err for err in payload["errors"])


def test_h14_cli_public_remains_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """H14: CLI pública permanece incondicionalmente bloqueada por APPLY_NOT_ENABLED_C2A (Exit 1)."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste para H14 com mais de 15 caracteres",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    assert "APPLY_NOT_ENABLED_C2A" in captured.err


def test_h15_dry_run_remains_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """H15: Modo dry-run continua read-only, sai com Exit 0 e não realiza escritas."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--dry-run",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--fallback-dir", str(env["fb_dir"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 0
    assert _get_sqlite_count(env["sqlite_path"]) == 0
    assert _get_jsonl_count(env["jsonl_path"]) == 0


def test_h16_real_lock_conflict_returns_exit_3(tmp_path: Path):
    """H16: Conflito físico de concorrência com processo ativo retorna Exit 3 coerente."""
    env = _setup_sandbox(tmp_path)
    competing_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="competing_process")
    competing_lock.acquire()

    try:
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Concorrencia real de processo ativo no kernel",
        sandbox_root=env["tmp_path"]
        )
        assert code == 3
        assert payload["stage_reached"] == "LOCK_ACQUISITION"
    finally:
        competing_lock.release()


def test_h17_textual_error_not_in_lock_acquisition_returns_exit_1(tmp_path: Path):
    """H17: Erro textual contendo 'Conflito de exclusividade' em estágio que não seja LOCK_ACQUISITION retorna Exit 1."""
    env = _setup_sandbox(tmp_path)
    fake_res = ReconciliationTransactionResult(
        audit_id="00000000-0000-0000-0000-000000000000",
        environment_uuid=env["env_uuid"],
        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
        stage_reached="PREFLIGHT",  # Não é LOCK_ACQUISITION!
        errors=["Simulando texto acidental: Conflito de exclusividade detectado em fixture."],
    )

    with mock.patch("tests.helpers.cli_apply_integration_harness.execute_reconciliation_transaction_hermetic", return_value=fake_res):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Texto com conflito sem ser no estágio de aquisição",
        sandbox_root=env["tmp_path"]
        )
    assert code == 1, "Erro fora de LOCK_ACQUISITION foi erroneamente classificado como Exit 3!"
    assert payload["exit_code"] == 1


def test_h18_persistence_uncertain_preserves_exit_5(tmp_path: Path):
    """H18: PERSISTENCE_UNCERTAIN preserva Exit Code 5 e marca requires_manual_recovery=True."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.record_reconciliation_audit", side_effect=IOError("Falha fisica")):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Incerteza de persistencia preservando exit 5",
        sandbox_root=env["tmp_path"]
        )
    assert code == 5
    assert payload["status"] == "PERSISTENCE_UNCERTAIN"
    assert payload["requires_manual_recovery"] is True


def test_h19_prepared_incomplete_preserves_exit_4(tmp_path: Path):
    """H19: PREPARED_INCOMPLETE preserva Exit Code 4 e marca requires_manual_recovery=True."""
    env = _setup_sandbox(tmp_path)
    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Falha SQLite")):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Falha pos-prepared preservando exit 4",
        sandbox_root=env["tmp_path"]
        )
    assert code == 4
    assert payload["status"] == "PREPARED_INCOMPLETE"
    assert payload["requires_manual_recovery"] is True


def test_h20_no_runtime_state_changed_to_clean(tmp_path: Path):
    """H20: Falhas transacionais nunca convertem estado persistido de erro (DIRTY) para CLEAN."""
    env = _setup_sandbox(tmp_path)
    dirty_lock = OperationalRuntimeLock(lock_path=env["lock_path"], holder_name="crashed_holder")
    dirty_lock.acquire()
    dirty_lock.update_state(RuntimeState.DIRTY, reason="crash_inicial_h20")
    dirty_lock._file_obj.close()
    dirty_lock._is_locked = False

    with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Falha H20")):
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            operator_reason="Preservacao do estado DIRTY sob falha",
        sandbox_root=env["tmp_path"]
        )

    persisted = OperationalRuntimeLock.read_persisted_state(env["lock_path"])
    assert persisted["status"] == RuntimeState.DIRTY


def test_h21_no_retry_or_automatic_rollback(tmp_path: Path):
    """H21: Sem retentativa automática e sem rollback automático que expurgue evidências."""
    env = _setup_sandbox(tmp_path)
    call_count = 0
    orig_execute = execute_reconciliation_transaction_hermetic

    def counting_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_execute(*args, **kwargs)

    with mock.patch("tests.helpers.cli_apply_integration_harness.execute_reconciliation_transaction_hermetic", side_effect=counting_exec):
        with mock.patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("Falha H21")):
            code, payload = execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=env["sqlite_path"],
                jsonl_path=env["jsonl_path"],
                archive_dir=env["archive_dir"],
                backup_dir=env["backup_dir"],
                lock_path=env["lock_path"],
                manifest_path=env["manifest_path"],
                ledger_path=env["ledger_path"],
                operator_reason="Verificacao de unicidade de chamada sem auto-rollback",
        sandbox_root=env["tmp_path"]
            )
            assert code == 4
            assert call_count == 1, "Houve retentativa indevida!"

    # Evidência física PREPARED permanece intacta no ledger
    val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert val["has_pending_transaction"] is True, "PREPARED foi apagado indevidamente por auto-rollback!"


def test_h22_internal_function_no_operational_bypass(tmp_path: Path):
    """H22: A função interna não oferece bypass operacional por variantes de argumentos arbitrários."""
    env = _setup_sandbox(tmp_path)
    real_prod_dir = PROJECT_ROOT / "dados"

    permutations = [
        {"sqlite_path": real_prod_dir / "trading_bot.db"},
        {"jsonl_path": real_prod_dir / "eventos_fluxo.jsonl"},
        {"manifest_path": real_prod_dir / MANIFEST_FILENAME},
        {"ledger_path": real_prod_dir / RECONCILIATION_AUDIT_FILENAME},
        {"lock_path": real_prod_dir / ".operational_runtime.lock"},
        {"backup_dir": real_prod_dir / "backups"},
        {"fallback_file": real_prod_dir / "fake.jsonl"},
    ]

    for variant in permutations:
        kw = {
            "fallback_file": env["fallback_file"],
            "sqlite_path": env["sqlite_path"],
            "jsonl_path": env["jsonl_path"],
            "archive_dir": env["archive_dir"],
            "backup_dir": env["backup_dir"],
            "lock_path": env["lock_path"],
            "manifest_path": env["manifest_path"],
            "ledger_path": env["ledger_path"],
            "operator_reason": "Tentativa de permutacao com caminhos operacionais",
        }
        kw.update(variant)
        with pytest.raises(PermissionError) as exc_info:
            execute_apply_integration_hermetic(**kw)
        assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value) or "INCONSISTÊNCIA DE SANDBOX" in str(exc_info.value)



# =============================================================================
# BLOCO 4: TESTES ARQUITETURAIS DE ISOLAMENTO H2-01 A H2-18 (Gate C2-B2.4-B-H2)
# =============================================================================

def test_h2_01_cli_production_module_does_not_export_integration_function():
    """H2-01: Módulo CLI de produção não exporta nem contém execute_apply_integration_hermetic()."""
    import scripts.maintenance.reconcile_fallback_events as rfe
    assert not hasattr(rfe, "execute_apply_integration_hermetic"), (
        "VULNERABILIDADE ARQUITETURAL: scripts.maintenance.reconcile_fallback_events "
        "continua exportando execute_apply_integration_hermetic!"
    )
    if hasattr(rfe, "__all__"):
        assert "execute_apply_integration_hermetic" not in rfe.__all__


def test_h2_02_cli_public_apply_strictly_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """H2-02: CLI pública com --apply continua estritamente bloqueada por APPLY_NOT_ENABLED_C2A (Exit 1)."""
    env = _setup_sandbox(tmp_path)
    test_args = [
        "reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason", "Justificativa valida de teste H2-02 com mais de 15 caracteres",
        "--sqlite-path", str(env["sqlite_path"]),
        "--jsonl-path", str(env["jsonl_path"]),
        "--lock-path", str(env["lock_path"]),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    code = cli_main()
    assert code == 1
    captured = capsys.readouterr()
    assert "APPLY_NOT_ENABLED_C2A" in captured.err


def test_h2_03_harness_rejects_missing_or_none_sandbox_context(tmp_path: Path):
    """H2-03: Harness rejeita execução sem contexto de sandbox válido (sandbox_root=None)."""
    env = _setup_sandbox(tmp_path)
    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            sandbox_root=None,
            operator_reason="Tentativa sem contexto de sandbox valido",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_04_harness_accepts_valid_sandbox_context(tmp_path: Path):
    """H2-04: Harness aceita execução legítima exclusivamente em sandbox criado pelo teste."""
    env = _setup_sandbox(tmp_path)
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
        operator_reason="Execucao legitima em sandbox controlado de teste",
    )
    assert exit_code == 0
    assert payload["status"] == "SUCCESS"
    assert payload["sqlite_inserted"] == 2
    assert payload["jsonl_appended"] == 2


def test_h2_05_external_sqlite_rejected_before_connect(tmp_path: Path):
    """H2-05: SQLite externo ao sandbox é rejeitado antes de conexão física."""
    env = _setup_sandbox(tmp_path)
    external_sqlite = Path("D:/external_fake_bot/dados/trading_bot.db") if sys.platform == "win32" else Path("/var/external_bot/dados/trading_bot.db")

    with mock.patch("sqlite3.connect") as mock_conn:
        with pytest.raises(PermissionError) as exc_info:
            execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=external_sqlite,
                jsonl_path=env["jsonl_path"],
                archive_dir=env["archive_dir"],
                backup_dir=env["backup_dir"],
                sandbox_root=env["tmp_path"],
                operator_reason="SQLite externo ao sandbox autorizado",
            )
        assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)
        assert mock_conn.call_count == 0


def test_h2_06_external_jsonl_rejected_before_open(tmp_path: Path):
    """H2-06: JSONL externo ao sandbox é rejeitado antes de qualquer abertura."""
    env = _setup_sandbox(tmp_path)
    external_jsonl = Path("D:/external_fake_bot/dados/eventos.jsonl") if sys.platform == "win32" else Path("/var/external_bot/dados/eventos.jsonl")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=external_jsonl,
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            sandbox_root=env["tmp_path"],
            operator_reason="JSONL externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_07_external_manifest_rejected(tmp_path: Path):
    """H2-07: Manifesto externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_manifest = Path("D:/external_fake_bot/dados/manifest.json") if sys.platform == "win32" else Path("/var/external_bot/dados/manifest.json")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            manifest_path=external_manifest,
            sandbox_root=env["tmp_path"],
            operator_reason="Manifesto externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_08_external_ledger_rejected(tmp_path: Path):
    """H2-08: Ledger externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_ledger = Path("D:/external_fake_bot/dados/reconciliation_audit.jsonl") if sys.platform == "win32" else Path("/var/external_bot/dados/reconciliation_audit.jsonl")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            ledger_path=external_ledger,
            sandbox_root=env["tmp_path"],
            operator_reason="Ledger externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_09_external_lock_rejected(tmp_path: Path):
    """H2-09: Lock externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_lock = Path("D:/external_fake_bot/dados/.operational_runtime.lock") if sys.platform == "win32" else Path("/var/external_bot/dados/.operational_runtime.lock")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=external_lock,
            sandbox_root=env["tmp_path"],
            operator_reason="Lock externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_10_external_backup_rejected(tmp_path: Path):
    """H2-10: Diretório de backup externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_backup = Path("D:/external_fake_bot/backups") if sys.platform == "win32" else Path("/var/external_bot/backups")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=external_backup,
            sandbox_root=env["tmp_path"],
            operator_reason="Backup externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_11_external_fallback_rejected(tmp_path: Path):
    """H2-11: Arquivo fallback externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_fb = Path("D:/external_fake_bot/fb.jsonl") if sys.platform == "win32" else Path("/var/external_bot/fb.jsonl")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=external_fb,
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            sandbox_root=env["tmp_path"],
            operator_reason="Fallback externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_12_external_archive_rejected(tmp_path: Path):
    """H2-12: Diretório Archive externo ao sandbox é rejeitado."""
    env = _setup_sandbox(tmp_path)
    external_archive = Path("D:/external_fake_bot/archive") if sys.platform == "win32" else Path("/var/external_bot/archive")

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=external_archive,
            backup_dir=env["backup_dir"],
            sandbox_root=env["tmp_path"],
            operator_reason="Archive externo ao sandbox autorizado",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)


def test_h2_13_second_operational_installation_rejected_without_opening_files(tmp_path: Path):
    """H2-13: Segunda instalação operacional completa é rejeitada sem abrir arquivos reais."""
    sandbox_legit = tmp_path / "sandbox_autorizado"
    sandbox_legit.mkdir(parents=True, exist_ok=True)
    env = _setup_sandbox(sandbox_legit)

    second_install = tmp_path / "segundo_bot_operacional" / "dados"
    second_install.mkdir(parents=True, exist_ok=True)
    fake_second_db = second_install / "trading_bot.db"
    fake_second_db.touch()

    with mock.patch("sqlite3.connect") as mock_conn:
        with pytest.raises(PermissionError) as exc_info:
            execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=fake_second_db,
                jsonl_path=second_install / "eventos_fluxo.jsonl",
                archive_dir=second_install / "archive",
                sandbox_root=sandbox_legit,
                operator_reason="Tentativa de apontar para segunda instalacao operacional",
            )
        assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value)
        assert mock_conn.call_count == 0


def test_h2_14_mixture_of_two_different_sandboxes_rejected(tmp_path: Path):
    """H2-14: Mistura de artefatos de dois sandboxes distintos é rejeitada."""
    sb1 = tmp_path / "sandbox_1"
    sb2 = tmp_path / "sandbox_2"
    sb1.mkdir(parents=True, exist_ok=True)
    sb2.mkdir(parents=True, exist_ok=True)

    env1 = _setup_sandbox(sb1)
    env2 = _setup_sandbox(sb2)

    with pytest.raises(PermissionError) as exc_info:
        execute_apply_integration_hermetic(
            fallback_file=env1["fallback_file"],
            sqlite_path=env1["sqlite_path"],
            jsonl_path=env1["jsonl_path"],
            archive_dir=env1["archive_dir"],
            backup_dir=env1["backup_dir"],
            manifest_path=env2["manifest_path"],
            sandbox_root=sb1,
            operator_reason="Tentativa de misturar dois sandboxes distintos",
        )
    assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "INCONSISTÊNCIA DE SANDBOX" in str(exc_info.value)


def test_h2_15_symlink_final_rejected_deterministic(tmp_path: Path):
    """H2-15: Symlink final é rejeitado via mock determinístico na checagem pré-resolve."""
    env = _setup_sandbox(tmp_path)
    target_fb = env["fallback_file"]

    def fake_is_symlink(self):
        return self == target_fb

    with mock.patch.object(Path, "is_symlink", side_effect=fake_is_symlink, autospec=True):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            sandbox_root=env["tmp_path"],
            operator_reason="Symlink final pre-resolve detectado",
        )
    assert code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any("link simbólico detectado" in err for err in payload["errors"])


def test_h2_16_symlink_intermediate_rejected_pre_resolve(tmp_path: Path):
    """H2-16: Symlink intermediário é inspecionado e rejeitado antes de qualquer resolve()."""
    env = _setup_sandbox(tmp_path)
    target_parent = env["data_dir"]

    def fake_is_symlink(self):
        return self == target_parent

    with mock.patch.object(Path, "is_symlink", side_effect=fake_is_symlink, autospec=True):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            sandbox_root=env["tmp_path"],
            operator_reason="Symlink intermediario pre-resolve detectado",
        )
    assert code == 1
    assert payload["status"] == "FAILED_BEFORE_PREPARED"
    assert any("link simbólico" in err for err in payload["errors"])


def test_h2_17_lock_exit_preserves_fail_closed_classification(tmp_path: Path):
    """H2-17: Saída de lock preserva classificação fail-closed (Exit 1 em dúvida, Exit 3 em concorrência)."""
    env = _setup_sandbox(tmp_path)
    fake_res = ReconciliationTransactionResult(
        audit_id="00000000-0000-0000-0000-000000000000",
        environment_uuid=env["env_uuid"],
        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
        stage_reached="PREFLIGHT",
        errors=["Mensagem acidental com Conflito de exclusividade em preflight."],
    )

    with mock.patch("tests.helpers.cli_apply_integration_harness.execute_reconciliation_transaction_hermetic", return_value=fake_res):
        code, payload = execute_apply_integration_hermetic(
            fallback_file=env["fallback_file"],
            sqlite_path=env["sqlite_path"],
            jsonl_path=env["jsonl_path"],
            archive_dir=env["archive_dir"],
            backup_dir=env["backup_dir"],
            lock_path=env["lock_path"],
            manifest_path=env["manifest_path"],
            ledger_path=env["ledger_path"],
            sandbox_root=env["tmp_path"],
            operator_reason="Verificacao de classificacao fail-closed",
        )
    assert code == 1, "Erro em PREFLIGHT não pode retornar Exit 3!"
    assert payload["exit_code"] == 1


def test_h2_18_direct_harness_call_cannot_bypass_sandbox_containment(tmp_path: Path):
    """H2-18: Chamada direta ao harness não permite ignorar a contenção do sandbox por argumentos arbitrários."""
    env = _setup_sandbox(tmp_path)
    real_prod = PROJECT_ROOT / "dados"

    permutations = [
        {"sqlite_path": real_prod / "trading_bot.db"},
        {"jsonl_path": real_prod / "eventos_fluxo.jsonl"},
        {"manifest_path": real_prod / MANIFEST_FILENAME},
        {"ledger_path": real_prod / RECONCILIATION_AUDIT_FILENAME},
        {"lock_path": real_prod / ".operational_runtime.lock"},
        {"backup_dir": real_prod / "backups"},
        {"fallback_file": real_prod / "fake.jsonl"},
    ]

    for variant in permutations:
        kw = {
            "fallback_file": env["fallback_file"],
            "sqlite_path": env["sqlite_path"],
            "jsonl_path": env["jsonl_path"],
            "archive_dir": env["archive_dir"],
            "backup_dir": env["backup_dir"],
            "lock_path": env["lock_path"],
            "manifest_path": env["manifest_path"],
            "ledger_path": env["ledger_path"],
            "sandbox_root": env["tmp_path"],
            "operator_reason": "Tentativa direta de bypass de sandbox",
        }
        kw.update(variant)
        with pytest.raises(PermissionError) as exc_info:
            execute_apply_integration_hermetic(**kw)
        assert "CONTESTAÇÃO DE SANDBOX" in str(exc_info.value) or "PRODUÇÃO BLOQUEADA" in str(exc_info.value)
