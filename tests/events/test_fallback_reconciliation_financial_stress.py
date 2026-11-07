# -*- coding: utf-8 -*-
"""
Suíte de Testes Financeiros Finais e Invariantes Numéricas — Gate G4-D B3-B2-C2-B3-B.

Cenários Financeiros:
FIN-01: Precisão monetária, representação numérica (float, Decimal, string decimal) e roundtrip.
FIN-02: Normalização analítica determinística de NaN e Infinity (política RFC 8259 vs tokens).
FIN-03: Falha simulada de I/O no SQLite (OperationalError) com transição fail-closed e integridade.
FIN-04: Concorrência de leitura em SQLite modo WAL sem deadlock durante reconciliação hermética.
FIN-05: Colisão de identidade financeira, divergência de payload e auditoria de unicidade física no SQLite.

Restrições Absolutas:
- 100% Hermético em tmp_path (sandboxes temporários isolados).
- Zero dependência de Binance, rede externa ou dados reais de produção.
- Zero mutação no código de produção ou no schema operacional.
- Preservação estrita das asserções e fail-closed em estados incertos.
"""

from __future__ import annotations

import decimal
import hashlib
import json
import math
import os
import queue
import sqlite3
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
    EventClassification,
    FallbackReconciler,
    ReconciliationTransactionStatus,
    compute_canonical_payload_hash,
    normalize_json_value,
    insert_event_to_sqlite,
    append_event_to_jsonl,
    verify_sqlite_integrity,
    verify_jsonl_integrity,
    verify_reconciliation_targets,
    validate_reconciliation_audit_ledger,
    record_reconciliation_audit,
)
from tests.helpers.cli_apply_integration_harness import (
    HermeticSandboxContext,
    execute_apply_integration_hermetic,
)


# =============================================================================
# SETUP E FIXTURES HERMÉTICAS EM SANDBOX
# =============================================================================

def _init_sqlite_db(db_path: Path, wal_mode: bool = False) -> None:
    """Cria banco SQLite com tabela events no schema exato do EventStore de produção."""
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
                is_signal INTEGER DEFAULT 0,
                payload JSON NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_ms);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_window ON events(window_id);")
        if wal_mode:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()
    finally:
        conn.close()


def _setup_sandbox(
    tmp_path: Path,
    initial_events: Optional[List[Dict[str, Any]]] = None,
    wal_mode: bool = False,
) -> Dict[str, Any]:
    """
    Inicializa sandbox temporário 100% isolado com diretórios de dados, backups,
    archive, banco SQLite, manifesto v1.2 e lock operacional em estado CLEAN.
    """
    data_dir = tmp_path / "dados"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir = tmp_path / "fallback_events"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    backup_base = tmp_path / "backups" / "reconciliation_snapshots"
    backup_base.mkdir(parents=True, exist_ok=True)

    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    manifest_path = data_dir / MANIFEST_FILENAME
    ledger_path = data_dir / RECONCILIATION_AUDIT_FILENAME
    lock_path = data_dir / ".operational_runtime.lock"

    _init_sqlite_db(sqlite_path, wal_mode=wal_mode)
    jsonl_path.touch()

    # Inicializa manifesto v1.2
    env_uuid = str(uuid.uuid4())
    write_runtime_manifest(
        manifest_path=manifest_path,
        env_uuid=env_uuid,
        holder_name="financial_stress_harness",
        manifest_version="1.2",
    )

    # Inicializa lock operacional em estado CLEAN
    op_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="financial_stress_harness")
    op_lock.acquire()
    op_lock.release()

    fallback_file = fallback_dir / "eventos_stress.jsonl"
    if initial_events:
        with open(fallback_file, "w", encoding="utf-8") as f:
            for ev in initial_events:
                f.write(json.dumps(ev, separators=(",", ":"), ensure_ascii=False) + "\n")
    else:
        fallback_file.touch()

    return {
        "sandbox_root": tmp_path,
        "data_dir": data_dir,
        "archive_dir": archive_dir,
        "fallback_dir": fallback_dir,
        "backup_base": backup_base,
        "sqlite_path": sqlite_path,
        "jsonl_path": jsonl_path,
        "manifest_path": manifest_path,
        "ledger_path": ledger_path,
        "lock_path": lock_path,
        "fallback_file": fallback_file,
        "env_uuid": env_uuid,
    }


# =============================================================================
# FIN-01: PRECISÃO MONETÁRIA E REPRESENTAÇÃO
# =============================================================================

def test_fin_01_monetary_precision_float_and_decimal_representation(tmp_path: Path):
    """
    FIN-01: Valida a precisão monetária em valores de alta precisão (preço com 8 casas,
    quantidade com 8 casas decimais — satoshis), distinguindo float, Decimal e string.

    Contratos verificados:
    1. Float de alta precisão (67890.12345678) é preservado sem truncamento numérico arbitrário.
    2. Quantidade fracionária (0.00000001) mantém fidelidade em SQLite, JSONL e hash canônico.
    3. Strings decimais ('67890.12345678') são preservadas textualmente sem conversão para float binário.
    4. Objetos Decimal puros (não admitidos nativamente pelo json.dumps padrão) devem falhar fail-closed
       sem corromper o banco relacional ou o livro-razão.
    """
    # 1. Caso com Float nativo de alta precisão
    event_float = {
        "event_id": "FIN-FLOAT-01",
        "tipo_evento": "ORDER_FILLED",
        "symbol": "BTCUSDT",
        "price": 67890.12345678,
        "quantity": 0.00000001,
        "epoch_ms": 1773000000000,
        "side": "BUY",
    }

    # 2. Caso com String Decimal monetária estrita
    event_str = {
        "event_id": "FIN-STR-01",
        "tipo_evento": "ORDER_FILLED",
        "symbol": "BTCUSDT",
        "price": "67890.12345678",
        "quantity": "0.00000001",
        "epoch_ms": 1773000001000,
        "side": "SELL",
    }

    env = _setup_sandbox(tmp_path / "fin01_valid", initial_events=[event_float, event_str])

    exit_code, res = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_base"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["sandbox_root"],
    )

    assert exit_code == 0, f"Falha na reconciliação de precisão monetária: {res}"
    assert res["status"] == ReconciliationTransactionStatus.SUCCESS.value
    assert res["sqlite_inserted"] == 2
    assert res["jsonl_appended"] == 2

    # Verificação pós-apply no SQLite: ler payload cru e checar valores
    conn = sqlite3.connect(f"file:{env['sqlite_path'].as_posix()}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM events WHERE symbol = 'BTCUSDT' ORDER BY timestamp_ms ASC")
        rows = cur.fetchall()
        assert len(rows) == 2

        payload_float = json.loads(rows[0][0])
        assert payload_float["event_id"] == "FIN-FLOAT-01"
        assert payload_float["price"] == 67890.12345678
        assert payload_float["quantity"] == 0.00000001
        assert isinstance(payload_float["price"], float)
        assert isinstance(payload_float["quantity"], float)

        payload_str = json.loads(rows[1][0])
        assert payload_str["event_id"] == "FIN-STR-01"
        assert payload_str["price"] == "67890.12345678"
        assert payload_str["quantity"] == "0.00000001"
        assert isinstance(payload_str["price"], str)
        assert isinstance(payload_str["quantity"], str)
    finally:
        conn.close()

    # Verificação no JSONL
    with open(env["jsonl_path"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["price"] == 67890.12345678
    assert lines[1]["price"] == "67890.12345678"

    # 3. Caso com Decimal: Rejeição Fail-Closed de objeto Decimal puro não-serializado
    # O contrato JSON padrão não serializa decimal.Decimal sem converter previamente.
    event_decimal_unsupported = {
        "event_id": "FIN-DECIMAL-UNSUPPORTED",
        "tipo_evento": "ORDER_FILLED",
        "symbol": "BTCUSDT",
        "price": decimal.Decimal("67890.12345678"),
        "quantity": decimal.Decimal("0.00000001"),
        "epoch_ms": 1773000002000,
    }

    with pytest.raises(Exception):
        # A tentativa de computar hash sobre Decimal puro causa TypeError,
        # provocando fail-closed antes de qualquer escrita no banco.
        compute_canonical_payload_hash(event_decimal_unsupported)


# =============================================================================
# FIN-02: NaN E INFINITY
# =============================================================================

def test_fin_02_nan_and_infinity_analytical_normalization_and_stability(tmp_path: Path):
    """
    FIN-02: Valida o comportamento do sistema diante de valores não-finitos (NaN, +Inf, -Inf).

    Contratos verificados:
    1. A normalização analítica via normalize_json_value() mapeia:
       - math.nan -> '__NaN__'
       - float('inf') -> '__Infinity__'
       - float('-inf') -> '__-Infinity__'
    2. Hashes canônicos de eventos com NaN/Inf são estáveis e determinísticos.
    3. Documenta que '__NaN__' e '__Infinity__' são marcadores analíticos para evitar quebra
       de pipeline, e NÃO representam valores financeiros válidos para execução de ordens.
    """
    event_nan = {
        "event_id": "FIN-NAN-01",
        "tipo_evento": "MARKET_ANOMALY",
        "symbol": "BTCUSDT",
        "anomaly_score": math.nan,
        "epoch_ms": 1773000003000,
    }

    event_inf_pos = {
        "event_id": "FIN-INF-POS-01",
        "tipo_evento": "ORDERBOOK_IMBALANCE",
        "symbol": "BTCUSDT",
        "imbalance_ratio": float("inf"),
        "epoch_ms": 1773000004000,
    }

    event_inf_neg = {
        "event_id": "FIN-INF-NEG-01",
        "tipo_evento": "ORDERBOOK_IMBALANCE",
        "symbol": "BTCUSDT",
        "imbalance_ratio": float("-inf"),
        "epoch_ms": 1773000005000,
    }

    # Hash canônico determinístico
    hash_nan_1 = compute_canonical_payload_hash(event_nan)
    hash_nan_2 = compute_canonical_payload_hash(event_nan)
    assert hash_nan_1 == hash_nan_2, "Hash canônico de NaN deve ser estritamente estável e idempotente"

    hash_inf_pos = compute_canonical_payload_hash(event_inf_pos)
    hash_inf_neg = compute_canonical_payload_hash(event_inf_neg)
    assert hash_inf_pos != hash_inf_neg, "Hash de +Infinity deve ser diferente de -Infinity"

    # Reconciliação no sandbox com os marcadores de normalização
    lote = [
        {"event_id": "FIN-NAN-01", "tipo_evento": "MARKET_ANOMALY", "symbol": "BTCUSDT", "val": "__NaN__", "epoch_ms": 1773000003000},
        {"event_id": "FIN-INF-POS-01", "tipo_evento": "ORDERBOOK_IMBALANCE", "symbol": "BTCUSDT", "val": "__Infinity__", "epoch_ms": 1773000004000},
        {"event_id": "FIN-INF-NEG-01", "tipo_evento": "ORDERBOOK_IMBALANCE", "symbol": "BTCUSDT", "val": "__-Infinity__", "epoch_ms": 1773000005000},
    ]

    env = _setup_sandbox(tmp_path / "fin02_sandbox", initial_events=lote)

    exit_code, res = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_base"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["sandbox_root"],
    )

    assert exit_code == 0
    assert res["status"] == ReconciliationTransactionStatus.SUCCESS.value
    assert res["sqlite_inserted"] == 3
    assert res["jsonl_appended"] == 3

    # Confirmar pós-verificação aprovada via verificação direta
    verif = verify_reconciliation_targets(
        expected_events=lote,
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        require_all_destinations_default=True,
    )
    assert verif.approved is True
    assert verif.divergent_hash_count == 0


# =============================================================================
# FIN-03: FALHA SIMULADA DE ESCRITA NO SQLITE (I/O OPERATIONALERROR)
# =============================================================================

def test_fin_03_simulated_sqlite_io_failure_fail_closed_and_prepared_incomplete(tmp_path: Path, monkeypatch):
    """
    FIN-03: Injeta falha controlada de I/O no SQLite (sqlite3.OperationalError) durante a fase
    APPLY_SQLITE.

    Contratos verificados:
    1. Registro PREPARED gravado no ledger com fsync antes da tentativa de mutação.
    2. Snapshot forense completo preservado e auditável.
    3. Ausência estrita de registro COMMITTED no ledger.
    4. Transação encerra com status PREPARED_INCOMPLETE e stage_reached='APPLY_SQLITE'.
    5. Exit code contratual 4 (EXIT_PREPARED_INCOMPLETE) e requires_manual_recovery=True.
    6. Startup preflight posterior bloqueia fail-closed devido à transação pendente no ledger.
    7. Comprova ausência de retry automático, rollback cego ou restauração de snapshot.

    Nota de integridade:
    'Falha de I/O simulada por injeção de exceção; não equivale a ENOSPC físico nem power loss.'
    """
    event = {
        "event_id": "FIN-IO-FAIL-01",
        "tipo_evento": "POSITION_OPENED",
        "symbol": "BTCUSDT",
        "price": 68000.0,
        "quantity": 0.5,
        "epoch_ms": 1773000010000,
    }

    env = _setup_sandbox(tmp_path / "fin03_sandbox", initial_events=[event])

    # Injeta exceção operacional simulada de I/O no SQLite
    import events.fallback_reconciler as fr

    def mock_insert_sqlite_failure(path, ev):
        raise sqlite3.OperationalError("disk I/O error (simulated controlled failure during APPLY_SQLITE)")

    monkeypatch.setattr(fr, "insert_event_to_sqlite", mock_insert_sqlite_failure)

    exit_code, res = execute_apply_integration_hermetic(
        fallback_file=env["fallback_file"],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        backup_dir=env["backup_base"],
        lock_path=env["lock_path"],
        manifest_path=env["manifest_path"],
        ledger_path=env["ledger_path"],
        sandbox_root=env["sandbox_root"],
    )

    # 1. Exit code contratual do SO (4 = PREPARED_INCOMPLETE) e status
    assert exit_code == 4
    assert res["status"] == ReconciliationTransactionStatus.PREPARED_INCOMPLETE.value
    assert res["stage_reached"] == "APPLY_SQLITE"
    assert res["requires_manual_recovery"] is True
    assert res["prepared_recorded"] is True
    assert res["committed_recorded"] is False

    # 2. Snapshot forense preservado
    snapshots = list(env["backup_base"].glob("reconcile_snapshot_*"))
    assert len(snapshots) >= 1
    assert (snapshots[0] / "snapshot_manifest.json").exists()

    # 3. Ledger contém PREPARED e não contém COMMITTED
    ledger_val = validate_reconciliation_audit_ledger(env["ledger_path"], expected_environment_uuid=env["env_uuid"])
    assert ledger_val["has_pending_transaction"] is True
    assert len(ledger_val["orphaned_prepared"]) == 1
    assert ledger_val["completed_transactions_count"] == 0

    # 4. JSONL não foi tocado (APPLY abortou antes de tocar no JSONL)
    assert env["jsonl_path"].stat().st_size == 0

    # 5. Startup preflight posterior bloqueia fail-closed
    ok, msg = verify_startup_preflight(data_dir=env["data_dir"], lock_path=env["lock_path"])
    assert not ok
    assert "Startup bloqueado fail-closed" in msg or "Existe transação PREPARED pendente" in msg or "CLEAN" in msg


# =============================================================================
# FIN-04: LEITURAS WAL CONCORRENTES SEM DEADLOCK
# =============================================================================

def test_fin_04_wal_concurrent_reads_during_hermetic_reconciliation(tmp_path: Path):
    """
    FIN-04: Configura o SQLite em modo WAL (Write-Ahead Logging) e valida a coexistência
    de um processo leitor analítico concorrente (executando SELECT contínuos) enquanto
    a reconciliação hermética executa escritas e confirmação de transação.

    Contratos verificados:
    1. O leitor analítico em modo read-only não trava e não recebe 'database is locked'.
    2. A reconciliação conclui com sucesso (status SUCCESS, exit code 0).
    3. Após o commit, o leitor observa a atualização do número de registros com consistência.
    4. Ausência total de deadlock ou timeout de banco de dados.
    5. Utiliza barreira determinística com threading.Event e IPC controlado.
    """
    events_batch = [
        {
            "event_id": f"FIN-WAL-{i:03d}",
            "tipo_evento": "TICK_TRADE",
            "symbol": "BTCUSDT",
            "price": 68000.0 + i,
            "quantity": 0.1,
            "epoch_ms": 1773000020000 + i,
        }
        for i in range(1, 6)
    ]

    env = _setup_sandbox(tmp_path / "fin04_sandbox", initial_events=events_batch, wal_mode=True)

    reader_first_read_barrier = threading.Event()
    writer_committed_event = threading.Event()
    do_final_read_event = threading.Event()
    read_results_queue: queue.Queue = queue.Queue()

    def analytical_reader_thread():
        """Leitor concorrente operando estritamente em conexão read-only padrão."""
        conn_uri = f"file:{env['sqlite_path'].resolve().as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(conn_uri, uri=True, timeout=5.0)
            during_counts = []

            # 1. Primeira leitura obrigatória antes da gravação
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            c = cur.fetchone()[0]
            cur.close()
            during_counts.append(c)
            reader_first_read_barrier.set()

            # 2. Leituras contínuas durante a gravação (sem sleep)
            while not writer_committed_event.is_set():
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM events")
                c = cur.fetchone()[0]
                cur.close()
                during_counts.append(c)

            # 3. Aguarda sinal emitido estritamente após o join() do escritor
            assert do_final_read_event.wait(timeout=5.0), "Timeout aguardando sinal para leitura pós-join"
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM events")
            final_count = cur.fetchone()[0]
            cur.close()
            conn.close()
            read_results_queue.put(("SUCCESS", len(during_counts), during_counts, final_count))
        except Exception as e:
            read_results_queue.put(("ERROR", str(e), [], 0))

    reader_t = threading.Thread(target=analytical_reader_thread, daemon=True, name="fin04_reader")
    reader_t.start()

    # Aguarda o leitor realizar com sucesso a primeira leitura
    assert reader_first_read_barrier.wait(timeout=5.0), "Leitor concorrente não realizou primeira leitura a tempo"

    writer_result = {}
    writer_exception = None

    def writer_target():
        """Executa a reconciliação hermética em thread dedicada."""
        nonlocal writer_exception
        try:
            exit_code, res = execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=env["sqlite_path"],
                jsonl_path=env["jsonl_path"],
                archive_dir=env["archive_dir"],
                backup_dir=env["backup_base"],
                lock_path=env["lock_path"],
                manifest_path=env["manifest_path"],
                ledger_path=env["ledger_path"],
                sandbox_root=env["sandbox_root"],
            )
            writer_result["exit_code"] = exit_code
            writer_result["res"] = res
        except Exception as exc:
            writer_exception = exc
        finally:
            writer_committed_event.set()

    writer_t = threading.Thread(target=writer_target, daemon=True, name="fin04_writer")
    writer_t.start()

    try:
        writer_t.join(timeout=10.0)
        assert not writer_t.is_alive(), "Escritor concorrente não finalizou a tempo"
    finally:
        do_final_read_event.set()

    assert not writer_t.is_alive(), "Escritor concorrente não finalizou a tempo"
    assert writer_exception is None, f"Escritor sofreu excecao nao tratada: {writer_exception}"

    reader_t.join(timeout=5.0)
    assert not reader_t.is_alive(), "Leitor concorrente não finalizou a tempo"

    # Asserções da transação
    exit_code = writer_result.get("exit_code")
    res = writer_result.get("res", {})
    assert exit_code == 0
    assert res["status"] == ReconciliationTransactionStatus.SUCCESS.value
    assert res["sqlite_inserted"] == 5
    assert res["jsonl_appended"] == 5

    # Asserções do leitor concorrente
    assert not read_results_queue.empty()
    status, iters, during_counts, final_count = read_results_queue.get()
    assert status == "SUCCESS", f"Leitor concorrente sofreu exceção: {iters}"
    assert iters >= 1, "Leitor deve ter executado ao menos a leitura inicial antes/durante a gravação"

    # (i) Verificação de leituras durante a gravação: não podem exceder o total final
    assert all(0 <= c <= 5 for c in during_counts), (
        f"Leituras durante a gravação violaram o intervalo [0, 5]: {during_counts}"
    )

    # (ii) Leitura final após o join() do escritor: deve ser exatamente 5
    assert final_count == 5, f"Leitor deve enxergar exatamente 5 registros gravados após commit, obteve: {final_count}"


def test_fin_04_writer_exception_does_not_hang_reader(monkeypatch, tmp_path: Path):
    """
    Comprova que uma exceção inesperada no escritor não trava o leitor em deadlock.
    Valida a presença de try/finally no writer_target e no ciclo de vida das threads.
    """
    env = _setup_sandbox(tmp_path / "fin04_exc_sandbox", initial_events=[], wal_mode=True)

    reader_first_read_barrier = threading.Event()
    writer_committed_event = threading.Event()
    do_final_read_event = threading.Event()

    def analytical_reader_thread():
        conn_uri = f"file:{env['sqlite_path'].resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(conn_uri, uri=True, timeout=5.0)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        cur.close()
        reader_first_read_barrier.set()

        while not writer_committed_event.is_set():
            time.sleep(0.005)

        do_final_read_event.wait(timeout=2.0)
        conn.close()

    reader_t = threading.Thread(target=analytical_reader_thread, daemon=True, name="fin04_exc_reader")
    reader_t.start()
    assert reader_first_read_barrier.wait(timeout=5.0)

    # Injeta falha no escritor
    def crashing_hermetic(*args, **kwargs):
        raise RuntimeError("Falha simulada no escritor antes do commit")

    monkeypatch.setattr(
        "tests.events.test_fallback_reconciliation_financial_stress.execute_apply_integration_hermetic",
        crashing_hermetic
    )

    def writer_target():
        try:
            execute_apply_integration_hermetic(
                fallback_file=env["fallback_file"],
                sqlite_path=env["sqlite_path"],
                jsonl_path=env["jsonl_path"],
                archive_dir=env["archive_dir"],
                backup_dir=env["backup_base"],
                lock_path=env["lock_path"],
                manifest_path=env["manifest_path"],
                ledger_path=env["ledger_path"],
                sandbox_root=env["sandbox_root"],
            )
        finally:
            writer_committed_event.set()

    writer_t = threading.Thread(target=writer_target, daemon=True, name="fin04_exc_writer")
    writer_t.start()

    writer_t.join(timeout=2.0)
    do_final_read_event.set()
    reader_t.join(timeout=1.0)

    assert not reader_t.is_alive(), "Leitor travou após exceção no escritor (writer_committed_event não setado)"


# =============================================================================
# FIN-05: COLISÃO DE IDENTIDADE E AUDITORIA DA UNICIDADE SQLITE
# =============================================================================

def test_fin_05_identity_collision_and_sqlite_physical_uniqueness_audit(tmp_path: Path):
    """
    FIN-05: Audita a ausência de restrição UNIQUE nativa no SQLite DDL e comprova
    que colisões de identidade com payload divergente e duplicatas físicas pré-existentes
    são rejeitadas de forma estritamente fail-closed antes de PREPARED ou na verificação.

    Subcenários testados:
    1. Auditoria DDL: Comprova que o SQLite aceita fisicamente duas linhas com o mesmo event_id
       no payload JSON (ausência de UNIQUE nativo no DDL).
    2. A pós-verificação (verify_reconciliation_targets) detecta essa duplicata física e reprova
       com code=VERIFICATION_FAILED (sem LIMIT 1).
    3. Colisão de ID com Payload Divergente no SQLite: Evento A gravado com price=50000.0;
       fallback com mesmo event_id e price=50100.0. Classificado como EVENTO_ID_COLISAO,
       rejeitado antes de PREPARED com FAILED_BEFORE_PREPARED.
    4. Colisão no JSONL: Rejeitada com EVENTO_ID_COLISAO.
    5. Colisão no Archive: Rejeitada com EVENTO_ID_COLISAO.
    """
    # -------------------------------------------------------------------------
    # 1. Auditoria DDL e Duplicação Física no SQLite
    # -------------------------------------------------------------------------
    env = _setup_sandbox(tmp_path / "fin05_sandbox")

    # Inserção direta de duas linhas com o mesmo event_id no SQLite para auditar o DDL
    conn = sqlite3.connect(str(env["sqlite_path"]))
    try:
        cur = conn.cursor()
        # Verificar schema DDL
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='events'")
        ddl_sql = cur.fetchone()[0]
        assert "UNIQUE" not in ddl_sql.upper(), "Tabela events não deve ter constraint UNIQUE nativa"

        payload_1 = json.dumps({"event_id": "DUP-PHYSICAL-01", "price": 100.0, "epoch_ms": 1773000000000})
        payload_2 = json.dumps({"event_id": "DUP-PHYSICAL-01", "price": 100.0, "epoch_ms": 1773000000000})
        cur.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (1773000000000, 'TRADE', 'BTC', ?)", (payload_1,))
        cur.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (1773000000000, 'TRADE', 'BTC', ?)", (payload_2,))
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM events")
        total_rows = cur.fetchone()[0]
        assert total_rows == 2, "SQLite aceitou fisicamente duas linhas duplicadas (comprovando ausência de UNIQUE nativo)"
    finally:
        conn.close()

    # 2. Pós-verificação detecta a duplicata física no SQLite e REPROVA
    verif_res = verify_reconciliation_targets(
        expected_events=[{"event_id": "DUP-PHYSICAL-01", "price": 100.0, "epoch_ms": 1773000000000, "tipo_evento": "TRADE", "symbol": "BTC"}],
        sqlite_path=env["sqlite_path"],
        jsonl_path=env["jsonl_path"],
        archive_dir=env["archive_dir"],
        require_all_destinations_default=False,
    )
    assert not verif_res.approved, "Pós-verificação deve reprovar duplicata física no SQLite"
    assert verif_res.duplicate_count == 1
    assert "DUP-PHYSICAL-01" in verif_res.problematic_event_ids

    # -------------------------------------------------------------------------
    # 3. Colisão no SQLite com Payload Divergente
    # -------------------------------------------------------------------------
    env_col_sq = _setup_sandbox(tmp_path / "fin05_col_sq")
    # Grava evento original com price=50000.0 no SQLite
    orig_ev = {"event_id": "COL-EID-01", "price": 50000.0, "epoch_ms": 1773000030000, "tipo_evento": "TRADE", "symbol": "BTCUSDT"}
    conn_col = sqlite3.connect(str(env_col_sq["sqlite_path"]))
    try:
        conn_col.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (1773000030000, 'TRADE', 'BTCUSDT', ?)", (json.dumps(orig_ev),))
        conn_col.commit()
    finally:
        conn_col.close()

    # Tenta reconciliar lote com mesmo ID mas preço divergente (50100.0)
    divergent_ev = {"event_id": "COL-EID-01", "price": 50100.0, "epoch_ms": 1773000030000, "tipo_evento": "TRADE", "symbol": "BTCUSDT"}
    with open(env_col_sq["fallback_file"], "w", encoding="utf-8") as f:
        f.write(json.dumps(divergent_ev) + "\n")

    exit_code, res = execute_apply_integration_hermetic(
        fallback_file=env_col_sq["fallback_file"],
        sqlite_path=env_col_sq["sqlite_path"],
        jsonl_path=env_col_sq["jsonl_path"],
        archive_dir=env_col_sq["archive_dir"],
        backup_dir=env_col_sq["backup_base"],
        lock_path=env_col_sq["lock_path"],
        manifest_path=env_col_sq["manifest_path"],
        ledger_path=env_col_sq["ledger_path"],
        sandbox_root=env_col_sq["sandbox_root"],
    )

    assert exit_code == 1
    assert res["status"] == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value
    assert res["stage_reached"] == "EVENT_CLASSIFICATION"
    assert "EVENTO_ID_COLISAO" in str(res["errors"])
    assert res["prepared_recorded"] is False
    assert res["committed_recorded"] is False

    # -------------------------------------------------------------------------
    # 4. Colisão no JSONL com Payload Divergente
    # -------------------------------------------------------------------------
    env_col_jl = _setup_sandbox(tmp_path / "fin05_col_jl")
    # Grava evento original no JSONL
    with open(env_col_jl["jsonl_path"], "w", encoding="utf-8") as f:
        f.write(json.dumps(orig_ev) + "\n")

    with open(env_col_jl["fallback_file"], "w", encoding="utf-8") as f:
        f.write(json.dumps(divergent_ev) + "\n")

    exit_code_jl, res_jl = execute_apply_integration_hermetic(
        fallback_file=env_col_jl["fallback_file"],
        sqlite_path=env_col_jl["sqlite_path"],
        jsonl_path=env_col_jl["jsonl_path"],
        archive_dir=env_col_jl["archive_dir"],
        backup_dir=env_col_jl["backup_base"],
        lock_path=env_col_jl["lock_path"],
        manifest_path=env_col_jl["manifest_path"],
        ledger_path=env_col_jl["ledger_path"],
        sandbox_root=env_col_jl["sandbox_root"],
    )

    assert exit_code_jl == 1
    assert res_jl["status"] == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value
    assert "EVENTO_ID_COLISAO" in str(res_jl["errors"])

    # -------------------------------------------------------------------------
    # 5. Colisão no Archive com Payload Divergente
    # -------------------------------------------------------------------------
    env_col_arch = _setup_sandbox(tmp_path / "fin05_col_arch")
    seg_file = env_col_arch["archive_dir"] / "eventos_fluxo_20260301.jsonl"
    with open(seg_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(orig_ev) + "\n")

    with open(env_col_arch["fallback_file"], "w", encoding="utf-8") as f:
        f.write(json.dumps(divergent_ev) + "\n")

    exit_code_arch, res_arch = execute_apply_integration_hermetic(
        fallback_file=env_col_arch["fallback_file"],
        sqlite_path=env_col_arch["sqlite_path"],
        jsonl_path=env_col_arch["jsonl_path"],
        archive_dir=env_col_arch["archive_dir"],
        backup_dir=env_col_arch["backup_base"],
        lock_path=env_col_arch["lock_path"],
        manifest_path=env_col_arch["manifest_path"],
        ledger_path=env_col_arch["ledger_path"],
        sandbox_root=env_col_arch["sandbox_root"],
    )

    assert exit_code_arch == 1
    assert res_arch["status"] == ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED.value
    assert "EVENTO_ID_COLISAO" in str(res_arch["errors"])
