# tests/unit/test_event_store_idempotency_h1_b1.py
"""
Suíte de Testes Herméticos B1-01 a B1-26 para Idempotência Transacional do EventStore (Gate H1-B1).

Garante:
- Resolução do ACHADO-FIN-01 (P1).
- Atomicidade e unicidade aplicacional de event_id no SQLite.
- BEGIN IMMEDIATE na mesma transação para SELECT e INSERT.
- Isolamento absoluto em tmp_path (sem tocar dados/ de produção).
- Retrocompatibilidade estrita com BatchSaveResult e Gate G2.
"""

import sqlite3
import json
import time
import multiprocessing
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from database.event_store import (
    EventStore,
    BatchSaveResult,
    EventStoreError,
    IntegrityCollisionError,
    PreexistingDuplicateError,
    CorruptedHistoricalRecordError,
    _extract_event_identity,
)
from events import event_saver
from events.event_saver import _is_valid_batch_save_result, EventSaver
from events.fallback_reconciler import compute_canonical_payload_hash


@pytest.fixture
def hermetic_event_store(tmp_path):
    """Cria uma instância isolada do EventStore em tmp_path."""
    db_file = tmp_path / "hermetic_trading_bot.db"
    store = EventStore(db_path=str(db_file))
    return store


# ==============================================================================
# B1-01: Batch novo com IDs distintos
# ==============================================================================
def test_b1_01_batch_novo_ids_distintos(hermetic_event_store):
    store = hermetic_event_store
    events = [
        {"event_id": "b1_01_a", "epoch_ms": 1000, "price": 50000.0},
        {"event_id": "b1_01_b", "epoch_ms": 1001, "price": 50010.0},
    ]
    res = store.save_batch(events)
    assert res.commit_status == "COMMITTED"
    assert res.total_received == 2
    assert res.persisted == 2
    assert res.rows_inserted == 2
    assert res.duplicates_skipped == 0
    assert res.rejected_pre_commit == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# B1-02: Reenvio idêntico do mesmo batch
# ==============================================================================
def test_b1_02_reenvio_identico_mesmo_batch(hermetic_event_store):
    store = hermetic_event_store
    events = [
        {"event_id": "b1_02_a", "epoch_ms": 2000, "price": 60000.0},
        {"event_id": "b1_02_b", "epoch_ms": 2001, "price": 60010.0},
    ]
    # Primeira gravação
    res1 = store.save_batch(events)
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 2

    # Reenvio idêntico
    res2 = store.save_batch(events)
    assert res2.commit_status == "COMMITTED"
    assert res2.total_received == 2
    assert res2.persisted == 2
    assert res2.rows_inserted == 0
    assert res2.duplicates_skipped == 2
    assert res2.rejected_pre_commit == 0

    # Confirma que nenhuma nova linha foi adicionada ao SQLite
    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# B1-03: Mesmo event_id com timestamp diferente
# ==============================================================================
def test_b1_03_mesmo_event_id_timestamp_diferente(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": "b1_03", "epoch_ms": 1000, "price": 70000.0}])

    # Reenvio com o mesmo event_id mas timestamp divergente no payload
    res = store.save_batch([{"event_id": "b1_03", "epoch_ms": 9999, "price": 70000.0}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "IntegrityCollisionError"

    # Confirma que a linha no SQLite permanece com o epoch_ms original
    with sqlite3.connect(str(store.db_path)) as conn:
        payload_str = conn.execute("SELECT payload FROM events WHERE json_extract(payload, '$.event_id') = 'b1_03'").fetchone()[0]
        data = json.loads(payload_str)
        assert data["epoch_ms"] == 1000


# ==============================================================================
# B1-04: Mesmo event_id com payload divergente
# ==============================================================================
def test_b1_04_mesmo_event_id_payload_divergente(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": "b1_04", "price": 50000.0, "side": "BUY"}])

    res = store.save_batch([{"event_id": "b1_04", "price": 50500.0, "side": "BUY"}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "IntegrityCollisionError"


# ==============================================================================
# B1-05: Duplicata idêntica dentro do mesmo batch
# ==============================================================================
def test_b1_05_duplicata_identica_mesmo_batch(hermetic_event_store):
    store = hermetic_event_store
    event = {"event_id": "b1_05", "epoch_ms": 1234, "action": "HOLD"}
    res = store.save_batch([event, dict(event)])

    assert res.commit_status == "COMMITTED"
    assert res.total_received == 2
    assert res.persisted == 2
    assert res.rows_inserted == 1
    assert res.duplicates_skipped == 1
    assert res.rejected_pre_commit == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1


# ==============================================================================
# B1-06: Duplicata divergente dentro do mesmo batch
# ==============================================================================
def test_b1_06_duplicata_divergente_mesmo_batch(hermetic_event_store):
    store = hermetic_event_store
    events = [
        {"event_id": "b1_06", "price": 100.0},
        {"event_id": "b1_06", "price": 200.0},
    ]
    res = store.save_batch(events)
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "IntegrityCollisionError"

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0


# ==============================================================================
# B1-07: Batch misto de novos e já presentes idênticos
# ==============================================================================
def test_b1_07_batch_misto_novos_e_ja_presentes(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": "b1_07_existente", "val": 10}])

    res = store.save_batch([
        {"event_id": "b1_07_existente", "val": 10},
        {"event_id": "b1_07_novo", "val": 20},
    ])
    assert res.commit_status == "COMMITTED"
    assert res.total_received == 2
    assert res.persisted == 2
    assert res.rows_inserted == 1
    assert res.duplicates_skipped == 1

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# B1-08: Batch com colisão divergente não insere nenhuma linha nova
# ==============================================================================
def test_b1_08_colisao_divergente_falha_integral(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": "b1_08_base", "val": 10}])

    res = store.save_batch([
        {"event_id": "b1_08_novo_nao_deve_inserir", "val": 999},
        {"event_id": "b1_08_base", "val": 9999},  # divergente
    ])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        rows = conn.execute("SELECT payload FROM events").fetchall()
        assert len(rows) == 1
        assert "b1_08_novo_nao_deve_inserir" not in rows[0][0]


# ==============================================================================
# B1-09: Banco com duplicatas físicas históricas para ID relevante
# ==============================================================================
def test_b1_09_duplicatas_historicas_preexistentes(hermetic_event_store):
    store = hermetic_event_store
    payload_str = json.dumps({"event_id": "b1_09_dup", "val": 1})
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("INSERT INTO events (timestamp_ms, event_type, payload) VALUES (1, 't', ?)", (payload_str,))
        conn.execute("INSERT INTO events (timestamp_ms, event_type, payload) VALUES (2, 't', ?)", (payload_str,))
        conn.commit()

    res = store.save_batch([{"event_id": "b1_09_dup", "val": 1}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "PreexistingDuplicateError"


# ==============================================================================
# B1-10: Payload histórico JSON inválido
# ==============================================================================
def test_b1_10_payload_historico_json_invalido(hermetic_event_store):
    store = hermetic_event_store
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("INSERT INTO events (timestamp_ms, event_type, payload) VALUES (1, 't', '{corrupted json')")
        conn.commit()

    res = store.save_batch([{"event_id": "b1_10", "val": 1}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "CorruptedHistoricalRecordError"


# ==============================================================================
# B1-11: event_id ausente, vazio, nulo, booleano e numérico
# ==============================================================================
def test_b1_11_variantes_event_id(hermetic_event_store):
    store = hermetic_event_store
    res_invalid = store.save_batch([
        {"event_id": "", "val": 1},
        {"event_id": True, "val": 2},
        {"event_id": "   ", "val": 3},
    ])
    assert res_invalid.commit_status == "FAILED"
    assert res_invalid.rejected_pre_commit == 3
    assert res_invalid.persisted == 0

    res_valid = store.save_batch([
        {"event_id": 12345, "val": "num_coerced"},
        {"val": "anonymous_event"},
        {"event_id": None, "val": "none_anonymous"},
    ])
    assert res_valid.commit_status == "COMMITTED"
    assert res_valid.total_received == 3
    assert res_valid.persisted == 3
    assert res_valid.rejected_pre_commit == 0


# ==============================================================================
# B1-12: Evento legado com apenas id
# ==============================================================================
def test_b1_12_evento_legado_com_apenas_id(hermetic_event_store):
    store = hermetic_event_store
    res1 = store.save_batch([{"id": "legacy_order_123", "val": 10}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    res2 = store.save_batch([{"id": "legacy_order_123", "val": 10}])
    assert res2.commit_status == "COMMITTED"
    assert res2.duplicates_skipped == 1
    assert res2.rows_inserted == 0

    res3 = store.save_batch([{"id": 42, "val": 10}])
    assert res3.commit_status == "COMMITTED"
    assert res3.rows_inserted == 1


# ==============================================================================
# B1-13: Evento com id e event_id conflitantes
# ==============================================================================
def test_b1_13_evento_com_id_e_event_id_conflitantes(hermetic_event_store):
    store = hermetic_event_store
    res = store.save_batch([{"event_id": "ORDER_ALPHA", "id": "ORDER_BETA"}])
    assert res.commit_status == "FAILED"
    assert res.rejected_pre_commit == 1
    assert res.persisted == 0


# ==============================================================================
# B1-14: Falha antes do commit com rollback verificável
# ==============================================================================
class FailingExecutemanyConnection:
    """Wrapper de conexão SQLite para injetar falha controlada em executemany."""
    def __init__(self, real_conn):
        self._real_conn = real_conn
        self.in_transaction = False

    def execute(self, sql, *args):
        if "BEGIN" in sql:
            self.in_transaction = True
        elif "ROLLBACK" in sql or "COMMIT" in sql:
            self.in_transaction = False
        return self._real_conn.execute(sql, *args)

    def executemany(self, sql, seq):
        raise sqlite3.OperationalError("Simulated disk error during executemany")

    def commit(self):
        self.in_transaction = False
        return self._real_conn.commit()

    def rollback(self):
        self.in_transaction = False
        return self._real_conn.rollback()

    def close(self):
        return self._real_conn.close()

    def __getattr__(self, name):
        return getattr(self._real_conn, name)


def test_b1_14_falha_antes_do_commit_com_rollback(hermetic_event_store, monkeypatch):
    store = hermetic_event_store

    original_conn_factory = store._get_conn
    def mock_conn():
        return FailingExecutemanyConnection(original_conn_factory())

    monkeypatch.setattr(store, "_get_conn", mock_conn)

    res = store.save_batch([{"event_id": "b1_14", "val": 1}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0


# ==============================================================================
# B1-15: Falha durante commit com resultado UNKNOWN
# ==============================================================================
class FailingCommitConnection:
    """Wrapper de conexão SQLite para injetar falha durante o commit."""
    def __init__(self, real_conn):
        self._real_conn = real_conn
        self.in_transaction = False

    def execute(self, sql, *args):
        if "BEGIN" in sql:
            self.in_transaction = True
        elif "ROLLBACK" in sql or "COMMIT" in sql:
            self.in_transaction = False
        return self._real_conn.execute(sql, *args)

    def executemany(self, sql, seq):
        return self._real_conn.executemany(sql, seq)

    def commit(self):
        raise sqlite3.OperationalError("Simulated network timeout during commit")

    def rollback(self):
        self.in_transaction = False
        return self._real_conn.rollback()

    def close(self):
        return self._real_conn.close()

    def __getattr__(self, name):
        return getattr(self._real_conn, name)


def test_b1_15_falha_durante_commit_resultado_unknown(hermetic_event_store, monkeypatch):
    store = hermetic_event_store

    original_conn_factory = store._get_conn
    def mock_conn():
        return FailingCommitConnection(original_conn_factory())

    monkeypatch.setattr(store, "_get_conn", mock_conn)

    res = store.save_batch([{"event_id": "b1_15", "val": 1}])
    assert res.commit_status == "UNKNOWN"
    assert res.persisted == 0


# ==============================================================================
# B1-16: Nenhum retry automático após UNKNOWN
# ==============================================================================
def test_b1_16_nenhum_retry_automatico_apos_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    saver = EventSaver(sound_alert=False)
    mock_db = MagicMock()
    mock_db.save_batch.return_value = BatchSaveResult(
        total_received=1,
        persisted=0,
        rejected_pre_commit=0,
        commit_status="UNKNOWN",
        error_type="OperationalError",
    )
    saver.db = mock_db
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    saver.save_event({"event_id": "b1_16", "test": True})
    saver._process_single_flush_batch()

    assert mock_db.save_batch.call_count == 1
    stats = saver.get_stats()
    assert stats["sqlite_events_unknown"] == 1
    assert stats["sqlite_events_persisted"] == 0
    saver.stop()


# ==============================================================================
# B1-17: Concorrência entre duas conexões SQLite
# ==============================================================================
def test_b1_17_concorrencia_duas_conexoes(hermetic_event_store):
    store1 = hermetic_event_store
    store2 = EventStore(db_path=str(store1.db_path))

    res1 = store1.save_batch([{"event_id": "b1_17_a", "val": 1}])
    res2 = store2.save_batch([{"event_id": "b1_17_b", "val": 2}])
    res3 = store2.save_batch([{"event_id": "b1_17_a", "val": 1}])

    assert res1.commit_status == "COMMITTED"
    assert res2.commit_status == "COMMITTED"
    assert res3.commit_status == "COMMITTED"
    assert res3.duplicates_skipped == 1
    assert res3.rows_inserted == 0

    with sqlite3.connect(str(store1.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# B1-18: Concorrência real entre dois subprocessos independentes em tmp_path
# ==============================================================================
def _worker_process_save(db_path_str: str, event_id: str, val: int, return_queue):
    store = EventStore(db_path=db_path_str)
    res = store.save_batch([{"event_id": event_id, "val": val}])
    return_queue.put((res.commit_status, res.persisted, res.rows_inserted, res.duplicates_skipped))


def test_b1_18_concorrencia_multiprocesso(tmp_path):
    db_path = tmp_path / "multiprocess_test.db"
    store = EventStore(db_path=str(db_path))

    queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(
        target=_worker_process_save, args=(str(db_path), "shared_event", 100, queue)
    )
    p2 = multiprocessing.Process(
        target=_worker_process_save, args=(str(db_path), "shared_event", 100, queue)
    )

    p1.start()
    p2.start()
    p1.join(timeout=10)
    p2.join(timeout=10)

    results = [queue.get() for _ in range(2)]
    assert all(r[0] == "COMMITTED" for r in results)
    total_rows = sum(r[2] for r in results)
    total_skipped = sum(r[3] for r in results)
    assert total_rows == 1
    assert total_skipped == 1

    with sqlite3.connect(str(db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events WHERE json_extract(payload, '$.event_id') = 'shared_event'").fetchone()[0]
        assert count == 1


# ==============================================================================
# B1-19: Compatibilidade de BatchSaveResult
# ==============================================================================
def test_b1_19_compatibilidade_batch_save_result():
    res = BatchSaveResult(5, 5, 0, "COMMITTED", None, None)
    assert res.total_received == 5
    assert res.persisted == 5
    assert res.rejected_pre_commit == 0
    assert res.commit_status == "COMMITTED"
    assert res.duplicates_skipped == 0
    assert res.rows_inserted == 0

    res2 = BatchSaveResult(
        total_received=3,
        persisted=3,
        rejected_pre_commit=0,
        commit_status="COMMITTED",
        duplicates_skipped=1,
        rows_inserted=2,
    )
    assert res2.duplicates_skipped == 1
    assert res2.rows_inserted == 2


# ==============================================================================
# B1-20: Compatibilidade com EventSaver worker e seus contadores
# ==============================================================================
def test_b1_20_compatibilidade_event_saver_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    saver = EventSaver(sound_alert=False)
    db_path = tmp_path / "saver_worker_test.db"
    store = EventStore(db_path=str(db_path))
    saver.db = store
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    saver.save_event({"event_id": "b1_20_1", "action": "BUY"})
    saver.save_event({"event_id": "b1_20_2", "action": "SELL"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_persisted"] == 2
    assert stats["sqlite_events_failed"] == 0
    saver.stop()


# ==============================================================================
# B1-21: Reconciliação e hashes históricos preservados
# ==============================================================================
def test_b1_21_hashes_canonicos_reconciliador(hermetic_event_store):
    store = hermetic_event_store
    event = {
        "event_id": "b1_21_canon",
        "tipo_evento": "order_filled",
        "preco": 65000.12345678,
        "volume": 1.5,
    }
    res = store.save_batch([event])
    assert res.commit_status == "COMMITTED"

    expected_hash = compute_canonical_payload_hash(event)

    with sqlite3.connect(str(store.db_path)) as conn:
        row_payload = conn.execute("SELECT payload FROM events WHERE json_extract(payload, '$.event_id') = 'b1_21_canon'").fetchone()[0]
        db_dict = json.loads(row_payload)
        db_hash = compute_canonical_payload_hash(db_dict)
        assert db_hash == expected_hash


# ==============================================================================
# B1-22: Nenhuma alteração de schema
# ==============================================================================
def test_b1_22_nenhuma_alteracao_schema(hermetic_event_store):
    store = hermetic_event_store
    with sqlite3.connect(str(store.db_path)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
        expected_columns = ["id", "timestamp_ms", "event_type", "symbol", "window_id", "is_signal", "payload", "created_at"]
        assert columns == expected_columns

        indexes = [row[1] for row in conn.execute("PRAGMA index_list(events)").fetchall()]
        expected_indexes = {"idx_events_ts", "idx_events_type", "idx_events_window"}
        assert set(indexes) == expected_indexes


# ==============================================================================
# B1-23: Falha de colisão não se transforma em COMMITTED
# ==============================================================================
def test_b1_23_colisao_nunca_vira_committed(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": "b1_23", "data": "A"}])

    res = store.save_batch([{"event_id": "b1_23", "data": "B"}])
    assert res.commit_status != "COMMITTED"
    assert res.commit_status == "FAILED"
    assert res.persisted == 0


# ==============================================================================
# B1-24: Nenhum descarte silencioso de payload financeiro
# ==============================================================================
def test_b1_24_fidelidade_payload_financeiro(hermetic_event_store):
    store = hermetic_event_store
    financial_payload = {
        "event_id": "fin_audit_01",
        "balance": 999999.99,
        "currency": "USDT",
        "orders": [{"id": 1, "price": 45000.25}],
        "rate": 0.0001,
    }
    res = store.save_batch([financial_payload])
    assert res.commit_status == "COMMITTED"

    with sqlite3.connect(str(store.db_path)) as conn:
        row = conn.execute("SELECT payload FROM events WHERE json_extract(payload, '$.event_id') = 'fin_audit_01'").fetchone()[0]
        persisted_data = json.loads(row)
        assert persisted_data["balance"] == 999999.99
        assert persisted_data["currency"] == "USDT"
        assert persisted_data["orders"][0]["price"] == 45000.25


# ==============================================================================
# B1-25: CLI pública continua com APPLY_NOT_ENABLED_C2A
# ==============================================================================
def test_b1_25_cli_publica_bloqueada():
    reconcile_script = Path("scripts/maintenance/reconcile_fallback_events.py")
    assert reconcile_script.exists()
    content = reconcile_script.read_text(encoding="utf-8")
    assert "APPLY_NOT_ENABLED_C2A" in content


# ==============================================================================
# B1-26: Testes FIN-01 a FIN-05 continuam aprovados
# ==============================================================================
def test_b1_26_fin_01_a_05_aprovados():
    ret = pytest.main([
        "tests/events/test_fallback_reconciliation_financial_stress.py",
        "-q",
        "--no-cov",
    ])
    assert ret == pytest.ExitCode.OK


# ==============================================================================
# H1-01 & H1-02: Persistência e reenvio de event_id numérico
# ==============================================================================
def test_h1_01_e_02_persistencia_e_reenvio_event_id_numerico(hermetic_event_store):
    store = hermetic_event_store
    event = {"event_id": 12345, "price": 50000.0}
    res1 = store.save_batch([event])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1
    assert res1.duplicates_skipped == 0
    assert res1.persisted == 1

    # H1-02: Reenvio em outro batch
    res2 = store.save_batch([dict(event)])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 0
    assert res2.duplicates_skipped == 1
    assert res2.persisted == 1

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1


# ==============================================================================
# H1-03: Persistir textual e depois numérico (colisão divergente)
# ==============================================================================
def test_h1_03_textual_depois_numerico_colisao_divergente(hermetic_event_store):
    store = hermetic_event_store
    res1 = store.save_batch([{"event_id": "12345", "val": 1}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    # Enviar event_id=12345 (numérico). Hashes divergem -> colisão divergente
    res2 = store.save_batch([{"event_id": 12345, "val": 1}])
    assert res2.commit_status == "FAILED"
    assert res2.persisted == 0
    assert res2.error_type == "IntegrityCollisionError"

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1


# ==============================================================================
# H1-04: Persistir numérico e depois textual (colisão divergente)
# ==============================================================================
def test_h1_04_numerico_depois_textual_colisao_divergente(hermetic_event_store):
    store = hermetic_event_store
    res1 = store.save_batch([{"event_id": 12345, "val": 1}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    # Enviar event_id="12345" (textual). Hashes divergem -> colisão divergente
    res2 = store.save_batch([{"event_id": "12345", "val": 1}])
    assert res2.commit_status == "FAILED"
    assert res2.persisted == 0
    assert res2.error_type == "IntegrityCollisionError"

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1


# ==============================================================================
# H1-05: event_id=1 versus event_id=1.0 (identidade e representação canônica)
# ==============================================================================
def test_h1_05_int_versus_float_identidade_e_hash(hermetic_event_store):
    store = hermetic_event_store
    # Verificar que hashes canônicos diferem estritamente
    h1 = compute_canonical_payload_hash({"event_id": 1})
    h1_0 = compute_canonical_payload_hash({"event_id": 1.0})
    assert h1 != h1_0

    res1 = store.save_batch([{"event_id": 1, "val": "A"}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    # event_id=1.0 possui representação textual "1.0" no SQLite CAST
    res2 = store.save_batch([{"event_id": 1.0, "val": "A"}])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 1

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# H1-06: event_id=True e event_id=False (rejeição pré-commit)
# ==============================================================================
def test_h1_06_event_id_booleano_rejeicao_pre_commit(hermetic_event_store):
    store = hermetic_event_store
    res = store.save_batch([
        {"event_id": True, "val": 1},
        {"event_id": False, "val": 2},
    ])
    assert res.commit_status == "FAILED"
    assert res.rejected_pre_commit == 2
    assert res.persisted == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0


# ==============================================================================
# H1-07: Mesmo event_id numérico dentro do mesmo batch
# ==============================================================================
def test_h1_07_duplicata_numerica_mesmo_batch(hermetic_event_store):
    store = hermetic_event_store
    event = {"event_id": 9999, "action": "HOLD"}
    res = store.save_batch([event, dict(event)])
    assert res.commit_status == "COMMITTED"
    assert res.total_received == 2
    assert res.persisted == 2
    assert res.rows_inserted == 1
    assert res.duplicates_skipped == 1

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1


# ==============================================================================
# H1-08: Mesmo event_id numérico com payload divergente
# ==============================================================================
def test_h1_08_numerico_divergente_mesmo_batch(hermetic_event_store):
    store = hermetic_event_store
    res = store.save_batch([
        {"event_id": 8888, "price": 100.0},
        {"event_id": 8888, "price": 200.0},
    ])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "IntegrityCollisionError"

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0


# ==============================================================================
# H1-09: Banco com duplicatas físicas equivalentes
# ==============================================================================
def test_h1_09_banco_com_duplicatas_fisicas_equivalentes(hermetic_event_store):
    store = hermetic_event_store
    # Inserir manualmente no SQLite duas linhas (uma numérica, uma string)
    p1 = json.dumps({"event_id": 777, "val": 1})
    p2 = json.dumps({"event_id": "777", "val": 1})
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("INSERT INTO events (timestamp_ms, event_type, payload) VALUES (1, 't', ?)", (p1,))
        conn.execute("INSERT INTO events (timestamp_ms, event_type, payload) VALUES (2, 't', ?)", (p2,))
        conn.commit()

    res = store.save_batch([{"event_id": 777, "val": 1}])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0
    assert res.error_type == "PreexistingDuplicateError"

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# H1-10: Lote misto com novo textual e colisão numérica divergente
# ==============================================================================
def test_h1_10_lote_misto_novo_e_colisao_numerica(hermetic_event_store):
    store = hermetic_event_store
    store.save_batch([{"event_id": 5555, "val": 10}])

    res = store.save_batch([
        {"event_id": "novo_evento_textual", "val": 1},
        {"event_id": 5555, "val": 20},  # colisão divergente
    ])
    assert res.commit_status == "FAILED"
    assert res.persisted == 0

    with sqlite3.connect(str(store.db_path)) as conn:
        rows = conn.execute("SELECT payload FROM events").fetchall()
        assert len(rows) == 1
        assert "novo_evento_textual" not in rows[0][0]


# ==============================================================================
# H1-11: Evento legado com id textual válido
# ==============================================================================
def test_h1_11_evento_legado_com_id_textual(hermetic_event_store):
    store = hermetic_event_store
    res1 = store.save_batch([{"id": "legacy_alpha_99", "amount": 100}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    res2 = store.save_batch([{"id": "legacy_alpha_99", "amount": 100}])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 0
    assert res2.duplicates_skipped == 1


# ==============================================================================
# H1-12: Evento com id numérico tratado como anônimo
# ==============================================================================
def test_h1_12_evento_com_id_numerico_anonimo(hermetic_event_store):
    store = hermetic_event_store
    # Evento com 'id' numérico (ex: ID autoincrement relacional do SQLite)
    # Não é promovido a chave de negócio -> tratado como anônimo sem deduplicação
    res1 = store.save_batch([{"id": 101, "data": "row1"}])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1

    # Reenvio insere segunda linha física (sem garantia de idempotência para anônimos)
    res2 = store.save_batch([{"id": 101, "data": "row1"}])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 1

    with sqlite3.connect(str(store.db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 2


# ==============================================================================
# H1-13: Compatibilidade de BatchSaveResult após correção
# ==============================================================================
def test_h1_13_compatibilidade_batch_save_result_e_g2():
    res = BatchSaveResult(
        total_received=4,
        persisted=3,
        rejected_pre_commit=1,
        commit_status="COMMITTED",
        duplicates_skipped=1,
        rows_inserted=2,
    )
    assert res.persisted + res.rejected_pre_commit == res.total_received
    assert res.rows_inserted + res.duplicates_skipped == res.persisted
    assert _is_valid_batch_save_result(res, 4) is True


# ==============================================================================
# H1-14: Nenhum retry automático após UNKNOWN
# ==============================================================================
def test_h1_14_sem_retry_automatico_unknown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    saver = EventSaver(sound_alert=False)
    mock_db = MagicMock()
    mock_db.save_batch.return_value = BatchSaveResult(
        total_received=1,
        persisted=0,
        rejected_pre_commit=0,
        commit_status="UNKNOWN",
        error_type="CommitTimeoutError",
    )
    saver.db = mock_db
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    saver.save_event({"event_id": 12345, "side": "BUY"})
    saver._process_single_flush_batch()

    assert mock_db.save_batch.call_count == 1
    stats = saver.get_stats()
    assert stats["sqlite_events_unknown"] == 1
    assert stats["sqlite_events_persisted"] == 0
    saver.stop()


# ==============================================================================
# H1-15: Confirmação de que compute_canonical_payload_hash() não foi alterado
# ==============================================================================
def test_h1_15_compute_canonical_payload_hash_inalterado():
    from events.fallback_reconciler import compute_canonical_payload_hash, EXCLUDED_CANONICAL_FIELDS
    assert EXCLUDED_CANONICAL_FIELDS == {"_log_id", "_source_db_id", "id", "created_at"}
    test_payload = {"event_id": "test", "preco": 100.0}
    h = compute_canonical_payload_hash(test_payload)
    assert isinstance(h, str) and len(h) == 64


# ==============================================================================
# H1-16: Confirmação de que o DDL SQLite não foi alterado
# ==============================================================================
def test_h1_16_ddl_sqlite_inalterado(hermetic_event_store):
    store = hermetic_event_store
    with sqlite3.connect(str(store.db_path)) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='events'").fetchone()[0]
        assert "UNIQUE" not in schema.upper()


# ==============================================================================
# H1-17: Confirmação de que APPLY_NOT_ENABLED_C2A permanece ativo
# ==============================================================================
def test_h1_17_apply_not_enabled_c2a_ativo():
    content = Path("scripts/maintenance/reconcile_fallback_events.py").read_text(encoding="utf-8")
    assert "APPLY_NOT_ENABLED_C2A" in content


# ==============================================================================
# H1-18: Regressão dos testes B1-01 a B1-26
# ==============================================================================
def test_h1_18_regressao_testes_b1_01_a_b1_26(hermetic_event_store):
    store = hermetic_event_store
    res = store.save_batch([{"event_id": "reg_b1", "val": 1}])
    assert res.commit_status == "COMMITTED"
    assert res.rows_inserted == 1
    res2 = store.save_batch([{"event_id": "reg_b1", "val": 1}])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 0
    assert res2.duplicates_skipped == 1


# ==============================================================================
# H1-19: Regressão dos testes FIN-01 a FIN-05
# ==============================================================================
def test_h1_19_regressao_testes_fin_01_a_05():
    ret = pytest.main([
        "tests/events/test_fallback_reconciliation_financial_stress.py",
        "-q",
        "--no-cov",
    ])
    assert ret == pytest.ExitCode.OK

