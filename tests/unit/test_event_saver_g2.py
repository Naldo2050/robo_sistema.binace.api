"""
Testes herméticos para o Gate G2 — Contrato de Persistência e Confirmação de Escrita.

Valida:
1. SQLite batch vazio.
2. SQLite commit integral.
3. SQLite rejeição parcial pré-commit.
4. SQLite rejeição total.
5. SQLite falha antes do commit.
6. SQLite erro durante commit.
7. SQLite resultado desconhecido.
8. SQLite retorno inválido.
9. SQLite MagicMock sem configuração.
10. SQLite contabilidade exata.
11. JSONL sucesso.
12. JSONL falha com retries esgotados.
13. JSONL sucesso após retry.
14. JSONL desabilitado.
15. JSONL resultado desconhecido.
16. Fallback sucesso.
17. Fallback falha.
18. SQLite FAIL / JSONL OK.
19. SQLite OK / JSONL FAIL.
20. SQLite FAIL / JSONL FAIL.
21. Rejeição parcial pré-commit.
22. Contadores independentes por destino.
23. Conservação G1 preservada.
24. Warning rate-limited no worker.
25. Exception no logger não mata worker.
26. Nenhum logging no caller.
27. Shutdown durante I/O.
28. Mocks legados retornando None.
"""

import time
import threading
import sqlite3
from unittest.mock import MagicMock, patch
import pytest

from database.event_store import EventStore, BatchSaveResult
from events import event_saver
from events.event_saver import EventSaver


def _patch_mainthread_sleep(monkeypatch):
    """Bypass sleep apenas na MainThread para testes sem travar threads de background."""
    real_sleep = time.sleep
    monkeypatch.setattr(
        time,
        "sleep",
        lambda s: None if threading.current_thread() is threading.main_thread() else real_sleep(s),
    )


def test_sleep_isolation_secondary_threads_preserve_sleep(monkeypatch):
    """
    Regressão: valida que o mock de sleep afeta apenas a MainThread e
    conserva o sleep real em threads secundárias, mesmo com coverage.
    """
    _patch_mainthread_sleep(monkeypatch)

    # 1. MainThread tem sleep ignorado (instantâneo)
    t0 = time.perf_counter()
    time.sleep(1.0)
    main_elapsed = time.perf_counter() - t0
    assert main_elapsed < 0.1, f"MainThread deveria ignorar sleep, demorou {main_elapsed:.4f}s"

    # 2. Thread secundária preserva sleep real
    secondary_elapsed = []
    def worker():
        w_t0 = time.perf_counter()
        time.sleep(0.05)
        secondary_elapsed.append(time.perf_counter() - w_t0)

    th = threading.Thread(target=worker)
    th.start()
    th.join(timeout=2.0)

    assert len(secondary_elapsed) == 1
    assert secondary_elapsed[0] >= 0.04, (
        f"Thread secundária deveria executar sleep real (>= 0.04s), obteve {secondary_elapsed[0]:.4f}s"
    )


@pytest.fixture
def hermetic_saver_g2(tmp_path, monkeypatch):
    """Fixture para inicializar EventSaver hermético com banco SQLite real isolado."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    db_path = tmp_path / "test_events.db"
    store = EventStore(db_path=str(db_path))

    saver = EventSaver(sound_alert=False)
    saver.db = store
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    yield saver

    try:
        saver.stop(timeout=1.0)
    except Exception:
        pass


# 1. SQLite batch vazio
def test_sqlite_batch_vazio(tmp_path):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    res = store.save_batch([])
    assert isinstance(res, BatchSaveResult)
    assert res.total_received == 0
    assert res.persisted == 0
    assert res.rejected_pre_commit == 0
    assert res.commit_status == "COMMITTED"
    assert res.error_type is None


# 2. SQLite commit integral
def test_sqlite_commit_integral(tmp_path):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [
        {"epoch_ms": 1000, "tipo_evento": "TEST_1", "symbol": "BTCUSDT"},
        {"epoch_ms": 2000, "tipo_evento": "TEST_2", "symbol": "BTCUSDT"},
    ]
    res = store.save_batch(events)
    assert res.total_received == 2
    assert res.persisted == 2
    assert res.rejected_pre_commit == 0
    assert res.commit_status == "COMMITTED"
    assert res.error_type is None


# 3. SQLite rejeição parcial pré-commit
def test_sqlite_rejeicao_parcial_pre_commit(tmp_path, monkeypatch):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [
        {"epoch_ms": 1000, "tipo_evento": "VALID_1"},
        {"epoch_ms": "invalid_epoch", "tipo_evento": "BAD_EVENT"},
        {"epoch_ms": 3000, "tipo_evento": "VALID_2"},
    ]
    def bad_sanitize(e):
        if e.get("tipo_evento") == "BAD_EVENT":
            raise ValueError("Corrupted event")
        return e

    monkeypatch.setattr("database.event_store.sanitize_json_safe", bad_sanitize)
    res = store.save_batch(events)
    assert res.total_received == 3
    assert res.persisted == 2
    assert res.rejected_pre_commit == 1
    assert res.commit_status == "COMMITTED"


# 4. SQLite rejeição total
def test_sqlite_rejeicao_total(tmp_path, monkeypatch):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [{"tipo_evento": f"BAD_{i}"} for i in range(3)]

    def fail_sanitize(e):
        raise TypeError("Fatal sanitize error")

    monkeypatch.setattr("database.event_store.sanitize_json_safe", fail_sanitize)
    res = store.save_batch(events)
    assert res.total_received == 3
    assert res.persisted == 0
    assert res.rejected_pre_commit == 3
    assert res.commit_status == "FAILED"
    assert res.error_type == "PreCommitValidationError"


# 5. SQLite falha antes do commit (erro em executemany)
def test_sqlite_falha_antes_do_commit(tmp_path, monkeypatch):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [{"tipo_evento": "EVT_1"}]

    real_conn = store._get_conn()
    mock_conn = MagicMock(wraps=real_conn)
    mock_conn.executemany.side_effect = sqlite3.OperationalError("disk I/O error during insert")
    monkeypatch.setattr(store, "_get_conn", lambda: mock_conn)

    res = store.save_batch(events)
    assert res.total_received == 1
    assert res.persisted == 0
    assert res.commit_status == "FAILED"
    assert res.error_type == "OperationalError"


# 6. SQLite erro durante commit
def test_sqlite_erro_durante_commit(tmp_path, monkeypatch):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [{"tipo_evento": "EVT_1"}]

    real_conn = store._get_conn()
    mock_conn = MagicMock(wraps=real_conn)
    mock_conn.commit.side_effect = sqlite3.OperationalError("database is locked on commit")
    monkeypatch.setattr(store, "_get_conn", lambda: mock_conn)

    res = store.save_batch(events)
    assert res.total_received == 1
    assert res.persisted == 0
    assert res.commit_status == "UNKNOWN"
    assert res.error_type == "OperationalError"


# 6b. SQLite executemany falha e rollback falha -> UNKNOWN (ETAPA 2)
def test_sqlite_falha_executemany_com_rollback_falho(tmp_path, monkeypatch):
    store = EventStore(db_path=str(tmp_path / "db.sqlite"))
    events = [{"tipo_evento": "EVT_1"}]

    real_conn = store._get_conn()
    mock_conn = MagicMock(wraps=real_conn)
    mock_conn.executemany.side_effect = sqlite3.OperationalError("disk I/O error during insert")
    mock_conn.rollback.side_effect = sqlite3.OperationalError("disk I/O error during rollback")
    monkeypatch.setattr(store, "_get_conn", lambda: mock_conn)

    res = store.save_batch(events)
    assert res.total_received == 1
    assert res.persisted == 0
    assert res.commit_status == "UNKNOWN"
    assert res.error_type == "OperationalError"


# 7. SQLite resultado desconhecido
def test_sqlite_resultado_desconhecido(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    mock_store.save_batch.return_value = BatchSaveResult(
        total_received=2,
        persisted=0,
        rejected_pre_commit=0,
        commit_status="UNKNOWN",
        error_type="NetworkTimeout",
    )
    saver.db = mock_store

    saver.save_event({"id": "1"})
    saver.save_event({"id": "2"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 2
    assert stats["sqlite_events_unknown"] == 2
    assert stats["sqlite_events_persisted"] == 0
    assert stats["sqlite_events_failed"] == 0


# 8. SQLite retorno inválido
def test_sqlite_retorno_invalido(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    mock_store.save_batch.return_value = {"not_a": "BatchSaveResult"}
    saver.db = mock_store

    saver.save_event({"id": "1"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 1
    assert stats["sqlite_events_unknown"] == 1
    assert stats["sqlite_events_persisted"] == 0
    assert stats["sqlite_events_failed"] == 0


# 9. SQLite MagicMock sem configuração
def test_sqlite_magic_mock_sem_configuracao(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    saver.db = mock_store

    saver.save_event({"id": "mock_event"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 1
    assert stats["sqlite_events_unknown"] == 1
    assert stats["sqlite_events_persisted"] == 0
    assert stats["sqlite_events_failed"] == 0


# 10. SQLite contabilidade exata
def test_sqlite_contabilidade_exata(hermetic_saver_g2):
    saver = hermetic_saver_g2
    for i in range(5):
        saver.save_event({"id": f"ok_{i}"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    attempted = stats["sqlite_events_attempted"]
    persisted = stats["sqlite_events_persisted"]
    failed = stats["sqlite_events_failed"]
    unknown = stats["sqlite_events_unknown"]
    assert attempted == 5
    assert persisted == 5
    assert failed == 0
    assert unknown == 0
    assert attempted == persisted + failed + unknown


# 11. JSONL sucesso
def test_jsonl_sucesso(hermetic_saver_g2):
    saver = hermetic_saver_g2
    saver.save_event({"id": "jsonl_ok"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_attempted"] == 1
    assert stats["jsonl_events_written"] == 1
    assert stats["jsonl_events_failed"] == 0
    assert stats["jsonl_events_unknown"] == 0
    assert stats["jsonl_events_attempted"] == stats["jsonl_events_written"] + stats["jsonl_events_failed"] + stats["jsonl_events_unknown"]


# 12. JSONL falha com retries esgotados
def test_jsonl_falha_com_retries_esgotados(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    _patch_mainthread_sleep(monkeypatch)

    real_open = open
    def mock_open(path, mode="r", *args, **kwargs):
        if "events.jsonl" in str(path):
            raise OSError("Disk write error")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)
    saver.save_event({"id": "jsonl_fail"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_attempted"] == 1
    assert stats["jsonl_events_written"] == 0
    assert stats["jsonl_events_failed"] == 1
    assert stats["fallback_events_attempted"] == 1
    assert stats["fallback_events_written"] == 1


# 13. JSONL sucesso após retry
def test_jsonl_sucesso_apos_retry(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    _patch_mainthread_sleep(monkeypatch)

    real_open = open
    attempts = [0]
    def mock_open(path, mode="r", *args, **kwargs):
        if "events.jsonl" in str(path):
            attempts[0] += 1
            if attempts[0] == 1:
                raise OSError("Transient I/O failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)
    saver.save_event({"id": "jsonl_retry"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_attempted"] == 1
    assert stats["jsonl_events_written"] == 1
    assert stats["jsonl_events_failed"] == 0
    assert stats["fallback_events_attempted"] == 0


# 14. JSONL desabilitado
def test_jsonl_desabilitado(hermetic_saver_g2):
    saver = hermetic_saver_g2
    saver.write_jsonl = False
    saver.save_event({"id": "jsonl_disabled"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_attempted"] == 0
    assert stats["jsonl_events_written"] == 0


# 15. JSONL resultado desconhecido
def test_jsonl_resultado_desconhecido(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    def raise_runtime(evt):
        with saver._buffer_lock:
            saver._jsonl_events_attempted += 1
            saver._jsonl_events_unknown += 1
        raise RuntimeError("Worker process interrupted")

    monkeypatch.setattr(saver, "_save_to_jsonl", raise_runtime)
    saver.save_event({"id": "jsonl_unknown"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_attempted"] == 1
    assert stats["jsonl_events_unknown"] == 1
    assert stats["jsonl_events_written"] == 0


# 16. Fallback sucesso
def test_fallback_sucesso(hermetic_saver_g2):
    saver = hermetic_saver_g2
    res = saver._save_fallback({"id": "fallback_test"}, "jsonl")
    assert res == "WRITTEN"
    stats = saver.get_stats()
    assert stats["fallback_events_attempted"] == 1
    assert stats["fallback_events_written"] == 1
    assert stats["fallback_events_failed"] == 0


# 17. Fallback falha
def test_fallback_falha(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Access denied")
    monkeypatch.setattr("pathlib.Path.mkdir", mock_mkdir)

    res = saver._save_fallback({"id": "fallback_fail"}, "jsonl")
    assert res == "FAILED"
    stats = saver.get_stats()
    assert stats["fallback_events_attempted"] == 1
    assert stats["fallback_events_written"] == 0
    assert stats["fallback_events_failed"] == 1


# 18. SQLite FAIL / JSONL OK
def test_scenario_sqlite_fail_jsonl_ok(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    mock_store.save_batch.return_value = BatchSaveResult(
        total_received=1,
        persisted=0,
        rejected_pre_commit=0,
        commit_status="FAILED",
        error_type="OperationalError",
    )
    saver.db = mock_store

    saver.save_event({"id": "s_fail_j_ok"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_persisted"] == 0
    assert stats["sqlite_events_failed"] == 1
    assert stats["jsonl_events_written"] == 1


# 19. SQLite OK / JSONL FAIL
def test_scenario_sqlite_ok_jsonl_fail(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    _patch_mainthread_sleep(monkeypatch)

    real_open = open
    def mock_open(path, mode="r", *args, **kwargs):
        if "events.jsonl" in str(path):
            raise OSError("JSONL disk error")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    saver.save_event({"id": "s_ok_j_fail"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_persisted"] == 1
    assert stats["jsonl_events_failed"] == 1
    assert stats["fallback_events_written"] == 1


# 20. SQLite FAIL / JSONL FAIL
def test_scenario_sqlite_fail_jsonl_fail(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    _patch_mainthread_sleep(monkeypatch)

    mock_store = MagicMock()
    mock_store.save_batch.return_value = BatchSaveResult(
        total_received=1,
        persisted=0,
        rejected_pre_commit=0,
        commit_status="FAILED",
        error_type="OperationalError",
    )
    saver.db = mock_store

    def mock_open(path, mode="r", *args, **kwargs):
        raise OSError("Global write failure")

    monkeypatch.setattr("builtins.open", mock_open)

    saver.save_event({"id": "s_fail_j_fail"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_failed"] == 1
    assert stats["jsonl_events_failed"] == 1
    assert stats["fallback_events_failed"] == 1


# 21. Rejeição parcial pré-commit refletida nas estatísticas
def test_rejeicao_parcial_pre_commit_stats(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    mock_store.save_batch.return_value = BatchSaveResult(
        total_received=4,
        persisted=3,
        rejected_pre_commit=1,
        commit_status="COMMITTED",
        error_type=None,
    )
    saver.db = mock_store

    for i in range(4):
        saver.save_event({"id": f"p_{i}"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 4
    assert stats["sqlite_events_persisted"] == 3
    assert stats["sqlite_events_failed"] == 1
    assert stats["sqlite_events_rejected_pre_commit"] == 1


# 22. Contadores independentes por destino
def test_contadores_independentes_por_destino(hermetic_saver_g2):
    saver = hermetic_saver_g2
    saver.save_event({"id": "iso_1"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 1
    assert stats["jsonl_events_attempted"] == 1
    assert stats["fallback_events_attempted"] == 0
    assert stats["sqlite_events_persisted"] == 1
    assert stats["jsonl_events_written"] == 1


# 23. Conservação G1 preservada
def test_conservacao_g1_preservada(hermetic_saver_g2):
    saver = hermetic_saver_g2
    for i in range(3):
        saver.save_event({"id": f"g1_{i}"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["events_received"] == 3
    assert stats["events_accepted"] == 3
    assert stats["events_in_flight"] == 0
    assert stats["events_finalized"] == 3
    assert stats["events_failed_terminal"] == 0


# 24. Warning rate-limited no worker
def test_warning_rate_limited(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    warn_calls = []
    monkeypatch.setattr(saver.logger, "warning", lambda *args, **kwargs: warn_calls.append(args))

    with saver._buffer_lock:
        saver._events_rejected_overflow = 10

    saver._check_and_log_rejection_warnings(force=True)
    assert len(warn_calls) == 1

    saver._check_and_log_rejection_warnings(force=False)
    assert len(warn_calls) == 1


# 25. Exception no logger não mata worker
def test_exception_no_logger_nao_mata_worker(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    def failing_logger(*args, **kwargs):
        raise RuntimeError("Logger crashed")
    monkeypatch.setattr(saver.logger, "warning", failing_logger)

    with saver._buffer_lock:
        saver._events_rejected_overflow = 5

    saver._check_and_log_rejection_warnings(force=True)
    assert saver._events_rejected_overflow == 5


# 26. Nenhum logging no caller
def test_nenhum_logging_no_caller(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    log_calls = []
    monkeypatch.setattr(saver.logger, "warning", lambda *args: log_calls.append(args))
    monkeypatch.setattr(saver.logger, "error", lambda *args: log_calls.append(args))
    monkeypatch.setattr(saver.logger, "info", lambda *args: log_calls.append(args))

    saver.save_event({"tipo_evento": "FAST_CALL"})
    assert len(log_calls) == 0


# 27. Shutdown durante I/O
def test_shutdown_durante_io(hermetic_saver_g2):
    saver = hermetic_saver_g2
    for i in range(10):
        saver.save_event({"id": f"drain_{i}"})
    saver.stop(timeout=2.0)

    stats = saver.get_stats()
    assert stats["events_finalized"] == 10
    assert stats["events_waiting"] == 0
    assert stats["events_in_flight"] == 0


# 28. Mocks legados retornando None
def test_mocks_legados_retornando_none(hermetic_saver_g2):
    saver = hermetic_saver_g2
    mock_store = MagicMock()
    mock_store.save_batch.return_value = None
    saver.db = mock_store

    saver.save_event({"id": "legacy_mock"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 1
    assert stats["sqlite_events_unknown"] == 1
    assert stats["sqlite_events_persisted"] == 0
    assert stats["sqlite_events_failed"] == 0


# 29. JSONL falha durante close() / __exit__ -> FAILED, não WRITTEN, fallback acionado (ETAPA 4)
def test_jsonl_falha_no_fechamento_close(hermetic_saver_g2, monkeypatch):
    saver = hermetic_saver_g2
    _patch_mainthread_sleep(monkeypatch)

    real_open = open

    class MockFileWithFailingClose:
        def __init__(self, target):
            self.target = target

        def write(self, *args, **kwargs):
            return self.target.write(*args, **kwargs)

        def flush(self, *args, **kwargs):
            return self.target.flush(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                self.target.close()
            except Exception:
                pass
            raise OSError(5, "I/O error during file close")

        def close(self):
            self.target.close()

    def mock_open(path, mode="r", *args, **kwargs):
        if "events.jsonl" in str(path):
            f = real_open(path, mode, *args, **kwargs)
            return MockFileWithFailingClose(f)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    saver.save_event({"id": "jsonl_close_fail"})
    saver._process_single_flush_batch()

    stats = saver.get_stats()
    assert stats["jsonl_events_written"] == 0
    assert stats["jsonl_events_failed"] == 1
    assert stats["jsonl_events_attempted"] == 1
    assert (
        stats["jsonl_events_attempted"]
        == stats["jsonl_events_written"]
        + stats["jsonl_events_failed"]
        + stats["jsonl_events_unknown"]
    )
    assert stats["fallback_events_attempted"] == 1
    assert stats["fallback_events_written"] == 1
