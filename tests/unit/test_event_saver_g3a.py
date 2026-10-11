# -*- coding: utf-8 -*-
"""
tests/unit/test_event_saver_g3a.py
Testes herméticos para a Fase 5.11 - Gate G3-A (Shutdown State & Worker Lifecycle).

Cobre rigorosamente os 15 requisitos de Gate G3-A:
1. Clean stop (sucesso integral).
2. Worker morto com buffer pendente (anomalia detectada, buffer preservado).
3. Worker morto com buffer vazio (anomalia detectada, shutdown_incomplete=True).
4. Timeout com worker vivo (INCOMPLETE_STOP declarado).
5. Worker termina após timeout (preservação de INCOMPLETE_STOP).
6. Stop duplicado (idempotência preserva CLEAN_STOP).
7. Stop concorrente (exclusão mútua e conclusão limpa).
8. Save durante stop (rejeição correta para events_rejected_after_stop).
9. Exception fatal no worker (diagnóstico capturado em worker_exception).
10. Get_stats expõe estado completo e consistente.
11. STOPPING durante drain ativo.
12. Timeout não reverte para CLEAN_STOP.
13. Worker não declara clean stop antes de terminar o drain.
14. Invariantes contábeis G1 estritamente preservadas.
15. Contadores por destino G2 preservados.
"""

import threading
import time
from unittest.mock import MagicMock
import pytest

from events import event_saver
from events.event_saver import EventSaver


@pytest.fixture
def hermetic_saver_g3a(tmp_path, monkeypatch):
    """Fixture hermética isolando dependências externas do EventSaver."""
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    mock_store = MagicMock()
    monkeypatch.setattr(event_saver, "EventStore", lambda: mock_store)

    saver = EventSaver(sound_alert=False)
    saver.db = mock_store
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    yield saver

    try:
        saver.stop(timeout=1.0)
    except Exception:
        pass


# =====================================================================
# 1. Clean stop (sucesso integral)
# =====================================================================
def test_clean_stop(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    flushed = []

    def mock_flush(batch):
        flushed.extend(batch)

    monkeypatch.setattr(saver, "_flush_buffer", mock_flush)

    saver.save_event({"id": "evt_1"})
    saver.save_event({"id": "evt_2"})

    saver.stop(timeout=2.0)

    assert saver.shutdown_status == "CLEAN_STOP"
    assert saver.shutdown_incomplete is False
    assert saver.worker_alive is False
    assert saver.worker_exited_cleanly is True
    assert saver.events_waiting == 0
    assert saver.events_in_flight == 0
    assert saver.events_finalized == 2
    assert len(flushed) == 2


# =====================================================================
# 2. Worker morto com buffer pendente
# =====================================================================
def test_worker_died_before_stop_with_pending_buffer(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    worker_died = threading.Event()

    def fatal_flush_batch():
        worker_died.set()
        raise KeyboardInterrupt("Simulated fatal kill")

    monkeypatch.setattr(saver, "_process_single_flush_batch", fatal_flush_batch)
    saver._wake_flush_event.set()

    assert worker_died.wait(timeout=2.0)
    saver._flush_thread.join(timeout=2.0)
    assert not saver.worker_alive

    # Enfileira evento com worker já morto (antes do stop)
    saver.save_event({"id": "pending_unprocessed"})
    assert saver.events_waiting == 1

    # Antes do stop, status já reflete a morte da thread
    assert saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"

    # Agora chama stop()
    saver.stop(timeout=1.0)

    # stop() deve detectar que o worker encerrou de forma anômala
    assert saver.shutdown_incomplete is True
    assert saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"
    assert saver.worker_exited_cleanly is False
    assert saver.events_waiting == 1  # Buffer residual NÃO foi descartado


# =====================================================================
# 3. Worker morto com buffer vazio
# =====================================================================
def test_worker_died_before_stop_with_empty_buffer(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    worker_died = threading.Event()

    def fatal_flush_batch():
        worker_died.set()
        raise KeyboardInterrupt("Simulated fatal kill")

    monkeypatch.setattr(saver, "_process_single_flush_batch", fatal_flush_batch)
    saver._wake_flush_event.set()

    assert worker_died.wait(timeout=2.0)
    saver._flush_thread.join(timeout=2.0)
    assert not saver.worker_alive
    assert saver.events_waiting == 0

    assert saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"

    saver.stop(timeout=1.0)

    assert saver.shutdown_incomplete is True
    assert saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"
    assert saver.worker_exited_cleanly is False


# =====================================================================
# 4. Timeout com worker vivo (INCOMPLETE_STOP)
# =====================================================================
def test_timeout_with_worker_alive(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    block_flush = threading.Event()
    worker_started = threading.Event()

    def slow_flush(batch):
        worker_started.set()
        block_flush.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", slow_flush)

    saver.save_event({"id": "blocking_evt"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    # Executa stop com timeout muito curto (50ms)
    saver.stop(timeout=0.05)

    assert saver.shutdown_incomplete is True
    assert saver.worker_alive is True
    assert saver.shutdown_status == "INCOMPLETE_STOP"

    block_flush.set()


# =====================================================================
# 5. Worker termina após timeout (preservação de INCOMPLETE_STOP)
# =====================================================================
def test_worker_finishes_after_timeout_preserves_incomplete(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    block_flush = threading.Event()
    worker_started = threading.Event()

    def slow_flush(batch):
        worker_started.set()
        block_flush.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", slow_flush)

    saver.save_event({"id": "late_evt"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    saver.stop(timeout=0.05)
    assert saver.shutdown_incomplete is True
    assert saver.shutdown_status == "INCOMPLETE_STOP"

    # Libera o worker para que ele termine seu processamento
    block_flush.set()
    saver._flush_thread.join(timeout=2.0)
    assert not saver._flush_thread.is_alive()

    # O status deve PERMANECER INCOMPLETE_STOP porque houve timeout no stop()
    assert saver.shutdown_incomplete is True
    assert saver.shutdown_status == "INCOMPLETE_STOP"


# =====================================================================
# 6. Stop duplicado (idempotência preserva CLEAN_STOP)
# =====================================================================
def test_duplicate_stop_preserves_clean_stop(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    monkeypatch.setattr(saver, "_flush_buffer", lambda batch: None)

    saver.save_event({"id": "evt_1"})
    saver.stop(timeout=2.0)

    assert saver.shutdown_status == "CLEAN_STOP"
    assert saver.shutdown_incomplete is False

    # Segunda e terceira chamadas não devem alterar o status nem lançar exceção
    saver.stop(timeout=1.0)
    assert saver.shutdown_status == "CLEAN_STOP"
    assert saver.shutdown_incomplete is False

    saver.stop(timeout=1.0)
    assert saver.shutdown_status == "CLEAN_STOP"
    assert saver.shutdown_incomplete is False


# =====================================================================
# 7. Stop concorrente (exclusão mútua)
# =====================================================================
def test_concurrent_stop(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    monkeypatch.setattr(saver, "_flush_buffer", lambda batch: None)

    saver.save_event({"id": "evt_conc"})

    def invoke_stop():
        saver.stop(timeout=2.0)

    t1 = threading.Thread(target=invoke_stop)
    t2 = threading.Thread(target=invoke_stop)

    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert saver.shutdown_status == "CLEAN_STOP"
    assert saver.shutdown_incomplete is False


# =====================================================================
# 8. Save durante stop (rejeição após stop)
# =====================================================================
def test_save_event_during_stop(hermetic_saver_g3a):
    saver = hermetic_saver_g3a
    saver.stop(timeout=2.0)

    # Tenta salvar após parada
    saver.save_event({"id": "too_late"})

    assert saver.events_rejected_after_stop == 1
    assert saver.events_accepted == 0
    assert saver.events_waiting == 0
    assert saver.shutdown_status == "CLEAN_STOP"


# =====================================================================
# 9. Exception fatal no worker (diagnóstico em worker_exception)
# =====================================================================
def test_fatal_exception_in_worker_captured(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    exception_thrown = threading.Event()

    def broken_single_flush():
        exception_thrown.set()
        raise KeyboardInterrupt("Simulando excecao fatal fora do try/except")

    # Força a exception na chamada do loop
    monkeypatch.setattr(saver, "_process_single_flush_batch", broken_single_flush)
    saver._wake_flush_event.set()

    assert exception_thrown.wait(timeout=2.0)
    saver._flush_thread.join(timeout=2.0)

    assert not saver.worker_alive
    assert saver.worker_exception is not None
    assert "KeyboardInterrupt" in saver.worker_exception
    assert "Simulando excecao fatal" in saver.worker_exception
    assert saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"


# =====================================================================
# 10. Get_stats expõe estado completo e consistente
# =====================================================================
def test_get_stats_exposes_complete_lifecycle(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    monkeypatch.setattr(saver, "_flush_buffer", lambda b: None)

    # Em execução normal
    stats_running = saver.get_stats()
    assert stats_running["shutdown_status"] == "RUNNING"
    assert stats_running["worker_alive"] is True
    assert stats_running["worker_exited_cleanly"] is False
    assert stats_running["worker_exception"] is None
    assert stats_running["shutdown_incomplete"] is False

    # Após parada limpa
    saver.save_event({"id": "stats_evt"})
    saver.stop(timeout=2.0)

    stats_stopped = saver.get_stats()
    assert stats_stopped["shutdown_status"] == "CLEAN_STOP"
    assert stats_stopped["worker_alive"] is False
    assert stats_stopped["worker_exited_cleanly"] is True
    assert stats_stopped["shutdown_incomplete"] is False


# =====================================================================
# 11. STOPPING durante drain ativo
# =====================================================================
def test_stopping_state_during_active_drain(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    drain_entered = threading.Event()
    unblock_drain = threading.Event()

    def slow_drain_flush(batch):
        drain_entered.set()
        unblock_drain.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", slow_drain_flush)

    saver.save_event({"id": "draining_evt"})

    def stop_in_background():
        saver.stop(timeout=2.0)

    t = threading.Thread(target=stop_in_background)
    t.start()

    assert drain_entered.wait(timeout=2.0)
    # Enquanto o worker drena antes do timeout, status deve ser STOPPING
    assert saver.shutdown_status == "STOPPING"

    unblock_drain.set()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert saver.shutdown_status == "CLEAN_STOP"


# =====================================================================
# 12. Timeout não vira CLEAN_STOP
# =====================================================================
def test_timeout_never_transitions_to_clean_stop(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    block_worker = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        block_worker.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    saver.save_event({"id": "timeout_evt"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    saver.stop(timeout=0.05)
    assert saver.shutdown_status == "INCOMPLETE_STOP"
    assert saver.shutdown_incomplete is True

    # Libera worker e chama stop() novamente
    block_worker.set()
    saver._flush_thread.join(timeout=2.0)
    saver.stop(timeout=1.0)

    # Mesmo após término com buffer vazio, timeout anterior impede CLEAN_STOP
    assert saver.events_waiting == 0
    assert saver.shutdown_incomplete is True
    assert saver.shutdown_status == "INCOMPLETE_STOP"


# =====================================================================
# 13. Worker não declara clean stop antes de terminar drain
# =====================================================================
def test_worker_does_not_declare_clean_before_drain_finishes(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    in_drain = threading.Event()
    proceed = threading.Event()

    def checked_flush(batch):
        # Durante o flush, worker_exited_cleanly DEVE ser False
        assert saver.worker_exited_cleanly is False
        in_drain.set()
        proceed.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", checked_flush)

    saver.save_event({"id": "check_clean"})

    t = threading.Thread(target=lambda: saver.stop(timeout=2.0))
    t.start()

    assert in_drain.wait(timeout=2.0)
    assert saver.worker_exited_cleanly is False

    proceed.set()
    t.join(timeout=3.0)
    assert saver.worker_exited_cleanly is True
    assert saver.shutdown_status == "CLEAN_STOP"


# =====================================================================
# 14. Invariantes contábeis G1 preservadas
# =====================================================================
def test_accounting_invariants_preserved(hermetic_saver_g3a, monkeypatch):
    saver = hermetic_saver_g3a
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 2)
    monkeypatch.setattr(saver, "_flush_buffer", lambda b: None)

    saver.save_event({"id": "ok_1"})
    saver.save_event({"id": "ok_2"})
    saver.save_event({"id": "overflow_3"})  # overflow

    saver.stop(timeout=2.0)
    saver.save_event({"id": "after_stop_4"})  # rejected after stop

    stats = saver.get_stats()
    # Invariante 1: received = accepted + overflow + after_stop
    assert stats["events_received"] == stats["events_accepted"] + stats["events_rejected_overflow"] + stats["events_rejected_after_stop"]
    # Invariante 2: accepted = waiting + in_flight + finalized + failed_terminal
    assert stats["events_accepted"] == stats["events_waiting"] + stats["events_in_flight"] + stats["events_finalized"] + stats["events_failed_terminal"]

    assert stats["events_accepted"] == 2
    assert stats["events_rejected_overflow"] == 1
    assert stats["events_rejected_after_stop"] == 1
    assert stats["events_finalized"] == 2


# =====================================================================
# 15. Contadores por destino G2 preservados
# =====================================================================
def test_g2_destination_counters_preserved(hermetic_saver_g3a):
    saver = hermetic_saver_g3a
    from database.event_store import BatchSaveResult

    # Mock do SQLite retornando resultado estruturado G2
    mock_res = BatchSaveResult(
        total_received=2,
        persisted=2,
        rejected_pre_commit=0,
        commit_status="COMMITTED",
        error_type=None,
    )
    saver.db.save_batch = MagicMock(return_value=mock_res)

    saver.save_event({"id": "g2_1"})
    saver.save_event({"id": "g2_2"})

    saver.stop(timeout=2.0)

    stats = saver.get_stats()
    assert stats["sqlite_events_attempted"] == 2
    assert stats["sqlite_events_persisted"] == 2
    assert stats["sqlite_events_failed"] == 0
    assert stats["jsonl_events_written"] == 2
    assert stats["shutdown_status"] == "CLEAN_STOP"
