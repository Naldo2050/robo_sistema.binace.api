# -*- coding: utf-8 -*-
"""
tests/unit/test_event_saver_g1.py
Testes herméticos para a Fase 5.11 - Gate G1 (Implementação Controlada do EventSaver).

Cobre os 16 requisitos mandatórios de Gate G1:
1. save_event não executa flush na thread caller.
2. save_event permanece responsivo com worker bloqueado em I/O simulado.
3. Overflow rejeita evento novo.
4. Overflow não descarta eventos antigos.
5. Contadores de received/accepted/rejected permanecem consistentes.
6. Worker acorda por backlog.
7. Worker executa flush periódico.
8. Shutdown drena buffer quando I/O responde normalmente.
9. Shutdown retorna no timeout quando worker está bloqueado.
10. Shutdown incompleto não contabiliza in-flight como perda definitiva.
11. save_event após stop é rejeitado.
12. stop duplicado é seguro.
13. Nunca ocorre flush concorrente.
14. Exception inesperada no flush não desaparece da contabilidade.
15. Worker não executa busy loop.
16. Corrida entre save_event e stop.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from events import event_saver
from events.event_saver import EventSaver


@pytest.fixture
def hermetic_saver(tmp_path, monkeypatch):
    """Fixture para inicializar EventSaver de forma hermética."""
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)
    if hasattr(event_saver, "config") and event_saver.config is not None:
        monkeypatch.setattr(
            event_saver.config,
            "EVENT_SAVER_JSONL_ARCHIVE_DIR",
            str(tmp_path / "archive"),
            raising=False,
        )

    mock_store = MagicMock()
    monkeypatch.setattr(event_saver, "EventStore", lambda: mock_store)
    monkeypatch.chdir(tmp_path)

    saver = EventSaver(sound_alert=False)
    saver.db = mock_store
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"
    saver.archive_dir = tmp_path / "archive"

    yield saver

    # Teardown seguro
    try:
        saver.stop(timeout=1.0)
    except Exception:
        pass


# =====================================================================
# 1. save_event não executa flush na thread caller
# =====================================================================
def test_save_event_does_not_flush_on_caller_thread(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    flush_threads = []
    flush_event = threading.Event()

    def mock_flush(batch):
        flush_threads.append(threading.current_thread().ident)
        flush_event.set()

    monkeypatch.setattr(saver, "_flush_buffer", mock_flush)

    caller_tid = threading.current_thread().ident
    test_event = {"tipo_evento": "TEST_EVENT", "symbol": "BTCUSDT"}

    saver.save_event(test_event)
    # Acorda o worker para processar o lote
    saver._wake_flush_event.set()

    # Aguarda o worker processar
    assert flush_event.wait(timeout=2.0)
    assert caller_tid not in flush_threads
    assert len(flush_threads) == 1
    assert flush_threads[0] == saver._flush_thread.ident


# =====================================================================
# 2. save_event permanece responsivo com worker bloqueado em I/O
# =====================================================================
def test_save_event_remains_responsive_with_worker_blocked_in_io(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    io_block = threading.Event()
    worker_entered_io = threading.Event()

    def blocking_flush(batch):
        worker_entered_io.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    # Envia primeiro evento e acorda o worker para engajá-lo em I/O
    saver.save_event({"tipo_evento": "FIRST_EVENT"})
    saver._wake_flush_event.set()
    assert worker_entered_io.wait(timeout=2.0)

    # Worker está agora travado em I/O. As próximas chamadas devem ser não-bloqueantes.
    t0 = time.perf_counter()
    for i in range(10):
        saver.save_event({"tipo_evento": f"FAST_EVENT_{i}"})
    duration = time.perf_counter() - t0

    # Chamadas em memória não devem ter esperado o I/O (muito abaixo de 0.5s)
    assert duration < 0.5

    # Libera o worker
    io_block.set()
    saver.stop(timeout=2.0)


# =====================================================================
# 3. Overflow rejeita evento novo
# =====================================================================
def test_overflow_rejects_new_event(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 3)

    # Bloqueia o worker para acumular no buffer
    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    # 1 evento vai para o worker (in-flight)
    saver.save_event({"id": "in_flight"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    # Enche o buffer até o limite (3 eventos em _write_buffer)
    for i in range(3):
        saver.save_event({"id": f"buf_{i}"})

    assert saver.events_waiting == 3

    # Tentativa de inserir além da capacidade
    overflow_event = {"id": "overflow_event"}
    saver.save_event(overflow_event)

    assert saver.events_rejected_overflow == 1
    # Verifica que overflow_event não foi admitido
    with saver._buffer_lock:
        ids_in_buf = [e.get("id") for e in saver._write_buffer]
        assert "overflow_event" not in ids_in_buf

    io_block.set()
    saver.stop(timeout=2.0)


# =====================================================================
# 4. Overflow não descarta eventos antigos
# =====================================================================
def test_overflow_does_not_evict_old_events(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 3)

    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    saver.save_event({"id": "first_active"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    for i in range(3):
        saver.save_event({"id": f"old_{i}"})

    # Tenta inserir no overflow
    saver.save_event({"id": "new_rejected"})

    with saver._buffer_lock:
        assert len(saver._write_buffer) == 3
        # Eventos antigos permanecem na ordem exata (sem descarte FIFO dos antigos)
        assert saver._write_buffer[0]["id"] == "old_0"
        assert saver._write_buffer[1]["id"] == "old_1"
        assert saver._write_buffer[2]["id"] == "old_2"

    io_block.set()
    saver.stop(timeout=2.0)


# =====================================================================
# 5. Contadores de received/accepted/rejected permanecem consistentes
# =====================================================================
def test_accounting_counters_remain_consistent(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 2)

    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    # 1 evento em voo
    saver.save_event({"id": "e_flight"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    # 2 eventos no buffer
    saver.save_event({"id": "e_buf_1"})
    saver.save_event({"id": "e_buf_2"})

    # 2 eventos rejeitados por overflow
    saver.save_event({"id": "e_over_1"})
    saver.save_event({"id": "e_over_2"})

    # Desbloqueia e encerra
    io_block.set()
    saver.stop(timeout=2.0)

    # 2 eventos após stop
    saver.save_event({"id": "e_stop_1"})
    saver.save_event({"id": "e_stop_2"})

    stats = saver.get_stats()

    # Invariante 1: received = accepted + rejected_overflow + rejected_after_stop
    assert stats["events_received"] == 7
    assert stats["events_accepted"] == 3
    assert stats["events_rejected_overflow"] == 2
    assert stats["events_rejected_after_stop"] == 2
    assert stats["events_received"] == (
        stats["events_accepted"]
        + stats["events_rejected_overflow"]
        + stats["events_rejected_after_stop"]
    )

    # Invariante 2: accepted = waiting + in_flight + finalized + failed_terminal
    assert stats["events_accepted"] == (
        stats["events_waiting"]
        + stats["events_in_flight"]
        + stats["events_finalized"]
        + stats["events_failed_terminal"]
    )
    assert stats["events_finalized"] == 3


# =====================================================================
# 6. Worker acorda por backlog
# =====================================================================
def test_worker_wakes_up_on_backlog(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    # Intervalo longo para provar que o wake-up ocorreu antes do timer periódico
    saver._flush_interval = 60.0
    monkeypatch.setattr(event_saver, "BACKPRESSURE_WAKE_THRESHOLD", 3)
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 10)

    flushed_event = threading.Event()

    def mock_flush(batch):
        flushed_event.set()

    monkeypatch.setattr(saver, "_flush_buffer", mock_flush)

    # Envia 3 eventos para acionar o threshold
    for i in range(3):
        saver.save_event({"id": f"wake_{i}"})

    # Deve acordar quase instantaneamente (bem antes de 60s)
    assert flushed_event.wait(timeout=2.0)


# =====================================================================
# 7. Worker executa flush periódico
# =====================================================================
def test_worker_executes_periodic_flush(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    saver._flush_interval = 0.05  # 50ms
    # Acorda o worker para renovar o timeout no wait com o novo interval
    saver._wake_flush_event.set()
    time.sleep(0.01)

    flushed_event = threading.Event()

    def mock_flush(batch):
        flushed_event.set()

    monkeypatch.setattr(saver, "_flush_buffer", mock_flush)

    # Envia 1 evento (abaixo do threshold de backlog)
    saver.save_event({"id": "periodic_event"})

    # Deve disparar pelo timer periódico (50ms)
    assert flushed_event.wait(timeout=2.0)


# =====================================================================
# 8. Shutdown drena buffer quando I/O responde normalmente
# =====================================================================
def test_shutdown_drains_buffer_when_io_responds(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    saver._flush_interval = 60.0
    monkeypatch.setattr(event_saver, "BACKPRESSURE_WAKE_THRESHOLD", 1000)

    flushed_batches = []

    def mock_flush(batch):
        flushed_batches.append(list(batch))

    monkeypatch.setattr(saver, "_flush_buffer", mock_flush)

    # Envia 5 eventos
    for i in range(5):
        saver.save_event({"id": f"drain_{i}"})

    # stop() deve solicitar drain e aguardar conclusão
    saver.stop(timeout=5.0)

    assert saver.shutdown_incomplete is False
    assert saver.events_waiting == 0
    assert saver.events_in_flight == 0
    assert saver.events_finalized == 5
    assert saver.events_accepted == 5


# =====================================================================
# 9. Shutdown retorna no timeout quando worker está bloqueado
# =====================================================================
def test_shutdown_returns_on_timeout_when_worker_blocked(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    saver.save_event({"id": "blocked_event"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    # Shutdown com timeout pequeno (100ms)
    t0 = time.perf_counter()
    saver.stop(timeout=0.1)
    duration = time.perf_counter() - t0

    assert duration < 0.6  # Não ficou preso indefinidamente
    assert saver.shutdown_incomplete is True

    io_block.set()


# =====================================================================
# 10. Shutdown incompleto não contabiliza in-flight como perda definitiva
# =====================================================================
def test_shutdown_incomplete_does_not_count_in_flight_as_terminal_loss(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    saver.save_event({"id": "in_flight_event"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    saver.stop(timeout=0.1)

    assert saver.shutdown_incomplete is True
    # In-flight continua registrado como in-flight e NÃO como failed_terminal
    assert saver.events_in_flight == 1
    assert saver.events_failed_terminal == 0

    io_block.set()


# =====================================================================
# 11. save_event após stop é rejeitado
# =====================================================================
def test_save_event_after_stop_is_rejected(hermetic_saver):
    saver = hermetic_saver
    saver.stop(timeout=2.0)

    saver.save_event({"id": "rejected_after_stop"})

    assert saver.events_rejected_after_stop == 1
    assert saver.events_accepted == 0
    assert saver.events_waiting == 0


# =====================================================================
# 12. stop duplicado é seguro
# =====================================================================
def test_duplicate_stop_is_safe(hermetic_saver):
    saver = hermetic_saver
    saver.save_event({"id": "test_event"})

    # Múltiplas chamadas não causam erro nem deadlock
    saver.stop(timeout=1.0)
    saver.stop(timeout=1.0)
    saver.stop(timeout=1.0)

    assert saver.events_finalized == 1
    assert not saver._flush_thread.is_alive()


# =====================================================================
# 13. Nunca ocorre flush concorrente
# =====================================================================
def test_never_concurrent_flush(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    active_flushes = 0
    max_concurrent = 0
    concurrency_lock = threading.Lock()

    def monitored_flush(batch):
        nonlocal active_flushes, max_concurrent
        with concurrency_lock:
            active_flushes += 1
            if active_flushes > max_concurrent:
                max_concurrent = active_flushes
        time.sleep(0.01)
        with concurrency_lock:
            active_flushes -= 1

    monkeypatch.setattr(saver, "_flush_buffer", monitored_flush)

    # Dispara múltiplos threads chamando save_event
    barrier = threading.Barrier(5)

    def producer():
        barrier.wait()
        for i in range(10):
            saver.save_event({"id": f"conc_{i}"})

    threads = [threading.Thread(target=producer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    saver.stop(timeout=3.0)

    assert max_concurrent <= 1


# =====================================================================
# 14. Exception inesperada no flush não desaparece da contabilidade
# =====================================================================
def test_unexpected_exception_in_flush_retains_accounting(hermetic_saver, monkeypatch):
    saver = hermetic_saver

    def failing_flush(batch):
        raise RuntimeError("Simulated Database / Disk Crash")

    monkeypatch.setattr(saver, "_flush_buffer", failing_flush)

    saver.save_event({"id": "fail_1"})
    saver.save_event({"id": "fail_2"})

    saver.stop(timeout=2.0)

    # Os eventos não foram finalizados com sucesso
    assert saver.events_finalized == 0
    # Mas foram retidos na contabilidade como failed_terminal
    assert saver.events_failed_terminal == 2
    assert saver.events_accepted == 2
    # Invariante preservada
    assert saver.events_accepted == (
        saver.events_waiting
        + saver.events_in_flight
        + saver.events_finalized
        + saver.events_failed_terminal
    )


# =====================================================================
# 15. Worker não executa busy loop
# =====================================================================
def test_worker_does_not_busy_loop(hermetic_saver, monkeypatch):
    saver = hermetic_saver
    saver._flush_interval = 1.0

    batch_process_calls = 0
    orig_batch = saver._process_single_flush_batch

    def tracked_batch():
        nonlocal batch_process_calls
        batch_process_calls += 1
        return orig_batch()

    monkeypatch.setattr(saver, "_process_single_flush_batch", tracked_batch)

    # Ocioso durante 100ms
    time.sleep(0.1)

    # Em 100ms com interval=1.0s e buffer vazio, não deve ter iterado centenas de vezes
    assert batch_process_calls <= 2

    saver.stop(timeout=1.0)


# =====================================================================
# 16. Corrida entre save_event e stop
# =====================================================================
def test_race_between_save_event_and_stop(hermetic_saver):
    saver = hermetic_saver
    num_threads = 4
    events_per_thread = 25
    barrier = threading.Barrier(num_threads + 1)

    def producer():
        barrier.wait()
        for i in range(events_per_thread):
            saver.save_event({"id": f"race_{i}"})

    def stopper():
        barrier.wait()
        saver.stop(timeout=3.0)

    threads = [threading.Thread(target=producer) for _ in range(num_threads)]
    stop_thread = threading.Thread(target=stopper)

    for t in threads:
        t.start()
    stop_thread.start()

    for t in threads:
        t.join()
    stop_thread.join()

    stats = saver.get_stats()
    # Equação estrita de conservação de eventos
    assert stats["events_received"] == (
        stats["events_accepted"]
        + stats["events_rejected_overflow"]
        + stats["events_rejected_after_stop"]
    )
    assert stats["events_accepted"] == (
        stats["events_waiting"]
        + stats["events_in_flight"]
        + stats["events_finalized"]
        + stats["events_failed_terminal"]
    )


# =====================================================================
# 17. Overflow não executa logging síncrono no hot path
# =====================================================================
def test_overflow_does_not_invoke_synchronous_logging(hermetic_saver, monkeypatch):
    """Garante que no overflow save_event() não executa logging síncrono no hot path."""
    saver = hermetic_saver
    monkeypatch.setattr(event_saver, "MAX_BUFFER_SIZE", 2)

    io_block = threading.Event()
    worker_started = threading.Event()

    def blocking_flush(batch):
        worker_started.set()
        io_block.wait(timeout=5.0)

    monkeypatch.setattr(saver, "_flush_buffer", blocking_flush)

    # 1 evento em voo
    saver.save_event({"id": "in_flight"})
    saver._wake_flush_event.set()
    assert worker_started.wait(timeout=2.0)

    # Enche o buffer até o limite (2 eventos)
    saver.save_event({"id": "buf_1"})
    saver.save_event({"id": "buf_2"})

    # Espiona chamadas de logging no logger do saver
    mock_warning = MagicMock()
    mock_error = MagicMock()
    mock_critical = MagicMock()
    monkeypatch.setattr(saver.logger, "warning", mock_warning)
    monkeypatch.setattr(saver.logger, "error", mock_error)
    monkeypatch.setattr(saver.logger, "critical", mock_critical)

    # Tentativa de inserir além do limite (overflow)
    res = saver.save_event({"id": "overflow_candidate"})

    assert res is None
    assert saver.events_rejected_overflow == 1
    assert mock_warning.call_count == 0
    assert mock_error.call_count == 0
    assert mock_critical.call_count == 0

    io_block.set()
    saver.stop(timeout=2.0)
