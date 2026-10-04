# -*- coding: utf-8 -*-
"""
Testes específicos para o ciclo de vida do ThreadPoolExecutor em WindowProcessor (Fase 5.8A - Achado #8).
"""
import concurrent.futures
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from market_orchestrator.windows import window_processor as wp


class DummyBot:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.window_count = 0
        self.should_stop = False


def _create_processor(symbol="BTCUSDT"):
    return wp.WindowProcessor(
        symbol=symbol,
        windows_minutes=[1],
        event_bus=MagicMock(),
        time_manager=MagicMock(),
    )


def test_success_calls_shutdown_wait_true():
    """1. Sucesso: executor.shutdown(wait=True, cancel_futures=True) deve ser chamado."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    shutdown_calls = []
    orig_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def _spy_shutdown(self, *args, **kwargs):
        shutdown_calls.append({"args": args, "kwargs": kwargs})
        return orig_shutdown(self, *args, **kwargs)

    def _dummy_process(b, wd, close_ms):
        pass

    with patch.object(concurrent.futures.ThreadPoolExecutor, "shutdown", _spy_shutdown), \
         patch.object(wp, "process_window_snapshot", _dummy_process):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(shutdown_calls) == 1
    assert shutdown_calls[0]["kwargs"] == {"wait": True, "cancel_futures": True}
    assert processor.window_count == 1


def test_worker_exception_calls_shutdown_wait_true():
    """2. Worker exception: executor.shutdown(wait=True, cancel_futures=True) deve ser chamado."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    shutdown_calls = []
    orig_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def _spy_shutdown(self, *args, **kwargs):
        shutdown_calls.append({"args": args, "kwargs": kwargs})
        return orig_shutdown(self, *args, **kwargs)

    def _failing_process(b, wd, close_ms):
        raise ValueError("Simulated pipeline failure")

    with patch.object(concurrent.futures.ThreadPoolExecutor, "shutdown", _spy_shutdown), \
         patch.object(wp, "process_window_snapshot", _failing_process):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(shutdown_calls) == 1
    assert shutdown_calls[0]["kwargs"] == {"wait": True, "cancel_futures": True}
    assert processor.window_count == 0


def test_submit_exception_calls_shutdown():
    """3. Submit exception: executor já criado deve ser fechado mesmo se submit falhar."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    shutdown_calls = []
    orig_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def _spy_shutdown(self, *args, **kwargs):
        shutdown_calls.append({"args": args, "kwargs": kwargs})
        return orig_shutdown(self, *args, **kwargs)

    def _failing_submit(self, *args, **kwargs):
        raise RuntimeError("Cannot schedule task")

    with patch.object(concurrent.futures.ThreadPoolExecutor, "shutdown", _spy_shutdown), \
         patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _failing_submit):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(shutdown_calls) == 1
    assert shutdown_calls[0]["kwargs"] == {"wait": True, "cancel_futures": True}
    assert processor.window_count == 0


def test_timeout_calls_shutdown_wait_false():
    """4. Timeout: executor.shutdown(wait=False, cancel_futures=True) deve ser chamado."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    shutdown_calls = []
    orig_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def _spy_shutdown(self, *args, **kwargs):
        shutdown_calls.append({"args": args, "kwargs": kwargs})
        return orig_shutdown(self, *args, **kwargs)

    orig_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _timeout_submit(self, fn, *args, **kwargs):
        future = orig_submit(self, fn, *args, **kwargs)
        # Mock result to raise TimeoutError
        future.result = MagicMock(side_effect=concurrent.futures.TimeoutError("Execution exceeded 45s"))
        return future

    with patch.object(concurrent.futures.ThreadPoolExecutor, "shutdown", _spy_shutdown), \
         patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _timeout_submit):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(shutdown_calls) == 1
    assert shutdown_calls[0]["kwargs"] == {"wait": False, "cancel_futures": True}
    assert processor.window_count == 0


def test_timeout_returns_control_immediately_without_waiting_long_thread():
    """5. Timeout retorna controle ao worker loop sem esperar thread longa."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    started_ev = threading.Event()
    release_ev = threading.Event()

    def _slow_task(b, wd, close_ms):
        started_ev.set()
        release_ev.wait(timeout=5.0)

    orig_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _fast_timeout_submit(self, fn, *args, **kwargs):
        future = orig_submit(self, fn, *args, **kwargs)

        def _mock_result(timeout=None):
            started_ev.wait(timeout=1.0)
            raise concurrent.futures.TimeoutError("Simulated 45s timeout")

        future.result = _mock_result
        return future

    t0 = time.perf_counter()
    with patch.object(wp, "process_window_snapshot", _slow_task), \
         patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _fast_timeout_submit):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    elapsed = time.perf_counter() - t0
    # Liberar a thread de background para cleanup limpo
    release_ev.set()

    # O worker loop deve ter retornado rapidamente (< 1.0s) em vez de esperar os 5s da thread lenta
    assert elapsed < 1.0
    assert processor.window_count == 0


def test_20_failing_windows_do_not_accumulate_orphan_threads():
    """6. 20 janelas que lançam exceção não acumulam executors/threads órfãos."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    def _failing_task(b, wd, close_ms):
        raise RuntimeError("Task failed")

    baseline_threads = threading.active_count()

    with patch.object(wp, "process_window_snapshot", _failing_task):
        for i in range(20):
            processor._queue.put_nowait((bot, window_data, 1000 + i))
        processor._worker_stop.set()
        processor._worker_loop()

    # Como o shutdown(wait=True) é garantido no finally de cada erro,
    # as 20 threads do executor foram finalizadas e unidas no loop.
    active_now = threading.active_count()
    assert active_now <= baseline_threads + 1
    assert processor.window_count == 0


def test_queue_task_done_called_exactly_once_per_item():
    """7. queue.task_done ocorre exatamente uma vez por item."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    task_done_calls = 0
    orig_task_done = processor._queue.task_done

    def _spy_task_done():
        nonlocal task_done_calls
        task_done_calls += 1
        orig_task_done()

    processor._queue.task_done = _spy_task_done

    call_index = 0

    def _variable_behavior(b, wd, close_ms):
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return  # Sucesso
        elif call_index == 2:
            raise ValueError("Falha")
        elif call_index == 3:
            raise RuntimeError("Falha 2")

    with patch.object(wp, "process_window_snapshot", _variable_behavior):
        processor._queue.put_nowait((bot, window_data, 1001))  # sucesso
        processor._queue.put_nowait((bot, window_data, 1002))  # exception
        processor._queue.put_nowait((bot, window_data, 1003))  # exception
        processor._queue.put_nowait(None)                     # sentinela
        processor._worker_stop.set()
        processor._worker_loop()

    assert task_done_calls == 4
    assert processor._queue.unfinished_tasks == 0


def test_nominal_window_count_behavior_preserved():
    """8. Comportamento nominal de window_count preservado."""
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    calls = 0

    def _selective_process(b, wd, close_ms):
        nonlocal calls
        calls += 1
        if calls in (2, 4):
            raise ValueError("Erro simulado")

    with patch.object(wp, "process_window_snapshot", _selective_process):
        # 5 janelas: 1 (ok), 2 (erro), 3 (ok), 4 (erro), 5 (ok)
        for i in range(5):
            processor._queue.put_nowait((bot, window_data, 1000 + i))
        processor._worker_stop.set()
        processor._worker_loop()

    # Apenas as 3 janelas com sucesso devem incrementar window_count
    assert processor.window_count == 3
