# tests/unit/test_ai_thread_pool_concurrency.py
# -*- coding: utf-8 -*-
"""
Testes de concorrência e não-bloqueio para o pool de IA (Achados #7 e #7b).

Valida:
1. Eliminação do join bloqueante em run_ai_analysis_threaded.
2. Invariante estrita len(ai_thread_pool) <= max_ai_threads.
3. Descarte fail-fast de análise em saturação com tempo < 0.5s.
4. Liberação de slots por workers finalizados ou com erro.
5. Início atômico (Thread.start dentro do lock), eliminando a race de reserva.
6. Worker ultrarrápido finalizando enquanto caller libera lock (sem deadlock).
7. Falha em Thread.start remove a reserva atomicamente.
8. Não bloqueio do EventBus sob saturação de IA.
9. Teardown / shutdown seguro.
"""

import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from market_orchestrator.ai.ai_runner import run_ai_analysis_threaded
from events.event_bus import EventBus


@pytest.fixture(autouse=True)
def always_allow_throttler(monkeypatch):
    """Garante que o singleton global de rate limiter de IA não bloqueie os testes de concorrência."""
    monkeypatch.setattr(
        "market_orchestrator.ai.ai_runner._ai_throttler",
        type(
            "_AlwaysAllowThrottler",
            (),
            {"should_call_ai": lambda *a, **k: True, "get_status": lambda *a, **k: {}},
        )(),
    )


class DummyRateLimiter:
    def acquire(self):
        pass


class FakeBot:
    """Implementação hermética mínima compatível com BotAIRuntimeProtocol."""

    def __init__(self, max_ai_threads: int = 3):
        self.symbol = "BTCUSDT"
        self.should_stop = False
        self.ai_runner = None
        self.ai_analyzer = MagicMock()
        self.ai_initialization_attempted = True
        self.ai_test_passed = True
        self.ai_thread_pool = []
        self.max_ai_threads = max_ai_threads
        self.ml_engine = None

        self._ai_init_lock = threading.Lock()
        self._ai_pool_lock = threading.Lock()
        self.ai_semaphore = threading.Semaphore(max_ai_threads)
        self.ai_rate_limiter = DummyRateLimiter()

        self.health_monitor = MagicMock()
        self.event_saver = MagicMock()
        self.feature_calc = None
        self.slog = MagicMock()
        self.tracer = MagicMock()

    def _wait_for_ai_threads(self, timeout_per_thread: float = 2.0) -> None:
        with self._ai_pool_lock:
            threads = list(self.ai_thread_pool)
        for t in threads:
            try:
                t.join(timeout=timeout_per_thread)
            except Exception:
                pass


@pytest.fixture
def fake_bot():
    bot = FakeBot(max_ai_threads=3)
    yield bot
    bot._wait_for_ai_threads(timeout_per_thread=0.5)


@pytest.fixture
def fake_event_data():
    return {
        "tipo_evento": "ABSORCAO",
        "resultado_da_batalha": "BULLISH",
        "volume_total": 500000.0,
        "regime_analysis": {"volatility_regime": "TRENDING"},
    }


# ===================================================================
# 1. Pool vazio -> 1 worker inicia
# ===================================================================
def test_empty_pool_starts_one_worker(fake_bot, fake_event_data):
    worker_started = threading.Event()
    worker_release = threading.Event()

    def mock_analyze(*args, **kwargs):
        worker_started.set()
        worker_release.wait(timeout=2.0)
        return {"action": "buy", "confidence": 0.9}

    fake_bot.ai_analyzer.analyze = mock_analyze

    run_ai_analysis_threaded(fake_bot, fake_event_data)
    assert worker_started.wait(timeout=5.0), "Worker não iniciou"

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 1

    worker_release.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)

    with fake_bot._ai_pool_lock:
        fake_bot.ai_thread_pool = [
            th for th in fake_bot.ai_thread_pool if th.is_alive()
        ]
        assert len(fake_bot.ai_thread_pool) == 0


# ===================================================================
# 2. Pool abaixo do limite -> workers iniciam até max
# ===================================================================
def test_pool_below_limit_starts_workers_up_to_max(fake_bot, fake_event_data):
    events_started = [threading.Event() for _ in range(3)]
    release_all = threading.Event()

    def make_mock(idx):
        def _analyze(*args, **kwargs):
            events_started[idx].set()
            release_all.wait(timeout=2.0)
            return {"action": "hold", "confidence": 0.8}
        return _analyze

    for i in range(3):
        fake_bot.ai_analyzer.analyze = make_mock(i)
        run_ai_analysis_threaded(fake_bot, fake_event_data)
        assert events_started[i].wait(timeout=3.0)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 3

    release_all.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 3. Pool == max -> nova análise retorna rapidamente sem criar thread
# ===================================================================
def test_pool_at_max_returns_immediately_without_creating_thread(
    fake_bot, fake_event_data
):
    release_workers = threading.Event()
    started_count = [0]
    count_lock = threading.Lock()

    def blocking_analyze(*args, **kwargs):
        with count_lock:
            started_count[0] += 1
        release_workers.wait(timeout=3.0)
        return {"action": "wait", "confidence": 0.5}

    fake_bot.ai_analyzer.analyze = blocking_analyze

    for _ in range(3):
        run_ai_analysis_threaded(fake_bot, fake_event_data)

    time.sleep(0.1)
    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 3

    # Tenta 4ª análise: deve ser descartada imediatamente
    run_ai_analysis_threaded(fake_bot, fake_event_data)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 3
    with count_lock:
        assert started_count[0] == 3

    release_workers.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 4. Medir que saturação não espera ~5s (bound generoso < 0.5s)
# ===================================================================
def test_saturation_does_not_wait_5_seconds(fake_bot, fake_event_data):
    release_workers = threading.Event()

    def slow_analyze(*args, **kwargs):
        release_workers.wait(timeout=5.0)
        return {"action": "wait", "confidence": 0.5}

    fake_bot.ai_analyzer.analyze = slow_analyze

    for _ in range(3):
        run_ai_analysis_threaded(fake_bot, fake_event_data)

    time.sleep(0.05)

    t0 = time.perf_counter()
    run_ai_analysis_threaded(fake_bot, fake_event_data)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.5, f"Saturação demorou {elapsed:.3f}s (esperado < 0.5s, longe de 5s)"

    release_workers.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 5. Pool nunca excede max sob rajada concorrente
# ===================================================================
def test_burst_concurrency_never_exceeds_max(fake_bot, fake_event_data):
    release_workers = threading.Event()

    def busy_analyze(*args, **kwargs):
        release_workers.wait(timeout=2.0)
        return {"action": "buy", "confidence": 0.8}

    fake_bot.ai_analyzer.analyze = busy_analyze

    threads = []
    for _ in range(20):
        th = threading.Thread(
            target=run_ai_analysis_threaded, args=(fake_bot, fake_event_data)
        )
        threads.append(th)

    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=1.0)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) <= 3

    release_workers.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 6. Worker terminado libera slot e nova análise pode iniciar
# ===================================================================
def test_finished_worker_releases_slot_and_allows_new(fake_bot, fake_event_data):
    release_1 = threading.Event()
    w1_started = threading.Event()

    def w1_analyze(*args, **kwargs):
        w1_started.set()
        release_1.wait(timeout=2.0)
        return {"action": "sell", "confidence": 0.9}

    fake_bot.ai_analyzer.analyze = w1_analyze
    fake_bot.max_ai_threads = 1
    run_ai_analysis_threaded(fake_bot, fake_event_data)
    assert w1_started.wait(timeout=3.0)

    run_ai_analysis_threaded(fake_bot, fake_event_data)
    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 1

    release_1.set()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with fake_bot._ai_pool_lock:
            if len(fake_bot.ai_thread_pool) == 0:
                break
        time.sleep(0.02)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 0

    w2_started = threading.Event()

    def w2_analyze(*args, **kwargs):
        w2_started.set()
        return {"action": "buy", "confidence": 0.85}

    fake_bot.ai_analyzer.analyze = w2_analyze
    run_ai_analysis_threaded(fake_bot, fake_event_data)
    assert w2_started.wait(timeout=3.0), "W2 não conseguiu iniciar após liberação de slot"

    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 7. Worker com exception limpa slot
# ===================================================================
def test_worker_with_exception_cleans_slot(fake_bot, fake_event_data):
    fake_bot.max_ai_threads = 1
    w_error_started = threading.Event()

    def failing_analyze(*args, **kwargs):
        w_error_started.set()
        raise RuntimeError("Falha de API de teste")

    fake_bot.ai_analyzer.analyze = failing_analyze
    run_ai_analysis_threaded(fake_bot, fake_event_data)
    assert w_error_started.wait(timeout=3.0)

    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 0, "Worker com erro não limpou slot"


# ===================================================================
# 8. max_ai_threads=1 sob rajada continua <= 1
# ===================================================================
def test_max_ai_threads_1_under_burst(fake_bot, fake_event_data):
    fake_bot.max_ai_threads = 1
    release = threading.Event()

    def slow_analyze(*args, **kwargs):
        release.wait(timeout=2.0)
        return {"action": "wait", "confidence": 0.5}

    fake_bot.ai_analyzer.analyze = slow_analyze

    threads = [
        threading.Thread(
            target=run_ai_analysis_threaded, args=(fake_bot, fake_event_data)
        )
        for _ in range(10)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=1.0)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 1

    release.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 9. Nenhuma chamada a Thread.join no hot path de saturação
# ===================================================================
def test_no_thread_join_called_in_hot_path(fake_bot, fake_event_data, monkeypatch):
    release = threading.Event()

    def hold_analyze(*args, **kwargs):
        release.wait(timeout=2.0)
        return {"action": "wait", "confidence": 0.5}

    fake_bot.ai_analyzer.analyze = hold_analyze

    for _ in range(fake_bot.max_ai_threads):
        run_ai_analysis_threaded(fake_bot, fake_event_data)

    join_calls = []
    real_join = threading.Thread.join

    def spy_join(self, *args, **kwargs):
        join_calls.append(self)
        return real_join(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "join", spy_join)

    run_ai_analysis_threaded(fake_bot, fake_event_data)
    assert len(join_calls) == 0, "Thread.join foi chamado indevidamente no hot path"

    monkeypatch.undo()
    release.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 10. PARTE C — Worker Ultrarrápido: Sem deadlock e slot limpo
# ===================================================================
def test_ultrafast_worker_cleans_slot_without_deadlock(fake_bot, fake_event_data):
    """
    Testa que um worker que termina em microssegundos (enquanto o caller
    está retornando) limpa o slot sem deadlock e permite nova análise imediata.
    """
    fake_bot.max_ai_threads = 1

    def instant_analyze(*args, **kwargs):
        # Retorna instantaneamente
        return {"action": "buy", "confidence": 0.95}

    fake_bot.ai_analyzer.analyze = instant_analyze

    for _ in range(5):
        run_ai_analysis_threaded(fake_bot, fake_event_data)
        # Pequeno intervalo para a thread terminar
        deadline = time.time() + 1.0
        while time.time() < deadline:
            with fake_bot._ai_pool_lock:
                if len(fake_bot.ai_thread_pool) == 0:
                    break
            time.sleep(0.01)

        with fake_bot._ai_pool_lock:
            assert len(fake_bot.ai_thread_pool) == 0, "Slot não foi liberado pelo worker ultrarrápido"


# ===================================================================
# 11. PARTE D — Falha em Thread.start remove a reserva atomicamente
# ===================================================================
def test_thread_start_failure_removes_reservation(fake_bot, fake_event_data, monkeypatch):
    """
    Testa que se Thread.start() levantar RuntimeError (ex: exaustão de threads no SO),
    a reserva é removida atomicamente sob o lock e não causa vazamento de slot.
    """
    fake_bot.max_ai_threads = 1

    def mock_start_fail(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", mock_start_fail)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        run_ai_analysis_threaded(fake_bot, fake_event_data)

    # Verifica que a thread foi removida da lista mesmo após a exceção
    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 0, "Slot vazou após falha no start()"

    monkeypatch.undo()

    # Prova que após o erro, novas chamadas normais conseguem usar o slot
    fake_bot.ai_analyzer.analyze = lambda *a, **k: {"action": "wait", "confidence": 0.5}
    run_ai_analysis_threaded(fake_bot, fake_event_data)

    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 1

    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)


# ===================================================================
# 12. PARTE G — EventBus não é bloqueado por saturação de IA
# ===================================================================
def test_event_bus_not_blocked_on_ai_saturation(fake_bot, fake_event_data):
    bus = EventBus(max_queue_size=100)
    fake_bot.max_ai_threads = 1
    ai_release = threading.Event()

    def slow_ai_analyze(*args, **kwargs):
        ai_release.wait(timeout=3.0)
        return {"action": "buy", "confidence": 0.9}

    fake_bot.ai_analyzer.analyze = slow_ai_analyze

    def signal_handler(evt):
        run_ai_analysis_threaded(fake_bot, evt)

    control_received = threading.Event()
    control_delivery_time = [0.0]

    def control_handler(evt):
        control_delivery_time[0] = time.perf_counter()
        control_received.set()

    bus.subscribe("signal", signal_handler)
    bus.subscribe("control", control_handler)

    bus.publish("signal", fake_event_data)
    time.sleep(0.05)
    with fake_bot._ai_pool_lock:
        assert len(fake_bot.ai_thread_pool) == 1

    t_publish = time.perf_counter()
    bus.publish("signal", fake_event_data)
    bus.publish("control", {"tipo": "PING"})

    assert control_received.wait(timeout=0.8), "EventBus ficou congelado pelo handler de IA!"
    delay = control_delivery_time[0] - t_publish
    assert delay < 0.4, f"Atraso do EventBus foi {delay:.3f}s (esperado < 0.4s)"

    ai_release.set()
    fake_bot._wait_for_ai_threads(timeout_per_thread=1.0)
    bus._stop = True
