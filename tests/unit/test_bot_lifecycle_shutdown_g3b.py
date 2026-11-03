"""
Testes Unitários para o Lifecycle de Shutdown do EnhancedMarketBot (Gate G3-B).
Validação do shutdown assíncrono do EventSaver fora do event loop principal,
resiliência a cancelamentos (inclusive repetidos), remoção do fallback síncrono,
espera fora do lock e preservação dos contratos G1/G2/G3-A.
"""

import asyncio
import logging
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from events import event_saver
from events.event_saver import EventSaver
from market_orchestrator.market_orchestrator import EnhancedMarketBot


@pytest.fixture(autouse=True)
def isolate_bot_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(EnhancedMarketBot, "_initialize_ai_async", lambda self: None)
    monkeypatch.setattr(EnhancedMarketBot, "_init_shadow_runtime", lambda self: None)
    monkeypatch.setattr("market_orchestrator.market_orchestrator.HealthMonitor", MagicMock)
    monkeypatch.setattr("market_orchestrator.market_orchestrator.FeatureStore", MagicMock)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(
        event_saver.config,
        "EVENT_SAVER_JSONL_ARCHIVE_DIR",
        str(tmp_path / "archive"),
        raising=False,
    )

    mock_store = MagicMock()
    mock_store.save_batch.return_value = event_saver.BatchSaveResult(
        total_received=1, persisted=1, rejected_pre_commit=0, commit_status="COMMITTED"
    )
    monkeypatch.setattr(event_saver, "EventStore", lambda: mock_store)
    try:
        import database.event_store
        monkeypatch.setattr(database.event_store, "EventStore", lambda *args, **kwargs: mock_store)
    except Exception:
        pass
    yield


def _make_dummy_bot():
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager"), \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer"), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor"), \
         patch("market_orchestrator.market_orchestrator.OrderBookAnalyzer"), \
         patch("market_orchestrator.market_orchestrator.EventBus"), \
         patch("market_orchestrator.market_orchestrator.HealthMonitor"):
        bot = EnhancedMarketBot(
            stream_url="wss://test.stream",
            symbol="BTCUSDT",
            window_size_minutes=1,
            vol_factor_exh=1.0,
            history_size=10,
            delta_std_dev_factor=1.0,
            context_sma_period=10,
            liquidity_flow_alert_percentage=10.0,
            wall_std_dev_factor=1.0,
        )
        if hasattr(bot, "event_saver") and bot.event_saver and hasattr(bot.event_saver, "stop"):
            bot.event_saver.stop(timeout=1.0)
        return bot


@pytest.mark.asyncio
async def test_event_loop_remains_responsive_during_event_saver_stop():
    """1. Garante que o event loop continua responsivo enquanto EventSaver.stop executa em thread."""
    bot = _make_dummy_bot()

    def slow_stop():
        time.sleep(0.12)

    bot.event_saver = MagicMock()
    bot.event_saver.stop = slow_stop

    loop_ticks = 0

    async def loop_heartbeat():
        nonlocal loop_ticks
        while not bot._is_shutdown or (bot._shutdown_completed_event and not bot._shutdown_completed_event.is_set()):
            await asyncio.sleep(0.02)
            loop_ticks += 1

    heartbeat_task = asyncio.create_task(loop_heartbeat())
    await bot.shutdown()
    await heartbeat_task

    assert loop_ticks >= 3, f"Event loop congelou durante shutdown! loop_ticks={loop_ticks}"
    assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_event_saver_stop_runs_in_auxiliary_thread():
    """2. Garante que EventSaver.stop executa em thread separada do event loop principal."""
    bot = _make_dummy_bot()
    main_thread_id = threading.get_ident()
    stop_thread_id = None

    def record_thread_stop():
        nonlocal stop_thread_id
        stop_thread_id = threading.get_ident()

    bot.event_saver = MagicMock()
    bot.event_saver.stop = record_thread_stop

    await bot.shutdown()

    assert stop_thread_id is not None
    assert stop_thread_id != main_thread_id, "EventSaver.stop executou dentro da MainThread do loop!"


@pytest.mark.asyncio
async def test_cancellation_during_event_saver_stop_propagates_and_waits_for_task():
    """3. Garante que cancelamento durante EventSaver.stop aguarda a conclusão da tarefa e propaga CancelledError."""
    bot = _make_dummy_bot()
    stop_started = threading.Event()
    stop_finished = threading.Event()

    def blocking_stop():
        stop_started.set()
        time.sleep(0.08)
        stop_finished.set()

    bot.event_saver = MagicMock()
    bot.event_saver.stop = blocking_stop

    shutdown_task = asyncio.create_task(bot.shutdown())

    while not stop_started.is_set():
        await asyncio.sleep(0.01)

    shutdown_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert stop_finished.is_set(), "Thread do EventSaver foi abandonada antes de concluir stop!"
    assert bot._shutdown_completed_event.is_set(), "Completion event não foi setado após término cooperativo sob cancelamento!"


@pytest.mark.asyncio
async def test_repeated_cancellation_during_stop_does_not_abandon_thread():
    """4. Garante que cancelamento repetido não faz o shutdown abandonar a task do EventSaver.stop."""
    bot = _make_dummy_bot()
    stop_started = threading.Event()
    stop_finished = threading.Event()

    def blocking_stop():
        stop_started.set()
        time.sleep(0.1)
        stop_finished.set()

    bot.event_saver = MagicMock()
    bot.event_saver.stop = blocking_stop

    shutdown_task = asyncio.create_task(bot.shutdown())

    while not stop_started.is_set():
        await asyncio.sleep(0.01)

    # Primeiro cancelamento
    shutdown_task.cancel()
    await asyncio.sleep(0.02)
    # Segundo cancelamento repetido durante a espera
    shutdown_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert stop_finished.is_set(), "Thread do EventSaver foi abandonada sob cancelamento repetido!"
    assert bot._shutdown_completed_event.is_set()


@pytest.mark.asyncio
async def test_stop_finishes_before_subsequent_cleanups_run():
    """5. Garante que etapas posteriores de cleanup não iniciam enquanto stop_task está em andamento."""
    bot = _make_dummy_bot()
    stop_active = threading.Event()
    stop_done = threading.Event()

    def slow_stop():
        stop_active.set()
        time.sleep(0.08)
        stop_active.clear()
        stop_done.set()

    bot.event_saver = MagicMock()
    bot.event_saver.stop = slow_stop

    orderbook_closed_while_stop_active = False

    async def mock_ob_close():
        nonlocal orderbook_closed_while_stop_active
        if stop_active.is_set():
            orderbook_closed_while_stop_active = True

    bot.orderbook_analyzer = MagicMock()
    bot.orderbook_analyzer.close = mock_ob_close

    await bot.shutdown()

    assert stop_done.is_set()
    assert orderbook_closed_while_stop_active is False, "OrderBookAnalyzer.close rodou prematuramente com stop_task ativo!"


@pytest.mark.asyncio
async def test_completion_event_not_set_while_stop_is_active():
    """6. Garante que _shutdown_completed_event não é sinalizado enquanto stop_task ainda está ativa."""
    bot = _make_dummy_bot()
    stop_started = threading.Event()
    release_stop = threading.Event()
    event_was_set_prematurely = False

    def blocking_stop():
        stop_started.set()
        release_stop.wait(timeout=2.0)

    bot.event_saver = MagicMock()
    bot.event_saver.stop = blocking_stop

    shutdown_task = asyncio.create_task(bot.shutdown())

    while not stop_started.is_set():
        await asyncio.sleep(0.01)

    if bot._shutdown_completed_event is not None and bot._shutdown_completed_event.is_set():
        event_was_set_prematurely = True

    release_stop.set()
    await shutdown_task

    assert event_was_set_prematurely is False, "Completion event foi sinalizado antes do término de stop_task!"
    assert bot._shutdown_completed_event.is_set() is True


@pytest.mark.asyncio
async def test_second_shutdown_waits_outside_lock_for_completion():
    """7. Garante que segunda chamada concorrente aguarda fora do lock sem duplicar stop."""
    bot = _make_dummy_bot()
    stop_call_count = 0

    def slow_stop():
        nonlocal stop_call_count
        stop_call_count += 1
        time.sleep(0.08)

    bot.event_saver = MagicMock()
    bot.event_saver.stop = slow_stop

    t1 = asyncio.create_task(bot.shutdown())
    await asyncio.sleep(0.02)
    # Lock deve estar liberado enquanto o shutdown progride
    assert bot._shutdown_async_lock.locked() is False, "Lock continuou retido durante o shutdown!"

    t2 = asyncio.create_task(bot.shutdown())
    await asyncio.gather(t1, t2)

    assert stop_call_count == 1, f"EventSaver.stop duplicado! count={stop_call_count}"
    assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_second_shutdown_after_cancellation_observes_real_completion():
    """8. Garante que segunda chamada após cancelamento observa a conclusão real da primeira execução."""
    bot = _make_dummy_bot()
    stop_started = threading.Event()
    stop_finished = threading.Event()

    def slow_stop():
        stop_started.set()
        time.sleep(0.08)
        stop_finished.set()

    bot.event_saver = MagicMock()
    bot.event_saver.stop = slow_stop

    t1 = asyncio.create_task(bot.shutdown())
    while not stop_started.is_set():
        await asyncio.sleep(0.01)

    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1

    assert stop_finished.is_set()

    # Segunda chamada pós-cancelamento observa conclusão e não duplica execução
    await bot.shutdown()
    assert bot._is_shutdown is True
    assert bot._shutdown_completed_event.is_set() is True


@pytest.mark.asyncio
async def test_internal_runtime_error_does_not_trigger_sync_fallback():
    """9. Garante que RuntimeError originado dentro de stop() não provoca segunda execução síncrona."""
    bot = _make_dummy_bot()
    stop_call_count = 0

    def failing_stop():
        nonlocal stop_call_count
        stop_call_count += 1
        raise RuntimeError("Internal SQLite corruption error")

    bot.event_saver = MagicMock()
    bot.event_saver.stop = failing_stop

    await bot.shutdown()

    assert stop_call_count == 1, f"Stop foi executado {stop_call_count} vezes (segunda chamada indevida)!"
    assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_unavailable_executor_does_not_trigger_sync_fallback():
    """10. Garante que executor indisponível não executa fallback síncrono no event loop."""
    bot = _make_dummy_bot()
    sync_called = False

    def sync_stop():
        nonlocal sync_called
        sync_called = True

    bot.event_saver = MagicMock()
    bot.event_saver.stop = sync_stop

    with patch("asyncio.to_thread", side_effect=RuntimeError("cannot schedule new futures after shutdown")):
        await bot.shutdown()

    assert sync_called is False, "Fallback síncrono indevido foi executado quando executor estava indisponível!"
    assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_exception_in_stop_task_is_observable(caplog):
    """11. Garante que exceções de stop_task não são engolidas silenciosamente e permanecem observáveis nos logs."""
    bot = _make_dummy_bot()

    def failing_stop():
        raise ValueError("Simulated unexpected stop crash")

    bot.event_saver = MagicMock()
    bot.event_saver.stop = failing_stop

    with caplog.at_level(logging.WARNING):
        await bot.shutdown()

    matching_logs = [record for record in caplog.records if "Falha ao parar EventSaver" in record.message]
    assert len(matching_logs) > 0, "Exceção em stop_task foi silenciada sem log observável!"


@pytest.mark.asyncio
async def test_preservation_of_g3a_lifecycle_states():
    """12. Valida que instância real do EventSaver atinge CLEAN_STOP e preserva WORKER_DIED sob falha prévia."""
    bot = _make_dummy_bot()
    real_saver = EventSaver(sound_alert=None, health_monitor=bot.health_monitor)
    bot.event_saver = real_saver

    await bot.shutdown()

    assert bot.event_saver.shutdown_status == "CLEAN_STOP"
    assert bot.event_saver._shutdown_incomplete is False
    assert bot.event_saver._worker_exited_cleanly is True

    # Cenário de falha prévia do worker: estado não deve ser mascarado como CLEAN_STOP
    bot2 = _make_dummy_bot()
    crashed_saver = EventSaver(sound_alert=None, health_monitor=bot2.health_monitor)

    def fatal_flush_batch():
        raise KeyboardInterrupt("Simulated fatal worker crash")

    crashed_saver._process_single_flush_batch = fatal_flush_batch
    crashed_saver._wake_flush_event.set()
    crashed_saver._flush_thread.join(timeout=2.0)
    assert not crashed_saver.worker_alive
    assert crashed_saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"

    bot2.event_saver = crashed_saver
    await bot2.shutdown()

    assert bot2.event_saver.shutdown_status == "WORKER_DIED_UNEXPECTEDLY"
    assert bot2.event_saver._shutdown_incomplete is True


@pytest.mark.asyncio
async def test_no_pending_orphan_tasks_after_cooperative_shutdown():
    """13. Garante que nenhuma task interna do bot permanece pendente ou vazando após encerramento cooperativo."""
    bot = _make_dummy_bot()
    loop = asyncio.get_running_loop()

    # Inicia tarefa periódica simulando inicialização do bot
    async def dummy_sync_loop():
        while not bot.should_stop:
            await asyncio.sleep(0.05)

    bot._periodic_sync_task = asyncio.create_task(dummy_sync_loop())

    await bot.shutdown()

    all_tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    # Todas as tasks pertencentes ao bot devem estar concluídas/canceladas
    bot_tasks = [t for t in all_tasks if not t.done()]
    assert len(bot_tasks) == 0, f"Tarefas órfãs pendentes encontradas: {bot_tasks}"
