# tests/unit/test_bot_lifecycle_shutdown.py
# -*- coding: utf-8 -*-
"""
Testes de Lifecycle e Shutdown Cooperativo (Cloud Hardening C1).

Valida:
1. Idempotência estrita de EnhancedMarketBot.shutdown().
2. Encerramento gracioso em resposta a cancelamento/evento de shutdown.
3. Tratamento seguro de shutdown duplicado sem exceções.
"""

import asyncio
import signal
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from events import event_saver
from market_orchestrator import EnhancedMarketBot


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
    try:
        import common.runtime_lock
        monkeypatch.setattr(common.runtime_lock, "verify_startup_preflight", lambda *a, **kw: (True, "OK"))
        monkeypatch.setattr(common.runtime_lock.OperationalRuntimeLock, "acquire", lambda *a, **kw: True)
        monkeypatch.setattr(common.runtime_lock.OperationalRuntimeLock, "release", lambda *a, **kw: True)
    except Exception:
        pass
    try:
        import main
        monkeypatch.setattr(main.io, "TextIOWrapper", lambda *a, **kw: sys.stdout)
    except Exception:
        pass
    yield


@pytest.mark.asyncio
async def test_bot_shutdown_is_strictly_idempotent():
    """Garante que múltiplas chamadas consecutivas a shutdown() executam apenas uma vez."""
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

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

        assert bot._is_shutdown is False

        # Primeira execução
        await bot.shutdown()
        assert bot._is_shutdown is True
        assert bot.should_stop is True

        # Segunda execução: deve retornar imediatamente como no-op
        await bot.shutdown()
        assert bot._is_shutdown is True

        # Terceira execução concorrente
        await asyncio.gather(bot.shutdown(), bot.shutdown())
        assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_cooperative_shutdown_event_triggers_cleanly():
    """Simula um shutdown_event ativado por sinal e valida a finalização do bot."""
    shutdown_event = asyncio.Event()

    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

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

        async def fake_run():
            while not bot.should_stop:
                await asyncio.sleep(0.01)

        bot.run = fake_run

        bot_task = asyncio.create_task(bot.run())
        waiter = asyncio.create_task(shutdown_event.wait())

        # Simula envio de sinal após 30ms
        async def send_signal_soon():
            await asyncio.sleep(0.03)
            shutdown_event.set()

        asyncio.create_task(send_signal_soon())

        done, pending = await asyncio.wait([bot_task, waiter], return_when=asyncio.FIRST_COMPLETED)

        assert shutdown_event.is_set()
        assert waiter in done

        # Executa shutdown do bot
        await bot.shutdown()
        await bot_task

        assert bot._is_shutdown is True
        assert bot.should_stop is True


# ==============================================================================
# NOVOS TESTES - FASE 5.2: Propagacao de Falhas Fatais e Exit Code (#4)
# ==============================================================================

@pytest.mark.asyncio
async def test_bot_run_propagates_initialize_failure():
    """Garante que falha em initialize() propaga para o caller e executa shutdown limpo."""
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

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

        bot.initialize = AsyncMock(side_effect=RuntimeError("Simulated init failure"))
        bot.context_collector = MagicMock()
        bot.connection_manager.disconnect = AsyncMock()

        with pytest.raises(RuntimeError, match="Simulated init failure"):
            await bot.run()

        assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_bot_run_propagates_connection_failure():
    """Garante que falha em connection_manager.connect() propaga para o caller."""
    from common.exceptions import APIConnectionError

    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

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

        bot.initialize = AsyncMock(return_value=None)
        bot.context_collector = MagicMock()
        bot.connection_manager = MagicMock()
        bot.connection_manager.connect = AsyncMock(
            side_effect=APIConnectionError("Simulated terminal WS failure")
        )
        bot.connection_manager.disconnect = AsyncMock()

        with pytest.raises(APIConnectionError, match="Simulated terminal WS failure"):
            await bot.run()

        assert bot._is_shutdown is True
        assert bot.connection_manager.disconnect.await_count >= 1


@pytest.mark.asyncio
async def test_bot_run_fatal_failure_still_shutdowns():
    """Garante que qualquer erro fatal em run() ainda executa shutdown de recursos."""
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

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

        bot.initialize = AsyncMock(side_effect=ValueError("Corrupted internal state"))
        bot.context_collector = MagicMock()
        bot.connection_manager.disconnect = AsyncMock()

        with pytest.raises(ValueError, match="Corrupted internal state"):
            await bot.run()

        assert bot._is_shutdown is True
        assert bot.should_stop is True

        # Idempotencia estrita preservada
        await bot.shutdown()
        assert bot._is_shutdown is True


@pytest.mark.asyncio
async def test_main_returns_1_on_fatal_bot_error():
    """Garante que main() retorna exit code 1 quando bot_task falha com erro fatal."""
    import main

    mock_bot = MagicMock()
    mock_bot.run = AsyncMock(side_effect=RuntimeError("Fatal bot crash"))
    mock_bot.shutdown = AsyncMock()

    with patch("main.EnhancedMarketBot", return_value=mock_bot), \
         patch("monitoring.pipeline_health.serve_metrics_and_health"), \
         patch("monitoring.pipeline_health.start_pipeline_health"), \
         patch("monitoring.pipeline_health.attach_ws_connected_provider"), \
         patch("monitoring.pipeline_health.attach_message_age_provider"), \
         patch("monitoring.pipeline_health.attach_window_age_provider"), \
         patch("fetchers.macro_update_service.start_macro_service", new_callable=AsyncMock), \
         patch("fetchers.macro_update_service.stop_macro_service", new_callable=AsyncMock):

        exit_code = await main.main()
        assert exit_code == 1, f"Esperado exit code 1, obteve {exit_code}"
        assert mock_bot.shutdown.await_count >= 1


@pytest.mark.asyncio
async def test_main_returns_0_on_graceful_shutdown():
    """Garante que main() retorna exit code 0 em shutdown cooperativo normal."""
    import main

    mock_bot = MagicMock()
    stop_event = asyncio.Event()

    async def fake_run():
        await stop_event.wait()

    mock_bot.run = fake_run
    mock_bot.shutdown = AsyncMock(side_effect=lambda: stop_event.set())

    with patch("main.EnhancedMarketBot", return_value=mock_bot), \
         patch("monitoring.pipeline_health.serve_metrics_and_health"), \
         patch("monitoring.pipeline_health.start_pipeline_health"), \
         patch("monitoring.pipeline_health.attach_ws_connected_provider"), \
         patch("monitoring.pipeline_health.attach_message_age_provider"), \
         patch("monitoring.pipeline_health.attach_window_age_provider"), \
         patch("fetchers.macro_update_service.start_macro_service", new_callable=AsyncMock), \
         patch("fetchers.macro_update_service.stop_macro_service", new_callable=AsyncMock), \
         patch("sys.argv", ["main.py", "--duration-seconds", "0"]):

        async def trigger_shutdown():
            await asyncio.sleep(0.05)
            stop_event.set()

        asyncio.create_task(trigger_shutdown())

        exit_code = await main.main()
        assert exit_code == 0, f"Esperado exit code 0, obteve {exit_code}"
