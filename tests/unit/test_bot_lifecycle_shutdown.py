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
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from market_orchestrator import EnhancedMarketBot


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
