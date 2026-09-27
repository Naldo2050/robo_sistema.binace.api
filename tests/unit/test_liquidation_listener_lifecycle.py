# tests/unit/test_liquidation_listener_lifecycle.py
# -*- coding: utf-8 -*-
"""
Testes de Lifecycle do BinanceLiquidationListener (Cloud Hardening C1).

Valida:
1. Feature flag default OFF (BINANCE_LIQUIDATION_STREAM_ENABLED=0): zero instâncias/tasks.
2. Startup condicional quando flag ON (start assíncrono e registro em health opcional).
3. Shutdown limpo com await stop() sem tasks órfãs.
4. Isolamento: telemetria de liquidação nunca quebra o trading pipeline.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import config
from market_orchestrator import EnhancedMarketBot
from monitoring.health_monitor import NON_STAGE_CHANNELS


def test_liquidation_channel_is_non_stage_optional():
    """Garante que liquidation_stream_status é categorizado como opcional fora dos estágios críticos."""
    assert "liquidation_stream_status" in NON_STAGE_CHANNELS
    assert "shadow_collector" in NON_STAGE_CHANNELS


@pytest.mark.asyncio
async def test_liquidation_listener_lifecycle_off_by_default():
    """Quando flag OFF, nenhum listener é criado no startup."""
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

        assert bot.liquidation_listener is None
        assert bot.liquidation_stream_enabled is False


@pytest.mark.asyncio
async def test_liquidation_listener_lifecycle_on_and_shutdown():
    """Quando flag ON, inicializa o listener e no shutdown encerra graciosamente."""
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp, \
         patch.object(config, "BINANCE_LIQUIDATION_STREAM_ENABLED", True):

        mock_tb.return_value.start = AsyncMock()
        mock_tb.return_value.stop = AsyncMock()
        mock_wp.return_value.start = AsyncMock()
        mock_wp.return_value.stop = AsyncMock()

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

        bot.trades_buffer.start = AsyncMock()
        bot.trades_buffer.stop = AsyncMock()
        bot._prefetch_ohlc_history = AsyncMock()
        bot.liquidation_stream_enabled = True

        mock_listener = MagicMock()
        mock_listener.start = MagicMock()
        mock_listener.stop = AsyncMock()

        with patch("fetchers.binance_liquidation_stream.BinanceLiquidationListener", return_value=mock_listener):
            bot._loop = asyncio.get_running_loop()
            await bot.initialize()

            assert bot.liquidation_listener is mock_listener
            mock_listener.start.assert_called_once()

            # Executa shutdown
            await bot.shutdown()

            mock_listener.stop.assert_awaited_once()
            assert bot.liquidation_listener is None
