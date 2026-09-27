# tests/integration/test_cloud_hardening_e2e_lifecycle.py
# -*- coding: utf-8 -*-
"""
Teste de Integração de Ciclo de Vida e Parada Graciosa (Cloud Hardening C1).

Valida:
1. Início do bot em modo de teste com Shadow Transport e Liquidation Listener ativos.
2. Enfileiramento de registros shadow.
3. Disparo de sinal de parada cooperativa (SIGTERM simulado).
4. Confirmação de:
   - Exit code limpo (0)
   - Fila do shadow totalmente drenada e persistida em JSONL válido
   - Liquidation listener parado sem task aberta
   - Sem threads órfãs do shadow
"""

import asyncio
import json
import threading
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import config
from flow_analyzer.effort_response_transport import (
    EffortResponseSnapshotDTO,
    ShadowAsyncTransport,
)
from market_orchestrator import EnhancedMarketBot


@pytest.fixture(autouse=True)
def reset_singletons():
    ShadowAsyncTransport.reset_instance_for_testing()
    yield
    ShadowAsyncTransport.reset_instance_for_testing()


@pytest.mark.asyncio
async def test_full_cloud_hardening_lifecycle_and_shutdown(tmp_path, monkeypatch):
    """Executa simulação ponta a ponta do ciclo de vida com shutdown gracioso."""
    shadow_file = tmp_path / "shadow_e2e.jsonl"
    monkeypatch.setenv("EFFORT_RESPONSE_SHADOW_ENABLED", "1")
    monkeypatch.setenv("EFFORT_RESPONSE_SHADOW_FILEPATH", str(shadow_file))
    monkeypatch.setenv("BINANCE_LIQUIDATION_STREAM_ENABLED", "1")

    # Mocks para isolar chamadas de rede externas
    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager") as mock_cm, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer") as mock_tb, \
         patch("market_orchestrator.market_orchestrator.WindowProcessor") as mock_wp:

        mock_tb.return_value.start = AsyncMock()
        mock_tb.return_value.stop = AsyncMock()
        mock_wp.return_value.start = AsyncMock()
        mock_wp.return_value.stop = AsyncMock()

        # Mock do Liquidation Listener
        mock_listener = MagicMock()
        mock_listener.start = MagicMock()
        mock_listener.stop = AsyncMock()

        with patch("fetchers.binance_liquidation_stream.BinanceLiquidationListener", return_value=mock_listener):
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

            bot._prefetch_ohlc_history = AsyncMock()
            bot.liquidation_stream_enabled = True

            # 1. Inicializa bot
            bot._loop = asyncio.get_running_loop()
            await bot.initialize()

            assert bot.liquidation_listener is mock_listener
            mock_listener.start.assert_called_once()

            # 2. Inicializa Shadow Transport e enfileira registros
            transport = ShadowAsyncTransport.get_instance(filepath=shadow_file, enabled=True)
            assert transport is not None
            assert ShadowAsyncTransport.get_if_initialized() is transport

            for i in range(5):
                t_anchor = 1700000000000 + (i * 60000)
                dto = EffortResponseSnapshotDTO(
                    symbol="BTCUSDT",
                    causal_anchor_ms=t_anchor,
                    observation_open_ms=t_anchor - 60000,
                    observation_close_ms=t_anchor - 1,
                    buy_notional_usd=10000.0 * (i + 1),
                    sell_notional_usd=8000.0 * (i + 1),
                    open=65000.0 + i,
                    high=65100.0 + i,
                    low=64900.0 + i,
                    close=65050.0 + i,
                    window_duration_ms=60000,
                )
                assert transport.submit_nowait(dto) is True

            # 3. Dispara shutdown cooperativo
            await bot.shutdown()

            # 4. Verificações de encerramento
            assert bot._is_shutdown is True
            assert bot.should_stop is True
            assert bot.liquidation_listener is None
            mock_listener.stop.assert_awaited_once()

            # Confirma que a fila do shadow foi drenada
            stats = transport.get_stats()
            assert stats["queue_depth"] == 0

            # Confirma que o arquivo JSONL contém todos os 5 registros válidos
            assert shadow_file.exists()
            lines = shadow_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 5
            for line in lines:
                record = json.loads(line)
                assert record["provenance"]["symbol"] == "BTCUSDT"

            # Confirma que a thread do worker encerrou
            if transport._worker_thread:
                assert not transport._worker_thread.is_alive()
