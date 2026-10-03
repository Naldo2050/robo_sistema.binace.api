# tests/unit/test_robust_connection.py
# -*- coding: utf-8 -*-
"""
Testes herméticos para o gerenciador de conexão RobustConnectionManager (Achado #6).
Valida:
- Não reconectar em JSON inválido;
- Não bloquear mensagens válidas após inválidas;
- Tratamento de exceção no callback com caminho único de reconnect;
- Contagem unitária exata de falhas reais de conexão;
- Interrupção imediata do backoff durante shutdown/disconnect;
- Propagação correta de asyncio.CancelledError.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp
from aiohttp import WSMsgType

from market_orchestrator.connection.robust_connection import RobustConnectionManager


# ==============================================================================
# 1. test_invalid_json_does_not_reconnect
# ==============================================================================
@pytest.mark.asyncio
async def test_invalid_json_does_not_reconnect():
    manager = RobustConnectionManager(stream_url="wss://dummy/ws", symbol="BTCUSDT")
    with patch.object(manager, "_handle_reconnect", new_callable=AsyncMock) as mock_recon:
        invalid_msg = MagicMock(type=WSMsgType.TEXT, data="{invalid_json_string")
        await manager._handle_message(invalid_msg)

        assert mock_recon.call_count == 0, "_handle_reconnect não deve ser chamado para JSON inválido!"
        assert manager.reconnect_count == 0, "reconnect_count deve permanecer 0"
        assert manager.total_reconnects == 0, "total_reconnects deve permanecer 0"
        assert manager.total_messages_received == 1


# ==============================================================================
# 2. test_invalid_json_does_not_block_next_valid_message
# ==============================================================================
@pytest.mark.asyncio
async def test_invalid_json_does_not_block_next_valid_message():
    manager = RobustConnectionManager(stream_url="wss://dummy/ws", symbol="BTCUSDT")
    received_payloads = []

    def on_msg(ws, data):
        received_payloads.append(data)

    manager.set_callbacks(on_message=on_msg)

    with patch.object(manager, "_handle_reconnect", new_callable=AsyncMock) as mock_recon:
        invalid_msg = MagicMock(type=WSMsgType.TEXT, data="bad-json-payload-not-dict")
        valid_payload = '{"e": "depthUpdate", "s": "BTCUSDT", "b": [["50000", "1.0"]]}'
        valid_msg = MagicMock(type=WSMsgType.TEXT, data=valid_payload)

        await manager._handle_message(invalid_msg)
        await manager._handle_message(valid_msg)

        assert mock_recon.call_count == 0
        assert manager.total_messages_received == 2
        assert len(received_payloads) == 1
        assert received_payloads[0] == valid_payload


# ==============================================================================
# 3. test_callback_exception_causes_single_reconnect_path
# ==============================================================================
@pytest.mark.asyncio
async def test_callback_exception_causes_single_reconnect_path():
    manager = RobustConnectionManager(
        stream_url="wss://dummy/ws",
        symbol="BTCUSDT",
        initial_delay=0.01,
        max_reconnect_attempts=5,
    )

    def failing_callback(ws, data):
        raise ValueError("Simulated callback explosion")

    manager.set_callbacks(on_message=failing_callback)

    # Passo 1: Em _handle_message(), a exceção deve ser re-lançada SEM chamar _handle_reconnect
    valid_msg = MagicMock(type=WSMsgType.TEXT, data='{"e": "depthUpdate"}')
    with patch.object(manager, "_handle_reconnect", new_callable=AsyncMock) as mock_recon:
        with pytest.raises(ValueError, match="Simulated callback explosion"):
            await manager._handle_message(valid_msg)

        assert mock_recon.call_count == 0, "Exceção do callback não deve chamar _handle_reconnect dentro de _handle_message!"
        assert manager.total_reconnects == 0

    # Passo 2: No fluxo de connect(), a exceção escapa do loop do WS e atinge o handler externo exatamente UMA vez
    mock_ws = AsyncMock()

    async def async_msg_gen():
        yield valid_msg

    mock_ws.__aiter__.side_effect = lambda: async_msg_gen()
    mock_ws.closed = False

    mock_session = MagicMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_ws
    mock_ctx.__aexit__.return_value = None
    mock_session.ws_connect.return_value = mock_ctx

    async def stop_after_first_reconnect():
        manager.should_stop = True

    with patch("aiohttp.ClientSession") as mock_session_cls, \
         patch.object(manager, "_handle_reconnect", side_effect=stop_after_first_reconnect) as mock_recon:
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None

        await manager.connect()

        # Comprova que _handle_reconnect foi chamado exatamente UMA vez (sem double-reconnect)
        assert mock_recon.call_count == 1, f"_handle_reconnect chamado {mock_recon.call_count} vezes (esperado 1)"


# ==============================================================================
# 4. test_real_connection_failure_counts_once
# ==============================================================================
@pytest.mark.asyncio
async def test_real_connection_failure_counts_once():
    manager = RobustConnectionManager(
        stream_url="wss://dummy/ws",
        symbol="BTCUSDT",
        initial_delay=0.01,
        max_reconnect_attempts=5,
    )

    mock_session = MagicMock()
    mock_session.ws_connect.side_effect = aiohttp.ClientError("Simulated Network Drop")

    orig_handle_reconnect = manager._handle_reconnect

    async def tracked_handle_reconnect():
        await orig_handle_reconnect()
        manager.should_stop = True

    with patch("aiohttp.ClientSession") as mock_session_cls, \
         patch.object(manager, "_handle_reconnect", side_effect=tracked_handle_reconnect):
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None

        await manager.connect()

        assert manager.reconnect_count == 1
        assert manager.total_reconnects == 1


# ==============================================================================
# 5. test_shutdown_interrupts_reconnect_backoff
# ==============================================================================
@pytest.mark.asyncio
async def test_shutdown_interrupts_reconnect_backoff():
    # Delay longo de 60 segundos
    manager = RobustConnectionManager(
        stream_url="wss://dummy/ws",
        symbol="BTCUSDT",
        initial_delay=60.0,
    )

    start_t = time.monotonic()
    reconnect_task = asyncio.create_task(manager._handle_reconnect())

    # Aguarda a task iniciar e entrar no wait_for
    await asyncio.sleep(0.03)

    # Executa disconnect
    await manager.disconnect()

    # Deve concluir rapidamente sem esperar os 60 segundos
    await asyncio.wait_for(reconnect_task, timeout=1.0)

    elapsed = time.monotonic() - start_t
    assert elapsed < 0.5, f"Shutdown demorou {elapsed:.2f}s (esperado < 0.5s)"
    assert manager.should_stop is True


@pytest.mark.asyncio
async def test_should_stop_property_interrupts_backoff():
    manager = RobustConnectionManager(
        stream_url="wss://dummy/ws",
        symbol="BTCUSDT",
        initial_delay=60.0,
    )

    start_t = time.monotonic()
    reconnect_task = asyncio.create_task(manager._handle_reconnect())
    await asyncio.sleep(0.03)

    # Set direto da propriedade should_stop
    manager.should_stop = True

    await asyncio.wait_for(reconnect_task, timeout=1.0)
    elapsed = time.monotonic() - start_t
    assert elapsed < 0.5, f"should_stop direto demorou {elapsed:.2f}s"
    assert manager.should_stop is True


# ==============================================================================
# 6. test_cancelled_error_is_not_swallowed
# ==============================================================================
@pytest.mark.asyncio
async def test_cancelled_error_is_not_swallowed():
    manager = RobustConnectionManager(
        stream_url="wss://dummy/ws",
        symbol="BTCUSDT",
        initial_delay=60.0,
    )

    reconnect_task = asyncio.create_task(manager._handle_reconnect())
    await asyncio.sleep(0.03)

    reconnect_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await reconnect_task
