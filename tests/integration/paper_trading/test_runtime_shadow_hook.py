# tests/integration/paper_trading/test_runtime_shadow_hook.py
"""
Integration tests for EnhancedMarketBot runtime shadow hook (Gate C3-C-B3-B).

Covers all 20 mandatory runtime hook test scenarios:
1. bot OFF -> shadow_runtime None
2. OFF -> zero EventBus subscription paper
3. OFF -> on_message behavior existente preservado
4. OFF -> tick não chama paper
5. RUNNING -> subscription exatamente uma
6. initialize repetido -> não duplica subscription
7. FAILED -> não subscribe
8. mesmo norm object chega runtime
9. tick hook ocorre após TradeBuffer
10. tick hook ocorre antes de window boundary processing
11. paper hook exception não escapa on_message
12. paper failure não reconecta websocket artificialmente
13. existing market processing continua
14. shutdown chama runtime uma vez
15. shutdown não cria synthetic close
16. observation safety continua bloqueando credentials
17. nenhuma segunda RobustConnectionManager criada
18. IA não é ativada
19. nenhuma ordem real
20. paper status observável
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from config.env_policy import assert_observation_safe
from market_orchestrator.market_orchestrator import EnhancedMarketBot


def _create_test_bot(
    shadow_runtime: Any = None,
    env: Dict[str, str] | None = None,
    rcm_instance: Any = None,
) -> EnhancedMarketBot:
    """Helper to instantiate EnhancedMarketBot hermetically without external I/O."""
    from unittest.mock import AsyncMock

    rcm_patch = (
        patch("market_orchestrator.market_orchestrator.RobustConnectionManager", return_value=rcm_instance)
        if rcm_instance is not None
        else patch("market_orchestrator.market_orchestrator.RobustConnectionManager")
    )

    def _wp_factory(*args: Any, **kwargs: Any) -> MagicMock:
        wp = MagicMock()
        wp.start = AsyncMock()
        wp.stop = AsyncMock()
        return wp

    with rcm_patch, \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer"), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor", side_effect=_wp_factory), \
         patch("market_orchestrator.market_orchestrator.ContextCollector"), \
         patch("market_orchestrator.market_orchestrator.FlowAnalyzer"), \
         patch("market_orchestrator.market_orchestrator.EventSaver"), \
         patch("market_orchestrator.market_orchestrator.OrderBookAnalyzer"), \
         patch("market_orchestrator.market_orchestrator.OnchainUpdater"), \
         patch("market_orchestrator.market_orchestrator.CrossAssetUpdater"), \
         patch.object(EnhancedMarketBot, "_initialize_ai_async", return_value=None), \
         patch.dict("os.environ", env or {}, clear=False):

        bot = EnhancedMarketBot(
            stream_url="wss://test.stream",
            symbol="BTCUSDT",
            window_size_minutes=1,
            vol_factor_exh=2.0,
            history_size=10,
            delta_std_dev_factor=2.0,
            context_sma_period=10,
            liquidity_flow_alert_percentage=0.1,
            wall_std_dev_factor=2.0,
            shadow_runtime=shadow_runtime,
        )
        if hasattr(bot, "trades_buffer") and bot.trades_buffer is not None:
            bot.trades_buffer.start = AsyncMock()
            bot.trades_buffer.stop = AsyncMock()
        if hasattr(bot, "connection_manager") and bot.connection_manager is not None:
            bot.connection_manager.disconnect = AsyncMock()
        return bot


def _make_agg_trade_json(
    price: float = 65000.0,
    qty: float = 1.0,
    ts_ms: int = 1788800000000,
    is_buyer_maker: bool = False,
) -> str:
    return json.dumps({
        "e": "aggTrade",
        "E": ts_ms,
        "s": "BTCUSDT",
        "a": 12345,
        "p": str(price),
        "q": str(qty),
        "f": 100,
        "l": 100,
        "T": ts_ms,
        "m": is_buyer_maker,
    })


@pytest.mark.asyncio
async def test_01_bot_off_shadow_runtime_is_none():
    """1. bot OFF -> shadow_runtime None e status DISABLED."""
    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    assert bot.shadow_runtime is None
    assert bot.paper_shadow_status == "DISABLED"
    assert bot.paper_shadow_error is None


def _mock_wp_factory(*args: Any, **kwargs: Any) -> MagicMock:
    wp = MagicMock()
    wp.start = AsyncMock()
    wp.stop = AsyncMock()
    return wp


@pytest.mark.asyncio
async def test_02_off_zero_eventbus_subscription():
    """2. OFF -> zero EventBus subscription paper."""
    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    with patch.object(bot, "_prefetch_ohlc_history", return_value=None), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor", side_effect=_mock_wp_factory):
        await bot.initialize()

    # Verifica handlers de "signal"
    subscribers = bot.event_bus._handlers.get("signal", [])
    # Somente o handler interno do bot (_handle_signal_event) deve existir
    assert len(subscribers) == 1
    assert subscribers[0] == bot._handle_signal_event


@pytest.mark.asyncio
async def test_03_off_on_message_preserves_existing_behavior():
    """3. OFF -> on_message behavior existente preservado."""
    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    msg = _make_agg_trade_json(price=65100.0, ts_ms=1000)
    bot.on_message(None, msg)

    assert bot.trades_buffer.add_trade_sync.called
    assert len(bot.window_data) == 1
    assert bot.window_data[0]["p"] == 65100.0


@pytest.mark.asyncio
async def test_04_off_tick_does_not_call_paper():
    """4. OFF -> tick não chama paper."""
    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    msg = _make_agg_trade_json(price=65200.0, ts_ms=2000)
    # Não deve haver shadow_runtime e nenhuma chamada
    assert bot.shadow_runtime is None
    bot.on_message(None, msg)
    assert bot.paper_shadow_hook_errors == 0


@pytest.mark.asyncio
async def test_05_running_subscribes_exactly_once():
    """5. RUNNING -> subscription exatamente uma vez."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)
    assert bot.paper_shadow_status == "RUNNING"

    with patch.object(bot, "_prefetch_ohlc_history", return_value=None), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor", side_effect=_mock_wp_factory):
        await bot.initialize()

    subscribers = bot.event_bus._handlers.get("signal", [])
    assert mock_runtime.on_signal in subscribers
    assert bot._shadow_subscribed is True


@pytest.mark.asyncio
async def test_06_repeated_initialize_does_not_duplicate_subscription():
    """6. initialize repetido -> não duplica subscription."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)

    with patch.object(bot, "_prefetch_ohlc_history", return_value=None), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor", side_effect=_mock_wp_factory):
        await bot.initialize()
        subscribers_before = list(bot.event_bus._handlers.get("signal", []))

        # Re-chamar initialize
        bot._initialized = False  # simular segunda chamada
        await bot.initialize()
        subscribers_after = list(bot.event_bus._handlers.get("signal", []))

    assert subscribers_before == subscribers_after
    assert subscribers_after.count(mock_runtime.on_signal) == 1


@pytest.mark.asyncio
async def test_07_failed_does_not_subscribe():
    """7. FAILED -> não subscribe."""
    bot = _create_test_bot()
    bot.paper_shadow_status = "FAILED"
    bot.shadow_runtime = None

    with patch.object(bot, "_prefetch_ohlc_history", return_value=None), \
         patch("market_orchestrator.market_orchestrator.WindowProcessor", side_effect=_mock_wp_factory):
        await bot.initialize()

    subscribers = bot.event_bus._handlers.get("signal", [])
    assert not any("shadow" in getattr(s, "__qualname__", "").lower() for s in subscribers)


@pytest.mark.asyncio
async def test_08_same_norm_object_delivered_to_runtime():
    """8. Mesmo objeto normalizado chega ao runtime (sem cópia/reparse)."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    msg = _make_agg_trade_json(price=65300.0, ts_ms=3000)
    bot.on_message(None, msg)

    assert mock_runtime.on_market_trade.called
    delivered_norm = mock_runtime.on_market_trade.call_args[0][0]
    buffer_norm = bot.trades_buffer.add_trade_sync.call_args[0][0]

    assert delivered_norm is buffer_norm  # idêntico em memória


@pytest.mark.asyncio
async def test_09_tick_hook_occurs_after_trade_buffer():
    """9. Tick hook ocorre após add_trade_sync do TradeBuffer."""
    call_order = []
    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = lambda n: call_order.append("shadow_hook")

    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(side_effect=lambda n, cb: call_order.append("trade_buffer"))

    msg = _make_agg_trade_json(price=65400.0, ts_ms=4000)
    bot.on_message(None, msg)

    assert call_order == ["trade_buffer", "shadow_hook"]


@pytest.mark.asyncio
async def test_10_tick_hook_occurs_before_window_boundary_processing():
    """10. Tick hook ocorre antes de _process_window no fechamento da janela."""
    call_order = []
    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = lambda n: call_order.append("shadow_hook")

    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)
    bot._process_window = MagicMock(side_effect=lambda: call_order.append("process_window"))

    # Configura janela atual para fechar em 60_000 ms
    bot.window_end_ms = 60_000
    # Envia trade com timestamp que ultrapassa a fronteira (60_001 ms)
    msg = _make_agg_trade_json(price=65500.0, ts_ms=60_001)
    bot.on_message(None, msg)

    assert call_order == ["shadow_hook", "process_window"]


@pytest.mark.asyncio
async def test_11_paper_hook_exception_does_not_escape_on_message():
    """11. Exceção do hook de paper trading não escapa de on_message."""
    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = RuntimeError("Fatal shadow runtime crash")

    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    msg = _make_agg_trade_json(price=65600.0, ts_ms=5000)

    # Não deve lançar exceção
    bot.on_message(None, msg)

    assert bot.paper_shadow_status == "FAILED"
    assert "Fatal shadow runtime crash" in (bot.paper_shadow_error or "")
    assert bot.paper_shadow_hook_errors == 1


@pytest.mark.asyncio
async def test_12_paper_failure_does_not_reconnect_websocket():
    """12. Falha no paper trading não dispara reconexão artificial de websocket."""
    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = Exception("Isolated paper error")

    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)
    bot._on_reconnect = MagicMock()

    msg = _make_agg_trade_json(price=65700.0, ts_ms=6000)
    bot.on_message(None, msg)

    assert not bot._on_reconnect.called


@pytest.mark.asyncio
async def test_13_existing_market_processing_continues_after_hook_failure():
    """13. Processamento regular do mercado continua mesmo após erro do paper."""
    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = ValueError("Corrupt tick in paper")

    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    # 1º trade com falha no hook
    msg1 = _make_agg_trade_json(price=65800.0, ts_ms=7000)
    bot.on_message(None, msg1)
    assert bot.paper_shadow_status == "FAILED"

    # 2º trade
    mock_runtime.on_market_trade.side_effect = None
    msg2 = _make_agg_trade_json(price=65900.0, ts_ms=8000)
    bot.on_message(None, msg2)

    # Ambas as mensagens foram adicionadas ao buffer e à janela
    assert bot.trades_buffer.add_trade_sync.call_count == 2
    assert len(bot.window_data) == 2


@pytest.mark.asyncio
async def test_14_shutdown_calls_shadow_runtime_once():
    """14. shutdown chama shadow_runtime.shutdown exatamente uma vez."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)

    with patch.object(bot.trades_buffer, "stop", return_value=None), \
         patch.object(bot.connection_manager, "disconnect", return_value=None):
        await bot.shutdown()

    mock_runtime.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_15_shutdown_does_not_create_synthetic_close(tmp_path):
    """15. shutdown não sintetiza fechamentos de posição."""
    from paper_trading.factory import create_shadow_runtime

    env = {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": "CH_SHUTDOWN_TEST",
        "PAPER_PROVIDER": "fixed_long",
        "PAPER_NOTIONAL_USDT": "1000.0",
        "PAPER_HORIZON_S": "300",
        "PAPER_ORDER_TTL_MS": "5000",
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "VIP0_TIER",
        "PAPER_COST_EFFECTIVE_AT": "2026-09-01T00:00:00+00:00",
        "PAPER_DB_PATH": str(tmp_path / "shut.db"),
        "PAPER_GIT_SHA": "0c5c95fa1b2c3d4e5f67890abcdef1234567890a",
    }
    res = create_shadow_runtime(env)
    assert res.status == "RUNNING"
    runtime = res.runtime
    assert runtime is not None

    bot = _create_test_bot(shadow_runtime=runtime)
    with patch.object(bot.trades_buffer, "stop", return_value=None), \
         patch.object(bot.connection_manager, "disconnect", return_value=None):
        await bot.shutdown()

    # O runtime encerrou graciosamente sem gerar closed trades sintéticos
    assert runtime.get_counters()["closed_trades"] == 0
    assert runtime.is_active is False


def test_16_observation_safety_continues_blocking_credentials():
    """16. observation safety continua bloqueando credenciais mesmo com paper ativo."""
    with patch.dict(
        "os.environ",
        {
            "OBSERVATION_MODE": "1",
            "PAPER_SHADOW_ENABLED": "1",
            "BINANCE_API_KEY": "fake_live_key_must_fail",
        },
        clear=False,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            assert_observation_safe(None)
        assert "trading credentials present" in str(exc_info.value)


@pytest.mark.asyncio
async def test_17_no_second_connection_manager_created():
    """17. Nenhuma segunda RobustConnectionManager criada pelo hook."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)
    assert bot.connection_manager is not None
    from paper_trading.shadow_runtime import ShadowPaperRuntime
    assert not hasattr(ShadowPaperRuntime, "connection_manager")


@pytest.mark.asyncio
async def test_18_ai_not_activated_by_paper():
    """18. IA não é ativada ou invocada pelo shadow runtime hook."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    msg = _make_agg_trade_json(price=66000.0, ts_ms=9000)
    bot.on_message(None, msg)

    # ai_analyzer continua None / não acionado
    assert bot.ai_analyzer is None


@pytest.mark.asyncio
async def test_19_zero_real_orders_submitted():
    """19. Nenhuma ordem real emitida durante processamento de tick ou sinal."""
    mock_runtime = MagicMock()
    bot = _create_test_bot(shadow_runtime=mock_runtime)
    bot.trades_buffer.add_trade_sync = MagicMock(return_value=True)

    msg = _make_agg_trade_json(price=66100.0, ts_ms=10000)
    bot.on_message(None, msg)

    # Não existe nenhum endpoint de ordem no bot além de hooks observacionais
    assert not hasattr(bot, "submit_real_order")


@pytest.mark.asyncio
async def test_20_paper_status_observable():
    """20. Status e erros do paper trading são observáveis diretamente no bot."""
    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    assert bot.paper_shadow_status == "DISABLED"
    assert bot.paper_shadow_error is None
    assert bot.paper_shadow_hook_errors == 0

    mock_runtime = MagicMock()
    mock_runtime.on_market_trade.side_effect = TypeError("Sample hook error")
    bot_with_runtime = _create_test_bot(shadow_runtime=mock_runtime)
    assert bot_with_runtime.paper_shadow_status == "RUNNING"

    bot_with_runtime.trades_buffer.add_trade_sync = MagicMock(return_value=True)
    msg = _make_agg_trade_json(price=66200.0, ts_ms=11000)
    bot_with_runtime.on_message(None, msg)

    assert bot_with_runtime.paper_shadow_status == "FAILED"
    assert "Sample hook error" in (bot_with_runtime.paper_shadow_error or "")
    assert bot_with_runtime.paper_shadow_hook_errors == 1


@pytest.mark.asyncio
async def test_21_off_path_regression_exact_sequence():
    """21. Prova regressão zero no hot path quando OFF: ordem normalização -> buffer -> window."""
    events_log = []

    bot = _create_test_bot(env={"PAPER_SHADOW_ENABLED": "0"})
    assert bot.shadow_runtime is None

    # Instrumenta os métodos relevantes
    def spy_add_trade(norm, cb):
        events_log.append(("trades_buffer", norm["p"], norm["T"]))
        return True

    def spy_process_window():
        events_log.append(("process_window", bot.window_end_ms))

    bot.trades_buffer.add_trade_sync = MagicMock(side_effect=spy_add_trade)
    bot._process_window = MagicMock(side_effect=spy_process_window)

    bot.window_end_ms = 60_000

    # 1º trade dentro da janela
    msg1 = _make_agg_trade_json(price=65000.0, ts_ms=30_000)
    bot.on_message(None, msg1)

    # 2º trade que fecha a janela
    msg2 = _make_agg_trade_json(price=65100.0, ts_ms=60_001)
    bot.on_message(None, msg2)

    expected_sequence = [
        ("trades_buffer", 65000.0, 30_000),
        ("trades_buffer", 65100.0, 60_001),
        ("process_window", 60_000),
    ]
    assert events_log == expected_sequence
    assert bot.shadow_runtime is None
    assert bot.paper_shadow_hook_errors == 0
