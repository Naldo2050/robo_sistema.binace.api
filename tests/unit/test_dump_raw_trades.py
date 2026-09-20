# tests/unit/test_dump_raw_trades.py
import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from market_orchestrator import EnhancedMarketBot
from scripts.diagnostics.validate_whale_threshold import analyze_whale_trades


def test_enhanced_market_bot_raw_trades_dump(tmp_path):
    dump_file = tmp_path / "test_trades.jsonl"

    with patch("market_orchestrator.market_orchestrator.RobustConnectionManager"), \
         patch("market_orchestrator.market_orchestrator.AsyncTradeBuffer"), \
         patch("market_orchestrator.market_orchestrator.ContextCollector"), \
         patch("market_orchestrator.market_orchestrator.FlowAnalyzer"), \
         patch("market_orchestrator.market_orchestrator.EventSaver"), \
         patch("market_orchestrator.market_orchestrator.OrderBookAnalyzer"), \
         patch.object(EnhancedMarketBot, "_initialize_ai_async", return_value=None):
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
            dump_raw_trades=str(dump_file),
        )

        assert bot.dump_raw_trades_path == dump_file
        assert bot._raw_trades_file is not None

        # Simular chegada de mensagem WebSocket (@aggTrade)
        sample_msg = {
            "e": "aggTrade",
            "E": 1788800000000,
            "s": "BTCUSDT",
            "a": 123456,
            "p": "65000.0",
            "q": "2.5",
            "f": 100,
            "l": 101,
            "T": 1788800000000,
            "m": False
        }
        bot.on_message(None, json.dumps(sample_msg))

        # Simular segundo trade
        sample_msg2 = {
            "e": "aggTrade",
            "E": 1788800000100,
            "s": "BTCUSDT",
            "a": 123457,
            "p": "65001.0",
            "q": "0.1",
            "f": 102,
            "l": 103,
            "T": 1788800000100,
            "m": True
        }
        bot.on_message(None, json.dumps(sample_msg2))

        bot._raw_trades_file.flush()

        # Verificar conteúdo escrito
        assert dump_file.exists()
        lines = dump_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        t1 = json.loads(lines[0])
        assert t1["trade_id"] == 123456
        assert t1["quantity"] == 2.5
        assert t1["price"] == 65000.0

        t2 = json.loads(lines[1])
        assert t2["trade_id"] == 123457
        assert t2["quantity"] == 0.1

        # Fechamento e shutdown do bot
        try:
            import asyncio
            asyncio.run(bot.shutdown())
        except Exception:
            if hasattr(bot, "_raw_trades_file") and bot._raw_trades_file:
                try:
                    bot._raw_trades_file.close()
                except Exception:
                    pass

        # Executar validação de whale sobre o arquivo gerado
        ret = analyze_whale_trades(
            dump_path=str(dump_file),
            db_path="dados/trading_bot.db",
            whale_threshold=2.0
        )
        assert ret == 0
