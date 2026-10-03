# tests/unit/test_ingestion_counters.py
# -*- coding: utf-8 -*-
"""
Testes de regressão para bug #5: contadores de ingestão não inicializados.

Verifica que EnhancedMarketBot.on_message() não gera AttributeError
quando recebe JSON inválido, aggTrade com campos ausentes (p/q/T),
ou trades com tipos inválidos.

Rede: NENHUMA.  Todos os componentes pesados são mockados.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from market_orchestrator.market_orchestrator import EnhancedMarketBot


# ---------------------------------------------------------------------------
# Helper: cria um EnhancedMarketBot hermético (sem I/O, sem rede)
# ---------------------------------------------------------------------------

_PATCHES = [
    "market_orchestrator.market_orchestrator.RobustConnectionManager",
    "market_orchestrator.market_orchestrator.AsyncTradeBuffer",
    "market_orchestrator.market_orchestrator.ContextCollector",
    "market_orchestrator.market_orchestrator.FlowAnalyzer",
    "market_orchestrator.market_orchestrator.EventSaver",
    "market_orchestrator.market_orchestrator.OrderBookAnalyzer",
]


@pytest.fixture()
def bot():
    """EnhancedMarketBot hermético — todos os componentes de I/O mockados."""
    patches = [patch(p) for p in _PATCHES]
    patches.append(
        patch.object(EnhancedMarketBot, "_initialize_ai_async", return_value=None)
    )
    for p in patches:
        p.start()
    try:
        b = EnhancedMarketBot(
            stream_url="wss://test.stream/ws/btcusdt@aggTrade",
            symbol="BTCUSDT",
            window_size_minutes=1,
            vol_factor_exh=2.0,
            history_size=10,
            delta_std_dev_factor=2.0,
            context_sma_period=10,
            liquidity_flow_alert_percentage=0.1,
            wall_std_dev_factor=2.0,
        )
        yield b
    finally:
        for p in patches:
            p.stop()


# ---------------------------------------------------------------------------
# Mensagens de referência (aggTrade Binance USD-M Futures)
# ---------------------------------------------------------------------------

_VALID_AGGTRADE = {
    "e": "aggTrade",
    "E": 1788800000000,
    "s": "BTCUSDT",
    "a": 999001,
    "p": "65000.00",
    "q": "1.500",
    "f": 100,
    "l": 102,
    "T": 1788800000000,
    "m": False,
}


# ===================================================================
# 1. JSON inválido — não deve gerar AttributeError
# ===================================================================


class TestInvalidJson:
    def test_no_attribute_error(self, bot):
        """on_message com JSON malformado não gera AttributeError."""
        bot.on_message(None, "<<<NOT-JSON>>>")
        # Se chegou aqui sem exceção, o bug #5 está corrigido.

    def test_counter_increments(self, bot):
        """_invalid_json_count incrementa a cada JSON inválido."""
        assert bot._invalid_json_count == 0
        bot.on_message(None, "{broken")
        assert bot._invalid_json_count == 1
        bot.on_message(None, "<<<>>>")
        assert bot._invalid_json_count == 2

    def test_log_step_is_positive_int(self, bot):
        """_invalid_json_log_step inicializado como int positivo."""
        assert isinstance(bot._invalid_json_log_step, int)
        assert bot._invalid_json_log_step > 0


# ===================================================================
# 2. aggTrade sem campo 'p'
# ===================================================================


class TestMissingFieldP:
    def test_no_crash(self, bot):
        """Mensagem sem 'p' não crashar."""
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "p"}
        bot.on_message(None, json.dumps(msg))

    def test_counter_increments(self, bot):
        """_missing_field_counts['p'] incrementa."""
        assert bot._missing_field_counts["p"] == 0
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "p"}
        bot.on_message(None, json.dumps(msg))
        assert bot._missing_field_counts["p"] == 1


# ===================================================================
# 3. aggTrade sem campo 'q'
# ===================================================================


class TestMissingFieldQ:
    def test_no_crash(self, bot):
        """Mensagem sem 'q' não crashar."""
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "q"}
        bot.on_message(None, json.dumps(msg))

    def test_counter_increments(self, bot):
        """_missing_field_counts['q'] incrementa."""
        assert bot._missing_field_counts["q"] == 0
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "q"}
        bot.on_message(None, json.dumps(msg))
        assert bot._missing_field_counts["q"] == 1


# ===================================================================
# 4. aggTrade sem campo 'T'
# ===================================================================


class TestMissingFieldT:
    def test_no_crash(self, bot):
        """Mensagem sem 'T' não crashar."""
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "T"}
        bot.on_message(None, json.dumps(msg))

    def test_counter_increments(self, bot):
        """_missing_field_counts['T'] incrementa."""
        assert bot._missing_field_counts["T"] == 0
        msg = {k: v for k, v in _VALID_AGGTRADE.items() if k != "T"}
        bot.on_message(None, json.dumps(msg))
        assert bot._missing_field_counts["T"] == 1


# ===================================================================
# 5. Mensagem válida — correção não altera processamento normal
# ===================================================================


class TestValidMessage:
    def test_counters_stay_zero(self, bot):
        """Mensagem válida não incrementa contadores de erro."""
        bot.on_message(None, json.dumps(_VALID_AGGTRADE))
        assert bot._invalid_json_count == 0
        assert bot._missing_field_counts == {"p": 0, "q": 0, "T": 0}

    def test_initial_state(self, bot):
        """Todos os contadores existem e estão zerados após __init__."""
        assert bot._invalid_json_count == 0
        assert isinstance(bot._invalid_json_log_step, int)
        assert bot._missing_field_counts == {"p": 0, "q": 0, "T": 0}
        assert isinstance(bot._missing_field_log_step, int)
        assert bot._invalid_trade_count == 0
        assert isinstance(bot._invalid_trade_log_step, int)


# ===================================================================
# 6. _invalid_trade_count (tipo/valor inválido)
# ===================================================================


class TestInvalidTradeCount:
    def test_non_numeric_price_increments(self, bot):
        """Trade com preço não numérico incrementa _invalid_trade_count."""
        msg = dict(_VALID_AGGTRADE, p="NOT_A_NUMBER")
        bot.on_message(None, json.dumps(msg))
        assert bot._invalid_trade_count == 1

    def test_negative_price_increments(self, bot):
        """Trade com preço negativo incrementa _invalid_trade_count."""
        msg = dict(_VALID_AGGTRADE, p="-1.0")
        bot.on_message(None, json.dumps(msg))
        assert bot._invalid_trade_count == 1
