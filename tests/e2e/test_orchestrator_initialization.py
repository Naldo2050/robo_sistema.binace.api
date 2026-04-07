# tests/test_orchestrator_initialization.py
import pytest
from unittest.mock import Mock

# Import the test configuration
try:
    from tests.fixtures import TestOrchestratorConfig
except ImportError:
    pytest.skip("TestOrchestratorConfig not available in tests.fixtures", allow_module_level=True)

# Import EnhancedMarketBot from the market_orchestrator module
from market_orchestrator import EnhancedMarketBot, adapt_orchestrator_runtime


def test_orchestrator_initialization(orchestrator_config_test):
    """Testa a inicialização do EnhancedMarketBot."""
    # Cria o bot com os parâmetros individuais, extraídos da configuração de teste
    bot = EnhancedMarketBot(
        stream_url=orchestrator_config_test.stream_url,
        symbol=orchestrator_config_test.symbol,
        window_size_minutes=orchestrator_config_test.window_size_minutes,
        vol_factor_exh=orchestrator_config_test.vol_factor_exh,
        history_size=orchestrator_config_test.history_size,
        delta_std_dev_factor=orchestrator_config_test.delta_std_dev_factor,
        context_sma_period=orchestrator_config_test.context_sma_period,
        liquidity_flow_alert_percentage=orchestrator_config_test.liquidity_flow_alert_percentage,
        wall_std_dev_factor=orchestrator_config_test.wall_std_dev_factor,
    )
    runtime = adapt_orchestrator_runtime(bot)
    snapshot = runtime.snapshot_state()
    assert bot.symbol == orchestrator_config_test.symbol
    assert bot.window_size_minutes == orchestrator_config_test.window_size_minutes
    # Verifica via contrato comum os componentes internos expostos para consumidores.
    assert snapshot["kind"] == "enhanced_market_bot"
    assert snapshot["symbol"] == orchestrator_config_test.symbol
    assert snapshot["health"]["health_monitor"] is True
    assert snapshot["health"]["event_bus"] is True
    assert bot.orderbook_analyzer is not None
    assert bot.flow_analyzer is not None


@pytest.fixture
def orchestrator_config_test():
    """Fixture que retorna uma instância de TestOrchestratorConfig para os testes."""
    return TestOrchestratorConfig()
