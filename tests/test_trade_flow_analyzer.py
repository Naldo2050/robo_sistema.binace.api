# tests/test_trade_flow_analyzer.py
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

# Adiciona o diretório raiz ao sys.path para garantir importações corretas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_orchestrator.flow.trade_flow_analyzer import TradeFlowAnalyzer

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_tz():
    return ZoneInfo("UTC")

@pytest.fixture
def analyzer(mock_tz):
    # Inicializa com fator de exaustão 2.0
    return TradeFlowAnalyzer(vol_factor_exh=2.0, tz_output=mock_tz)

@pytest.fixture
def sample_window_data():
    return [
        {"p": 100.0, "q": 1.0, "T": 1620000000000, "m": False},
        {"p": 101.0, "q": 2.0, "T": 1620000001000, "m": True},
        {"p": 100.5, "q": 0.5, "T": 1620000002000, "m": False},
    ]

# ==========================================
# TESTES
# ==========================================

def test_initialization(analyzer, mock_tz):
    """Testa a inicialização correta dos atributos."""
    assert analyzer.vol_factor_exh == 2.0
    assert analyzer.tz_output == mock_tz

def test_analyze_window_empty_data(analyzer):
    """
    Testa o comportamento quando a janela de dados está vazia.
    Deve retornar dicionários padrão 'zerados' e não chamar o data_handler.
    """
    with patch("market_orchestrator.flow.trade_flow_analyzer.create_absorption_event") as mock_abs, \
         patch("market_orchestrator.flow.trade_flow_analyzer.create_exhaustion_event") as mock_exh:
        
        abs_evt, exh_evt = analyzer.analyze_window(
            window_data=[],
            symbol="BTCUSDT",
            history_volumes=[100, 200],
            dynamic_delta_threshold=50.0
        )

        # Verificações
        assert abs_evt["is_signal"] is False
        assert abs_evt["volume_total"] == 0
        assert exh_evt["is_signal"] is False
        assert exh_evt["volume_total"] == 0
        
        # Garante que as funções pesadas não foram chamadas
        mock_abs.assert_not_called()
        mock_exh.assert_not_called()

def test_analyze_window_insufficient_data(analyzer):
    """
    Testa comportamento com menos de 2 trades.
    O código original exige len(window_data) >= 2.
    """
    single_trade = [{"p": 100.0, "q": 1.0, "T": 1000, "m": False}]
    
    abs_evt, exh_evt = analyzer.analyze_window(
        window_data=single_trade,
        symbol="BTCUSDT",
        history_volumes=[],
        dynamic_delta_threshold=10.0
    )
    
    assert abs_evt["is_signal"] is False
    assert exh_evt["is_signal"] is False

def test_analyze_window_valid_execution(analyzer, sample_window_data):
    """
    Testa o fluxo completo com dados válidos.
    Verifica se as funções do data_handler são chamadas com os argumentos corretos.
    """
    # Mocks para as funções importadas de data_handler
    # Precisamos patchear onde elas são usadas (no módulo trade_flow_analyzer)
    with patch("market_orchestrator.flow.trade_flow_analyzer.create_absorption_event") as mock_abs, \
         patch("market_orchestrator.flow.trade_flow_analyzer.create_exhaustion_event") as mock_exh:
        
        # Configura retornos simulados
        mock_abs.return_value = {"type": "ABSORPTION", "is_signal": True, "delta": 50}
        mock_exh.return_value = {"type": "EXHAUSTION", "is_signal": False}

        history = [10.0, 20.0, 15.0]
        threshold = 5.5
        symbol = "ETHUSDT"
        profile = {"poc": 100}

        # Executa
        res_abs, res_exh = analyzer.analyze_window(
            window_data=sample_window_data,
            symbol=symbol,
            history_volumes=history,
            dynamic_delta_threshold=threshold,
            historical_profile=profile
        )

        # Verifica retorno
        assert res_abs == mock_abs.return_value
        assert res_exh == mock_exh.return_value

        # Verifica chamada de create_absorption_event
        mock_abs.assert_called_once()
        call_args_abs = mock_abs.call_args
        assert call_args_abs[0][0] == sample_window_data # window_data
        assert call_args_abs[0][1] == symbol             # symbol
        assert call_args_abs[1]['delta_threshold'] == threshold
        assert call_args_abs[1]['tz_output'] == analyzer.tz_output
        assert call_args_abs[1]['historical_profile'] == profile

        # Verifica chamada de create_exhaustion_event
        mock_exh.assert_called_once()
        call_args_exh = mock_exh.call_args
        assert call_args_exh[0][0] == sample_window_data
        assert call_args_exh[0][1] == symbol
        assert call_args_exh[1]['history_volumes'] == history
        assert call_args_exh[1]['volume_factor'] == analyzer.vol_factor_exh # Deve usar o fator da classe
        assert call_args_exh[1]['tz_output'] == analyzer.tz_output
        assert call_args_exh[1]['historical_profile'] == profile