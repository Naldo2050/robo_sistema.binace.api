# tests/test_orderbook_analyzer.py
import pytest
import time
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock

# Importa a classe do arquivo original
# Ajuste o import conforme a estrutura de pastas do seu projeto
# Ex: from market_orchestrator.orderbook_analyzer import OrderBookAnalyzer
try:
    from orderbook_analyzer import OrderBookAnalyzer
except ImportError:
    # Fallback caso esteja rodando da raiz e o arquivo esteja lá
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from orderbook_analyzer import OrderBookAnalyzer

# ==========================================
# MOCKS E FIXTURES
# ==========================================

class MockTimeManager:
    """Simula o TimeManager para controlar o tempo nos testes."""
    def __init__(self):
        self._now = 1700000000000  # Timestamp fixo arbitrário (ms)

    def now_ms(self) -> int:
        return self._now

    def build_time_index(self, ts_ms, include_local=True, timespec="seconds") -> Dict[str, Any]:
        return {
            "epoch_ms": ts_ms,
            "timestamp_utc": "2023-11-14T22:13:20",
            "timestamp_ny": "2023-11-14T17:13:20"
        }

@pytest.fixture
def mock_tm():
    return MockTimeManager()

@pytest.fixture
def analyzer(mock_tm):
    """Instância do OrderBookAnalyzer com TimeManager mockado."""
    return OrderBookAnalyzer(
        symbol="BTCUSDT",
        liquidity_flow_alert_percentage=40.0,
        wall_std_dev_factor=3.0,
        time_manager=mock_tm,
        cache_ttl_seconds=10.0,
        max_stale_seconds=60.0,
        rate_limit_threshold=10,
        ob_limit_fetch=5
    )

@pytest.fixture
def valid_snapshot(mock_tm):
    """Gera um snapshot válido da Binance."""
    return {
        "lastUpdateId": 100,
        "E": mock_tm.now_ms() - 100,  # 100ms atrás
        "T": mock_tm.now_ms() - 100,
        # Formato: [price, qty]
        "bids": [
            ["50000.00", "1.0"],
            ["49990.00", "2.0"],
            ["49980.00", "5.0"]
        ],
        "asks": [
            ["50010.00", "1.0"],
            ["50020.00", "1.5"],
            ["50030.00", "3.0"]
        ]
    }

# ==========================================
# TESTES DE VALIDAÇÃO (_validate_snapshot)
# ==========================================

def test_validate_snapshot_success(analyzer, valid_snapshot):
    """Testa um snapshot perfeitamente válido."""
    is_valid, issues, converted = analyzer._validate_snapshot(valid_snapshot)
    
    assert is_valid is True
    assert len(issues) == 0
    assert len(converted["bids"]) == 3
    # Verifica conversão para float
    assert isinstance(converted["bids"][0][0], float)
    assert converted["bids"][0][0] == 50000.0

def test_validate_snapshot_stale_data(analyzer, valid_snapshot, mock_tm):
    """Testa rejeição de dados muito antigos."""
    # Define timestamp para 2 minutos atrás (120000ms)
    # O limite padrão no código é 60000ms (ORDERBOOK_MAX_AGE_MS)
    valid_snapshot["T"] = mock_tm.now_ms() - 120000
    valid_snapshot["E"] = mock_tm.now_ms() - 120000
    
    is_valid, issues, _ = analyzer._validate_snapshot(valid_snapshot)
    
    assert is_valid is False
    assert any("muito antigos" in issue for issue in issues)

def test_validate_snapshot_crossed_spread(analyzer, valid_snapshot):
    """Testa rejeição quando Bid >= Ask (Spread negativo)."""
    # Bid mais alto que o Ask
    valid_snapshot["bids"][0] = ["50020.00", "1.0"] 
    valid_snapshot["asks"][0] = ["50010.00", "1.0"] 
    
    is_valid, issues, _ = analyzer._validate_snapshot(valid_snapshot)
    
    assert is_valid is False
    assert any("spread negativo" in issue for issue in issues)

def test_validate_snapshot_zero_volume(analyzer, valid_snapshot):
    """Testa rejeição quando volume é zero ou inválido."""
    valid_snapshot["bids"][0] = ["50000.00", "0.0"] # Qty zero

    # O código soma os top 5 níveis. Vamos zerar todos para garantir falha
    valid_snapshot["bids"] = [["50000.00", "0.0"]]
    valid_snapshot["asks"] = [["50010.00", "0.0"]]

    is_valid, issues, _ = analyzer._validate_snapshot(valid_snapshot)

    assert is_valid is False
    assert len(issues) > 0

# ==========================================
# TESTES DE MÉTRICAS E CÁLCULOS
# ==========================================

def test_metrics_calculation(analyzer):
    """
    Testa cálculos matemáticos: Spread, Imbalance, Ratio.
    Cenário Controlado:
    Bid: $100 (qty 10) -> $1000
    Ask: $110 (qty 30) -> $3300
    Mid: $105
    Spread: $10
    Total Liquidez: $4300
    Imbalance: (1000 - 3300) / 4300 = -2300 / 4300 = -0.5348
    """
    bids = [(100.0, 10.0)]
    asks = [(110.0, 30.0)]
    
    # Testa _spread_and_depth
    sm = analyzer._spread_and_depth(bids, asks)
    assert sm["mid"] == 105.0
    assert sm["spread"] == 10.0
    assert sm["bid_depth_usd"] == 1000.0
    assert sm["ask_depth_usd"] == 3300.0
    
    # Testa _imbalance_ratio_pressure
    imb, ratio, pressure = analyzer._imbalance_ratio_pressure(sm["bid_depth_usd"], sm["ask_depth_usd"])
    
    expected_imbalance = (1000 - 3300) / (1000 + 3300)
    assert pytest.approx(imb, 0.0001) == expected_imbalance
    assert pytest.approx(ratio, 0.0001) == 1000/3300

def test_market_impact_simulation(analyzer):
    """
    Testa simulação de slippage.
    Livro:
    Ask 1: 100.0, Qty 1.0 (Total $100)
    Ask 2: 101.0, Qty 1.0 (Total $101)
    
    Ordem de Compra de $150:
    - Consome nível 1 inteiro ($100 @ 100.0)
    - Resta $50
    - Consome parcial nível 2 ($50 @ 101.0) -> Qty ~0.495
    
    Preço médio esperado: (100*1 + 101*(50/101)) / (1 + 50/101)
    Preço final: 101.0
    """
    levels = [(100.0, 1.0), (101.0, 1.0)]
    usd_amount = 150.0
    mid = 99.0 # Supondo mid price
    
    # Acessa função privada do módulo (ou método se for estático, mas no código original é função solta ou método)
    # No código fornecido, _simulate_market_impact é função solta no arquivo, mas estamos importando a classe.
    # Precisamos importar a função do módulo ou acessá-la se estiver disponível.
    # Assumindo que o teste roda no mesmo contexto, vamos usar a importação direta se possível,
    # caso contrário, adaptamos. No código original, é uma função solta.
    
    from orderbook_analyzer import _simulate_market_impact
    
    impact = _simulate_market_impact(levels, usd_amount, "buy", mid)
    
    assert impact["usd"] == 150.0
    assert impact["levels"] == 2 # Cruzou 2 níveis
    assert impact["final_price"] == 101.0
    assert impact["move_usd"] == 101.0 - mid
    assert impact["vwap"] > 100.0 and impact["vwap"] < 101.0

# ==========================================
# TESTES DE INTEGRAÇÃO (MÉTODO ANALYZE)
# ==========================================

@pytest.mark.asyncio
async def test_analyze_valid_flow(analyzer, valid_snapshot):
    """
    Testa o fluxo principal `analyze` injetando um snapshot válido.
    Não deve chamar rede.
    """
    result = await analyzer.analyze(current_snapshot=valid_snapshot)
    
    assert result["is_valid"] is True
    assert result["tipo_evento"] == "OrderBook"
    assert result["ativo"] == "BTCUSDT"
    
    # Verifica se métricas foram calculadas e populadas
    data = result["orderbook_data"]
    assert data["bid_depth_usd"] > 0
    assert data["ask_depth_usd"] > 0
    assert "imbalance" in data
    
    # Verifica qualidade dos dados
    assert result["data_quality"]["validation_passed"] is True
    assert result["data_quality"]["data_source"] == "external" # pois passamos current_snapshot

@pytest.mark.asyncio
async def test_analyze_critical_flags(analyzer, mock_tm):
    """Testa se flags críticas são ativadas com desequilíbrio extremo."""
    # Cria desequilíbrio massivo: Bids >>> Asks
    unbalanced_snapshot = {
        "E": mock_tm.now_ms(),
        "T": mock_tm.now_ms(),
        "bids": [["50000.0", "1000.0"]], # $50M
        "asks": [["50010.0", "0.1"]]     # $5k
    }
    
    result = await analyzer.analyze(current_snapshot=unbalanced_snapshot)
    
    assert result["is_valid"] is True
    assert result["critical_flags"]["is_critical"] is True
    assert "DESEQUILÍBRIO CRÍTICO" in str(result["alertas_liquidez"])

@pytest.mark.asyncio
async def test_analyze_fetch_failure(analyzer):
    """
    Testa comportamento quando fetch falha (retorna None) e não há snapshot injetado.
    Como _fetch_orderbook tentaria usar rede, precisamos mockar o método interno.
    """
    # Mockando _fetch_orderbook para retornar None (simulando falha de rede/timeout)
    analyzer._fetch_orderbook = AsyncMock(return_value=None)
    
    result = await analyzer.analyze()
    
    # Deve retornar um evento inválido, mas estruturado
    assert result["is_valid"] is False
    assert result["resultado_da_batalha"] == "INDISPONÍVEL"
    assert "fetch_failed" in result["erro"]

@pytest.mark.asyncio
async def test_analyze_partial_data_rejection(analyzer, valid_snapshot):
    """
    Testa se o analisador rejeita dados parciais (ex: um lado do livro vazio/zerado)
    quando a configuração ORDERBOOK_ALLOW_PARTIAL é False (padrão conservador).
    """
    # Zera o lado dos Asks
    valid_snapshot["asks"] = [["50010.0", "0.0"]] 
    
    result = await analyzer.analyze(current_snapshot=valid_snapshot)
    
    assert result["is_valid"] is False
    # O validador deve pegar "volume zero detectado" ou "liquidez muito baixa"
    assert len(result["data_quality"]["validation_issues"]) > 0 

# ==========================================
# TESTES DE STATS
# ==========================================

def test_get_stats(analyzer):
    """Testa se as estatísticas de saúde são retornadas corretamente."""
    # Simula alguns erros
    analyzer._fetch_errors = 5
    analyzer._total_fetches = 10
    
    stats = analyzer.get_stats()
    
    assert stats["total_fetches"] == 10
    assert stats["fetch_errors"] == 5
    assert stats["error_rate_pct"] == 50.0
    assert "config" in stats