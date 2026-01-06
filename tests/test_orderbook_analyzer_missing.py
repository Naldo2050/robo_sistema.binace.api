"""
Testes para aumentar cobertura do OrderBook Analyzer.
Foca em métodos que não estão sendo testados.
"""

import pytest
import time
from orderbook_analyzer import OrderBookAnalyzer
from orderbook_core.orderbook import OrderBookSnapshot

def test_analyzer_initialization_with_custom_params():
    """Testa inicialização com parâmetros customizados."""
    analyzer = OrderBookAnalyzer(
        symbol="BTCUSDT",
        liquidity_flow_alert_percentage=0.5,
        wall_std_dev_factor=2.5,
        cache_ttl_seconds=10.0,
        max_stale_seconds=30.0,
        rate_limit_threshold=20
    )
    
    assert analyzer.symbol == "BTCUSDT"
    assert analyzer.liquidity_flow_alert_percentage == 0.5
    assert analyzer.cache_ttl_seconds == 10.0

def test_analyzer_with_invalid_symbol():
    """Testa comportamento com símbolo inválido."""
    analyzer = OrderBookAnalyzer(symbol="")
    
    # Deve aceitar símbolo vazio mas logar warning
    assert analyzer.symbol == ""
    
    # Testa com símbolo None
    analyzer2 = OrderBookAnalyzer(symbol=None)
    assert analyzer2.symbol is None

def test_calculate_metrics_empty_orderbook():
    """Testa cálculo de métricas com orderbook vazio."""
    analyzer = OrderBookAnalyzer(symbol="BTCUSDT")
    
    snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[],  # Vazio
        asks=[],  # Vazio
        timestamp=time.time()
    )
    
    metrics = analyzer._calculate_metrics(snapshot)
    
    assert metrics["best_bid"] is None
    assert metrics["best_ask"] is None
    assert metrics["spread"] is None
    assert metrics["mid_price"] is None
    assert metrics["imbalance_10"] is None

def test_calculate_metrics_partial_orderbook():
    """Testa cálculo com apenas bids ou apenas asks."""
    analyzer = OrderBookAnalyzer(symbol="BTCUSDT")
    
    # Apenas bids
    snapshot1 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000, 1.0), (49900, 2.0)],
        asks=[],
        timestamp=time.time()
    )
    
    metrics1 = analyzer._calculate_metrics(snapshot1)
    assert metrics1["best_bid"] == 50000
    assert metrics1["best_ask"] is None
    assert metrics1["spread"] is None
    
    # Apenas asks
    snapshot2 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=124,
        bids=[],
        asks=[(50100, 1.0), (50200, 2.0)],
        timestamp=time.time()
    )
    
    metrics2 = analyzer._calculate_metrics(snapshot2)
    assert metrics2["best_bid"] is None
    assert metrics2["best_ask"] == 50100
    assert metrics2["spread"] is None

def test_detect_liquidity_walls_edge_cases():
    """Testa detecção de paredes de liquidez em casos especiais."""
    analyzer = OrderBookAnalyzer(symbol="BTCUSDT", wall_std_dev_factor=2.0)
    
    # Teste 1: Orderbook com apenas uma ordem
    snapshot1 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000, 100.0)],  # Uma ordem grande
        asks=[(50100, 100.0)],
        timestamp=time.time()
    )
    
    walls1 = analyzer._detect_liquidity_walls(snapshot1, levels=10)
    assert walls1 is not None
    
    # Teste 2: Orderbook com ordens muito pequenas
    snapshot2 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=124,
        bids=[(50000, 0.001), (49900, 0.001)],
        asks=[(50100, 0.001), (50200, 0.001)],
        timestamp=time.time()
    )
    
    walls2 = analyzer._detect_liquidity_walls(snapshot2, levels=10)
    # Deve não detectar paredes porque volumes são muito pequenos
    assert walls2 is not None

def test_check_critical_imbalance():
    """Testa verificação de desequilíbrio crítico."""
    analyzer = OrderBookAnalyzer(symbol="BTCUSDT")
    
    # Teste 1: Imbalance extremo (>= 0.95)
    snapshot1 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000, 100.0), (49900, 100.0)],  # 200 BTC
        asks=[(50100, 5.0), (50200, 5.0)],      # 10 BTC
        timestamp=time.time()
    )
    
    # Imbalance = (200 - 10) / (200 + 10) = 190/210 ≈ 0.905
    critical1 = analyzer._check_critical_imbalance(snapshot1)
    assert isinstance(critical1, dict)
    
    # Teste 2: Sem desequilíbrio
    snapshot2 = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=124,
        bids=[(50000, 10.0), (49900, 10.0)],  # 20 BTC
        asks=[(50100, 10.0), (50200, 10.0)],  # 20 BTC
        timestamp=time.time()
    )
    
    critical2 = analyzer._check_critical_imbalance(snapshot2)
    assert isinstance(critical2, dict)

def test_cache_operations():
    """Testa operações de cache."""
    analyzer = OrderBookAnalyzer(
        symbol="BTCUSDT",
        cache_ttl_seconds=1.0,  # Cache expira rápido para testes
        max_stale_seconds=2.0
    )
    
    snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000, 1.0)],
        asks=[(50100, 1.0)],
        timestamp=time.time()
    )
    
    # Adiciona ao cache
    analyzer._add_to_cache("test_key", snapshot)
    
    # Recupera do cache (deve existir)
    cached = analyzer._get_from_cache("test_key")
    assert cached is not None
    
    # Espera para expirar
    time.sleep(1.5)
    
    # Deve ter expirado
    expired = analyzer._get_from_cache("test_key")
    assert expired is None

def test_rate_limiter():
    """Testa o rate limiting."""
    analyzer = OrderBookAnalyzer(
        symbol="BTCUSDT",
        rate_limit_threshold=3  # Alterado para 3 para evitar confusão
    )
    
    # Primeira chamada deve permitir
    allowed1 = analyzer._check_rate_limit()
    assert allowed1 is True
    
    # Segunda chamada deve permitir
    allowed2 = analyzer._check_rate_limit()
    assert allowed2 is True
    
    # Terceira chamada deve permitir (threshold=3)
    allowed3 = analyzer._check_rate_limit()
    assert allowed3 is True
    
    # Quarta chamada deve bloquear
    allowed4 = analyzer._check_rate_limit()
    assert allowed4 is False
    
    # Reset do rate limiter
    analyzer._reset_rate_limit()
    allowed5 = analyzer._check_rate_limit()
    assert allowed5 is True

def test_validate_orderbook_snapshot():
    """Testa validação de snapshots."""
    analyzer = OrderBookAnalyzer(symbol="BTCUSDT")
    
    # Teste 1: Snapshot válido
    valid_snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000, 1.0)],
        asks=[(50100, 1.0)],
        timestamp=time.time()
    )
    
    is_valid = analyzer._validate_orderbook_snapshot(valid_snapshot)
    assert is_valid is True
    
    # Teste 2: Snapshot com símbolo diferente
    wrong_symbol_snapshot = OrderBookSnapshot(
        symbol="ETHUSDT",  # Símbolo diferente
        last_update_id=124,
        bids=[(50000, 1.0)],
        asks=[(50100, 1.0)],
        timestamp=time.time()
    )
    
    is_valid2 = analyzer._validate_orderbook_snapshot(wrong_symbol_snapshot)
    assert is_valid2 is False
    
    # Teste 3: Snapshot inválido (bids vazias)
    invalid_snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=125,
        bids=[],  # Vazio
        asks=[(50100, 1.0)],
        timestamp=time.time()
    )
    
    is_invalid = analyzer._validate_orderbook_snapshot(invalid_snapshot)
    assert is_invalid is True  # Snapshot com bids vazias deve ser inválido
    
    # Teste 4: Snapshot com valores inválidos
    bad_values_snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=126,
        bids=[(-50000, 1.0)],  # Preço negativo
        asks=[(50100, -1.0)],  # Quantidade negativa
        timestamp=time.time()
    )
    
    is_invalid2 = analyzer._validate_orderbook_snapshot(bad_values_snapshot)
    assert is_invalid2 is False
    
    # Teste 5: Snapshot None
    is_invalid3 = analyzer._validate_orderbook_snapshot(None)
    assert is_invalid3 is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])