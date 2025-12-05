import os
import sys
from decimal import Decimal
import pytest

# Garante que a pasta raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flow_analyzer import FlowAnalyzer, _to_decimal, _ui_safe_round_btc

# ==========================================
# 1. Testes de Funções Auxiliares (Pure)
# ==========================================

def test_to_decimal_conversion():
    """Testa conversão segura para Decimal."""
    assert _to_decimal(10) == Decimal('10')
    assert _to_decimal("0.0001") == Decimal('0.0001')
    assert _to_decimal(None) == Decimal('0')
    # Teste de float impreciso
    assert _to_decimal(0.1) == Decimal('0.1')

def test_ui_safe_round_btc_invariance():
    """Garante que buy + sell == total após arredondamento."""
    buy = Decimal('0.333333339')
    sell = Decimal('0.666666669')
    
    b_float, s_float, total_float, diff = _ui_safe_round_btc(buy, sell)
    
    # A soma dos componentes arredondados deve ser igual ao total retornado
    assert (b_float + s_float) == total_float
    # O diff deve ser pequeno
    assert diff < 1e-8

# ==========================================
# 2. Testes de Lógica de Negócio (Static)
# ==========================================

@pytest.mark.parametrize("delta, eps, esperado", [
    (-10.0, 1.0, "Absorção de Compra"), # Agressão Venda absorvida
    (10.0, 1.0, "Absorção de Venda"),   # Agressão Compra absorvida
    (0.5, 1.0, "Neutra"),               # Abaixo do threshold
    (-0.5, 1.0, "Neutra")
])
def test_classificar_absorcao_por_delta(delta, eps, esperado):
    assert FlowAnalyzer.classificar_absorcao_por_delta(delta, eps) == esperado

def test_classificar_absorcao_contextual_compra():
    """
    Testa detecção de Absorção de Compra (Sell Absorption).
    Cenário: Muita venda (Delta Negativo), mas preço fecha no topo.
    """
    analyzer = FlowAnalyzer()
    
    resultado = analyzer.classificar_absorcao_contextual(
        delta_btc=-100.0, # Muita agressão de venda
        open_p=100.0,
        high_p=100.5,
        low_p=99.0,
        close_p=100.4,    # Fechou perto da máxima
        eps=1.0
    )
    assert resultado == "Absorção de Compra"

def test_classificar_absorcao_contextual_neutra():
    """
    Cenário: Muita venda, preço cai (Movimento normal, sem absorção).
    """
    analyzer = FlowAnalyzer()
    
    resultado = analyzer.classificar_absorcao_contextual(
        delta_btc=-100.0,
        open_p=100.0,
        high_p=100.0,
        low_p=90.0,
        close_p=90.5,     # Fechou na mínima
        eps=1.0
    )
    assert resultado == "Neutra"

# ==========================================
# 3. Testes de Estado da Classe
# ==========================================

@pytest.fixture
def flow_analyzer():
    """Fixture para criar uma instância limpa para cada teste."""
    return FlowAnalyzer()

def test_process_trade_cvd_calculation(flow_analyzer):
    """Testa se o CVD é atualizado corretamente com trades."""
    # Trade de Compra (m=False -> Taker Buy)
    trade_buy = {'p': 50000.0, 'q': 1.5, 'T': 1620000000000, 'm': False}
    flow_analyzer.process_trade(trade_buy)
    
    # Trade de Venda (m=True -> Taker Sell)
    trade_sell = {'p': 50000.0, 'q': 0.5, 'T': 1620000000100, 'm': True}
    flow_analyzer.process_trade(trade_sell)
    
    # CVD esperado: +1.5 - 0.5 = 1.0
    stats = flow_analyzer.get_stats()
    assert float(stats['cvd']) == 1.0

def test_process_trade_whale_detection(flow_analyzer):
    """Testa se trades grandes são contabilizados como Whale."""
    # Configura threshold para teste
    flow_analyzer.whale_threshold = 5.0
    
    # Trade Baleia (Compra 10 BTC)
    whale_trade = {'p': 50000.0, 'q': 10.0, 'T': 1620000000000, 'm': False}
    flow_analyzer.process_trade(whale_trade)
    
    # Trade Varejo (Venda 0.1 BTC)
    retail_trade = {'p': 50000.0, 'q': 0.1, 'T': 1620000000100, 'm': True}
    flow_analyzer.process_trade(retail_trade)
    
    stats = flow_analyzer.get_stats()
    
    # Whale Delta deve considerar apenas os 10.0, ignorar o 0.1
    assert float(stats['whale_delta']) == 10.0
    # CVD geral considera tudo: 10.0 - 0.1 = 9.9
    assert float(stats['cvd']) == 9.9