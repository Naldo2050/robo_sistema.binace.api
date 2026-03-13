# tests/test_flow_analyzer.py
"""
Testes do FlowAnalyzer - Módulo principal.

Testa:
- Funções auxiliares (to_decimal, ui_safe_round_btc)
- Classificação de absorção
- Processamento de trades
- CVD e whale detection
"""

import os
import sys
from decimal import Decimal
import pytest

# Garante que a pasta raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports do módulo modularizado
from flow_analyzer import FlowAnalyzer
from flow_analyzer.utils import to_decimal, ui_safe_round_btc


# ==========================================
# 1. Testes de Funções Auxiliares (Pure)
# ==========================================

class TestUtilFunctions:
    """Testes para funções utilitárias."""
    
    def test_to_decimal_from_int(self):
        """Testa conversão de int para Decimal."""
        assert to_decimal(10) == Decimal('10')
        assert to_decimal(0) == Decimal('0')
        assert to_decimal(-5) == Decimal('-5')
    
    def test_to_decimal_from_string(self):
        """Testa conversão de string para Decimal."""
        assert to_decimal("0.0001") == Decimal('0.0001')
        assert to_decimal("123.456") == Decimal('123.456')
        assert to_decimal("-99.99") == Decimal('-99.99')
    
    def test_to_decimal_from_none(self):
        """Testa conversão de None para Decimal (deve retornar 0)."""
        assert to_decimal(None) == Decimal('0')
    
    def test_to_decimal_from_float(self):
        """Testa conversão de float para Decimal."""
        # Float é convertido via string para evitar imprecisão
        result = to_decimal(0.1)
        assert result == Decimal('0.1')
    
    def test_to_decimal_from_decimal(self):
        """Testa que Decimal passa direto."""
        original = Decimal('123.456')
        assert to_decimal(original) is original
    
    def test_to_decimal_invalid_returns_zero(self):
        """Testa que valores inválidos retornam zero."""
        assert to_decimal("invalid") == Decimal('0')
        assert to_decimal(object()) == Decimal('0')


class TestUISafeRoundBTC:
    """Testes para função ui_safe_round_btc."""
    
    def test_invariance_basic(self):
        """Garante que buy + sell == total após arredondamento."""
        buy = Decimal('0.333333339')
        sell = Decimal('0.666666669')
        
        b_float, s_float, total_float, diff = ui_safe_round_btc(buy, sell)
        
        # A soma dos componentes arredondados deve ser igual ao total retornado
        assert abs((b_float + s_float) - total_float) < 1e-10
        # O diff deve ser pequeno
        assert diff < 1e-7
    
    def test_invariance_zero_values(self):
        """Testa com valores zero."""
        buy = Decimal('0')
        sell = Decimal('0')
        
        b_float, s_float, total_float, diff = ui_safe_round_btc(buy, sell)
        
        assert b_float == 0.0
        assert s_float == 0.0
        assert total_float == 0.0
        assert diff == 0.0
    
    def test_invariance_large_values(self):
        """Testa com valores grandes."""
        buy = Decimal('1000000.12345678')
        sell = Decimal('500000.87654322')
        
        b_float, s_float, total_float, diff = ui_safe_round_btc(buy, sell)
        
        # Invariante deve ser mantida
        assert abs((b_float + s_float) - total_float) < 1e-7
    
    def test_invariance_negative_handling(self):
        """Testa comportamento com Decimals que poderiam ser negativos."""
        # Embora buy/sell não devam ser negativos, a função deve ser robusta
        buy = Decimal('1.5')
        sell = Decimal('0.5')
        
        b_float, s_float, total_float, diff = ui_safe_round_btc(buy, sell)
        
        assert b_float == 1.5
        assert s_float == 0.5
        assert total_float == 2.0


# ==========================================
# 2. Testes de Lógica de Negócio (Static)
# ==========================================

class TestAbsorptionClassification:
    """Testes para classificação de absorção."""
    
    @pytest.mark.parametrize("delta, eps, esperado", [
        (-10.0, 1.0, "Absorção de Compra"),  # Agressão Venda absorvida
        (10.0, 1.0, "Absorção de Venda"),    # Agressão Compra absorvida
        (0.5, 1.0, "Neutra"),                # Abaixo do threshold
        (-0.5, 1.0, "Neutra"),               # Abaixo do threshold (negativo)
        (0.0, 1.0, "Neutra"),                # Exatamente zero
        (-1.0, 1.0, "Neutra"),               # No limite (não passa)
        (1.0, 1.0, "Neutra"),                # No limite (não passa)
        (-1.01, 1.0, "Absorção de Compra"),  # Logo acima do limite
        (1.01, 1.0, "Absorção de Venda"),    # Logo acima do limite
    ])
    def test_classificar_absorcao_por_delta(self, delta, eps, esperado):
        """Testa classificação simples por delta."""
        resultado = FlowAnalyzer.classificar_absorcao_por_delta(delta, eps)
        assert resultado == esperado
    
    def test_classificar_absorcao_contextual_compra(self):
        """
        Testa detecção de Absorção de Compra (Sell Absorption).
        Cenário: Muita venda (Delta Negativo), mas preço fecha no topo.
        """
        analyzer = FlowAnalyzer()
        
        resultado = analyzer.classificar_absorcao_contextual(
            delta_btc=-100.0,  # Muita agressão de venda
            open_p=100.0,
            high_p=100.5,
            low_p=99.0,
            close_p=100.4,     # Fechou perto da máxima (topo do candle)
            eps=1.0
        )
        assert resultado == "Absorção de Compra"
    
    def test_classificar_absorcao_contextual_venda(self):
        """
        Testa detecção de Absorção de Venda (Buy Absorption).
        Cenário: Muita compra (Delta Positivo), mas preço fecha na mínima.
        """
        analyzer = FlowAnalyzer()
        
        resultado = analyzer.classificar_absorcao_contextual(
            delta_btc=100.0,   # Muita agressão de compra
            open_p=100.0,
            high_p=101.0,
            low_p=99.5,
            close_p=99.6,      # Fechou perto da mínima
            eps=1.0
        )
        assert resultado == "Absorção de Venda"
    
    def test_classificar_absorcao_contextual_neutra_movimento_normal(self):
        """
        Cenário: Muita venda, preço cai (Movimento normal, sem absorção).
        """
        analyzer = FlowAnalyzer()
        
        resultado = analyzer.classificar_absorcao_contextual(
            delta_btc=-100.0,
            open_p=100.0,
            high_p=100.0,
            low_p=90.0,
            close_p=90.5,      # Fechou na mínima (movimento seguiu delta)
            eps=1.0
        )
        assert resultado == "Neutra"
    
    def test_classificar_absorcao_contextual_neutra_delta_pequeno(self):
        """
        Cenário: Delta pequeno, não importa o OHLC.
        """
        analyzer = FlowAnalyzer()
        
        resultado = analyzer.classificar_absorcao_contextual(
            delta_btc=0.5,     # Delta pequeno (< eps)
            open_p=100.0,
            high_p=105.0,
            low_p=95.0,
            close_p=100.0,
            eps=1.0
        )
        assert resultado == "Neutra"


# ==========================================
# 3. Testes de Estado da Classe
# ==========================================

class TestFlowAnalyzerState:
    """Testes de estado e processamento de trades."""
    
    @pytest.fixture
    def flow_analyzer(self):
        """Fixture para criar uma instância limpa para cada teste."""
        return FlowAnalyzer()
    
    def test_initial_state(self, flow_analyzer):
        """Testa estado inicial do analyzer."""
        stats = flow_analyzer.get_stats()
        
        assert stats['total_trades_processed'] == 0
        assert stats['invalid_trades'] == 0
        assert float(stats['cvd']) == 0.0
        assert float(stats['whale_delta']) == 0.0
    
    def test_process_trade_cvd_calculation(self, flow_analyzer):
        """Testa se o CVD é atualizado corretamente com trades."""
        # Trade de Compra (m=False -> Taker Buy -> delta positivo)
        trade_buy = {'p': 50000.0, 'q': 1.5, 'T': 1620000000000, 'm': False}
        flow_analyzer.process_trade(trade_buy)
        
        # Trade de Venda (m=True -> Taker Sell -> delta negativo)
        trade_sell = {'p': 50000.0, 'q': 0.5, 'T': 1620000000100, 'm': True}
        flow_analyzer.process_trade(trade_sell)
        
        # CVD esperado: +1.5 - 0.5 = 1.0
        stats = flow_analyzer.get_stats()
        assert float(stats['cvd']) == 1.0
    
    def test_process_trade_whale_detection(self, flow_analyzer):
        """Testa se trades grandes são contabilizados como Whale."""
        # Configura threshold para teste
        flow_analyzer.whale_threshold = Decimal('5.0')
        
        # Trade Baleia (Compra 10 BTC)
        whale_trade = {'p': 50000.0, 'q': 10.0, 'T': 1620000000000, 'm': False}
        flow_analyzer.process_trade(whale_trade)
        
        # Trade Varejo (Venda 0.1 BTC) - abaixo do threshold
        retail_trade = {'p': 50000.0, 'q': 0.1, 'T': 1620000000100, 'm': True}
        flow_analyzer.process_trade(retail_trade)
        
        stats = flow_analyzer.get_stats()
        
        # Whale Delta deve considerar apenas os 10.0 (compra whale)
        # Retail trade (0.1) não é whale
        assert float(stats['whale_delta']) == 10.0
        
        # CVD geral considera tudo: 10.0 - 0.1 = 9.9
        assert float(stats['cvd']) == 9.9
    
    def test_process_trade_invalid_trade(self, flow_analyzer):
        """Testa que trades inválidos são contabilizados."""
        # Trade sem campos obrigatórios
        invalid_trade = {'invalid': 'data'}
        flow_analyzer.process_trade(invalid_trade)
        
        stats = flow_analyzer.get_stats()
        
        assert stats['total_trades_processed'] == 1
        assert stats['invalid_trades'] == 1
        assert float(stats['cvd']) == 0.0
    
    def test_process_trade_batch(self, flow_analyzer):
        """Testa processamento de batch de trades."""
        trades = [
            {'p': 50000.0, 'q': 1.0, 'T': 1620000000000, 'm': False},
            {'p': 50000.0, 'q': 1.0, 'T': 1620000000100, 'm': False},
            {'p': 50000.0, 'q': 0.5, 'T': 1620000000200, 'm': True},
        ]
        
        count = flow_analyzer.process_batch(trades)
        
        assert count == 3
        stats = flow_analyzer.get_stats()
        assert stats['total_trades_processed'] == 3
        # CVD: 1.0 + 1.0 - 0.5 = 1.5
        assert float(stats['cvd']) == 1.5
    
    def test_health_check(self, flow_analyzer):
        """Testa health check básico."""
        health = flow_analyzer.health_check()
        
        assert 'status' in health
        assert health['status'] in ['HEALTHY', 'DEGRADED', 'UNHEALTHY']
        assert 'issues' in health
        assert 'stats' in health


# ==========================================
# 4. Testes de Métricas
# ==========================================

class TestFlowMetrics:
    """Testes para get_flow_metrics."""
    
    @pytest.fixture
    def flow_analyzer_with_trades(self):
        """Fixture com trades pré-processados."""
        analyzer = FlowAnalyzer()
        
        # Simula alguns trades
        base_ts = 1620000000000
        trades = [
            {'p': 50000.0, 'q': 2.0, 'T': base_ts, 'm': False},        # Compra whale
            {'p': 50100.0, 'q': 1.0, 'T': base_ts + 1000, 'm': True},  # Venda
            {'p': 50050.0, 'q': 0.5, 'T': base_ts + 2000, 'm': False}, # Compra retail
        ]
        
        for trade in trades:
            analyzer.process_trade(trade)
        
        return analyzer
    
    def test_get_flow_metrics_structure(self, flow_analyzer_with_trades):
        """Testa estrutura do retorno de get_flow_metrics."""
        metrics = flow_analyzer_with_trades.get_flow_metrics()
        
        # Campos obrigatórios
        assert 'cvd' in metrics
        assert 'whale_delta' in metrics
        assert 'timestamp' in metrics
        assert 'data_quality' in metrics
    
    def test_get_flow_metrics_values(self, flow_analyzer_with_trades):
        """Testa valores corretos em get_flow_metrics."""
        metrics = flow_analyzer_with_trades.get_flow_metrics()
        
        # CVD: 2.0 - 1.0 + 0.5 = 1.5
        assert metrics['cvd'] == 1.5
        
        # Whale delta: apenas o trade de 2.0 BTC (se threshold padrão for 5.0, nenhum é whale)
        # Com threshold padrão de 5.0, nenhum trade é whale
        # Se quiser testar whale, precisa ajustar threshold


# ==========================================
# 5. Testes de Configuração Dinâmica
# ==========================================

class TestDynamicConfig:
    """Testes para update_config."""
    
    @pytest.fixture
    def flow_analyzer(self):
        return FlowAnalyzer()
    
    def test_update_config_whale_threshold(self, flow_analyzer):
        """Testa atualização de whale threshold."""
        original = float(flow_analyzer.whale_threshold)
        
        success = flow_analyzer.update_config({'whale_threshold': 10.0})
        
        assert success is True
        assert float(flow_analyzer.whale_threshold) == 10.0
    
    def test_update_config_invalid_value(self, flow_analyzer):
        """Testa que valores inválidos são rejeitados."""
        success = flow_analyzer.update_config({'whale_threshold': -1.0})
        
        assert success is False
    
    def test_update_config_absorcao_eps(self, flow_analyzer):
        """Testa atualização de epsilon de absorção."""
        success = flow_analyzer.update_config({'absorcao_eps': 2.5})
        
        assert success is True
        assert flow_analyzer.absorcao_eps == 2.5
    
    def test_update_config_guard_mode(self, flow_analyzer):
        """Testa atualização de guard mode."""
        success = flow_analyzer.update_config({'absorcao_guard_mode': 'off'})
        
        assert success is True
        assert flow_analyzer.absorcao_guard_mode == 'off'
    
    def test_update_config_invalid_guard_mode(self, flow_analyzer):
        """Testa que guard mode inválido é rejeitado."""
        success = flow_analyzer.update_config({'absorcao_guard_mode': 'invalid'})
        
        assert success is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])