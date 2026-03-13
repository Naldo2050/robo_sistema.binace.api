# tests/test_out_of_order_pruning.py
"""
Testes para detecção e tratamento de trades out-of-order.

Verifica:
- Detecção de trades fora de ordem
- Pruning robusto que mantém trades corretos
- Reset do flag após pruning

NOTA: Usa mock de timestamp para testes determinísticos.
"""

import sys
import os
import pytest
from collections import deque
from decimal import Decimal
from unittest.mock import patch, MagicMock, PropertyMock

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flow_analyzer import FlowAnalyzer


class TestOutOfOrderPruning:
    """Testes para detecção e pruning de trades out-of-order."""
    
    @pytest.fixture
    def analyzer(self):
        """
        Fixture para FlowAnalyzer com timestamp mockado.
        
        Usa timestamps na faixa de 1000-100000 ms para testes,
        com referência mockada para 50000 ms.
        """
        analyzer = FlowAnalyzer()
        
        # Configura para teste
        analyzer.net_flow_windows_min = [1]  # 1 minute window = 60,000 ms
        analyzer.flow_trades_maxlen = 100
        analyzer.flow_trades = deque(maxlen=100)
        
        # Reset estado
        analyzer._max_ts_seen = 0
        analyzer._out_of_order_seen = False
        
        return analyzer
    
    def _process_trade_with_mock_time(self, analyzer, trade, reference_ts):
        """
        Processa trade com timestamp de referência mockado.
        
        Isso evita que o pruning automático remova trades
        com timestamps "antigos" em relação ao tempo real.
        """
        with patch.object(analyzer, '_get_synced_timestamp_ms', return_value=reference_ts):
            analyzer.process_trade(trade)
    
    def test_initial_state(self, analyzer):
        """Testa estado inicial."""
        assert analyzer._max_ts_seen == 0
        assert analyzer._out_of_order_seen is False
        assert len(analyzer.flow_trades) == 0
    
    def test_sequential_trades_no_out_of_order(self, analyzer):
        """Testa que trades sequenciais não acionam flag OOO."""
        # Reference time: 50000 (dentro da janela de 60000ms)
        ref_ts = 50000
        
        # Trade 1: ts=10000
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 10000, 'p': 100.0, 'm': False
        }, ref_ts)
        assert analyzer._max_ts_seen == 10000
        assert analyzer._out_of_order_seen is False
        
        # Trade 2: ts=20000 (depois)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 20000, 'p': 100.0, 'm': False
        }, ref_ts)
        assert analyzer._max_ts_seen == 20000
        assert analyzer._out_of_order_seen is False
        
        # Trade 3: ts=30000 (depois)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 30000, 'p': 100.0, 'm': False
        }, ref_ts)
        assert analyzer._max_ts_seen == 30000
        assert analyzer._out_of_order_seen is False
    
    def test_out_of_order_detection(self, analyzer):
        """Testa se trades out-of-order são detectados."""
        # Reference time grande o suficiente para não fazer prune
        ref_ts = 100000
        
        # Trade 1: ts=50000
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 50000, 'p': 100.0, 'm': False
        }, ref_ts)
        assert analyzer._max_ts_seen == 50000
        assert analyzer._out_of_order_seen is False
        
        # Trade 2: ts=80000 (futuro)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 80000, 'p': 100.0, 'm': False
        }, ref_ts)
        assert analyzer._max_ts_seen == 80000
        assert analyzer._out_of_order_seen is False
        
        # Trade 3: ts=60000 (LATE - chegou depois do 80000)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 60000, 'p': 100.0, 'm': False
        }, ref_ts)
        
        # Deve detectar out-of-order porque 60000 < 80000 (max_seen)
        assert analyzer._out_of_order_count > 0
        # max_ts_seen não deve mudar (ainda é 80000)
        assert analyzer._max_ts_seen == 80000
    
    def test_all_trades_stored_despite_ooo(self, analyzer):
        """Testa que todos os trades são armazenados mesmo com OOO."""
        ref_ts = 100000
        
        trades = [
            {'q': 1.0, 'T': 50000, 'p': 100.0, 'm': False},
            {'q': 1.0, 'T': 80000, 'p': 100.0, 'm': False},
            {'q': 1.0, 'T': 60000, 'p': 100.0, 'm': False},  # OOO
        ]
        
        for trade in trades:
            self._process_trade_with_mock_time(analyzer, trade, ref_ts)
        
        # Todos devem estar armazenados
        assert len(analyzer.flow_trades) == 3
        
        # Verifica timestamps
        stored_ts = sorted([t['ts'] for t in analyzer.flow_trades])
        assert stored_ts == [50000, 60000, 80000]
    
    def test_pruning_robustness_with_ooo(self, analyzer):
        """Testa que pruning robusto mantém trades corretos mesmo com OOO."""
        # Window é 1 minuto (60,000 ms)
        # Trades:
        # t1: 25000 (antigo - deve ser removido se cutoff >= 25000)
        # t2: 75000 (recente - deve ser mantido)
        # t3: 65000 (recente, chegou LATE após t2 - deve ser mantido)

        ref_ts = 80000

        # 1. Add t1 (25000)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 25000, 'p': 100, 'm': False
        }, ref_ts)

        # 2. Add t2 (75000)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 75000, 'p': 100, 'm': False
        }, ref_ts)

        # 3. Add t3 (65000) - LATE
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 65000, 'p': 100, 'm': False
        }, ref_ts)

        assert analyzer._out_of_order_count > 0
        assert len(analyzer.flow_trades) == 3

        # now = 90,000. cutoff = 90,000 - 60,000 = 30,000
        # t1 (25000) < 30000 -> REMOVE
        # t2 (75000) >= 30000 -> KEEP
        # t3 (65000) >= 30000 -> KEEP

        analyzer._prune_flow_history(now_ms=90000)
        
        # Deve ter 2 trades restantes
        assert len(analyzer.flow_trades) == 2
        
        timestamps = sorted([t['ts'] for t in analyzer.flow_trades])
        assert timestamps == [65000, 75000]
        
        # Flag deve ser resetado após pruning robusto
        assert analyzer._out_of_order_seen is False
    
    def test_pruning_fast_path_when_ordered(self, analyzer):
        """Testa que pruning usa caminho rápido quando não há OOO."""
        ref_ts = 100000
        
        # Adiciona trades em ordem
        for ts in [50000, 60000, 70000, 80000, 90000]:
            self._process_trade_with_mock_time(analyzer, {
                'q': 1.0, 'T': ts, 'p': 100, 'm': False
            }, ref_ts)
        
        assert analyzer._out_of_order_seen is False
        assert len(analyzer.flow_trades) == 5
        
        # Prune com now=100000, cutoff = 40000
        # Deve remover 50000? Não, 50000 >= 40000
        # Todos devem permanecer
        analyzer._prune_flow_history(now_ms=100000)
        
        assert len(analyzer.flow_trades) == 5
        
        # Agora prune mais agressivo: now=150000, cutoff=90000
        # Remove 50000, 60000, 70000, 80000
        # Mantém 90000
        analyzer._prune_flow_history(now_ms=150000)
        
        assert len(analyzer.flow_trades) == 1
        assert analyzer.flow_trades[0]['ts'] == 90000
    
    def test_pruning_removes_all_old(self, analyzer):
        """Testa que todos os trades antigos são removidos."""
        ref_ts = 100000
        
        # Adiciona trades
        for ts in [50000, 60000, 70000]:
            self._process_trade_with_mock_time(analyzer, {
                'q': 1.0, 'T': ts, 'p': 100, 'm': False
            }, ref_ts)
        
        assert len(analyzer.flow_trades) == 3
        
        # Prune com now muito no futuro: cutoff = 1000000 - 60000 = 940000
        analyzer._prune_flow_history(now_ms=1000000)
        
        # Todos devem ser removidos
        assert len(analyzer.flow_trades) == 0
    
    def test_pruning_keeps_all_recent(self, analyzer):
        """Testa que todos os trades recentes são mantidos."""
        ref_ts = 150000
        
        # Adiciona trades recentes
        for ts in [100000, 110000, 120000]:
            self._process_trade_with_mock_time(analyzer, {
                'q': 1.0, 'T': ts, 'p': 100, 'm': False
            }, ref_ts)
        
        assert len(analyzer.flow_trades) == 3
        
        # Prune com now = 130000, cutoff = 70000
        # Todos os trades são >= 70000
        analyzer._prune_flow_history(now_ms=130000)
        
        # Todos devem ser mantidos
        assert len(analyzer.flow_trades) == 3
    
    def test_cache_degradation_on_ooo(self, analyzer):
        """Testa que cache é degradado quando há OOO."""
        ref_ts = 200000
        initial_degraded = analyzer._cache_degraded_until_ms
        
        # Trade normal
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 100000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Trade no futuro
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 180000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Trade OOO (atrasado)
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 150000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Cache deve ser degradado
        assert analyzer._cache_degraded_until_ms > initial_degraded


class TestEdgeCases:
    """Testes de casos extremos."""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para FlowAnalyzer."""
        analyzer = FlowAnalyzer()
        analyzer.net_flow_windows_min = [1]
        analyzer.flow_trades = deque(maxlen=100)
        return analyzer
    
    def _process_trade_with_mock_time(self, analyzer, trade, reference_ts):
        """Processa trade com timestamp de referência mockado."""
        with patch.object(analyzer, '_get_synced_timestamp_ms', return_value=reference_ts):
            analyzer.process_trade(trade)
    
    def test_empty_prune(self, analyzer):
        """Testa pruning de deque vazio."""
        analyzer._prune_flow_history(now_ms=100000)
        assert len(analyzer.flow_trades) == 0
    
    def test_single_trade_prune_keep(self, analyzer):
        """Testa pruning com único trade que deve ser mantido."""
        ref_ts = 100000
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 80000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Prune com cutoff = 50000 (trade 80000 >= 50000)
        analyzer._prune_flow_history(now_ms=110000)  # cutoff = 50000
        assert len(analyzer.flow_trades) == 1
    
    def test_single_trade_prune_remove(self, analyzer):
        """Testa pruning com único trade que deve ser removido."""
        ref_ts = 100000
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 50000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Prune com cutoff = 140000 (trade 50000 < 140000)
        analyzer._prune_flow_history(now_ms=200000)  # cutoff = 140000
        assert len(analyzer.flow_trades) == 0
    
    def test_same_timestamp_trades(self, analyzer):
        """Testa múltiplos trades com mesmo timestamp."""
        ref_ts = 100000
        
        for i in range(5):
            self._process_trade_with_mock_time(analyzer, {
                'q': float(i + 1), 'T': 80000, 'p': 100, 'm': False
            }, ref_ts)
        
        assert len(analyzer.flow_trades) == 5
        assert analyzer._out_of_order_seen is False  # Mesmo ts não é OOO
        
        # Prune que mantém todos (cutoff = 40000)
        analyzer._prune_flow_history(now_ms=100000)
        assert len(analyzer.flow_trades) == 5
    
    def test_trades_at_cutoff_boundary(self, analyzer):
        """Testa trades exatamente no limite do cutoff."""
        ref_ts = 120000
        
        # Trade exatamente no cutoff
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 60000, 'p': 100, 'm': False
        }, ref_ts)
        
        # now = 120000, cutoff = 60000
        # Trade com ts=60000 está NO limite (60000 >= 60000) -> KEEP
        analyzer._prune_flow_history(now_ms=120000)
        assert len(analyzer.flow_trades) == 1
        
        # now = 120001, cutoff = 60001
        # Trade com ts=60000 está ABAIXO do limite (60000 < 60001) -> REMOVE
        analyzer._prune_flow_history(now_ms=120001)
        assert len(analyzer.flow_trades) == 0


class TestCVDAfterOOO:
    """Testes para garantir que CVD está correto após operações OOO."""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para FlowAnalyzer."""
        return FlowAnalyzer()
    
    def _process_trade_with_mock_time(self, analyzer, trade, reference_ts):
        """Processa trade com timestamp de referência mockado."""
        with patch.object(analyzer, '_get_synced_timestamp_ms', return_value=reference_ts):
            analyzer.process_trade(trade)
    
    def test_cvd_correct_with_ooo_trades(self, analyzer):
        """Testa que CVD está correto mesmo com trades OOO."""
        ref_ts = 200000
        
        # Trades em ordem: buy 2.0, sell 1.0, buy 0.5
        # CVD esperado: 2.0 - 1.0 + 0.5 = 1.5
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 2.0, 'T': 100000, 'p': 100, 'm': False  # buy
        }, ref_ts)
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 0.5, 'T': 180000, 'p': 100, 'm': False  # buy (futuro)
        }, ref_ts)
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 150000, 'p': 100, 'm': True  # sell (OOO)
        }, ref_ts)
        
        stats = analyzer.get_stats()
        # CVD: 2.0 + 0.5 - 1.0 = 1.5
        assert abs(float(stats['cvd']) - 1.5) < 0.001
    
    def test_cvd_correct_after_pruning(self, analyzer):
        """Testa que CVD permanece correto após pruning."""
        ref_ts = 200000
        
        # Trade antigo (será removido)
        self._process_trade_with_mock_time(analyzer, {
            'q': 10.0, 'T': 50000, 'p': 100, 'm': False
        }, ref_ts)
        
        # Trades recentes
        self._process_trade_with_mock_time(analyzer, {
            'q': 2.0, 'T': 150000, 'p': 100, 'm': False
        }, ref_ts)
        
        self._process_trade_with_mock_time(analyzer, {
            'q': 1.0, 'T': 160000, 'p': 100, 'm': True
        }, ref_ts)
        
        # CVD inicial: 10.0 + 2.0 - 1.0 = 11.0
        stats = analyzer.get_stats()
        assert abs(float(stats['cvd']) - 11.0) < 0.001
        
        # Prune (remove trade de 50000)
        # now=1040000, cutoff=140000 (1040000 - 15*60000)
        analyzer._prune_flow_history(now_ms=1040000)
        
        # CVD acumulado NÃO muda com prune (é acumulado total)
        # Apenas flow_trades é filtrado
        stats = analyzer.get_stats()
        assert abs(float(stats['cvd']) - 11.0) < 0.001
        
        # Mas flow_trades deve ter apenas 2 trades
        assert len(analyzer.flow_trades) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])