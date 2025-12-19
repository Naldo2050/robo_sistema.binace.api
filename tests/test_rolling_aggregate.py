# tests/test_rolling_aggregate.py
"""
Testes para RollingAggregate.

Testa:
- Inicialização
- Adição de trades
- Subtração correta no prune
- Eviction por max_trades
- Recompute de OHLC
"""

import sys
import os
import pytest
from decimal import Decimal
from collections import defaultdict

# Garante que a pasta raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flow_analyzer import RollingAggregate
from flow_analyzer.constants import DECIMAL_ZERO


class TestRollingAggregateInitialization:
    """Testes de inicialização."""
    
    def test_default_initialization(self):
        """Testa estado inicial padrão."""
        agg = RollingAggregate(window_min=1, max_trades=100)
        
        assert agg.window_min == 1
        assert agg.window_ms == 60000
        assert agg.max_trades == 100
        assert agg.sum_delta_btc == DECIMAL_ZERO
        assert agg.sum_buy_btc == DECIMAL_ZERO
        assert agg.sum_sell_btc == DECIMAL_ZERO
        assert len(agg.trades) == 0
    
    def test_ohlc_initial_state(self):
        """Testa estado inicial do OHLC."""
        agg = RollingAggregate(window_min=5, max_trades=50)
        
        assert agg._open is None
        assert agg._high is None
        assert agg._low is None
        assert agg._close is None
        assert agg._dirty_hilo is False
    
    def test_reset(self):
        """Testa método reset."""
        agg = RollingAggregate(window_min=1, max_trades=5)
        
        # Adiciona alguns trades
        agg.add_trade({
            'ts': 1000, 'qty': 1.0, 'price': 100.0,
            'delta_btc': 1.0, 'side': 'buy', 'sector': None
        }, whale_threshold=5.0)
        
        assert len(agg.trades) == 1
        assert agg.sum_buy_btc > DECIMAL_ZERO
        
        # Reset
        agg.reset()
        
        assert len(agg.trades) == 0
        assert agg.sum_buy_btc == DECIMAL_ZERO
        assert agg._open is None


class TestRollingAggregateAddTrade:
    """Testes para adição de trades."""
    
    @pytest.fixture
    def rolling_agg(self):
        """Fixture para criar instância limpa."""
        return RollingAggregate(window_min=1, max_trades=5)
    
    def test_add_single_buy_trade(self, rolling_agg):
        """Testa adição de trade de compra."""
        trade = {
            'ts': 1000,
            'qty': 1.0,
            'price': 50000.0,
            'delta_btc': 1.0,
            'side': 'buy',
            'sector': 'whale'
        }
        
        result = rolling_agg.add_trade(trade, whale_threshold=0.5)
        
        assert result is True
        assert len(rolling_agg.trades) == 1
        assert rolling_agg.sum_buy_btc == Decimal('1.0')
        assert rolling_agg.sum_sell_btc == DECIMAL_ZERO
        assert rolling_agg.sum_delta_btc == Decimal('1.0')
    
    def test_add_single_sell_trade(self, rolling_agg):
        """Testa adição de trade de venda."""
        trade = {
            'ts': 1000,
            'qty': 0.5,
            'price': 50000.0,
            'delta_btc': -0.5,
            'side': 'sell',
            'sector': 'retail'
        }
        
        rolling_agg.add_trade(trade, whale_threshold=1.0)
        
        assert rolling_agg.sum_sell_btc == Decimal('0.5')
        assert rolling_agg.sum_buy_btc == DECIMAL_ZERO
        assert rolling_agg.sum_delta_btc == Decimal('-0.5')
    
    def test_add_whale_trade(self, rolling_agg):
        """Testa tracking de whale trades."""
        # Trade whale (qty >= threshold)
        trade = {
            'ts': 1000,
            'qty': 10.0,
            'price': 50000.0,
            'delta_btc': 10.0,
            'side': 'buy'
        }
        
        rolling_agg.add_trade(trade, whale_threshold=5.0)
        
        assert rolling_agg.whale_buy == Decimal('10.0')
        assert rolling_agg.whale_sell == DECIMAL_ZERO
    
    def test_add_non_whale_trade(self, rolling_agg):
        """Testa que trades pequenos não contam como whale."""
        trade = {
            'ts': 1000,
            'qty': 0.5,
            'price': 50000.0,
            'delta_btc': 0.5,
            'side': 'buy'
        }
        
        rolling_agg.add_trade(trade, whale_threshold=5.0)
        
        assert rolling_agg.whale_buy == DECIMAL_ZERO
    
    def test_ohlc_single_trade(self, rolling_agg):
        """Testa OHLC com único trade."""
        trade = {
            'ts': 1000, 'qty': 1.0, 'price': 50000.0,
            'delta_btc': 1.0, 'side': 'buy'
        }
        
        rolling_agg.add_trade(trade, whale_threshold=5.0)
        
        metrics = rolling_agg.get_metrics(last_price=50000.0)
        
        assert metrics['ohlc'] == (50000.0, 50000.0, 50000.0, 50000.0)
    
    def test_ohlc_multiple_trades(self, rolling_agg):
        """Testa OHLC com múltiplos trades."""
        trades = [
            {'ts': 1, 'qty': 1, 'price': 100, 'delta_btc': 1, 'side': 'buy'},
            {'ts': 2, 'qty': 1, 'price': 150, 'delta_btc': 1, 'side': 'buy'},  # High
            {'ts': 3, 'qty': 1, 'price': 80, 'delta_btc': -1, 'side': 'sell'},  # Low
            {'ts': 4, 'qty': 1, 'price': 120, 'delta_btc': 1, 'side': 'buy'},   # Close
        ]
        
        for t in trades:
            rolling_agg.add_trade(t, whale_threshold=99)
        
        metrics = rolling_agg.get_metrics(0)
        o, h, l, c = metrics['ohlc']
        
        assert o == 100   # Primeiro
        assert h == 150   # Máximo
        assert l == 80    # Mínimo
        assert c == 120   # Último
    
    def test_reject_out_of_order_trade(self, rolling_agg):
        """Testa que trades out-of-order são rejeitados."""
        # Trade 1
        rolling_agg.add_trade({
            'ts': 2000, 'qty': 1, 'price': 100,
            'delta_btc': 1, 'side': 'buy'
        }, whale_threshold=99)
        
        # Trade 2 (out of order)
        result = rolling_agg.add_trade({
            'ts': 1000, 'qty': 1, 'price': 100,
            'delta_btc': 1, 'side': 'buy'
        }, whale_threshold=99)
        
        assert result is False
        assert len(rolling_agg.trades) == 1


class TestRollingAggregatePruning:
    """Testes para pruning."""
    
    @pytest.fixture
    def rolling_agg(self):
        return RollingAggregate(window_min=1, max_trades=10)
    
    def test_prune_subtracts_correctly(self, rolling_agg):
        """Testa que prune subtrai valores corretamente."""
        # t0: Compra 1.0
        rolling_agg.add_trade({
            'ts': 1000, 'qty': 1.0, 'price': 50000.0,
            'delta_btc': 1.0, 'side': 'buy', 'sector': None
        }, whale_threshold=5.0)
        
        # t1: Venda 0.5
        rolling_agg.add_trade({
            'ts': 2000, 'qty': 0.5, 'price': 50100.0,
            'delta_btc': -0.5, 'side': 'sell', 'sector': None
        }, whale_threshold=5.0)
        
        # Estado inicial
        assert rolling_agg.sum_buy_btc == Decimal('1.0')
        assert rolling_agg.sum_sell_btc == Decimal('0.5')
        assert rolling_agg.sum_delta_btc == Decimal('0.5')
        
        # Prune t0 (cutoff 1500 remove ts=1000)
        removed = rolling_agg.prune(cutoff_ms=1500)
        
        assert removed == 1
        assert len(rolling_agg.trades) == 1
        
        # Deve sobrar apenas a venda de 0.5
        assert rolling_agg.sum_buy_btc == DECIMAL_ZERO
        assert rolling_agg.sum_sell_btc == Decimal('0.5')
        assert rolling_agg.sum_delta_btc == Decimal('-0.5')
    
    def test_prune_updates_ohlc(self, rolling_agg):
        """Testa que OHLC é atualizado após prune."""
        # Trade 1: price 100
        rolling_agg.add_trade({
            'ts': 1000, 'qty': 1, 'price': 100,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # Trade 2: price 150
        rolling_agg.add_trade({
            'ts': 2000, 'qty': 1, 'price': 150,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # Antes do prune: open=100, close=150
        metrics = rolling_agg.get_metrics(0)
        assert metrics['ohlc'][0] == 100
        assert metrics['ohlc'][3] == 150
        
        # Prune trade 1
        rolling_agg.prune(1500)
        
        # Após prune: open=150, close=150
        metrics = rolling_agg.get_metrics(0)
        assert metrics['ohlc'][0] == 150
        assert metrics['ohlc'][3] == 150
    
    def test_prune_all(self, rolling_agg):
        """Testa prune de todos os trades."""
        for ts in [1000, 2000, 3000]:
            rolling_agg.add_trade({
                'ts': ts, 'qty': 1, 'price': 100,
                'delta_btc': 1, 'side': 'buy'
            }, whale_threshold=99)
        
        assert len(rolling_agg.trades) == 3
        
        # Prune tudo
        rolling_agg.prune(cutoff_ms=10000)
        
        assert len(rolling_agg.trades) == 0
        assert rolling_agg.sum_buy_btc == DECIMAL_ZERO
        assert rolling_agg._open is None


class TestRollingAggregateEviction:
    """Testes para eviction por max_trades."""
    
    def test_eviction_by_max_trades(self):
        """Testa que max_trades força eviction (FIFO)."""
        agg = RollingAggregate(window_min=1, max_trades=5)
        
        # Adiciona 10 trades
        for i in range(10):
            agg.add_trade({
                'ts': 1000 + i,
                'qty': 1.0,
                'price': 100.0 + i,
                'delta_btc': 1.0,
                'side': 'buy'
            }, whale_threshold=999.0)
        
        # Deve ter apenas 5 trades
        assert len(agg.trades) == 5
        
        # Evictions devem ter ocorrido
        assert agg.capacity_evictions == 5
        
        # Trades devem ser os últimos 5
        # Preços: 105, 106, 107, 108, 109
        metrics = agg.get_metrics(100.0)
        assert metrics['ohlc'][0] == 105.0  # Open (primeiro da fila)
        assert metrics['ohlc'][3] == 109.0  # Close (último)
        
        # Soma deve ser 5.0 (5 trades de 1.0 cada)
        assert agg.sum_buy_btc == Decimal('5.0')


class TestRollingAggregateOHLCRecompute:
    """Testes para recompute de OHLC."""
    
    @pytest.fixture
    def rolling_agg(self):
        return RollingAggregate(window_min=1, max_trades=10)
    
    def test_ohlc_recompute_after_high_removed(self, rolling_agg):
        """Testa recálculo de High após remoção do trade com preço máximo."""
        # 1. Price 100
        rolling_agg.add_trade({
            'ts': 1, 'qty': 1, 'price': 100,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # 2. Price 200 (High)
        rolling_agg.add_trade({
            'ts': 2, 'qty': 1, 'price': 200,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # 3. Price 150
        rolling_agg.add_trade({
            'ts': 3, 'qty': 1, 'price': 150,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        assert rolling_agg._high == 200.0
        
        # Remove trade 1 (Price 100) -> High ainda é 200
        rolling_agg.prune(1.5)
        metrics = rolling_agg.get_metrics(0)
        assert metrics['ohlc'][1] == 200.0
        
        # Remove trade 2 (Price 200) -> High deve cair para 150
        rolling_agg.prune(2.5)
        metrics = rolling_agg.get_metrics(0)
        assert metrics['ohlc'][1] == 150.0
    
    def test_ohlc_recompute_after_low_removed(self, rolling_agg):
        """Testa recálculo de Low após remoção do trade com preço mínimo."""
        # 1. Price 50 (Low)
        rolling_agg.add_trade({
            'ts': 1, 'qty': 1, 'price': 50,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # 2. Price 100
        rolling_agg.add_trade({
            'ts': 2, 'qty': 1, 'price': 100,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        # 3. Price 80
        rolling_agg.add_trade({
            'ts': 3, 'qty': 1, 'price': 80,
            'delta_btc': 0, 'side': 'buy'
        }, whale_threshold=99)
        
        assert rolling_agg._low == 50.0
        
        # Remove trade 1 (Low) -> Low deve ser recalculado para 80
        rolling_agg.prune(1.5)
        metrics = rolling_agg.get_metrics(0)
        assert metrics['ohlc'][2] == 80.0
    
    def test_ohlc_fallback_empty(self, rolling_agg):
        """Testa OHLC quando não há trades."""
        metrics = rolling_agg.get_metrics(last_price=50000.0)
        
        # Deve usar last_price
        assert metrics['ohlc'] == (50000.0, 50000.0, 50000.0, 50000.0)
    
    def test_ohlc_fallback_zero(self, rolling_agg):
        """Testa OHLC quando não há trades e last_price é 0."""
        metrics = rolling_agg.get_metrics(last_price=0.0)
        
        assert metrics['ohlc'] == (0.0, 0.0, 0.0, 0.0)


class TestRollingAggregateMetrics:
    """Testes para get_metrics."""
    
    @pytest.fixture
    def rolling_agg_with_trades(self):
        """Fixture com trades pré-adicionados."""
        agg = RollingAggregate(window_min=1, max_trades=100)
        
        trades = [
            {'ts': 1000, 'qty': 2.0, 'price': 50000, 'delta_btc': 2.0, 'side': 'buy', 'sector': 'whale'},
            {'ts': 2000, 'qty': 1.0, 'price': 50100, 'delta_btc': -1.0, 'side': 'sell', 'sector': 'mid'},
            {'ts': 3000, 'qty': 0.5, 'price': 50050, 'delta_btc': 0.5, 'side': 'buy', 'sector': 'retail'},
        ]
        
        for t in trades:
            agg.add_trade(t, whale_threshold=1.5)
        
        return agg
    
    def test_metrics_structure(self, rolling_agg_with_trades):
        """Testa estrutura do retorno de get_metrics."""
        metrics = rolling_agg_with_trades.get_metrics(50000.0)
        
        required_keys = [
            'sum_delta_btc', 'sum_delta_usd',
            'sum_buy_btc', 'sum_sell_btc',
            'sum_buy_usd', 'sum_sell_usd',
            'whale_buy', 'whale_sell', 'whale_delta',
            'ohlc', 'last_update', 'trade_count',
            'sector_agg'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_metrics_values(self, rolling_agg_with_trades):
        """Testa valores corretos em get_metrics."""
        metrics = rolling_agg_with_trades.get_metrics(50000.0)
        
        # Delta BTC: 2.0 - 1.0 + 0.5 = 1.5
        assert metrics['sum_delta_btc'] == 1.5
        
        # Whale: trade de 2.0 BTC (threshold=1.5)
        assert metrics['whale_buy'] == 2.0
        assert metrics['whale_sell'] == 0.0
        
        # Trade count
        assert metrics['trade_count'] == 3
    
    def test_metrics_sector_agg(self, rolling_agg_with_trades):
        """Testa agregação por sector."""
        metrics = rolling_agg_with_trades.get_metrics(50000.0)
        
        assert 'whale' in metrics['sector_agg']
        assert 'mid' in metrics['sector_agg']
        assert 'retail' in metrics['sector_agg']
        
        # Whale sector: 2.0 buy
        assert metrics['sector_agg']['whale']['buy_btc'] == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])