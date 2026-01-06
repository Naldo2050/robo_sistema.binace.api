# tests/test_orderbook_validate_snapshot.py - VERSÃO CORRIGIDA
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Definindo a constante que estava faltando
ORDERBOOK_MAX_AGE_MS = 5000  # 5 segundos

class TestOrderBookSnapshotValidation:
    """Testes para validação de snapshot do orderbook - VERSÃO CORRIGIDA"""
    
    def test_snapshot_too_old(self):
        """Testa snapshot muito antigo"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        # Cria snapshot antigo
        old_timestamp = datetime.now() - timedelta(seconds=10)
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=old_timestamp,
            spread=1.0,
            mid_price=50000.5
        )
        
        # Calcula idade
        age_ms = (datetime.now() - snapshot.timestamp).total_seconds() * 1000
        
        assert age_ms > ORDERBOOK_MAX_AGE_MS
        assert snapshot.is_too_old(max_age_ms=ORDERBOOK_MAX_AGE_MS) is True
    
    def test_snapshot_fresh(self):
        """Testa snapshot recente"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        # Cria snapshot recente
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=datetime.now(),
            spread=1.0,
            mid_price=50000.5
        )
        
        assert snapshot.is_too_old(max_age_ms=ORDERBOOK_MAX_AGE_MS) is False
    
    def test_snapshot_validation(self):
        """Testa validação completa do snapshot"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[50000, 1.0], [49999, 2.0]],
            asks=[[50001, 1.0], [50002, 1.5]],
            timestamp=datetime.now(),
            spread=1.0,
            mid_price=50000.5
        )
        
        # Validações básicas
        assert snapshot.validate() is True
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.spread > 0
        assert snapshot.mid_price == 50000.5
    
    def test_snapshot_with_invalid_data(self):
        """Testa snapshot com dados inválidos"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        # Testa com bids vazio
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[],
            asks=[[50001, 1.0]],
            timestamp=datetime.now(),
            spread=1.0,
            mid_price=50000.5
        )
        
        assert snapshot.validate() is False
        
        # Testa com preços inválidos
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[-50000, 1.0]],  # Preço negativo
            asks=[[50001, 1.0]],
            timestamp=datetime.now(),
            spread=1.0,
            mid_price=50000.5
        )
        
        assert snapshot.validate() is False
    
    def test_snapshot_serialization(self):
        """Testa serialização do snapshot"""
        from orderbook_core.orderbook import OrderBookSnapshot
        import json
        
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=datetime.now(),
            spread=1.0,
            mid_price=50000.5
        )
        
        # Serializa para JSON
        json_str = snapshot.to_json()
        data = json.loads(json_str)
        
        assert data['symbol'] == 'BTCUSDT'
        assert data['sequence'] == 100
        assert 'bids' in data
        assert 'asks' in data
        assert 'timestamp' in data
    
    def test_snapshot_from_json(self):
        """Testa criação de snapshot a partir de JSON"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        json_data = {
            'symbol': 'BTCUSDT',
            'sequence': 100,
            'bids': [[50000.0, 1.5], [49999.0, 2.3]],
            'asks': [[50001.0, 2.0], [50002.0, 1.5]],
            'timestamp': datetime.now().isoformat(),
            'spread': 2.0,
            'mid_price': 50000.5
        }
        
        snapshot = OrderBookSnapshot.from_json(json_data)
        
        assert snapshot.symbol == 'BTCUSDT'
        assert snapshot.sequence == 100
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.spread == 2.0
    
    @pytest.mark.parametrize("age_seconds,expected_valid", [
        (1, True),    # 1 segundo - válido
        (3, True),    # 3 segundos - válido
        (6, False),   # 6 segundos - inválido (mais que 5s)
        (10, False),  # 10 segundos - inválido
    ])
    def test_snapshot_age_validation(self, age_seconds, expected_valid):
        """Testa validação por idade do snapshot"""
        from orderbook_core.orderbook import OrderBookSnapshot
        
        timestamp = datetime.now() - timedelta(seconds=age_seconds)
        snapshot = OrderBookSnapshot(
            symbol='BTCUSDT',
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=timestamp,
            spread=1.0,
            mid_price=50000.5
        )
        
        is_valid = not snapshot.is_too_old(max_age_ms=5000)  # 5 segundos
        assert is_valid == expected_valid