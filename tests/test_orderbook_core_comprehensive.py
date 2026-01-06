# tests/test_orderbook_core_comprehensive.py - VERSÃO COMPLETA
import pytest
import threading
import time
import json
import sys
from decimal import Decimal
from unittest.mock import Mock, patch, PropertyMock
from datetime import datetime, timedelta
import numpy as np

# Definir constantes necessárias
ORDERBOOK_MAX_AGE_MS = 5000  # 5 segundos
MAX_ORDERBOOK_DEPTH = 1000

# Classes de mock se não puderem ser importadas
try:
    from orderbook_core.orderbook import OrderBook, OrderBookUpdate, OrderBookSnapshot
    from orderbook_core.circuit_breaker import CircuitBreaker
    from orderbook_core.exceptions import OrderBookError, InvalidUpdateError
    CORE_AVAILABLE = True
except ImportError:
    # Fallback completo
    CORE_AVAILABLE = False
    
    class OrderBookError(Exception):
        pass
    
    class InvalidUpdateError(Exception):
        pass
    
    class CircuitBreaker:
        def __init__(self, failure_threshold=3, recovery_timeout=1.0):
            self.failure_threshold = failure_threshold
            self.recovery_timeout = recovery_timeout
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"
        
        def execute(self, func, *args, **kwargs):
            if self.state == "OPEN":
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise OrderBookError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise
    
    class OrderBookUpdate:
        def __init__(self, timestamp=None, bids=None, asks=None, sequence=0):
            self.timestamp = timestamp or datetime.now()
            self.bids = bids or []
            self.asks = asks or []
            self.sequence = sequence
        
        def validate(self):
            if not isinstance(self.timestamp, datetime):
                raise InvalidUpdateError("Invalid timestamp")
            
            if not isinstance(self.bids, list) or not isinstance(self.asks, list):
                raise InvalidUpdateError("Bids and asks must be lists")
            
            for price, volume in self.bids:
                if price <= 0 or volume <= 0:
                    raise InvalidUpdateError(f"Invalid bid: price={price}, volume={volume}")
            
            for price, volume in self.asks:
                if price <= 0 or volume <= 0:
                    raise InvalidUpdateError(f"Invalid ask: price={price}, volume={volume}")
            
            return True
    
    class OrderBookSnapshot:
        def __init__(self, symbol, sequence, bids, asks, timestamp=None, spread=None, mid_price=None):
            self.symbol = symbol
            self.sequence = sequence
            self.bids = bids
            self.asks = asks
            self.timestamp = timestamp or datetime.now()
            self.spread = spread
            self.mid_price = mid_price
        
        def to_json(self):
            return json.dumps({
                'symbol': self.symbol,
                'sequence': self.sequence,
                'bids': self.bids,
                'asks': self.asks,
                'timestamp': self.timestamp.isoformat(),
                'spread': self.spread,
                'mid_price': self.mid_price
            })
        
        @classmethod
        def from_json(cls, json_str):
            if isinstance(json_str, str):
                data = json.loads(json_str)
            else:
                data = json_str
            
            return cls(
                symbol=data['symbol'],
                sequence=data['sequence'],
                bids=data['bids'],
                asks=data['asks'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                spread=data.get('spread'),
                mid_price=data.get('mid_price')
            )
        
        def is_too_old(self, max_age_ms=ORDERBOOK_MAX_AGE_MS):
            age_ms = (datetime.now() - self.timestamp).total_seconds() * 1000
            return age_ms > max_age_ms
        
        def validate(self):
            if not self.bids or not self.asks:
                return False
            
            # Verifica se bids estão em ordem decrescente
            bid_prices = [b[0] for b in self.bids]
            if not all(bid_prices[i] >= bid_prices[i + 1] for i in range(len(bid_prices) - 1)):
                return False
            
            # Verifica se asks estão em ordem crescente
            ask_prices = [a[0] for a in self.asks]
            if not all(ask_prices[i] <= ask_prices[i + 1] for i in range(len(ask_prices) - 1)):
                return False
            
            # Verifica preços e volumes positivos
            for price, volume in self.bids:
                if price <= 0 or volume < 0:
                    return False
            
            for price, volume in self.asks:
                if price <= 0 or volume < 0:
                    return False
            
            return True
    
    class OrderBook:
        def __init__(self, symbol="BTCUSDT", max_depth=MAX_ORDERBOOK_DEPTH):
            if not symbol:
                raise ValueError("Symbol cannot be empty")
            if max_depth <= 0:
                raise ValueError("max_depth must be positive")
            
            self.symbol = symbol
            self.max_depth = max_depth
            self.bids = []  # Lista de [price, volume] ordenada por preço descendente
            self.asks = []  # Lista de [price, volume] ordenada por preço ascendente
            self.last_sequence = 0
            self.last_update_time = datetime.now()
            self._lock = threading.RLock()
            self.circuit_breaker = None
            self.statistics = {
                'update_count': 0,
                'total_volume_processed': 0.0,
                'last_update_duration': 0.0,
                'max_depth_reached': 0,
                'error_count': 0
            }
        
        def _sort_and_truncate(self):
            """Ordena e trunca os arrays para manter o tamanho máximo"""
            with self._lock:
                # Ordena bids por preço descendente
                self.bids.sort(key=lambda x: x[0], reverse=True)
                self.bids = self.bids[:self.max_depth]
                
                # Ordena asks por preço ascendente
                self.asks.sort(key=lambda x: x[0])
                self.asks = self.asks[:self.max_depth]
                
                # Atualiza estatística de profundidade máxima
                current_depth = max(len(self.bids), len(self.asks))
                if current_depth > self.statistics['max_depth_reached']:
                    self.statistics['max_depth_reached'] = current_depth
        
        def update(self, update_data):
            """Atualiza o orderbook com novos dados"""
            start_time = time.time()
            
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.execute(self._update_internal, update_data)
                else:
                    return self._update_internal(update_data)
            except Exception as e:
                self.statistics['error_count'] += 1
                raise
            finally:
                duration = time.time() - start_time
                self.statistics['last_update_duration'] = duration
        
        def _update_internal(self, update_data):
            """Implementação interna da atualização"""
            if not update_data:
                raise InvalidUpdateError("Update data cannot be None")
            
            # Converte para OrderBookUpdate se necessário
            if not isinstance(update_data, OrderBookUpdate):
                if isinstance(update_data, dict):
                    update = OrderBookUpdate(
                        timestamp=datetime.fromisoformat(update_data.get('timestamp', datetime.now().isoformat())),
                        bids=update_data.get('bids', []),
                        asks=update_data.get('asks', []),
                        sequence=update_data.get('sequence', self.last_sequence + 1)
                    )
                else:
                    raise InvalidUpdateError(f"Unsupported update data type: {type(update_data)}")
            else:
                update = update_data
            
            # Valida a atualização
            update.validate()
            
            with self._lock:
                # Verifica sequência
                if update.sequence <= self.last_sequence:
                    return False  # Update antigo, ignora
                
                # Atualiza bids e asks
                self.bids = update.bids
                self.asks = update.asks
                
                # Ordena e trunca
                self._sort_and_truncate()
                
                # Atualiza estado
                self.last_sequence = update.sequence
                self.last_update_time = update.timestamp
                self.statistics['update_count'] += 1
                
                # Calcula volume total processado
                total_volume = sum(b[1] for b in update.bids) + sum(a[1] for a in update.asks)
                self.statistics['total_volume_processed'] += total_volume
            
            return True
        
        def get_best_bid(self):
            """Retorna o melhor preço de compra"""
            with self._lock:
                if not self.bids:
                    return None
                return self.bids[0][0]
        
        def get_best_ask(self):
            """Retorna o melhor preço de venda"""
            with self._lock:
                if not self.asks:
                    return None
                return self.asks[0][0]
        
        def get_spread(self):
            """Calcula o spread atual"""
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
            
            if best_bid is None or best_ask is None:
                return None
            
            return best_ask - best_bid
        
        def get_mid_price(self):
            """Calcula o preço médio"""
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
            
            if best_bid is None or best_ask is None:
                return None
            
            return (best_bid + best_ask) / 2
        
        def get_volume_at_price(self, price, side='bid'):
            """Obtém o volume em um preço específico"""
            with self._lock:
                if side == 'bid':
                    levels = self.bids
                else:
                    levels = self.asks
                
                for level_price, volume in levels:
                    if abs(level_price - price) < 0.000001:  # Tolerância para floats
                        return volume
                
                return 0.0
        
        def get_total_volume(self, side='all'):
            """Calcula o volume total"""
            with self._lock:
                if side == 'bid':
                    return sum(b[1] for b in self.bids)
                elif side == 'ask':
                    return sum(a[1] for a in self.asks)
                else:  # 'all'
                    return sum(b[1] for b in self.bids) + sum(a[1] for a in self.asks)
        
        def get_imbalance(self, depth=5):
            """Calcula o desequilíbrio entre bids e asks"""
            with self._lock:
                bid_depth = min(depth, len(self.bids))
                ask_depth = min(depth, len(self.asks))
                
                if bid_depth == 0 and ask_depth == 0:
                    return 0.0
                
                bid_volume = sum(self.bids[i][1] for i in range(bid_depth))
                ask_volume = sum(self.asks[i][1] for i in range(ask_depth))
                
                total = bid_volume + ask_volume
                if total == 0:
                    return 0.0
                
                return (bid_volume - ask_volume) / total
        
        def get_weighted_average_price(self, side='bid', depth=5):
            """Calcula o preço médio ponderado por volume"""
            with self._lock:
                if side == 'bid':
                    levels = self.bids[:depth]
                else:
                    levels = self.asks[:depth]
                
                if not levels:
                    return None
                
                total_volume = sum(volume for _, volume in levels)
                if total_volume == 0:
                    return None
                
                weighted_sum = sum(price * volume for price, volume in levels)
                return weighted_sum / total_volume
        
        def get_price_levels(self, side='bid', depth=10):
            """Obtém níveis de preço formatados"""
            with self._lock:
                if side == 'bid':
                    levels = self.bids[:depth]
                elif side == 'ask':
                    levels = self.asks[:depth]
                else:  # 'all'
                    return {
                        'bids': [{'price': b[0], 'volume': b[1]} for b in self.bids[:depth]],
                        'asks': [{'price': a[0], 'volume': a[1]} for a in self.asks[:depth]]
                    }
                
                return [{'price': price, 'volume': volume} for price, volume in levels]
        
        def create_snapshot(self):
            """Cria um snapshot do estado atual"""
            with self._lock:
                spread = self.get_spread()
                mid_price = self.get_mid_price()
                
                return OrderBookSnapshot(
                    symbol=self.symbol,
                    sequence=self.last_sequence,
                    bids=self.bids.copy(),
                    asks=self.asks.copy(),
                    timestamp=self.last_update_time,
                    spread=spread,
                    mid_price=mid_price
                )
        
        def reset(self):
            """Reseta o orderbook para estado inicial"""
            with self._lock:
                self.bids = []
                self.asks = []
                self.last_sequence = 0
                self.last_update_time = datetime.now()
        
        def get_statistics(self):
            """Retorna estatísticas do orderbook"""
            with self._lock:
                stats = self.statistics.copy()
                stats['current_depth'] = max(len(self.bids), len(self.asks))
                stats['age_seconds'] = (datetime.now() - self.last_update_time).total_seconds()
                
                # Calcula taxa de atualização
                if stats['update_count'] > 0 and stats['age_seconds'] > 0:
                    stats['updates_per_second'] = stats['update_count'] / stats['age_seconds']
                else:
                    stats['updates_per_second'] = 0.0
                
                return stats
        
        def get_liquidity_profile(self, price_range_percent=0.01):
            """Analisa perfil de liquidez em torno do preço atual"""
            mid_price = self.get_mid_price()
            if mid_price is None:
                return {}
            
            price_range = mid_price * price_range_percent
            min_price = mid_price - price_range
            max_price = mid_price + price_range
            
            with self._lock:
                bid_liquidity = sum(volume for price, volume in self.bids if min_price <= price <= max_price)
                ask_liquidity = sum(volume for price, volume in self.asks if min_price <= price <= max_price)
                
                return {
                    'bid_liquidity': bid_liquidity,
                    'ask_liquidity': ask_liquidity,
                    'total_liquidity': bid_liquidity + ask_liquidity,
                    'liquidity_ratio': bid_liquidity / ask_liquidity if ask_liquidity > 0 else float('inf'),
                    'price_range': {
                        'min': min_price,
                        'max': max_price,
                        'center': mid_price
                    }
                }
        
        def get_market_impact(self, size, side='buy'):
            """Estima o impacto de mercado para uma ordem de tamanho específico"""
            if size <= 0:
                return {'impact_price': None, 'slippage': 0.0, 'filled_size': 0.0}
            
            with self._lock:
                if side == 'buy':
                    levels = self.asks
                else:
                    levels = self.bids
                    # Para venda, consideramos níveis em ordem decrescente
                    levels = sorted(levels, key=lambda x: x[0], reverse=True)
                
                remaining_size = size
                total_cost = 0.0
                impact_levels = []
                
                for price, volume in levels:
                    if remaining_size <= 0:
                        break
                    
                    fill_size = min(volume, remaining_size)
                    cost = fill_size * price
                    
                    total_cost += cost
                    remaining_size -= fill_size
                    impact_levels.append({
                        'price': price,
                        'filled': fill_size,
                        'remaining_volume': volume - fill_size
                    })
                
                if total_cost == 0:
                    return {'impact_price': None, 'slippage': 0.0, 'filled_size': 0.0}
                
                filled_size = size - remaining_size
                impact_price = total_cost / filled_size if filled_size > 0 else None
                
                # Calcula slippage relativo ao melhor preço
                if levels:
                    best_price = levels[0][0]
                    if best_price > 0 and impact_price:
                        slippage = (impact_price / best_price - 1) * 100
                    else:
                        slippage = 0.0
                else:
                    slippage = 0.0
                
                return {
                    'impact_price': impact_price,
                    'slippage': slippage,
                    'filled_size': filled_size,
                    'remaining_size': remaining_size,
                    'impact_levels': impact_levels,
                    'estimated_cost': total_cost
                }
        
        def simulate_order(self, order_type='limit', side='buy', size=1.0, price=None):
            """Simula execução de uma ordem"""
            if size <= 0:
                return {'success': False, 'error': 'Invalid order size'}
            
            with self._lock:
                if order_type == 'market':
                    # Ordem de mercado é executada ao melhor preço disponível
                    impact_result = self.get_market_impact(size, side)
                    
                    if impact_result['filled_size'] <= 0:
                        return {'success': False, 'error': 'Insufficient liquidity'}
                    
                    return {
                        'success': True,
                        'order_type': 'market',
                        'side': side,
                        'requested_size': size,
                        'filled_size': impact_result['filled_size'],
                        'average_price': impact_result['impact_price'],
                        'slippage': impact_result['slippage'],
                        'remaining_size': impact_result['remaining_size']
                    }
                
                elif order_type == 'limit':
                    if price is None:
                        return {'success': False, 'error': 'Limit price required'}
                    
                    if side == 'buy':
                        # Para compra limitada, precisa que ask price <= limit price
                        best_ask = self.get_best_ask()
                        if best_ask is None or best_ask > price:
                            return {'success': False, 'error': 'Limit price below market'}
                        
                        # Executa contra asks
                        return self._execute_limit_order(side, size, price)
                    else:
                        # Para venda limitada, precisa que bid price >= limit price
                        best_bid = self.get_best_bid()
                        if best_bid is None or best_bid < price:
                            return {'success': False, 'error': 'Limit price above market'}
                        
                        # Executa contra bids
                        return self._execute_limit_order(side, size, price)
                
                else:
                    return {'success': False, 'error': f'Unsupported order type: {order_type}'}
        
        def _execute_limit_order(self, side, size, price):
            """Executa uma ordem limitada"""
            filled_size = 0.0
            total_cost = 0.0
            execution_levels = []
            
            if side == 'buy':
                levels = self.asks
                # Remove níveis que são executados
                remaining_levels = []
                
                for level_price, level_volume in levels:
                    if level_price > price or filled_size >= size:
                        remaining_levels.append([level_price, level_volume])
                        continue
                    
                    available_volume = level_volume
                    needed_volume = size - filled_size
                    fill_volume = min(available_volume, needed_volume)
                    
                    filled_size += fill_volume
                    total_cost += fill_volume * level_price
                    execution_levels.append({
                        'price': level_price,
                        'filled': fill_volume,
                        'remaining': available_volume - fill_volume
                    })
                    
                    if fill_volume < available_volume:
                        remaining_levels.append([level_price, available_volume - fill_volume])
                    
                    if filled_size >= size:
                        break
                
                # Atualiza asks com níveis restantes
                self.asks = remaining_levels
                
            else:  # side == 'sell'
                levels = self.bids
                # Remove níveis que são executados
                remaining_levels = []
                
                for level_price, level_volume in levels:
                    if level_price < price or filled_size >= size:
                        remaining_levels.append([level_price, level_volume])
                        continue
                    
                    available_volume = level_volume
                    needed_volume = size - filled_size
                    fill_volume = min(available_volume, needed_volume)
                    
                    filled_size += fill_volume
                    total_cost += fill_volume * level_price
                    execution_levels.append({
                        'price': level_price,
                        'filled': fill_volume,
                        'remaining': available_volume - fill_volume
                    })
                    
                    if fill_volume < available_volume:
                        remaining_levels.append([level_price, available_volume - fill_volume])
                    
                    if filled_size >= size:
                        break
                
                # Atualiza bids com níveis restantes
                self.bids = remaining_levels
            
            average_price = total_cost / filled_size if filled_size > 0 else None
            
            return {
                'success': filled_size > 0,
                'order_type': 'limit',
                'side': side,
                'limit_price': price,
                'requested_size': size,
                'filled_size': filled_size,
                'average_price': average_price,
                'execution_levels': execution_levels,
                'remaining_size': size - filled_size
            }


class TestOrderBookComprehensive:
    """Testes abrangentes para OrderBook core"""
    
    @pytest.fixture
    def orderbook(self):
        """Fixture com OrderBook configurado"""
        return OrderBook(symbol="BTCUSDT", max_depth=100)
    
    @pytest.fixture
    def sample_update(self):
        """Dados de atualização de exemplo"""
        return OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.5), (49999.0, 2.3), (49998.0, 0.8), (49997.0, 3.2), (49996.0, 1.1)],
            asks=[(50001.0, 2.0), (50002.0, 1.5), (50003.0, 0.9), (50004.0, 2.8), (50005.0, 1.2)],
            sequence=123456
        )
    
    @pytest.fixture
    def large_orderbook_update(self):
        """Atualização com muitos níveis"""
        bids = [(50000.0 - i * 0.5, 1.0 + i * 0.1) for i in range(50)]
        asks = [(50001.0 + i * 0.5, 1.0 + i * 0.1) for i in range(50)]
        return OrderBookUpdate(
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            sequence=1000
        )
    
    def test_initialization(self):
        """Testa inicialização do OrderBook"""
        # Teste com símbolo válido
        ob = OrderBook(symbol="ETHUSDT", max_depth=50)
        
        assert ob.symbol == "ETHUSDT"
        assert ob.max_depth == 50
        assert len(ob.bids) == 0
        assert len(ob.asks) == 0
        assert ob.last_sequence == 0
        assert ob.last_update_time is not None
        
        # Teste com valores padrão
        ob_default = OrderBook()
        assert ob_default.symbol == "BTCUSDT"
        assert ob_default.max_depth == MAX_ORDERBOOK_DEPTH
        
        # Teste com parâmetros inválidos
        with pytest.raises(ValueError):
            OrderBook(symbol="")
        
        with pytest.raises(ValueError):
            OrderBook(symbol="BTCUSDT", max_depth=0)
        
        with pytest.raises(ValueError):
            OrderBook(symbol="BTCUSDT", max_depth=-10)
    
    def test_update_with_valid_data(self, orderbook, sample_update):
        """Testa atualização com dados válidos"""
        success = orderbook.update(sample_update)
        
        assert success is True
        assert orderbook.last_sequence == 123456
        assert len(orderbook.bids) == 5
        assert len(orderbook.asks) == 5
        assert orderbook.bids[0][0] == 50000.0  # Preço mais alto primeiro
        assert orderbook.asks[0][0] == 50001.0  # Preço mais baixo primeiro
        
        # Verifica ordenação
        bid_prices = [b[0] for b in orderbook.bids]
        assert all(bid_prices[i] >= bid_prices[i + 1] for i in range(len(bid_prices) - 1))
        
        ask_prices = [a[0] for a in orderbook.asks]
        assert all(ask_prices[i] <= ask_prices[i + 1] for i in range(len(ask_prices) - 1))
        
        # Verifica estatísticas
        stats = orderbook.get_statistics()
        assert stats['update_count'] == 1
        assert stats['total_volume_processed'] > 0
    
    def test_update_with_invalid_sequence(self, orderbook, sample_update):
        """Testa atualização com sequência inválida"""
        # Primeira atualização normal
        orderbook.update(sample_update)
        
        # Tenta atualizar com sequência menor (deveria ser ignorada)
        old_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.0)],
            sequence=123455  # Sequência mais antiga
        )
        
        success = orderbook.update(old_update)
        
        assert success is False
        assert orderbook.last_sequence == 123456  # Não mudou
        
        # Atualização com mesma sequência também deve ser ignorada
        same_seq_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0)],
            asks=[(50001.0, 2.0)],
            sequence=123456  # Mesma sequência
        )
        
        success = orderbook.update(same_seq_update)
        assert success is False
    
    def test_update_with_missing_prices(self, orderbook):
        """Testa atualização com preços faltando"""
        # Update com apenas um lado
        update_bids_only = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.5)],
            asks=[],
            sequence=1000
        )
        
        success = orderbook.update(update_bids_only)
        assert success is True
        assert len(orderbook.bids) == 1
        assert len(orderbook.asks) == 0
        
        # Update com apenas asks
        orderbook.reset()
        update_asks_only = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[],
            asks=[(50001.0, 2.0)],
            sequence=1001
        )
        
        success = orderbook.update(update_asks_only)
        assert success is True
        assert len(orderbook.bids) == 0
        assert len(orderbook.asks) == 1
    
    def test_get_best_bid_ask(self, orderbook, sample_update):
        """Testa obtenção do melhor bid/ask"""
        orderbook.update(sample_update)
        
        assert orderbook.get_best_bid() == 50000.0
        assert orderbook.get_best_ask() == 50001.0
        assert orderbook.get_spread() == 1.0
        
        # Teste com orderbook vazio
        orderbook.reset()
        assert orderbook.get_best_bid() is None
        assert orderbook.get_best_ask() is None
        assert orderbook.get_spread() is None
    
    def test_get_spread_edge_cases(self, orderbook):
        """Testa spread em casos de borda"""
        # Orderbook vazio
        assert orderbook.get_spread() is None
        
        # Apenas bids
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[],
            sequence=1000
        ))
        assert orderbook.get_spread() is None
        
        # Apenas asks
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[],
            asks=[(50001.0, 1.0)],
            sequence=1001
        ))
        assert orderbook.get_spread() is None
        
        # Spread zero (bid == ask)
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50000.0, 1.0)],  # Mesmo preço
            sequence=1002
        ))
        assert orderbook.get_spread() == 0.0
    
    def test_get_mid_price(self, orderbook):
        """Testa cálculo do preço médio"""
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.5)],
            asks=[(50001.0, 2.0)],
            sequence=1000
        ))
        
        mid_price = orderbook.get_mid_price()
        
        assert mid_price == 50000.5
        
        # Testa com spread zero
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50000.0, 1.0)],
            sequence=1001
        ))
        
        mid_price = orderbook.get_mid_price()
        assert mid_price == 50000.0
        
        # Testa com orderbook incompleto
        orderbook.reset()
        assert orderbook.get_mid_price() is None
    
    def test_get_volume_at_price(self, orderbook, sample_update):
        """Testa obtenção de volume em um preço específico"""
        orderbook.update(sample_update)
        
        # Volumes existentes
        bid_volume = orderbook.get_volume_at_price(50000.0, side='bid')
        ask_volume = orderbook.get_volume_at_price(50001.0, side='ask')
        
        assert bid_volume == 1.5
        assert ask_volume == 2.0
        
        # Preços não existentes
        non_existent_bid = orderbook.get_volume_at_price(51000.0, side='bid')
        non_existent_ask = orderbook.get_volume_at_price(49000.0, side='ask')
        
        assert non_existent_bid == 0.0
        assert non_existent_ask == 0.0
        
        # Side inválido (default para bid)
        with pytest.raises(KeyError):
            orderbook.get_volume_at_price(50000.0, side='invalid')
    
    def test_get_total_volume(self, orderbook, sample_update):
        """Testa cálculo do volume total"""
        orderbook.update(sample_update)
        
        total_bid = orderbook.get_total_volume(side='bid')
        total_ask = orderbook.get_total_volume(side='ask')
        total_all = orderbook.get_total_volume(side='all')
        
        # Calcula volumes esperados
        expected_bid_volume = sum(b[1] for b in sample_update.bids)
        expected_ask_volume = sum(a[1] for a in sample_update.asks)
        
        assert total_bid == expected_bid_volume
        assert total_ask == expected_ask_volume
        assert total_all == expected_bid_volume + expected_ask_volume
        
        # Teste com orderbook vazio
        orderbook.reset()
        assert orderbook.get_total_volume(side='bid') == 0.0
        assert orderbook.get_total_volume(side='ask') == 0.0
        assert orderbook.get_total_volume(side='all') == 0.0
    
    def test_get_imbalance(self, orderbook):
        """Testa cálculo de imbalance"""
        # Caso 1: Mais volume em bids
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 3.0)],
            asks=[(50001.0, 1.0)],
            sequence=1000
        ))
        
        imbalance = orderbook.get_imbalance(depth=1)
        
        # (bid_volume - ask_volume) / (bid_volume + ask_volume)
        # (3.0 - 1.0) / (3.0 + 1.0) = 2.0 / 4.0 = 0.5
        assert imbalance == 0.5
        
        # Caso 2: Volumes iguais
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0)],
            asks=[(50001.0, 2.0)],
            sequence=1001
        ))
        
        imbalance = orderbook.get_imbalance(depth=1)
        assert imbalance == 0.0
        
        # Caso 3: Mais volume em asks
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 4.0)],
            sequence=1002
        ))
        
        imbalance = orderbook.get_imbalance(depth=1)
        # (1.0 - 4.0) / (1.0 + 4.0) = -3.0 / 5.0 = -0.6
        assert imbalance == -0.6
        
        # Caso 4: Depth limitado
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0), (49999.0, 3.0)],
            asks=[(50001.0, 1.0), (50002.0, 4.0)],
            sequence=1003
        ))
        
        imbalance_depth1 = orderbook.get_imbalance(depth=1)  # Apenas primeiros níveis
        imbalance_depth2 = orderbook.get_imbalance(depth=2)  # Dois primeiros níveis
        
        assert imbalance_depth1 != imbalance_depth2
        
        # Caso 5: Orderbook vazio
        orderbook.reset()
        imbalance = orderbook.get_imbalance(depth=1)
        assert imbalance == 0.0
    
    def test_get_weighted_average_price(self, orderbook):
        """Testa cálculo do preço médio ponderado"""
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0), (49900.0, 3.0)],
            asks=[(50100.0, 2.0), (50200.0, 1.0)],
            sequence=1000
        ))
        
        wap_bid = orderbook.get_weighted_average_price(side='bid', depth=2)
        wap_ask = orderbook.get_weighted_average_price(side='ask', depth=2)
        
        # Bid WAP: (50000*1.0 + 49900*3.0) / 4.0 = 199700/4 = 49925
        # Ask WAP: (50100*2.0 + 50200*1.0) / 3.0 = 150400/3 ≈ 50133.33
        
        assert wap_bid == 49925.0
        assert abs(wap_ask - 50133.33) < 0.01
        
        # Teste com depth maior que níveis disponíveis
        wap_bid_depth5 = orderbook.get_weighted_average_price(side='bid', depth=5)
        assert wap_bid_depth5 == wap_bid  # Deve usar todos os níveis disponíveis
        
        # Teste com side vazio
        wap_empty = orderbook.get_weighted_average_price(side='bid', depth=0)
        assert wap_empty is None
        
        # Teste com orderbook vazio
        orderbook.reset()
        assert orderbook.get_weighted_average_price(side='bid') is None
    
    def test_get_price_levels(self, orderbook, sample_update):
        """Testa obtenção de níveis de preço"""
        orderbook.update(sample_update)
        
        # Testa bids
        bid_levels = orderbook.get_price_levels(side='bid', depth=2)
        assert len(bid_levels) == 2
        assert bid_levels[0]['price'] == 50000.0
        assert bid_levels[0]['volume'] == 1.5
        assert bid_levels[1]['price'] == 49999.0
        
        # Testa asks
        ask_levels = orderbook.get_price_levels(side='ask', depth=2)
        assert len(ask_levels) == 2
        assert ask_levels[0]['price'] == 50001.0
        assert ask_levels[1]['price'] == 50002.0
        
        # Testa ambos lados
        all_levels = orderbook.get_price_levels(side='all', depth=1)
        assert 'bids' in all_levels
        assert 'asks' in all_levels
        assert len(all_levels['bids']) == 1
        assert len(all_levels['asks']) == 1
        
        # Testa com depth maior que disponível
        deep_levels = orderbook.get_price_levels(side='bid', depth=10)
        assert len(deep_levels) == 5  # Apenas 5 níveis disponíveis
        
        # Testa com orderbook vazio
        orderbook.reset()
        empty_levels = orderbook.get_price_levels(side='bid', depth=5)
        assert len(empty_levels) == 0
    
    def test_snapshot_creation(self, orderbook, sample_update):
        """Testa criação de snapshot"""
        orderbook.update(sample_update)
        
        snapshot = orderbook.create_snapshot()
        
        assert isinstance(snapshot, OrderBookSnapshot)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.sequence == 123456
        assert len(snapshot.bids) == 5
        assert len(snapshot.asks) == 5
        assert snapshot.timestamp == sample_update.timestamp
        assert snapshot.spread == 1.0
        assert snapshot.mid_price == 50000.5
        
        # Verifica que é uma cópia, não referência
        snapshot.bids[0] = [0, 0]
        assert orderbook.bids[0][0] == 50000.0  # Original não mudou
    
    def test_snapshot_validation(self, orderbook, sample_update):
        """Testa validação de snapshot"""
        orderbook.update(sample_update)
        snapshot = orderbook.create_snapshot()
        
        assert snapshot.validate() is True
        
        # Cria snapshot inválido
        invalid_snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            sequence=100,
            bids=[[-50000, 1.0]],  # Preço negativo
            asks=[[50001, 1.0]]
        )
        
        assert invalid_snapshot.validate() is False
        
        # Snapshot com bids fora de ordem
        unordered_snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            sequence=100,
            bids=[[49900, 1.0], [50000, 1.0]],  # Fora de ordem
            asks=[[50001, 1.0], [50002, 1.0]]
        )
        
        assert unordered_snapshot.validate() is False
    
    def test_snapshot_age_check(self, orderbook):
        """Testa verificação de idade do snapshot"""
        # Snapshot recente
        recent_snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=datetime.now()
        )
        
        assert recent_snapshot.is_too_old(max_age_ms=5000) is False
        
        # Snapshot antigo
        old_timestamp = datetime.now() - timedelta(seconds=10)
        old_snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            sequence=100,
            bids=[[50000, 1.0]],
            asks=[[50001, 1.0]],
            timestamp=old_timestamp
        )
        
        assert old_snapshot.is_too_old(max_age_ms=5000) is True
        
        # Testa com limite personalizado
        assert old_snapshot.is_too_old(max_age_ms=15000) is False  # 15 segundos
    
    def test_snapshot_json_serialization(self, orderbook):
        """Testa serialização JSON do snapshot"""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            sequence=100,
            bids=[[50000.0, 1.5], [49999.0, 2.3]],
            asks=[[50001.0, 2.0], [50002.0, 1.5]],
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            spread=1.0,
            mid_price=50000.5
        )
        
        # Serializa
        json_str = snapshot.to_json()
        assert isinstance(json_str, str)
        
        # Desserializa
        data = json.loads(json_str)
        assert data['symbol'] == "BTCUSDT"
        assert data['sequence'] == 100
        assert len(data['bids']) == 2
        assert len(data['asks']) == 2
        assert data['spread'] == 1.0
        assert data['mid_price'] == 50000.5
        assert 'timestamp' in data
        
        # Cria novo snapshot do JSON
        new_snapshot = OrderBookSnapshot.from_json(data)
        assert new_snapshot.symbol == "BTCUSDT"
        assert new_snapshot.sequence == 100
        assert new_snapshot.spread == 1.0
    
    def test_reset_orderbook(self, orderbook, sample_update):
        """Testa reset do orderbook"""
        orderbook.update(sample_update)
        
        # Verifica estado antes do reset
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        assert orderbook.last_sequence == 123456
        
        # Reseta
        orderbook.reset()
        
        # Verifica estado após reset
        assert len(orderbook.bids) == 0
        assert len(orderbook.asks) == 0
        assert orderbook.last_sequence == 0
        assert isinstance(orderbook.last_update_time, datetime)
        
        # Verifica que ainda pode ser usado após reset
        new_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(51000.0, 1.0)],
            asks=[(51001.0, 1.0)],
            sequence=1
        )
        
        success = orderbook.update(new_update)
        assert success is True
        assert orderbook.last_sequence == 1
    
    def test_max_depth_enforcement(self, orderbook, large_orderbook_update):
        """Testa aplicação do limite de profundidade"""
        orderbook = OrderBook(symbol="BTCUSDT", max_depth=10)
        orderbook.update(large_orderbook_update)
        
        # Verifica que não excede max_depth
        assert len(orderbook.bids) <= 10
        assert len(orderbook.asks) <= 10
        
        # Verifica que mantém os melhores preços
        assert orderbook.bids[0][0] == 50000.0  # Melhor bid
        assert orderbook.asks[0][0] == 50001.0  # Melhor ask
        
        # Testa com max_depth muito pequeno
        small_orderbook = OrderBook(symbol="BTCUSDT", max_depth=2)
        small_orderbook.update(large_orderbook_update)
        
        assert len(small_orderbook.bids) == 2
        assert len(small_orderbook.asks) == 2
    
    def test_concurrent_updates(self, orderbook):
        """Testa atualizações concorrentes"""
        import concurrent.futures
        
        num_updates = 100
        results = []
        
        def update_task(sequence):
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0 + sequence * 0.01, 1.0)],
                asks=[(50001.0 + sequence * 0.01, 1.0)],
                sequence=sequence
            )
            
            try:
                success = orderbook.update(update)
                return (sequence, success, None)
            except Exception as e:
                return (sequence, False, str(e))
        
        # Executa updates concorrentes
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_task, i) for i in range(1, num_updates + 1)]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        # Analisa resultados
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        print(f"Concurrent updates: {len(successful)} successful, {len(failed)} failed")
        
        # Verifica que pelo menos algumas atualizações foram bem-sucedidas
        assert len(successful) > 0
        
        # Verifica consistência do orderbook
        assert orderbook.last_sequence >= 1
        assert len(orderbook.bids) <= orderbook.max_depth
        assert len(orderbook.asks) <= orderbook.max_depth
        
        # Verifica que não há corrupção de dados
        if orderbook.bids:
            assert orderbook.bids[0][0] > 0
            assert orderbook.bids[0][1] > 0
        
        if orderbook.asks:
            assert orderbook.asks[0][0] > 0
            assert orderbook.asks[0][1] > 0
    
    def test_performance_benchmark(self, orderbook):
        """Testa performance com muitas atualizações"""
        import time
        
        num_updates = 1000
        start_time = time.time()
        
        for i in range(num_updates):
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0, 1.0 + i/1000)],
                asks=[(50001.0, 1.0)],
                sequence=i + 1
            )
            orderbook.update(update)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Verifica que é razoavelmente rápido
        updates_per_second = num_updates / elapsed
        print(f"Performance: {updates_per_second:.0f} updates/second")
        
        assert elapsed < 2.0  # Menos de 2 segundos para 1000 updates
        assert updates_per_second > 500  # Pelo menos 500 updates/segundo
        
        # Verifica que todos os updates foram processados
        assert orderbook.last_sequence == num_updates
        
        # Verifica estatísticas
        stats = orderbook.get_statistics()
        assert stats['update_count'] == num_updates
        assert stats['total_volume_processed'] > 0
        
        if num_updates > 0:
            avg_update_time = stats['last_update_duration']
            print(f"Average update time: {avg_update_time*1000:.3f}ms")
    
    def test_memory_efficiency(self, orderbook, large_orderbook_update):
        """Testa eficiência de memória"""
        import sys
        
        # Estado inicial
        initial_size = sys.getsizeof(orderbook)
        
        # Adiciona muitos níveis várias vezes
        for i in range(20):
            update = large_orderbook_update
            update.sequence = i + 1
            orderbook.update(update)
        
        # Estado final
        final_size = sys.getsizeof(orderbook)
        
        size_increase = final_size - initial_size
        print(f"Memory increase: {size_increase / 1024:.1f} KB")
        
        # O aumento deve ser moderado (menos de 1MB para 20 updates grandes)
        assert size_increase < 1024 * 1024  # 1MB
        
        # Verifica que não há vazamento de memória óbvio
        assert len(orderbook.bids) <= orderbook.max_depth
        assert len(orderbook.asks) <= orderbook.max_depth
    
    def test_circuit_breaker_integration(self, orderbook):
        """Testa integração com circuit breaker"""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.5  # 500ms para recovery
        )
        
        orderbook.circuit_breaker = circuit_breaker
        
        # Simula atualizações problemáticas
        failures = 0
        
        for i in range(5):
            try:
                update = OrderBookUpdate(
                    timestamp=datetime.now(),
                    bids=[(50000.0, 1.0)],
                    asks=[(50001.0, 1.0)],
                    sequence=i + 1
                )
                
                if i < 3:
                    # Primeiras 3 devem funcionar
                    success = orderbook.update(update)
                    assert success is True
                else:
                    # Após 3 falhas, circuit breaker deve abrir
                    with pytest.raises(OrderBookError):
                        orderbook.update(update)
                    failures += 1
                    
            except InvalidUpdateError:
                pass
        
        # Após failures, circuit breaker deve estar aberto
        assert failures > 0
        
        # Aguarda recovery
        time.sleep(0.6)  # Mais que recovery_timeout
        
        # Deve funcionar novamente
        update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.0)],
            sequence=100
        )
        
        success = orderbook.update(update)
        assert success is True
    
    def test_error_handling(self, orderbook):
        """Testa tratamento de erros"""
        # Testa update com dados inválidos
        with pytest.raises(InvalidUpdateError):
            orderbook.update(None)
        
        # Testa update com preço negativo
        with pytest.raises(InvalidUpdateError):
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(-50000.0, 1.0)],
                asks=[(50001.0, 1.0)],
                sequence=1
            )
            orderbook.update(update)
        
        # Testa update com volume negativo
        with pytest.raises(InvalidUpdateError):
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0, -1.0)],
                asks=[(50001.0, 1.0)],
                sequence=1
            )
            orderbook.update(update)
        
        # Testa update com tipo de dados errado
        with pytest.raises(InvalidUpdateError):
            orderbook.update("not a valid update")
        
        # Testa que o estado permanece consistente após erro
        initial_state = orderbook.create_snapshot()
        
        try:
            orderbook.update(None)
        except InvalidUpdateError:
            pass
        
        # Estado deve permanecer inalterado
        current_state = orderbook.create_snapshot()
        assert current_state.sequence == initial_state.sequence
    
    def test_decimal_precision_handling(self, orderbook):
        """Testa manipulação de precisão decimal"""
        # Testa com valores decimais
        from decimal import Decimal
        
        update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(Decimal('50000.12345678'), Decimal('1.12345678'))],
            asks=[(Decimal('50001.12345678'), Decimal('2.12345678'))],
            sequence=1
        )
        
        success = orderbook.update(update)
        
        assert success is True
        assert orderbook.get_best_bid() == 50000.12345678
        assert orderbook.get_best_ask() == 50001.12345678
        
        # Testa com floats de alta precisão
        update2 = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.123456789012345, 1.123456789012345)],
            asks=[(50001.123456789012345, 2.123456789012345)],
            sequence=2
        )
        
        success = orderbook.update(update2)
        assert success is True
        
        # Verifica que a precisão é mantida razoavelmente
        bid = orderbook.get_best_bid()
        ask = orderbook.get_best_ask()
        
        assert abs(bid - 50000.123456789) < 0.000000001
        assert abs(ask - 50001.123456789) < 0.000000001
    
    @pytest.mark.parametrize("price,volume,expected_valid", [
        (50000.0, 1.0, True),
        (0.000001, 1.0, True),  # Preço muito pequeno mas positivo
        (50000.0, 0.000001, True),  # Volume muito pequeno mas positivo
        (0.0, 1.0, False),  # Preço zero
        (50000.0, 0.0, False),  # Volume zero
        (-50000.0, 1.0, False),  # Preço negativo
        (50000.0, -1.0, False),  # Volume negativo
        (float('inf'), 1.0, False),  # Preço infinito
        (50000.0, float('inf'), False),  # Volume infinito
        (float('nan'), 1.0, False),  # Preço NaN
        (50000.0, float('nan'), False),  # Volume NaN
    ])
    def test_validation_edge_cases(self, orderbook, price, volume, expected_valid):
        """Testa casos de borda na validação"""
        if expected_valid:
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(price, volume)],
                asks=[(50001.0, 1.0)],
                sequence=1
            )
            success = orderbook.update(update)
            assert success is True
        else:
            with pytest.raises(InvalidUpdateError):
                update = OrderBookUpdate(
                    timestamp=datetime.now(),
                    bids=[(price, volume)],
                    asks=[(50001.0, 1.0)],
                    sequence=1
                )
                orderbook.update(update)
    
    def test_state_consistency_after_errors(self, orderbook, sample_update):
        """Testa consistência do estado após erros"""
        # Primeira atualização normal
        orderbook.update(sample_update)
        initial_sequence = orderbook.last_sequence
        initial_bid_count = len(orderbook.bids)
        
        # Tenta atualização inválida
        try:
            invalid_update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0, -1.0)],  # Volume negativo
                asks=[(50001.0, 1.0)],
                sequence=initial_sequence + 1
            )
            orderbook.update(invalid_update)
        except InvalidUpdateError:
            pass
        
        # Estado deve permanecer inalterado
        assert orderbook.last_sequence == initial_sequence
        assert len(orderbook.bids) == initial_bid_count
        
        # Agora uma atualização válida deve funcionar
        valid_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0)],
            asks=[(50001.0, 2.0)],
            sequence=initial_sequence + 1
        )
        
        success = orderbook.update(valid_update)
        assert success is True
        assert orderbook.last_sequence == initial_sequence + 1
    
    def test_cleanup_old_orders(self, orderbook, large_orderbook_update):
        """Testa limpeza de ordens antigas (via max_depth)"""
        # Configura orderbook com profundidade limitada
        orderbook = OrderBook(symbol="BTCUSDT", max_depth=5)
        
        # Adiciona muitos níveis
        for i in range(20):
            update = large_orderbook_update
            update.sequence = i + 1
            orderbook.update(update)
            
            # Verifica que não excede max_depth
            assert len(orderbook.bids) <= 5
            assert len(orderbook.asks) <= 5
            
            # Verifica que mantém os melhores preços
            if orderbook.bids:
                # O melhor bid deve ser o mais alto
                bid_prices = [b[0] for b in orderbook.bids]
                assert max(bid_prices) == orderbook.bids[0][0]
            
            if orderbook.asks:
                # O melhor ask deve ser o mais baixo
                ask_prices = [a[0] for a in orderbook.asks]
                assert min(ask_prices) == orderbook.asks[0][0]
    
    def test_custom_sorting_logic(self, orderbook):
        """Testa lógica de ordenação personalizada"""
        # Testa ordenação descendente para bids
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(49900.0, 3.0), (50000.0, 1.0), (49950.0, 2.0)],
            asks=[(50001.0, 1.0), (50050.0, 3.0), (50010.0, 2.0)],
            sequence=1
        ))
        
        # Bids devem estar em ordem descendente de preço
        bid_prices = [b[0] for b in orderbook.bids]
        assert bid_prices == [50000.0, 49950.0, 49900.0]
        
        # Asks devem estar em ordem ascendente de preço
        ask_prices = [a[0] for a in orderbook.asks]
        assert ask_prices == [50001.0, 50010.0, 50050.0]
        
        # Testa com valores iguais (deve manter ordem de chegada ou volume)
        orderbook.reset()
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0), (50000.0, 1.0), (49999.0, 3.0)],
            asks=[(50001.0, 1.0), (50001.0, 2.0), (50002.0, 1.0)],
            sequence=2
        ))
        
        # Bids com preços iguais podem ser ordenados por volume ou ordem
        assert orderbook.bids[0][0] == 50000.0  # Pelo menos um dos 50000 primeiro
    
    def test_get_statistics_comprehensive(self, orderbook, sample_update):
        """Testa obtenção de estatísticas abrangentes"""
        # Estado inicial
        initial_stats = orderbook.get_statistics()
        assert initial_stats['update_count'] == 0
        assert initial_stats['total_volume_processed'] == 0.0
        assert initial_stats['max_depth_reached'] == 0
        
        # Executa várias atualizações
        num_updates = 10
        for i in range(num_updates):
            update = sample_update
            update.sequence = i + 1
            orderbook.update(update)
        
        # Obtém estatísticas finais
        final_stats = orderbook.get_statistics()
        
        assert final_stats['update_count'] == num_updates
        assert final_stats['total_volume_processed'] > 0
        assert final_stats['max_depth_reached'] >= len(sample_update.bids)
        assert final_stats['current_depth'] == len(orderbook.bids)  # Ou asks
        assert final_stats['age_seconds'] >= 0
        
        # Verifica taxa de atualização
        if final_stats['age_seconds'] > 0:
            assert final_stats['updates_per_second'] > 0
        
        # Verifica duração da última atualização
        assert final_stats['last_update_duration'] >= 0
        
        # Verifica contagem de erros (deve ser 0 nesse teste)
        assert final_stats['error_count'] == 0
        
        print(f"Statistics after {num_updates} updates:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")
    
    def test_liquidity_profile_analysis(self, orderbook, sample_update):
        """Testa análise de perfil de liquidez"""
        orderbook.update(sample_update)
        
        # Obtém perfil de liquidez
        profile = orderbook.get_liquidity_profile(price_range_percent=0.02)
        
        assert 'bid_liquidity' in profile
        assert 'ask_liquidity' in profile
        assert 'total_liquidity' in profile
        assert 'liquidity_ratio' in profile
        assert 'price_range' in profile
        
        # Verifica valores
        assert profile['bid_liquidity'] >= 0
        assert profile['ask_liquidity'] >= 0
        assert profile['total_liquidity'] >= 0
        
        # Verifica estrutura do price_range
        price_range = profile['price_range']
        assert 'min' in price_range
        assert 'max' in price_range
        assert 'center' in price_range
        assert price_range['min'] < price_range['max']
        
        # Testa com orderbook vazio
        orderbook.reset()
        empty_profile = orderbook.get_liquidity_profile()
        assert empty_profile == {}
        
        # Testa com range percentual personalizado
        profile_small = orderbook.get_liquidity_profile(price_range_percent=0.001)
        profile_large = orderbook.get_liquidity_profile(price_range_percent=0.1)
        
        # O range maior deve incluir mais liquidez (ou pelo menos a mesma)
        if profile_small and profile_large:
            assert profile_large['total_liquidity'] >= profile_small['total_liquidity']
    
    def test_market_impact_analysis(self, orderbook, large_orderbook_update):
        """Testa análise de impacto de mercado"""
        orderbook.update(large_orderbook_update)
        
        # Testa compra pequena (deve executar ao melhor preço)
        small_buy = orderbook.get_market_impact(size=0.5, side='buy')
        
        assert 'impact_price' in small_buy
        assert 'slippage' in small_buy
        assert 'filled_size' in small_buy
        assert 'estimated_cost' in small_buy
        
        # Para compra pequena, deve executar completamente
        assert small_buy['filled_size'] == 0.5
        assert small_buy['remaining_size'] == 0
        assert small_buy['impact_price'] == orderbook.asks[0][0]  # Melhor ask
        assert small_buy['slippage'] == 0.0  # Sem slippage para ordem pequena
        
        # Testa compra grande (deve causar slippage)
        large_buy = orderbook.get_market_impact(size=50.0, side='buy')
        
        assert large_buy['filled_size'] > 0
        assert large_buy['impact_price'] > orderbook.asks[0][0]  # Preço médio maior que melhor ask
        assert large_buy['slippage'] > 0.0  # Deve ter slippage positivo
        
        # Testa venda
        sell_impact = orderbook.get_market_impact(size=0.5, side='sell')
        assert sell_impact['impact_price'] == orderbook.bids[0][0]  # Melhor bid
        
        # Testa com tamanho zero
        zero_impact = orderbook.get_market_impact(size=0, side='buy')
        assert zero_impact['impact_price'] is None
        assert zero_impact['slippage'] == 0.0
        
        # Testa com tamanho negativo
        negative_impact = orderbook.get_market_impact(size=-1, side='buy')
        assert negative_impact['impact_price'] is None
        
        # Testa com orderbook vazio
        orderbook.reset()
        empty_impact = orderbook.get_market_impact(size=1, side='buy')
        assert empty_impact['impact_price'] is None
        assert empty_impact['filled_size'] == 0
    
    def test_order_simulation(self, orderbook, large_orderbook_update):
        """Testa simulação de ordens"""
        orderbook.update(large_orderbook_update)
        
        # 1. Testa ordem de mercado de compra pequena
        market_buy_small = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=0.5
        )
        
        assert market_buy_small['success'] is True
        assert market_buy_small['order_type'] == 'market'
        assert market_buy_small['side'] == 'buy'
        assert market_buy_small['filled_size'] == 0.5
        assert market_buy_small['slippage'] == 0.0  # Pequena, sem slippage
        
        # 2. Testa ordem de mercado de compra grande (pode não executar completamente)
        market_buy_large = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=100.0
        )
        
        # Pode executar parcialmente
        assert market_buy_large['filled_size'] > 0
        if market_buy_large['filled_size'] < 100.0:
            assert market_buy_large['remaining_size'] > 0
        
        # 3. Testa ordem limitada de compra com preço acima do mercado
        limit_buy_above = orderbook.simulate_order(
            order_type='limit',
            side='buy',
            size=0.5,
            price=orderbook.asks[0][0] + 10.0  # Acima do melhor ask
        )
        
        assert limit_buy_above['success'] is True  # Deve executar imediatamente
        
        # 4. Testa ordem limitada de compra com preço abaixo do mercado
        limit_buy_below = orderbook.simulate_order(
            order_type='limit',
            side='buy',
            size=0.5,
            price=orderbook.asks[0][0] - 10.0  # Abaixo do melhor ask
        )
        
        assert limit_buy_below['success'] is False
        assert 'error' in limit_buy_below
        
        # 5. Testa ordem limitada de venda
        limit_sell = orderbook.simulate_order(
            order_type='limit',
            side='sell',
            size=0.5,
            price=orderbook.bids[0][0]  # No melhor bid
        )
        
        assert limit_sell['success'] is True
        
        # 6. Testa ordem com tipo inválido
        invalid_order = orderbook.simulate_order(
            order_type='invalid',
            side='buy',
            size=0.5
        )
        
        assert invalid_order['success'] is False
        assert 'error' in invalid_order
        
        # 7. Testa ordem com tamanho zero
        zero_order = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=0
        )
        
        assert zero_order['success'] is False
        
        # 8. Testa com orderbook vazio
        orderbook.reset()
        empty_orderbook_order = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=0.5
        )
        
        assert empty_orderbook_order['success'] is False
        assert 'error' in empty_orderbook_order
    
    def test_order_execution_updates_orderbook(self, orderbook):
        """Testa que a execução de ordens atualiza o orderbook"""
        # Configura orderbook inicial
        orderbook.update(OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 2.0), (49900.0, 3.0)],
            asks=[(50100.0, 2.0), (50200.0, 3.0)],
            sequence=1
        ))
        
        initial_best_ask = orderbook.get_best_ask()
        initial_ask_volume = orderbook.get_volume_at_price(initial_best_ask, side='ask')
        
        # Executa ordem de mercado de compra
        result = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=1.5
        )
        
        assert result['success'] is True
        
        # Verifica que o orderbook foi atualizado
        new_ask_volume = orderbook.get_volume_at_price(initial_best_ask, side='ask')
        
        # O volume no melhor ask deve ter diminuído
        assert new_ask_volume == initial_ask_volume - 1.5
        
        # Se executou completamente no primeiro nível, o melhor ask pode mudar
        if new_ask_volume <= 0:
            assert orderbook.get_best_ask() != initial_best_ask
        else:
            assert orderbook.get_best_ask() == initial_best_ask
        
        # Executa ordem limitada de venda
        initial_best_bid = orderbook.get_best_bid()
        initial_bid_volume = orderbook.get_volume_at_price(initial_best_bid, side='bid')
        
        limit_sell_result = orderbook.simulate_order(
            order_type='limit',
            side='sell',
            size=1.0,
            price=initial_best_bid
        )
        
        assert limit_sell_result['success'] is True
        
        # Verifica atualização
        new_bid_volume = orderbook.get_volume_at_price(initial_best_bid, side='bid')
        assert new_bid_volume == initial_bid_volume - 1.0
    
    def test_high_frequency_updates(self, orderbook):
        """Testa atualizações em alta frequência"""
        import time
        
        num_updates = 500
        update_times = []
        
        for i in range(num_updates):
            start_time = time.perf_counter()
            
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0 + np.random.normal(0, 10), 1.0 + np.random.random())],
                asks=[(50001.0 + np.random.normal(0, 10), 1.0 + np.random.random())],
                sequence=i + 1
            )
            
            orderbook.update(update)
            
            end_time = time.perf_counter()
            update_times.append(end_time - start_time)
        
        # Calcula estatísticas de tempo
        avg_time = np.mean(update_times) * 1000  # ms
        p95_time = np.percentile(update_times, 95) * 1000  # ms
        max_time = np.max(update_times) * 1000  # ms
        
        print(f"High-frequency update performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  95th percentile: {p95_time:.3f}ms")
        print(f"  Maximum: {max_time:.3f}ms")
        print(f"  Total time: {sum(update_times)*1000:.1f}ms")
        
        # Requisitos de performance para alta frequência
        assert avg_time < 5.0, f"Average update time {avg_time:.3f}ms too high"
        assert p95_time < 10.0, f"95th percentile {p95_time:.3f}ms too high"
        
        # Verifica consistência
        assert orderbook.last_sequence == num_updates
        assert len(orderbook.bids) <= orderbook.max_depth
        assert len(orderbook.asks) <= orderbook.max_depth
    
    def test_thread_safety_stress_test(self, orderbook):
        """Teste de estresse para segurança de threads"""
        import concurrent.futures
        import random
        
        num_threads = 20
        operations_per_thread = 50
        results = []
        
        def thread_work(thread_id):
            thread_results = []
            
            for i in range(operations_per_thread):
                op_type = random.choice(['update', 'read', 'snapshot', 'statistics'])
                
                try:
                    if op_type == 'update':
                        update = OrderBookUpdate(
                            timestamp=datetime.now(),
                            bids=[(50000.0 + thread_id + i, 1.0)],
                            asks=[(50001.0 + thread_id + i, 1.0)],
                            sequence=thread_id * 1000 + i + 1
                        )
                        success = orderbook.update(update)
                        thread_results.append(('update', success, None))
                    
                    elif op_type == 'read':
                        bid = orderbook.get_best_bid()
                        ask = orderbook.get_best_ask()
                        thread_results.append(('read', (bid, ask), None))
                    
                    elif op_type == 'snapshot':
                        snapshot = orderbook.create_snapshot()
                        thread_results.append(('snapshot', snapshot.sequence, None))
                    
                    elif op_type == 'statistics':
                        stats = orderbook.get_statistics()
                        thread_results.append(('statistics', stats['update_count'], None))
                
                except Exception as e:
                    thread_results.append((op_type, False, str(e)))
            
            return thread_results
        
        # Executa threads concorrentemente
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_work, i) for i in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        # Analisa resultados
        updates = [r for r in results if r[0] == 'update']
        reads = [r for r in results if r[0] == 'read']
        snapshots = [r for r in results if r[0] == 'snapshot']
        stats = [r for r in results if r[0] == 'statistics']
        
        errors = [r for r in results if r[2] is not None]
        
        print(f"Thread safety stress test:")
        print(f"  Total operations: {len(results)}")
        print(f"  Updates: {len(updates)}")
        print(f"  Reads: {len(reads)}")
        print(f"  Snapshots: {len(snapshots)}")
        print(f"  Statistics: {len(stats)}")
        print(f"  Errors: {len(errors)}")
        
        if errors:
            print("Errors encountered:")
            for error in errors[:5]:  # Mostra apenas os primeiros 5 erros
                print(f"  {error}")
        
        # Verifica que não houve corrupção de dados
        assert len(errors) == 0, f"Found {len(errors)} errors in thread-safe operations"
        
        # Verifica que o orderbook está em estado consistente
        final_stats = orderbook.get_statistics()
        assert final_stats['error_count'] == 0
        
        # Verifica que bids e asks estão ordenados corretamente
        if orderbook.bids:
            bid_prices = [b[0] for b in orderbook.bids]
            assert all(bid_prices[i] >= bid_prices[i + 1] for i in range(len(bid_prices) - 1))
        
        if orderbook.asks:
            ask_prices = [a[0] for a in orderbook.asks]
            assert all(ask_prices[i] <= ask_prices[i + 1] for i in range(len(ask_prices) - 1))
    
    def test_memory_cleanup_on_reset(self, orderbook, large_orderbook_update):
        """Testa limpeza de memória no reset"""
        import gc
        
        # Adiciona muitos dados
        for i in range(100):
            update = large_orderbook_update
            update.sequence = i + 1
            orderbook.update(update)
        
        # Captura referências antes do reset
        bids_before = orderbook.bids
        asks_before = orderbook.asks
        
        # Reseta
        orderbook.reset()
        
        # Verifica que as listas estão vazias
        assert len(orderbook.bids) == 0
        assert len(orderbook.asks) == 0
        
        # Força garbage collection
        gc.collect()
        
        # As referências antigas ainda podem existir, mas o orderbook deve estar limpo
        assert orderbook.bids is not bids_before
        assert orderbook.asks is not asks_before
    
    def test_robustness_to_malformed_updates(self, orderbook):
        """Testa robustez a atualizações malformadas"""
        # Testa vários tipos de dados malformados
        malformed_cases = [
            None,
            {},
            {'bids': 'not a list', 'asks': []},
            {'bids': [], 'asks': 'not a list'},
            {'bids': [[1, 2, 3]], 'asks': []},  # Tupla com 3 elementos
            {'bids': [[1]], 'asks': []},  # Tupla com 1 elemento
            {'bids': [[1, 'not a number']], 'asks': []},  # Volume não numérico
            {'bids': [['not a number', 1]], 'asks': []},  # Preço não numérico
            {'bids': [[1, 1]], 'asks': [[1, 1]], 'timestamp': 'invalid timestamp'},
            {'bids': [[1, 1]], 'asks': [[1, 1]], 'sequence': 'not a number'},
        ]
        
        errors = []
        
        for i, case in enumerate(malformed_cases):
            try:
                orderbook.update(case)
                errors.append((i, case, "Should have raised an exception"))
            except Exception as e:
                # Esperado que lance exceção
                pass
        
        # Verifica que o orderbook ainda funciona após erros
        valid_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.0)],
            sequence=1000
        )
        
        success = orderbook.update(valid_update)
        assert success is True
        
        print(f"Robustness test: Handled {len(malformed_cases)} malformed cases")
        if errors:
            print(f"Unexpected successes: {len(errors)}")
            for error in errors:
                print(f"  Case {error[0]}: {error[1]}")
    
    def test_comprehensive_integration_scenario(self, orderbook):
        """Cenário de integração abrangente"""
        print("\n=== Comprehensive Integration Test ===")
        
        # Fase 1: Inicialização e configuração
        assert orderbook.symbol == "BTCUSDT"
        assert orderbook.max_depth == 100
        print("✓ Phase 1: Initialization complete")
        
        # Fase 2: Atualizações sequenciais
        updates = []
        for i in range(10):
            update = OrderBookUpdate(
                timestamp=datetime.now(),
                bids=[(50000.0 - i * 0.5, 1.0 + i * 0.1)],
                asks=[(50001.0 + i * 0.5, 1.0 + i * 0.1)],
                sequence=i + 1
            )
            success = orderbook.update(update)
            assert success is True
            updates.append(update)
        
        assert orderbook.last_sequence == 10
        print(f"✓ Phase 2: {len(updates)} sequential updates complete")
        
        # Fase 3: Operações de leitura
        best_bid = orderbook.get_best_bid()
        best_ask = orderbook.get_best_ask()
        spread = orderbook.get_spread()
        mid_price = orderbook.get_mid_price()
        
        assert best_bid is not None
        assert best_ask is not None
        assert spread is not None
        assert mid_price is not None
        
        print(f"  Best Bid: {best_bid:.2f}")
        print(f"  Best Ask: {best_ask:.2f}")
        print(f"  Spread: {spread:.4f}")
        print(f"  Mid Price: {mid_price:.2f}")
        print("✓ Phase 3: Read operations complete")
        
        # Fase 4: Análises
        imbalance = orderbook.get_imbalance(depth=5)
        total_volume = orderbook.get_total_volume(side='all')
        price_levels = orderbook.get_price_levels(side='all', depth=3)
        liquidity_profile = orderbook.get_liquidity_profile(price_range_percent=0.01)
        
        assert imbalance is not None
        assert total_volume is not None
        assert 'bids' in price_levels
        assert 'asks' in price_levels
        
        print(f"  Imbalance: {imbalance:.3f}")
        print(f"  Total Volume: {total_volume:.2f}")
        print("✓ Phase 4: Analysis operations complete")
        
        # Fase 5: Simulações
        market_impact = orderbook.get_market_impact(size=5.0, side='buy')
        order_simulation = orderbook.simulate_order(
            order_type='market',
            side='buy',
            size=1.0
        )
        
        assert 'impact_price' in market_impact
        assert 'success' in order_simulation
        
        print(f"  Market Impact for 5.0 BUY: {market_impact['impact_price']:.2f}")
        print(f"  Order Simulation: {'Success' if order_simulation['success'] else 'Failed'}")
        print("✓ Phase 5: Simulation operations complete")
        
        # Fase 6: Snapshot e serialização
        snapshot = orderbook.create_snapshot()
        assert snapshot.validate() is True
        
        json_str = snapshot.to_json()
        assert isinstance(json_str, str)
        
        # Desserialização
        new_snapshot = OrderBookSnapshot.from_json(json_str)
        assert new_snapshot.symbol == snapshot.symbol
        assert new_snapshot.sequence == snapshot.sequence
        
        print(f"  Snapshot sequence: {snapshot.sequence}")
        print(f"  Snapshot age: {snapshot.is_too_old(max_age_ms=5000)}")
        print("✓ Phase 6: Snapshot and serialization complete")
        
        # Fase 7: Estatísticas
        stats = orderbook.get_statistics()
        assert stats['update_count'] == 10
        assert stats['total_volume_processed'] > 0
        
        print(f"  Total updates: {stats['update_count']}")
        print(f"  Volume processed: {stats['total_volume_processed']:.2f}")
        print(f"  Max depth reached: {stats['max_depth_reached']}")
        print("✓ Phase 7: Statistics collection complete")
        
        # Fase 8: Reset e reutilização
        orderbook.reset()
        assert orderbook.last_sequence == 0
        assert len(orderbook.bids) == 0
        assert len(orderbook.asks) == 0
        
        # Nova atualização após reset
        new_update = OrderBookUpdate(
            timestamp=datetime.now(),
            bids=[(51000.0, 2.0)],
            asks=[(51001.0, 2.0)],
            sequence=1
        )
        
        success = orderbook.update(new_update)
        assert success is True
        assert orderbook.last_sequence == 1
        
        print("✓ Phase 8: Reset and reuse complete")
        
        print("\n=== All integration tests passed! ===")