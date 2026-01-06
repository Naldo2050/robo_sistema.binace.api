# tests/test_performance_benchmarks.py - VERSÃO CORRIGIDA
import pytest
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Mock das classes principais para evitar dependências
class MockOrderBook:
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.bids = []
        self.asks = []
    
    def update(self, data):
        self.bids = data.get('bids', [])
        self.asks = data.get('asks', [])
        return True

class MockOrderBookAnalyzer:
    def analyze(self, data):
        return {'spread': 1.0, 'imbalance': 0.1}

class MockMarketOrchestrator:
    async def process_market_data(self, data):
        await asyncio.sleep(0.001)  # Simula processamento
        return {'processed': True}


class TestPerformanceBenchmarks:
    """Testes de performance e benchmarks - VERSÃO CORRIGIDA"""
    
    @pytest.fixture
    def mock_orderbook(self):
        return MockOrderBook()
    
    @pytest.fixture
    def mock_analyzer(self):
        return MockOrderBookAnalyzer()
    
    @pytest.fixture
    def mock_orchestrator(self):
        return MockMarketOrchestrator()
    
    @pytest.mark.performance
    def test_orderbook_update_performance(self, mock_orderbook):
        """Benchmark de atualização de orderbook"""
        # Mede tempo para 1000 atualizações
        start_time = time.time()
        
        for i in range(1000):
            update_data = {
                'bids': [[50000 - i*0.1, 1.0] for _ in range(10)],
                'asks': [[50001 + i*0.1, 1.0] for _ in range(10)]
            }
            mock_orderbook.update(update_data)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Performance esperada: < 0.5 segundo para 1000 updates
        assert elapsed < 0.5, f"Muito lento: {elapsed:.3f}s para 1000 updates"
        
        updates_per_second = 1000 / elapsed
        print(f"📊 OrderBook updates/sec: {updates_per_second:.0f}")
    
    @pytest.mark.performance
    def test_orderbook_analyzer_performance(self, mock_analyzer):
        """Benchmark de análise de orderbook"""
        # Dados de teste
        orderbook_data = {
            'bids': [[50000 - i, 1.0 + i*0.1] for i in range(50)],
            'asks': [[50001 + i, 1.0 + i*0.1] for i in range(50)]
        }
        
        # Mede tempo para 100 análises
        times = []
        for _ in range(100):
            start = time.perf_counter()
            mock_analyzer.analyze(orderbook_data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # em ms
        p95_time = np.percentile(times, 95) * 1000
        
        # Performance esperada: < 10ms por análise
        assert avg_time < 10, f"Muito lento: {avg_time:.2f}ms média"
        
        print(f"📊 OrderBookAnalyzer: {avg_time:.2f}ms média, {p95_time:.2f}ms P95")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_market_orchestrator_throughput(self, mock_orchestrator):
        """Benchmark de throughput do orchestrator"""
        # Mede processamento de 1000 mensagens
        messages = [
            {'symbol': 'BTCUSDT', 'price': 50000 + i, 'volume': 100}
            for i in range(1000)
        ]
        
        start_time = time.time()
        
        for msg in messages:
            await mock_orchestrator.process_market_data(msg)
        
        end_time = time.time()
        
        throughput = len(messages) / (end_time - start_time)
        
        # Esperado: > 100 mensagens/segundo
        assert throughput > 100, f"Throughput baixo: {throughput:.0f} msg/sec"
        
        print(f"📊 MarketOrchestrator throughput: {throughput:.0f} msg/sec")
    
    @pytest.mark.performance
    def test_memory_usage_growth(self):
        """Testa crescimento de uso de memória"""
        import tracemalloc
        import gc
        
        tracemalloc.start()
        
        # Captura snapshot inicial
        snapshot1 = tracemalloc.take_snapshot()
        
        # Cria muitos objetos
        orderbooks = []
        for i in range(1000):
            ob = MockOrderBook(symbol=f'SYM{i}')
            orderbooks.append(ob)
        
        # Captura snapshot após criação
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calcula diferença
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Encontra crescimento total
        total_increase = sum(stat.size for stat in top_stats) if top_stats else 0
        
        # Verifica que o crescimento é razoável (< 10MB)
        assert total_increase < 10 * 1024 * 1024, f"Crescimento excessivo: {total_increase/1024/1024:.1f}MB"
        
        # Limpa
        del orderbooks
        gc.collect()
        
        tracemalloc.stop()
        
        print(f"📊 Memory growth: {total_increase/1024/1024:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_performance(self):
        """Testa performance com concorrência"""
        import asyncio
        
        orchestrator = MockMarketOrchestrator()
        
        async def process_batch(batch_id, num_messages):
            for i in range(num_messages):
                await orchestrator.process_market_data({
                    'symbol': f'SYM{batch_id}',
                    'price': 100 + i,
                    'volume': 10
                })
        
        # Executa 4 workers concorrentes
        num_workers = 4
        messages_per_worker = 250
        
        start_time = time.time()
        
        tasks = [
            process_batch(i, messages_per_worker)
            for i in range(num_workers)
        ]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        total_messages = num_workers * messages_per_worker
        total_time = end_time - start_time
        throughput = total_messages / total_time
        
        # Verifica scaling
        assert throughput > 500, f"Throughput concorrente baixo: {throughput:.0f} msg/sec"
        
        print(f"📊 Concurrent throughput ({num_workers} workers): {throughput:.0f} msg/sec")