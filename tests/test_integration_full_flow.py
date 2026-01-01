# tests/test_integration_full_flow.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from orderbook_analyzer import OrderBookAnalyzer
from market_orchestrator import EnhancedMarketBot
from ai_runner import AIRunner
from risk_management.risk_manager import RiskManager


class TestFullTradingFlow:
    """Testes de integração do fluxo completo de trading"""
    
    @pytest.fixture
    def full_trading_system(self):
        """Sistema de trading completo com mocks"""
        with patch('orderbook_analyzer.OrderBookAnalyzer'), \
             patch('risk_management.risk_manager.RiskManager'), \
             patch('ai_runner.ai_runner.AIRunner'):
            
            # Configura orchestrator
            orchestrator = EnhancedMarketBot(
                stream_url="wss://test.stream.com",
                symbol="BTCUSDT",
                window_size_minutes=5,
                vol_factor_exh=2.0,
                history_size=100,
                delta_std_dev_factor=1.5,
                context_sma_period=20,
                liquidity_flow_alert_percentage=5.0,
                wall_std_dev_factor=2.0
            )
            
            # Configura componentes mockados
            orchestrator.orderbook_analyzer = Mock()
            orchestrator.risk_manager = Mock()
            orchestrator.ai_runner = Mock()
            orchestrator.trade_executor = Mock()
            
            return orchestrator
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_trade_signal_flow(self, full_trading_system):
        """Testa fluxo completo de sinal de trade"""
        # 1. Dados de mercado
        market_data = {
            'symbol': 'BTCUSDT',
            'price': 50000,
            'volume': 1000,
            'orderbook': {'bids': [[49999, 10]], 'asks': [[50001, 8]]}
        }
        
        # 2. Análise do orderbook
        full_trading_system.orderbook_analyzer.process_orderbook_update.return_value = {
            'success': True,
            'spread': 2.0,
            'imbalance': 0.3,
            'metrics': {'volatility': 0.02}
        }
        
        # 3. Análise de IA
        full_trading_system.ai_runner.analyze_orderbook.return_value = {
            'success': True,
            'signal': 'STRONG_BUY',
            'confidence': 0.92,
            'reasoning': 'Strong metrics',
            'price_target': 51000
        }
        
        # 4. Verificação de risco
        full_trading_system.risk_manager.check_trade_request.return_value = {
            'approved': True,
            'max_size': 1.0,
            'reason': 'Within limits'
        }
        
        # 5. Execução do trade
        full_trading_system.trade_executor.execute_trade.return_value = {
            'success': True,
            'order_id': '12345',
            'filled_price': 50000.5,
            'filled_size': 0.5
        }
        
        # Executa fluxo completo
        result = await full_trading_system.execute_complete_flow(market_data)
        
        # Verificações
        assert result['success'] is True
        assert 'order_id' in result
        assert result['signal_strength'] == 'STRONG_BUY'
        
        # Verifica que todos os componentes foram chamados
        full_trading_system.orderbook_analyzer.process_orderbook_update.assert_called_once()
        full_trading_system.ai_runner.analyze_orderbook.assert_called_once()
        full_trading_system.risk_manager.check_trade_request.assert_called_once()
        full_trading_system.trade_executor.execute_trade.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_flow_with_risk_rejection(self, full_trading_system):
        """Testa fluxo com rejeição de risco"""
        # Configura risco para rejeitar
        full_trading_system.risk_manager.check_trade_request.return_value = {
            'approved': False,
            'reason': 'Daily loss limit exceeded'
        }
        
        result = await full_trading_system.execute_complete_flow({})
        
        assert result['success'] is False
        assert 'risk_rejected' in result
        full_trading_system.trade_executor.execute_trade.assert_not_called()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_flow_with_execution_failure(self, full_trading_system):
        """Testa fluxo com falha na execução"""
        # Configura execução para falhar
        full_trading_system.trade_executor.execute_trade.side_effect = Exception("Exchange error")
        
        result = await full_trading_system.execute_complete_flow({})
        
        assert result['success'] is False
        assert 'execution_error' in result
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_market_data_processing(self, full_trading_system):
        """Testa processamento concorrente de dados de mercado"""
        import asyncio
        
        # Cria múltiplos streams de dados
        market_data_streams = [
            [{'symbol': 'BTCUSDT', 'price': 50000 + i} for i in range(10)],
            [{'symbol': 'ETHUSDT', 'price': 3000 + i} for i in range(10)],
            [{'symbol': 'SOLUSDT', 'price': 100 + i} for i in range(10)]
        ]
        
        results = []
        
        async def process_stream(stream_data):
            for data in stream_data:
                result = await full_trading_system.process_market_data(data)
                results.append(result)
        
        # Processa streams concorrentemente
        tasks = [process_stream(stream) for stream in market_data_streams]
        await asyncio.gather(*tasks)
        
        assert len(results) == 30  # 3 streams * 10 dados cada
    
    @pytest.mark.integration
    def test_system_health_monitoring(self, full_trading_system):
        """Testa monitoramento de saúde do sistema"""
        # Configura saúde dos componentes
        full_trading_system.orderbook_analyzer.health_check.return_value = {'status': 'HEALTHY'}
        full_trading_system.risk_manager.health_check.return_value = {'status': 'HEALTHY'}
        full_trading_system.ai_runner.health_check.return_value = {'status': 'DEGRADED'}
        full_trading_system.trade_executor.health_check.return_value = {'status': 'HEALTHY'}
        
        system_health = full_trading_system.check_system_health()
        
        assert 'overall_status' in system_health
        assert 'components' in system_health
        assert 'degraded_components' in system_health
        assert 'unhealthy_components' in system_health
        
        assert 'ai_runner' in system_health['degraded_components']
        assert system_health['overall_status'] == 'DEGRADED'
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_endurance_test(self, full_trading_system):
        """Teste de resistência com múltiplos ciclos"""
        # Executa múltiplos ciclos de processamento
        for cycle in range(100):
            result = await full_trading_system.process_market_data({
                'symbol': 'BTCUSDT',
                'price': 50000 + cycle,
                'volume': 1000
            })
            
            # Verifica que não há degradação de performance
            assert 'processing_time' in result
            assert result['processing_time'] < 1.0  # Menos de 1 segundo por ciclo
        
        # Verifica estatísticas finais
        stats = full_trading_system.get_performance_statistics()
        assert stats['total_cycles'] == 100
        assert stats['success_rate'] >= 0.95  # Pelo menos 95% de sucesso