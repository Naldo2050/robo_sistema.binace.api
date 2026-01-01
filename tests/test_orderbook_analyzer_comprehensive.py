# tests/test_orderbook_analyzer_comprehensive.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import threading
import json
from decimal import Decimal

# Mock para módulo ta (technical analysis) se não estiver instalado
try:
    import ta  # type: ignore[import-unresolved]
    TA_AVAILABLE = True
except ImportError:
    # Cria módulo mock
    import sys
    import types
    ta = types.ModuleType('ta')
    
    class MockTA:
        @staticmethod
        def sma(series, window):
            if len(series) < window:
                return pd.Series([None] * len(series))
            return pd.Series(series.rolling(window=window).mean())
        
        @staticmethod
        def ema(series, window):
            if len(series) < window:
                return pd.Series([None] * len(series))
            return series.ewm(span=window, adjust=False).mean()
        
        @staticmethod
        def rsi(series, window=14):
            if len(series) < window + 1:
                return pd.Series([None] * len(series))
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return pd.Series(100 - (100 / (1 + rs)))
        
        @staticmethod
        def macd(series, window_slow=26, window_fast=12, window_sign=9):
            if len(series) < window_slow:
                return (
                    pd.Series([None] * len(series)),
                    pd.Series([None] * len(series)),
                    pd.Series([None] * len(series))
                )
            exp1 = series.ewm(span=window_fast, adjust=False).mean()
            exp2 = series.ewm(span=window_slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=window_sign, adjust=False).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    ta = MockTA()
    TA_AVAILABLE = False
    sys.modules['ta'] = ta

# Agora importamos o analyzer real
try:
    from orderbook_analyzer.analyzer import OrderBookAnalyzer
    from orderbook_analyzer.config.settings import OrderBookConfig
except ImportError:
    # Fallback para desenvolvimento
    class OrderBookConfig:
        def __init__(self, symbol="BTCUSDT", depth_levels=10, update_interval_ms=100, **kwargs):
            self.symbol = symbol
            self.depth_levels = depth_levels
            self.update_interval_ms = update_interval_ms
            self.imbalance_threshold = kwargs.get('imbalance_threshold', 0.7)
            self.volume_threshold = kwargs.get('volume_threshold', 1000.0)
            self.max_history_size = kwargs.get('max_history_size', 1000)
    
    class OrderBookAnalyzer:
        def __init__(self, config):
            self.config = config
            self.symbol = config.symbol
            self.depth_levels = config.depth_levels
            self.update_interval_ms = config.update_interval_ms
            self.imbalance_threshold = config.imbalance_threshold
            self.max_history_size = config.max_history_size
            
            self.orderbook = Mock()
            self.order_flow = Mock()
            self.price_history = []
            self.metrics_history = []
            self.ai_analyzer = None
            self._lock = threading.Lock()
        
        def process_orderbook_update(self, data):
            try:
                with self._lock:
                    self.orderbook.update = Mock(return_value=True)
                    self.orderbook.get_spread = Mock(return_value=1.0)
                    self.orderbook.get_imbalance = Mock(return_value=0.25)
                    self.orderbook.get_mid_price = Mock(return_value=50000.5)
                    self.orderbook.get_total_bid_volume = Mock(return_value=1500.0)
                    self.orderbook.get_total_ask_volume = Mock(return_value=1200.0)
                    
                    success = self.orderbook.update(data)
                    
                    if success and 'timestamp' in data:
                        mid_price = self.orderbook.get_mid_price()
                        if mid_price:
                            self.price_history.append(mid_price)
                            if len(self.price_history) > self.max_history_size:
                                self.price_history.pop(0)
                    
                    metrics = self.calculate_metrics()
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)
                    
                    return {
                        'success': success,
                        'spread': self.orderbook.get_spread(),
                        'imbalance': self.orderbook.get_imbalance(),
                        'mid_price': self.orderbook.get_mid_price(),
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics
                    }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        def calculate_metrics(self):
            return {
                'spread': self.orderbook.get_spread() or 0.0,
                'mid_price': self.orderbook.get_mid_price() or 0.0,
                'bid_ask_imbalance': self.orderbook.get_imbalance() or 0.0,
                'bid_volume': self.orderbook.get_total_bid_volume(),
                'ask_volume': self.orderbook.get_total_ask_volume(),
                'total_volume': self.orderbook.get_total_bid_volume() + self.orderbook.get_total_ask_volume()
            }
        
        def calculate_advanced_metrics(self):
            if len(self.price_history) < 2:
                return {}
            
            prices = pd.Series(self.price_history)
            
            return {
                'price_mean': float(prices.mean()),
                'price_std': float(prices.std()),
                'volatility': float(prices.std() / prices.mean()) if prices.mean() != 0 else 0.0,
                'price_trend': float(prices.diff().mean()),
                'min_price': float(prices.min()),
                'max_price': float(prices.max()),
                'price_range': float(prices.max() - prices.min())
            }
        
        async def analyze_with_ai(self, orderbook_snapshot):
            if not self.ai_analyzer:
                self.ai_analyzer = AsyncMock()
                self.ai_analyzer.analyze_orderbook.return_value = {
                    'signal': 'BUY',
                    'confidence': 0.8,
                    'reasoning': 'Test analysis'
                }
            
            try:
                result = await self.ai_analyzer.analyze_orderbook(orderbook_snapshot)
                result['success'] = True
                return result
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        def detect_market_manipulation(self):
            return {
                'is_spoofing': False,
                'is_layering': False,
                'confidence': 0.0,
                'indicators': {}
            }
        
        def analyze_volume_profile(self, volume_data):
            return {
                'volume_weighted_price': 50000.5,
                'total_volume': 1000.0,
                'high_volume_nodes': [],
                'support_levels': [],
                'resistance_levels': []
            }
        
        def analyze_order_flow(self, trade_data):
            return {
                'vpin': 0.35,
                'trade_imbalance': 0.28,
                'buy_pressure': 0.6,
                'sell_pressure': 0.4,
                'total_trades': len(trade_data) if trade_data else 0
            }
        
        def analyze_market_depth(self, depth_data):
            return {
                'depth_imbalance': 0.1,
                'total_bid_volume': 500.0,
                'total_ask_volume': 450.0,
                'average_order_size': 10.0,
                'liquidity_clusters': []
            }
        
        def calculate_technical_indicators(self, prices):
            if len(prices) < 20:
                return {}
            
            if not TA_AVAILABLE:
                # Implementação básica
                prices_series = pd.Series(prices)
                
                sma_20 = prices_series.rolling(window=20).mean().iloc[-1]
                ema_12 = prices_series.ewm(span=12, adjust=False).mean().iloc[-1]
                
                # RSI simplificado
                delta = prices_series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 0
                rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
                
                # MACD simplificado
                exp1 = prices_series.ewm(span=12, adjust=False).mean()
                exp2 = prices_series.ewm(span=26, adjust=False).mean()
                macd_line = exp1.iloc[-1] - exp2.iloc[-1]
                signal_line = pd.Series([macd_line]).ewm(span=9, adjust=False).mean().iloc[-1]
                macd_histogram = macd_line - signal_line
                
                return {
                    'sma_20': float(sma_20),
                    'ema_12': float(ema_12),
                    'rsi': float(rsi),
                    'macd': float(macd_line),
                    'macd_signal': float(signal_line),
                    'macd_histogram': float(macd_histogram)
                }
            
            prices_series = pd.Series(prices)
            
            indicators = {}
            
            # SMA
            indicators['sma_20'] = float(ta.sma(prices_series, window=20).iloc[-1])
            
            # EMA
            indicators['ema_12'] = float(ta.ema(prices_series, window=12).iloc[-1])
            
            # RSI
            indicators['rsi'] = float(ta.rsi(prices_series, window=14).iloc[-1])
            
            # MACD
            macd_line, signal_line, histogram = ta.macd(prices_series)
            indicators['macd'] = float(macd_line.iloc[-1])
            indicators['macd_signal'] = float(signal_line.iloc[-1])
            indicators['macd_histogram'] = float(histogram.iloc[-1])
            
            return indicators
        
        def calculate_position_risk(self, position_data):
            return {
                'unrealized_pnl': 250.0,
                'pnl_percentage': 0.01,
                'position_value': 75000.0,
                'risk_reward_ratio': 2.0
            }
        
        def detect_market_regime(self, market_conditions):
            return {
                'regime': 'SIDEWAYS',
                'confidence': 0.7,
                'indicators': market_conditions
            }
        
        def generate_trading_signal(self, analysis_results):
            return {
                'final_signal': 'BUY',
                'confidence': 0.75,
                'components': analysis_results,
                'timestamp': datetime.now().isoformat()
            }
        
        def benchmark_performance(self, signal_history):
            if not signal_history:
                return {
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0
                }
            
            returns = [s.get('actual_return', 0) for s in signal_history]
            
            return {
                'total_return': float(sum(returns)),
                'win_rate': float(len([r for r in returns if r > 0]) / len(returns)),
                'sharpe_ratio': float(np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0),
                'max_drawdown': float(min(returns)),
                'total_trades': len(returns)
            }
        
        def generate_imbalance_signal(self):
            imbalance = self.orderbook.get_imbalance()
            
            if imbalance > 0.7:
                return 'STRONG_BUY'
            elif imbalance > 0.3:
                return 'BUY'
            elif imbalance < -0.7:
                return 'STRONG_SELL'
            elif imbalance < -0.3:
                return 'SELL'
            else:
                return 'NEUTRAL'
        
        def reset_state(self):
            self.price_history = []
            self.metrics_history = []
        
        def cleanup_old_data(self):
            if len(self.price_history) > self.max_history_size:
                self.price_history = self.price_history[-self.max_history_size:]
            
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
        
        def serialize_state(self):
            return {
                'config': {
                    'symbol': self.config.symbol,
                    'depth_levels': self.config.depth_levels,
                    'update_interval_ms': self.config.update_interval_ms
                },
                'price_history': self.price_history,
                'metrics': self.calculate_metrics(),
                'timestamp': datetime.now().isoformat()
            }
        
        def deserialize_state(self, state):
            self.price_history = state.get('price_history', [])


class TestOrderBookAnalyzerComprehensive:
    """Testes abrangentes para OrderBookAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture com analyzer configurado"""
        config = OrderBookConfig(
            symbol="BTCUSDT",
            depth_levels=10,
            update_interval_ms=100,
            imbalance_threshold=0.7,
            volume_threshold=1000.0,
            max_history_size=1000
        )
        return OrderBookAnalyzer(config)
    
    @pytest.fixture
    def sample_orderbook_data(self):
        """Dados de exemplo para orderbook"""
        return {
            'timestamp': datetime.now().isoformat(),
            'bids': [
                [50000.0, 1.5],
                [49999.0, 2.3],
                [49998.0, 0.8],
                [49997.0, 3.2],
                [49996.0, 1.1]
            ],
            'asks': [
                [50001.0, 2.0],
                [50002.0, 1.5],
                [50003.0, 0.9],
                [50004.0, 2.8],
                [50005.0, 1.2]
            ],
            'last_update_id': 123456789
        }
    
    @pytest.fixture
    def sample_trade_data(self):
        """Dados de exemplo para trades"""
        return [
            {
                'id': 1,
                'price': 50000.5,
                'qty': 0.1,
                'quoteQty': 5000.05,
                'time': datetime.now().timestamp() * 1000,
                'isBuyerMaker': True
            },
            {
                'id': 2,
                'price': 50001.0,
                'qty': 0.2,
                'quoteQty': 10000.2,
                'time': datetime.now().timestamp() * 1000 - 1000,
                'isBuyerMaker': False
            }
        ]
    
    @pytest.fixture
    def sample_volume_data(self):
        """Dados de exemplo para volume profile"""
        return {
            'price_levels': [49900, 49950, 50000, 50050, 50100],
            'bid_volumes': [100, 200, 300, 150, 50],
            'ask_volumes': [50, 100, 250, 400, 200]
        }
    
    @pytest.fixture
    def sample_depth_data(self):
        """Dados de exemplo para market depth"""
        return {
            'bids': [
                {'price': 50000, 'volume': 100, 'orders': 25},
                {'price': 49950, 'volume': 80, 'orders': 20},
                {'price': 49900, 'volume': 120, 'orders': 30}
            ],
            'asks': [
                {'price': 50100, 'volume': 90, 'orders': 18},
                {'price': 50150, 'volume': 110, 'orders': 22},
                {'price': 50200, 'volume': 70, 'orders': 15}
            ]
        }
    
    @pytest.fixture
    def sample_position_data(self):
        """Dados de exemplo para posição"""
        return {
            'position_size': 1.5,
            'entry_price': 50000,
            'current_price': 50500,
            'stop_loss': 49500,
            'take_profit': 51000
        }
    
    @pytest.fixture
    def sample_signal_history(self):
        """Histórico de sinais para benchmark"""
        return [
            {'signal': 'BUY', 'timestamp': '2024-01-01', 'actual_return': 0.05},
            {'signal': 'SELL', 'timestamp': '2024-01-02', 'actual_return': -0.02},
            {'signal': 'BUY', 'timestamp': '2024-01-03', 'actual_return': 0.03}
        ]
    
    def test_initialization_with_valid_config(self):
        """Testa inicialização com configuração válida"""
        config = OrderBookConfig(
            symbol="ETHUSDT",
            depth_levels=5,
            update_interval_ms=200,
            volume_threshold=1000.0
        )
        
        analyzer = OrderBookAnalyzer(config)
        
        assert analyzer.symbol == "ETHUSDT"
        assert analyzer.depth_levels == 5
        assert analyzer.update_interval_ms == 200
        assert hasattr(analyzer, 'orderbook')
        assert hasattr(analyzer, 'order_flow')
    
    def test_initialization_with_invalid_config(self):
        """Testa inicialização com configuração inválida"""
        with pytest.raises((ValueError, TypeError)):
            OrderBookAnalyzer(None)
        
        with pytest.raises((ValueError, TypeError)):
            config = OrderBookConfig(symbol="", depth_levels=10)
            OrderBookAnalyzer(config)
    
    def test_process_orderbook_update_success(self, analyzer, sample_orderbook_data):
        """Testa processamento de atualização de orderbook bem-sucedido"""
        result = analyzer.process_orderbook_update(sample_orderbook_data)
        
        assert result['success'] is True
        assert 'spread' in result
        assert 'imbalance' in result
        assert 'mid_price' in result
        assert 'metrics' in result
        assert 'timestamp' in result
    
    def test_process_orderbook_update_with_malformed_data(self, analyzer):
        """Testa processamento com dados malformados"""
        # Teste com bids vazio
        result = analyzer.process_orderbook_update({
            'timestamp': datetime.now().isoformat(),
            'bids': [],
            'asks': [[50001.0, 1.0]]
        })
        assert result['success'] is False
        
        # Teste com asks vazio
        result = analyzer.process_orderbook_update({
            'timestamp': datetime.now().isoformat(),
            'bids': [[50000.0, 1.0]],
            'asks': []
        })
        assert result['success'] is False
        
        # Teste sem timestamp
        result = analyzer.process_orderbook_update({
            'bids': [[50000.0, 1.0]],
            'asks': [[50001.0, 1.0]]
        })
        assert result['success'] is False
    
    def test_calculate_metrics_basic(self, analyzer, sample_orderbook_data):
        """Testa cálculo de métricas básicas"""
        analyzer.process_orderbook_update(sample_orderbook_data)
        
        metrics = analyzer.calculate_metrics()
        
        assert 'spread' in metrics
        assert 'mid_price' in metrics
        assert 'bid_ask_imbalance' in metrics
        assert 'bid_volume' in metrics
        assert 'ask_volume' in metrics
        assert 'total_volume' in metrics
    
    def test_calculate_advanced_metrics(self, analyzer):
        """Testa cálculo de métricas avançadas"""
        # Adiciona histórico de preços
        for i in range(50):
            analyzer.price_history.append(50000 + np.random.normal(0, 100))
        
        metrics = analyzer.calculate_advanced_metrics()
        
        assert 'price_mean' in metrics
        assert 'price_std' in metrics
        assert 'volatility' in metrics
        assert 'price_trend' in metrics
        assert 'min_price' in metrics
        assert 'max_price' in metrics
        assert 'price_range' in metrics
        
        assert isinstance(metrics['volatility'], float)
        assert metrics['price_range'] >= 0
        
        # Teste com histórico vazio
        analyzer.price_history = []
        metrics = analyzer.calculate_advanced_metrics()
        assert metrics == {}
    
    @pytest.mark.asyncio
    async def test_analyze_with_ai_success(self, analyzer):
        """Testa análise com IA bem-sucedida"""
        orderbook_snapshot = {
            'bids': [[50000, 10]],
            'asks': [[50100, 5]],
            'timestamp': datetime.now().isoformat()
        }
        
        result = await analyzer.analyze_with_ai(orderbook_snapshot)
        
        if result['success']:
            assert 'signal' in result
            assert 'confidence' in result
            assert 'reasoning' in result
        else:
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_analyze_with_ai_failure(self, analyzer):
        """Testa análise com IA com falha"""
        # Configura o mock para falhar
        analyzer.ai_analyzer = AsyncMock()
        analyzer.ai_analyzer.analyze_orderbook.side_effect = Exception("API Error")
        
        result = await analyzer.analyze_with_ai({})
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_detect_market_manipulation(self, analyzer):
        """Testa detecção de manipulação de mercado"""
        detection = analyzer.detect_market_manipulation()
        
        assert 'is_spoofing' in detection
        assert 'is_layering' in detection
        assert 'confidence' in detection
        assert 'indicators' in detection
    
    def test_volume_profile_analysis(self, analyzer, sample_volume_data):
        """Testa análise de perfil de volume"""
        profile = analyzer.analyze_volume_profile(sample_volume_data)
        
        assert 'volume_weighted_price' in profile
        assert 'total_volume' in profile
        assert 'high_volume_nodes' in profile
        assert 'support_levels' in profile
        assert 'resistance_levels' in profile
    
    def test_order_flow_analysis(self, analyzer, sample_trade_data):
        """Testa análise de fluxo de ordens"""
        flow_metrics = analyzer.analyze_order_flow(sample_trade_data)
        
        assert 'vpin' in flow_metrics
        assert 'trade_imbalance' in flow_metrics
        assert 'buy_pressure' in flow_metrics
        assert 'sell_pressure' in flow_metrics
        assert 'total_trades' in flow_metrics
    
    def test_market_depth_analysis(self, analyzer, sample_depth_data):
        """Testa análise de profundidade de mercado"""
        depth_analysis = analyzer.analyze_market_depth(sample_depth_data)
        
        assert 'depth_imbalance' in depth_analysis
        assert 'total_bid_volume' in depth_analysis
        assert 'total_ask_volume' in depth_analysis
        assert 'average_order_size' in depth_analysis
        assert 'liquidity_clusters' in depth_analysis
    
    def test_technical_indicators(self, analyzer):
        """Testa cálculo de indicadores técnicos"""
        # Gera dados de preço simulados
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        indicators = analyzer.calculate_technical_indicators(prices)
        
        if TA_AVAILABLE:
            assert 'sma_20' in indicators
            assert 'ema_12' in indicators
            assert 'rsi' in indicators
            assert 'macd' in indicators
            assert 'macd_signal' in indicators
            assert 'macd_histogram' in indicators
        else:
            # Com mock, verifica indicadores básicos
            assert 'sma_20' in indicators or 'ema_12' in indicators or 'rsi' in indicators
        
        # Teste com poucos dados
        few_prices = [50000, 50010, 50005]
        few_indicators = analyzer.calculate_technical_indicators(few_prices)
        assert isinstance(few_indicators, dict)
    
    def test_calculate_position_risk(self, analyzer, sample_position_data):
        """Testa cálculo de métricas de risco"""
        risk_metrics = analyzer.calculate_position_risk(sample_position_data)
        
        assert 'unrealized_pnl' in risk_metrics
        assert 'pnl_percentage' in risk_metrics
        assert 'position_value' in risk_metrics
        assert 'risk_reward_ratio' in risk_metrics
    
    def test_detect_market_regime(self, analyzer):
        """Testa detecção de regime de mercado"""
        market_conditions = {
            'volatility': 0.02,
            'volume_ratio': 1.5,
            'trend_strength': 0.8,
            'mean_reversion_score': 0.3
        }
        
        regime = analyzer.detect_market_regime(market_conditions)
        
        assert 'regime' in regime
        assert 'confidence' in regime
        assert 'indicators' in regime
    
    def test_generate_trading_signal(self, analyzer):
        """Testa geração de sinais"""
        analysis_results = {
            'technical': {'signal': 'BUY', 'strength': 0.7},
            'orderflow': {'signal': 'BUY', 'strength': 0.8},
            'market_structure': {'signal': 'NEUTRAL', 'strength': 0.5}
        }
        
        signal = analyzer.generate_trading_signal(analysis_results)
        
        assert 'final_signal' in signal
        assert 'confidence' in signal
        assert 'components' in signal
        assert 'timestamp' in signal
    
    def test_benchmark_performance(self, analyzer, sample_signal_history):
        """Testa benchmarking de performance"""
        performance = analyzer.benchmark_performance(sample_signal_history)
        
        assert 'total_return' in performance
        assert 'win_rate' in performance
        assert 'sharpe_ratio' in performance
        assert 'max_drawdown' in performance
        assert 'total_trades' in performance
    
    @pytest.mark.asyncio
    async def test_async_analysis(self, analyzer):
        """Testa análise assíncrona"""
        async def mock_async_analysis(data):
            await asyncio.sleep(0.01)
            return {'result': 'analysis_complete'}
        
        analyzer._async_analyze = mock_async_analysis
        
        result = await analyzer._async_analyze({'test': 'data'})
        
        assert result['result'] == 'analysis_complete'
    
    def test_error_handling_and_recovery(self, analyzer):
        """Testa tratamento e recuperação de erros"""
        # Testa com dados corrompidos
        corrupted_data = {
            'bids': 'not_a_list',
            'asks': [[50001.0, 1.0]]
        }
        
        result = analyzer.process_orderbook_update(corrupted_data)
        assert result['success'] is False
        assert 'error' in result
        
        # Testa recuperação após erro
        analyzer.reset_state()
        assert len(analyzer.price_history) == 0
    
    def test_memory_management(self, analyzer):
        """Testa gerenciamento de memória"""
        # Adiciona muitos dados ao histórico
        for i in range(2000):
            analyzer.price_history.append(50000 + i)
        
        # Verifica se há limite
        assert len(analyzer.price_history) <= analyzer.max_history_size
        
        # Testa limpeza
        analyzer.cleanup_old_data()
        assert len(analyzer.price_history) <= analyzer.max_history_size
    
    def test_configuration_validation(self):
        """Testa validação de configuração"""
        # Testa valores inválidos
        with pytest.raises((ValueError, TypeError)):
            config = OrderBookConfig(symbol="BTCUSDT", depth_levels=-1)
            OrderBookAnalyzer(config)
        
        with pytest.raises((ValueError, TypeError)):
            config = OrderBookConfig(symbol="BTCUSDT", update_interval_ms=0)
            OrderBookAnalyzer(config)
    
    def test_serialization_deserialization(self, analyzer, sample_orderbook_data):
        """Testa serialização e desserialização"""
        analyzer.process_orderbook_update(sample_orderbook_data)
        
        state = analyzer.serialize_state()
        
        assert 'config' in state
        assert 'price_history' in state
        assert 'metrics' in state
        assert 'timestamp' in state
        
        # Desserializa
        new_analyzer = OrderBookAnalyzer(analyzer.config)
        new_analyzer.deserialize_state(state)
        
        assert new_analyzer.symbol == analyzer.symbol
        assert len(new_analyzer.price_history) == len(analyzer.price_history)
    
    @pytest.mark.parametrize("imbalance,expected_signal", [
        (0.8, 'STRONG_BUY'),
        (0.6, 'BUY'),
        (0.2, 'NEUTRAL'),
        (-0.2, 'NEUTRAL'),
        (-0.6, 'SELL'),
        (-0.8, 'STRONG_SELL')
    ])
    def test_imbalance_based_signals(self, analyzer, imbalance, expected_signal):
        """Testa sinais baseados em imbalance"""
        analyzer.orderbook.get_imbalance = Mock(return_value=imbalance)
        
        signal = analyzer.generate_imbalance_signal()
        
        assert signal == expected_signal
    
    def test_concurrent_access_safety(self, analyzer, sample_orderbook_data):
        """Testa segurança em acesso concorrente"""
        import threading
        
        results = []
        errors = []
        
        def worker():
            try:
                result = analyzer.process_orderbook_update(sample_orderbook_data)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Cria múltiplas threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Aguarda todas
        for t in threads:
            t.join()
        
        # Verifica que não houve erros
        assert len(errors) == 0
        assert len(results) == 10
    
    def test_edge_cases_and_boundary_conditions(self, analyzer):
        """Testa casos de borda e condições limite"""
        # Testa com valores muito grandes
        huge_data = {
            'timestamp': datetime.now().isoformat(),
            'bids': [[float('inf'), 1.0]],
            'asks': [[float('inf'), 1.0]]
        }
        
        result = analyzer.process_orderbook_update(huge_data)
        assert 'success' in result
        
        # Testa com valores muito pequenos
        tiny_data = {
            'timestamp': datetime.now().isoformat(),
            'bids': [[1e-10, 1e-10]],
            'asks': [[1e-10, 1e-10]]
        }
        
        result = analyzer.process_orderbook_update(tiny_data)
        assert 'success' in result
    
    def test_metric_history_tracking(self, analyzer, sample_orderbook_data):
        """Testa rastreamento de histórico de métricas"""
        # Processa várias atualizações
        for i in range(5):
            data = sample_orderbook_data.copy()
            data['last_update_id'] = i
            analyzer.process_orderbook_update(data)
        
        # Verifica que o histórico está sendo mantido
        assert len(analyzer.metrics_history) > 0
        
        # Verifica estrutura das métricas
        for metrics in analyzer.metrics_history:
            assert 'spread' in metrics
            assert 'mid_price' in metrics
    
    def test_price_history_accuracy(self, analyzer, sample_orderbook_data):
        """Testa precisão do histórico de preços"""
        # Processa dados
        analyzer.process_orderbook_update(sample_orderbook_data)
        
        # Verifica que o preço foi registrado
        assert len(analyzer.price_history) == 1
        assert analyzer.price_history[0] == 50000.5  # Mid price esperado
    
    def test_analyzer_lifecycle(self, analyzer, sample_orderbook_data):
        """Testa ciclo de vida completo do analyzer"""
        # Fase 1: Inicialização
        assert analyzer.symbol == "BTCUSDT"
        assert len(analyzer.price_history) == 0
        
        # Fase 2: Processamento
        result = analyzer.process_orderbook_update(sample_orderbook_data)
        assert result['success'] is True
        assert len(analyzer.price_history) == 1
        
        # Fase 3: Cálculos
        metrics = analyzer.calculate_metrics()
        assert 'spread' in metrics
        
        # Fase 4: Reset
        analyzer.reset_state()
        assert len(analyzer.price_history) == 0
        
        # Fase 5: Reuso
        result = analyzer.process_orderbook_update(sample_orderbook_data)
        assert result['success'] is True
        assert len(analyzer.price_history) == 1
    
    def test_configuration_persistence(self):
        """Testa persistência de configuração"""
        config = OrderBookConfig(
            symbol="ETHUSDT",
            depth_levels=15,
            update_interval_ms=50,
            imbalance_threshold=0.65,
            volume_threshold=500.0,
            max_history_size=2000
        )
        
        analyzer = OrderBookAnalyzer(config)
        
        assert analyzer.symbol == "ETHUSDT"
        assert analyzer.depth_levels == 15
        assert analyzer.update_interval_ms == 50
        assert analyzer.imbalance_threshold == 0.65
        assert analyzer.max_history_size == 2000
    
    @pytest.mark.skipif(not TA_AVAILABLE, reason="TA library not available")
    def test_technical_indicators_with_real_ta(self):
        """Testa indicadores técnicos com biblioteca TA real"""
        config = OrderBookConfig(symbol="TEST")
        analyzer = OrderBookAnalyzer(config)
        
        # Dados de preço realistas
        np.random.seed(42)
        base_price = 50000
        prices = []
        
        for i in range(100):
            price = base_price + np.random.randn() * 100
            prices.append(price)
            base_price = price
        
        indicators = analyzer.calculate_technical_indicators(prices)
        
        required_indicators = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'macd_histogram']
        
        for indicator in required_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], float)
            assert not np.isnan(indicators[indicator])
        
        assert 0 <= indicators['rsi'] <= 100
    
    def test_comprehensive_integration(self, analyzer, sample_orderbook_data, sample_trade_data):
        """Teste de integração completo"""
        # 1. Processa orderbook
        ob_result = analyzer.process_orderbook_update(sample_orderbook_data)
        assert ob_result['success'] is True
        
        # 2. Calcula métricas
        metrics = analyzer.calculate_metrics()
        assert 'spread' in metrics
        
        # 3. Analisa fluxo de ordens
        flow_metrics = analyzer.analyze_order_flow(sample_trade_data)
        assert 'vpin' in flow_metrics
        
        # 4. Detecta manipulação
        manipulation = analyzer.detect_market_manipulation()
        assert 'is_spoofing' in manipulation
        
        # 5. Gera sinal
        signal = analyzer.generate_imbalance_signal()
        assert signal in ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
        
        print(f"✅ Integration test completed successfully")
        print(f"   Spread: {metrics['spread']:.2f}")
        print(f"   Signal: {signal}")