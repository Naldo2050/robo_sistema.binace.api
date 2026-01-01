# tests/test_risk_manager_comprehensive.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import numpy as np

from risk_management.risk_manager import (
    RiskManager, 
    RiskConfig,
    Position,
    TradeRequest,
    RiskMetrics
)
from risk_management.exceptions import (
    RiskLimitExceeded,
    PositionLimitError,
    DailyLossLimitError
)


class TestRiskManagerComprehensive:
    """Testes abrangentes para RiskManager"""
    
    @pytest.fixture
    def risk_config(self):
        """Configuração de risco"""
        return RiskConfig(
            max_position_size=100000,
            max_daily_loss=0.05,  # 5%
            max_loss_per_trade=0.02,  # 2%
            max_open_positions=10,
            max_correlation=0.8,
            var_confidence_level=0.95,
            enable_stress_testing=True
        )
    
    @pytest.fixture
    def risk_manager(self, risk_config):
        """RiskManager configurado"""
        return RiskManager(risk_config)
    
    @pytest.fixture
    def sample_position(self):
        """Posição de exemplo"""
        return Position(
            symbol='BTCUSDT',
            side='BUY',
            size=0.5,
            entry_price=50000,
            current_price=50500,
            stop_loss=49500,
            take_profit=51000,
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_trade_request(self):
        """Requisição de trade de exemplo"""
        return TradeRequest(
            symbol='BTCUSDT',
            side='BUY',
            size=0.5,
            price=50000,
            stop_loss=49500,
            take_profit=50500,
            strategy='momentum',
            confidence=0.8
        )
    
    def test_initialization(self, risk_config):
        """Testa inicialização do RiskManager"""
        rm = RiskManager(risk_config)
        
        assert rm.max_position_size == 100000
        assert rm.max_daily_loss == 0.05
        assert rm.max_loss_per_trade == 0.02
        assert rm.max_open_positions == 10
        assert rm.max_correlation == 0.8
        assert rm.var_confidence_level == 0.95
        assert rm.enable_stress_testing is True
        assert rm.positions == {}
        assert rm.daily_pnl == 0.0
    
    def test_add_position(self, risk_manager, sample_position):
        """Testa adição de posição"""
        risk_manager.add_position(sample_position)
        
        assert 'BTCUSDT' in risk_manager.positions
        assert risk_manager.positions['BTCUSDT'].size == 0.5
        assert risk_manager.positions['BTCUSDT'].side == 'BUY'
        
        # Testa adição de posição duplicada
        with pytest.raises(ValueError):
            risk_manager.add_position(sample_position)
    
    def test_update_position(self, risk_manager, sample_position):
        """Testa atualização de posição"""
        risk_manager.add_position(sample_position)
        
        # Atualiza preço
        risk_manager.update_position('BTCUSDT', 51000)
        
        assert risk_manager.positions['BTCUSDT'].current_price == 51000
        assert risk_manager.positions['BTCUSDT'].unrealized_pnl == 500  # 0.5 * (51000-50000)
        
        # Testa atualização de posição inexistente
        with pytest.raises(KeyError):
            risk_manager.update_position('ETHUSDT', 3000)
    
    def test_remove_position(self, risk_manager, sample_position):
        """Testa remoção de posição"""
        risk_manager.add_position(sample_position)
        
        # Remove com realização de PnL
        realized_pnl = risk_manager.remove_position('BTCUSDT', exit_price=51000)
        
        assert realized_pnl == 500  # 0.5 * (51000-50000)
        assert 'BTCUSDT' not in risk_manager.positions
        assert risk_manager.daily_pnl == 500
        
        # Testa remoção de posição inexistente
        with pytest.raises(KeyError):
            risk_manager.remove_position('BTCUSDT')
    
    def test_check_trade_request_approved(self, risk_manager, sample_trade_request):
        """Testa aprovação de requisição de trade"""
        # Configura posições existentes
        risk_manager.positions = {
            'BTCUSDT': Position(symbol='BTCUSDT', size=0.2, entry_price=49000, current_price=49500),
            'ETHUSDT': Position(symbol='ETHUSDT', size=5, entry_price=2800, current_price=2850)
        }
        
        result = risk_manager.check_trade_request(sample_trade_request)
        
        assert result['approved'] is True
        assert 'max_size' in result
        assert 'reason' in result
        assert result['max_size'] > 0
    
    def test_check_trade_request_position_limit(self, risk_manager, sample_trade_request):
        """Testa limite de posição"""
        # Adiciona muitas posições
        for i in range(10):
            risk_manager.positions[f'SYM{i}'] = Position(
                symbol=f'SYM{i}',
                size=1000,
                entry_price=100
            )
        
        result = risk_manager.check_trade_request(sample_trade_request)
        
        assert result['approved'] is False
        assert 'position limit' in result['reason'].lower()
    
    def test_check_trade_request_size_limit(self, risk_manager):
        """Testa limite de tamanho"""
        trade_request = TradeRequest(
            symbol='BTCUSDT',
            side='BUY',
            size=3.0,  # 3 BTC a 50000 = 150000 > max_position_size
            price=50000
        )
        
        result = risk_manager.check_trade_request(trade_request)
        
        assert result['approved'] is False
        assert 'position size' in result['reason'].lower()
    
    def test_check_trade_request_daily_loss_limit(self, risk_manager, sample_trade_request):
        """Testa limite de perda diária"""
        # Configura perda diária significativa
        risk_manager.daily_pnl = -0.06  # -6%, acima do limite de 5%
        
        result = risk_manager.check_trade_request(sample_trade_request)
        
        assert result['approved'] is False
        assert 'daily loss' in result['reason'].lower()
    
    def test_check_trade_request_loss_per_trade_limit(self, risk_manager):
        """Testa limite de perda por trade"""
        trade_request = TradeRequest(
            symbol='BTCUSDT',
            side='BUY',
            size=1.0,
            price=50000,
            stop_loss=45000  # Perda de 5000 (10%) > 2% limite
        )
        
        result = risk_manager.check_trade_request(trade_request)
        
        assert result['approved'] is False
        assert 'per trade' in result['reason'].lower()
    
    def test_calculate_position_risk_metrics(self, risk_manager, sample_position):
        """Testa cálculo de métricas de risco da posição"""
        risk_manager.add_position(sample_position)
        
        metrics = risk_manager.calculate_position_risk('BTCUSDT')
        
        assert 'unrealized_pnl' in metrics
        assert 'pnl_percentage' in metrics
        assert 'distance_to_stop' in metrics
        assert 'distance_to_take' in metrics
        assert 'risk_reward_ratio' in metrics
        assert 'var' in metrics
        
        assert metrics['unrealized_pnl'] == 250  # 0.5 * (50500-50000)
        assert metrics['pnl_percentage'] == 0.01  # 1%
    
    def test_calculate_portfolio_risk_metrics(self, risk_manager):
        """Testa cálculo de métricas de risco do portfólio"""
        # Adiciona múltiplas posições
        positions = [
            Position('BTCUSDT', 'BUY', 0.5, 50000, 50500),
            Position('ETHUSDT', 'SELL', 10, 3000, 2950),
            Position('SOLUSDT', 'BUY', 100, 100, 105)
        ]
        
        for pos in positions:
            risk_manager.add_position(pos)
        
        portfolio_metrics = risk_manager.calculate_portfolio_risk()
        
        assert 'total_exposure' in portfolio_metrics
        assert 'total_unrealized_pnl' in portfolio_metrics
        assert 'portfolio_var' in portfolio_metrics
        assert 'expected_shortfall' in portfolio_metrics
        assert 'concentration_risk' in portfolio_metrics
        assert 'correlation_matrix' in portfolio_metrics
        
        assert portfolio_metrics['total_exposure'] > 0
    
    def test_stress_testing(self, risk_manager):
        """Testa stress testing"""
        # Adiciona posições
        risk_manager.positions = {
            'BTCUSDT': Position('BTCUSDT', 'BUY', 0.5, 50000, 50000),
            'ETHUSDT': Position('ETHUSDT', 'SELL', 10, 3000, 3000)
        }
        
        stress_scenarios = {
            'market_crash': {'BTCUSDT': -0.20, 'ETHUSDT': -0.15},  # -20%, -15%
            'volatility_spike': {'BTCUSDT': -0.10, 'ETHUSDT': 0.05},  # -10%, +5%
            'recovery': {'BTCUSDT': 0.15, 'ETHUSDT': 0.10}  # +15%, +10%
        }
        
        stress_results = risk_manager.run_stress_tests(stress_scenarios)
        
        assert 'market_crash' in stress_results
        assert 'volatility_spike' in stress_results
        assert 'recovery' in stress_results
        
        for scenario, result in stress_results.items():
            assert 'portfolio_pnl' in result
            assert 'max_drawdown' in result
            assert 'liquidity_impact' in result
    
    def test_var_calculation(self, risk_manager):
        """Testa cálculo de Value at Risk (VaR)"""
        # Simula histórico de retornos
        returns = np.random.normal(0.001, 0.02, 1000)  # Média 0.1%, vol 2%
        
        var = risk_manager.calculate_var(returns, confidence_level=0.95)
        expected_shortfall = risk_manager.calculate_expected_shortfall(returns, confidence_level=0.95)
        
        assert var < 0  # VaR é tipicamente negativo para perdas
        assert expected_shortfall <= var  # ES deve ser mais conservador que VaR
    
    def test_correlation_analysis(self, risk_manager):
        """Testa análise de correlação"""
        # Dados de preços simulados
        price_data = pd.DataFrame({
            'BTCUSDT': np.random.normal(50000, 1000, 100),
            'ETHUSDT': np.random.normal(3000, 100, 100),
            'SOLUSDT': np.random.normal(100, 10, 100)
        })
        
        correlation_matrix = risk_manager.analyze_correlations(price_data)
        
        assert correlation_matrix.shape == (3, 3)
        assert np.all(correlation_matrix.values >= -1) and np.all(correlation_matrix.values <= 1)
        
        # Verifica diagonal (auto-correlação)
        assert np.all(np.diag(correlation_matrix) == 1)
    
    def test_liquidity_risk_assessment(self, risk_manager):
        """Testa avaliação de risco de liquidez"""
        orderbook_data = {
            'symbol': 'BTCUSDT',
            'bids': [
                {'price': 50000, 'volume': 10},
                {'price': 49900, 'volume': 5},
                {'price': 49800, 'volume': 3}
            ],
            'asks': [
                {'price': 50100, 'volume': 8},
                {'price': 50200, 'volume': 6},
                {'price': 50300, 'volume': 4}
            ]
        }
        
        position_size = 2.0  # 2 BTC
        
        liquidity_risk = risk_manager.assess_liquidity_risk(orderbook_data, position_size)
        
        assert 'slippage_estimate' in liquidity_risk
        assert 'market_impact' in liquidity_risk
        assert 'liquidation_time' in liquidity_risk
        assert 'spread_ratio' in liquidity_risk
        
        assert liquidity_risk['slippage_estimate'] >= 0
    
    def test_margin_requirement_calculation(self, risk_manager, sample_position):
        """Testa cálculo de requisito de margem"""
        risk_manager.add_position(sample_position)
        
        margin_req = risk_manager.calculate_margin_requirement('BTCUSDT')
        
        assert 'initial_margin' in margin_req
        assert 'maintenance_margin' in margin_req
        assert 'margin_ratio' in margin_req
        assert 'available_margin' in margin_req
        
        assert margin_req['initial_margin'] > 0
    
    def test_concentration_risk(self, risk_manager):
        """Testa risco de concentração"""
        # Adiciona posições concentradas em um ativo
        risk_manager.positions = {
            'BTCUSDT': Position('BTCUSDT', 'BUY', 2.0, 50000, 50000, notional_value=100000),
            'ETHUSDT': Position('ETHUSDT', 'BUY', 5.0, 3000, 3000, notional_value=15000),
            'SOLUSDT': Position('SOLUSDT', 'BUY', 50.0, 100, 100, notional_value=5000)
        }
        
        concentration = risk_manager.calculate_concentration_risk()
        
        assert 'herfindahl_index' in concentration
        assert 'largest_position_pct' in concentration
        assert 'sector_concentration' in concentration
        assert 'diversification_score' in concentration
        
        assert concentration['largest_position_pct'] > 0.5  # BTC deve ser >50%
    
    def test_scenario_analysis(self, risk_manager):
        """Testa análise de cenários"""
        scenarios = {
            'base_case': {
                'BTCUSDT': 0.00,  # 0%
                'ETHUSDT': 0.00,
                'USD': 0.00
            },
            'bear_market': {
                'BTCUSDT': -0.30,  # -30%
                'ETHUSDT': -0.40,  # -40%
                'USD': 0.05  # +5% (flight to safety)
            },
            'bull_market': {
                'BTCUSDT': 0.50,  # +50%
                'ETHUSDT': 0.60,  # +60%
                'USD': -0.02  # -2%
            }
        }
        
        # Configura posições
        risk_manager.positions = {
            'BTCUSDT': Position('BTCUSDT', 'BUY', 0.5, 50000, 50000),
            'ETHUSDT': Position('ETHUSDT', 'BUY', 10, 3000, 3000)
        }
        
        scenario_results = risk_manager.run_scenario_analysis(scenarios)
        
        for scenario_name, result in scenario_results.items():
            assert 'portfolio_pnl' in result
            assert 'var_impact' in result
            assert 'liquidity_impact' in result
            assert 'risk_adjusted_return' in result
    
    def test_risk_adjusted_return_metrics(self, risk_manager):
        """Testa métricas de retorno ajustado ao risco"""
        # Simula histórico de retornos
        returns = np.random.normal(0.001, 0.02, 252)  # 1 ano de dados diários
        
        metrics = risk_manager.calculate_risk_adjusted_metrics(returns)
        
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'omega_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)
    
    def test_volatility_forecasting(self, risk_manager):
        """Testa previsão de volatilidade"""
        # Dados de preços históricos
        prices = pd.Series(np.random.normal(50000, 1000, 100))
        
        volatility_forecast = risk_manager.forecast_volatility(prices)
        
        assert 'historical_vol' in volatility_forecast
        assert 'garch_vol' in volatility_forecast
        assert 'ewma_vol' in volatility_forecast
        assert 'volatility_ratio' in volatility_forecast
        assert 'regime' in volatility_forecast
        
        assert volatility_forecast['regime'] in ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
    
    def test_counterparty_risk_assessment(self, risk_manager):
        """Testa avaliação de risco de contraparte"""
        counterparties = {
            'exchange_a': {
                'credit_rating': 'AA',
                'volume_share': 0.4,
                'default_probability': 0.001
            },
            'exchange_b': {
                'credit_rating': 'A',
                'volume_share': 0.3,
                'default_probability': 0.005
            },
            'exchange_c': {
                'credit_rating': 'BBB',
                'volume_share': 0.3,
                'default_probability': 0.01
            }
        }
        
        exposure = {
            'exchange_a': 50000,
            'exchange_b': 30000,
            'exchange_c': 20000
        }
        
        counterparty_risk = risk_manager.assess_counterparty_risk(counterparties, exposure)
        
        assert 'total_exposure' in counterparty_risk
        assert 'weighted_default_prob' in counterparty_risk
        assert 'expected_loss' in counterparty_risk
        assert 'concentration_risk' in counterparty_risk
        assert 'diversification_score' in counterparty_risk
    
    def test_regulatory_compliance_check(self, risk_manager):
        """Testa verificação de conformidade regulatória"""
        # Configura limites regulatórios
        regulatory_limits = {
            'max_leverage': 20,
            'min_liquidity_ratio': 0.3,
            'max_concentration': 0.25,
            'stress_test_frequency': 'daily'
        }
        
        # Configura dados atuais
        current_state = {
            'leverage': 15,
            'liquidity_ratio': 0.4,
            'largest_position_pct': 0.2,
            'last_stress_test': datetime.now() - timedelta(hours=12)
        }
        
        compliance = risk_manager.check_regulatory_compliance(regulatory_limits, current_state)
        
        assert 'all_compliant' in compliance
        assert 'violations' in compliance
        assert 'warnings' in compliance
        assert 'required_actions' in compliance
        
        assert isinstance(compliance['all_compliant'], bool)
    
    def test_risk_limit_dynamic_adjustment(self, risk_manager):
        """Testa ajuste dinâmico de limites de risco"""
        # Simula condições de mercado
        market_conditions = {
            'volatility': 0.035,  # 3.5%
            'volume': 1500000000,
            'spread': 0.0002,  # 0.02%
            'vix': 25.0
        }
        
        # Ajusta limites baseado nas condições
        adjusted_limits = risk_manager.adjust_risk_limits(market_conditions)
        
        assert 'position_size_multiplier' in adjusted_limits
        assert 'leverage_multiplier' in adjusted_limits
        assert 'stop_loss_adjustment' in adjusted_limits
        assert 'recommended_action' in adjusted_limits
        
        # Em alta volatilidade, limites devem ser mais restritivos
        assert adjusted_limits['position_size_multiplier'] <= 1.0
    
    def test_risk_report_generation(self, risk_manager):
        """Testa geração de relatório de risco"""
        # Configura dados de exemplo
        risk_manager.positions = {
            'BTCUSDT': Position('BTCUSDT', 'BUY', 0.5, 50000, 50500),
            'ETHUSDT': Position('ETHUSDT', 'SELL', 10, 3000, 2950)
        }
        risk_manager.daily_pnl = 750
        
        report = risk_manager.generate_risk_report()
        
        assert 'executive_summary' in report
        assert 'portfolio_overview' in report
        assert 'risk_metrics' in report
        assert 'stress_test_results' in report
        assert 'recommendations' in report
        assert 'timestamp' in report
        
        assert report['portfolio_overview']['total_positions'] == 2
        assert report['portfolio_overview']['total_exposure'] > 0
    
    def test_error_handling_and_recovery(self, risk_manager):
        """Testa tratamento de erros e recuperação"""
        # Testa com dados inválidos
        invalid_position = Position(
            symbol='',
            side='INVALID',
            size=-1.0,
            entry_price=-50000
        )
        
        with pytest.raises(ValueError):
            risk_manager.add_position(invalid_position)
        
        # Verifica que o estado permanece consistente
        assert len(risk_manager.positions) == 0
        
        # Testa recuperação após erro
        risk_manager.reset_daily_pnl()
        assert risk_manager.daily_pnl == 0.0