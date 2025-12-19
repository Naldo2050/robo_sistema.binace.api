"""
TESTS CONSOLIDADOS PARA O SISTEMA DE SUPORTE E RESISTÊNCIA
==========================================================

Arquivo de testes compatível com o módulo support_resistance.py

Versão: 2.0.0 (Corrigida e Compatível)
Última atualização: 2024
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from support_resistance import (
    # Classes principais
    InstitutionalSupportResistanceSystem,
    AdvancedSupportResistance,
    VolumeProfileAnalyzer,
    InstitutionalMarketMonitor,
    InstitutionalPivotPoints,
    
    # Configurações
    InstitutionalConfig,
    SRConfig,
    VolumeProfileConfig,
    MonitorConfig,
    PivotConfig,
    
    # Validação
    validate_dataframe,
    validate_series,
    
    # Enums
    LevelType,
    ReactionType,
    ConfidenceLevel,
    MarketBias,
    QualityRating,
    
    # Utilitários
    StatisticalUtils,
    HealthCheckResult,
    
    # Constantes
    CONSTANTS
)


# ============================================================================
# FIXTURES COMPARTILHADAS
# ============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    DataFrame sintético realista para testes (OHLCV)
    
    Gera dados com tendência, ruído e níveis de S/R embutidos para
    testes realistas de detecção de níveis.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    
    base_price = 100.0
    trend = np.linspace(0, 25, len(dates))
    noise = np.random.normal(0, 2.0, len(dates))
    
    prices = base_price + trend + noise
    
    # Injetar níveis de S/R realistas para teste de detecção
    support_levels = [95, 97, 102]
    resistance_levels = [105, 108, 112]
    
    for i, price in enumerate(prices):
        for level in support_levels:
            if abs(price - level) < 1:
                prices[i] = level + np.random.random() * 0.5
        
        for level in resistance_levels:
            if abs(price - level) < 1:
                prices[i] = level - np.random.random() * 0.5
    
    return pd.DataFrame(
        {
            "open": prices - np.random.random(len(dates)) * 2,
            "high": prices + np.random.random(len(dates)) * 3,
            "low": prices - np.random.random(len(dates)) * 3,
            "close": prices,
            "volume": np.random.lognormal(10, 1, len(dates)) * 1000,
        },
        index=dates,
    )


@pytest.fixture
def constant_price_df() -> pd.DataFrame:
    """DataFrame com preço constante para testar edge cases"""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    price = 100.0
    return pd.DataFrame(
        {
            "open": [price] * len(dates),
            "high": [price] * len(dates),
            "low": [price] * len(dates),
            "close": [price] * len(dates),
            "volume": [10.0] * len(dates),
        },
        index=dates,
    )


@pytest.fixture
def malformed_df() -> pd.DataFrame:
    """DataFrame malformado para testar validações"""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "close": [100.0] * len(dates),
            "volume": [1000.0] * len(dates),
        },
        index=dates,
    )


@pytest.fixture
def test_config() -> InstitutionalConfig:
    """Configuração de teste padrão"""
    return InstitutionalConfig(
        sr=SRConfig(
            lookback_period=100,
            merge_tolerance=0.01,
            # 6 pesos que somam 1.0
            weights=(0.20, 0.15, 0.20, 0.15, 0.15, 0.15)
        ),
        volume_profile=VolumeProfileConfig(bins=50),
        monitor=MonitorConfig(),
        pivot=PivotConfig(),
        enable_cache=False
    )


# ============================================================================
# TESTES PARA FUNÇÕES DE VALIDAÇÃO
# ============================================================================

class TestValidation:
    """Testes para funções de validação"""
    
    def test_validate_dataframe_success(self, sample_df):
        """Validação deve passar para DataFrame válido"""
        validate_dataframe(sample_df)
    
    def test_validate_dataframe_missing_columns(self, malformed_df):
        """Validação deve falhar para colunas faltando"""
        with pytest.raises(ValueError, match="Colunas faltando"):
            validate_dataframe(
                malformed_df,
                required_cols=['open', 'high', 'low', 'close', 'volume']
            )
    
    def test_validate_dataframe_insufficient_rows(self):
        """Validação deve falhar para poucas linhas"""
        small_df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })
        
        with pytest.raises(ValueError, match="precisa ter pelo menos"):
            validate_dataframe(small_df, min_rows=50)
    
    def test_validate_dataframe_zero_or_negative_prices(self):
        """Validação deve falhar para preços ≤ 0"""
        bad_df = pd.DataFrame({
            'open': [100.0, 0.0],
            'high': [101.0, 101.0],
            'low': [99.0, 99.0],
            'close': [100.5, 100.5],
            'volume': [1000.0, 1000.0]
        })
        
        with pytest.raises(ValueError, match="contém .* valores <= 0"):
            validate_dataframe(bad_df)
    
    def test_validate_dataframe_negative_volume(self):
        """Validação deve falhar para volume negativo"""
        bad_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'close': [100.5, 101.5],
            'volume': [1000.0, -500.0]
        })
        
        with pytest.raises(ValueError, match="contém valores negativos"):
            validate_dataframe(bad_df)
    
    def test_validate_dataframe_nan_values(self):
        """Validação deve falhar para valores NaN"""
        bad_df = pd.DataFrame({
            'open': [100.0, np.nan],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'close': [100.5, 101.5],
            'volume': [1000.0, 1100.0]
        })
        
        with pytest.raises(ValueError, match="contém .* valores NaN"):
            validate_dataframe(bad_df)
    
    def test_validate_series_success(self):
        """Validação de série deve passar para dados válidos"""
        series = pd.Series([1.0, 2.0, 3.0, 4.0] * 10)  # 40 elementos
        validate_series(series, min_length=20, allow_nan=False)
    
    def test_validate_series_nan_without_permission(self):
        """Validação deve falhar para NaN quando não permitido"""
        series = pd.Series([1.0, np.nan, 3.0] * 10)
        with pytest.raises(ValueError, match="contém .* valores NaN"):
            validate_series(series, min_length=20, allow_nan=False)
    
    def test_validate_series_nan_with_permission(self):
        """Validação deve passar para NaN quando permitido"""
        series = pd.Series([1.0, np.nan, 3.0] * 10)
        validate_series(series, min_length=20, allow_nan=True)


# ============================================================================
# TESTES PARA CLASSES DE CONFIGURAÇÃO
# ============================================================================

class TestConfiguration:
    """Testes para classes de configuração"""
    
    def test_sr_config_defaults(self):
        """SRConfig deve ter valores padrão corretos"""
        config = SRConfig()
        assert config.lookback_period == 100
        assert config.merge_tolerance == 0.01
        assert len(config.weights) == 6  # 6 pesos no código atual
        assert config.min_cluster_size == 3
    
    def test_sr_config_validation_correct(self):
        """Validação deve aceitar 6 pesos que somam 1.0"""
        cfg = SRConfig(weights=(0.25, 0.12, 0.23, 0.12, 0.14, 0.14))
        assert len(cfg.weights) == 6
        assert abs(sum(cfg.weights) - 1.0) < 1e-10
    
    def test_sr_config_validation_wrong_count(self):
        """Validação deve rejeitar número incorreto de pesos"""
        with pytest.raises(ValueError, match="exatamente 6 valores"):
            SRConfig(weights=(0.5, 0.5))
    
    def test_sr_config_validation_wrong_sum(self):
        """Validação deve rejeitar pesos que não somam 1.0"""
        with pytest.raises(ValueError, match="deve somar 1.0"):
            SRConfig(weights=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1))  # Soma 0.6
    
    def test_volume_profile_config_defaults(self):
        """VolumeProfileConfig deve ter valores padrão corretos"""
        config = VolumeProfileConfig()
        assert config.bins == 50
        assert config.value_area_percent == 0.70
        assert config.min_data_points == 20
    
    def test_volume_profile_config_custom_values(self):
        """VolumeProfileConfig deve aceitar valores customizados"""
        config = VolumeProfileConfig(bins=30, value_area_percent=0.80)
        assert config.bins == 30
        assert config.value_area_percent == 0.80
    
    def test_institutional_config_defaults(self):
        """InstitutionalConfig deve ter valores padrão corretos"""
        config = InstitutionalConfig()
        assert config.min_data_points == 50
        assert config.confidence_level == 0.95
        assert config.enable_cache == True
    
    def test_monitor_config_defaults(self):
        """MonitorConfig deve ter valores padrão corretos"""
        config = MonitorConfig()
        assert config.tolerance_percent == 0.5
        assert config.lookback_ticks == 100
        assert config.max_test_history == 1000


# ============================================================================
# TESTES PARA SISTEMA COMPLETO (InstitutionalSupportResistanceSystem)
# ============================================================================

class TestInstitutionalSupportResistanceSystem:
    """Testes para a classe InstitutionalSupportResistanceSystem"""
    
    def test_initialization_default(self):
        """Sistema deve inicializar com configurações padrão"""
        system = InstitutionalSupportResistanceSystem()
        assert system is not None
        assert hasattr(system, 'config')
        assert system.config.sr.lookback_period == 100
    
    def test_initialization_custom_config(self, test_config):
        """Sistema deve inicializar com configuração customizada"""
        system = InstitutionalSupportResistanceSystem(config=test_config)
        assert system.config.sr.lookback_period == 100
        assert system.config.volume_profile.bins == 50
    
    def test_analyze_basic_execution(self, sample_df):
        """Análise básica deve executar sem erros"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        # Verificar estrutura básica
        assert isinstance(result, dict)
        required_keys = [
            "timestamp",
            "data_points",
            "sr_analysis",
            "pivot_analysis",
            "volume_profile",
            "confluence_analysis",
            "consolidated_report",
            "performance"
        ]
        
        for key in required_keys:
            assert key in result, f"Chave '{key}' faltando no resultado"
        
        # Verificar sr_analysis
        sr = result["sr_analysis"]
        assert "support_levels" in sr
        assert "resistance_levels" in sr
        assert "current_price" in sr
        assert sr["current_price"] > 0
    
    def test_analyze_with_custom_config(self, sample_df, test_config):
        """Análise deve funcionar com configuração customizada"""
        system = InstitutionalSupportResistanceSystem(config=test_config)
        result = system.analyze_market(sample_df)
        
        sr = result["sr_analysis"]
        assert "support_levels" in sr
        assert "resistance_levels" in sr
        
        # Verificar estrutura dos níveis
        if sr["support_levels"]:
            level = sr["support_levels"][0]
            assert "price" in level
            assert "composite_score" in level
            assert "touches" in level
            assert "volume_strength" in level
    
    def test_analyze_constant_price(self, constant_price_df):
        """Sistema deve lidar com preço constante"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(constant_price_df)
        
        sr = result["sr_analysis"]
        assert sr["current_price"] == 100.0
        assert isinstance(sr["support_levels"], list)
        assert isinstance(sr["resistance_levels"], list)
    
    def test_analyze_without_volume(self):
        """Sistema deve funcionar sem dados de volume significativos"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.random.normal(0, 5, len(dates))
        
        df_no_volume = pd.DataFrame({
            'open': prices - 1,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': [0.001] * len(dates)  # Volume quase zero
        }, index=dates)
        
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(df_no_volume)
        
        assert "sr_analysis" in result
        assert "volume_profile" in result
    
    def test_analyze_empty_data_handling(self):
        """Sistema deve lidar com dados insuficientes"""
        system = InstitutionalSupportResistanceSystem()
        
        small_df = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [1000.0] * 10
        })
        
        with pytest.raises(ValueError, match="pelo menos"):
            system.analyze_market(small_df)
    
    def test_levels_structure_and_scores(self, sample_df):
        """Níveis detectados devem ter estrutura e scores válidos"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        sr = result["sr_analysis"]
        all_levels = sr["support_levels"] + sr["resistance_levels"]
        
        for level in all_levels:
            # Campos obrigatórios
            assert "price" in level
            assert "composite_score" in level
            assert "touches" in level
            assert "volume_strength" in level
            assert "recency_score" in level
            assert "stability_score" in level
            assert "reaction_score" in level
            assert "confidence_interval" in level
            assert "cluster_quality" in level
            assert "type" in level
            
            # Tipos corretos
            assert isinstance(level["price"], float)
            assert isinstance(level["composite_score"], float)
            assert isinstance(level["touches"], int)
            
            # Valores válidos
            assert level["price"] > 0
            assert 0.0 <= level["composite_score"] <= 10.0
            assert level["touches"] >= 0
    
    def test_volume_profile_structure(self, sample_df):
        """Volume Profile deve ter estrutura correta"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        vp = result["volume_profile"]
        
        # POC
        assert "poc" in vp
        poc = vp["poc"]
        assert "price" in poc
        assert "volume" in poc
        assert "strength" in poc
        assert poc["price"] > 0
        assert 0.0 <= poc["strength"] <= 10.0
        
        # Value Area
        assert "value_area" in vp
        va = vp["value_area"]
        assert "low" in va
        assert "high" in va
        assert "width" in va
        assert va["low"] <= va["high"]
    
    def test_confluence_analysis_structure(self, sample_df):
        """Análise de confluência deve ter estrutura correta"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        confluence = result["confluence_analysis"]
        assert "levels" in confluence
        assert "strongest_levels" in confluence
        assert "overall_confluence_score" in confluence
        assert "metadata" in confluence
        
        assert 0.0 <= confluence["overall_confluence_score"] <= 10.0
    
    def test_consolidated_report_structure(self, sample_df):
        """Relatório consolidado deve ter estrutura correta"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        report = result["consolidated_report"]
        assert "summary" in report
        assert "market_context" in report
        assert "key_levels" in report
        assert "risk_assessment" in report
        assert "recommendations" in report
        
        summary = report["summary"]
        assert "current_price" in summary
        assert "market_regime" in summary
        assert "sr_quality_score" in summary
    
    def test_context_manager(self, sample_df):
        """Sistema deve funcionar como context manager"""
        with InstitutionalSupportResistanceSystem() as system:
            result = system.analyze_market(sample_df)
            assert "sr_analysis" in result
        
        # Após sair do context, last_analysis deve ser None
        assert system.last_analysis is None
    
    def test_reset_method(self, sample_df):
        """Método reset deve limpar estado"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        assert system.last_analysis is not None
        
        system.reset()
        
        assert system.last_analysis is None
        assert len(system.performance_metrics) == 0
    
    def test_health_check(self):
        """Health check deve funcionar corretamente"""
        system = InstitutionalSupportResistanceSystem()
        health = system.health_check()
        
        assert isinstance(health, HealthCheckResult)
        assert health.status in ["healthy", "degraded", "unhealthy"]
        assert "config_valid" in health.checks
        assert "cache_functional" in health.checks
        assert "analysis_functional" in health.checks
        assert "volume_profile_functional" in health.checks
    
    def test_create_market_monitor(self, sample_df):
        """Deve criar monitor de mercado após análise"""
        system = InstitutionalSupportResistanceSystem()
        system.analyze_market(sample_df)
        
        monitor = system.create_market_monitor()
        
        assert isinstance(monitor, InstitutionalMarketMonitor)
        assert len(monitor.support_levels) >= 0
        assert len(monitor.resistance_levels) >= 0
    
    def test_create_monitor_without_analysis(self):
        """Deve falhar ao criar monitor sem análise prévia"""
        system = InstitutionalSupportResistanceSystem()
        
        with pytest.raises(RuntimeError, match="Nenhuma análise encontrada"):
            system.create_market_monitor()


# ============================================================================
# TESTES PARA MARKET MONITOR
# ============================================================================

class TestInstitutionalMarketMonitor:
    """Testes para InstitutionalMarketMonitor"""
    
    @pytest.fixture
    def monitor_with_levels(self):
        """Cria monitor com níveis de teste"""
        support_levels = [
            {"price": 95.0, "composite_score": 7.5},
            {"price": 90.0, "composite_score": 6.0}
        ]
        resistance_levels = [
            {"price": 105.0, "composite_score": 8.0},
            {"price": 110.0, "composite_score": 6.5}
        ]
        return InstitutionalMarketMonitor(
            support_levels=support_levels,
            resistance_levels=resistance_levels
        )
    
    def test_process_tick_no_signal(self, monitor_with_levels):
        """Tick longe dos níveis não deve gerar sinal"""
        signal = monitor_with_levels.process_tick(
            price=100.0,
            volume=1000.0,
            delta=100.0
        )
        assert signal is None
    
    def test_process_tick_near_support(self, monitor_with_levels):
        """Tick perto de suporte deve gerar sinal"""
        signal = monitor_with_levels.process_tick(
            price=95.1,  # Perto do suporte em 95.0
            volume=1000.0,
            delta=500.0  # Delta positivo (compradores)
        )
        
        assert signal is not None
        assert signal["level_type"] == "SUPPORT"
        assert signal["level_price"] == 95.0
    
    def test_process_tick_near_resistance(self, monitor_with_levels):
        """Tick perto de resistência deve gerar sinal"""
        signal = monitor_with_levels.process_tick(
            price=104.9,  # Perto da resistência em 105.0
            volume=1000.0,
            delta=-500.0  # Delta negativo (vendedores)
        )
        
        assert signal is not None
        assert signal["level_type"] == "RESISTANCE"
        assert signal["level_price"] == 105.0
    
    def test_deque_limits(self, monitor_with_levels):
        """Históricos devem respeitar limites"""
        config = MonitorConfig(lookback_ticks=10, max_test_history=5)
        monitor = InstitutionalMarketMonitor(
            support_levels=[{"price": 100.0, "composite_score": 5.0}],
            resistance_levels=[],
            config=config
        )
        
        # Processar muitos ticks
        for i in range(20):
            monitor.process_tick(100.0 + (i % 2) * 0.1, 1000.0, 0.0)
        
        # Verificar limites
        assert len(monitor.price_history) <= 10
        assert len(monitor.level_tests) <= 5
    
    def test_reset(self, monitor_with_levels):
        """Reset deve limpar históricos"""
        monitor_with_levels.process_tick(95.0, 1000.0, 100.0)
        
        assert len(monitor_with_levels.level_tests) > 0
        
        monitor_with_levels.reset()
        
        assert len(monitor_with_levels.level_tests) == 0
        assert len(monitor_with_levels.price_history) == 0
        assert monitor_with_levels.stats["total_tests"] == 0
    
    def test_update_levels(self, monitor_with_levels):
        """update_levels deve atualizar níveis"""
        new_supports = [{"price": 92.0, "composite_score": 8.0}]
        new_resistances = [{"price": 108.0, "composite_score": 7.0}]
        
        monitor_with_levels.update_levels(new_supports, new_resistances)
        
        assert monitor_with_levels.support_levels == new_supports
        assert monitor_with_levels.resistance_levels == new_resistances
    
    def test_summary_report(self, monitor_with_levels):
        """Relatório resumido deve ter estrutura correta"""
        # Gerar alguns testes
        monitor_with_levels.process_tick(95.0, 1000.0, 100.0)
        monitor_with_levels.process_tick(105.0, 1000.0, -100.0)
        
        report = monitor_with_levels.get_summary_report()
        
        assert "status" in report
        assert "stats" in report
        assert "success_rate" in report
        assert report["stats"]["total_tests"] >= 2


# ============================================================================
# TESTES PARA STATISTICAL UTILS
# ============================================================================

class TestStatisticalUtils:
    """Testes para StatisticalUtils"""
    
    def test_clamp(self):
        """clamp deve limitar valores corretamente"""
        assert StatisticalUtils.clamp(5.0, 0.0, 10.0) == 5.0
        assert StatisticalUtils.clamp(-1.0, 0.0, 10.0) == 0.0
        assert StatisticalUtils.clamp(15.0, 0.0, 10.0) == 10.0
        assert StatisticalUtils.clamp(np.nan, 0.0, 10.0) == 0.0
        assert StatisticalUtils.clamp(np.inf, 0.0, 10.0) == 0.0
    
    def test_safe_divide(self):
        """safe_divide deve evitar divisão por zero"""
        assert StatisticalUtils.safe_divide(10.0, 2.0) == 5.0
        assert StatisticalUtils.safe_divide(10.0, 0.0) == 0.0
        assert StatisticalUtils.safe_divide(10.0, 0.0, default=1.0) == 1.0
        assert StatisticalUtils.safe_divide(10.0, np.nan) == 0.0
    
    def test_confidence_interval(self):
        """Intervalo de confiança deve ser calculado corretamente"""
        data = np.array([100.0, 101.0, 99.0, 100.5, 99.5])
        result = StatisticalUtils.calculate_confidence_interval(data)
        
        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "stability_score" in result
        
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]
        assert 0.0 <= result["stability_score"] <= 10.0
    
    def test_confidence_interval_empty(self):
        """CI com array vazio deve retornar valores padrão"""
        result = StatisticalUtils.calculate_confidence_interval(np.array([]))
        
        assert result["mean"] == 0.0
        assert result["sample_size"] == 0
    
    def test_confidence_interval_single_value(self):
        """CI com único valor deve retornar intervalo degenerado"""
        result = StatisticalUtils.calculate_confidence_interval(np.array([100.0]))
        
        assert result["mean"] == 100.0
        assert result["ci_lower"] == 100.0
        assert result["ci_upper"] == 100.0
        assert result["ci_width"] == 0.0
    
    def test_cluster_prices(self):
        """cluster_prices deve agrupar preços corretamente"""
        prices = np.array([100.0, 100.1, 100.2, 110.0, 110.1, 110.2])
        clusters = StatisticalUtils.cluster_prices(prices, eps_percent=0.02, min_cluster_size=2)
        
        assert len(clusters) == 2
        assert len(clusters[0]) == 3
        assert len(clusters[1]) == 3
    
    def test_cache_thread_safety(self):
        """Cache deve ser thread-safe"""
        import threading
        
        errors = []
        
        def worker():
            try:
                for _ in range(100):
                    data = np.random.random(10)
                    StatisticalUtils.bootstrap_ci(data, n=10)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_cache_clear(self):
        """Cache deve ser limpo corretamente"""
        data = np.array([1.0, 2.0, 3.0])
        StatisticalUtils.bootstrap_ci(data, n=10)
        
        StatisticalUtils.clear_cache()
        
        # Não deve lançar erro
        StatisticalUtils.bootstrap_ci(data, n=10)


# ============================================================================
# TESTES PARA EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Testes para casos especiais e edge cases"""
    
    def test_single_row_dataframe(self):
        """Sistema deve lidar com DataFrame de uma linha"""
        single_df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000.0]
        })
        
        system = InstitutionalSupportResistanceSystem()
        
        with pytest.raises(ValueError):
            system.analyze_market(single_df)
    
    def test_very_small_price_range(self):
        """Range de preço muito pequeno"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = np.full(len(dates), 100.0) + np.random.normal(0, 0.001, len(dates))
        
        df_small_range = pd.DataFrame({
            'open': prices - 0.0005,
            'high': prices + 0.001,
            'low': prices - 0.001,
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates)) * 1000
        }, index=dates)
        
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(df_small_range)
        
        assert result["sr_analysis"]["current_price"] > 0
    
    def test_very_large_price_range(self):
        """Range de preço muito grande"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = np.linspace(10, 1000, len(dates))
        
        df_large_range = pd.DataFrame({
            'open': prices - 5,
            'high': prices + 10,
            'low': prices - 9,  # ✅ Corrigido: 10 - 9 = 1.0 (válido!)
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates)) * 1000
        }, index=dates)
        
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(df_large_range)
        
        assert result["sr_analysis"]["current_price"] > 0
    
    def test_extreme_volume_values(self):
        """Valores de volume extremos"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = 100 + np.random.normal(0, 5, len(dates))
        
        df_extreme_volume = pd.DataFrame({
            'open': prices - 1,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.lognormal(20, 3, len(dates)) * 1000000
        }, index=dates)
        
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(df_extreme_volume)
        
        assert "sr_analysis" in result


# ============================================================================
# TESTES DE INTEGRAÇÃO
# ============================================================================

class TestIntegration:
    """Testes de integração"""
    
    def test_full_pipeline(self, sample_df):
        """Pipeline completo deve executar todas as etapas"""
        # Etapa 1: Validação
        validate_dataframe(sample_df)
        
        # Etapa 2: Análise
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        # Etapa 3: Monitor
        monitor = system.create_market_monitor()
        
        # Etapa 4: Processar ticks
        for i in range(10):
            price = sample_df["close"].iloc[-1] + np.random.uniform(-1, 1)
            monitor.process_tick(price, 1000.0, np.random.uniform(-100, 100))
        
        # Verificar resultados
        report = monitor.get_summary_report()
        assert report["stats"]["total_tests"] >= 0
    
    def test_multiple_analyses_different_configs(self, sample_df):
        """Múltiplas análises com configurações diferentes"""
        config1 = InstitutionalConfig(
            sr=SRConfig(lookback_period=50, merge_tolerance=0.02),
            volume_profile=VolumeProfileConfig(bins=20),
            enable_cache=False
        )
        
        config2 = InstitutionalConfig(
            sr=SRConfig(lookback_period=100, merge_tolerance=0.005),
            volume_profile=VolumeProfileConfig(bins=80),
            enable_cache=False
        )
        
        system1 = InstitutionalSupportResistanceSystem(config=config1)
        system2 = InstitutionalSupportResistanceSystem(config=config2)
        
        result1 = system1.analyze_market(sample_df)
        result2 = system2.analyze_market(sample_df)
        
        assert "sr_analysis" in result1
        assert "sr_analysis" in result2
    
    def test_system_reproducibility(self, sample_df):
        """Análises com mesma seed devem ser reprodutíveis"""
        np.random.seed(42)
        system1 = InstitutionalSupportResistanceSystem()
        result1 = system1.analyze_market(sample_df)
        
        np.random.seed(42)
        system2 = InstitutionalSupportResistanceSystem()
        result2 = system2.analyze_market(sample_df)
        
        # Preços atuais devem ser iguais
        assert result1["sr_analysis"]["current_price"] == result2["sr_analysis"]["current_price"]


# ============================================================================
# TESTES PARA MÉTRICAS ESPECÍFICAS
# ============================================================================

class TestSpecificMetrics:
    """Testes para métricas específicas do sistema"""
    
    def test_support_below_current_price(self, sample_df):
        """Suportes devem estar abaixo ou no preço atual"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        sr = result["sr_analysis"]
        current_price = sr["current_price"]
        tolerance = current_price * 0.002
        
        for support in sr["support_levels"]:
            assert support["price"] <= current_price + tolerance
    
    def test_resistance_above_current_price(self, sample_df):
        """Resistências devem estar acima ou no preço atual"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        sr = result["sr_analysis"]
        current_price = sr["current_price"]
        tolerance = current_price * 0.002
        
        for resistance in sr["resistance_levels"]:
            assert resistance["price"] >= current_price - tolerance
    
    def test_poc_in_value_area(self, sample_df):
        """POC deve estar dentro da Value Area"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        vp = result["volume_profile"]
        poc_price = vp["poc"]["price"]
        va_low = vp["value_area"]["low"]
        va_high = vp["value_area"]["high"]
        
        # POC deve estar dentro (com tolerância para arredondamentos)
        tolerance = (va_high - va_low) * 0.01
        assert va_low - tolerance <= poc_price <= va_high + tolerance
    
    def test_quality_rating_values(self, sample_df):
        """Quality rating deve ter valores válidos"""
        system = InstitutionalSupportResistanceSystem()
        result = system.analyze_market(sample_df)
        
        sr = result["sr_analysis"]
        quality_report = sr["quality_report"]
        
        valid_ratings = [r.value for r in QualityRating]
        assert quality_report["quality_rating"] in valid_ratings
        
        assert 0.0 <= quality_report["overall_quality"] <= 10.0


# ============================================================================
# RUN DOS TESTES
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])