# test_enhanced_cross_asset.py
"""
Testes para as novas funcionalidades enhanced cross-asset.

Testa:
- VIX metrics
- Treasury Yields 
- Crypto Dominance
- Commodities (Gold, Oil)
- Regime Detection
- Novas correlações
"""

import pytest
import logging
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Importa os módulos a serem testados
try:
    from cross_asset_correlations import (
        get_enhanced_cross_asset_correlations,
        _calculate_correlation_regime,
        _corr_last_window,
        _log_returns
    )
    from macro_data_fetcher import (
        fetch_crypto_dominance,
        fetch_vix_data,
        fetch_treasury_yields,
        fetch_commodities_data,
        calculate_macro_regime,
        calculate_correlation_regime
    )
    from ml_features import calculate_cross_asset_features, generate_ml_features
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


class TestEnhancedCrossAsset:
    """Testes para funcionalidades enhanced cross-asset."""
    
    def setup_method(self):
        """Setup para cada teste."""
        logging.basicConfig(level=logging.DEBUG)
    
    def test_correlation_regime_calculation(self):
        """Testa cálculo de regime de correlação."""
        # Correlação inversa forte
        regime = _calculate_correlation_regime(-0.6)
        assert regime == "INVERSE"
        
        # Correlação baixa
        regime = _calculate_correlation_regime(0.1)
        assert regime == "DECORRELATED"
        
        # Correlação positiva
        regime = _calculate_correlation_regime(0.5)
        assert regime == "CORRELATED"
        
        # Valor inválido
        regime = _calculate_correlation_regime(float('nan'))
        assert regime == "UNKNOWN"
    
    def test_enhanced_correlations_structure(self):
        """Testa estrutura básica das enhanced correlations."""
        # Mock dos dados macro
        with patch('cross_asset_correlations._MACRO_DATA_OK', False):
            result = get_enhanced_cross_asset_correlations()
            
            # Deve ter status
            assert 'status' in result
            assert 'timestamp' in result
            
            # Deve incluir métricas tradicionais
            expected_traditional = [
                'btc_eth_corr_7d', 'btc_eth_corr_30d',
                'btc_dxy_corr_30d', 'btc_dxy_corr_90d',
                'dxy_return_5d', 'dxy_return_20d'
            ]
            
            for metric in expected_traditional:
                assert metric in result, f"Métrica tradicional {metric} ausente"
    
    def test_vix_data_structure(self):
        """Testa estrutura dos dados VIX."""
        with patch('macro_data_fetcher.requests.get') as mock_get:
            # Mock da resposta da CoinGecko
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": {
                    "market_cap_percentage": {
                        "btc": 45.2,
                        "eth": 18.7,
                        "usdt": 7.1
                    }
                }
            }
            mock_get.return_value = mock_response
            
            result = fetch_crypto_dominance()
            
            assert result['status'] == 'ok'
            assert result['btc_dominance'] == 45.2
            assert result['eth_dominance'] == 18.7
            assert result['usdt_dominance'] == 7.1
            assert 'timestamp' in result
    
    @patch('macro_data_fetcher._fetch_yfinance_data_with_fallbacks')
    def test_vix_metrics_calculation(self, mock_yf):
        """Testa cálculo de métricas VIX."""
        # Mock DataFrame do VIX
        import pandas as pd
        mock_df = pd.DataFrame({
            'close': [20.5, 21.2, 22.8, 23.1, 24.5]
        })
        mock_yf.return_value = mock_df
        
        result = fetch_vix_data()
        
        assert result['status'] == 'ok'
        assert result['vix_current'] == 24.5
        assert abs(result['vix_change_1d'] - 5.71) < 0.1  # (24.5-23.1)/23.1 * 100
        assert not result['historical'].empty
    
    @patch('macro_data_fetcher._fetch_yfinance_data_with_fallbacks')
    def test_treasury_yields_calculation(self, mock_yf):
        """Testa cálculo de Treasury Yields."""
        import pandas as pd
        
        # Mock DataFrames para yields
        mock_10y = pd.DataFrame({'close': [4.25, 4.30, 4.28, 4.32, 4.35]})
        mock_2y = pd.DataFrame({'close': [4.80, 4.85, 4.83, 4.87, 4.90]})
        
        def mock_fallback(name, period, interval):
            if name == "US10Y":
                return mock_10y
            elif name == "US2Y":
                return mock_2y
            return pd.DataFrame()
        
        mock_yf.side_effect = mock_fallback
        
        result = fetch_treasury_yields()
        
        assert result['status'] == 'ok'
        assert result['us10y_yield'] == 4.35
        assert result['us2y_yield'] == 4.90
        assert abs(result['us10y_change_1d'] - 0.69) < 0.1  # (4.35-4.32)/4.32 * 100
        assert abs(result['us2y_change_1d'] - 0.62) < 0.1   # (4.90-4.87)/4.87 * 100
    
    def test_macro_regime_calculation(self):
        """Testa cálculo de regime macro."""
        vix_data = {"vix_current": 28.5}  # High VIX = risk off
        dominance_data = {"btc_dominance": 52.0}  # High BTC dominance = risk off
        treasury_data = {"us10y_change_1d": 0.1}  # Rising yields = risk off
        
        regime = calculate_macro_regime(vix_data, dominance_data, treasury_data)
        assert regime == "RISK_OFF"
        
        # Teste risk on
        vix_data_low = {"vix_current": 12.5}  # Low VIX = risk on
        dominance_data_low = {"btc_dominance": 38.0}  # Low BTC dominance = risk on
        treasury_data_down = {"us10y_change_1d": -0.1}  # Falling yields = risk on
        
        regime = calculate_macro_regime(vix_data_low, dominance_data_low, treasury_data_down)
        assert regime == "RISK_ON"
    
    def test_commodities_data_structure(self):
        """Testa estrutura dos dados de commodities."""
        with patch('macro_data_fetcher._fetch_yfinance_data_with_fallbacks') as mock_yf:
            import pandas as pd
            
            # Mock DataFrames para commodities
            mock_gold = pd.DataFrame({'close': [2000, 2010, 1995, 2005, 2015]})
            mock_oil = pd.DataFrame({'close': [75, 76, 74, 75, 77]})
            
            def mock_fallback(name, period, interval):
                if name == "GOLD":
                    return mock_gold
                elif name == "OIL":
                    return mock_oil
                return pd.DataFrame()
            
            mock_yf.side_effect = mock_fallback
            
            result = fetch_commodities_data()
            
            assert result['status'] == 'ok'
            assert result['gold_price'] == 2015
            assert result['oil_price'] == 77
            assert abs(result['gold_change_1d'] - 0.5) < 0.1  # (2015-2005)/2005 * 100
            assert abs(result['oil_change_1d'] - 2.67) < 0.1  # (77-75)/75 * 100
    
    def test_ml_features_integration(self):
        """Testa integração com ml_features."""
        # Mock dos dados de correlação
        mock_correlations = {
            "status": "ok",
            "btc_eth_corr_7d": 0.85,
            "btc_eth_corr_30d": 0.78,
            "btc_dxy_corr_30d": -0.45,
            "dxy_return_5d": 1.2,
            "vix_current": 25.5,
            "btc_dominance": 48.5,
            "gold_price": 2000.0,
            "macro_regime": "RISK_OFF",
            "correlation_regime": "INVERSE"
        }
        
        with patch('ml_features.get_cross_asset_features') as mock_get:
            mock_get.return_value = mock_correlations
            
            features = calculate_cross_asset_features("BTCUSDT")
            
            # Verifica métricas tradicionais
            assert features["btc_eth_corr_7d"] == 0.85
            assert features["btc_dxy_corr_30d"] == -0.45
            
            # Verifica novas métricas
            assert features["vix_current"] == 25.5
            assert features["btc_dominance"] == 48.5
            assert features["gold_price"] == 2000.0
            assert features["macro_regime"] == "RISK_OFF"
            assert features["correlation_regime"] == "INVERSE"
    
    def test_enhanced_features_count(self):
        """Testa se enhanced features estão sendo contadas corretamente."""
        with patch('ml_features.get_cross_asset_features') as mock_get:
            # Mock com todas as novas métricas
            mock_correlations = {
                "status": "ok",
                "btc_eth_corr_7d": 0.85,
                "btc_eth_corr_30d": 0.78,
                "btc_dxy_corr_30d": -0.45,
                "btc_dxy_corr_90d": -0.42,
                "btc_ndx_corr_30d": 0.65,
                "dxy_return_5d": 1.2,
                "dxy_return_20d": 2.1,
                # Novas métricas
                "vix_current": 25.5,
                "vix_change_1d": 2.1,
                "btc_vix_corr_30d": -0.35,
                "us10y_yield": 4.25,
                "us10y_change_1d": 0.05,
                "us2y_yield": 4.80,
                "us2y_change_1d": 0.03,
                "btc_yields_corr_30d": -0.28,
                "btc_dominance": 48.5,
                "btc_dominance_change_7d": 0.5,
                "eth_dominance": 18.2,
                "usdt_dominance": 7.1,
                "gold_price": 2000.0,
                "gold_change_1d": 0.5,
                "btc_gold_corr_30d": 0.15,
                "oil_price": 76.5,
                "oil_change_1d": 1.2,
                "btc_oil_corr_30d": 0.22,
                "macro_regime": "RISK_OFF",
                "correlation_regime": "INVERSE"
            }
            
            mock_get.return_value = mock_correlations
            
            features = calculate_cross_asset_features("BTCUSDT")
            
            # Deve ter pelo menos 25+ features
            assert len(features) >= 25
            
            # Verifica algumas métricas-chave
            expected_metrics = [
                "btc_eth_corr_7d", "btc_dxy_corr_30d", "dxy_return_5d",
                "vix_current", "btc_dominance", "gold_price", 
                "macro_regime", "correlation_regime"
            ]
            
            for metric in expected_metrics:
                assert metric in features, f"Métrica {metric} ausente"
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Testa quando macro_data_fetcher não está disponível
        with patch('cross_asset_correlations._MACRO_DATA_OK', False):
            result = get_enhanced_cross_asset_correlations()
            assert 'enhanced_status' in result
            assert result['enhanced_status'] == 'unavailable'
    
    def test_fallback_behavior(self):
        """Testa comportamento de fallback."""
        # Mock falha no yfinance
        with patch('macro_data_fetcher._fetch_yfinance_data_with_fallbacks') as mock_yf:
            mock_yf.return_value = None  # Dados vazios
            
            result = fetch_vix_data()
            
            assert result['status'] == 'failed'
            assert result['vix_current'] is None
            assert result['vix_change_1d'] is None


def run_integration_test():
    """Teste de integração completo."""
    print("\n" + "="*80)
    print("🧪 TESTE DE INTEGRAÇÃO - ENHANCED CROSS-ASSET")
    print("="*80 + "\n")
    
    try:
        # Teste 1: Crypto Dominance
        print("📊 Testando Crypto Dominance...")
        dominance = fetch_crypto_dominance()
        print(f"  Status: {dominance['status']}")
        if dominance['status'] == 'ok':
            print(f"  BTC Dominance: {dominance['btc_dominance']:.2f}%")
            print(f"  ETH Dominance: {dominance['eth_dominance']:.2f}%")
            print(f"  USDT Dominance: {dominance['usdt_dominance']:.2f}%")
        
        # Teste 2: Enhanced Correlations
        print("\n📊 Testando Enhanced Correlations...")
        correlations = get_enhanced_cross_asset_correlations()
        print(f"  Status: {correlations['status']}")
        
        traditional_count = sum(1 for k in correlations.keys() 
                              if k in ['btc_eth_corr_7d', 'btc_eth_corr_30d', 'btc_dxy_corr_30d', 
                                      'btc_dxy_corr_90d', 'dxy_return_5d', 'dxy_return_20d'])
        
        enhanced_count = len(correlations) - traditional_count - 2  # -2 para status e timestamp
        
        print(f"  Total Features: {len(correlations)}")
        print(f"  Traditional: {traditional_count}")
        print(f"  Enhanced: {enhanced_count}")
        
        # Teste 3: ML Features Integration
        print("\n📊 Testando ML Features Integration...")
        ml_features = calculate_cross_asset_features("BTCUSDT")
        print(f"  ML Features Count: {len(ml_features)}")
        
        print("\n✅ Teste de Integração Concluído!")
        
    except Exception as e:
        print(f"\n❌ Erro no teste de integração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Executa teste de integração quando rodado diretamente
    run_integration_test()