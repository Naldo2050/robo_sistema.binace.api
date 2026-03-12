#!/usr/bin/env python3
"""
Teste das funcionalidades de correlação cross-asset
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import logging

# Adicionar o diretório ao path para importações
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_correlation_functions():
    """Testa as funções básicas de correlação"""
    import numpy as np
    import pandas as pd
    from cross_asset_correlations import _log_returns, _corr_last_window
    
    # Criar dados sintéticos para teste
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='h')
    np.random.seed(42)
    
    # Série sintética 1
    prices1 = 50000 + np.cumsum(np.random.randn(len(dates)) * 100)
    series1 = pd.Series(prices1, index=dates)
    
    # Série sintética 2 (correlacionada com 1)
    prices2 = 3000 + 0.8 * np.cumsum(np.random.randn(len(dates)) * 50) + 0.2 * np.cumsum(np.random.randn(len(dates)) * 100)
    series2 = pd.Series(prices2, index=dates)
    
    # Testar log returns
    log_returns1 = _log_returns(series1)
    log_returns2 = _log_returns(series2)
    
    print(f"Log returns série 1: {len(log_returns1)} pontos")
    print(f"Log returns série 2: {len(log_returns2)} pontos")
    
    # Testar correlação
    corr_24h = _corr_last_window(log_returns1, log_returns2, 24)
    corr_168h = _corr_last_window(log_returns1, log_returns2, 168)
    
    print(f"Correlação 24h: {corr_24h:.4f}")
    print(f"Correlação 168h (7d): {corr_168h:.4f}")
    
    return True

def test_btc_eth_correlation():
    """Testa a função principal de correlação BTC-ETH"""
    import pandas as pd
    from cross_asset_correlations import get_btc_eth_correlations
    
    try:
        # Testar com datetime atual
        now_utc = datetime.now(timezone.utc)
        print(f"Testando correlacao BTC-ETH para {now_utc}")
        
        result = get_btc_eth_correlations(now_utc)
        print(f"Resultado: {result}")
        
        # Verificar se os valores estao validos
        for key, value in result.items():
            if pd.isna(value):
                print(f"[AVISO] {key} eh NaN")
            else:
                if isinstance(value, (int, float)):
                    print(f"[OK] {key}: {value:.4f}")
                else:
                    print(f"[OK] {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Erro no teste BTC-ETH: {e}")
        return False

def test_btc_macro_correlations():
    """Testa a função de correlação BTC com indicadores macro"""
    import pandas as pd
    from cross_asset_correlations import get_btc_macro_correlations
    
    try:
        # Testar com datetime atual
        now_utc = datetime.now(timezone.utc)
        print(f"Testando correlacao BTC-Macro para {now_utc}")
        
        result = get_btc_macro_correlations(now_utc)
        print(f"Resultado: {result}")
        
        # Verificar se os valores estao validos
        for key, value in result.items():
            if pd.isna(value):
                print(f"[AVISO] {key} eh NaN")
            else:
                if isinstance(value, (int, float)):
                    print(f"[OK] {key}: {value:.4f}")
                else:
                    print(f"[OK] {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"[ERRO] Erro no teste BTC-Macro: {e}")
        return False

def test_all_correlations():
    """Testa todas as funcionalidades"""
    print("=" * 60)
    print("TESTE DAS FUNCIONALIDADES DE CORRELAÇÃO CROSS-ASSET")
    print("=" * 60)
    
    # Teste 1: Funções básicas
    print("\n1. Testando funções básicas...")
    if test_correlation_functions():
        print("[OK] Funcoes basicas")
    else:
        print("[ERRO] Funcoes basicas: FALHOU")
        return False
    
    # Teste 2: BTC-ETH
    print("\n2. Testando correlação BTC-ETH...")
    if test_btc_eth_correlation():
        print("[OK] Correlacao BTC-ETH: OK")
    else:
        print("[ERRO] Correlacao BTC-ETH: FALHOU")
        return False
    
    # Teste 3: BTC-Macro
    print("\n3. Testando correlação BTC-Macro...")
    if test_btc_macro_correlations():
        print("[OK] Correlacao BTC-Macro: OK")
    else:
        print("[ERRO] Correlacao BTC-Macro: FALHOU")
        return False
    
    print("\n" + "=" * 60)
    print("TODOS OS TESTES CONCLUIDOS")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_all_correlations()
    sys.exit(0 if success else 1)