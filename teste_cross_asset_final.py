#!/usr/bin/env python3
"""
Teste final completo da implementação de Cross-Asset Correlations
"""
import sys
import os
from datetime import datetime, timezone
import json

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cross_asset_correlations import get_btc_eth_correlations, get_btc_macro_correlations
from ml_features import calculate_cross_asset_features
from market_orchestrator.ai.ai_payload_builder import build_payload_with_cross_asset

def teste_completo():
    """Teste completo da funcionalidade cross-asset"""
    print("=== TESTE FINAL - Cross-Asset Correlations ===\n")
    
    # 1. Teste das funcoes de correlacao
    print("1. Testando get_btc_eth_correlations...")
    try:
        now_utc = datetime.now(timezone.utc)
        btc_eth = get_btc_eth_correlations(now_utc)
        print(f"[OK] BTC-ETH correlations: {btc_eth}")
    except Exception as e:
        print(f"[ERRO] Erro em BTC-ETH correlations: {e}")
        return False
    
    print("\n2. Testando get_btc_macro_correlations...")
    try:
        btc_macro = get_btc_macro_correlations(now_utc)
        print(f"[OK] BTC-Macro correlations: {btc_macro}")
    except Exception as e:
        print(f"[ERRO] Erro em BTC-Macro correlations: {e}")
        return False
    
    # 3. Teste das features ML
    print("\n3. Testando calculate_cross_asset_features...")
    try:
        features = calculate_cross_asset_features(now_utc)
        print(f"[OK] Features calculadas: {len(features)} features")
        print("Primeiras 5 features:")
        for key, value in list(features.items())[:5]:
            print(f"  - {key}: {value}")
    except Exception as e:
        print(f"[ERRO] Erro em calculate_cross_asset_features: {e}")
        return False
    
    # 4. Teste de integração completa
    print("\n4. Testando build_payload_with_cross_asset...")
    try:
        # Simular um evento BTCUSDT
        symbol = "BTCUSDT"
        event_type = "AI_ANALYSIS"
        timestamp = int(now_utc.timestamp() * 1000)
        
        # Dados base do payload
        base_payload = {
            "event_type": event_type,
            "symbol": symbol,
            "timestamp": timestamp,
            "market_data": {
                "price": 45000.0,
                "volume": 100.0
            },
            "technical_analysis": {
                "rsi": 65.5,
                "macd": 120.5
            },
            "flow_analysis": {
                "trend": "bullish",
                "momentum": 0.75
            }
        }
        
        # Build do payload completo
        final_payload = build_payload_with_cross_asset(
            symbol=symbol,
            event_type=event_type,
            timestamp=timestamp,
            base_payload=base_payload
        )
        
        print(f"[OK] Payload construido com sucesso!")
        print(f"Payload size: {len(json.dumps(final_payload))} bytes")
        
        # Verificar se o cross_asset_context foi adicionado
        if "cross_asset_context" in final_payload:
            print("[OK] cross_asset_context presente no payload!")
            print("Features incluídas:")
            for key in final_payload["cross_asset_context"].get("features", {}).keys():
                print(f"  - {key}")
        else:
            print("[ERRO] cross_asset_context NAO encontrado no payload!")
            return False
            
    except Exception as e:
        print(f"[ERRO] Erro no build_payload_with_cross_asset: {e}")
        return False
    
    print("\n=== TODOS OS TESTES PASSARAM! ===")
    return True

def test_integration_with_ml_features():
    """Teste de integração com ml_features"""
    print("\n=== TESTE DE INTEGRAÇÃO COM ML_FEATURES ===")
    
    # Verificar se a nova função foi adicionada corretamente
    try:
        from ml_features import calculate_cross_asset_features
        print("[OK] calculate_cross_asset_features importada com sucesso!")
        
        # Verificar se a função está registrada no ANALYSIS_TRIGGER
        now_utc = datetime.now(timezone.utc)
        features = calculate_cross_asset_features(now_utc)
        
        expected_features = [
            "btc_eth_corr_7d",
            "btc_eth_corr_30d", 
            "btc_dxy_corr_7d",
            "btc_dxy_corr_30d",
            "btc_ndx_corr_7d",
            "btc_ndx_corr_30d"
        ]
        
        print(f"Features esperadas: {expected_features}")
        print(f"Features calculadas: {list(features.keys())}")
        
        for feature in expected_features:
            if feature in features:
                print(f"[OK] {feature}: {features[feature]}")
            else:
                print(f"[ERRO] {feature}: NAO ENCONTRADO!")
                
    except ImportError as e:
        print(f"[ERRO] Erro de importacao: {e}")
        return False
    except Exception as e:
        print(f"[ERRO] Erro na integracao: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success1 = teste_completo()
    success2 = test_integration_with_ml_features()
    
    if success1 and success2:
        print("\n[OK] IMPLEMENTACAO COMPLETA E FUNCIONAL!")
        sys.exit(0)
    else:
        print("\n[ERRO] ALGUNS TESTES FALHARAM!")
        sys.exit(1)