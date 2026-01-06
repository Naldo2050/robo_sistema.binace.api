#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste simples para verificar se a correção da função enrich_event_with_advanced_analysis funcionou.
"""

import sys
import os
import logging

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_enricher import DataEnricher

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_enrich_function():
    """Testa a função enrich_event_with_advanced_analysis."""
    print("\n=== TESTE: Funcao enrich_event_with_advanced_analysis ===")
    
    # Criar dados de teste
    config_dict = {
        "SYMBOL": "BTCUSDT",
        "ABSORPTION_THRESHOLD_BASE": 0.15,
        "FLOW_THRESHOLD_BASE": 0.10,
        "MIN_VOL_FACTOR": 0.5,
        "MAX_VOL_FACTOR": 2.0
    }
    
    enricher = DataEnricher(config_dict)
    
    # Evento de teste
    test_event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "raw_event": {
            "symbol": "BTCUSDT",
            "preco_fechamento": 50000.0,
            "volume_total": 100.0,
            "multi_tf": {
                "1d": {"realized_vol": 0.02},
                "4h": {"realized_vol": 0.025}
            },
            "historical_vp": {
                "daily": {
                    "vah": 51000.0,
                    "val": 49000.0,
                    "poc": 50000.0
                }
            },
            "timestamp_utc": "2024-01-01T12:00:00Z"
        }
    }
    
    try:
        print("Chamando enrich_event_with_advanced_analysis...")
        result = enricher.enrich_event_with_advanced_analysis(test_event)
        print(f"Funcao executada com sucesso - tipo: {type(result)}")
        
        # Verificar se o raw_event foi atualizado
        raw_event = result.get("raw_event", {})
        if "advanced_analysis" in raw_event:
            advanced = raw_event["advanced_analysis"]
            print(f"SUCCESS: advanced_analysis encontrado")
            print(f"Chaves: {list(advanced.keys()) if isinstance(advanced, dict) else 'N/A'}")
            
            # Verificar campos específicos
            if isinstance(advanced, dict):
                print(f"Symbol: {advanced.get('symbol')}")
                print(f"Price: {advanced.get('price')}")
                print(f"Price targets: {len(advanced.get('price_targets', []))} alvos")
                print(f"Adaptive thresholds: {'current_volatility' in advanced.get('adaptive_thresholds', {})}")
        else:
            print("ERROR: advanced_analysis NAO foi adicionado")
            print(f"Raw event keys: {list(raw_event.keys())}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Erro na execucao: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa o teste."""
    print("TESTANDO CORRECAO DA FUNCAO enrich_event_with_advanced_analysis")
    print("=" * 60)
    
    success = test_enrich_function()
    
    print("\n" + "=" * 60)
    if success:
        print("RESULTADO: Teste passou - a correcao funcionou!")
    else:
        print("RESULTADO: Teste falhou - verificar logs acima.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)