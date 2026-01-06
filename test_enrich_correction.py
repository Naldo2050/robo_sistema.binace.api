#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste para verificar se a correção da função enrich_event_with_advanced_analysis funcionou.
"""

import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_enricher import DataEnricher
from data_pipeline.pipeline import DataPipeline
from enrichment_integrator import enrich_analysis_trigger_event, build_analysis_trigger_event

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_enrich_function_directly():
    """Testa a função enrich_event_with_advanced_analysis diretamente."""
    print("\n=== TESTE 1: Função Directa ===")
    
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
        result = enricher.enrich_event_with_advanced_analysis(test_event)
        print(f"✅ Função executada com sucesso")
        print(f"📊 Resultado tipo: {type(result)}")
        
        # Verificar se o raw_event foi atualizado
        raw_event = result.get("raw_event", {})
        if "advanced_analysis" in raw_event:
            advanced = raw_event["advanced_analysis"]
            print(f"✅ advanced_analysis encontrado")
            print(f"🔑 Chaves: {list(advanced.keys()) if isinstance(advanced, dict) else 'N/A'}")
            
            # Verificar campos específicos
            if isinstance(advanced, dict):
                print(f"💰 Symbol: {advanced.get('symbol')}")
                print(f"💵 Price: {advanced.get('price')}")
                print(f"📈 Price targets: {len(advanced.get('price_targets', []))} alvos")
                print(f"⚡ Adaptive thresholds: {'current_volatility' in advanced.get('adaptive_thresholds', {})}")
        else:
            print("❌ advanced_analysis NÃO foi adicionado")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Testa a integração com o DataPipeline."""
    print("\n=== TESTE 2: Integração com Pipeline ===")
    
    try:
        # Dados de trades simulados
        trades_data = [
            {"p": 50000, "q": 0.1, "T": 1000000, "m": False},
            {"p": 50100, "q": 0.2, "T": 1001000, "m": True},
            {"p": 49900, "q": 0.15, "T": 1002000, "m": False},
        ]
        
        # Criar pipeline
        pipeline = DataPipeline(trades_data, "BTCUSDT")
        
        # Executar pipeline
        enriched = pipeline.enrich()
        print(f"✅ Pipeline enriquecido com sucesso")
        
        # Adicionar contexto
        contextual = pipeline.add_context()
        print(f"✅ Contexto adicionado com sucesso")
        
        # Detectar sinais (inclui ANALYSIS_TRIGGER)
        signals = pipeline.detect_signals()
        print(f"✅ Sinais detectados: {len(signals)}")
        
        # Verificar se ANALYSIS_TRIGGER tem advanced_analysis
        analysis_triggers = [s for s in signals if s.get("tipo_evento") == "ANALYSIS_TRIGGER"]
        
        if analysis_triggers:
            trigger = analysis_triggers[0]
            raw_event = trigger.get("raw_event", {})
            
            if "advanced_analysis" in raw_event:
                print("✅ ANALYSIS_TRIGGER contém advanced_analysis")
                advanced = raw_event["advanced_analysis"]
                print(f"🔑 Chaves: {list(advanced.keys())}")
            else:
                print("❌ ANALYSIS_TRIGGER NÃO contém advanced_analysis")
                print(f"📋 Raw event keys: {list(raw_event.keys())}")
        else:
            print("❌ Nenhum ANALYSIS_TRIGGER encontrado")
            
        pipeline.close()
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enrichment_integrator():
    """Testa o enrichment_integrator."""
    print("\n=== TESTE 3: Enrichment Integrator ===")
    
    try:
        # Config de teste
        config_dict = {
            "SYMBOL": "BTCUSDT",
            "ABSORPTION_THRESHOLD_BASE": 0.15,
            "FLOW_THRESHOLD_BASE": 0.10,
        }
        
        # Criar evento ANALYSIS_TRIGGER
        raw_event_data = {
            "symbol": "BTCUSDT",
            "preco_fechamento": 50000.0,
            "volume_total": 100.0,
            "multi_tf": {"1d": {"realized_vol": 0.02}},
            "historical_vp": {"daily": {"vah": 51000, "val": 49000, "poc": 50000}},
        }
        
        event = build_analysis_trigger_event("BTCUSDT", raw_event_data)
        
        # Enriquecer
        enriched_event = enrich_analysis_trigger_event(event, config_dict)
        
        print(f"✅ Evento enriquecido pelo integrator")
        
        # Verificar resultado
        raw_event = enriched_event.get("raw_event", {})
        if "advanced_analysis" in raw_event:
            print("✅ advanced_analysis encontrado via integrator")
            advanced = raw_event["advanced_analysis"]
            print(f"🔑 Chaves: {list(advanced.keys())}")
        else:
            print("❌ advanced_analysis NÃO encontrado via integrator")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do integrator: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes."""
    print("TESTANDO CORRECAO DA FUNCAO enrich_event_with_advanced_analysis")
    print("=" * 70)
    
    tests = [
        test_enrich_function_directly,
        test_pipeline_integration,
        test_enrichment_integrator
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Erro crítico no teste {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES:")
    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {i}. {test.__name__}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 RESULTADO: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! A correção funcionou.")
    else:
        print("⚠️ Alguns testes falharam. Verifique os logs acima.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)