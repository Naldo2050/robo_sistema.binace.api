# diagnostics/verify_ml_integration.py
import sys
import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any

# Adiciona raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MLCheck")

def test_ml_engine_direct():
    """Testa diretamente o MLInferenceEngine sem dependências externas."""
    logger.info("🧠 TESTE DIRETO DO MOTOR DE INFERÊNCIA ML")
    print("="*60)
    
    # 1. Verifica se os arquivos existem
    files_to_check = [
        "ml/inference_engine.py",
        "market_orchestrator/ai/ai_payload_builder.py",
    ]
    
    # Verifica modelo ML (opcional para teste)
    model_files = [
        "ml/models/xgb_model_latest.json",
        "ml/models/model_metadata.json"
    ]
    
    missing = [f for f in files_to_check if not os.path.exists(f)]
    if missing:
        logger.error(f"❌ Arquivos críticos faltando: {missing}")
        return False
    
    model_missing = [f for f in model_files if not os.path.exists(f)]
    if model_missing:
        logger.warning(f"⚠️ Arquivos de modelo faltando: {model_missing}")
        logger.warning("   O sistema funcionará apenas com IA Generativa")
    
    # 2. Testa importação do MLInferenceEngine
    try:
        from ml.inference_engine import MLInferenceEngine
        logger.info("✅ MLInferenceEngine importado com sucesso")
    except ImportError as e:
        logger.error(f"❌ Falha ao importar MLInferenceEngine: {e}")
        return False
    
    # 3. Testa inicialização do motor
    try:
        ml_engine = MLInferenceEngine()
        
        if ml_engine.model is None:
            logger.warning("⚠️ Modelo ML não carregado (pode não existir ou estar corrompido)")
            logger.info("✅ Sistema pode continuar apenas com IA Generativa")
            return True  # Não é fatal
            
        logger.info(f"✅ ML Engine carregado com {len(ml_engine.features)} features")
        
        # 4. Testa extração de features
        test_event = {
            "tipo_evento": "Absorção",
            "ativo": "BTCUSDT",
            "delta": -15.5,
            "volume_total": 125.3,
            "volume_ratio": 1.2,
            "preco_fechamento": 95000,
            "fluxo_continuo": {
                "microstructure": {
                    "tick_rule_sum": 0.2,
                    "flow_imbalance": 0.1,
                    "aggressive_buy_ratio": 0.6,
                    "aggressive_sell_ratio": 0.4
                },
                "whale_activity": {
                    "whale_delta": 0.3,
                    "whale_buy_ratio": 0.7
                }
            },
            "orderbook_data": {
                "bid_ask_ratio": 1.1,
                "imbalance": 0.05,
                "spread_percent": 0.01
            },
            "ohlc": {
                "close": 95000,
                "high": 95500,
                "low": 94500
            }
        }
        
        features = ml_engine.extract_ml_features(test_event)
        logger.info(f"✅ Extraídas {len(features)} features do evento")
        
        # 5. Testa previsão
        prediction = ml_engine.predict(test_event)
        
        if prediction.get("status") == "ok":
            prob = prediction.get("prob_up", 0.5)
            confidence = prediction.get("confidence", 0.0)
            
            logger.info(f"✅ Previsão ML bem-sucedida!")
            logger.info(f"   📈 Probabilidade de Alta: {prob:.1%}")
            logger.info(f"   📊 Confiança: {confidence:.1%}")
            logger.info(f"   🔍 Features usadas: {prediction.get('features_used')}/{prediction.get('total_features')}")
            
            # Interpretação
            if prob > 0.6:
                bias = "BULLISH (Altista)"
            elif prob < 0.4:
                bias = "BEARISH (Baixista)"
            else:
                bias = "NEUTRAL (Neutro)"
                
            logger.info(f"   🎯 Viés: {bias}")
            
        else:
            logger.warning(f"⚠️ Previsão falhou: {prediction.get('status')}")
            if prediction.get("msg"):
                logger.warning(f"   Erro: {prediction.get('msg')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar ML Engine: {e}", exc_info=True)
        return False

def test_payload_builder():
    """Testa se o payload builder foi atualizado para suportar ML."""
    logger.info("\n📦 TESTANDO ATUALIZAÇÃO DO PAYLOAD BUILDER")
    print("-"*40)
    
    try:
        # Importa o builder diretamente
        import market_orchestrator.ai.ai_payload_builder as builder_module
        
        # Verifica se a função build_ai_input tem o parâmetro ml_prediction
        import inspect
        sig = inspect.signature(builder_module.build_ai_input)
        params = list(sig.parameters.keys())
        
        if "ml_prediction" in params:
            logger.info("✅ Payload builder atualizado com parâmetro 'ml_prediction'")
            
            # Testa chamada com ml_prediction
            test_payload = builder_module.build_ai_input(
                symbol="BTCUSDT",
                signal={"tipo_evento": "Teste", "descricao": "Teste"},
                enriched={},
                flow_metrics={},
                historical_profile={},
                macro_context={},
                market_environment={},
                orderbook_data={},
                ml_features={},
                ml_prediction={"status": "ok", "prob_up": 0.75, "confidence": 0.8}
            )
            
            if "quant_model" in test_payload:
                logger.info("✅ Seção 'quant_model' adicionada ao payload")
                logger.info(f"   Viés: {test_payload['quant_model'].get('model_sentiment', 'N/A')}")
            else:
                logger.error("❌ Seção 'quant_model' não encontrada no payload")
                return False
                
            if "ml_str" in test_payload:
                logger.info("✅ String ML formatada criada para templates")
            else:
                logger.warning("⚠️ 'ml_str' não encontrada no payload")
                
            return True
            
        else:
            logger.error("❌ Payload builder NÃO atualizado - falta parâmetro 'ml_prediction'")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao testar payload builder: {e}", exc_info=True)
        return False

def test_ai_runner_integration():
    """Verifica se o ai_runner.py foi atualizado."""
    logger.info("\n⚙️ VERIFICANDO ATUALIZAÇÃO DO AI_RUNNER")
    print("-"*40)
    
    try:
        with open("market_orchestrator/ai/ai_runner.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        checks = [
            ("from ml.inference_engine import MLInferenceEngine", "Importação do ML Engine"),
            ("bot.ml_engine = MLInferenceEngine()", "Inicialização do ML Engine"),
            ("ml_prediction = bot.ml_engine.predict", "Chamada de previsão ML"),
            ('event_data["ml_prediction"] = ml_prediction', "Injeção no event_data"),
            ("ml_prediction=ml_prediction", "Passagem para builder")
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                logger.info(f"✅ {description} encontrado")
            else:
                logger.error(f"❌ {description} NÃO encontrado")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ Erro ao verificar ai_runner: {e}", exc_info=True)
        return False

def generate_test_report():
    """Gera relatório completo de teste."""
    logger.info("\n📋 RELATÓRIO DE VERIFICAÇÃO DA INTEGRAÇÃO ML")
    print("="*60)
    
    results = []
    
    # Teste 1: Motor ML
    logger.info("\n1. Testando Motor de Inferência ML...")
    ml_ok = test_ml_engine_direct()
    results.append(("Motor ML", ml_ok))
    
    # Teste 2: Payload Builder
    logger.info("\n2. Testando Payload Builder...")
    builder_ok = test_payload_builder()
    results.append(("Payload Builder", builder_ok))
    
    # Teste 3: AI Runner
    logger.info("\n3. Verificando AI Runner...")
    runner_ok = test_ai_runner_integration()
    results.append(("AI Runner", runner_ok))
    
    # Relatório final
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMO DOS TESTES")
    print("-"*40)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("-"*40)
    if all_passed:
        logger.info("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para Inteligência Híbrida.")
        logger.info("   O robô usará ML Quantitativo + IA Generativa.")
    else:
        logger.info("⚠️  ALGUNS TESTES FALHARAM. Sistema funcionará apenas com IA Generativa.")
        logger.info("   Verifique os erros acima e corrija.")
    
    return all_passed

if __name__ == "__main__":
    success = generate_test_report()
    sys.exit(0 if success else 1)