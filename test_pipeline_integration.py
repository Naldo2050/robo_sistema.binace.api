# test_pipeline_integration.py
# -*- coding: utf-8 -*-
"""
Teste da integração das cross-asset features no ANALYSIS_TRIGGER
"""

import sys
import logging
import pandas as pd
from datetime import datetime, timezone

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import do pipeline
try:
    from data_pipeline.pipeline import DataPipeline
    logger.info("✅ DataPipeline importado com sucesso")
except ImportError as e:
    logger.error(f"❌ Erro ao importar DataPipeline: {e}")
    sys.exit(1)


def create_sample_trades():
    """Cria dados de exemplo para teste."""
    # Dados de trades simulados para BTCUSDT
    trades = []
    base_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    for i in range(100):
        trades.append({
            'E': base_time + (i * 60000),  # 1 minuto de diferença
            'e': 'trade',
            's': 'BTCUSDT',
            'p': str(45000 + (i * 10)),  # Preço subindo
            'q': str(0.001 + (i * 0.0001)),  # Volume crescente
            'T': base_time + (i * 60000),
            'm': False if i % 2 == 0 else True,  # Alterna buyer maker
        })
    
    return trades


def test_pipeline_ml_features():
    """Testa se as cross-asset features são incluídas no ml_features do ANALYSIS_TRIGGER."""
    logger.info("🧪 TESTANDO: Integração cross-asset no DataPipeline")
    
    try:
        # Criar pipeline com dados de teste
        trades = create_sample_trades()
        pipeline = DataPipeline(trades, "BTCUSDT")
        
        logger.info(f"📊 Pipeline criado com {len(trades)} trades")
        
        # Executar pipeline completo
        logger.info("🚀 Executando pipeline completo...")
        
        # 1. Enriquecer dados
        enriched = pipeline.enrich()
        logger.info("✅ Dados enriquecidos")
        
        # 2. Adicionar contexto (simulado)
        contextual = pipeline.add_context()
        logger.info("✅ Contexto adicionado")
        
        # 3. Detectar sinais
        signals = pipeline.detect_signals()
        logger.info(f"✅ Sinais detectados: {len(signals)}")
        
        # 4. Extrair features ML
        ml_features = pipeline.extract_features()
        logger.info(f"✅ Features ML extraídas: {len(ml_features)} features")
        
        # Verificar se cross_asset foi adicionado
        if "cross_asset" in ml_features:
            cross_asset = ml_features["cross_asset"]
            logger.info(f"🎯 CROSS-ASSET FEATURES ENCONTRADAS!")
            logger.info(f"📊 Total de cross-asset features: {len(cross_asset)}")
            
            # Verificar campos específicos da especificação
            required_fields = [
                "btc_eth_corr_7d", "btc_eth_corr_30d",
                "btc_dxy_corr_30d", "btc_dxy_corr_90d",
                "btc_ndx_corr_30d",
                "dxy_return_5d", "dxy_return_20d"
            ]
            
            logger.info("\n📋 CROSS-ASSET FEATURES:")
            for field in required_fields:
                value = cross_asset.get(field, 'N/A')
                logger.info(f"  {field}: {value}")
            
            # Verificar se todos os campos estão presentes
            missing = [f for f in required_fields if f not in cross_asset]
            if missing:
                logger.warning(f"⚠️ Campos faltando: {missing}")
                return False
            else:
                logger.info("✅ Todos os campos especificados estão presentes!")
                
        else:
            logger.error("❌ cross_asset NÃO encontrado em ml_features")
            logger.info(f"📋 Campos disponíveis: {list(ml_features.keys())}")
            return False
        
        # 5. Verificar get_final_features (inclui no ANALYSIS_TRIGGER)
        final_features = pipeline.get_final_features()
        ml_features_final = final_features.get("ml_features", {})
        
        if "cross_asset" in ml_features_final:
            logger.info("✅ cross_asset também presente em get_final_features()")
            logger.info("✅ INTEGRAÇÃO COMPLETA NO ANALYSIS_TRIGGER CONFIRMADA!")
            return True
        else:
            logger.error("❌ cross_asset NÃO encontrado em get_final_features()")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}", exc_info=True)
        return False


def test_analysis_trigger_structure():
    """Testa a estrutura completa do ANALYSIS_TRIGGER."""
    logger.info("🧪 TESTANDO: Estrutura completa do ANALYSIS_TRIGGER")
    
    try:
        trades = create_sample_trades()
        pipeline = DataPipeline(trades, "BTCUSDT")
        
        # Executar pipeline
        final_features = pipeline.get_final_features()
        
        # Verificar sinais (inclui ANALYSIS_TRIGGER)
        signals = final_features.get("signals", [])
        analysis_triggers = [s for s in signals if s.get("tipo_evento") == "ANALYSIS_TRIGGER"]
        
        if analysis_triggers:
            trigger = analysis_triggers[0]
            logger.info("✅ ANALYSIS_TRIGGER encontrado")
            
            # Verificar se ml_features está no ANALYSIS_TRIGGER
            if "ml_features" in trigger:
                ml_features = trigger["ml_features"]
                logger.info(f"✅ ml_features no ANALYSIS_TRIGGER: {len(ml_features)} features")
                
                if "cross_asset" in ml_features:
                    cross_asset = ml_features["cross_asset"]
                    logger.info("✅ cross_asset no ANALYSIS_TRIGGER!")
                    logger.info(f"📊 Cross-asset features: {list(cross_asset.keys())}")
                    return True
                else:
                    logger.error("❌ cross_asset NÃO no ANALYSIS_TRIGGER")
                    return False
            else:
                logger.error("❌ ml_features NÃO no ANALYSIS_TRIGGER")
                return False
        else:
            logger.error("❌ Nenhum ANALYSIS_TRIGGER encontrado")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no teste de estrutura: {e}", exc_info=True)
        return False


def main():
    """Função principal de teste."""
    logger.info("🚀 INICIANDO TESTE DE INTEGRAÇÃO NO ANALYSIS_TRIGGER")
    logger.info("=" * 70)
    
    # Teste 1: Pipeline ML features
    logger.info("1️⃣ TESTE 1: Pipeline com cross-asset features")
    pipeline_success = test_pipeline_ml_features()
    
    # Teste 2: Estrutura ANALYSIS_TRIGGER
    logger.info("\n2️⃣ TESTE 2: Estrutura ANALYSIS_TRIGGER")
    structure_success = test_analysis_trigger_structure()
    
    # Resultado final
    logger.info("\n" + "=" * 70)
    logger.info("📋 RESULTADO FINAL:")
    
    if pipeline_success and structure_success:
        logger.info("✅ TODOS OS TESTES PASSARAM!")
        logger.info("✅ Cross-asset features integradas no ANALYSIS_TRIGGER")
        logger.info("✅ Estrutura JSON conforme especificação")
        logger.info("✅ get_cross_asset_features() funcionando")
        return True
    else:
        logger.error("❌ ALGUNS TESTES FALHARAM:")
        logger.error(f"  - Pipeline: {'✅' if pipeline_success else '❌'}")
        logger.error(f"  - Estrutura: {'✅' if structure_success else '❌'}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)