# test_cross_asset_integration.py
# -*- coding: utf-8 -*-
"""
Teste de Integração das Features Cross-Asset

Este script testa a integração completa:
1. ml_features.py com calculate_cross_asset_features
2. ai_payload_builder.py com cross_asset_context
3. Eventos ANALYSIS_TRIGGER e AI_ANALYSIS
"""

import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports dos módulos modificados
try:
    from ml_features import generate_ml_features
    from market_orchestrator.ai.ai_payload_builder import build_ai_input
    logger.info("✅ Imports dos módulos bem-sucedidos")
except ImportError as e:
    logger.error(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)


def create_test_data():
    """Cria dados de teste para simular um evento ANALYSIS_TRIGGER."""
    
    # Dados de preços simulados (BTCUSDT)
    test_df = pd.DataFrame({
        'p': [45000, 45100, 45050, 45200, 45150, 45300, 45250],
        'q': [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5],
        'm': [False, True, True, False, True, False, True],
    })
    
    # Dados de orderbook simulados
    orderbook_data = {
        "order_book_depth": {
            "L1": {"bids": 100000, "asks": 95000},
            "L10": {"bids": 500000, "asks": 480000},
        },
        "spread_metrics": {"spread": 5.0},
        "bid_depth_usd": 100000,
        "ask_depth_usd": 95000,
        "imbalance": 0.05,
        "spread_percent": 0.011
    }
    
    # Dados de flow metrics simulados
    flow_metrics = {
        "order_flow": {
            "net_flow_1m": -1500.0,
            "buy_sell_ratio": 0.8,
            "flow_imbalance": -0.11,
            "tick_rule_sum": -2.0,
        },
        "bursts": {"count": 5},
        "metadata": {"burst_window_ms": 1000},
        "cvd": -5000,
        "whale_delta": 1000,
        "whale_buy_volume": 500,
        "whale_sell_volume": 1500,
        "tipo_absorcao": "Venda"
    }
    
    # Dados de perfil histórico simulados
    historical_profile = {
        "daily": {
            "poc": 45100,
            "vah": 45300,
            "val": 44900
        }
    }
    
    # Contexto macro simulado
    macro_context = {
        "trading_session": "NY",
        "session_phase": "ACTIVE",
        "mtf_trends": {
            "1h": {
                "tendencia": "Alta",
                "rsi_short": 65.5,
                "macd": 0.002,
                "macd_signal": 0.001,
                "adx": 25.3
            }
        },
        "atr": 500.0
    }
    
    # Ambiente de mercado simulado
    market_environment = {
        "volatility_regime": "NORMAL",
        "trend_direction": "UP",
        "market_structure": "ACCUMULATION",
        "risk_sentiment": "BULLISH",
        "correlation_spy": 0.3,
        "correlation_dxy": -0.6
    }
    
    # Evento ANALYSIS_TRIGGER simulado
    signal = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "preco_fechamento": 45250.0,
        "volume_total": 15.5,
        "delta": -250.0,
        "descricao": "Evento de teste para cross-asset features",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "resultado_da_batalha": "VENDEDOR_VENCEDOR",
        "janela_numero": 1001
    }
    
    return {
        "df": test_df,
        "orderbook_data": orderbook_data,
        "flow_metrics": flow_metrics,
        "historical_profile": historical_profile,
        "macro_context": macro_context,
        "market_environment": market_environment,
        "signal": signal
    }


def test_ml_features_cross_asset():
    """Testa a geração de features ML incluindo cross-asset."""
    logger.info("🧪 TESTANDO: ml_features.py com cross-asset features")
    
    test_data = create_test_data()
    
    try:
        # Gera features ML
        ml_features = generate_ml_features(
            df=test_data["df"],
            orderbook_data=test_data["orderbook_data"],
            flow_metrics=test_data["flow_metrics"],
            symbol="BTCUSDT"  # Especifica símbolo para cross-asset
        )
        
        # Verifica se cross_asset foi adicionado
        if "cross_asset" in ml_features:
            cross_asset = ml_features["cross_asset"]
            logger.info(f"✅ cross_asset features encontradas: {len(cross_asset)} features")
            
            # Lista as features encontradas
            for key, value in cross_asset.items():
                logger.info(f"  📊 {key}: {value}")
            
            # Verifica features específicas esperadas
            expected_features = [
                "btc_eth_corr_7d", "btc_eth_corr_30d",
                "btc_dxy_corr_7d", "btc_dxy_corr_30d",
                "btc_ndx_corr_7d", "btc_ndx_corr_30d"
            ]
            
            found_features = []
            missing_features = []
            
            for feature in expected_features:
                if feature in cross_asset:
                    found_features.append(feature)
                else:
                    missing_features.append(feature)
            
            logger.info(f"✅ Features encontradas: {found_features}")
            if missing_features:
                logger.warning(f"⚠️ Features faltando: {missing_features}")
            
            return True, cross_asset
        else:
            logger.error("❌ cross_asset não encontrado nas features")
            return False, {}
            
    except Exception as e:
        logger.error(f"❌ Erro ao testar ml_features: {e}", exc_info=True)
        return False, {}


def test_ai_payload_cross_asset():
    """Testa a construção do ai_payload com cross_asset_context."""
    logger.info("🧪 TESTANDO: ai_payload_builder.py com cross_asset_context")
    
    test_data = create_test_data()
    
    try:
        # Primeiro gera as features
        ml_features, _ = test_ml_features_cross_asset()
        if not ml_features:
            logger.error("❌ Não foi possível gerar ml_features para testar ai_payload")
            return False
        
        # Constrói ai_payload
        ai_payload = build_ai_input(
            symbol="BTCUSDT",
            signal=test_data["signal"],
            enriched={"ohlc": {"close": 45250}},
            flow_metrics=test_data["flow_metrics"],
            historical_profile=test_data["historical_profile"],
            macro_context=test_data["macro_context"],
            market_environment=test_data["market_environment"],
            orderbook_data=test_data["orderbook_data"],
            ml_features=ml_features
        )
        
        # Verifica se cross_asset_context foi adicionado
        if "cross_asset_context" in ai_payload:
            cross_asset_ctx = ai_payload["cross_asset_context"]
            logger.info(f"✅ cross_asset_context encontrado no ai_payload")
            
            # Lista os contextos encontrados
            for key, value in cross_asset_ctx.items():
                logger.info(f"  🔗 {key}: {value}")
            
            # Verifica contextos específicos
            expected_contexts = [
                "btc_eth_correlations",
                "btc_dxy_correlations", 
                "btc_ndx_correlations"
            ]
            
            found_contexts = []
            missing_contexts = []
            
            for context in expected_contexts:
                if context in cross_asset_ctx:
                    found_contexts.append(context)
                else:
                    missing_contexts.append(context)
            
            logger.info(f"✅ Contextos encontrados: {found_contexts}")
            if missing_contexts:
                logger.warning(f"⚠️ Contextos faltando: {missing_contexts}")
            
            return True, cross_asset_ctx
        else:
            logger.error("❌ cross_asset_context não encontrado no ai_payload")
            return False, {}
            
    except Exception as e:
        logger.error(f"❌ Erro ao testar ai_payload: {e}", exc_info=True)
        return False, {}


def test_event_workflow():
    """Testa o fluxo completo de evento ANALYSIS_TRIGGER -> AI_ANALYSIS."""
    logger.info("🧪 TESTANDO: Fluxo completo ANALYSIS_TRIGGER -> AI_ANALYSIS")
    
    try:
        # Simula evento ANALYSIS_TRIGGER
        logger.info("📋 Simulando evento ANALYSIS_TRIGGER...")
        
        # Gera features ML (como faria o sistema real)
        ml_success, cross_asset_features = test_ml_features_cross_asset()
        if not ml_success:
            logger.error("❌ Falha na geração de features ML")
            return False
        
        # Simula ai_payload para AI_ANALYSIS
        logger.info("📋 Simulando evento AI_ANALYSIS...")
        
        ai_success, cross_asset_context = test_ai_payload_cross_asset()
        if not ai_success:
            logger.error("❌ Falha na construção de ai_payload")
            return False
        
        logger.info("✅ Fluxo completo executado com sucesso!")
        
        # Resume resultados
        logger.info("📊 RESUMO DOS RESULTADOS:")
        logger.info(f"  - Features ML cross-asset: {len(cross_asset_features)}")
        logger.info(f"  - Contextos AI cross-asset: {len(cross_asset_context)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no fluxo completo: {e}", exc_info=True)
        return False


def main():
    """Função principal de teste."""
    logger.info("🚀 INICIANDO TESTE DE INTEGRAÇÃO CROSS-ASSET")
    logger.info("=" * 60)
    
    # Teste 1: ml_features
    logger.info("1️⃣ TESTE 1: ml_features.py")
    ml_success, cross_asset_features = test_ml_features_cross_asset()
    
    # Teste 2: ai_payload_builder
    logger.info("\n2️⃣ TESTE 2: ai_payload_builder.py")
    ai_success, cross_asset_context = test_ai_payload_cross_asset()
    
    # Teste 3: Fluxo completo
    logger.info("\n3️⃣ TESTE 3: Fluxo completo")
    workflow_success = test_event_workflow()
    
    # Resultado final
    logger.info("\n" + "=" * 60)
    logger.info("📋 RESULTADO FINAL:")
    
    if ml_success and ai_success and workflow_success:
        logger.info("✅ TODOS OS TESTES PASSARAM!")
        logger.info("✅ Integração cross-asset implementada com sucesso")
        logger.info("✅ Features disponíveis em eventos ANALYSIS_TRIGGER")
        logger.info("✅ Contextos disponíveis em eventos AI_ANALYSIS")
        return True
    else:
        logger.error("❌ ALGUNS TESTES FALHARAM:")
        logger.error(f"  - ml_features: {'✅' if ml_success else '❌'}")
        logger.error(f"  - ai_payload: {'✅' if ai_success else '❌'}")
        logger.error(f"  - workflow: {'✅' if workflow_success else '❌'}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)