"""
Script de validação rápida do sistema de regime.
Executa todos os componentes e verifica se estão funcionando.
"""
import asyncio
import sys
sys.path.insert(0, 'src')

from data.macro_data_provider import MacroDataProvider
from analysis.regime_detector import EnhancedRegimeDetector
from rules.regime_rules import RegimeBasedRules

# Configurar codificação para UTF-8 para evitar problemas com caracteres especiais
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def validate_system():
    """Valida todos os componentes do sistema de regime"""
    
    print("="*60)
    print("    VALIDAÇÃO DO SISTEMA DE REGIME")
    print("="*60)
    
    results = {
        "macro_provider": False,
        "regime_detector": False,
        "regime_rules": False,
        "integration": False,
    }
    
    # 1. Testar MacroDataProvider
    print("\n[1/4] Testando MacroDataProvider...")
    try:
        provider = MacroDataProvider()
        macro_data = await provider.get_all_macro_data()
        
        if macro_data and "btc_dominance" in macro_data:
            print(f"   ✅ MacroDataProvider OK")
            print(f"      BTC Dominance: {macro_data.get('btc_dominance', 'N/A')}")
            print(f"      VIX: {macro_data.get('vix', 'N/A')}")
            results["macro_provider"] = True
        else:
            print(f"   ❌ MacroDataProvider - dados incompletos")
    except Exception as e:
        print(f"   ❌ MacroDataProvider ERRO: {e}")
    
    # 2. Testar EnhancedRegimeDetector
    print("\n[2/4] Testando EnhancedRegimeDetector...")
    try:
        detector = EnhancedRegimeDetector()
        
        # Mock data para teste
        mock_macro = {
            "vix": 18.0,
            "btc_dominance": 48.0,
            "usdt_dominance": 5.0,
        }
        mock_cross_asset = {
            "correlation_spy": 0.5,
            "btc_dxy_corr_30d": -0.2,
        }
        mock_price = {"current_price": 92000}
        
        regime = detector.detect_regime(mock_macro, mock_cross_asset, mock_price)
        
        if regime and hasattr(regime, 'market_regime'):
            print(f"   ✅ EnhancedRegimeDetector OK")
            print(f"      Market Regime: {regime.market_regime}")
            print(f"      Volatility: {regime.volatility_regime}")
            print(f"      Risk Score: {regime.risk_score:.2f}")
            results["regime_detector"] = True
        else:
            print(f"   ❌ EnhancedRegimeDetector - regime não detectado")
    except Exception as e:
        print(f"   ❌ EnhancedRegimeDetector ERRO: {e}")
    
    # 3. Testar RegimeBasedRules
    print("\n[3/4] Testando RegimeBasedRules...")
    try:
        rules = RegimeBasedRules()
        
        mock_regime = {
            "market_regime": "RISK_ON",
            "volatility_regime": "LOW_VOL",
            "correlation_regime": "MACRO_CORRELATED",
            "risk_score": 0.5,
            "regime_change_warning": False,
        }
        
        adjustment = rules.get_regime_adjustment(mock_regime)
        should_trade, reason = rules.should_trade(mock_regime, "long", 0.7)
        
        print(f"   ✅ RegimeBasedRules OK")
        print(f"      Position Multiplier: {adjustment.position_size_multiplier:.2f}x")
        print(f"      Should Trade: {should_trade}")
        print(f"      Allowed Directions: {adjustment.allowed_directions}")
        results["regime_rules"] = True
    except Exception as e:
        print(f"   ❌ RegimeBasedRules ERRO: {e}")
    
    # 4. Testar Integração Completa
    print("\n[4/4] Testando Integração Completa...")
    try:
        # Simular fluxo completo
        provider = MacroDataProvider()
        detector = EnhancedRegimeDetector()
        rules = RegimeBasedRules()
        
        # Dados
        macro_data = await provider.get_all_macro_data()
        
        # Detectar regime
        regime_result = detector.detect_regime(
            macro_data,
            {"correlation_spy": 0.4},
            {"current_price": 92000}
        )
        
        # Converter para dict
        regime_dict = {
            "market_regime": regime_result.market_regime.value,
            "volatility_regime": regime_result.volatility_regime.value,
            "correlation_regime": regime_result.correlation_regime.value,
            "risk_score": regime_result.risk_score,
            "regime_change_warning": regime_result.regime_change_warning,
        }
        
        # Aplicar regras
        adjustment = rules.get_regime_adjustment(regime_dict)
        should_trade, reason = rules.should_trade(regime_dict, "long", 0.65)
        
        print(f"   ✅ Integração OK")
        print(f"      Flow: MacroData → RegimeDetector → RegimeRules")
        print(f"      Final Decision: {'TRADE' if should_trade else 'BLOCKED'}")
        results["integration"] = True
    except Exception as e:
        print(f"   ❌ Integração ERRO: {e}")
    
    # Resumo Final
    print("\n" + "="*60)
    print("    RESUMO DA VALIDAÇÃO")
    print("="*60)
    
    all_passed = all(results.values())
    
    for component, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {component}")
    
    print("\n" + "="*60)
    if all_passed:
        print("   🎉 TODOS OS COMPONENTES VALIDADOS COM SUCESSO!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"   ⚠️  COMPONENTES COM FALHA: {failed}")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(validate_system())
    sys.exit(0 if success else 1)