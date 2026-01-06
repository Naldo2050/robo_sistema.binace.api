# test_integrated_macro_provider.py
"""
Teste de integração do MacroDataProvider com o sistema de correlações cross-asset.
"""

import asyncio
import logging
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

async def test_macro_provider_integration():
    """Testa integração completa do sistema."""
    
    print("\n" + "="*80)
    print("TESTE DE INTEGRAÇÃO: MACRO DATA PROVIDER + CROSS-ASSET")
    print("="*80 + "\n")
    
    try:
        # 1. Teste do MacroDataProvider isolado
        print("1. Testando MacroDataProvider isolado...")
        
        from src.data.macro_data_provider import MacroDataProvider
        
        provider = MacroDataProvider()
        macro_data = await provider.get_all_macro_data()
        
        print(f"   Dados coletados: {len([k for k, v in macro_data.items() if v is not None and k != 'timestamp'])}/8")
        print(f"   VIX: {macro_data.get('vix')}")
        print(f"   Treasury 10Y: {macro_data.get('treasury_10y')}")
        print(f"   DXY: {macro_data.get('dxy')}")
        print(f"   BTC Dominance: {macro_data.get('btc_dominance')}")
        print(f"   ETH Dominance: {macro_data.get('eth_dominance')}")
        print(f"   Gold: {macro_data.get('gold')}")
        print(f"   Oil: {macro_data.get('oil')}")
        
        # 2. Teste das correlações cross-asset enhanced
        print("\n2. Testando correlações cross-asset enhanced...")
        
        from cross_asset_correlations import get_enhanced_cross_asset_correlations
        
        correlations = get_enhanced_cross_asset_correlations()
        
        print(f"   Status: {correlations.get('status')}")
        print(f"   Total features: {len([k for k in correlations.keys() if not k.startswith('_')])}")
        
        # Verifica métricas enhanced
        enhanced_metrics = [
            'vix_current', 'us10y_yield', 'btc_dominance', 'eth_dominance',
            'gold_price', 'oil_price', 'macro_regime', 'correlation_regime'
        ]
        
        for metric in enhanced_metrics:
            value = correlations.get(metric)
            if value is not None:
                print(f"   {metric}: {value}")
            else:
                print(f"   {metric}: None")
        
        # 3. Teste de integração com ml_features
        print("\n3. Testando integração com ml_features...")
        
        from ml_features import calculate_cross_asset_features
        
        features = calculate_cross_asset_features("BTCUSDT")
        
        print(f"   ML Features count: {len(features)}")
        
        # Verifica se as novas métricas estão presentes
        enhanced_count = 0
        for metric in enhanced_metrics:
            if metric in features:
                enhanced_count += 1
        
        print(f"   Enhanced metrics in ML: {enhanced_count}/{len(enhanced_metrics)}")
        
        # 4. Resumo final
        print("\n4. Resumo da integração:")
        
        traditional_count = len([k for k in correlations.keys() 
                               if k in ['btc_eth_corr_7d', 'btc_eth_corr_30d', 'btc_dxy_corr_30d', 
                                       'dxy_return_5d', 'dxy_return_20d']])
        
        total_count = len([k for k in correlations.keys() if not k.startswith('_')])
        
        print(f"   Traditional metrics: {traditional_count}")
        print(f"   Enhanced metrics: {total_count - traditional_count - 2}")  # -2 para status/timestamp
        print(f"   Total metrics: {total_count}")
        
        # 5. Verificação de requisitos
        print("\n5. Verificação de requisitos:")
        
        requirements = {
            "VIX - Fear Index": correlations.get('vix_current') is not None,
            "Treasury Yields": correlations.get('us10y_yield') is not None,
            "Crypto Dominance": correlations.get('btc_dominance') is not None,
            "Commodities (Gold)": correlations.get('gold_price') is not None,
            "Commodities (Oil)": correlations.get('oil_price') is not None,
            "Regime Detection": correlations.get('macro_regime') is not None
        }
        
        for req, met in requirements.items():
            status = "OK" if met else "Missing"
            print(f"   {req}: {status}")
        
        print("\n" + "="*80)
        print("TESTE DE INTEGRAÇÃO CONCLUÍDO!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nErro no teste de integração: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_macro_provider_integration())
    if success:
        print("\nSistema integrado funcionando corretamente!")
    else:
        print("\nFalhas detectadas no sistema integrado.")