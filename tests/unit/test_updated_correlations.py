# test_updated_correlations.py
# -*- coding: utf-8 -*-
"""
Teste das correlações cross-asset atualizadas conforme especificação
"""

import sys
from datetime import datetime

# Teste direto do módulo
try:
    from market_analysis.cross_asset_correlations import get_cross_asset_features
    
    print("🧪 Testando função get_cross_asset_features...")
    
    now_utc = datetime.utcnow()
    result = get_cross_asset_features(now_utc)
    
    print(f"Status: {result.get('status')}")
    
    if result.get('status') == 'ok':
        print("✅ Função funcionando!")
        
        # Verifica campos específicos da especificação
        print("\n📊 RESULTADOS (conforme especificação):")
        print(f"btc_eth_corr_7d: {result.get('btc_eth_corr_7d', 'N/A')}")
        print(f"btc_eth_corr_30d: {result.get('btc_eth_corr_30d', 'N/A')}")
        print(f"btc_dxy_corr_30d: {result.get('btc_dxy_corr_30d', 'N/A')}")
        print(f"btc_dxy_corr_90d: {result.get('btc_dxy_corr_90d', 'N/A')}")
        print(f"btc_ndx_corr_30d: {result.get('btc_ndx_corr_30d', 'N/A')}")
        print(f"dxy_return_5d: {result.get('dxy_return_5d', 'N/A')}")
        print(f"dxy_return_20d: {result.get('dxy_return_20d', 'N/A')}")
        
        # Verifica se todos os campos requeridos estão presentes
        required_fields = [
            'btc_eth_corr_7d', 'btc_eth_corr_30d',
            'btc_dxy_corr_30d', 'btc_dxy_corr_90d',
            'btc_ndx_corr_30d',
            'dxy_return_5d', 'dxy_return_20d'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in result:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"\n⚠️ Campos faltando: {missing_fields}")
        else:
            print(f"\n✅ Todos os {len(required_fields)} campos especificados estão presentes!")
        
        print(f"\n📋 Total de campos: {len(result)}")
        
    else:
        print(f"❌ Erro: {result.get('error')}")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Erro de import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)