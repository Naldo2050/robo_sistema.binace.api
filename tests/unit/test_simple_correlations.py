# test_simple_correlations.py
# -*- coding: utf-8 -*-
"""
Teste simples das correlações cross-asset
"""

import asyncio
import sys
from datetime import datetime

# Teste direto do módulo
try:
    from market_analysis.cross_asset_correlations import get_all_cross_asset_correlations
    
    async def test_simple():
        print("🧪 Teste simples das correlações...")
        now_utc = datetime.utcnow()
        result = await get_all_cross_asset_correlations(now_utc)
        
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'ok':
            print("✅ Módulo funcionando!")
            
            btc_eth = result.get('btc_eth', {})
            print(f"BTC x ETH 7d: {btc_eth.get('btc_eth_corr_7d', 'N/A')}")
            print(f"BTC x ETH 30d: {btc_eth.get('btc_eth_corr_30d', 'N/A')}")
            
            btc_dxy = result.get('btc_dxy', {})
            print(f"BTC x DXY 7d: {btc_dxy.get('btc_dxy_corr_7d', 'N/A')}")
            print(f"BTC x DXY 30d: {btc_dxy.get('btc_dxy_corr_30d', 'N/A')}")
        else:
            print(f"Erro: {result.get('error')}")
    
    asyncio.run(test_simple())
    
except Exception as e:
    print(f"❌ Erro: {e}")
    sys.exit(1)