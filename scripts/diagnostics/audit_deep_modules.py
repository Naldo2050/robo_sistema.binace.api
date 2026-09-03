# -*- coding: utf-8 -*-
"""
Auditoria detalhada de módulos específicos do sistema.
"""
import sys
import os
import inspect

def audit_all():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 80)
    print("AUDITORIA DETALHADA DAS CAPACIDADES ESPECIFICAS")
    print("=" * 80)

    # 1. Kyle Lambda / Market Impact
    print("\n--- 20. Kyle Lambda / Market Impact ---")
    if os.path.exists('market_analysis/market_impact.py'):
        with open('market_analysis/market_impact.py', 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("market_analysis/market_impact.py NAO ENCONTRADO")

    # 2. StatArb / Cointegration
    print("\n--- 13. Statistical Arbitrage / Cointegration ---")
    if os.path.exists('market_analysis/cross_asset_correlations.py'):
        with open('market_analysis/cross_asset_correlations.py', 'r', encoding='utf-8') as f:
            content = f.read()
            c_lines = [l for l in content.splitlines() if any(k in l.lower() for k in ['coint', 'statarb', 'arbitrag', 'spread', 'adf', 'stationar'])]
            print(f"cross_asset_correlations.py matches ({len(c_lines)}):", c_lines[:10])

    # 3. On-chain Fetcher (MVRV, SOPR, NVT, Realized Price, Miners)
    print("\n--- 32-39. On-Chain Metrics (MVRV, SOPR, NVT, Whale Wallets, Miners) ---")
    if os.path.exists('fetchers/onchain_fetcher.py'):
        with open('fetchers/onchain_fetcher.py', 'r', encoding='utf-8') as f:
            print(f.read()[:1500])

    # 4. Derivatives & Positioning (Funding, Open Interest, Long/Short ratios, Liquidation)
    print("\n--- 28-30 & 42-45. Derivatives & Positioning (OI, Funding, L/S Ratios, Liquidation) ---")
    if os.path.exists('fetchers/funding_aggregator.py'):
        with open('fetchers/funding_aggregator.py', 'r', encoding='utf-8') as f:
            print("funding_aggregator.py:\n", f.read()[:600])

    if os.path.exists('institutional/crypto_cot.py'):
        with open('institutional/crypto_cot.py', 'r', encoding='utf-8') as f:
            print("institutional/crypto_cot.py:\n", f.read()[:600])

    # 5. Market Structure & SMC (BOS, MSS, FVG, Order Blocks, Liquidity Sweep, Wyckoff)
    print("\n--- 46-53. Market Structure & SMC ---")
    if os.path.exists('institutional/smart_money.py'):
        with open('institutional/smart_money.py', 'r', encoding='utf-8') as f:
            print("institutional/smart_money.py:\n", f.read()[:1000])

    # 6. Options & GEX
    print("\n--- 31 & 54-57. Options, Greeks, GEX, Black-Scholes ---")
    opt_found = []
    for root, _, files in os.walk('.'):
        if any(p in root for p in ['.git', '.venv', '__pycache__', 'archive', 'backups', 'src_old', 'legacy', 'logs', 'dados']):
            continue
        for f in files:
            if f.endswith('.py') and any(k in f.lower() for k in ['option', 'greek', 'black_scholes', 'gex']):
                opt_found.append(os.path.join(root, f))
    print(f"Option files in codebase: {opt_found}")

if __name__ == '__main__':
    audit_all()
