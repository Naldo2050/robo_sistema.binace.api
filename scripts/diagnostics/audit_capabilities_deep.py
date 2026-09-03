# -*- coding: utf-8 -*-
"""
Deep scan of all 57 institutional capabilities across codebase.
"""
import os
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    files_content = {}
    for root, _, files in os.walk('.'):
        if any(p in root for p in ['.git', '.venv', '__pycache__', 'archive', 'backups', 'src_old', 'legacy', 'logs', 'dados']):
            continue
        for f in files:
            if f.endswith('.py'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as fp:
                        files_content[fpath] = fp.read().lower()
                except Exception:
                    pass

    print(f"Cached {len(files_content)} Python files.")

    items = [
        # Order flow
        ("1. Footprint", ["footprint", "cluster", "footprintanalyzer"]),
        ("2. DOM/Orderbook", ["orderbook_analyzer", "order_book_depth", "orderbookcore"]),
        ("3. CVD", ["cvd_divergence", "flow_analyzer_cvd", "cvdanalyzer"]),
        ("4. Time & Sales", ["trade_buffer", "valid_window_data"]),
        ("5. Order Flow Imbalance", ["orderflowimbalanceanalyzer", "flow_imbalance", "ofi"]),
        ("6. Iceberg", ["icebergdetector", "iceberg_activity"]),
        ("7. Absorption", ["absorptiondetector", "absorptionzonemapper", "absorcao", "absorção"]),
        # Volume/Auction
        ("8. Volume Profile", ["volumeprofile", "dynamicvolumeprofile", "vah", "val", "poc"]),
        ("9. Market Profile/TPO", ["market_profile", "tpo"]),
        ("10. Auction Market Theory", ["auction_market_theory", "auction_state"]),
        ("11. VWAP", ["vwap", "vwaptwapanalyzer"]),
        ("12. TWAP", ["twap", "twap_validator"]),
        # Quant
        ("13. StatArb/Cointegration", ["cointegration", "statarb", "statistical_arbitrage"]),
        ("14. Mean Reversion", ["meanreversionanalyzer", "bollinger_bands"]),
        ("15. Momentum/Trend", ["momentum", "adx", "macd"]),
        ("16. Machine Learning", ["mlinferenceengine", "xgboost"]),
        ("17. Monte Carlo", ["montecarlosimulator", "monte_carlo"]),
        ("18. HMM / Market Regime", ["marketregimehmm", "regime_probabilities", "regimedetector"]),
        ("19. GARCH", ["garchmodel", "garch_forecast_1h", "garch_volatility"]),
        ("20. Kyle Lambda / Market Impact", ["kyle", "market_impact", "slippage_1k"]),
        ("21. Hurst", ["hurstcalculator", "hurst_exponent"]),
        ("22. Fractal Analysis", ["fractal_dimension", "dfa"]),
        ("23. Shannon Entropy", ["entropyanalyzer", "shannon_entropy"]),
        ("24. Fourier / Cycles", ["fouriercycleanalyzer", "dominant_cycles"]),
        ("25. Kalman", ["kalmantrendfilter", "kalman_filter"]),
        ("26. Dynamic Regression", ["regression_channel", "slope_per_bar"]),
        # Liquidity / Derivatives
        ("27. Liquidity Heatmap", ["liquidityheatmap", "liquidity_heatmap"]),
        ("28. Liquidation Map", ["liquidation_map", "estimated_liquidations"]),
        ("29. Open Interest", ["open_interest", "oi_contracts", "oi_change"]),
        ("30. Funding Rate", ["funding_rate", "funding_aggregator"]),
        ("31. Gamma Exposure", ["gamma_exposure", "gex"]),
        # On-chain
        ("32. Whale trades", ["whaledetector", "large_orders_1h", "whale_score"]),
        ("33. Whale wallet tracking", ["whale_wallet", "wallet_tracker"]),
        ("34. Exchange Inflow/Outflow", ["exchange_inflow", "exchange_outflow"]),
        ("35. MVRV", ["mvrv"]),
        ("36. SOPR", ["sopr"]),
        ("37. NVT", ["nvt"]),
        ("38. Realized Price", ["realized_price"]),
        ("39. Miner metrics", ["hashrate", "puell_multiple"]),
        # Sentiment
        ("40. Fear & Greed", ["fear_and_greed", "fear&greed", "fear_greed"]),
        ("41. NLP Sentiment", ["nlp_sentiment", "news_sentiment"]),
        ("42. Global L/S Ratio", ["global_long_short", "global_ratio"]),
        ("43. Top Trader Account Ratio", ["top_trader_long_short_account", "top_trader_account"]),
        ("44. Top Trader Position Ratio", ["top_trader_long_short_position", "top_trader_position"]),
        ("45. COT / Crypto COT", ["cryptocot", "crypto_cot"]),
        # Market Structure
        ("46. Wyckoff", ["wyckoff"]),
        ("47. SMC / ICT", ["smartmoneyanalyzer", "smc"]),
        ("48. Order Blocks", ["order_block", "orderblock"]),
        ("49. FVG", ["fair_value_gap", "fvg"]),
        ("50. Liquidity Sweep", ["liquidity_sweep"]),
        ("51. Supply / Demand", ["defense_zones", "defensezonedetector", "supply_demand"]),
        ("52. BOS", ["break_of_structure", "bos"]),
        ("53. MSS", ["market_structure_shift", "mss", "choch"]),
        # Options
        ("54. Options Flow", ["options_flow"]),
        ("55. Black-Scholes", ["black_scholes", "bs_model"]),
        ("56. Greeks", ["greeks", "delta_gamma_vega"]),
        ("57. GEX", ["gex", "gamma_exposure"])
    ]

    for label, keys in items:
        matched_files = []
        for fpath, content in files_content.items():
            if any(k in content for k in keys):
                matched_files.append(fpath)
        print(f"{label:35s} -> {len(matched_files):2d} files: {[os.path.basename(m) for m in matched_files[:6]]}")

if __name__ == '__main__':
    main()
