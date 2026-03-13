"""
Teste de integração do EnhancedRegimeDetector no ai_payload_builder.
"""
import asyncio
import sys
from pathlib import Path

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(str(Path(__file__).parent))

from market_orchestrator.ai.ai_payload_builder import build_ai_input


# Dados de exemplo para o teste
def create_sample_data():
    """Cria dados de exemplo para testar a integração."""
    symbol = "BTCUSDT"
    signal = {
        "tipo_evento": "AI_ANALYSIS",
        "descricao": "Teste de integração",
        "delta": 100.0,
        "volume_total": 5000.0,
        "preco_fechamento": 50000.0,
        "timestamp": "2026-01-05T00:00:00Z",
        "janela_numero": 1,
        "timestamp_utc": "2026-01-05T00:00:00Z"
    }
    enriched = {
        "ohlc": {
            "open": 49500.0,
            "high": 50500.0,
            "low": 49000.0,
            "close": 50000.0,
            "vwap": 49800.0
        }
    }
    flow_metrics = {
        "order_flow": {
            "net_flow_1m": 50.0,
            "flow_imbalance": 0.1,
            "aggressive_buy_pct": 0.6,
            "aggressive_sell_pct": 0.4
        },
        "whale_delta": 20.0,
        "whale_buy_volume": 1000.0,
        "whale_sell_volume": 800.0,
        "cvd": 150.0,
        "liquidity_heatmap": {
            "clusters": []
        }
    }
    historical_profile = {
        "daily": {
            "poc": 49800.0,
            "vah": 50200.0,
            "val": 49400.0
        }
    }
    macro_context = {
        "trading_session": "NY",
        "session_phase": "ACTIVE",
        "mtf_trends": {},
        "atr": 200.0
    }
    market_environment = {
        "volatility_regime": "NORMAL",
        "trend_direction": "UP",
        "market_structure": "TRENDING",
        "risk_sentiment": "BULLISH",
        "correlation_spy": 0.7,
        "correlation_dxy": -0.3
    }
    orderbook_data = {
        "bid_depth_usd": 1000000.0,
        "ask_depth_usd": 1200000.0,
        "imbalance": 0.1,
        "spread_percent": 0.05,
        "pressure": 0.2
    }
    ml_features = {
        "cross_asset": {
            "btc_eth_corr_7d": 0.8,
            "btc_eth_corr_30d": 0.75,
            "btc_dxy_corr_30d": -0.4,
            "btc_ndx_corr_30d": 0.6,
            "dxy_return_5d": 0.5,
            "dxy_return_20d": 1.2,
            "dxy_momentum": 0.3
        }
    }
    
    return symbol, signal, enriched, flow_metrics, historical_profile, macro_context, market_environment, orderbook_data, ml_features


async def main():
    """Testa a integração do EnhancedRegimeDetector."""
    print("Testando integração do EnhancedRegimeDetector...")
    
    # Cria dados de exemplo
    symbol, signal, enriched, flow_metrics, historical_profile, macro_context, market_environment, orderbook_data, ml_features = create_sample_data()
    
    # Constrói o ai_payload
    ai_payload = build_ai_input(
        symbol=symbol,
        signal=signal,
        enriched=enriched,
        flow_metrics=flow_metrics,
        historical_profile=historical_profile,
        macro_context=macro_context,
        market_environment=market_environment,
        orderbook_data=orderbook_data,
        ml_features=ml_features
    )
    
    # Verifica se a análise de regime foi adicionada
    if "regime_analysis" in ai_payload:
        regime_analysis = ai_payload["regime_analysis"]
        print("Analise de regime adicionada ao ai_payload!")
        print("=" * 80)
        print(f"Market Regime: {regime_analysis.get('market_regime', 'N/A')}")
        print(f"Correlation Regime: {regime_analysis.get('correlation_regime', 'N/A')}")
        print(f"Volatility Regime: {regime_analysis.get('volatility_regime', 'N/A')}")
        print(f"Risk Score: {regime_analysis.get('risk_score', 'N/A')}")
        print(f"Fear & Greed Proxy: {regime_analysis.get('fear_greed_proxy', 'N/A')}")
        print(f"Regime Change Warning: {regime_analysis.get('regime_change_warning', 'N/A')}")
        print(f"Divergence Alert: {regime_analysis.get('divergence_alert', 'N/A')}")
        print(f"Primary Driver: {regime_analysis.get('primary_driver', 'N/A')}")
        print("=" * 80)
        print("Signals Summary:")
        for key, value in regime_analysis.get('signals_summary', {}).items():
            print(f"  {key}: {value}")
    else:
        print("Aviso: Analise de regime nao encontrada no ai_payload.")


if __name__ == "__main__":
    asyncio.run(main())
