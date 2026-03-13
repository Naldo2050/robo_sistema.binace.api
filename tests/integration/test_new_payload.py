import sys
import json
import logging
from market_orchestrator.ai import ai_payload_builder

# Configuração mínima de Logging
logging.basicConfig(level=logging.INFO)

# Dados Mockados
mock_signal = {
    "tipo_evento": "TEST_EVENT",
    "timestamp": "2025-12-11T12:00:00Z",
    "preco_fechamento": 50000.0,
    "descricao": "Teste de Payload",
    "janela_numero": 123
}

mock_enriched = {
    "ohlc": {
        "open": 49950.0,
        "high": 50050.0,
        "low": 49900.0,
        "close": 50000.0, # Fechamento perto do topo
        "vwap": 49980.0
    }
}

mock_flow = {
    "order_flow": {
        "net_flow_1m": 120.5,
        "flow_imbalance": 0.15
    },
    "whale_delta": 50.0
}

mock_ob_data = {
    "bid_depth_usd": 100000.0,
    "ask_depth_usd": 80000.0,
    "imbalance": 20000.0,
    # 🆕 Métricas de profundidade que o OrderbookAnalyzer vai gerar
    "depth_metrics": {
        "bid_liquidity_top5": 50000.0,
        "ask_liquidity_top5": 30000.0,
        "depth_imbalance": 0.25
    }
}

mock_macro = {
    "trading_session": "NY",
    "session_phase": "OPEN",
    "market_structure": "TRENDING_UP"
}

# Executa o Builder
try:
    payload = ai_payload_builder.build_ai_input(
        symbol="BTCUSDT",
        signal=mock_signal,
        enriched=mock_enriched,
        flow_metrics=mock_flow,
        historical_profile={},
        macro_context=mock_macro,
        market_environment={},
        orderbook_data=mock_ob_data,
        ml_features={}
    )
    
    # Validações
    print("\n=== VALIDAÇÃO DO PAYLOAD ===\n")
    
    # 1. Price Action
    pa = payload["price_context"].get("price_action", {})
    print(f"[Price Action] Body Pct: {pa.get('candle_body_pct', -1):.4f}% (Esperado > 0)")
    print(f"[Price Action] Close Position: {pa.get('close_position', -1):.2f} (Esperado ~0.66 [Top 2/3])")
    
    # 2. Depth Metrics
    depth = payload["orderbook_context"].get("depth_metrics", {})
    print(f"[Depth] Bid Top5: {depth.get('bid_liquidity_top5', -1)}")
    print(f"[Depth] Imbalance: {depth.get('depth_imbalance', -99):.2f}")
    
    # 3. Output JSON Completo
    print("\n=== JSON GERADO (Snippet) ===")
    print(json.dumps(payload, indent=2)[:1000])

except Exception as e:
    print(f"❌ Erro ao construir payload: {e}")
    import traceback
    traceback.print_exc()
