from ai_payload_optimizer import AIPayloadOptimizer
import json

# Simular evento típico
evento_teste = {
    "raw_event": {
        "raw_event": {
            "preco_fechamento": 92662,
            "enriched_snapshot": {
                "ohlc": {"open": 92695, "high": 92728, "low": 92649, "close": 92662, "vwap": 92695}
            },
            "fluxo_continuo": {
                "cvd": -4.12,
                "order_flow": {
                    "net_flow_1m": -15094, 
                    "net_flow_5m": -95150,
                    "net_flow_15m": -382161,
                    "flow_imbalance": -0.78,
                    "aggressive_buy_pct": 11.04,
                    "aggressive_sell_pct": 88.96
                },
                "tipo_absorcao": "Neutra",
                "absorption_analysis": {
                    "current_absorption": {
                        "label": "Neutra",
                        "index": 0.03
                    }
                },
                "liquidity_heatmap": {
                    "clusters": [{
                        "center": 92661,
                        "imbalance": -0.67
                    }]
                }
            },
            "orderbook_data": {
                "spread": 0.1, 
                "imbalance": -0.02,
                "pressure": -0.02,
                "bid_depth_usd": 782727,
                "ask_depth_usd": 815251,
                "depth_metrics": {
                    "depth_imbalance": 0.056
                }
            },
            "multi_tf": {
                "15m": {"tendencia": "Baixa", "regime": "Range", "rsi_short": 46, "adx": 36, "preco_atual": 92671, "mme_21": 92795},
                "1h": {"tendencia": "Baixa", "regime": "Range", "rsi_short": 35, "adx": 25, "preco_atual": 92671, "mme_21": 93120},
                "4h": {"tendencia": "Baixa", "regime": "Range", "rsi_short": 26, "adx": 43, "preco_atual": 92671, "mme_21": 93980},
                "1d": {"tendencia": "Alta", "regime": "Manipulação", "rsi_short": 46, "adx": 39, "preco_atual": 92671, "mme_21": 92515}
            },
            "historical_vp": {
                "daily": {"poc": 92430, "vah": 92702, "val": 92370, "status": "success", "hvns": [92430, 92505, 92619]},
                "weekly": {"poc": 95604, "vah": 96544, "val": 91090, "status": "success", "hvns": [92100, 92149]},
                "monthly": {"poc": 89020, "vah": 90300, "val": 87580, "status": "success"}
            },
            "derivatives": {
                "BTCUSDT": {
                    "funding_rate_percent": 0.01, 
                    "open_interest": 94082, 
                    "long_short_ratio": 2.04
                }
            },
            "market_context": {"trading_session": "NY", "session_phase": "ACTIVE"},
            "market_environment": {
                "volatility_regime": "LOW", 
                "trend_direction": "DOWN",
                "market_structure": "RANGE_BOUND",
                "risk_sentiment": "BEARISH"
            },
            "timestamp_utc": "2026-01-20T00:48:00Z"
        }
    },
    "janela_numero": 1
}

print("=" * 70)
print("DIAGNÓSTICO DE OTIMIZAÇÃO")
print("=" * 70)

# Testar otimização
try:
    resultado = AIPayloadOptimizer.optimize(evento_teste)
    economia = AIPayloadOptimizer.estimate_savings(evento_teste)
    
    print(f"\n[OK] Tamanho Original: {economia['original_bytes']:,} bytes")
    print(f"[OK] Tamanho Otimizado: {economia['optimized_bytes']:,} bytes")
    print(f"[OK] Reducao: {economia['reduction_pct']}%")
    print(f"[OK] Tokens Originais (estimado): ~{economia.get('original_tokens_est', 'N/A'):,}")
    print(f"[OK] Tokens Otimizados (estimado): ~{economia.get('optimized_tokens_est', 'N/A'):,}")
    
    # Calcular economia de tokens manualmente se a chave não existir
    if 'tokens_saved' in economia:
        print(f"[OK] Tokens Economizados: ~{economia['tokens_saved']:,}")
    else:
        tokens_saved = economia.get('original_tokens_est', 0) - economia.get('optimized_tokens_est', 0)
        print(f"[OK] Tokens Economizados: ~{tokens_saved:,}")
    
    print("\n" + "=" * 70)
    print("PAYLOAD OTIMIZADO:")
    print("=" * 70)
    payload_json = json.dumps(resultado, indent=2, ensure_ascii=False)
    print(payload_json)
    
    print("\n" + "=" * 70)
    print("VALIDAÇÃO:")
    print("=" * 70)
    
    # Verificar campos essenciais
    campos_esperados = {
        'price': 'Dados de preço',
        'flow': 'Fluxo de ordens',
        'ob': 'Orderbook',
        'tf': 'Timeframes',
        'vp': 'Volume Profile',
        'ctx': 'Contexto de mercado',
        'deriv': 'Derivativos'
    }
    
    for campo, descricao in campos_esperados.items():
        if campo in resultado:
            valor = resultado[campo]
            if isinstance(valor, dict) and valor:
                print(f"  [OK]  {campo:8s} | {descricao:25s} | {len(str(valor)):4d} bytes")
            else:
                print(f"  [WARN] {campo:8s} | {descricao:25s} | VAZIO ou INVALIDO")
        else:
            print(f"  [ERRO] {campo:8s} | {descricao:25s} | AUSENTE")
    
    print("\n" + "=" * 70)
    
    # Diagnóstico de tamanho
    if economia['optimized_bytes'] < 100:
        print("[WARN] ALERTA: Payload muito pequeno (< 100 bytes)")
        print("   Possível problema na extração de dados do evento original")
    elif economia['optimized_bytes'] < 500:
        print("[WARN] AVISO: Payload menor que esperado (< 500 bytes)")
        print("   Alguns dados podem estar faltando")
    elif economia['optimized_bytes'] < 3000:
        print("[OK] Tamanho adequado (500-3000 bytes)")
    else:
        print("[WARN] Payload maior que esperado (> 3000 bytes)")
        print("   Otimização pode estar incompleta")
    
except Exception as e:
    print(f"\n[ERRO] {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
