import pytest
import json
import sys
import os

# Ajuste de path para importar módulos da raiz se necessário
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importa o otimizador existente em src/utils/
from src.utils.ai_payload_optimizer import AIPayloadOptimizer


def test_payload_compression_ratio():
    """Testa se a compressão está reduzindo drasticamente o tamanho sem perder dados críticos."""
    # Simulação do JSON gigante fornecido pelo usuário
    dummy_huge_event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "timestamp_utc": "2026-01-30T12:30:00Z",
        "raw_event": {
            "raw_event": {  # Simula aninhamento duplo
                "price": 82747.1,
                "volume": 1.344,
                "delta": -0.515,
                "multi_tf": {
                    "15m": {"tendencia": "Baixa", "rsi_short": 49.42, "regime": "Range"},
                    "1h": {"tendencia": "Baixa", "rsi_short": 40.52, "regime": "Range"},
                    "4h": {"tendencia": "Baixa", "rsi_short": 22.46, "regime": "Manipulação"}
                },
                "orderbook_data": {
                    "imbalance": -0.376,
                    "bid_depth_usd": 482960.43,
                    "ask_depth_usd": 1065554.61
                }
            }
        },
        "contextual_snapshot": {"lixo": "dados duplicados enormes aqui..."}
    }

    # Tamanho original estimado
    original_str = json.dumps(dummy_huge_event)
    original_size = len(original_str)

    # Otimizar - o método retorna um dict
    optimized_dict = AIPayloadOptimizer.optimize(dummy_huge_event)
    optimized_str = json.dumps(optimized_dict, separators=(',', ':'))
    optimized_size = len(optimized_str)
    
    print(f"\nOriginal Size: {original_size} chars")
    print(f"Optimized Size: {optimized_size} chars")
    print(f"Payload Otimizado: {optimized_str}")

    # Validações
    assert optimized_size < original_size * 0.7, "A compressão deve reduzir pelo menos 30% do tamanho"
    assert '"lixo"' not in optimized_str, "Dados redundantes devem ser removidos"
    assert 'tf' in optimized_dict, "Dados Multi-Timeframe devem estar presentes"
    assert '15m' in optimized_dict.get('tf', {}), "Timeframes devem ser preservados"
    assert optimized_dict.get('symbol') == 'BTCUSDT', "Symbol deve ser preservado"


def test_payload_optimizer_with_realistic_data():
    """Testa o otimizador com dados mais realistas de mercado."""
    event_data = {
        "symbol": "ETHUSDT",
        "timestamp_utc": "2026-01-30T12:30:00Z",
        "raw_event": {
            "preco_fechamento": 3250.75,
            "volume_total": 12345.67,
            "delta": 0.234,
            "multi_tf": {
                "15m": {"tendencia": "Alta", "rsi_short": 65.5, "regime": "Tendência"},
                "1h": {"tendencia": "Alta", "rsi_short": 58.2, "regime": "Tendência"},
                "4h": {"tendencia": "Neutro", "rsi_short": 52.1, "regime": "Range"},
                "1d": {"tendencia": "Alta", "rsi_short": 61.3, "regime": "Tendência"}
            },
            "orderbook_data": {
                "imbalance": 0.123,
                "bid_depth_usd": 892345.67,
                "ask_depth_usd": 765432.10,
                "walls_detected": True
            },
            "historical_vp": {
                "daily": {
                    "poc": 3248.0,
                    "vah": 3260.5,
                    "val": 3235.0
                }
            }
        },
        "fluxo_continuo": {
            "cvd": 456.78,
            "order_flow": {
                "flow_imbalance": 0.345
            },
            "absorption_analysis": {
                "current_absorption": {
                    "label": "Alta Pressão Compradora"
                }
            }
        }
    }
    
    original_str = json.dumps(event_data, separators=(',', ':'))
    original_size = len(original_str)
    
    optimized_dict = AIPayloadOptimizer.optimize(event_data)
    optimized_str = json.dumps(optimized_dict, separators=(',', ':'))
    optimized_size = len(optimized_str)
    
    print(f"\n[Teste Realista] Original: {original_size} chars, Otimizado: {optimized_size} chars")
    print(f"Redução: {((original_size - optimized_size) / original_size * 100):.1f}%")
    print(f"Payload: {optimized_str}")
    
    # Verifica estrutura
    assert optimized_dict.get('symbol') == 'ETHUSDT', "Symbol deve estar presente"
    assert 'price' in optimized_dict, "Preço deve estar presente"
    assert optimized_dict['price'].get('c') == 3250.75, "Preço de fechamento correto"
    assert 'ob' in optimized_dict, "Orderbook deve estar presente"
    assert 'vp' in optimized_dict, "Volume Profile deve estar presente"
    assert 'tf' in optimized_dict, "Multi-timeframe deve estar presente"


def test_payload_optimizer_handles_missing_data():
    """Testa se o otimizador lida corretamente com dados ausentes."""
    minimal_event = {
        "symbol": "BTCUSDT",
        "timestamp": "2026-01-30T12:30:00Z",
        "raw_event": {
            "preco_fechamento": 50000.0  # Usando nome de campo correto
        }
    }
    
    optimized = AIPayloadOptimizer.optimize(minimal_event)
    
    assert optimized.get('symbol') == 'BTCUSDT'
    assert optimized.get('price', {}).get('c') == 50000.0


def test_payload_optimizer_estimate_savings():
    """Testa a função de estimativa de economia."""
    event_data = {
        "symbol": "BTCUSDT",
        "timestamp_utc": "2026-01-30T12:30:00Z",
        "raw_event": {
            "preco_fechamento": 82747.1,
            "volume_total": 1.344,
            "delta": -0.515,
            "multi_tf": {
                "15m": {"tendencia": "Baixa", "rsi_short": 49.42, "regime": "Range"}
            },
            "orderbook_data": {
                "imbalance": -0.376,
                "bid_depth_usd": 482960.43,
                "ask_depth_usd": 1065554.61
            }
        }
    }
    
    savings = AIPayloadOptimizer.estimate_savings(event_data)
    
    assert 'bytes_before' in savings
    assert 'bytes_after' in savings
    assert 'saved_bytes' in savings
    assert 'saved_pct' in savings
    assert savings['saved_bytes'] >= 0
    assert savings['saved_pct'] >= 0
    
    print(f"\nEconomia estimada: {savings['saved_pct']:.1f}%")
    print(f"Bytes antes: {savings['bytes_before']}, depois: {savings['bytes_after']}")


def test_payload_optimizer_compact_keys():
    """Testa se as chaves estão sendo compactadas corretamente."""
    event_data = {
        "symbol": "BTCUSDT",
        "timestamp_utc": "2026-01-30T12:30:00Z",
        "raw_event": {
            "preco_fechamento": 82747.1,
            "volume_total": 1.344,
            "delta": -0.515,
            "multi_tf": {
                "1h": {"tendencia": "Alta", "rsi_short": 55.0, "regime": "Tendência"}
            },
            "orderbook_data": {
                "imbalance": -0.376,
                "bid_depth_usd": 482960.43,
                "ask_depth_usd": 1065554.61
            },
            "historical_vp": {
                "daily": {"poc": 82500, "vah": 83000, "val": 82000}
            }
        }
    }
    
    optimized = AIPayloadOptimizer.optimize(event_data)
    
    # Verifica chaves compactas no orderbook
    assert 'ob' in optimized
    ob = optimized['ob']
    assert 'imb' in ob or 'imbalance' in ob, "Imbalance deve estar presente"
    assert 'bid' in ob or 'bid_depth_usd' in ob, "Bid depth deve estar presente"
    assert 'ask' in ob or 'ask_depth_usd' in ob, "Ask depth deve estar presente"
    
    # Verifica multi-timeframe
    assert 'tf' in optimized
    tf = optimized['tf']
    if tf:
        for tf_name, tf_data in tf.items():
            assert 'trend' in tf_data or 'tendencia' in tf_data, "Tendência deve estar presente"
            assert 'rsi' in tf_data or 'rsi_short' in tf_data, "RSI deve estar presente"
