# Salvar como: test_patch_compressor.py
# Rodar: python test_patch_compressor.py

import sys
sys.path.insert(0, ".")

from common.ai_payload_compressor import PayloadCompressor

print("=" * 60)
print("INICIANDO TESTES DO COMPRESSOR")
print("=" * 60)

compressor = PayloadCompressor()

# TESTE 1: epoch_ms preservado
print("\nTESTE 1: epoch_ms preservado apos compressao")
signal = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "tipo_evento": "ANALYSIS_TRIGGER",
    "janela_numero": 2,
    "contextual_snapshot": {
        "ohlc": {
            "close": 68972.1,
            "open": 68959.2,
            "high": 69020.0,
            "low": 68959.2,
            "vwap": 68994.4
        }
    }
}
result = compressor.compress(signal)
assert "epoch_ms" in result, "FALHOU: epoch_ms nao encontrado no resultado!"
assert isinstance(result["epoch_ms"], int), f"FALHOU: epoch_ms nao e int: {type(result['epoch_ms'])}"
assert result["epoch_ms"] == 1773106319933, f"FALHOU: valor errado: {result['epoch_ms']}"
assert "t" in result, "FALHOU: chave 't' nao encontrada!"
assert result["t"] == 1773106319933, f"FALHOU: 't' valor errado: {result['t']}"
print(f"  [OK] epoch_ms = {result['epoch_ms']}")
print(f"  [OK] t = {result['t']}")

# TESTE 2: multi_tf com chave "trend"
print("\nTESTE 2: multi_tf com chave 'trend' (signal atual)")
signal2 = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "contextual_snapshot": {
        "ohlc": {"close": 68972.1},
        "multi_tf": {
            "1m": {"trend": "bullish", "rsi": 65.5},
            "5m": {"trend": "neutral", "rsi": 52.0},
            "15m": {"trend": "bearish", "rsi": 44.0},
        }
    }
}
result2 = compressor.compress(signal2)
tf = result2.get("tf", {})
assert "1m" in tf, f"FALHOU: '1m' nao encontrado em tf: {tf}"
assert tf["1m"].get("tr") == "bullish", f"FALHOU: trend errado: {tf['1m']}"
assert tf["1m"].get("rs") == 65.5, f"FALHOU: rsi errado: {tf['1m']}"
print(f"  [OK] tf[1m] = {tf.get('1m')}")
print(f"  [OK] tf[5m] = {tf.get('5m')}")
print(f"  [OK] tf[15m] = {tf.get('15m')}")

# TESTE 3: epoch_ms fallback via fluxo_continuo
print("\nTESTE 3: epoch_ms via fluxo_continuo.time_index")
signal3 = {
    "symbol": "BTCUSDT",
    "tipo_evento": "Absorcao",
    "contextual_snapshot": {
        "ohlc": {"close": 68972.1}
    },
    "fluxo_continuo": {
        "time_index": {
            "epoch_ms": 1773106320000
        }
    }
}
result3 = compressor.compress(signal3)
assert result3.get("epoch_ms") == 1773106320000, \
    f"FALHOU: epoch_ms via fluxo_continuo: {result3.get('epoch_ms')}"
print(f"  [OK] epoch_ms via fluxo_continuo = {result3['epoch_ms']}")

# TESTE 4: macro sempre presente
print("\nTESTE 4: secao macro sempre presente (secao critica)")
signal4 = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "contextual_snapshot": {
        "ohlc": {"close": 68972.1},
        "market_context": {
            "trading_session": "NY",
            "session_phase": "ACTIVE"
        },
        "market_environment": {
            "volatility_regime": "LOW",
            "trend_direction": "UP"
        }
    }
}
# Comprime 3 vezes seguidas (simula janelas consecutivas)
r4a = compressor.compress(signal4)
r4b = compressor.compress(signal4)
r4c = compressor.compress(signal4)
for i, r in enumerate([r4a, r4b, r4c], 1):
    assert r.get("macro") is not None, \
        f"FALHOU: macro ausente na janela {i}!"
    print(f"  [OK] Janela {i}: macro presente = {r['macro']}")

print("\n" + "=" * 60)
print("TODOS OS TESTES PASSARAM!")
print("=" * 60)
