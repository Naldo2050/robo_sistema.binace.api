# Salvar como: test_patch_compressor_v3.py
# Rodar: python test_patch_compressor_v3.py

import sys
sys.path.insert(0, ".")

from market_orchestrator.ai.payload_compressor_v3 import (
    compress_payload_v3,
    REGIME_MAP,
    _compress_derivatives,
    _compress_volume_profile,
)

print("=" * 60)
print("INICIANDO TESTES DO COMPRESSOR V3")
print("=" * 60)

# ── TESTE 1: epoch_ms preservado ────────────────────────────
print("\nTESTE 1: epoch_ms preservado com fallback")
payload1 = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "tipo_evento": "Absorcao",
    "preco_fechamento": 68972.1,
}
result1 = compress_payload_v3(payload1)
assert result1.get("epoch_ms") == 1773106319933, \
    f"FALHOU: epoch_ms={result1.get('epoch_ms')}"
print(f"  [OK] epoch_ms = {result1['epoch_ms']}")

# Sem epoch_ms → usa fallback (now)
payload1b = {"symbol": "BTCUSDT", "preco_fechamento": 68972.1}
result1b = compress_payload_v3(payload1b)
assert isinstance(result1b.get("epoch_ms"), int), \
    f"FALHOU: epoch_ms deve ser int, got {type(result1b.get('epoch_ms'))}"
assert result1b["epoch_ms"] > 1_000_000_000_000, \
    f"FALHOU: epoch_ms suspeito: {result1b['epoch_ms']}"
print(f"  [OK] epoch_ms fallback = {result1b['epoch_ms']} (now)")

# ── TESTE 2: REGIME_MAP com chaves inglesas ──────────────────
print("\nTESTE 2: REGIME_MAP com chaves inglesas")
assert REGIME_MAP.get("neutral") == "NEUT", \
    f"FALHOU: neutral={REGIME_MAP.get('neutral')}"
assert REGIME_MAP.get("bullish") == "UP", \
    f"FALHOU: bullish={REGIME_MAP.get('bullish')}"
assert REGIME_MAP.get("bearish") == "DOWN", \
    f"FALHOU: bearish={REGIME_MAP.get('bearish')}"
assert REGIME_MAP.get("Alta") == "UP", \
    f"FALHOU: Alta={REGIME_MAP.get('Alta')}"
print(f"  [OK] neutral -> {REGIME_MAP['neutral']}")
print(f"  [OK] bullish -> {REGIME_MAP['bullish']}")
print(f"  [OK] bearish -> {REGIME_MAP['bearish']}")

# ── TESTE 3: derivatives busca campo correto ────────────────
print("\nTESTE 3: derivatives busca 'derivatives' nao 'derivatives_context'")
payload3 = {
    "symbol": "BTCUSDT",
    "derivatives": {
        "BTCUSDT": {
            "funding_rate_percent": 0.01,
            "open_interest": 82416.199,
            "open_interest_usd": 5678451386.24,
            "long_short_ratio": 1.48
        },
        "ETHUSDT": {
            "long_short_ratio": 1.77
        }
    }
}
result3 = _compress_derivatives(payload3)
assert result3 is not None, "FALHOU: derivatives retornou None!"
assert "btc_lsr" in result3, f"FALHOU: btc_lsr ausente: {result3}"
assert result3["btc_lsr"] == 1.48, \
    f"FALHOU: btc_lsr={result3['btc_lsr']}"
assert "eth_lsr" in result3, f"FALHOU: eth_lsr ausente"
print(f"  [OK] btc_lsr = {result3['btc_lsr']}")
print(f"  [OK] btc_oi = {result3.get('btc_oi')}")
print(f"  [OK] eth_lsr = {result3['eth_lsr']}")

# ── TESTE 4: volume profile com historical_vp ───────────────
print("\nTESTE 4: volume profile busca historical_vp")
payload4 = {
    "symbol": "BTCUSDT",
    "historical_vp": {
        "daily": {
            "poc": 68426, "vah": 69153, "val": 68425,
            "status": "success"
        },
        "weekly": {
            "poc": 70600, "vah": 73168, "val": 68319,
            "status": "success"
        },
        "monthly": {
            "poc": 68177, "vah": 70929, "val": 67532,
            "status": "success"
        }
    }
}
result4 = _compress_volume_profile(payload4)
assert result4 is not None, "FALHOU: volume_profile retornou None!"
assert "daily" in result4, f"FALHOU: daily ausente: {result4}"
assert result4["daily"]["poc"] == 68426, \
    f"FALHOU: poc={result4['daily']['poc']}"
assert "weekly" in result4, f"FALHOU: weekly ausente"
assert "monthly" in result4, f"FALHOU: monthly ausente"
print(f"  [OK] daily POC = {result4['daily']['poc']}")
print(f"  [OK] weekly POC = {result4['weekly']['poc']}")
print(f"  [OK] monthly POC = {result4['monthly']['poc']}")

# ── TESTE 5: timeframes com "neutral" mapeado ───────────────
print("\nTESTE 5: timeframes com 'neutral' mapeado corretamente")
payload5 = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "preco_fechamento": 68972.1,
    "multi_tf": {
        "1m": {"trend": "neutral"},
        "5m": {"trend": "bullish", "rsi": 65.5},
        "15m": {"trend": "bearish"},
    }
}
result5 = compress_payload_v3(payload5)
tf = result5.get("tf", {})
assert tf.get("1m", {}).get("t") == "NEUT", \
    f"FALHOU: 1m trend={tf.get('1m', {}).get('t')}"
assert tf.get("5m", {}).get("t") == "UP", \
    f"FALHOU: 5m trend={tf.get('5m', {}).get('t')}"
assert tf.get("15m", {}).get("t") == "DOWN", \
    f"FALHOU: 15m trend={tf.get('15m', {}).get('t')}"
print(f"  [OK] 1m: neutral -> {tf['1m']['t']}")
print(f"  [OK] 5m: bullish -> {tf['5m']['t']}")
print(f"  [OK] 15m: bearish -> {tf['15m']['t']}")

print("\n" + "=" * 60)
print("TODOS OS TESTES PASSARAM!")
print("=" * 60)
