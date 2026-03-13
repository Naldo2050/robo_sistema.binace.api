# Salvar como: test_patch_guardrail.py
# Rodar: python test_patch_guardrail.py

import sys
sys.path.insert(0, ".")

from market_orchestrator.ai.llm_payload_guardrail import (
    ensure_safe_llm_payload,
    _is_already_compressed,
    FORBIDDEN_KEYS,
)

print("=" * 60)
print("INICIANDO TESTES DO GUARDRAIL")
print("=" * 60)

# --- TESTE 1: historical_vp NAO ESTA MAIS EM FORBIDDEN_KEYS ---
print("\nTESTE 1: historical_vp removido de FORBIDDEN_KEYS")
assert "historical_vp" not in FORBIDDEN_KEYS, \
    "FALHOU: historical_vp ainda está em FORBIDDEN_KEYS!"
assert "raw_event" in FORBIDDEN_KEYS, \
    "FALHOU: raw_event deve estar em FORBIDDEN_KEYS!"
assert "contextual_snapshot" in FORBIDDEN_KEYS, \
    "FALHOU: contextual_snapshot deve estar em FORBIDDEN_KEYS!"
print(f"  OK: FORBIDDEN_KEYS = {FORBIDDEN_KEYS}")

# ── TESTE 2: payload limpo passa direto ─────────────────────────────
print("\nTESTE 2: payload limpo e pequeno passa direto")
clean_payload = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "price": {"c": 68972.1},
    "ob": {"bid": 760529, "ask": 135115},
    "flow": {"imb": 0.44},
    "quant": {"prob_up": 0.86}
}
result2 = ensure_safe_llm_payload(clean_payload)
assert result2 is not None, "FALHOU: payload limpo retornou None!"
assert result2.get("epoch_ms") == 1773106319933, \
    f"FALHOU: epoch_ms perdido: {result2.get('epoch_ms')}"
assert "ai_payload" not in result2, \
    "FALHOU: wrapper 'ai_payload' não deveria existir!"
print(f"  OK: Payload passou direto, epoch_ms={result2['epoch_ms']}")

# --- TESTE 3: payload com forbidden_keys -> extrai ai_payload ---
print("\nTESTE 3: payload com raw_event extrai ai_payload corretamente")
full_signal = {
    "symbol": "BTCUSDT",
    "tipo_evento": "Absorção",
    "raw_event": {"dados": "brutos" * 1000},        # FORBIDDEN
    "contextual_snapshot": {"dados": "dup" * 1000}, # FORBIDDEN
    "ai_payload": {
        "symbol": "BTCUSDT",
        "epoch_ms": 1773106319933,
        "price": {"c": 68972.1},
        "ob": {"bid": 760529, "ask": 135115},
        "flow": {"imb": 0.44},
        "quant": {"prob_up": 0.86}
    }
}
result3 = ensure_safe_llm_payload(full_signal)
assert result3 is not None, "FALHOU: retornou None com ai_payload presente!"
assert "raw_event" not in result3, \
    "FALHOU: raw_event vazou para o resultado!"
assert "contextual_snapshot" not in result3, \
    "FALHOU: contextual_snapshot vazou!"
assert result3.get("epoch_ms") == 1773106319933, \
    f"FALHOU: epoch_ms perdido: {result3.get('epoch_ms')}"
assert "ai_payload" not in result3, \
    "FALHOU: resultado não deve ter wrapper 'ai_payload'!"
print(f"  OK: Extraiu ai_payload corretamente")
print(f"  OK: epoch_ms = {result3.get('epoch_ms')}")
print(f"  OK: Sem wrapper 'ai_payload'")
print(f"  OK: tipo_evento = {result3.get('tipo_evento')}")

# --- TESTE 4: deteccao de payload ja comprimido ---
print("\nTESTE 4: detecção de payload já comprimido")
compressed_payload = {
    "symbol": "BTCUSDT",
    "epoch_ms": 1773106319933,
    "price": {"c": 68972.1},
    "ob": {"bid": 760529},
    "flow": {"imb": 0.44},
    "tf": {"1m": {"tr": "bullish"}},
    "quant": {"pu": 0.86}
}
assert _is_already_compressed(compressed_payload) == True, \
    "FALHOU: não detectou payload comprimido!"
print(f"  OK: Payload comprimido detectado corretamente")

not_compressed = {
    "symbol": "BTCUSDT",
    "tipo_evento": "Absorção",
    "raw_event": {},
    "preco_fechamento": 68972.1
}
assert _is_already_compressed(not_compressed) == False, \
    "FALHOU: detectou compressão incorretamente!"
print(f"  OK: Payload nao comprimido detectado corretamente")

# --- TESTE 5: payload sem ai_payload retorna None ---
print("\nTESTE 5: payload sem ai_payload e sem dados minimos -> None")
empty_signal = {
    "raw_event": {"dados": "brutos"},
    "contextual_snapshot": {"dados": "dup"},
}
result5 = ensure_safe_llm_payload(empty_signal)
assert result5 is None, \
    f"FALHOU: deveria retornar None, retornou: {result5}"
print(f"  OK: Retornou None corretamente")

print("\n" + "=" * 60)
print("TODOS OS TESTES PASSARAM!")
print("=" * 60)
