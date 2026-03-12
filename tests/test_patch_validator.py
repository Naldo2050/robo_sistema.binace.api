import sys
sys.path.insert(0, ".")

from market_orchestrator.ai.llm_response_validator import (
    validate_llm_response,
    validate_json_structure,
    is_truncated_json,
    ACTION_ALIASES,
    FALLBACK_RESPONSE,
)

print("=" * 60)
print("INICIANDO TESTES DO VALIDATOR")
print("=" * 60)

# ── TESTE 1: action 'short' mapeado para 'sell' ──────────────
print("\nTESTE 1: action 'short' mapeado para 'sell'")
response_short = '{"sentiment":"bearish","confidence":0.68,"action":"short","rationale":"venda detectada","entry_zone":null,"invalidation_zone":null,"region_type":null}'
result1 = validate_llm_response(response_short)
assert result1.valid == True, \
    f"FALHOU: deveria ser valido! error={result1.error_reason}"
assert result1.parsed["action"] == "sell", \
    f"FALHOU: action deveria ser 'sell', got '{result1.parsed['action']}'"
assert result1.is_fallback == False, "FALHOU: nao deveria ser fallback!"
print(f"  [OK] 'short' -> mapeado para '{result1.parsed['action']}'")

# ── TESTE 2: action 'long' mapeado para 'buy' ────────────────
print("\nTESTE 2: action 'long' mapeado para 'buy'")
response_long = '{"sentiment":"bullish","confidence":0.75,"action":"long","rationale":"compra","entry_zone":null,"invalidation_zone":null,"region_type":null}'
result2 = validate_llm_response(response_long)
assert result2.valid == True, f"FALHOU: error={result2.error_reason}"
assert result2.parsed["action"] == "buy", \
    f"FALHOU: got '{result2.parsed['action']}'"
print(f"  [OK] 'long' -> mapeado para '{result2.parsed['action']}'")

# ── TESTE 3: entry_zone normalizada ──────────────────────────
print("\nTESTE 3: entry_zone em diferentes formatos")
test_cases = [
    ('[68425, 69153]', [68425.0, 69153.0]),
    ('null', None),
    # Numero unico e convertido para ponto unico [x, x]
    ('68425', [68425.0, 68425.0]),
]
for zone_input, expected in test_cases:
    data = {
        "sentiment": "bullish",
        "confidence": 0.8,
        "action": "buy",
        "entry_zone": eval(zone_input) if zone_input != 'null' else None,
        "invalidation_zone": None
    }
    valid, err = validate_json_structure(data)
    assert valid == True, f"FALHOU com zone={zone_input}: {err}"
    assert data["entry_zone"] == expected, \
        f"FALHOU: zone={zone_input} -> got {data['entry_zone']}, expected {expected}"
    print(f"  [OK] entry_zone={zone_input} -> {data['entry_zone']}")

# ── TESTE 4: confidence fora do range → clamp ────────────────
print("\nTESTE 4: confidence clamp em vez de rejeitar")
data4 = {
    "sentiment": "bullish",
    "confidence": 1.15,    # fora do range!
    "action": "buy",
    "entry_zone": None,
    "invalidation_zone": None
}
valid4, err4 = validate_json_structure(data4)
assert valid4 == True, f"FALHOU: deveria fazer clamp! error={err4}"
assert data4["confidence"] == 1.0, \
    f"FALHOU: confidence deveria ser 1.0, got {data4['confidence']}"
print(f"  [OK] confidence=1.15 -> clamped para {data4['confidence']}")

data4b = {"sentiment":"bearish","confidence":-0.1,"action":"sell","entry_zone":None,"invalidation_zone":None}
valid4b, err4b = validate_json_structure(data4b)
assert valid4b == True, f"FALHOU: deveria fazer clamp! error={err4b}"
assert data4b["confidence"] == 0.0, \
    f"FALHOU: confidence deveria ser 0.0, got {data4b['confidence']}"
print(f"  [OK] confidence=-0.1 -> clamped para {data4b['confidence']}")

# ── TESTE 5: JSON com texto extra apos (strict=False) ─────────
print("\nTESTE 5: JSON valido com texto extra apos (strict=False)")
response5 = '{"sentiment":"bullish","confidence":0.85,"action":"buy","rationale":"compra forte","entry_zone":null,"invalidation_zone":null,"region_type":null}\nObservacao: aguardar confirmacao'
result5 = validate_llm_response(response5, strict=False)
assert result5.valid == True, \
    f"FALHOU: deveria aceitar! error={result5.error_reason}"
assert result5.parsed["action"] == "buy", \
    f"FALHOU: action={result5.parsed['action']}"
print(f"  [OK] Texto extra ignorado, JSON valido extrado")

# ── TESTE 6: is_truncated sem falso positivo ─────────────────
print("\nTESTE 6: is_truncated_json sem falso positivo")
valid_with_newline = '{"sentiment":"bullish","action":"buy","confidence":0.8}\n'
assert is_truncated_json(valid_with_newline) == False, \
    "FALHOU: JSON valido com newline detectado como truncado!"
print(f"  [OK] JSON com newline nao e truncado")

actually_truncated = '{"sentiment":"bullish","action":"buy","confidence":'
assert is_truncated_json(actually_truncated) == True, \
    "FALHOU: JSON truncado nao detectado!"
print(f"  [OK] JSON truncado detectado corretamente")

# ── TESTE 7: _is_fallback consistente ────────────────────────
print("\nTESTE 7: FALLBACK_RESPONSE tem _is_fallback")
assert "_is_fallback" in FALLBACK_RESPONSE, \
    "FALHOU: _is_fallback nao esta no FALLBACK_RESPONSE!"
assert FALLBACK_RESPONSE["_is_fallback"] == True, \
    "FALHOU: _is_fallback deveria ser True!"
assert "_is_valid" in FALLBACK_RESPONSE, \
    "FALHOU: _is_valid nao esta no FALLBACK_RESPONSE!"
print(f"  [OK] FALLBACK_RESPONSE tem _is_fallback e _is_valid")

print("\n" + "=" * 60)
print("TODOS OS TESTES PASSARAM!")
print("=" * 60)
