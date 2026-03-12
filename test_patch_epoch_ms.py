# Teste Rapido - Verificar correcoes do patch
import sys
sys.path.insert(0, ".")

from market_orchestrator.ai.ai_payload_builder import _safe_epoch_ms

print("=" * 60)
print("TESTE 1: epoch_ms como int valido")
signal_1 = {"epoch_ms": 1773106319933}
result_1 = _safe_epoch_ms(signal_1)
assert isinstance(result_1, int), f"FALHOU: {type(result_1)}"
assert result_1 == 1773106319933, f"FALHOU: {result_1}"
print(f"  OK -> {result_1}")

print("TESTE 2: timestamp como STRING (bug original)")
signal_2 = {"timestamp": "2026-03-09 21:32:00-04:00"}
result_2 = _safe_epoch_ms(signal_2)
assert isinstance(result_2, int), f"FALHOU: {type(result_2)}"
assert result_2 > 1_000_000_000_000, f"FALHOU: {result_2}"
print(f"  OK -> {result_2}")

print("TESTE 3: epoch_ms em fluxo_continuo")
signal_3 = {
    "fluxo_continuo": {
        "time_index": {"epoch_ms": 1773106320000}
    }
}
result_3 = _safe_epoch_ms(signal_3)
assert result_3 == 1773106320000, f"FALHOU: {result_3}"
print(f"  OK -> {result_3}")

print("TESTE 4: sem nenhum timestamp (fallback now)")
signal_4 = {"tipo_evento": "TEST"}
result_4 = _safe_epoch_ms(signal_4)
assert isinstance(result_4, int), f"FALHOU: {type(result_4)}"
assert result_4 > 1_000_000_000_000, f"FALHOU: {result_4}"
print(f"  OK -> {result_4} (agora)")

print("=" * 60)
print("TODOS OS TESTES PASSARAM!")
