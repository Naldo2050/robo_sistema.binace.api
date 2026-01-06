# debug_validator.py - Script temporário para debug
from data_validator import DataValidator

print("=" * 50)
print("TESTE 1: whale_sell_volume > volume_venda")
print("=" * 50)

validator1 = DataValidator()
event1 = {
    "epoch_ms": 1733918400000,
    "timestamp": "2024-12-11T10:00:00Z",
    "volume_compra": 10.0,
    "volume_venda": 8.0,
    "whale_buy_volume": 5.0,
    "whale_sell_volume": 9.0,  # > volume_venda
    "volume_total": 18.0,
    "delta": 2.0,
    "preco_fechamento": 50000.0,
    "enriched_snapshot": {"volume_total": 18.0, "delta_fechamento": 2.0}
}
result1 = validator1.validate_and_clean(event1)
print(f"result is None: {result1 is None}")
if result1:
    print(f"whale_sell_volume: {result1.get('whale_sell_volume')}")
    print(f"volume_venda: {result1.get('volume_venda')}")

print()
print("=" * 50)
print("TESTE 2: Precisão numérica")
print("=" * 50)

validator2 = DataValidator()
event2 = {
    "epoch_ms": 1733918401000,
    "timestamp": "2024-12-11T10:00:01Z",
    "volume_compra": 0.623456789999,
    "volume_venda": 0.500000009999,
    "delta": 0.999999999999,
    "preco_fechamento": 50000.123456789,
    "volume_total": 100.0,
}
result2 = validator2.validate_and_clean(event2)
print(f"result is None: {result2 is None}")
if result2:
    print(f"volume_compra: {result2.get('volume_compra')}")
    print(f"volume_venda: {result2.get('volume_venda')}")
    print(f"delta: {result2.get('delta')}")
    print(f"volume_total: {result2.get('volume_total')}")
    print(f"preco_fechamento: {result2.get('preco_fechamento')}")
