import json

with open('dados/eventos_parsed.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

print(f'=== AUDITORIA DE {len(events)} EVENTOS ===')
print()

for i, evt in enumerate(events):
    print(f'--- EVENTO {i+1} ---')
    print(f"Tipo: {evt.get('tipo_evento', 'N/A')}")
    print(f"Symbol: {evt.get('symbol', 'N/A')}")
    print(f"Data Context: {evt.get('data_context', 'N/A')}")
    
    raw = evt.get('raw_event', {})
    if raw:
        print(f"  Timestamp: {raw.get('timestamp', 'N/A')}")
        print(f"  Delta: {raw.get('delta', 'N/A')}")
        print(f"  Volume Total: {raw.get('volume_total', 'N/A')}")
        print(f"  Preco Fechamento: {raw.get('preco_fechamento', 'N/A')}")
        
        ml = raw.get('ml_features', {})
        if ml:
            print(f"  ML Features: {list(ml.keys())}")
        
        ob = raw.get('orderbook_data', {})
        if ob:
            print(f"  Orderbook: mid={ob.get('mid')}, spread={ob.get('spread')}")
    print()

# Mostrar estrutura completa do primeiro evento
print('=== ESTRUTURA COMPLETA DO PRIMEIRO EVENTO ===')
print(json.dumps(events[0], indent=2, ensure_ascii=False)[:5000])

print('\n=== ESTRUTURA COMPLETA DO ULTIMO EVENTO ===')
print(json.dumps(events[-1], indent=2, ensure_ascii=False)[:5000])
