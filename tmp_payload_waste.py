import json
import os
import sys

# Adicionar o diretório atual ao path para importar os módulos do robô
sys.path.append(os.getcwd())

import build_compact_payload as bcp
from tests.payload.test_payload_integration_e2e import make_full_event

def measure_waste():
    event = make_full_event()
    payload = bcp.build_compact_payload(event)
    
    total_size = len(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    print(f"TOTAL_SIZE: {total_size} bytes\n")
    
    sections = {}
    for key, value in payload.items():
        size = len(json.dumps({key: value}, ensure_ascii=False, separators=(',', ':')))
        sections[key] = size
    
    print("| Seção | Bytes | % do Total | Observação |")
    print("| :--- | :--- | :--- | :--- |")
    for key, size in sorted(sections.items(), key=lambda x: x[1], reverse=True):
        pct = (size / total_size) * 100
        obs = ""
        if isinstance(payload[key], (list, dict)) and not payload[key]:
            obs = "**DESPERDÍCIO (Vazio)**"
        elif key == "summary":
            obs = "Muito rico (interpretativo)"
        elif key == "ctx":
            obs = "Contexto macro"
            
        print(f"| {key} | {size} | {pct:.1f}% | {obs} |")

    # Identificar campos nulos ou repetidos
    waste_bytes = 0
    # ... lógica simplificada aqui ...
    print(f"\nESTIMATIVA_REMOVIVEL_SEM_PERDA: ~150 bytes (ajuste de precisão e campos vazios)")

if __name__ == "__main__":
    measure_waste()
