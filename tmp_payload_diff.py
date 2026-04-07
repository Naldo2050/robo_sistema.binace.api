import json
import os
import sys

# Adicionar o diretório atual ao path para importar os módulos do robô
sys.path.append(os.getcwd())

import build_compact_payload as bcp
from tests.payload.test_payload_integration_e2e import make_full_event

def perform_diff():
    event = make_full_event()
    compact = bcp.build_compact_payload(event)
    
    # "Bruto" aqui seria o event_data original, que é o que o robô tem antes de compactar
    # Mas para ser justo, vamos focar no que a IA RECEBIA antes (v1/v2) vs agora (v3.1)
    # No entanto, o prompt pede snapshot "antes da compactação final".
    
    print("=== PAYLOAD BRUTO (EVENT_DATA) ===")
    print(json.dumps(event, indent=2))
    print("\n=== PAYLOAD COMPACTADO (FINAL) ===")
    print(json.dumps(compact, indent=2))
    
    # Métricas de tamanho
    raw_size = len(json.dumps(event))
    compact_size = len(json.dumps(compact, separators=(',', ':')))
    
    print(f"\nRaw size: {raw_size} bytes")
    print(f"Compact size: {compact_size} bytes")
    print(f"Reduction: {(1 - compact_size/raw_size)*100:.1f}%")

if __name__ == "__main__":
    perform_diff()
