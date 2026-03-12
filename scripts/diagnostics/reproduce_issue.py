
from data_validator import DataValidator
import time

def reproduce():
    v = DataValidator()
    
    # Evento similar ao ANALYSIS_TRIGGER gerado pelo DataPipeline
    # Arredondado para 4 casas decimais (como MetricsProcessor faz)
    event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "epoch_ms": int(time.time() * 1000),
        "volume_compra": 1.5003,
        "volume_venda": 0.8003,
        "delta": 0.7001, # Simula uma pequena discrepância original que foi arredondada separadamente
        "volume_total": 2.3006,
        "preco_fechamento": 50000.0,
        "timestamp_utc": "2025-12-18T20:00:00.000Z"
    }
    
    print("--- Antes da validação ---")
    print(f"Delta original: {event['delta']}")
    
    result = v.validate_and_clean(event.copy())
    
    print("\n--- Depois da validação ---")
    if result:
        print(f"Delta corrigido: {result['delta']}")
        print(f"Contadores de correção: {v.corrections_count}")
        
    # Teste de inconsistência que gera correção:
    # 1.5003 - 0.8003 = 0.7000
    # Como o delta no evento é 0.7001, a diferença é 0.0001
    # 0.0001 > BTC_TOLERANCE (1e-8) -> CORREÇÃO!

if __name__ == "__main__":
    reproduce()
