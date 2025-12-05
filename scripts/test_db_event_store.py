# scripts/test_db_event_store.py
import sys
import os
import time
import logging
import shutil
from datetime import datetime

# Adiciona diretório raiz ao path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.event_store import EventStore

# Configura log para ver output no console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("\n" + "="*60)
    print("🧪 TESTE DE INTEGRAÇÃO: EventStore (SQLite)")
    print("="*60)

    # Definir caminho de banco de teste para não poluir o oficial
    test_db_path = "dados/test_trading_bot.db"
    
    # Limpar teste anterior se existir
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"🗑️  Banco de teste anterior removido: {test_db_path}")

    # 1. Instanciar
    print("\n1. Inicializando EventStore...")
    try:
        db = EventStore(db_path=test_db_path)
        print("   ✅ Banco inicializado com sucesso (tabelas criadas).")
    except Exception as e:
        print(f"   ❌ Falha ao inicializar: {e}")
        return

    # 2. Criar dados de exemplo
    print("\n2. Gerando eventos simulados...")
    now = int(time.time() * 1000)
    
    events = [
        {
            "tipo_evento": "OrderBook",
            "ativo": "BTCUSDT",
            "epoch_ms": now - 5000,
            "bid_depth": 1000000,
            "ask_depth": 1200000,
            "is_signal": False,
            "window_id": "W1001"
        },
        {
            "tipo_evento": "Absorção",
            "ativo": "BTCUSDT",
            "epoch_ms": now - 3000,
            "delta": -150.5,
            "is_signal": True,
            "resultado_da_batalha": "Absorção de Compra",
            "window_id": "W1001"
        },
        {
            "tipo_evento": "Alerta",
            "ativo": "ETHUSDT",
            "epoch_ms": now - 1000,
            "descricao": "Volume Spike detectado",
            "is_signal": True,
            "severity": "HIGH"
        }
    ]
    
    # 3. Testar Save Batch
    print(f"\n3. Salvando {len(events)} eventos...")
    start_t = time.time()
    db.save_batch(events)
    end_t = time.time()
    print(f"   ✅ Salvo em {(end_t - start_t)*1000:.2f}ms")

    # 4. Testar Leitura
    print("\n4. Lendo de volta (get_recent_events)...")
    loaded = db.get_recent_events(limit=10)
    print(f"   ✅ Recuperados {len(loaded)} eventos.")
    
    print("\n   📋 Conteúdo recuperado:")
    print("   " + "-" * 50)
    for i, evt in enumerate(loaded):
        ts = evt.get('epoch_ms', 0)
        dt = datetime.fromtimestamp(ts/1000).strftime('%H:%M:%S')
        tipo = evt.get('tipo_evento')
        flag = "🔔" if evt.get('is_signal') else "📄"
        print(f"   {i+1}. {flag} [{dt}] {tipo:<15} | Ativo: {evt.get('ativo')}")
    print("   " + "-" * 50)

    # 5. Validação de Integridade
    print("\n5. Validando integridade...")
    if len(loaded) == 3 and loaded[-1]['tipo_evento'] == "Alerta":
        print("   ✅ Ordem cronológica correta (último evento da lista é o mais recente).")
    else:
        print("   ❌ Ordem incorreta ou dados faltando.")

    # Stats
    print("\n6. Estatísticas do DB:")
    stats = db.get_stats()
    print(f"   {stats}")

    print("\n" + "="*60)
    print("✅ TESTE CONCLUÍDO")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()