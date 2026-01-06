# teste_separador.py - Debug de separadores

import logging
import time
from event_saver import EventSaver
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🧪 TESTE DE SEPARADORES\n")
print("="*80)

saver = EventSaver(sound_alert=False)

# Força criação de 5 janelas diferentes
for i in range(5):
    epoch_base = int(datetime.now().timestamp() * 1000)
    epoch_minuto = (epoch_base // 60000) * 60000  # Arredonda para minuto
    epoch_minuto += i * 60000  # Adiciona 1 minuto por iteração
    
    evento = {
        "tipo_evento": f"TESTE_JANELA_{i+1}",
        "is_signal": True,
        "epoch_ms": epoch_minuto,
        "window_id": epoch_minuto,
        "price_data": {
            "current": {
                "last": 110000 + i * 100,
                "volume": 100 + i
            }
        }
    }
    
    print(f"\n📌 Salvando evento {i+1} (epoch: {epoch_minuto})...")
    saver.save_event(evento)
    time.sleep(0.5)

print("\n⏳ Aguardando flush...")
time.sleep(7)

print("\n📊 Estatísticas:")
stats = saver.get_stats()
for k, v in stats.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    {k2}: {v2}")
    else:
        print(f"  {k}: {v}")

saver.stop()

print("\n✅ Teste concluído!")
print("\n🔍 Agora execute:")
print("   python diagnostico_janelas.py")
print("="*80)
