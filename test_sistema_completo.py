# test_sistema_completo.py
"""Teste do sistema completo após correções."""
import logging
from orderbook_analyzer import OrderBookAnalyzer
from ai_analyzer_qwen import AIAnalyzer
from time_manager import TimeManager

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)

print("\n" + "="*80)
print("🧪 TESTE COMPLETO DO SISTEMA")
print("="*80 + "\n")

# 1. TimeManager
tm = TimeManager()
print("✅ TimeManager criado\n")

# 2. OrderBookAnalyzer
oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
print("✅ OrderBookAnalyzer criado\n")

# 3. AIAnalyzer (SEM time_manager)
ai = AIAnalyzer()
print("✅ AIAnalyzer criado\n")

# 4. Obtém orderbook
print("📡 Buscando orderbook...\n")
event = oba.analyze()

print(f"✅ Orderbook obtido:")
print(f"   is_valid: {event['is_valid']}")
print(f"   bid: ${event['orderbook_data']['bid_depth_usd']:,.2f}")
print(f"   ask: ${event['orderbook_data']['ask_depth_usd']:,.2f}\n")

# 5. Extrai com AI
print("🧠 Extraindo com AIAnalyzer...\n")
ob_data = ai._extract_orderbook_data(event)

if ob_data:
    print(f"✅ Extração OK:")
    print(f"   bid: ${ob_data['bid_usd']:,.2f}")
    print(f"   ask: ${ob_data['ask_usd']:,.2f}\n")
else:
    print("❌ Extração falhou!\n")

# 6. Testa prompt builder (se tiver)
print("📝 Testando build_prompt...\n")

try:
    if hasattr(ai, 'build_prompt'):
        prompt = ai.build_prompt(
            tipo_evento="OrderBook",
            event_data=event,
            timeframe_data=None,
            event_memory=[]
        )
        
        if prompt and len(prompt) > 100:
            print(f"✅ Prompt gerado ({len(prompt)} chars)")
            print(f"\nPrimeiros 300 chars:\n{prompt[:300]}...\n")
        else:
            print(f"⚠️  Prompt vazio ou muito curto: {len(prompt) if prompt else 0} chars\n")
    else:
        print("⚠️  Método build_prompt não encontrado\n")
        
except Exception as e:
    print(f"❌ Erro ao gerar prompt: {e}\n")

print("="*80)
print("✅ TESTE COMPLETO CONCLUÍDO")
print("="*80 + "\n")