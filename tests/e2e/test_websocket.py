import asyncio
import websockets
import json
from datetime import datetime
import pytest

pytestmark = pytest.mark.network

@pytest.mark.network
async def test_binance_stream():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    
    print(f"🔌 Conectando em {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ CONECTADO! Aguardando trades...\n")
            
            for i in range(20):  # Recebe 20 trades para teste
                message = await websocket.recv()
                data = json.loads(message)
                
                timestamp = datetime.fromtimestamp(data['T'] / 1000)
                side = "🔴 VENDA" if data['m'] else "🟢 COMPRA"
                
                print(f"Trade #{i+1:2d} | {timestamp.strftime('%H:%M:%S')} | "
                      f"{side} | Preço: ${float(data['p']):,.2f} | "
                      f"Qtd: {float(data['q']):.4f} BTC")
            
            print("\n✅ Teste concluído! Binance WebSocket está funcionando.")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")

if __name__ == "__main__":
    asyncio.run(test_binance_stream())