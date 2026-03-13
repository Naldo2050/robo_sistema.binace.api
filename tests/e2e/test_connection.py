# test_connection.py
import asyncio
import logging
from market_orchestrator.connection.robust_connection import RobustConnectionManager

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

async def main():
    print("🔌 Criando RobustConnectionManager...")
    
    message_count = 0
    
    def on_message(ws, message):
        nonlocal message_count
        message_count += 1
        if message_count <= 5:
            print(f"📨 Mensagem {message_count}: {message[:100]}...")
    
    def on_open(ws):
        print("✅ WebSocket aberto!")
    
    def on_close(ws, code, reason):
        print(f"❌ WebSocket fechado: {code} - {reason}")
    
    def on_reconnect():
        print("🔄 Reconectando...")
    
    manager = RobustConnectionManager(
        stream_url="wss://fstream.binance.com/ws/btcusdt@aggTrade",
        symbol="BTCUSDT",
        max_reconnect_attempts=5
    )
    
    manager.set_callbacks(
        on_message=on_message,
        on_open=on_open,
        on_close=on_close,
        on_reconnect=on_reconnect
    )
    
    print("🚀 Conectando...")
    
    # Criar task de conexão
    connect_task = asyncio.create_task(manager.connect())
    
    # Aguardar 10 segundos
    print("⏳ Aguardando 10 segundos...")
    await asyncio.sleep(10)
    
    print(f"\n📊 Resultado:")
    print(f"   Mensagens recebidas: {message_count}")
    print(f"   Task finalizada: {connect_task.done()}")
    
    # Desconectar
    print("🛑 Desconectando...")
    manager.should_stop = True
    try:
        await manager.disconnect()
    except Exception as e:
        logger.error(f"Erro em operação async: {e}")
        raise
    
    # Aguardar task finalizar
    if not connect_task.done():
        connect_task.cancel()
        try:
            await connect_task
        except asyncio.CancelledError:
            print("   Task cancelada")

if __name__ == "__main__":
    asyncio.run(main())