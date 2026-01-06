import asyncio
import json
import aiohttp

# Spot (Binance Spot): wss://stream.binance.com:9443/ws/<stream>
WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

async def main():
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(WS_URL, heartbeat=20) as ws:
            print("CONNECTED:", WS_URL)
            count = 0
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print("TRADE:", data.get("p"), "qty:", data.get("q"), "ts:", data.get("T"))
                    count += 1
                    if count >= 5:
                        return
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    print("WS CLOSED/ERROR:", msg)
                    return

if __name__ == "__main__":
    asyncio.run(main())