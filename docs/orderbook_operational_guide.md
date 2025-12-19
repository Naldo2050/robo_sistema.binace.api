# Guia de Operação – OrderBook Analyzer & Wrapper

Este guia resume como:

- chamar o `OrderBookAnalyzer`,
- usar o `orderbook_wrapper` no bot,
- monitorar o módulo via métricas Prometheus, logs estruturados e Circuit Breaker.

---

## 1. Uso direto do OrderBookAnalyzer

```python
import asyncio
from orderbook_analyzer import OrderBookAnalyzer

async def main():
    async with OrderBookAnalyzer(symbol="BTCUSDT") as oba:
        event = await oba.analyze()
        if event["is_valid"]:
            ob = event["orderbook_data"]
            print("Bid depth:", ob["bid_depth_usd"])
            print("Ask depth:", ob["ask_depth_usd"])
        else:
            print("Erro de orderbook:", event.get("erro"))

if __name__ == "__main__":
    asyncio.run(main())