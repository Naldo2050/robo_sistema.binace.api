import asyncio
import json

import websockets

import config
from market_orchestrator import EnhancedMarketBot, adapt_orchestrator_runtime


async def test_bot():
    print("Criando bot...")

    bot = EnhancedMarketBot(
        stream_url=config.STREAM_URL,
        symbol=config.SYMBOL,
        window_size_minutes=config.WINDOW_SIZE_MINUTES,
        vol_factor_exh=config.VOL_FACTOR_EXH,
        history_size=config.HISTORY_SIZE,
        delta_std_dev_factor=config.DELTA_STD_DEV_FACTOR,
        context_sma_period=config.CONTEXT_SMA_PERIOD,
        liquidity_flow_alert_percentage=config.LIQUIDITY_FLOW_ALERT_PERCENTAGE,
        wall_std_dev_factor=config.WALL_STD_DEV_FACTOR,
    )
    runtime = adapt_orchestrator_runtime(bot)

    print("Inicializando...")
    await bot.initialize()
    print(f"Snapshot inicial: {runtime.snapshot_state()}")

    print("Iniciando run()...")

    # Snapshot comum ajuda o debug sem depender tanto da implementação concreta.
    try:
        await asyncio.wait_for(bot.run(), timeout=10.0)
    except asyncio.TimeoutError:
        print("TIMEOUT! Bot.run() nao esta processando mensagens!")
        print(f"Snapshot do runtime: {runtime.snapshot_state()}")
        print(f"Estado do bot: {dir(bot)}")


if __name__ == "__main__":
    asyncio.run(test_bot())
