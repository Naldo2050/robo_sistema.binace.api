# main.py v2.3.1 - ENTRY POINT SIMPLIFICADO
# -*- coding: utf-8 -*-
"""
Entry point para o Enhanced Market Bot v2.3.1

Toda a lógica pesada de orquestração está em:
- market_orchestrator.EnhancedMarketBot
"""

import sys
import io
import logging

from dotenv import load_dotenv

# 🔧 FORÇAR UTF-8 NO WINDOWS (DEVE SER A PRIMEIRA COISA)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        # Fallback para Python < 3.7
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Carrega variáveis de ambiente do .env
load_dotenv()

import config
from market_orchestrator import EnhancedMarketBot


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Validação de configuração (se existir no config.py)
        try:
            config.validate_config()
        except AttributeError:
            # Se a função não existir, apenas segue
            pass

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

        bot.run()
        return 0

    except KeyboardInterrupt:
        logging.info("Execução interrompida pelo usuário (Ctrl+C).")
        return 0

    except Exception as e:
        logging.critical(
            "❌ Erro crítico na inicialização do bot: %s",
            e,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())