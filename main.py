# main.py v2.3.2 - ENTRY POINT ROBUSTO
# -*- coding: utf-8 -*-
"""
Entry point para o Enhanced Market Bot v2.3.2

Correções:
  - Cleanup garantido mesmo em erro
  - Validação de config mais específica
  - Try/finally para recursos
  - Logging melhorado (usa LOG_LEVEL do config)
  - _validate_required_config para validar parâmetros obrigatórios (existência e valor básico)
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
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

# Carrega variáveis de ambiente do .env
load_dotenv()

import config
from market_orchestrator import EnhancedMarketBot


def _validate_required_config() -> None:
    """
    Valida a presença e os valores básicos dos parâmetros obrigatórios.

    Regras:
      - O atributo precisa existir em config
      - Não pode ser None
      - Se for string, não pode ser vazia/apenas espaços

    Lança ValueError em caso de problema.
    """
    required_params = [
        "STREAM_URL",
        "SYMBOL",
        "WINDOW_SIZE_MINUTES",
        "VOL_FACTOR_EXH",
        "HISTORY_SIZE",
        "DELTA_STD_DEV_FACTOR",
        "CONTEXT_SMA_PERIOD",
        "LIQUIDITY_FLOW_ALERT_PERCENTAGE",
        "WALL_STD_DEV_FACTOR",
    ]

    missing = []
    invalid_values = []

    for param in required_params:
        # Falta de atributo
        if not hasattr(config, param):
            missing.append(param)
            continue

        value = getattr(config, param)

        # Valor inválido básico
        if value is None:
            invalid_values.append(f"{param}=None")
        elif isinstance(value, str) and not value.strip():
            invalid_values.append(f"{param} vazio")

    messages = []
    if missing:
        messages.append(f"parâmetros faltando em config: {', '.join(missing)}")
    if invalid_values:
        messages.append(
            f"parâmetros com valores inválidos: {', '.join(invalid_values)}"
        )

    if messages:
        # Vai ser capturado pelo except ValueError no main()
        raise ValueError("❌ " + " | ".join(messages))


def main() -> int:
    """
    Entry point principal com cleanup garantido.
    
    Returns:
        0 para sucesso, 1 para erro
    """
    # Usa LOG_LEVEL definido no config, se existir
    log_level_name = getattr(config, "LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    bot = None  # ✅ Inicializa fora do try
    
    try:
        # ✅ Validação mais específica (não captura AttributeError genérico)
        validate = getattr(config, "validate_config", None)
        if callable(validate):
            try:
                validate()
                logging.info("✅ Configuração validada com sucesso")
            except Exception as e:
                logging.warning(f"⚠️ Erro na validação de config: {e}")
                # Continua mesmo com erro de validação (pode ser não-crítico)

        # ✅ Validação rigorosa de parâmetros obrigatórios usados no construtor
        _validate_required_config()

        logging.info(f"🚀 Iniciando bot para {config.SYMBOL}...")

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
        logging.info("⏹️ Execução interrompida pelo usuário (Ctrl+C).")
        return 0

    except ValueError as e:
        # Erros de configuração (inclui os do validate_config e os dos required_params)
        logging.critical(f"❌ Erro de configuração: {e}")
        return 1

    except Exception as e:
        logging.critical(
            "❌ Erro crítico na inicialização/execução do bot: %s",
            e,
            exc_info=True,
        )
        return 1

    finally:
        # ✅ GARANTE CLEANUP MESMO EM CASO DE ERRO
        if bot is not None and hasattr(bot, "_cleanup_handler"):
            try:
                logging.info("🧹 Iniciando cleanup de recursos...")
                bot._cleanup_handler()
                logging.info("✅ Cleanup concluído")
            except Exception as cleanup_err:
                logging.error(
                    f"❌ Erro no cleanup: {cleanup_err}",
                    exc_info=True
                )


if __name__ == "__main__":
    sys.exit(main())