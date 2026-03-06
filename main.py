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
import os
import logging
import asyncio
import traceback

from dotenv import load_dotenv

# 🔧 INSTRUMENTAÇÃO PARA DEBUG DE asyncio.create_task (opcional)
if os.getenv("DEBUG_CREATE_TASK") == "1":
    _real_create_task = asyncio.create_task

    def traced_create_task(coro, *args, **kwargs):
        print("\n[DEBUG] asyncio.create_task chamado. Stack:")
        print("".join(traceback.format_stack(limit=25)))
        return _real_create_task(coro, *args, **kwargs)

    asyncio.create_task = traced_create_task

# 🔧 FORÇAR UTF-8 NO WINDOWS (DEVE SER A PRIMEIRA COISA)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
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

# ---------------------------------------------------------------------------
# FIX: platform._wmi_query() pode travar indefinidamente no Windows.
# Vários módulos (prometheus_client, oci) chamam platform.system(),
# platform.platform() ou platform.processor() no import-time, e todos
# passam por _wmi_query() para obter dados do WMI.
# Correção: substituir _wmi_query por versão que usa os.environ/sys
# (instantâneo, sem WMI).  DEVE rodar ANTES de qualquer outro import.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import platform as _platform
    try:
        _wv = sys.getwindowsversion()
        _fake_wmi = {
            "Version": f"{_wv.major}.{_wv.minor}.{_wv.build}",
            "ProductType": str(_wv.product_type),
            "Caption": f"Microsoft Windows {_wv.major}",
            "CSName": os.environ.get("COMPUTERNAME", ""),
            "Architecture": {"AMD64": "9", "x86": "0", "ARM64": "12"}.get(
                os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"), "9"
            ),
            "Manufacturer": (
                os.environ.get("PROCESSOR_IDENTIFIER", "").split(",")[-1].strip()
                if os.environ.get("PROCESSOR_IDENTIFIER") else ""
            ),
        }

        def _safe_wmi_query(_table, *keys):  # type: ignore[misc]
            return (str(_fake_wmi.get(k, "")) for k in keys)

        _platform._wmi_query = _safe_wmi_query  # type: ignore[attr-defined]
        _platform._wmi_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass

import config
from market_orchestrator import EnhancedMarketBot
from utils import HeartbeatManager


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


async def _heartbeat_during_run(heartbeat: HeartbeatManager):
    """
    Task background que faz heartbeats regulares durante a execução do bot.
    Isso garante que o módulo main nunca fique sem heartbeat por muito tempo,
    mesmo durante operações longas ou espera de WebSocket.
    """
    try:
        while True:
            await asyncio.sleep(30)  # Heartbeat a cada 30s
            if heartbeat._running:
                heartbeat.beat()
                logging.debug(f"💓 Heartbeat durante execução - silence={heartbeat.get_silence_seconds():.1f}s")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logging.error(f"Erro na task de heartbeat: {e}")


async def main() -> int:
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
    logging.info(f"📊 Nível de log configurado: {log_level_name}")

    logger = logging.getLogger(__name__)

    # Inicializar heartbeat manager
    heartbeat = HeartbeatManager(
        "main",
        warning_threshold=60,
        critical_threshold=120,
        auto_beat_interval=30  # Heartbeat automático a cada 30s
    )

    bot = None  # ✅ Inicializa fora do try

    try:
        # ✅ Validação mais específica (não captura AttributeError genérico)
        validate = getattr(config, "validate_config", None)
        if callable(validate):
            try:
                validate()
                logging.info("✅ Configuração validada com sucesso")
            except ValueError as e:
                # ValueError indica erro crítico de configuração - deve parar
                raise
            except Exception as e:
                logging.warning(f"⚠️ Erro inesperado na validação de config: {e}")
                # Continua apenas para exceções não-críticas

        # ✅ Validação rigorosa de parâmetros obrigatórios usados no construtor
        _validate_required_config()

        # Iniciar heartbeat manager
        await heartbeat.start()

        logger.info(f"🚀 Iniciando bot para {config.SYMBOL}...")

        # ✅ PATCH 2.6: Iniciar servidor Prometheus para métricas
        try:
            from prometheus_client import start_http_server
            import os

            # Porta configurável via env var (default 8000)
            prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
            start_http_server(prometheus_port)
            logging.info(f"📊 Servidor Prometheus iniciado na porta {prometheus_port} (/metrics)")
        except ImportError:
            logging.warning("⚠️ prometheus_client não disponível - métricas não serão exportadas")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao iniciar servidor Prometheus: {e}")

        # ✅ PATCH 2.7: Iniciar serviço de atualização de macro data
        try:
            from src.services.macro_update_service import start_macro_service
            await start_macro_service()
            logging.info("📊 MacroUpdateService iniciado (atualização em background)")
        except ImportError:
            logging.warning("⚠️ macro_update_service não disponível")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao iniciar MacroUpdateService: {e}")

        # 1. Criar o bot (sem inicializar tasks)
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

        # ✅ Integrar HeartbeatManager com HealthMonitor do bot
        if hasattr(bot, 'health_monitor'):
            heartbeat.health_monitor = bot.health_monitor
            logging.info("✅ HeartbeatManager integrado com HealthMonitor do bot")

        # 2. ✅ ADICIONAR: Inicializar componentes assíncronos
        await bot.initialize()
        heartbeat.beat()  # Heartbeat após inicialização

        # 3. Iniciar task de heartbeat periódico durante execução do bot
        heartbeat_task = asyncio.create_task(_heartbeat_during_run(heartbeat))

        try:
            # 4. Executar o bot
            await bot.run()
            return 0
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info("⚠️ Interrupção manual detectada")
        if bot is not None:
            await bot.shutdown()
        await heartbeat.stop()
        try:
            from src.services.macro_update_service import stop_macro_service
            await stop_macro_service()
            logging.info("🛑 MacroUpdateService parado")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao parar MacroUpdateService: {e}")
        return 0

    except ValueError as e:
        # Erros de configuração (inclui os do validate_config e os dos required_params)
        logger.critical(f"❌ Erro de configuração: {e}")
        if bot is not None:
            await bot.shutdown()
        await heartbeat.stop()
        return 1

    except Exception as e:
        logger.critical(
            "❌ Erro crítico na inicialização/execução do bot: %s",
            e,
            exc_info=True,
        )
        if bot is not None:
            await bot.shutdown()
        await heartbeat.stop()
        return 1


if __name__ == "__main__":
    # ✅ Executar a função assíncrona corretamente
    exit_code = asyncio.run(main())
    sys.exit(exit_code)