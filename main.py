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
import os
import io
import time

# [THROTTLE] Fonte única de verdade: main.py inicializa o singleton ANTES
# de qualquer import de ai_runner/analyzer_qwen (que apenas consomem).
from common.ai_throttler import init_throttler

init_throttler(
    min_interval=60,
    hard_min_interval=30,
    daily_token_budget=85_000,
    max_calls_per_hour=10,
)

# ══════════════════════════════════════════════════════════════════
# FIX DE ENCODING PARA WINDOWS - VERSÃO SEGURA
# ══════════════════════════════════════════════════════════════════
def _fix_encoding_windows() -> None:
    """
    Corrige encoding no Windows sem fechar streams.
    Evita 'I/O operation on closed file'.
    """
    if sys.platform != "win32":
        return

    try:
        # Só reconfigura se o stream estiver aberto e for um TextIOWrapper
        # Usa getattr para evitar erro de tipo estático do Pylance
        reconfigure = getattr(sys.stdout, 'reconfigure', None)
        if reconfigure is not None and not sys.stdout.closed:
            reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    try:
        reconfigure = getattr(sys.stderr, 'reconfigure', None)
        if reconfigure is not None and not sys.stderr.closed:
            reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # Variável de ambiente para subprocessos
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

_fix_encoding_windows()

import logging
import asyncio
import signal
import traceback

from config.env_policy import maybe_load_dotenv

# 🔧 INSTRUMENTAÇÃO PARA DEBUG DE asyncio.create_task (opcional)
if os.getenv("DEBUG_CREATE_TASK") == "1":
    _real_create_task = asyncio.create_task

    def traced_create_task(coro, *args, **kwargs):
        print("\n[DEBUG] asyncio.create_task chamado. Stack:")
        print("".join(traceback.format_stack(limit=25)))
        return _real_create_task(coro, *args, **kwargs)

    asyncio.create_task = traced_create_task

# PF-D4: mesma política de config/settings.py — LOAD_DOTENV=0 ou
# OBSERVATION_MODE=1 => NÃO carrega (observation nunca carrega).
maybe_load_dotenv()

# Silenciar logs de nível HTTP (httpx/httpcore aparecem a cada chamada Groq)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
from monitoring.heartbeat_manager import HeartbeatManager


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

    # Garantir pasta de logs
    os.makedirs("logs", exist_ok=True)

    # Console UTF-8 -- evita UnicodeEncodeError com emojis no Windows
    if hasattr(sys.stdout, "buffer"):
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    else:
        utf8_stream = sys.stdout

    console_handler = logging.StreamHandler(stream=utf8_stream)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    from logging.handlers import RotatingFileHandler

    # Sanear issues.log legado com BOM UTF-16 (ff fe): o RotatingFileHandler
    # anexa em utf-8, mas um arquivo que começa com BOM UTF-16 quebra a leitura.
    # Renomeia para backup antes de abrir, para o handler recriar em utf-8 puro.
    issues_log_path = os.path.join("logs", "issues.log")
    if os.path.exists(issues_log_path):
        try:
            with open(issues_log_path, "rb") as _f:
                _head = _f.read(2)
            if _head == b"\xff\xfe":
                _legacy_path = f"{issues_log_path}.legacy-{int(time.time())}"
                os.replace(issues_log_path, _legacy_path)
                logging.warning(
                    "issues.log com BOM UTF-16 legado renomeado para %s",
                    _legacy_path,
                )
        except OSError as _e:
            logging.warning("Não foi possível sanear issues.log: %s", _e)

    # Arquivo de problemas: WARNING + ERROR + CRITICAL
    issues_handler = RotatingFileHandler(
        "logs/issues.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    issues_handler.setLevel(logging.WARNING)
    issues_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Limpar handlers antigos
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(issues_handler)

    # Log completo de execução (INFO+): recebe a saída que run.log deveria ter.
    run_handler = RotatingFileHandler(
        "logs/run.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    run_handler.setLevel(log_level)
    run_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )
    root_logger.addHandler(run_handler)

    logging.info(f"📊 Nível de log configurado: {log_level_name}")
    logging.info("🚨 Logs de problemas em: logs/issues.log")

    logger = logging.getLogger(__name__)

    # HeartbeatManager "main": o auto-beat representa APENAS "processo vivo".
    # NAO e usado nas decisoes de health/degraded do container
    # (monitoring/pipeline_health ignora "main"); o progresso real e medido
    # pelos heartbeats de estagio (ws, ai, trade_ingestion, trade_buffer,
    # window_processor, orderbook, event_saver).
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

        # PF-D3: observation guard — aborta startup ANTES do bot e de qualquer
        # side effect de rede quando OBSERVATION_MODE=1 e o processo estiver
        # inseguro (credenciais/IA/hybrid/execução). No-op com modo desligado.
        from config.env_policy import assert_observation_safe
        import config.settings as _settings_for_guard
        assert_observation_safe(_settings_for_guard)

        # Iniciar heartbeat manager
        await heartbeat.start()

        logger.info(f"🚀 Iniciando bot para {config.SYMBOL}...")

        # ✅ PATCH 2.6: Iniciar servidor HTTP (métricas Prometheus + /health)
        try:
            from monitoring.pipeline_health import serve_metrics_and_health

            # Porta configurável via env var (default 8000)
            prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
            serve_metrics_and_health(prometheus_port)
            logging.info(f"📊 Servidor HTTP iniciado na porta {prometheus_port} (/metrics e /health)")
        except ImportError:
            logging.warning("⚠️ prometheus_client não disponível - métricas não serão exportadas")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao iniciar servidor HTTP: {e}")

        # ✅ PATCH 2.7: Iniciar serviço de atualização de macro data
        try:
            from fetchers.macro_update_service import start_macro_service
            await start_macro_service()
            logging.info("📊 MacroUpdateService iniciado (atualização em background)")
        except ImportError:
            logging.warning("⚠️ macro_update_service não disponível")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao iniciar MacroUpdateService: {e}")

        # ✅ Suporte a flags CLI (ex: --dump-raw-trades para coleta contínua do Item 8)
        import argparse
        cli_parser = argparse.ArgumentParser(description="Enhanced Market Bot v2.3.2", add_help=False)
        cli_parser.add_argument(
            "--dump-raw-trades",
            nargs="?",
            const="dados/trades_collect_2h.jsonl",
            default=os.getenv("DUMP_RAW_TRADES", None),
            help="Caminho do arquivo para dump contínuo de trades brutos em JSONL",
        )
        cli_parser.add_argument(
            "--duration-seconds",
            type=int,
            default=int(os.getenv("BOT_DURATION_SECONDS", 0)),
            help="Duração máxima em segundos antes do shutdown automático gracioso (0 = infinito)",
        )
        cli_args, _ = cli_parser.parse_known_args()

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
            dump_raw_trades=cli_args.dump_raw_trades,
        )

        # ✅ Integrar HeartbeatManager com HealthMonitor do bot
        if hasattr(bot, 'health_monitor'):
            heartbeat.health_monitor = bot.health_monitor
            logging.info("✅ HeartbeatManager integrado com HealthMonitor do bot")

        # ✅ Health check do container: expor o HealthMonitor via /health e /metrics
        try:
            from monitoring.pipeline_health import (
                age_seconds,
                attach_message_age_provider,
                attach_window_age_provider,
                attach_ws_connected_provider,
                start_pipeline_health,
            )
            from market_orchestrator.windows import window_processor as _window_processor

            start_pipeline_health(bot.health_monitor)
            attach_ws_connected_provider(
                lambda: bool(bot.connection_manager.is_connected)
            )
            attach_message_age_provider(
                lambda: age_seconds(bot.connection_manager.last_message_time)
            )
            attach_window_age_provider(
                lambda: age_seconds(_window_processor.last_window_processed_ts)
            )
            logging.info("✅ Pipeline health exporter iniciado (gauges + /health)")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao iniciar pipeline health exporter: {e}")

        # NOTA: bot.run() já chama self.initialize() internamente.
        # Não chamar bot.initialize() aqui para evitar inicialização dupla.

        # 3. Gerenciamento cooperativo de sinais (SIGINT e SIGTERM)
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def _on_signal(sig_name: str) -> None:
            logger.info(f"🛑 Sinal recebido: {sig_name}. Acionando shutdown cooperativo gracioso...")
            shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_signal, sig.name)
            except (NotImplementedError, AttributeError, RuntimeError):
                try:
                    signal.signal(
                        sig,
                        lambda s, f, n=sig.name: loop.call_soon_threadsafe(_on_signal, n),
                    )
                except Exception as e_sig:
                    logger.debug(f"Não foi possível registrar signal handler síncrono para {sig}: {e_sig}")

        # Iniciar task de heartbeat periódico durante execução do bot
        heartbeat_task = asyncio.create_task(_heartbeat_during_run(heartbeat))

        # 4. Executar o bot cooperativamente
        bot_task = asyncio.create_task(bot.run())
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        wait_tasks = [bot_task, shutdown_waiter]

        timeout_task = None
        if cli_args.duration_seconds and cli_args.duration_seconds > 0:
            logging.info(f"⏱️ Execução com temporizador: {cli_args.duration_seconds} segundos...")
            timeout_task = asyncio.create_task(asyncio.sleep(float(cli_args.duration_seconds)))
            wait_tasks.append(timeout_task)

        try:
            done, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancelar waiters/timers auxiliares
            for t in pending:
                if t is not bot_task:
                    t.cancel()

            if shutdown_event.is_set():
                logger.info("🛑 Parada cooperativa solicitada via sinal (SIGINT/SIGTERM).")
            elif timeout_task and timeout_task in done:
                logger.info(f"⏱️ Tempo limite de {cli_args.duration_seconds}s atingido. Iniciando graceful shutdown...")

            # Acionar shutdown cooperativo no bot
            if bot is not None:
                await bot.shutdown()

            # Aguardar bot_task finalizar
            if not bot_task.done():
                try:
                    await asyncio.wait_for(bot_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    bot_task.cancel()
                    try:
                        await bot_task
                    except (asyncio.CancelledError, Exception):
                        pass

            if bot_task.done() and not bot_task.cancelled() and bot_task.exception():
                exc = bot_task.exception()
                if not isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise exc

            return 0
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            if bot is not None:
                await bot.shutdown()
            await heartbeat.stop()
            try:
                from fetchers.macro_update_service import stop_macro_service
                await stop_macro_service()
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.info("⚠️ Interrupção manual detectada")
        if bot is not None:
            await bot.shutdown()
        await heartbeat.stop()
        try:
            from fetchers.macro_update_service import stop_macro_service
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