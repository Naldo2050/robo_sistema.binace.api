# scripts/diagnostics/run_o1_shadow_observation.py
# -*- coding: utf-8 -*-
"""
RUNNER OFICIAL DE OBSERVAÇÃO SHADOW — FASE O1.
Ambiente de observação contínua estritamente PASSIVO e em MODO SEGURO:
- ZERO ordens financeiras (nenhum módulo de envio de ordem ativo ou existente).
- Coleta e grava simultaneamente em SQLite WAL (dados/trading_bot.db):
  1. EnhancedMarketBot (1m streams, Baseline, Session VWAP UTC, Market Structure Schema 1.1.0).
  2. PositioningShadowCollector (polling a cada 5m de Crypto COT da Binance USDM).
  3. MacroUpdateService e PipelineHealthExporter (/health e /metrics).

Uso:
  # Smoke run de 3 minutos:
  python scripts/diagnostics/run_o1_shadow_observation.py --duration-min 3

  # Observação contínua longa:
  python scripts/diagnostics/run_o1_shadow_observation.py
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import signal
import sys
import time
from typing import Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 1. THROTTLER & WINDOWS FIXES (deve rodar antes de outros imports)
from common.ai_throttler import init_throttler

init_throttler(
    min_interval=60,
    hard_min_interval=30,
    daily_token_budget=85_000,
    max_calls_per_hour=10,
)


def _fix_encoding_windows() -> None:
    if sys.platform != "win32":
        return
    try:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None and not sys.stdout.closed:
            reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        reconfigure = getattr(sys.stderr, "reconfigure", None)
        if reconfigure is not None and not sys.stderr.closed:
            reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


_fix_encoding_windows()

# WMI patch para evitar travamentos no Windows
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

        def _safe_wmi_query(_table, *keys):
            return (str(_fake_wmi.get(k, "")) for k in keys)

        _platform._wmi_query = _safe_wmi_query
        _platform._wmi_patched = True
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import config
from market_orchestrator import EnhancedMarketBot
from monitoring.heartbeat_manager import HeartbeatManager
from scripts.analytics.positioning_shadow_collector import run_positioning_shadow_daemon
from scripts.diagnostics.verify_safe_mode import verify_runtime_safe_mode

MARKER = "[SHADOW_O1]"
META_FILE = "logs/shadow_o1_run_meta.json"


def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_level_name = getattr(config, "LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    if hasattr(sys.stdout, "buffer"):
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    else:
        utf8_stream = sys.stdout

    root = logging.getLogger()
    root.setLevel(log_level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    ch = logging.StreamHandler(stream=utf8_stream)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    from logging.handlers import RotatingFileHandler
    ih = RotatingFileHandler(
        "logs/issues.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    ih.setLevel(logging.WARNING)
    ih.setFormatter(fmt)
    root.addHandler(ih)

    rh = RotatingFileHandler(
        "logs/run.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    rh.setLevel(log_level)
    rh.setFormatter(fmt)
    root.addHandler(rh)

    return logging.getLogger("ShadowO1Runner")


async def run_o1_observation(duration_minutes: Optional[float] = None) -> int:
    logger = _setup_logging()
    dur_desc = f"{duration_minutes:.1f} minutos" if duration_minutes else "CONTÍNUA (INDEFINIDA)"
    logger.info(f"{MARKER} ========================================================")
    logger.info(f"{MARKER} INICIANDO OBSERVACAO SHADOW FASE O1 — DURAÇÃO: {dur_desc}")
    logger.info(f"{MARKER} ========================================================")

    # 1. Auditoria Estrita de Modo Seguro em Runtime
    safe_proof = verify_runtime_safe_mode()
    if not safe_proof["is_safe_for_o1"]:
        logger.critical(f"{MARKER} FALHA NO MODO SEGURO! ABORTANDO INÍCIO DA FASE O1!")
        return 1

    logger.info(f"{MARKER} ✅ Modo seguro auditado com sucesso (ZERO ordens financeiras)")

    stop_event = asyncio.Event()

    # 2. Inicia Heartbeat Manager
    heartbeat = HeartbeatManager(
        "main",
        warning_threshold=60,
        critical_threshold=120,
        auto_beat_interval=30,
    )
    await heartbeat.start()

    # 3. Inicia servidor /metrics e /health
    port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    try:
        from monitoring.pipeline_health import serve_metrics_and_health
        serve_metrics_and_health(port)
        logger.info(f"{MARKER} Servidor Prometheus/Health ativo na porta {port}")
    except Exception as e:
        logger.warning(f"{MARKER} Pipeline health server não iniciado: {e}")

    # 4. Inicia MacroUpdateService
    try:
        from fetchers.macro_update_service import start_macro_service
        await start_macro_service()
        logger.info(f"{MARKER} MacroUpdateService iniciado em background")
    except Exception as e:
        logger.warning(f"{MARKER} MacroUpdateService aviso: {e}")

    # 5. Inicia Daemon de Posicionamento Binance USDM (5m interval)
    pos_task = asyncio.create_task(
        run_positioning_shadow_daemon(
            interval_seconds=300.0,
            symbol=config.SYMBOL,
            stop_event=stop_event,
        )
    )

    # 6. Instancia o EnhancedMarketBot
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

    if hasattr(bot, "health_monitor"):
        heartbeat.health_monitor = bot.health_monitor

    try:
        from monitoring.pipeline_health import (
            attach_message_age_provider,
            attach_window_age_provider,
            attach_ws_connected_provider,
            start_pipeline_health,
        )
        from market_orchestrator.windows import window_processor as _wp

        start_pipeline_health(bot.health_monitor)
        attach_ws_connected_provider(lambda: bool(bot.connection_manager.is_connected))
        attach_message_age_provider(
            lambda: (time.time() - bot.connection_manager.last_message_time)
            if bot.connection_manager.last_message_time > 0 else 9999
        )
        attach_window_age_provider(
            lambda: (time.time() - _wp.last_window_processed_ts)
            if _wp.last_window_processed_ts > 0 else 9999
        )
    except Exception as e:
        logger.warning(f"{MARKER} Pipeline health monitor integração: {e}")

    # 7. Agendamento de término (caso duration_minutes seja passado para smoke run)
    timer_task = None
    if duration_minutes and duration_minutes > 0:
        async def _timer_shutdown():
            await asyncio.sleep(duration_minutes * 60)
            logger.info(f"{MARKER} Tempo limite de smoke run ({duration_minutes}m) atingido. Encerrando...")
            stop_event.set()
            await bot.shutdown()

        timer_task = asyncio.create_task(_timer_shutdown())

    t0 = time.time()
    try:
        await bot.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info(f"{MARKER} Interrupção solicitada pelo usuário.")
    finally:
        stop_event.set()
        if timer_task and not timer_task.done():
            timer_task.cancel()
        pos_task.cancel()
        try:
            await pos_task
        except (asyncio.CancelledError, Exception):
            pass

        try:
            from fetchers.macro_update_service import stop_macro_service
            await stop_macro_service()
        except Exception:
            pass

        await heartbeat.stop()

    elapsed = time.time() - t0
    meta = {
        "phase": "O1_PRODUCTION_SHADOW_OBSERVATION",
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(elapsed, 1),
        "duration_minutes": round(elapsed / 60.0, 1),
        "symbol": config.SYMBOL,
        "operational_mode": "SHADOW_OBSERVATION",
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"{MARKER} Execução encerrada graciosamente após {elapsed/60.0:.1f} minutos.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Runner de Observação Shadow Fase O1")
    parser.add_argument(
        "--duration-min",
        type=float,
        default=None,
        help="Duração em minutos para smoke run (default: contínuo/indefinido)",
    )
    args = parser.parse_args()
    return asyncio.run(run_o1_observation(duration_minutes=args.duration_min))


if __name__ == "__main__":
    sys.exit(main())
