# -*- coding: utf-8 -*-
"""
RUNNER DE OBSERVACAO SHADOW (NENHUMA EXECUCAO FINANCEIRA).

Boota o EnhancedMarketBot com o MESMO bootstrap do main.py (throttler,
encoding, WMI patch, metrics server, macro service), mas com:

  - DUracao limitada (SHADOW_DURATION_MIN, default 75min);
  - Encerramento gracioso ao final (bot.shutdown());
  - SEM reset forcado de CVD (diferente de run_production_observation.py);
  - Nenhuma alteracao em codigo de producao envolvida.

O proprio sistema NAO executa ordens financeiras (nao ha codigo de
order placement); esta observacao documenta o comportamento shadow.

Meta de captura: >= 60 janelas de 1min BTCUSDT consecutivas com
ANALYSIS_TRIGGER, AI_ANALYSIS, orderbook, fluxo, macro/intermarket,
quality e logs.

Uso:
    python scripts/diagnostics/run_shadow_observation.py
    SHADOW_DURATION_MIN=90 python scripts/diagnostics/run_shadow_observation.py
"""

import sys
import os
import io
import time
import asyncio
import json
import logging

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# BOOTSTRAP IDENTICO AO main.py (deve rodar antes de qualquer outro import)
# ---------------------------------------------------------------------------
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
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


_fix_encoding_windows()

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

import config
from market_orchestrator import EnhancedMarketBot
from monitoring.heartbeat_manager import HeartbeatManager

DURATION_MIN = float(os.getenv("SHADOW_DURATION_MIN", "75"))
MARKER = "[SHADOW]"
META_FILE = "logs/shadow_run_meta.json"

PROMETHEUS_KEYS = [
    "orchestrator_trades_timestamp_corrected_total",
    "flow_analyzer_cvd",
    "flow_analyzer_trades_total",
    "flow_analyzer_trades_invalid_total",
    "pipeline_health",
    "window_processor",
    "event_saver_events_written_total",
]


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
    ih = RotatingFileHandler("logs/issues.log", maxBytes=5 * 1024 * 1024,
                             backupCount=3, encoding="utf-8")
    ih.setLevel(logging.WARNING)
    ih.setFormatter(fmt)
    root.addHandler(ih)

    rh = RotatingFileHandler("logs/run.log", maxBytes=10 * 1024 * 1024,
                             backupCount=5, encoding="utf-8")
    rh.setLevel(log_level)
    rh.setFormatter(fmt)
    root.addHandler(rh)
    return logging.getLogger(__name__)


async def _scheduled_shutdown(bot, delay_min: float, stop_evt: asyncio.Event) -> None:
    logger = logging.getLogger("shadow.end")
    await asyncio.sleep(delay_min * 60)
    if stop_evt.is_set():
        return
    logger.info(f"{MARKER} DURACAO DE OBSERVACAO ESGOTADA ({delay_min}min). "
                f"Encerrando graciosamente...")
    stop_evt.set()
    try:
        await bot.shutdown()
    except Exception as e:
        logger.warning(f"{MARKER} erro no shutdown programado: {e}")


def _write_meta(t0: float, duration_min: float, summary: dict) -> None:
    meta = {
        "purpose": "shadow observation - NO financial execution (bot has no order placement code)",
        "start_unix": t0,
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planned_duration_min": duration_min,
        "summary": summary,
    }
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.getLogger("shadow").warning(f"{MARKER} falha ao gravar meta: {e}")


async def main() -> int:
    logger = _setup_logging()
    logger.info(f"{MARKER} === INICIO DA OBSERVACAO SHADOW === duracao={DURATION_MIN}min")

    heartbeat = HeartbeatManager(
        "main",
        warning_threshold=60,
        critical_threshold=120,
        auto_beat_interval=30,
    )
    await heartbeat.start()

    port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    try:
        from monitoring.pipeline_health import serve_metrics_and_health
        serve_metrics_and_health(port)
        logger.info(f"{MARKER} servidor /metrics + /health na porta {port}")
    except Exception as e:
        logger.warning(f"{MARKER} sem metrics server: {e}")

    try:
        from fetchers.macro_update_service import start_macro_service
        await start_macro_service()
        logger.info(f"{MARKER} MacroUpdateService iniciado")
    except Exception as e:
        logger.warning(f"{MARKER} macro service nao iniciou: {e}")

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

    if hasattr(bot, 'health_monitor'):
        heartbeat.health_monitor = bot.health_monitor

    try:
        from monitoring.pipeline_health import (
            start_pipeline_health,
            attach_ws_connected_provider,
            attach_message_age_provider,
            attach_window_age_provider,
            age_seconds,
        )
        from market_orchestrator.windows import window_processor as _wp
        start_pipeline_health(bot.health_monitor)
        attach_ws_connected_provider(lambda: bool(bot.connection_manager.is_connected))
        attach_message_age_provider(
            lambda: age_seconds(bot.connection_manager.last_message_time)
        )
        attach_window_age_provider(
            lambda: age_seconds(_wp.last_window_processed_ts)
        )
    except Exception as e:
        logger.warning(f"{MARKER} pipeline health exporter: {e}")

    stop_evt = asyncio.Event()
    end_task = asyncio.create_task(_scheduled_shutdown(bot, DURATION_MIN, stop_evt))

    t0 = time.time()
    try:
        await bot.run()
    finally:
        stop_evt.set()
        end_task.cancel()
        try:
            await end_task
        except (asyncio.CancelledError, Exception):
            pass

    elapsed = time.time() - t0
    summary = {
        "duracao_seg": round(elapsed, 1),
        "duracao_min": round(elapsed / 60, 1),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preco_final": None,
    }
    try:
        if bot.flow_analyzer is not None:
            summary["preco_final"] = getattr(bot.flow_analyzer, "last_price", None)
    except Exception:
        pass
    _write_meta(t0, DURATION_MIN, summary)
    logger.info(f"{MARKER} === FIM DA OBSERVACAO SHADOW === {json.dumps(summary, ensure_ascii=False)}")

    try:
        from fetchers.macro_update_service import stop_macro_service
        await stop_macro_service()
    except Exception:
        pass
    await heartbeat.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))