# scripts/diagnostics/run_production_observation.py
# -*- coding: utf-8 -*-
"""
RUNNER DE OBSERVAÇÃO EM PRODUÇÃO (validacao - NAO altera codigo de producao).

Boota o EnhancedMarketBot com o MESMO bootstrap do main.py (throttler,
encoding, WMI patch, metrics server, macro service) e:

  - T_RESET (default 25min): FORCA um reset manual de CVD via
    flow_analyzer._reset_metrics() sob lock (mesma semantica do
    _check_reset natural de 4h) para validar o warmup de 300s do
    cvd_div com dados reais.
  - A cada 60s: grava logs/observation_metrics.jsonl com as metricas
    Prometheus relevantes (fetch real do /metrics) e
    logs/observation_stats.jsonl com stats in-process do FlowAnalyzer.
  - DURATION_MIN (default 40min): encerra graciosamente (shutdown).

Uso:
    python scripts/diagnostics/run_production_observation.py
    OBS_DURATION_MIN=20 OBS_RESET_AFTER_MIN=10 python scripts/...
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

DURATION_MIN = float(os.getenv("OBS_DURATION_MIN", "40"))
RESET_AFTER_MIN = float(os.getenv("OBS_RESET_AFTER_MIN", "25"))
STATS_INTERVAL_SEC = 60

MARKER = "[OBSERVACAO]"
METRICS_FILE = "logs/observation_metrics.jsonl"
STATS_FILE = "logs/observation_stats.jsonl"
SUMMARY_FILE = "logs/observation_final_summary.json"

PROMETHEUS_KEYS = [
    "orchestrator_trades_timestamp_corrected_total",
    "flow_analyzer_ooo_total",
    "flow_analyzer_cvd",
    "flow_analyzer_trades_total",
    "flow_analyzer_trades_invalid_total",
    "flow_analyzer_flow_trades_count",
    "flow_analyzer_whale_delta",
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


def _fetch_metrics(port: int) -> dict:
    """Busca /metrics do servidor Prometheus e extrai as chaves relevantes."""
    import urllib.request
    out = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=10) as r:
            body = r.read().decode("utf-8", errors="replace")
        for line in body.splitlines():
            for key in PROMETHEUS_KEYS:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            out[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
    except Exception as e:
        out["_fetch_error"] = str(e)
    return out


async def _stats_loop(bot, port: int, stop_evt: asyncio.Event) -> None:
    logger = logging.getLogger("observation.stats")
    while not stop_evt.is_set():
        try:
            fa = bot.flow_analyzer
            stats = fa.get_stats()
            metrics = await asyncio.to_thread(_fetch_metrics, port)
            record = {
                "ts": time.time(),
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "flow": {
                    "total_trades_processed": stats.get("total_trades_processed"),
                    "invalid_trades": stats.get("invalid_trades"),
                    "cvd": stats.get("cvd"),
                    "out_of_order_count": stats.get("out_of_order_count"),
                    "flow_trades_count": stats.get("flow_trades_count"),
                },
                "orchestrator": {
                    "ooo_trades_count": getattr(bot, "_ooo_trades_count", None),
                    "invalid_trade_count": getattr(bot, "_invalid_trade_count", None),
                    "connected": bool(bot.connection_manager.is_connected),
                },
                "prometheus": metrics,
            }
            with open(STATS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"{MARKER} stats ts={record['ts_utc']} "
                        f"trades={stats.get('total_trades_processed')} "
                        f"cvd={stats.get('cvd')} "
                        f"ooo_fa={stats.get('out_of_order_count')} "
                        f"ooo_orb={getattr(bot, '_ooo_trades_count', 0)} "
                        f"clamps={metrics.get('orchestrator_trades_timestamp_corrected_total', 0)}")
        except Exception as e:
            logger.warning(f"{MARKER} erro no stats loop: {e}")
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=STATS_INTERVAL_SEC)
        except asyncio.TimeoutError:
            continue


async def _scheduled_reset(bot, delay_min: float, stop_evt: asyncio.Event) -> None:
    logger = logging.getLogger("observation.reset")
    await asyncio.sleep(delay_min * 60)
    if stop_evt.is_set():
        return
    fa = bot.flow_analyzer
    try:
        with fa._lock:
            fa._reset_metrics()
        logger.info(f"{MARKER} RESET MANUAL DE CVD FORCADO em {delay_min}min | "
                    f"last_reset_ms={fa.last_reset_ms} price_at_reset={fa._price_at_reset}")
    except Exception as e:
        logger.error(f"{MARKER} falha ao forcar reset: {e}")


async def _scheduled_shutdown(bot, delay_min: float, stop_evt: asyncio.Event) -> None:
    logger = logging.getLogger("observation.end")
    await asyncio.sleep(delay_min * 60)
    if stop_evt.is_set():
        return
    logger.info(f"{MARKER} TEMPO DE OBSERVACAO ESGOTADO ({delay_min}min). "
                f"Encerrando graciosamente...")
    stop_evt.set()
    try:
        await bot.shutdown()
    except Exception as e:
        logger.warning(f"{MARKER} erro no shutdown programado: {e}")


async def main() -> int:
    logger = _setup_logging()
    logger.info(f"{MARKER} === INICIO DA OBSERVACAO EM PRODUCAO === "
                f"duracao={DURATION_MIN}min reset_forcado={RESET_AFTER_MIN}min")

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
        logger.info(f"{MARKER} servidor /metrics+ /health na porta {port}")
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
    stats_task = asyncio.create_task(_stats_loop(bot, port, stop_evt))
    reset_task = asyncio.create_task(_scheduled_reset(bot, RESET_AFTER_MIN, stop_evt))
    end_task = asyncio.create_task(_scheduled_shutdown(bot, DURATION_MIN, stop_evt))

    t0 = time.time()
    try:
        await bot.run()
    finally:
        stop_evt.set()
        for t in (stats_task, reset_task, end_task):
            t.cancel()
        for t in (stats_task, reset_task, end_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    elapsed = time.time() - t0
    fa = bot.flow_analyzer
    stats = fa.get_stats()
    try:
        from market_orchestrator.market_orchestrator import TRADES_CORRECTED_TOTAL
        clamps = None
        if TRADES_CORRECTED_TOTAL is not None:
            for m in TRADES_CORRECTED_TOTAL.collect():
                for s in m.samples:
                    if s.name.endswith("_created"):
                        continue
                    clamps = s.value
    except Exception:
        clamps = None

    summary = {
        "duracao_seg": round(elapsed, 1),
        "duracao_min": round(elapsed / 60, 1),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_trades_processed": stats.get("total_trades_processed"),
        "invalid_trades": stats.get("invalid_trades"),
        "cvd_final": stats.get("cvd"),
        "out_of_order_fa": stats.get("out_of_order_count"),
        "out_of_order_orchestrator": getattr(bot, "_ooo_trades_count", None),
        "timestamp_corrected_total": clamps,
        "flow_trades_count": stats.get("flow_trades_count"),
        "last_reset_ms": fa.last_reset_ms,
        "price_at_reset": fa._price_at_reset,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"{MARKER} === FIM DA OBSERVACAO === {json.dumps(summary, ensure_ascii=False)}")

    try:
        from fetchers.macro_update_service import stop_macro_service
        await stop_macro_service()
    except Exception:
        pass
    await heartbeat.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
