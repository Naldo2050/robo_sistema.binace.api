"""
Contrato unico de health do container (healthcheck do Docker).

Decisoes de arquitetura registradas:
  - mecanismo de verdade: endpoint HTTP dedicado /health (200/503).
    O Dockerfile consome via `curl -f http://localhost:8000/health`,
    sem parsing de texto Prometheus em shell.
  - agregador de exposicao: NAO detecta nada; apenas le get_stats() do
    HealthMonitor existente (monitoring/health_monitor.py), que ja cuida
    de heartbeats, WARNING 90s e CRITICAL 180s.
  - sinais: componentes de heartbeat registrados dinamicamente
    (main, ai, ws, ws_error, buffer_critical, buffer_overflow);
    cada um e serie propria em trading_heartbeat_silence_seconds.
  - zumbi: processo vivo mas sem nenhum heartbeat registrado -> unhealthy
    (evita falso "saudavel" quando nada nunca bateu).
  - thresholds herdados do config (HEALTH_CHECK_TIMEOUT/CRITICAL),
    sem duplicar constantes.
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import HEALTH_CHECK_INTERVAL

try:
    from prometheus_client import REGISTRY, Gauge, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    REGISTRY = Gauge = generate_latest = None

HEALTH_GAUGE = "trading_pipeline_healthy"
SILENCE_GAUGE = "trading_heartbeat_silence_seconds"

if _PROMETHEUS_AVAILABLE:
    _silence_gauge = Gauge(
        SILENCE_GAUGE,
        "Segundos desde o ultimo heartbeat do componente.",
        labelnames=["component"],
    )
    _healthy_gauge = Gauge(
        HEALTH_GAUGE,
        "1 se o pipeline esta saudavel (sem alertas criticos e com heartbeats).",
    )
else:
    _silence_gauge = None
    _healthy_gauge = None

_lock = threading.Lock()
_health_monitor = None
_stop_event = threading.Event()


def attach_health_monitor(monitor):
    """Define o HealthMonitor de onde /health le o estado (leitura de get_stats)."""
    global _health_monitor
    with _lock:
        _health_monitor = monitor


def _status_from_stats(stats) -> dict:
    modules = stats.get("monitored_modules") or []
    heartbeats = stats.get("heartbeats") or {}
    critical = stats.get("active_critical_alerts", 0)
    warning = stats.get("active_warning_alerts", 0)

    unhealthy_components = sorted(
        module
        for module, heartbeat in heartbeats.items()
        if heartbeat.get("alert_level") == "critical"
    )

    if not modules:
        healthy, reason = False, "no_heartbeats"
    elif critical > 0:
        healthy, reason = False, "critical_components"
    else:
        healthy, reason = True, "ok"

    return {
        "healthy": healthy,
        "reason": reason,
        "critical": critical,
        "warning": warning,
        "unhealthy_components": unhealthy_components,
        "monitored_components": sorted(modules),
    }


def compute_status(health_monitor) -> dict:
    """Estado agregado atual (usado pelo /health e pelos testes)."""
    return _status_from_stats(health_monitor.get_stats())


def refresh(health_monitor) -> dict:
    """Atualiza as gauges no /metrics a partir do HealthMonitor."""
    stats = health_monitor.get_stats()
    status = _status_from_stats(stats)
    if _PROMETHEUS_AVAILABLE:
        _healthy_gauge.set(1.0 if status["healthy"] else 0.0)
        for module, heartbeat in (stats.get("heartbeats") or {}).items():
            _silence_gauge.labels(component=module).set(
                heartbeat.get("silence_seconds", 0.0)
            )
    return status


def _refresh_loop(monitor, interval: int):
    while not _stop_event.is_set():
        try:
            refresh(monitor)
        except Exception as exc:
            logging.warning(f"Erro ao atualizar gauges de health: {exc!r}")
        _stop_event.wait(interval)


def start_pipeline_health(health_monitor, interval: int = HEALTH_CHECK_INTERVAL):
    """Inicia o loop que publica o estado do HealthMonitor nas gauges."""
    attach_health_monitor(health_monitor)
    thread = threading.Thread(
        target=_refresh_loop,
        args=(health_monitor, max(1, interval)),
        daemon=True,
        name="pipeline-health",
    )
    thread.start()
    return thread


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            with _lock:
                monitor = _health_monitor
            if monitor is None:
                self._send_json(503, {"healthy": False, "reason": "no_monitor"})
            else:
                status = compute_status(monitor)
                self._send_json(200 if status["healthy"] else 503, status)
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/plain; version=0.0.4; charset=utf-8",
            )
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logging.debug("health http: " + fmt % args)


def serve_metrics_and_health(port: int = 8000):
    """Servidor HTTP unico: /metrics (Prometheus) + /health (200/503)."""
    if not _PROMETHEUS_AVAILABLE:
        raise ImportError("prometheus_client nao disponivel")
    server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="health-http",
    )
    thread.start()
    return server
