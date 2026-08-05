"""
Contrato unico de health do container (healthcheck do Docker).

Decisoes de arquitetura registradas:
  - mecanismo de verdade: endpoint HTTP dedicado /health. Respostas:
      200 {"status": "healthy"}   - tudo saudavel
      200 {"status": "degraded"}  - WebSocket conectado mas sem mensagens
                                    (silence > PIPELINE_HEALTH_DEGRADED_WS_SILENCE_SECONDS)
      503 {"status": "unhealthy"} - estagio critico sem heartbeat apos
                                    intervalo esperado * multiplicador
    O Dockerfile consome via `curl -f http://localhost:8000/health`
    (curl -f aceita 2xx, entao degraded nao reinicia o container).
  - agregador de exposicao: NAO detecta nada; le get_stats() do HealthMonitor
    existente (monitoring/health_monitor.py).
  - sinais: SO os heartbeats de estagio participam das decisoes:
      ws, ai, trade_ingestion, trade_buffer, window_processor, orderbook,
      event_saver.
    EXCLUIDOS (nao sao evidencia de progresso):
      "main" (auto-beat = apenas processo vivo), "ws_error" e
      "buffer_critical"/"buffer_overflow" (marcadores de eventos).
  - zumbi: nenhum estagio registrado -> unhealthy (evita falso "saudavel").
  - thresholds por estagio herdados do config
    (PIPELINE_HEALTH_STAGE_INTERVALS * PIPELINE_HEALTH_THRESHOLD_MULTIPLIER).
  - ALERTAS DE TRANSIÇÃO: disparo ativo nas mudancas de estado
    (healthy<->degraded<->unhealthy), com cooldown por transicao e canal
    webhook opcional (ALERT_WEBHOOK_URL, formato Slack-compativel) + log
    estruturado sempre. O AlertManager (trading/alert_manager.py) NAO foi
    usado: e um framework de agregacao de metricas via record_metric/check_metrics,
    sem canal de notificacao embutido; wire-lo adicionaria indirecao sem canal.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from config import (
    ALERT_STATE_COOLDOWN_SECONDS,
    ALERT_WEBHOOK_URL,
    HEALTH_CHECK_INTERVAL,
    PIPELINE_HEALTH_DEGRADED_WS_SILENCE_SECONDS,
    PIPELINE_HEALTH_STAGE_INTERVALS,
    PIPELINE_HEALTH_THRESHOLD_MULTIPLIER,
)

try:
    from prometheus_client import REGISTRY, Gauge, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    REGISTRY = Gauge = generate_latest = None

HEALTH_GAUGE = "trading_pipeline_healthy"
SILENCE_GAUGE = "trading_heartbeat_silence_seconds"
MESSAGE_AGE_GAUGE = "trading_last_message_age_seconds"
WINDOW_AGE_GAUGE = "trading_last_window_processed_age_seconds"

# Estagios que participam das decisoes de health/degraded
STAGE_COMPONENTS = frozenset(
    {
        "ws",
        "ai",
        "trade_ingestion",
        "trade_buffer",
        "window_processor",
        "orderbook",
        "event_saver",
    }
)

# Threshold critico por estagio: intervalo esperado * multiplicador
STAGE_CRITICAL_THRESHOLDS = {
    stage: float(interval) * float(PIPELINE_HEALTH_THRESHOLD_MULTIPLIER)
    for stage, interval in PIPELINE_HEALTH_STAGE_INTERVALS.items()
}

DEGRADED_WS_SILENCE = float(PIPELINE_HEALTH_DEGRADED_WS_SILENCE_SECONDS)

if _PROMETHEUS_AVAILABLE:
    _silence_gauge = Gauge(
        SILENCE_GAUGE,
        "Segundos desde o ultimo heartbeat do componente.",
        labelnames=["component"],
    )
    _healthy_gauge = Gauge(
        HEALTH_GAUGE,
        "1 se o pipeline esta saudavel (sem estagios criticos e com heartbeats).",
    )
    _message_age_gauge = Gauge(
        MESSAGE_AGE_GAUGE,
        "Idade da ultima mensagem recebida da Binance (segundos).",
    )
    _window_age_gauge = Gauge(
        WINDOW_AGE_GAUGE,
        "Idade da ultima janela processada (segundos).",
    )
else:
    _silence_gauge = None
    _healthy_gauge = None
    _message_age_gauge = None
    _window_age_gauge = None

_lock = threading.Lock()
_health_monitor = None
_stop_event = threading.Event()

_ws_connected_provider: Callable[[], bool] = lambda: False
_message_age_provider: Optional[Callable[[], Optional[float]]] = None
_window_age_provider: Optional[Callable[[], Optional[float]]] = None

# ---- Estado de transicoes de health (alerta ativo) ----
_last_status: Optional[str] = None  # ultimo status observado no loop de refresh
_last_alert_ts: dict = {}           # chave "from->to" -> timestamp do ultimo alerta
_last_unhealthy_components: list = []  # estagios criticos no ultimo unhealthy


def reset_health_state():
    """Zera o estado de transicoes (usado em testes)."""
    global _last_status
    _last_status = None
    _last_alert_ts.clear()
    _last_unhealthy_components.clear()


def attach_health_monitor(monitor):
    """Define o HealthMonitor de onde /health le o estado (leitura de get_stats)."""
    global _health_monitor
    with _lock:
        _health_monitor = monitor


def attach_ws_connected_provider(provider: Callable[[], bool]):
    """Provider do estado da conexao WebSocket (usado no estado degraded)."""
    global _ws_connected_provider
    _ws_connected_provider = provider or (lambda: False)


def attach_message_age_provider(provider: Callable[[], Optional[float]]):
    """Provider da idade da ultima mensagem recebida (segundos ou None)."""
    global _message_age_provider
    _message_age_provider = provider


def attach_window_age_provider(provider: Callable[[], Optional[float]]):
    """Provider da idade da ultima janela processada (segundos ou None)."""
    global _window_age_provider
    _window_age_provider = provider


def age_seconds(value) -> Optional[float]:
    """Idade em segundos de um datetime (aware ou naive) ou epoch seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        ts = value.timestamp()
    else:
        try:
            ts = float(value)
        except (TypeError, ValueError):
            return None
    return max(0.0, time.time() - ts)


def _status_from_stats(stats, ws_connected: bool) -> dict:
    heartbeats = stats.get("heartbeats") or {}
    monitored = set(stats.get("monitored_modules") or [])
    registered_stages = monitored & STAGE_COMPONENTS

    critical_components = []
    degraded = False

    for stage in STAGE_COMPONENTS:
        heartbeat = heartbeats.get(stage)
        if heartbeat is None:
            continue  # estagio nunca registrado: nao avaliado
        silence = float(heartbeat.get("silence_seconds", 0.0))
        threshold = STAGE_CRITICAL_THRESHOLDS.get(stage, 600.0)
        if silence > threshold:
            critical_components.append(stage)
        elif (
            stage == "ws"
            and silence > DEGRADED_WS_SILENCE
            and ws_connected
        ):
            degraded = True

    if not registered_stages:
        return {
            "status": "unhealthy",
            "healthy": False,
            "reason": "no_heartbeats",
            "unhealthy_components": [],
            "monitored_components": [],
        }

    if critical_components:
        return {
            "status": "unhealthy",
            "healthy": False,
            "reason": "critical_components",
            "unhealthy_components": sorted(critical_components),
            "monitored_components": sorted(registered_stages),
        }

    if degraded:
        return {
            "status": "degraded",
            "healthy": False,
            "reason": "ws_silence_connected",
            "unhealthy_components": [],
            "monitored_components": sorted(registered_stages),
        }

    return {
        "status": "healthy",
        "healthy": True,
        "reason": "ok",
        "unhealthy_components": [],
        "monitored_components": sorted(registered_stages),
    }


def compute_status(health_monitor) -> dict:
    """Estado agregado atual (usado pelo /health e pelos testes)."""
    return _status_from_stats(
        health_monitor.get_stats(), _ws_connected_provider()
    )


def _stage_detail(stage: str, heartbeats: dict) -> dict:
    heartbeat = heartbeats.get(stage) or {}
    return {
        "stage": stage,
        "silence_seconds": round(float(heartbeat.get("silence_seconds", 0.0)), 1),
        "threshold_seconds": round(
            STAGE_CRITICAL_THRESHOLDS.get(stage, 600.0), 1
        ),
    }


def _build_alert_payload(previous: str, current: str, status: dict,
                         heartbeats: dict) -> dict:
    payload = {
        "event": "health_state_transition",
        "from": previous,
        "to": current,
        "reason": status.get("reason"),
        "monitored_components": status.get("monitored_components", []),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if current == "unhealthy":
        components = status.get("unhealthy_components", [])
        payload["unhealthy_components"] = [
            _stage_detail(stage, heartbeats) for stage in components
        ]
        if components:
            detail = _stage_detail(components[0], heartbeats)
            payload["stage"] = detail["stage"]
            payload["silence_seconds"] = detail["silence_seconds"]
            payload["threshold_seconds"] = detail["threshold_seconds"]
    elif current == "degraded":
        payload["stage"] = "ws"
        silence = float(
            (heartbeats.get("ws") or {}).get("silence_seconds", 0.0)
        )
        payload["silence_seconds"] = round(silence, 1)
        payload["threshold_seconds"] = round(DEGRADED_WS_SILENCE, 1)
    elif current == "healthy" and _last_unhealthy_components:
        payload["recovered_components"] = [
            {"stage": item["stage"],
             "last_silence_seconds": item["silence_seconds"]}
            for item in _last_unhealthy_components
        ]
    return payload


def evaluate_transition(status: dict, heartbeats: dict) -> Optional[dict]:
    """Detecta mudanca de estado com cooldown por transicao.

    Retorna o payload do alerta se a transicao deve disparar, None caso
    contrario (estado inalterado, primeira avaliacao ou cooldown ativo).
    """
    global _last_status
    current = status["status"]
    previous = _last_status
    _last_status = current

    if current == "unhealthy":
        _last_unhealthy_components[:] = [
            _stage_detail(stage, heartbeats)
            for stage in status.get("unhealthy_components", [])
        ]

    if previous is None or previous == current:
        return None

    key = f"{previous}->{current}"
    now = time.time()
    if now - _last_alert_ts.get(key, 0.0) < ALERT_STATE_COOLDOWN_SECONDS:
        return None

    _last_alert_ts[key] = now
    return _build_alert_payload(previous, current, status, heartbeats)


def _post_webhook(url: str, payload: dict) -> None:
    """POST do alerta para o webhook (formato Slack-compativel)."""
    import requests

    body = {"text": payload["event"], "attachments": [{"fields": [
        {"title": "de", "value": payload["from"], "short": True},
        {"title": "para", "value": payload["to"], "short": True},
        {"title": "estagio", "value": payload.get("stage", "-"), "short": True},
        {"title": "silence (s)", "value": str(payload.get("silence_seconds", "-")), "short": True},
        {"title": "threshold (s)", "value": str(payload.get("threshold_seconds", "-")), "short": True},
        {"title": "reason", "value": payload.get("reason", "-"), "short": False},
    ]}]}
    resp = requests.post(url, json=body, timeout=10.0)
    resp.raise_for_status()


def _dispatch_alert(payload: dict) -> None:
    """Canal do alerta: log estruturado sempre + webhook se configurado."""
    logging.warning("ALERTA HEALTH: %s", json.dumps(payload, ensure_ascii=False))
    if ALERT_WEBHOOK_URL:
        try:
            _post_webhook(ALERT_WEBHOOK_URL, payload)
        except Exception as exc:
            logging.warning(f"Falha ao enviar webhook de health: {exc!r}")


def refresh(health_monitor) -> dict:
    """Atualiza as gauges no /metrics a partir do HealthMonitor e providers."""
    stats = health_monitor.get_stats()
    status = _status_from_stats(stats, _ws_connected_provider())
    payload = evaluate_transition(status, stats.get("heartbeats") or {})
    if payload is not None:
        _dispatch_alert(payload)
    if _PROMETHEUS_AVAILABLE:
        _healthy_gauge.set(1.0 if status["status"] == "healthy" else 0.0)
        for module, heartbeat in (stats.get("heartbeats") or {}).items():
            _silence_gauge.labels(component=module).set(
                heartbeat.get("silence_seconds", 0.0)
            )
        if _message_age_provider is not None:
            age = _message_age_provider()
            if age is not None:
                _message_age_gauge.set(float(age))
        if _window_age_provider is not None:
            age = _window_age_provider()
            if age is not None:
                _window_age_gauge.set(float(age))
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
                self._send_json(
                    503, {"status": "unhealthy", "healthy": False,
                          "reason": "no_monitor"}
                )
            else:
                status = compute_status(monitor)
                code = 503 if status["status"] == "unhealthy" else 200
                self._send_json(code, status)
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
    """Servidor HTTP unico: /metrics (Prometheus) + /health (200/200-degraded/503)."""
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
