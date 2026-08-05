"""
Simula travamentos de estagios para validar o /health, os alertas de
transicao e os reminders de estado prolongado (TICKET A) manualmente.

Usa a MESMA logica de producao (monitoring.pipeline_health) com um
HealthMonitor fake. O endpoint /health sobe na porta 8999.

Sequencia (15s por cenario, ajustavel via SIM_STEP_SECONDS):
   1. TUDO SAUDAVEL                     -> 200 healthy
   2. WS SILENTE 120s + CONECTADO       -> 200 degraded   (alerta healthy->degraded)
   2b/2c. degraded persiste             -> reminder de degraded
   3. WS SILENTE 300s                   -> 503 unhealthy  (alerta degraded->unhealthy)
   4. WINDOW_PROCESSOR TRAVADO (700s)   -> 503 unhealthy  (sem alerta: mesmo estado)
   5. ORDERBOOK TRAVADO (200s)          -> 503 unhealthy  (reminder de unhealthy)
   6. VOLTA AO NORMAL                   -> 200 healthy    (alerta unhealthy->healthy)

Uso:
    python scripts/simulate_health_stall.py
    SIM_STEP_SECONDS=1 ALERT_REMINDER_SECONDS=2 python scripts/simulate_health_stall.py
        # demo rapida: reminder a cada 2s em vez de 1800s (intervalo real)

Sem ALERT_WEBHOOK_URL o alerta aparece como log estruturado
("ALERTA HEALTH: ...") - e o comportamento de producao sem webhook.
"""
import json
import logging
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import monitoring.pipeline_health as ph
from monitoring.pipeline_health import (
    attach_health_monitor,
    attach_message_age_provider,
    attach_window_age_provider,
    attach_ws_connected_provider,
    refresh,
    serve_metrics_and_health,
)

STAGES = [
    "ws",
    "ai",
    "trade_ingestion",
    "trade_buffer",
    "window_processor",
    "orderbook",
    "event_saver",
]

SIM_STEP_SECONDS = float(os.getenv("SIM_STEP_SECONDS", "15"))
# Intervalo do reminder configuravel para demos (default: o real de producao)
ph.ALERT_UNHEALTHY_REMINDER_SECONDS = float(
    os.getenv("ALERT_REMINDER_SECONDS", "1800")
)


class FakeHealthMonitor:
    """Stub minimo de HealthMonitor: heartbeats = {module: silence_seconds}."""

    def __init__(self, heartbeats=None):
        self._heartbeats = heartbeats or {}

    def set_silence(self, module, silence):
        self._heartbeats[module] = silence

    def get_stats(self):
        now = time.time()
        return {
            "warn_silence": 90,
            "critical_silence": 180,
            "check_interval": 30,
            "monitored_modules": list(self._heartbeats.keys()),
            "heartbeats": {
                module: {
                    "last_beat_ts": now - silence,
                    "silence_seconds": silence,
                    "alert_level": None,
                }
                for module, silence in self._heartbeats.items()
            },
            "active_critical_alerts": 0,
            "active_warning_alerts": 0,
            "oci_enabled": False,
        }


def _health_via_http(port):
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=5
        ) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _scenario(monitor, port, name, changes=None, connected=True):
    monitor._heartbeats = {m: 5 for m in STAGES}
    for module, silence in (changes or {}).items():
        monitor.set_silence(module, silence)
    attach_ws_connected_provider(lambda: connected)
    refresh(monitor)
    code, body = _health_via_http(port)
    print(f"\n=== CENARIO {name} ===")
    print(f"  /health -> HTTP {code} {json.dumps(body, ensure_ascii=False)}")


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    monitor = FakeHealthMonitor()
    attach_health_monitor(monitor)
    attach_message_age_provider(lambda: 5.0)
    attach_window_age_provider(lambda: 5.0)
    server = serve_metrics_and_health(port=8999)
    port = server.server_address[1]
    print(f"Servidor /health em http://localhost:{port}/health")
    print(f"Cenarios ({SIM_STEP_SECONDS:g}s cada):")

    scenarios = [
        ("1. TUDO SAUDAVEL", {}, True),
        ("2. WS SILENTE 120s + CONECTADO (degraded)", {"ws": 120}, True),
        ("2b. DEGRADED PERSISTE", {"ws": 120}, True),
        ("2c. DEGRADED PERSISTE", {"ws": 120}, True),
        ("3. WS SILENTE 300s (unhealthy)", {"ws": 300}, True),
        ("4. WINDOW_PROCESSOR TRAVADO 700s (unhealthy)", {"window_processor": 700}, True),
        ("5. ORDERBOOK TRAVADO 200s (unhealthy)", {"orderbook": 200}, True),
        ("6. VOLTA AO NORMAL", {}, True),
    ]
    try:
        for name, changes, connected in scenarios:
            _scenario(monitor, port, name, changes, connected)
            time.sleep(SIM_STEP_SECONDS)
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
