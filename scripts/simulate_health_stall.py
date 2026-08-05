"""
Simula travamentos de estagios para validar o /health manualmente.

Usa a MESMA logica de producao (monitoring.pipeline_health) com um
HealthMonitor fake. O endpoint /health sobe na porta 8999.

Uso:
    python scripts/simulate_health_stall.py
    # em outro terminal:
    curl -s http://localhost:8999/health      # healthy (200)
    # o script muda os cenario a cada 15s:
    #   1. tudo saudavel                    -> 200 healthy
    #   2. window_processor travado 700s    -> 503 unhealthy
    #   3. orderbook travado 200s           -> 503 unhealthy
    #   4. ws silente 120s + conectado      -> 200 degraded
    #   5. ws silente 300s                  -> 503 unhealthy
    #   6. volta ao normal                  -> 200 healthy
"""
import json
import time

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


def _scenario(monitor, name, changes=None, connected=True):
    monitor._heartbeats = {m: 5 for m in STAGES}
    for module, silence in (changes or {}).items():
        monitor.set_silence(module, silence)
    attach_ws_connected_provider(lambda: connected)
    refresh(monitor)
    print(f"\n=== CENARIO: {name} ===")
    print(f"  curl -s http://localhost:8999/health ->")


def main():
    monitor = FakeHealthMonitor()
    attach_health_monitor(monitor)
    attach_message_age_provider(lambda: 5.0)
    attach_window_age_provider(lambda: 5.0)
    server = serve_metrics_and_health(port=8999)
    port = server.server_address[1]
    print(f"Servidor /health em http://localhost:{port}/health")
    print("Cenarios (15s cada, observe com curl):")

    scenarios = [
        ("1. TUDO SAUDAVEL", {}, True),
        ("2. WINDOW_PROCESSOR TRAVADO (700s)", {"window_processor": 700}, True),
        ("3. ORDERBOOK TRAVADO (200s)", {"orderbook": 200}, True),
        ("4. WS SILENTE 120s + CONECTADO (degraded)", {"ws": 120}, True),
        ("5. WS SILENTE 300s (unhealthy)", {"ws": 300}, True),
        ("6. VOLTA AO NORMAL", {}, True),
    ]
    try:
        for name, changes, connected in scenarios:
            _scenario(monitor, name, changes, connected)
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
