"""
Testes do contrato unico de health do container (item healthcheck do Docker).

Decisoes de arquitetura registradas (aprovadas antes da implementacao):
  - mecanismo de verdade: HTTP via endpoint dedicado /health (200/503),
    sem parsing de texto Prometheus em shell
  - agregador de exposicao: le get_stats() do HealthMonitor existente,
    nao duplica logica de heartbeat
  - sinais: componentes de heartbeat existentes (main, ai, ws, ws_error,
    buffer_critical, buffer_overflow); cada um e serie propria
  - zumbi: processo vivo mas sem nenhum heartbeat registrado -> unhealthy
  - thresholds herdados do config (HEALTH_CHECK_TIMEOUT/CRITICAL)

Protocolo de prova: red (modulo nao existe / gauges ausentes) -> green.
Criterio de aceite: 4 cenarios do pedido original + endpoint /health.
"""
import json
import time
from http.client import HTTPConnection

import pytest

pytest.importorskip("prometheus_client")
from prometheus_client import REGISTRY

import monitoring.pipeline_health as ph

SILENCE_GAUGE = "trading_heartbeat_silence_seconds"
HEALTH_GAUGE = "trading_pipeline_healthy"


class FakeHealthMonitor:
    """Stub de monitoring/health_monitor.py (sem thread, sem OCI)."""

    def __init__(self, heartbeats=None, alerts=None):
        self._heartbeats = heartbeats or {}
        self._alerts = alerts or {}

    def get_stats(self):
        now = time.time()
        heartbeats = {
            module: {
                "last_beat_ts": ts,
                "silence_seconds": now - ts,
                "alert_level": self._alerts.get(module),
            }
            for module, ts in self._heartbeats.items()
        }
        return {
            "warn_silence": 90,
            "critical_silence": 180,
            "check_interval": 30,
            "monitored_modules": list(self._heartbeats.keys()),
            "heartbeats": heartbeats,
            "active_critical_alerts": sum(
                1 for level in self._alerts.values() if level == "critical"
            ),
            "active_warning_alerts": sum(
                1 for level in self._alerts.values() if level == "warning"
            ),
            "oci_enabled": False,
        }


def _fresh_monitor(seconds_ago=10, alerts=None):
    now = time.time()
    modules = {"main", "ai", "ws", "buffer_critical"}
    return FakeHealthMonitor(
        heartbeats={module: now - seconds_ago for module in modules},
        alerts=alerts or {},
    )


def _sample(name, labels=None):
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


# ══════════════════════════════════════════════════════════════════
# Cenario 1: tudo saudavel
# ══════════════════════════════════════════════════════════════════

def test_tudo_saudavel_pipeline_healthy():
    status = ph.compute_status(_fresh_monitor(seconds_ago=10))
    assert status["healthy"] is True
    assert status["critical"] == 0
    assert status["unhealthy_components"] == []


# ══════════════════════════════════════════════════════════════════
# Cenario 2: WebSocket parado (componente "ws" critico)
# ══════════════════════════════════════════════════════════════════

def test_ws_parado_torna_pipeline_unhealthy():
    status = ph.compute_status(
        _fresh_monitor(seconds_ago=10, alerts={"ws": "critical"})
    )
    assert status["healthy"] is False
    assert status["critical"] == 1
    assert status["unhealthy_components"] == ["ws"]


# ══════════════════════════════════════════════════════════════════
# Cenario 3: pipeline de IA travado (componente "ai" critico)
# ══════════════════════════════════════════════════════════════════

def test_ia_travada_torna_pipeline_unhealthy():
    status = ph.compute_status(
        _fresh_monitor(seconds_ago=10, alerts={"ai": "critical"})
    )
    assert status["healthy"] is False
    assert status["unhealthy_components"] == ["ai"]


# ══════════════════════════════════════════════════════════════════
# Cenario 4: zumbi - processo vivo mas nenhum heartbeat registrado
# ══════════════════════════════════════════════════════════════════

def test_zumbi_sem_heartbeats_e_unhealthy():
    status = ph.compute_status(FakeHealthMonitor(heartbeats={}, alerts={}))
    assert status["healthy"] is False
    assert status["reason"] == "no_heartbeats"


# ══════════════════════════════════════════════════════════════════
# Gauges no /metrics
# ══════════════════════════════════════════════════════════════════

def test_refresh_atualiza_gauges():
    ph.refresh(_fresh_monitor(seconds_ago=10))
    assert _sample(HEALTH_GAUGE) == 1.0
    assert _sample(SILENCE_GAUGE, {"component": "main"}) == pytest.approx(
        10.0, abs=2.0
    )


def test_refresh_reflete_falha_na_gauge():
    ph.refresh(_fresh_monitor(seconds_ago=10, alerts={"ws": "critical"}))
    assert _sample(HEALTH_GAUGE) == 0.0


# ══════════════════════════════════════════════════════════════════
# Endpoint /health (mecanismo de verdade do Docker)
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def health_server():
    server = ph.serve_metrics_and_health(port=0)
    yield server
    server.shutdown()
    server.server_close()


def _get(port, path):
    conn = HTTPConnection("127.0.0.1", port, timeout=3)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def test_endpoint_health_200_quando_saudavel(health_server):
    ph.attach_health_monitor(_fresh_monitor(seconds_ago=10))
    port = health_server.server_address[1]
    status, body = _get(port, "/health")
    assert status == 200
    assert json.loads(body)["healthy"] is True


def test_endpoint_health_503_quando_unhealthy(health_server):
    ph.attach_health_monitor(
        _fresh_monitor(seconds_ago=10, alerts={"ws": "critical"})
    )
    port = health_server.server_address[1]
    status, body = _get(port, "/health")
    assert status == 503
    assert json.loads(body)["healthy"] is False


def test_endpoint_metrics_continua_disponivel(health_server):
    port = health_server.server_address[1]
    status, body = _get(port, "/metrics")
    assert status == 200
    assert b"trading_pipeline_healthy" in body
