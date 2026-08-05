"""
Testes do contrato unico de health do container (item healthcheck do Docker).

Decisoes de arquitetura registradas (aprovadas antes da implementacao):
  - mecanismo de verdade: HTTP via endpoint dedicado /health:
      200 healthy | 200 degraded (WS vivo sem mensagens) | 503 unhealthy
  - agregador de exposicao: le get_stats() do HealthMonitor existente
  - sinais: SO os heartbeats de ESTAGIO participam (ws, ai, trade_ingestion,
    trade_buffer, window_processor, orderbook, event_saver); "main",
    "ws_error", "buffer_critical", "buffer_overflow" sao excluidos
  - zumbi: nenhum estagio registrado -> unhealthy
  - thresholds por estagio: intervalo esperado * multiplicador (config)

Protocolo de prova: red (logica antiga sem estagios/degraded) -> green.
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
MESSAGE_AGE_GAUGE = "trading_last_message_age_seconds"
WINDOW_AGE_GAUGE = "trading_last_window_processed_age_seconds"

STAGES = [
    "ws",
    "ai",
    "trade_ingestion",
    "trade_buffer",
    "window_processor",
    "orderbook",
    "event_saver",
]
NON_STAGES = ["main", "ws_error", "buffer_critical", "buffer_overflow"]


class FakeHealthMonitor:
    """Stub de monitoring/health_monitor.py (sem thread, sem OCI).

    heartbeats: {module: silence_seconds} (0 = agora)
    """

    def __init__(self, heartbeats=None):
        self._heartbeats = heartbeats or {}

    def get_stats(self):
        now = time.time()
        heartbeats = {
            module: {
                "last_beat_ts": now - silence,
                "silence_seconds": silence,
                "alert_level": None,
            }
            for module, silence in self._heartbeats.items()
        }
        return {
            "warn_silence": 90,
            "critical_silence": 180,
            "check_interval": 30,
            "monitored_modules": list(self._heartbeats.keys()),
            "heartbeats": heartbeats,
            "active_critical_alerts": 0,
            "active_warning_alerts": 0,
            "oci_enabled": False,
        }


def _fresh_monitor(silence_by_module=None):
    heartbeats = {module: 5 for module in STAGES}
    if silence_by_module:
        heartbeats.update(silence_by_module)
    return FakeHealthMonitor(heartbeats)


def _connected():
    ph.attach_ws_connected_provider(lambda: True)


def _sample(name, labels=None):
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


# ══════════════════════════════════════════════════════════════════
# Cenario 1: tudo saudavel
# ══════════════════════════════════════════════════════════════════

def test_tudo_saudavel_status_healthy():
    status = ph.compute_status(_fresh_monitor())
    assert status["status"] == "healthy"
    assert status["healthy"] is True
    assert status["unhealthy_components"] == []


# ══════════════════════════════════════════════════════════════════
# Cenario 2: window_processor travado -> unhealthy (503)
# ══════════════════════════════════════════════════════════════════

def test_window_processor_travado_unhealthy():
    status = ph.compute_status(
        _fresh_monitor(silence_by_module={"window_processor": 700})
    )
    assert status["status"] == "unhealthy"
    assert status["unhealthy_components"] == ["window_processor"]


# ══════════════════════════════════════════════════════════════════
# Cenario 3: orderbook travado -> unhealthy
# ══════════════════════════════════════════════════════════════════

def test_orderbook_travado_unhealthy():
    status = ph.compute_status(
        _fresh_monitor(silence_by_module={"orderbook": 200})
    )
    assert status["status"] == "unhealthy"
    assert status["unhealthy_components"] == ["orderbook"]


# ══════════════════════════════════════════════════════════════════
# Cenario 4: zumbi - nenhum estagio registrado -> unhealthy
# ══════════════════════════════════════════════════════════════════

def test_zumbi_sem_estagios_unhealthy():
    status = ph.compute_status(FakeHealthMonitor(heartbeats={}))
    assert status["status"] == "unhealthy"
    assert status["reason"] == "no_heartbeats"


# ══════════════════════════════════════════════════════════════════
# Degraded: WS conectado mas sem mensagens (90s < silence <= critical)
# ══════════════════════════════════════════════════════════════════

def test_ws_silente_conectado_degraded():
    _connected()
    status = ph.compute_status(
        _fresh_monitor(silence_by_module={"ws": 120})
    )
    assert status["status"] == "degraded"
    assert status["reason"] == "ws_silence_connected"


def test_ws_silente_desconectado_nao_degraded():
    ph.attach_ws_connected_provider(lambda: False)
    status = ph.compute_status(
        _fresh_monitor(silence_by_module={"ws": 120})
    )
    assert status["status"] == "healthy"


def test_ws_silence_maior_que_critical_unhealthy():
    _connected()
    status = ph.compute_status(
        _fresh_monitor(silence_by_module={"ws": 300})
    )
    assert status["status"] == "unhealthy"
    assert status["unhealthy_components"] == ["ws"]


# ══════════════════════════════════════════════════════════════════
# Exclusao: main / ws_error / buffer_* NAO afetam as decisoes
# ══════════════════════════════════════════════════════════════════

def test_marcadores_nao_estagio_sao_ignorados():
    silence = {module: 999999 for module in NON_STAGES}
    status = ph.compute_status(_fresh_monitor(silence_by_module=silence))
    assert status["status"] == "healthy"


# ══════════════════════════════════════════════════════════════════
# Gauges no /metrics
# ══════════════════════════════════════════════════════════════════

def test_refresh_atualiza_gauges_de_silence():
    ph.refresh(_fresh_monitor())
    assert _sample(HEALTH_GAUGE) == 1.0
    for stage in STAGES:
        assert _sample(SILENCE_GAUGE, {"component": stage}) == pytest.approx(
            5.0, abs=2.0
        )


def test_refresh_reflete_falha_na_gauge():
    ph.refresh(_fresh_monitor(silence_by_module={"orderbook": 200}))
    assert _sample(HEALTH_GAUGE) == 0.0


def test_refresh_publica_gauges_de_idade():
    ph.attach_message_age_provider(lambda: 12.5)
    ph.attach_window_age_provider(lambda: 77.0)
    ph.refresh(_fresh_monitor())
    assert _sample(MESSAGE_AGE_GAUGE) == 12.5
    assert _sample(WINDOW_AGE_GAUGE) == 77.0


def test_age_seconds_datetime_aware_naive_e_none():
    now = time.time()
    assert ph.age_seconds(None) is None
    assert ph.age_seconds(now - 30) == pytest.approx(30.0, abs=2.0)
    from datetime import datetime, timezone
    assert ph.age_seconds(
        datetime.now(timezone.utc) - __import__("datetime").timedelta(seconds=15)
    ) == pytest.approx(15.0, abs=2.0)
    assert ph.age_seconds(datetime.utcnow() - __import__("datetime").timedelta(seconds=10)) \
        == pytest.approx(10.0, abs=2.0)


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


def test_endpoint_health_200_saudavel(health_server):
    ph.attach_health_monitor(_fresh_monitor())
    status, body = _get(health_server.server_address[1], "/health")
    assert status == 200
    assert json.loads(body)["status"] == "healthy"


def test_endpoint_health_200_degraded(health_server):
    _connected()
    ph.attach_health_monitor(
        _fresh_monitor(silence_by_module={"ws": 120})
    )
    status, body = _get(health_server.server_address[1], "/health")
    assert status == 200
    assert json.loads(body)["status"] == "degraded"


def test_endpoint_health_503_unhealthy(health_server):
    ph.attach_health_monitor(
        _fresh_monitor(silence_by_module={"window_processor": 700})
    )
    status, body = _get(health_server.server_address[1], "/health")
    assert status == 503
    assert json.loads(body)["status"] == "unhealthy"


def test_endpoint_metrics_continua_disponivel(health_server):
    status, body = _get(health_server.server_address[1], "/metrics")
    assert status == 200
    assert b"trading_pipeline_healthy" in body
    assert b"trading_last_message_age_seconds" in body
