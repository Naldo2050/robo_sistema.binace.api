"""
Testes dos alertas de transicao de estado do health (TICKET A).

Transicoes obrigatorias:
  - healthy -> degraded
  - degraded -> unhealthy
  - unhealthy -> healthy (recuperacao, fecha o loop)
Requisitos: cooldown por transicao (1 alerta a cada X minutos), mensagem com
estagio causador + tempo sem heartbeat + threshold configurado, canal webhook
opcional + log estruturado sempre.

Protocolo de prova: red (evaluate_transition nao existe) -> green.
"""
import time

import pytest

pytest.importorskip("prometheus_client")

import monitoring.pipeline_health as ph

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
    """Stub de HealthMonitor: heartbeats = {module: silence_seconds}."""

    def __init__(self, heartbeats=None):
        self._heartbeats = heartbeats or {}

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


def _fresh(silence_by_module=None):
    heartbeats = {module: 5 for module in STAGES}
    if silence_by_module:
        heartbeats.update(silence_by_module)
    return FakeHealthMonitor(heartbeats)


def _status(monitor, connected=True):
    ph.attach_ws_connected_provider(lambda: connected)
    return ph.compute_status(monitor)


def _stats(monitor):
    return monitor.get_stats()


@pytest.fixture(autouse=True)
def _reset():
    ph.reset_health_state()
    yield


# ══════════════════════════════════════════════════════════════════
# Transicao 1: healthy -> degraded (ws conectado, sem mensagens)
# ══════════════════════════════════════════════════════════════════

def test_transicao_healthy_para_degraded():
    mon = _fresh()
    assert ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"]) is None

    mon = _fresh(silence_by_module={"ws": 120})
    payload = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert payload is not None
    assert payload["from"] == "healthy"
    assert payload["to"] == "degraded"
    assert payload["stage"] == "ws"
    assert payload["silence_seconds"] == pytest.approx(120.0)
    assert payload["threshold_seconds"] == pytest.approx(
        ph.DEGRADED_WS_SILENCE
    )


# ══════════════════════════════════════════════════════════════════
# Transicao 2: degraded -> unhealthy (ws passou do threshold critico)
# ══════════════════════════════════════════════════════════════════

def test_transicao_degraded_para_unhealthy():
    mon = _fresh(silence_by_module={"ws": 120})
    ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])

    mon = _fresh(silence_by_module={"ws": 300})
    payload = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert payload is not None
    assert payload["from"] == "degraded"
    assert payload["to"] == "unhealthy"
    assert payload["stage"] == "ws"
    assert payload["silence_seconds"] == pytest.approx(300.0)
    assert payload["threshold_seconds"] == pytest.approx(
        ph.STAGE_CRITICAL_THRESHOLDS["ws"]
    )


# ══════════════════════════════════════════════════════════════════
# Transicao 3: unhealthy -> healthy (recuperacao, fecha o loop)
# ══════════════════════════════════════════════════════════════════

def test_transicao_unhealthy_para_healthy_recuperacao():
    mon = _fresh(silence_by_module={"window_processor": 700})
    ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])

    mon = _fresh()
    payload = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert payload is not None
    assert payload["from"] == "unhealthy"
    assert payload["to"] == "healthy"
    recovered = payload.get("recovered_components", [])
    assert any(item["stage"] == "window_processor" for item in recovered)


# ══════════════════════════════════════════════════════════════════
# healthy -> unhealthy direto (sem passo degraded) tambem dispara
# ══════════════════════════════════════════════════════════════════

def test_transicao_healthy_para_unhealthy_direta():
    mon = _fresh()
    ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])

    mon = _fresh(silence_by_module={"orderbook": 200})
    payload = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert payload is not None
    assert payload["to"] == "unhealthy"
    assert payload["stage"] == "orderbook"
    assert payload["threshold_seconds"] == pytest.approx(
        ph.STAGE_CRITICAL_THRESHOLDS["orderbook"]
    )


# ══════════════════════════════════════════════════════════════════
# Regras: mesmo estado nao dispara; cooldown suprime repeticao
# ══════════════════════════════════════════════════════════════════

def test_mesmo_estado_nao_dispara():
    mon = _fresh()
    ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"]) is None


def test_cooldown_suprime_repeticao_da_mesma_transicao():
    mon = _fresh()
    ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])

    mon = _fresh(silence_by_module={"orderbook": 200})
    first = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert first is not None

    # simula flapping: volta para healthy sem passar o cooldown da transicao
    ph._last_status = "healthy"
    second = ph.evaluate_transition(_status(mon), _stats(mon)["heartbeats"])
    assert second is None  # cooldown ativo para healthy->unhealthy


# ══════════════════════════════════════════════════════════════════
# Canal: webhook quando configurado, log sempre
# ══════════════════════════════════════════════════════════════════

def test_dispatch_chama_webhook_quando_configurado(monkeypatch):
    calls = []
    monkeypatch.setattr(ph, "_post_webhook", lambda url, payload: calls.append(url))
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", "http://exemplo/webhook")

    ph._dispatch_alert({"event": "health_state_transition", "from": "x", "to": "y"})
    assert calls == ["http://exemplo/webhook"]


def test_dispatch_sem_webhook_apenas_loga(monkeypatch):
    calls = []
    monkeypatch.setattr(ph, "_post_webhook", lambda url, payload: calls.append(url))
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", None)

    ph._dispatch_alert({"event": "health_state_transition", "from": "x", "to": "y"})
    assert calls == []


# ══════════════════════════════════════════════════════════════════
# Integracao: refresh() dispara o alerta de transicao ponta a ponta
# ══════════════════════════════════════════════════════════════════

def test_refresh_dispara_alerta_na_transicao(monkeypatch):
    fired = []
    monkeypatch.setattr(ph, "_dispatch_alert", lambda payload: fired.append(payload))

    mon = _fresh()
    ph.refresh(mon)  # healthy (primeira avaliacao: sem alerta)
    assert fired == []

    mon = _fresh(silence_by_module={"ws": 120})
    ph.attach_ws_connected_provider(lambda: True)
    ph.refresh(mon)  # healthy -> degraded
    assert len(fired) == 1
    assert fired[0]["from"] == "healthy"
    assert fired[0]["to"] == "degraded"
    assert fired[0]["stage"] == "ws"
