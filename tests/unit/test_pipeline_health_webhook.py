"""
Testes REAIS do webhook de health (TICKET A, item 2).

Sem dependencia nova: servidor HTTP local (ThreadingHTTPServer em porta
efemera) captura a requisicao de verdade e verifica:
  1. formato Slack-compatible do JSON (text + attachments/fields)
  2. metodo POST + Content-Type application/json
  3. timeout respeitado (servidor lento) com excecao capturada, sem propagar
  4. falha real (500 / conexao recusada) logada, fluxo principal intacto
"""
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("prometheus_client")

import monitoring.pipeline_health as ph


class _WebhookRecorder:
    def __init__(self):
        self.requests = []
        self.status = 200
        self.delay = 0.0


def _webhook_server(recorder):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(recorder.delay)
            length = int(self.headers.get("Content-Length") or 0)
            recorder.requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type"),
                    "body": self.rfile.read(length) if length else b"",
                }
            )
            self.send_response(recorder.status)
            self.end_headers()
            try:
                self.wfile.write(b"ok")
            except OSError:
                pass  # cliente ja desistiu (timeout no teste de servidor lento)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def webhook_server():
    recorder = _WebhookRecorder()
    server = _webhook_server(recorder)
    url = f"http://127.0.0.1:{server.server_address[1]}/hooks/health"
    yield server, recorder, url
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _reset():
    ph.reset_health_state()
    yield


UNHEALTHY_PAYLOAD = {
    "event": "health_state_transition",
    "from": "healthy",
    "to": "unhealthy",
    "reason": "critical_components",
    "monitored_components": ["orderbook"],
    "stage": "orderbook",
    "silence_seconds": 200.0,
    "threshold_seconds": 120.0,
    "ts": "2026-08-05T00:00:00+00:00",
}

REMINDER_PAYLOAD = {
    "event": "health_state_reminder",
    "state": "unhealthy",
    "duration_in_state_seconds": 3600.0,
    "reminder_interval_seconds": 1800.0,
    "reason": "critical_components",
    "monitored_components": ["orderbook"],
    "stage": "orderbook",
    "silence_seconds": 200.0,
    "threshold_seconds": 120.0,
    "ts": "2026-08-05T00:00:00+00:00",
}


def _fields(body):
    return {f["title"]: f["value"] for f in body["attachments"][0]["fields"]}


# ══════════════════════════════════════════════════════════════════
# 1. Formato Slack-compatible (funcao pura)
# ══════════════════════════════════════════════════════════════════

def test_webhook_body_formato_slack_transicao():
    body = ph._build_webhook_body(UNHEALTHY_PAYLOAD)
    assert body["text"] == "health_state_transition"
    assert isinstance(body["attachments"], list)
    fields = _fields(body)
    assert fields["de"] == "healthy"
    assert fields["para"] == "unhealthy"
    assert fields["estagio"] == "orderbook"
    assert fields["silence (s)"] == "200.0"
    assert fields["threshold (s)"] == "120.0"
    assert fields["reason"] == "critical_components"


def test_webhook_body_reminder_usa_campo_estado_e_duracao():
    body = ph._build_webhook_body(REMINDER_PAYLOAD)
    assert body["text"] == "health_state_reminder"
    fields = _fields(body)
    assert fields["estado"] == "unhealthy"
    assert fields["ha quanto tempo (s)"] == "3600.0"
    assert "de" not in fields and "para" not in fields
    assert fields["estagio"] == "orderbook"


# ══════════════════════════════════════════════════════════════════
# 2. Metodo POST + Content-Type corretos (requisicao real)
# ══════════════════════════════════════════════════════════════════

def test_post_webhook_metodo_post_content_type_e_body(webhook_server):
    server, recorder, url = webhook_server
    ph._post_webhook(url, UNHEALTHY_PAYLOAD)

    assert len(recorder.requests) == 1
    req = recorder.requests[0]
    assert req["method"] == "POST"
    assert req["path"] == "/hooks/health"
    assert req["content_type"].startswith("application/json")
    received = json.loads(req["body"].decode("utf-8"))
    assert received == ph._build_webhook_body(UNHEALTHY_PAYLOAD)


# ══════════════════════════════════════════════════════════════════
# 3. Timeout: servidor lento -> excecao capturada, nao propaga
# ══════════════════════════════════════════════════════════════════

def test_timeout_padrao_e_10_segundos():
    assert ph.WEBHOOK_TIMEOUT_SECONDS == 10.0


def test_post_webhook_timeout_excecao_capturada_nao_propaga(
    webhook_server, caplog, monkeypatch
):
    server, recorder, url = webhook_server
    recorder.delay = 5.0  # resposta alem do timeout do teste
    monkeypatch.setattr(ph, "WEBHOOK_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", url)

    with caplog.at_level(logging.WARNING):
        ph._dispatch_alert(UNHEALTHY_PAYLOAD)  # nao deve levantar

    assert "Falha ao enviar webhook de health" in caplog.text
    # o log do alerta em si continua sendo emitido mesmo com webhook falho
    assert "ALERTA HEALTH" in caplog.text


# ══════════════════════════════════════════════════════════════════
# 4. Falha real: 500 e conexao recusada -> logada, fluxo intacto
# ══════════════════════════════════════════════════════════════════

def test_post_webhook_erro_500_logado_e_fluxo_intacto(
    webhook_server, caplog, monkeypatch
):
    server, recorder, url = webhook_server
    recorder.status = 500
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", url)

    with caplog.at_level(logging.WARNING):
        ph._dispatch_alert(UNHEALTHY_PAYLOAD)

    assert "Falha ao enviar webhook de health" in caplog.text


def test_post_webhook_conexao_recusada_logada(webhook_server, caplog, monkeypatch):
    server, recorder, url = webhook_server
    server.shutdown()
    server.server_close()
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", url)

    with caplog.at_level(logging.WARNING):
        ph._dispatch_alert(UNHEALTHY_PAYLOAD)

    assert "Falha ao enviar webhook de health" in caplog.text


def test_refresh_sobrevive_a_falha_do_webhook(webhook_server, caplog, monkeypatch):
    """O refresh() continua respondendo (status retornado) apos falha de POST."""
    server, recorder, url = webhook_server
    server.shutdown()
    server.server_close()
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", url)

    class FakeMonitor:
        def __init__(self, silence):
            self._silence = silence

        def set_silence(self, silence):
            self._silence = silence

        def get_stats(self):
            now = time.time()
            return {
                "warn_silence": 90,
                "critical_silence": 180,
                "check_interval": 30,
                "monitored_modules": ["ws"],
                "heartbeats": {
                    "ws": {
                        "last_beat_ts": now - self._silence,
                        "silence_seconds": self._silence,
                        "alert_level": None,
                    }
                },
                "active_critical_alerts": 0,
                "active_warning_alerts": 0,
                "oci_enabled": False,
            }

    monkeypatch.setattr(ph, "_ws_connected_provider", lambda: False)
    monitor = FakeMonitor(silence=5)
    with caplog.at_level(logging.WARNING):
        ph.refresh(monitor)  # healthy: primeira avaliacao, sem alerta
        monitor.set_silence(300)
        status = ph.refresh(monitor)  # healthy -> unhealthy: alerta, POST falha

    assert status["status"] == "unhealthy"  # fluxo vivo apos falha de POST
    assert "Falha ao enviar webhook de health" in caplog.text
