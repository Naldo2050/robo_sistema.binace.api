"""
Testes do reminder de estado prolongado (TICKET A, item 3).

Cobre unhealthy E degraded prolongados:
  1. estado permanece unhealthy > ALERT_UNHEALTHY_REMINDER_SECONDS -> reminder
  2. degraded prolongado tambem gera reminder
  3. volta a healthy antes do intervalo -> reminder nao dispara depois
  4. reminder reaproveita o canal do webhook (_post_webhook)
"""
import logging
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

REMINDER = float(ph.ALERT_UNHEALTHY_REMINDER_SECONDS)


class _FakeClock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeHealthMonitor:
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


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(ph.time, "time", fake.time)
    return fake


@pytest.fixture(autouse=True)
def _reset(clock):
    ph.reset_health_state()
    yield clock


# ══════════════════════════════════════════════════════════════════
# 1. Unhealthy prolongado -> reminder apos o intervalo
# ══════════════════════════════════════════════════════════════════

def test_reminder_unhealthy_apos_intervalo(clock):
    mon = _fresh()
    assert ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"]) is None

    mon = _fresh(silence_by_module={"ws": 300})
    clock.advance(5)
    transition = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert transition is not None and transition["to"] == "unhealthy"

    clock.advance(REMINDER - 1)
    assert ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"]) is None

    clock.advance(1)  # exatamente no intervalo desde a entrada
    reminder = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert reminder is not None
    assert reminder["event"] == "health_state_reminder"
    assert reminder["state"] == "unhealthy"
    assert reminder["duration_in_state_seconds"] == pytest.approx(
        REMINDER, abs=0.1
    )
    assert reminder["stage"] == "ws"
    assert reminder["silence_seconds"] == pytest.approx(300.0)
    assert reminder["threshold_seconds"] == pytest.approx(
        ph.STAGE_CRITICAL_THRESHOLDS["ws"]
    )

    clock.advance(REMINDER)  # continua unhealthy: novo reminder
    second = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert second is not None
    assert second["event"] == "health_state_reminder"
    assert second["duration_in_state_seconds"] == pytest.approx(
        2 * REMINDER, abs=0.1
    )


# ══════════════════════════════════════════════════════════════════
# 2. Degraded prolongado tambem gera reminder
# ══════════════════════════════════════════════════════════════════

def test_reminder_degraded_tambem_dispara(clock):
    mon = _fresh()
    ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])

    mon = _fresh(silence_by_module={"ws": 120})
    clock.advance(5)
    transition = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert transition is not None and transition["to"] == "degraded"

    clock.advance(REMINDER)
    reminder = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert reminder is not None
    assert reminder["event"] == "health_state_reminder"
    assert reminder["state"] == "degraded"
    assert reminder["stage"] == "ws"
    assert reminder["duration_in_state_seconds"] == pytest.approx(
        REMINDER, abs=0.1
    )


# ══════════════════════════════════════════════════════════════════
# 3. Volta a healthy antes do intervalo -> sem reminder indevido
# ══════════════════════════════════════════════════════════════════

def test_sem_reminder_se_voltar_a_healthy_antes(clock):
    mon = _fresh()
    ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])

    mon = _fresh(silence_by_module={"ws": 300})
    clock.advance(100)
    transition = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert transition is not None and transition["to"] == "unhealthy"

    clock.advance(REMINDER - 100)  # duracao no unhealthy: REMINDER - 100 < intervalo
    assert ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"]) is None

    mon = _fresh()  # volta a healthy ANTES do intervalo
    clock.advance(100)
    recovery = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert recovery is not None and recovery["to"] == "healthy"

    clock.advance(REMINDER)  # saudavel por muito tempo
    assert ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"]) is None

    # volta a unhealthy: o relogio do reminder reinicia (nao usa o passado antigo)
    mon = _fresh(silence_by_module={"ws": 300})
    clock.advance(5)
    transition2 = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert transition2 is not None and transition2["to"] == "unhealthy"

    clock.advance(REMINDER - 1)
    assert ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"]) is None

    clock.advance(1)
    reminder = ph.evaluate_transition(_status(mon), mon.get_stats()["heartbeats"])
    assert reminder is not None
    assert reminder["event"] == "health_state_reminder"
    assert reminder["duration_in_state_seconds"] == pytest.approx(
        REMINDER, abs=0.1
    )


# ══════════════════════════════════════════════════════════════════
# 4. Reminder passa pelo mesmo canal do webhook (reaproveita _post_webhook)
# ══════════════════════════════════════════════════════════════════

def test_reminder_usando_canal_do_webhook(clock, monkeypatch, caplog):
    posted = []
    monkeypatch.setattr(
        ph, "_post_webhook", lambda url, payload, timeout=None: posted.append(payload)
    )
    monkeypatch.setattr(ph, "ALERT_WEBHOOK_URL", "http://exemplo/webhook")

    mon = _fresh()
    ph.refresh(mon)

    mon = _fresh(silence_by_module={"ws": 300})
    clock.advance(5)
    ph.refresh(mon)  # healthy -> unhealthy (transicao via refresh)

    clock.advance(REMINDER)
    with caplog.at_level(logging.WARNING):
        status = ph.refresh(mon)  # unhealthy prolongado -> reminder via refresh

    assert status["status"] == "unhealthy"
    assert len(posted) == 2  # 1 transicao + 1 reminder
    assert posted[1]["event"] == "health_state_reminder"
    assert posted[1]["state"] == "unhealthy"
    assert "ALERTA HEALTH" in caplog.text
