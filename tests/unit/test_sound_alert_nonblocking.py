# tests/unit/test_sound_alert_nonblocking.py
"""
FASE E3-A: áudio nunca bloqueia o hot path da janela.

Bug: EventSaver.save_event chamava winsound.Beep(1000, 500) de forma
síncrona por sinal (~500ms/sinal no worker da janela; até 2s no Darwin).

Contrato exigido:
  - default OFF (server/headless/produção); opt-in via SOUND_ALERT=1
    ou sound_alert=True explícito;
  - habilitado => worker único limitado, fila limitada (sem crescimento
    ilimitado; excesso descartado com contador);
  - falha de áudio nunca quebra/atrasa o processamento;
  - 1 e 5 sinais com beep simulado de 500ms não crescem ~500ms/sinal;
  - stop() não deixa a thread de áudio viva.
"""

import queue
import threading
import time

import pytest

from events.event_saver import EventSaver


def _signal(i=0):
    return {
        "is_signal": True,
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000 + i,
        "preco_fechamento": 78989.5,
    }


def _slow_beep_500ms(self):
    time.sleep(0.5)


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("SOUND_ALERT", raising=False)
    saver = EventSaver(sound_alert=None)
    try:
        assert saver.sound_alert is False
    finally:
        saver.stop()


def test_env_opt_in(monkeypatch):
    monkeypatch.setenv("SOUND_ALERT", "1")
    saver = EventSaver(sound_alert=None)
    try:
        assert saver.sound_alert is True
    finally:
        saver.stop()


def test_explicit_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("SOUND_ALERT", "1")
    saver = EventSaver(sound_alert=False)
    try:
        assert saver.sound_alert is False
    finally:
        saver.stop()


@pytest.mark.parametrize("n_signals", [1, 5])
def test_beep_500ms_does_not_block_window(monkeypatch, n_signals):
    """Com beep simulado de 500ms, N sinais não custam ~500ms/sinal."""
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = EventSaver(sound_alert=True)
    try:
        start = time.perf_counter()
        for i in range(n_signals):
            saver.save_event(_signal(i))
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5 * n_signals, (
            f"{n_signals} sinais levaram {elapsed:.2f}s (~500ms/sinal = bug)"
        )
    finally:
        saver.stop()


def test_audio_failure_never_breaks_save(monkeypatch):
    def _boom(self):
        raise RuntimeError("sem dispositivo de áudio")

    monkeypatch.setattr(EventSaver, "_play_sound", _boom)
    saver = EventSaver(sound_alert=True)
    try:
        saver.save_event(_signal())  # não deve levantar
        time.sleep(0.3)  # dá chance ao worker falhar em background
        saver.save_event(_signal(1))  # segue funcionando
    finally:
        saver.stop()


def test_queue_is_bounded_and_single_worker(monkeypatch):
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = EventSaver(sound_alert=True)
    try:
        threads_before = threading.active_count()
        start = time.perf_counter()
        for i in range(100):
            saver.save_event(_signal(i))
        elapsed = time.perf_counter() - start
        # 100 sinais precisam retornar rápido (fila limitada descarta excesso)
        assert elapsed < 5.0, f"100 enfileiramentos levaram {elapsed:.2f}s"
        workers = [t for t in threading.enumerate()
                   if t.name == "eventsaver-sound" and t.is_alive()]
        assert len(workers) <= 1
        assert threading.active_count() <= threads_before + 3
    finally:
        saver.stop()


def test_stop_kills_sound_thread(monkeypatch):
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = EventSaver(sound_alert=True)
    saver.save_event(_signal())
    time.sleep(0.2)
    assert any(t.name == "eventsaver-sound" and t.is_alive()
               for t in threading.enumerate())
    saver.stop()
    assert not any(t.name == "eventsaver-sound" and t.is_alive()
                   for t in threading.enumerate())
