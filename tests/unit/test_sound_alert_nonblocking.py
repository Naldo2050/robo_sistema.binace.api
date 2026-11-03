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

INC5-A: todos os caminhos operacionais são redirecionados para tmp_path.
"""

import hashlib
import multiprocessing
import queue
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from database.event_store import EventStore as RealEventStore
from events import event_saver
from events.event_saver import EventSaver

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path):
    if not root.exists():
        return ()
    entries = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append(("dir", rel, None, None))
        else:
            entries.append(("file", rel, path.stat().st_size, _sha256(path)))
    return tuple(entries)


@pytest.fixture
def isolated_saver_factory(tmp_path, monkeypatch):
    data_dir = tmp_path / "dados"
    data_dir.mkdir()
    paths = {
        "data_dir": data_dir,
        "db_path": tmp_path / "events.db",
        "history_file": tmp_path / "eventos_fluxo.jsonl",
        "visual_log_file": tmp_path / "eventos_visuais.log",
        "snapshot_file": tmp_path / "eventos-fluxo.json",
        "archive_dir": tmp_path / "archive",
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(event_saver, "DATA_DIR", data_dir)
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)

    def _tmp_event_store(*args, **kwargs):
        kwargs["db_path"] = str(paths["db_path"])
        return RealEventStore(*args, **kwargs)

    monkeypatch.setattr(event_saver, "EventStore", _tmp_event_store)

    created = []

    def make_saver(sound_alert):
        saver = EventSaver(sound_alert=sound_alert)
        saver.history_file = paths["history_file"]
        saver.visual_log_file = paths["visual_log_file"]
        saver.snapshot_file = paths["snapshot_file"]
        saver.archive_dir = paths["archive_dir"]
        saver.write_json = False
        saver.write_jsonl = True
        saver.write_visual_log = True
        saver.auto_archive_jsonl = False
        saver.max_jsonl_file_bytes = 10 * 1024 * 1024
        saver.max_visual_log_bytes = 10 * 1024 * 1024
        created.append(saver)
        return saver

    yield make_saver, paths

    for saver in created:
        saver.stop(timeout=5.0)


def test_default_is_off(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.delenv("SOUND_ALERT", raising=False)
    saver = make_saver(None)
    try:
        assert saver.sound_alert is False
    finally:
        saver.stop()


def test_env_opt_in(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.setenv("SOUND_ALERT", "1")
    saver = make_saver(None)
    try:
        assert saver.sound_alert is True
    finally:
        saver.stop()


def test_explicit_flag_wins_over_env(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.setenv("SOUND_ALERT", "1")
    saver = make_saver(False)
    try:
        assert saver.sound_alert is False
    finally:
        saver.stop()


@pytest.mark.parametrize("n_signals", [1, 5])
def test_beep_500ms_does_not_block_window(monkeypatch, isolated_saver_factory, n_signals):
    """Com beep simulado de 500ms, N sinais não custam ~500ms/sinal."""
    make_saver, _ = isolated_saver_factory
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = make_saver(True)
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


def test_audio_failure_never_breaks_save(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory

    def _boom(self):
        raise RuntimeError("sem dispositivo de áudio")

    monkeypatch.setattr(EventSaver, "_play_sound", _boom)
    saver = make_saver(True)
    try:
        saver.save_event(_signal())  # não deve levantar
        time.sleep(0.3)  # dá chance ao worker falhar em background
        saver.save_event(_signal(1))  # segue funcionando
    finally:
        saver.stop()


def test_queue_is_bounded_and_single_worker(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = make_saver(True)
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


def test_stop_kills_sound_thread(monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.setattr(EventSaver, "_play_sound", _slow_beep_500ms)
    saver = make_saver(True)
    saver.save_event(_signal())
    time.sleep(0.2)
    assert any(t.name == "eventsaver-sound" and t.is_alive()
               for t in threading.enumerate())
    saver.stop()
    assert not any(t.name == "eventsaver-sound" and t.is_alive()
                   for t in threading.enumerate())


def test_fallback_path_is_isolated(tmp_path, monkeypatch, isolated_saver_factory):
    make_saver, _ = isolated_saver_factory
    monkeypatch.setattr(EventSaver, "_play_sound", lambda self: None)

    def _force_fallback(self, event):
        self._save_fallback(event, "jsonl")
        return "FAILED"

    monkeypatch.setattr(EventSaver, "_save_to_jsonl", _force_fallback)
    saver = make_saver(False)
    try:
        saver.save_event(_signal(0))
    finally:
        saver.stop(timeout=5.0)

    fallback_files = list((tmp_path / "fallback_events").glob("eventos_*.jsonl"))
    assert len(fallback_files) == 1
    assert fallback_files[0].stat().st_size > 0
    assert fallback_files[0].resolve().is_relative_to(tmp_path.resolve())
    assert not (REPO_ROOT / "fallback_events" / fallback_files[0].name).exists()


def test_archive_path_is_isolated(tmp_path, monkeypatch, isolated_saver_factory):
    make_saver, paths = isolated_saver_factory
    saver = make_saver(False)
    try:
        saver.auto_archive_jsonl = True
        saver.max_jsonl_file_bytes = 1
        paths["history_file"].write_text("{}\n", encoding="utf-8")
        assert saver._rotate_jsonl_if_needed() is True

        archived = list(paths["archive_dir"].glob("eventos_fluxo_*.jsonl"))
        assert len(archived) == 1
        assert archived[0].resolve().is_relative_to(tmp_path.resolve())
        assert not paths["history_file"].exists()
        assert not (REPO_ROOT / "dados" / "archive").exists()
    finally:
        saver.stop(timeout=5.0)


def test_operational_paths_are_not_touched(tmp_path, monkeypatch, isolated_saver_factory):
    make_saver, paths = isolated_saver_factory

    operational_before = _tree_manifest(REPO_ROOT / "dados")
    fallback_before = _tree_manifest(REPO_ROOT / "fallback_events")
    memory_before = _tree_manifest(REPO_ROOT / "memory")
    logs_before = _tree_manifest(REPO_ROOT / "logs")
    threads_before = set(threading.enumerate())

    monkeypatch.setattr(EventSaver, "_play_sound", lambda self: None)
    saver = make_saver(True)
    try:
        saver.save_event(_signal(0))
    finally:
        saver.stop(timeout=5.0)

    assert saver.db is not None
    assert saver.db.db_path.resolve().is_relative_to(tmp_path.resolve())
    assert paths["db_path"].exists()
    with sqlite3.connect(str(paths["db_path"])) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1

    assert paths["history_file"].exists()
    assert len(paths["history_file"].read_text(encoding="utf-8").splitlines()) == 1
    assert paths["visual_log_file"].exists()
    assert paths["visual_log_file"].stat().st_size > 0
    assert paths["visual_log_file"].with_suffix(".lock").exists()
    assert not paths["snapshot_file"].exists()
    assert paths["archive_dir"].resolve().is_relative_to(tmp_path.resolve())
    assert Path("fallback_events").resolve().is_relative_to(tmp_path.resolve())

    for path in (
        paths["data_dir"],
        paths["history_file"],
        paths["visual_log_file"],
        paths["snapshot_file"],
        paths["archive_dir"],
    ):
        assert path.resolve().is_relative_to(tmp_path.resolve())

    assert operational_before == _tree_manifest(REPO_ROOT / "dados")
    assert fallback_before == _tree_manifest(REPO_ROOT / "fallback_events")
    assert memory_before == _tree_manifest(REPO_ROOT / "memory")
    assert logs_before == _tree_manifest(REPO_ROOT / "logs")
    assert not (REPO_ROOT / "dados" / "quarantine").exists()

    new_threads = set(threading.enumerate()) - threads_before
    assert not any(t.is_alive() for t in new_threads)
    assert not multiprocessing.active_children()
