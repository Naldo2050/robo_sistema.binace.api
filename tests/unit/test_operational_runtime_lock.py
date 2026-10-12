# -*- coding: utf-8 -*-
"""
Suíte de testes para o Lock Cooperativo Operacional (Gate G4-D B2 Hardened).

Valida exclusividade mútua de runtime, fail-closed sob concorrência,
retenção de trava durante shutdown incompleto (anti-GC), proteção do
finalizador __del__, liberação pelo SO após crash e verificação de writers.
Todos os testes rodam estritamente em tmp_path. Zero acesso a dados operacionais.
"""

import errno
import gc
import json
import logging
import os
import platform
import sys
import time
import subprocess
import threading
from pathlib import Path
import pytest

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    _RETAINED_LOCKS,
    _RETAINED_FILE_HANDLES,
)
from main import check_operational_writers_alive


def test_1_acquisition_and_release(tmp_path):
    """1. Aquisição e liberação limpas."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="test_bot")

    assert not lock.is_locked
    lock.acquire()
    assert lock.is_locked
    assert lock_file.exists()

    lock.release()
    assert not lock.is_locked
    # Arquivo persiste no disco após release
    assert lock_file.exists()


def test_2_concurrent_processes_blocked(tmp_path):
    """2. Dois processos concorrentes: o segundo processo falha imediatamente."""
    lock_file = tmp_path / ".operational_runtime.lock"

    child_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{Path(__file__).resolve().parent.parent.parent}')
from common.runtime_lock import OperationalRuntimeLock
lock = OperationalRuntimeLock(lock_path=r'{lock_file}', holder_name='proc_1')
lock.acquire()
print('LOCKED', flush=True)
time.sleep(5)
lock.release()
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "LOCKED"

    try:
        lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="proc_2")
        with pytest.raises(RuntimeLockError) as exc_info:
            lock2.acquire(timeout=0.0)

        assert "Conflito de exclusividade" in str(exc_info.value)
        assert not lock2.is_locked
    finally:
        proc.kill()
        proc.wait()


def test_3_acquisition_failure_fail_closed(tmp_path):
    """3. Falha de aquisição dispara erro imediato sem travar em espera."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock1 = OperationalRuntimeLock(lock_path=lock_file, holder_name="owner_1")
    lock1.acquire()

    lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="owner_2")
    start = time.monotonic()
    with pytest.raises(RuntimeLockError):
        lock2.acquire(timeout=0.1)
    duration = time.monotonic() - start

    assert duration < 0.5
    lock1.release()


def test_4_crash_releases_lock(tmp_path):
    """4. Subprocesso morre abruptamente (crash/kill); SO libera o lock para outro processo."""
    lock_file = tmp_path / ".operational_runtime.lock"

    child_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{Path(__file__).resolve().parent.parent.parent}')
from common.runtime_lock import OperationalRuntimeLock
lock = OperationalRuntimeLock(lock_path=r'{lock_file}', holder_name='doomed_proc')
lock.acquire()
print('ACQUIRED_BY_CHILD', flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "ACQUIRED_BY_CHILD"

    lock_parent = OperationalRuntimeLock(lock_path=lock_file, holder_name="parent")
    with pytest.raises(RuntimeLockError):
        lock_parent.acquire(timeout=0.0)

    # Crash abrupto via kill()
    proc.kill()
    proc.wait()

    lock_parent.acquire(timeout=1.0)
    assert lock_parent.is_locked
    lock_parent.release()


def test_5_context_manager(tmp_path):
    """5. Suporte estrito a context manager."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="ctx_test")

    with lock as active_lock:
        assert active_lock.is_locked
        assert lock.is_locked

    assert not lock.is_locked


def test_6_relative_and_absolute_paths(tmp_path, monkeypatch):
    """6. Caminhos relativos e absolutos resolvem para o mesmo arquivo canônico."""
    monkeypatch.chdir(tmp_path)
    rel_path = Path("dados_test/.operational_runtime.lock")
    abs_path = (tmp_path / "dados_test/.operational_runtime.lock").resolve()

    lock_rel = OperationalRuntimeLock(lock_path=rel_path, holder_name="rel_holder")
    lock_abs = OperationalRuntimeLock(lock_path=abs_path, holder_name="abs_holder")

    assert lock_rel.lock_path == lock_abs.lock_path

    lock_rel.acquire()
    try:
        with pytest.raises(RuntimeLockError):
            lock_abs.acquire(timeout=0.0)
    finally:
        lock_rel.release()


def test_7_lock_file_persistent(tmp_path):
    """7. Arquivo de lock não sofre unlink no release para evitar corridas."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file)

    lock.acquire()
    assert lock_file.exists()
    lock.release()
    assert lock_file.exists()

    lock.acquire()
    assert lock.is_locked
    lock.release()
    assert lock_file.exists()


def test_8_double_release_safe(tmp_path):
    """8. Liberação dupla não levanta exceção (idempotência segura)."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file)

    lock.acquire()
    lock.release()
    assert not lock.is_locked

    lock.release()
    assert not lock.is_locked


def test_9_permission_error_fail_closed(tmp_path):
    """9. Falha ao acessar diretório sem permissão dispara RuntimeLockError (fail-closed)."""
    fake_file = tmp_path / "fake_file.txt"
    fake_file.write_text("not a dir", encoding="utf-8")
    invalid_lock_path = fake_file / "subdir" / "lock.file"

    lock = OperationalRuntimeLock(lock_path=invalid_lock_path)
    with pytest.raises(RuntimeLockError) as exc_info:
        lock.acquire()

    assert "Falha ao criar diretório" in str(exc_info.value) or "Não foi possível abrir" in str(exc_info.value)


def test_10_lock_retained_during_shutdown(tmp_path):
    """10. Simulação de retenção de lock durante drenagem de shutdown."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="shutdown_sim")
    lock.acquire()

    draining = True
    assert lock.is_locked

    draining = False
    if not draining:
        lock.release()

    assert not lock.is_locked


def test_11_worker_active_retains_lock(tmp_path):
    """11. Se o worker de persistência permanecer ativo, o lock NÃO é liberado e aciona retain()."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="active_worker_bot")
    lock.acquire()

    class MockWorker:
        ident = 1001
        def is_alive(self):
            return True

    class MockEventSaver:
        _flush_thread = MockWorker()

    class MockBot:
        event_saver = MockEventSaver()

    bot = MockBot()

    has_writers, details = check_operational_writers_alive(bot)
    assert has_writers is True
    assert any("EventSaverFlushWorker" in d for d in details)

    if has_writers:
        lock.retain(reason="worker ativo")
    else:
        lock.release()

    assert lock.is_locked
    assert lock.is_retained


def test_12_startup_failure_fail_closed(tmp_path):
    """12. Se o lock falhar no boot, componentes operacionais não são instanciados."""
    lock_file = tmp_path / ".operational_runtime.lock"
    existing_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="instance_1")
    existing_lock.acquire()

    components_created = False
    bot_connected = False
    exit_code = 0

    incoming_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="instance_2")
    try:
        incoming_lock.acquire(timeout=0.0)
        components_created = True
        bot_connected = True
    except RuntimeLockError:
        exit_code = 2

    assert exit_code == 2
    assert not components_created
    assert not bot_connected

    existing_lock.release()


def test_13_windows_primitive(tmp_path):
    """13. Validação explícita da primitiva msvcrt sob plataforma Windows."""
    if sys.platform != "win32":
        pytest.skip("Teste exclusivo para Windows")

    import msvcrt
    lock_file = tmp_path / "win_test.lock"
    f1 = open(lock_file, "a+b")
    f1.seek(0)
    msvcrt.locking(f1.fileno(), msvcrt.LK_NBLCK, 1)

    f2 = open(lock_file, "a+b")
    f2.seek(0)
    with pytest.raises(OSError):
        msvcrt.locking(f2.fileno(), msvcrt.LK_NBLCK, 1)
    f2.close()

    f1.seek(0)
    msvcrt.locking(f1.fileno(), msvcrt.LK_UNLCK, 1)
    f1.close()


def test_14_linux_primitive_mock(tmp_path, monkeypatch):
    """14. Validação da primitiva Linux (fcntl.flock) via mock defensivo."""
    lock_file = tmp_path / "linux_test.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="linux_sim")

    monkeypatch.setattr(lock, "_system", "Linux")
    flock_calls = []

    class MockFcntl:
        LOCK_EX = 2
        LOCK_NB = 4
        LOCK_UN = 8

        @staticmethod
        def flock(fd, op):
            flock_calls.append(op)

    monkeypatch.setitem(sys.modules, "fcntl", MockFcntl)

    lock.acquire()
    assert MockFcntl.LOCK_EX | MockFcntl.LOCK_NB in flock_calls

    lock.release()
    assert MockFcntl.LOCK_UN in flock_calls


# ==============================================================================
# TESTES DO HARDENING G4-D B2: ANTI-GC, FINALIZADORES E WRITERS ABRANGENTES
# ==============================================================================

def test_15_retained_lock_survives_gc(tmp_path):
    """15. Lock retido sobrevive a gc.collect() devido à âncora global de módulo."""
    lock_file = tmp_path / ".operational_runtime.lock"

    def _scope_with_lock():
        inner_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="inner_scope")
        inner_lock.acquire()
        inner_lock.retain(reason="incomplete shutdown test")
        # Retorna apenas o caminho, descartando a variável local inner_lock
        return lock_file

    target_path = _scope_with_lock()

    # Força coleta de lixo completa no Python
    gc.collect()

    # Confirma que a âncora global manteve o objeto e o handle abertos
    assert any(str(lk.lock_path) == str(target_path) for lk in _RETAINED_LOCKS)
    assert len(_RETAINED_FILE_HANDLES) > 0

    # Segundo processo/instância tenta adquirir e continua bloqueado no SO!
    lock_competitor = OperationalRuntimeLock(lock_path=target_path, holder_name="competitor")
    with pytest.raises(RuntimeLockError):
        lock_competitor.acquire(timeout=0.0)


def test_16_del_does_not_release_retained_lock(tmp_path):
    """16. Invocação manual de __del__() não libera lock retido intencionalmente."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="del_test")
    lock.acquire()
    lock.retain(reason="testing __del__ protection")

    # Invocação manual do finalizador
    lock.__del__()

    # Continua travado e retido
    assert lock.is_locked
    assert lock.is_retained

    # Outro processo continua bloqueado
    lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="competitor_2")
    with pytest.raises(RuntimeLockError):
        lock2.acquire(timeout=0.0)


def test_17_release_refuses_when_retained(tmp_path):
    """17. Invocação de release() em lock retido é ignorada com aviso (não destrava)."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="refuse_release")
    lock.acquire()
    lock.retain(reason="safety lock")

    # Tentativa de release explícito
    lock.release()

    # Lock permanece ativo e não foi liberado
    assert lock.is_locked
    assert lock.is_retained

    # Concorrência continua bloqueada
    lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="should_fail")
    with pytest.raises(RuntimeLockError):
        lock2.acquire(timeout=0.0)


def test_18_check_operational_writers_alive_all_components(tmp_path):
    """18. Validação exaustiva de detecção de todos os writers em check_operational_writers_alive."""
    # Cenário A: bot é None -> sem writers
    has_w, details = check_operational_writers_alive(None)
    assert not has_w
    assert len(details) == 0

    # Cenário B: bot vazio -> sem writers
    class EmptyBot:
        pass
    has_w, details = check_operational_writers_alive(EmptyBot())
    assert not has_w

    # Cenário C: EventSaver com flush thread ativa
    class AliveWorker:
        ident = 1001
        def is_alive(self):
            return True

    class DeadWorker:
        def is_alive(self):
            return False

    class MockSaver:
        _flush_thread = AliveWorker()
        _cleanup_thread = DeadWorker()
        _sound_thread = DeadWorker()
        _write_buffer = []
        _events_in_flight = 0
        _shutdown_incomplete = False

    class BotWithFlushWorker:
        event_saver = MockSaver()

    has_w, details = check_operational_writers_alive(BotWithFlushWorker())
    assert has_w is True
    assert any("EventSaverFlushWorker" in d for d in details)

    # Cenário D: EventSaver com buffer pendente
    class MockSaverBuffer:
        _flush_thread = DeadWorker()
        _cleanup_thread = DeadWorker()
        _sound_thread = DeadWorker()
        _write_buffer = [{"id": "ev1"}]
        _events_in_flight = 0
        _shutdown_incomplete = False

    class BotWithBuffer:
        event_saver = MockSaverBuffer()

    has_w, details = check_operational_writers_alive(BotWithBuffer())
    assert has_w is True
    assert any("_write_buffer com 1 eventos" in d for d in details)

    # Cenário E: Raw trades file aberto
    mock_file = tmp_path / "raw.jsonl"
    f_open = open(mock_file, "a", encoding="utf-8")
    try:
        class BotWithRawFile:
            event_saver = None
            _raw_trades_file = f_open

        has_w, details = check_operational_writers_alive(BotWithRawFile())
        assert has_w is True
        assert any("_raw_trades_file ainda aberto" in d for d in details)
    finally:
        f_open.close()

    # Cenário F: AI Thread Pool com worker ativo
    class BotWithAIThread:
        event_saver = None
        _raw_trades_file = None
        ai_thread_pool = [AliveWorker()]

    has_w, details = check_operational_writers_alive(BotWithAIThread())
    assert has_w is True
    assert any("AIThreadPool" in d for d in details)

    # Cenário G: Async loop thread dedicado ativo
    class BotWithAsyncLoop:
        event_saver = None
        _raw_trades_file = None
        ai_thread_pool = []
        _async_loop_thread = AliveWorker()

    has_w, details = check_operational_writers_alive(BotWithAsyncLoop())
    assert has_w is True
    assert any("_async_loop_thread" in d for d in details)


def test_19_main_simulation_incomplete_shutdown_blocks_second_process(tmp_path):
    """19. Simulação completa: main() retorna com writer ativo, retain() é chamado e novo processo é bloqueado."""
    lock_file = tmp_path / ".operational_runtime.lock"

    child_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{Path(__file__).resolve().parent.parent.parent}')
from common.runtime_lock import OperationalRuntimeLock

# 1. Adquire lock
lock = OperationalRuntimeLock(lock_path=r'{lock_file}', holder_name='simulated_bot')
lock.acquire()

# 2. Simula shutdown incompleto com retenção explícita
lock.retain(reason='worker alive during simulated main exit')

# 3. Notifica processo pai
print('RETAINED_AND_ACTIVE', flush=True)

# 4. Mantém thread viva aguardando drain
time.sleep(10)
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "RETAINED_AND_ACTIVE"

    try:
        # Segundo processo independente tenta abrir o lock e falha no SO
        competitor_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="competitor_proc")
        with pytest.raises(RuntimeLockError):
            competitor_lock.acquire(timeout=0.0)
    finally:
        proc.kill()
        proc.wait()


def test_20_main_simulation_clean_shutdown_allows_second_process(tmp_path):
    """20. Simulação de shutdown limpo: sem writers ativos, release() é executado e segundo processo sucede."""
    lock_file = tmp_path / ".operational_runtime.lock"

    lock1 = OperationalRuntimeLock(lock_path=lock_file, holder_name="first_bot")
    lock1.acquire()

    # Simula check_operational_writers_alive retornando (False, [])
    has_writers = False
    if not has_writers:
        lock1.release()

    # Segundo processo consegue adquirir normalmente
    lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="second_bot")
    lock2.acquire(timeout=0.0)
    assert lock2.is_locked
    lock2.release()


def test_21_writer_finishes_after_retention_deadlock_free(tmp_path):
    """21. Writer conclui sem deadlocks mesmo se o lock estiver em estado retido."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="deadlock_test")
    lock.acquire()
    lock.retain(reason="testing concurrency safety")

    completed = threading.Event()

    def _worker_work():
        time.sleep(0.1)
        completed.set()

    th = threading.Thread(target=_worker_work)
    th.start()
    th.join(timeout=2.0)

    assert completed.is_set()
    assert not th.is_alive()
    assert lock.is_locked


# ==============================================================================
# TESTES DO GATE G4-D B3-A: ESTADOS PERSISTENTES E SINALIZAÇÃO DE INTEGRIDADE
# ==============================================================================

def test_22_state_running_on_startup(tmp_path):
    """22. Após aquisição do lock, o estado persistido é estritamente RUNNING."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="startup_bot")
    lock.acquire()

    assert lock.persisted_state == RuntimeState.RUNNING
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.RUNNING
    assert info["metadata"]["holder"] == "startup_bot"
    assert info["metadata"]["pid"] == os.getpid()

    lock.release()


def test_23_state_clean_after_clean_shutdown(tmp_path):
    """23. Após shutdown comprovadamente limpo e release(), o estado é CLEAN."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="clean_bot")
    lock.acquire()
    assert lock.persisted_state == RuntimeState.RUNNING

    lock.release()
    assert lock.persisted_state == RuntimeState.CLEAN
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.CLEAN
    assert info["metadata"]["holder"] == "clean_bot"


def test_24_state_dirty_after_incomplete_shutdown(tmp_path):
    """24. Após shutdown incompleto com retain(), o estado persistido é DIRTY."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="dirty_bot")
    lock.acquire()

    lock.retain(reason="EventSaverFlushWorker ainda ativo")
    assert lock.persisted_state == RuntimeState.DIRTY
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.DIRTY
    assert "EventSaverFlushWorker" in info["metadata"].get("reason", "")


def test_25_state_unknown_after_crash_without_cleanup(tmp_path):
    """25. Se o processo sofreu crash com status RUNNING mas sem trava no SO, detecta UNKNOWN."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Simula arquivo gravado por processo que crasheou
    stale_payload = {
        "status": "RUNNING",
        "holder": "crashed_process",
        "pid": 99999,
        "timestamp_utc": "2026-10-12T00:00:00+00:00",
        "host": "TEST_HOST",
    }
    lock_file.write_text(json.dumps(stale_payload), encoding="utf-8")

    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.UNKNOWN
    assert info["reason"] == "crash_detected_running_status_without_kernel_lock"


def test_26_state_unknown_missing_file_or_empty(tmp_path):
    """26. Arquivo inexistente ou com 0 bytes é classificado como UNKNOWN."""
    non_existent = tmp_path / "non_existent.lock"
    info = OperationalRuntimeLock.read_persisted_state(non_existent)
    assert info["status"] == RuntimeState.UNKNOWN
    assert info["reason"] == "missing_or_empty_lock_file"

    empty_file = tmp_path / "empty.lock"
    empty_file.touch()
    info_empty = OperationalRuntimeLock.read_persisted_state(empty_file)
    assert info_empty["status"] == RuntimeState.UNKNOWN
    assert info_empty["reason"] == "missing_or_empty_lock_file"


def test_27_state_unknown_invalid_or_truncated_json(tmp_path):
    """27. JSON corrompido ou truncado por encerramento abrupto é classificado como UNKNOWN."""
    lock_file = tmp_path / ".operational_runtime.lock"

    # Truncado no meio
    lock_file.write_text('{"status": "RUNN', encoding="utf-8")
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.UNKNOWN
    assert "invalid_or_truncated_json" in info["reason"]

    # Conteúdo não-JSON
    lock_file.write_text("CORRUPTED_NON_JSON_DATA_12345", encoding="utf-8")
    info2 = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info2["status"] == RuntimeState.UNKNOWN
    assert "invalid_or_truncated_json" in info2["reason"]


def test_28_enospc_failure_fail_closed(tmp_path, monkeypatch):
    """28. Falha de ENOSPC na gravação do estado aborta aquisição (fail-closed) e desfaz lock no SO."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="enospc_test")

    def mock_write_fail(self, state, reason=""):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(OperationalRuntimeLock, "_persist_state", mock_write_fail)

    with pytest.raises(RuntimeLockError) as exc_info:
        lock.acquire()

    assert "No space left on device" in str(exc_info.value) or "Falha ao persistir" in str(exc_info.value)
    assert not lock.is_locked
    assert lock._file_obj is None

    # Verifica que outro processo consegue adquirir em seguida (lock foi desfeito)
    monkeypatch.undo()
    lock2 = OperationalRuntimeLock(lock_path=lock_file, holder_name="successor")
    lock2.acquire(timeout=0.0)
    assert lock2.is_locked
    lock2.release()


def test_29_fsync_failure_fail_closed(tmp_path, monkeypatch):
    """29. Falha de fsync durante a persistência do estado é tratada como fail-closed."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="fsync_test")

    def mock_fsync_fail(fd):
        raise OSError(errno.EIO, "I/O error during fsync")

    monkeypatch.setattr(os, "fsync", mock_fsync_fail)

    with pytest.raises(RuntimeLockError) as exc_info:
        lock.acquire()

    assert not lock.is_locked
    assert lock._file_obj is None


def test_30_concurrent_processes_state_not_corrupted(tmp_path):
    """30. Processo concorrente bloqueado não sobrescreve nem corrompe o estado RUNNING do detentor."""
    lock_file = tmp_path / ".operational_runtime.lock"

    child_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{Path(__file__).resolve().parent.parent.parent}')
from common.runtime_lock import OperationalRuntimeLock
lock = OperationalRuntimeLock(lock_path=r'{lock_file}', holder_name='owner_proc')
lock.acquire()
print('OWNER_ACQUIRED', flush=True)
time.sleep(5)
lock.release()
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "OWNER_ACQUIRED"

    try:
        # Segundo processo tenta e falha
        competitor = OperationalRuntimeLock(lock_path=lock_file, holder_name="competitor_proc")
        with pytest.raises(RuntimeLockError):
            competitor.acquire(timeout=0.0)

        # O estado em disco continua RUNNING com o owner_proc
        info = OperationalRuntimeLock.read_persisted_state(lock_file)
        assert info["status"] == RuntimeState.RUNNING
        assert info["metadata"]["holder"] == "owner_proc"
    finally:
        proc.kill()
        proc.wait()


def test_31_state_does_not_replace_kernel_lock(tmp_path):
    """31. O estado persistente serve como integridade, mas a exclusividade depende estritamente do kernel."""
    lock_file = tmp_path / ".operational_runtime.lock"
    # Grava estado "CLEAN" no arquivo
    clean_payload = {
        "status": "CLEAN",
        "holder": "previous_proc",
        "pid": 1234,
        "timestamp_utc": "2026-10-12T00:00:00+00:00",
        "host": "TEST_HOST",
    }
    lock_file.write_text(json.dumps(clean_payload), encoding="utf-8")

    # Porém, um descritor independente trava fisicamente o byte 0 via SO
    f_holder = open(lock_file, "a+b")
    system = platform.system()
    if system == "Windows":
        import msvcrt
        f_holder.seek(0)
        msvcrt.locking(f_holder.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(f_holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        # Novo processo tenta adquirir. Mesmo com o arquivo dizendo "CLEAN", o SO bloqueia!
        incoming = OperationalRuntimeLock(lock_path=lock_file, holder_name="incoming_proc")
        with pytest.raises(RuntimeLockError):
            incoming.acquire(timeout=0.0)
    finally:
        if system == "Windows":
            import msvcrt
            f_holder.seek(0)
            msvcrt.locking(f_holder.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f_holder.fileno(), fcntl.LOCK_UN)
        f_holder.close()


def test_32_gc_and_retention_preserves_dirty_state(tmp_path):
    """32. Retenção anti-GC preserva a trava física no SO e o estado persistido DIRTY."""
    lock_file = tmp_path / ".operational_runtime.lock"

    def _retain_scope():
        inner_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="dirty_scope")
        inner_lock.acquire()
        inner_lock.retain(reason="testing dirty GC survival")
        return lock_file

    path = _retain_scope()
    gc.collect()

    # O lock continua retido no SO e o arquivo diz DIRTY
    info = OperationalRuntimeLock.read_persisted_state(path)
    assert info["status"] == RuntimeState.DIRTY
    assert "testing dirty GC survival" in info["reason"]

    # Concorrente é bloqueado
    competitor = OperationalRuntimeLock(lock_path=path, holder_name="competitor")
    with pytest.raises(RuntimeLockError):
        competitor.acquire(timeout=0.0)


def test_33_state_cannot_be_overwritten_without_lock(tmp_path):
    """33. Processo sem exclusividade do lock é proibido de sobrescrever o estado."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="unauthorized")

    # Sem chamar acquire()
    with pytest.raises(RuntimeLockError) as exc_info:
        lock.update_state(RuntimeState.CLEAN)

    assert "sem deter o lock exclusivo" in str(exc_info.value)


def test_34_recovery_after_crash(tmp_path):
    """34. Recuperação consistente após crash: novo processo adquire, atualiza RUNNING e encerra CLEAN."""
    lock_file = tmp_path / ".operational_runtime.lock"

    child_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r'{Path(__file__).resolve().parent.parent.parent}')
from common.runtime_lock import OperationalRuntimeLock
lock = OperationalRuntimeLock(lock_path=r'{lock_file}', holder_name='crash_victim')
lock.acquire()
print('CRASH_READY', flush=True)
time.sleep(30)
"""
    proc = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    assert line == "CRASH_READY"

    # Crash abrupto via kill()
    proc.kill()
    proc.wait()

    # Imediatamente após o crash, o arquivo dizia RUNNING, mas sem lock físico => UNKNOWN (crash detectado)
    pre_recovery_info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert pre_recovery_info["status"] == RuntimeState.UNKNOWN
    assert pre_recovery_info["reason"] == "crash_detected_running_status_without_kernel_lock"

    # Novo processo recupera com sucesso
    recovery_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="recovery_bot")
    recovery_lock.acquire(timeout=1.0)
    assert recovery_lock.persisted_state == RuntimeState.RUNNING

    recovery_lock.release()
    assert recovery_lock.persisted_state == RuntimeState.CLEAN


def test_35_windows_and_linux_primitives_with_state(tmp_path):
    """35. Validação de ciclo completo de estados com as primitivas nativas do SO."""
    lock_file = tmp_path / "lifecycle_primitive.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="primitive_bot")

    assert OperationalRuntimeLock.read_persisted_state(lock_file)["status"] == RuntimeState.UNKNOWN

    lock.acquire()
    assert lock.persisted_state == RuntimeState.RUNNING

    lock.update_state(RuntimeState.RUNNING, reason="checkpoint")
    assert lock.persisted_state == RuntimeState.RUNNING

    lock.release()
    assert lock.persisted_state == RuntimeState.CLEAN


def test_36_fsync_failure_in_release_unlocks_cleanly_not_clean_state(tmp_path, monkeypatch, caplog):
    """36. Falha de fsync em release(): erro é registrado, lock é liberado e estado não é CLEAN válido."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="fsync_release_bot")
    lock.acquire()
    assert lock.is_locked

    # Mock fsync para falhar exclusivamente após o acquire inicial
    def mock_fsync_fail(fd):
        raise OSError(errno.EIO, "Simulated I/O error during fsync on release")

    monkeypatch.setattr(os, "fsync", mock_fsync_fail)

    with caplog.at_level(logging.ERROR):
        # A chamada release() não deve explodir com exceção inesperada
        lock.release()

    # Confirma que erro crítico/erro foi registrado nos logs
    assert any("Erro ao persistir estado CLEAN durante release" in record.message for record in caplog.records)

    # Lock físico foi liberado no SO e descritor anulado
    assert not lock.is_locked
    assert lock._file_obj is None

    # O estado no arquivo NÃO é CLEAN válido (destravado no SO sem CLEAN persistido => UNKNOWN)
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] != RuntimeState.CLEAN
    assert info["status"] == RuntimeState.UNKNOWN


def test_37_fsync_failure_in_retain_keeps_kernel_lock_strong_reference(tmp_path, monkeypatch, caplog):
    """37. Falha de fsync em retain(): lock e referências fortes continuam retidos no SO e estado é UNKNOWN."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="fsync_retain_bot")
    lock.acquire()
    assert lock.is_locked

    # Mock os.fsync para falhar durante retain
    def mock_fsync_fail(fd):
        raise OSError(errno.EIO, "I/O error during fsync in retain")

    monkeypatch.setattr(os, "fsync", mock_fsync_fail)

    with caplog.at_level(logging.CRITICAL):
        lock.retain(reason="falha de persistência simulada")

    # Erro de gravação foi devidamente registrado em log
    assert any("Falha ao gravar estado DIRTY no lock" in record.message for record in caplog.records)

    # Lock físico e descritores permanecem retidos no SO e nas âncoras globais
    assert lock.is_locked
    assert lock.is_retained
    assert lock in _RETAINED_LOCKS
    assert lock._file_obj in _RETAINED_FILE_HANDLES

    # Outro processo não consegue adquirir exclusividade
    competitor = OperationalRuntimeLock(lock_path=lock_file, holder_name="competitor_bot")
    with pytest.raises(RuntimeLockError):
        competitor.acquire(timeout=0.0)

    # Estado em disco considerado UNKNOWN caso persistência falhe
    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.UNKNOWN


def test_38_concurrent_metadata_read_with_byte_zero_locked(tmp_path):
    """38. Segundo processo lê metadados a partir do byte 1 enquanto o byte 0 está travado no SO (Windows msvcrt)."""
    lock_file = tmp_path / ".operational_runtime.lock"
    owner_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="primary_owner")
    owner_lock.acquire()

    try:
        # Segundo processo abre o arquivo e lê metadados via _read_existing_metadata_safe sem PermissionError
        with open(lock_file, "r+b") as competitor_file:
            competitor_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="reader_process")
            meta = competitor_lock._read_existing_metadata_safe(competitor_file)

            # Confirma que os metadados do titular foram lidos com precisão
            assert meta is not None
            assert meta.get("holder") == "primary_owner"
            assert meta.get("status") == RuntimeState.RUNNING.value
            assert meta.get("pid") == os.getpid()

        # Confirma que read_persisted_state também lê com sucesso
        info = OperationalRuntimeLock.read_persisted_state(lock_file)
        assert info["status"] == RuntimeState.RUNNING
        assert info["metadata"]["holder"] == "primary_owner"

        # Confirma que o lock original permanece intacto e sem alteração
        assert owner_lock.is_locked
        assert owner_lock.persisted_state == RuntimeState.RUNNING
    finally:
        owner_lock.release()


def test_39_concurrent_metadata_read_invalid_data_unknown(tmp_path):
    """39. Leitura de metadados com payload corrompido não lança exceção e é classificada como UNKNOWN."""
    lock_file = tmp_path / ".operational_runtime.lock"
    lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="corrupt_meta_test")

    # Arquivo contendo byte 0 '#' e payload inválido
    with open(lock_file, "wb") as f:
        f.write(b"#NOT_VALID_JSON_BODY_12345")

    with open(lock_file, "r+b") as f:
        meta = lock._read_existing_metadata_safe(f)
        assert meta is None  # Não lança exceção não tratada, retorna None

    info = OperationalRuntimeLock.read_persisted_state(lock_file)
    assert info["status"] == RuntimeState.UNKNOWN
    assert "invalid_or_truncated_json" in info["reason"]
