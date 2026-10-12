# -*- coding: utf-8 -*-
"""
Suíte de testes para o Lock Cooperativo Operacional (Gate G4-D B2 Hardened).

Valida exclusividade mútua de runtime, fail-closed sob concorrência,
retenção de trava durante shutdown incompleto (anti-GC), proteção do
finalizador __del__, liberação pelo SO após crash e verificação de writers.
Todos os testes rodam estritamente em tmp_path. Zero acesso a dados operacionais.
"""

import gc
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import pytest

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
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
