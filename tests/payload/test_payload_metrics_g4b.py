# -*- coding: utf-8 -*-
"""
Suíte de Testes Herméticos — Gate G4-B
Validação da Rotação, Retenção, Module Lock Bounded e Eliminação de Cache Stale.

Cobre integralmente:
1-24: Cenários de append, rotação, backups, ordenação, fail-open, formatos e regressões.
25-38: Module lock bounded, orçamento compartilhado time.monotonic(), cache stale prevention,
       isolamento de paths/last_n e concorrência estrita.
"""

import errno
import json
import logging
from pathlib import Path
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from market_orchestrator.ai import payload_metrics_aggregator as pma
from market_orchestrator.ai.payload_metrics_aggregator import (
    append_metric_line,
    summarize_metrics,
    _TRIPWIRE_THRESHOLDS,
    MAX_METRICS_FILE_BYTES,
    MAX_METRICS_BACKUPS,
    LOCK_TIMEOUT_SECONDS,
    _rotate_metrics_file_if_needed,
    acquire_file_lock,
    release_file_lock,
    file_lock_context,
    acquire_combined_locks,
)

pytestmark = pytest.mark.payload


# ---------------------------------------------------------------------------
# 1. Append básico
# ---------------------------------------------------------------------------
def test_01_append_basico(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    entry = {"payload_bytes": 128, "section": "test_section"}
    append_metric_line(entry, metrics_path=str(p))

    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert content.endswith("\n")
    loaded = json.loads(content.strip())
    assert loaded["payload_bytes"] == 128

    summary = summarize_metrics(str(p), last_n=10)
    assert summary["count"] == 1
    assert summary["bytes_p50"] == 128


# ---------------------------------------------------------------------------
# 2. Rotação no limite
# ---------------------------------------------------------------------------
def test_02_rotacao_no_limite(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    monkeypatch.setattr(pma, "MAX_METRICS_FILE_BYTES", 50)

    append_metric_line({"payload_bytes": 10}, metrics_path=str(p))
    assert p.exists()
    assert not b1.exists()

    append_metric_line({"payload_bytes": 20, "extra": "x" * 40}, metrics_path=str(p))
    append_metric_line({"payload_bytes": 30}, metrics_path=str(p))

    assert b1.exists()
    assert p.exists()
    active_lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(active_lines) == 1
    assert json.loads(active_lines[0])["payload_bytes"] == 30


# ---------------------------------------------------------------------------
# 3. Dois backups (.1 e .2, sem .3)
# ---------------------------------------------------------------------------
def test_03_dois_backups(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    b2 = tmp_path / "payload_metrics.jsonl.2"
    b3 = tmp_path / "payload_metrics.jsonl.3"

    monkeypatch.setattr(pma, "MAX_METRICS_FILE_BYTES", 30)

    for i in range(10):
        append_metric_line({"id": i, "fill": "x" * 20}, metrics_path=str(p))

    assert p.exists()
    assert b1.exists()
    assert b2.exists()
    assert not b3.exists(), "Não deve criar .3 além do limite de 2 backups"


# ---------------------------------------------------------------------------
# 4. Ordem cronológica (.2 mais antigo -> .1 -> ativo mais recente)
# ---------------------------------------------------------------------------
def test_04_ordem_cronologica(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    b2 = tmp_path / "payload_metrics.jsonl.2"

    b2.write_text(json.dumps({"payload_bytes": 10}) + "\n" + json.dumps({"payload_bytes": 20}) + "\n", encoding="utf-8")
    b1.write_text(json.dumps({"payload_bytes": 30}) + "\n" + json.dumps({"payload_bytes": 40}) + "\n", encoding="utf-8")
    p.write_text(json.dumps({"payload_bytes": 50}) + "\n" + json.dumps({"payload_bytes": 60}) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=6)
    assert summary["count"] == 6
    assert summary["bytes_max"] == 60


# ---------------------------------------------------------------------------
# 5. Leitura complementar pós-rotação
# ---------------------------------------------------------------------------
def test_05_leitura_complementar_pos_rotacao(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"

    lines_b1 = [json.dumps({"payload_bytes": v}) for v in [100, 200, 300, 400, 500]]
    b1.write_text("\n".join(lines_b1) + "\n", encoding="utf-8")
    p.write_text(json.dumps({"payload_bytes": 600}) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=4)
    assert summary["count"] == 4
    assert summary["bytes_max"] == 600


# ---------------------------------------------------------------------------
# 6. Últimas 2000 amostras
# ---------------------------------------------------------------------------
def test_06_ultimas_2000_amostras(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"

    active_lines = [json.dumps({"payload_bytes": 200}) for _ in range(1200)]
    p.write_text("\n".join(active_lines) + "\n", encoding="utf-8")

    b1_lines = [json.dumps({"payload_bytes": 100}) for _ in range(1000)]
    b1.write_text("\n".join(b1_lines) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=2000)
    assert summary["count"] == 2000


# ---------------------------------------------------------------------------
# 7. Arquivo ativo suficiente não lê backups desnecessariamente
# ---------------------------------------------------------------------------
def test_07_arquivo_ativo_suficiente(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"

    active_lines = [json.dumps({"payload_bytes": 50}) for _ in range(2000)]
    p.write_text("\n".join(active_lines) + "\n", encoding="utf-8")

    b1.write_text(json.dumps({"payload_bytes": 9999}) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=2000)
    assert summary["count"] == 2000
    assert summary["bytes_max"] == 50


# ---------------------------------------------------------------------------
# 8. Arquivo vazio retorna {}
# ---------------------------------------------------------------------------
def test_08_arquivo_vazio(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text("", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=2000)
    assert summary == {}


# ---------------------------------------------------------------------------
# 9. Arquivo ausente retorna {}
# ---------------------------------------------------------------------------
def test_09_arquivo_ausente(tmp_path):
    p = tmp_path / "inexistente.jsonl"
    summary = summarize_metrics(str(p), last_n=2000)
    assert summary == {}


# ---------------------------------------------------------------------------
# 10. JSON inválido ignorado sem corromper denominador
# ---------------------------------------------------------------------------
def test_10_json_invalido(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    lines = [
        json.dumps({"payload_bytes": 100}),
        "{invalid json",
        "",
        json.dumps({"payload_bytes": 200, "leak_blocked": True}),
        "   ",
        json.dumps({"payload_bytes": 300}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=2000)
    assert summary["count"] == 3
    assert summary["guardrail_block_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 11. Falha intermediária de rename interrompe e preserva ativo
# ---------------------------------------------------------------------------
def test_11_falha_intermediaria_de_rename(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    p.write_text("initial_content\n", encoding="utf-8")
    b1.write_text("backup_content\n", encoding="utf-8")

    original_replace = pma.os.replace

    def mock_replace(src, dst):
        if "payload_metrics.jsonl.1" in str(src):
            raise OSError("Falha simulada no rename do backup .1")
        return original_replace(src, dst)

    monkeypatch.setattr(pma.os, "replace", mock_replace)

    success = _rotate_metrics_file_if_needed(p, max_bytes=1, backups=2)
    assert success is False
    assert p.read_text(encoding="utf-8") == "initial_content\n"


# ---------------------------------------------------------------------------
# 12. PermissionError Windows preserva arquivo ativo
# ---------------------------------------------------------------------------
def test_12_permission_error_windows(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text("content\n", encoding="utf-8")

    def mock_replace(src, dst):
        raise PermissionError("Arquivo bloqueado por outro processo")

    monkeypatch.setattr(pma.os, "replace", mock_replace)
    monkeypatch.setattr(pma, "MAX_METRICS_FILE_BYTES", 1)

    append_metric_line({"new_entry": True}, metrics_path=str(p))

    content = p.read_text(encoding="utf-8")
    assert "content" in content
    assert "new_entry" in content


# ---------------------------------------------------------------------------
# 13. Concorrência entre writers
# ---------------------------------------------------------------------------
def test_13_concorrencia_entre_writers(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    num_threads = 6
    writes_per_thread = 25
    barrier = threading.Barrier(num_threads)

    def worker(tid):
        barrier.wait()
        for i in range(writes_per_thread):
            append_metric_line({"tid": tid, "i": i, "payload_bytes": 50}, metrics_path=str(p))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = summarize_metrics(str(p), last_n=1000)
    assert summary["count"] == num_threads * writes_per_thread


# ---------------------------------------------------------------------------
# 14. Leitura durante rotação
# ---------------------------------------------------------------------------
def test_14_leitura_durante_rotacao(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text(json.dumps({"payload_bytes": 100}) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=10)
    assert summary["count"] == 1


# ---------------------------------------------------------------------------
# 15. Falha de write capturada (fail-open)
# ---------------------------------------------------------------------------
def test_15_falha_de_write(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"

    class BadFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write(self, *args):
            raise IOError("Disk write error")
        def flush(self):
            pass

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: BadFile())

    append_metric_line({"payload_bytes": 100}, metrics_path=str(p))


# ---------------------------------------------------------------------------
# 16. Falha de flush/close capturada
# ---------------------------------------------------------------------------
def test_16_falha_de_flush_close(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"

    class BadFlushFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            raise OSError("Close error")
        def write(self, data):
            pass
        def flush(self):
            raise OSError("Flush error")

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: BadFlushFile())

    append_metric_line({"payload_bytes": 100}, metrics_path=str(p))


# ---------------------------------------------------------------------------
# 17. Lock estável sem unlink
# ---------------------------------------------------------------------------
def test_17_lock_estavel_sem_unlink(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    lock_file = tmp_path / "payload_metrics.jsonl.lock"

    append_metric_line({"payload_bytes": 100}, metrics_path=str(p))
    assert lock_file.exists(), "O arquivo de lock deve ser criado"

    summarize_metrics(str(p), last_n=10)
    assert lock_file.exists(), "O arquivo de lock NÃO pode ser removido (sem unlink)"


# ---------------------------------------------------------------------------
# 18. Nenhuma alteração dos percentis
# ---------------------------------------------------------------------------
def test_18_nenhuma_alteracao_dos_percentis(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    lines = [{"payload_bytes": v} for v in [10, 20, 30, 40, 50]]
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=10)
    assert summary["bytes_p50"] == 30.0
    assert summary["bytes_max"] == 50.0


# ---------------------------------------------------------------------------
# 19. Nenhuma alteração das taxas
# ---------------------------------------------------------------------------
def test_19_nenhuma_alteracao_das_taxas(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    records = [
        {"payload_bytes": 100, "leak_blocked": True},
        {"payload_bytes": 100, "error": "no_safe_candidate"},
        {"payload_bytes": 100, "fallback_v1": True},
        {"payload_bytes": 100, "cache_hit": True, "section": "s1"},
        {"payload_bytes": 100, "cache_hit": False, "section": "s1"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=10)
    assert summary["count"] == 5
    assert summary["guardrail_block_rate"] == pytest.approx(1 / 5)
    assert summary["abort_rate"] == pytest.approx(1 / 5)
    assert summary["fallback_rate"] == pytest.approx(1 / 5)
    assert summary["cache_hit_rate"]["s1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 20. Nenhuma alteração de thresholds
# ---------------------------------------------------------------------------
def test_20_nenhuma_alteracao_de_thresholds():
    assert _TRIPWIRE_THRESHOLDS["fallback_rate"] == 0.10
    assert _TRIPWIRE_THRESHOLDS["abort_rate"] == 0.12
    assert _TRIPWIRE_THRESHOLDS["guardrail_block_rate"] == 0.20


# ---------------------------------------------------------------------------
# 21. Fail-open sob ENOSPC simulado
# ---------------------------------------------------------------------------
def test_21_fail_open_sob_enospc_simulado(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"

    class ENOSPCFile:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def write(self, data):
            err = OSError("No space left on device")
            err.errno = errno.ENOSPC
            raise err
        def flush(self):
            pass

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: ENOSPCFile())

    append_metric_line({"payload_bytes": 100}, metrics_path=str(p))


# ---------------------------------------------------------------------------
# 22. Sem alteração de formato JSONL (UTF-8, delimitado por \n)
# ---------------------------------------------------------------------------
def test_22_sem_alteracao_de_formato_jsonl(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    entry = {"utf8_test": "áéíóú çãõ", "number": 123.45, "flag": True}
    append_metric_line(entry, metrics_path=str(p))

    raw = p.read_bytes()
    assert b"\\u" not in raw
    assert raw.endswith(b"\n")


# ---------------------------------------------------------------------------
# 23. Sem registros duplicados
# ---------------------------------------------------------------------------
def test_23_sem_registros_duplicados(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    b2 = tmp_path / "payload_metrics.jsonl.2"

    b2.write_text("\n".join(json.dumps({"id": i, "payload_bytes": 10}) for i in range(1, 6)) + "\n", encoding="utf-8")
    b1.write_text("\n".join(json.dumps({"id": i, "payload_bytes": 10}) for i in range(6, 11)) + "\n", encoding="utf-8")
    p.write_text("\n".join(json.dumps({"id": i, "payload_bytes": 10}) for i in range(11, 16)) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=15)
    assert summary["count"] == 15


# ---------------------------------------------------------------------------
# 24. Sem vazamento de threads
# ---------------------------------------------------------------------------
def test_24_sem_vazamento_de_threads(tmp_path):
    threads_before = threading.active_count()

    p = tmp_path / "payload_metrics.jsonl"
    for i in range(10):
        append_metric_line({"payload_bytes": i}, metrics_path=str(p))
    summarize_metrics(str(p), last_n=10)

    threads_after = threading.active_count()
    assert threads_after == threads_before


# ===========================================================================
# TESTES G4-B — MODULE LOCK BOUNDED, DEADLINE E CACHE FAIL-CLOSED (25 a 38)
# ===========================================================================

# ---------------------------------------------------------------------------
# 25. _MODULE_LOCK ocupado no writer
# ---------------------------------------------------------------------------
def test_25_module_lock_ocupado_no_writer(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"

    pma._MODULE_LOCK.acquire()
    try:
        t0 = time.monotonic()
        append_metric_line({"payload_bytes": 100}, metrics_path=str(p))
        duration = time.monotonic() - t0
        assert duration < 1.0
        assert not p.exists(), "Não deve criar arquivo se o _MODULE_LOCK não for obtido"
    finally:
        pma._MODULE_LOCK.release()


# ---------------------------------------------------------------------------
# 26. _MODULE_LOCK ocupado no reader
# ---------------------------------------------------------------------------
def test_26_module_lock_ocupado_no_reader(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text(json.dumps({"payload_bytes": 100}) + "\n", encoding="utf-8")

    pma._CACHE_SUMMARY = None
    pma._CACHE_KEY = None

    pma._MODULE_LOCK.acquire()
    try:
        t0 = time.monotonic()
        summary = summarize_metrics(str(p), last_n=10)
        duration = time.monotonic() - t0
        assert duration < 1.0
        assert summary == {}
    finally:
        pma._MODULE_LOCK.release()


# ---------------------------------------------------------------------------
# 27. Timeout bounded no module lock
# ---------------------------------------------------------------------------
def test_27_timeout_bounded_no_module_lock(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    lock_path = Path(f"{p}.lock")

    pma._MODULE_LOCK.acquire()
    try:
        t0 = time.monotonic()
        with acquire_combined_locks(lock_path, timeout=0.1) as locked:
            assert locked is False
        duration = time.monotonic() - t0
        assert 0.08 <= duration <= 0.5
    finally:
        pma._MODULE_LOCK.release()


# ---------------------------------------------------------------------------
# 28. Orçamento compartilhado entre module lock e file lock
# ---------------------------------------------------------------------------
def test_28_orcamento_compartilhado_time_monotonic(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    lock_path = Path(f"{p}.lock")

    recorded_timeouts = []

    def mock_acquire_file_lock(file_obj, blocking=True, timeout=0.2):
        recorded_timeouts.append(timeout)
        return False

    monkeypatch.setattr(pma, "acquire_file_lock", mock_acquire_file_lock)

    t0 = time.monotonic()
    with acquire_combined_locks(lock_path, timeout=0.2) as locked:
        assert locked is False
    total_duration = time.monotonic() - t0

    assert len(recorded_timeouts) == 1
    # Tolerância de precisão temporal (<= 0.205s)
    assert recorded_timeouts[0] <= 0.205
    assert total_duration < 1.0


# ---------------------------------------------------------------------------
# 29. Nenhum I/O quando module lock não é adquirido
# ---------------------------------------------------------------------------
def test_29_nenhum_io_quando_module_lock_nao_adquirido(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text("initial\n", encoding="utf-8")
    initial_stat = p.stat()

    fake_lock = MagicMock()
    fake_lock.acquire.return_value = False
    monkeypatch.setattr(pma, "_MODULE_LOCK", fake_lock)

    append_metric_line({"payload_bytes": 999}, metrics_path=str(p))

    assert p.stat().st_size == initial_stat.st_size
    assert p.stat().st_mtime == initial_stat.st_mtime


# ---------------------------------------------------------------------------
# 30. Nenhum I/O quando file lock não é adquirido
# ---------------------------------------------------------------------------
def test_30_nenhum_io_quando_file_lock_nao_adquirido(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text("initial\n", encoding="utf-8")
    initial_stat = p.stat()

    monkeypatch.setattr(pma, "acquire_file_lock", lambda *args, **kwargs: False)

    append_metric_line({"payload_bytes": 999}, metrics_path=str(p))

    assert p.stat().st_size == initial_stat.st_size
    assert p.stat().st_mtime == initial_stat.st_mtime


# ---------------------------------------------------------------------------
# 31. Cache stale NÃO é reutilizado após falha de lock (fail-closed com {})
# ---------------------------------------------------------------------------
def test_31_cache_stale_nao_reutilizado_apos_falha_de_lock(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text(json.dumps({"payload_bytes": 100}) + "\n", encoding="utf-8")

    summary1 = summarize_metrics(str(p), last_n=10)
    assert summary1["count"] == 1

    p.write_text(json.dumps({"payload_bytes": 200}) + "\n" + json.dumps({"payload_bytes": 300}) + "\n", encoding="utf-8")

    monkeypatch.setattr(pma, "acquire_file_lock", lambda *args, **kwargs: False)

    summary2 = summarize_metrics(str(p), last_n=10)
    assert summary2 == {}, "Deve retornar {} e nunca dados obsoletos sob falha de lock"


# ---------------------------------------------------------------------------
# 32. Cache válido em operação normal (metadados idênticos) funciona
# ---------------------------------------------------------------------------
def test_32_cache_valido_operacao_normal(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text(json.dumps({"payload_bytes": 100}) + "\n", encoding="utf-8")

    summary1 = summarize_metrics(str(p), last_n=10)
    assert summary1["count"] == 1

    spy_open = MagicMock(side_effect=p.open)
    monkeypatch.setattr(Path, "open", spy_open)

    summary2 = summarize_metrics(str(p), last_n=10)
    assert summary2["count"] == 1
    assert spy_open.call_count == 0


# ---------------------------------------------------------------------------
# 33. Path diferente NÃO reutiliza cache
# ---------------------------------------------------------------------------
def test_33_path_diferente_nao_reutiliza_cache(tmp_path):
    p1 = tmp_path / "p1.jsonl"
    p2 = tmp_path / "p2.jsonl"

    p1.write_text(json.dumps({"payload_bytes": 10}) + "\n", encoding="utf-8")
    p2.write_text(json.dumps({"payload_bytes": 20}) + "\n", encoding="utf-8")

    s1 = summarize_metrics(str(p1), last_n=10)
    s2 = summarize_metrics(str(p2), last_n=10)

    assert s1["bytes_p50"] == 10
    assert s2["bytes_p50"] == 20


# ---------------------------------------------------------------------------
# 34. last_n diferente NÃO reutiliza cache
# ---------------------------------------------------------------------------
def test_34_last_n_diferente_nao_reutiliza_cache(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    lines = [json.dumps({"payload_bytes": i * 10}) for i in range(1, 11)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    s_all = summarize_metrics(str(p), last_n=10)
    assert s_all["count"] == 10

    s_few = summarize_metrics(str(p), last_n=3)
    assert s_few["count"] == 3


# ---------------------------------------------------------------------------
# 35. Release correto após exception (sem travar locks para chamadas futuras)
# ---------------------------------------------------------------------------
def test_35_release_correto_apos_exception(tmp_path, monkeypatch):
    p = tmp_path / "payload_metrics.jsonl"
    p.write_text(json.dumps({"payload_bytes": 100}) + "\n", encoding="utf-8")

    def broken_read(*args, **kwargs):
        raise RuntimeError("Erro inesperado durante leitura de disco")

    with patch.object(Path, "open", broken_read):
        summary = summarize_metrics(str(p), last_n=10)
        assert summary == {}

    append_metric_line({"payload_bytes": 200}, metrics_path=str(p))
    summary_ok = summarize_metrics(str(p), last_n=10)
    assert summary_ok["count"] == 2


# ---------------------------------------------------------------------------
# 36. Sem deadlock sob concorrência intensa
# ---------------------------------------------------------------------------
def test_36_sem_deadlock_sob_concorrencia(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    num_threads = 8
    ops_per_thread = 15
    barrier = threading.Barrier(num_threads)

    def worker(tid):
        barrier.wait()
        for i in range(ops_per_thread):
            if tid % 2 == 0:
                append_metric_line({"tid": tid, "i": i, "payload_bytes": 50}, metrics_path=str(p))
            else:
                summarize_metrics(str(p), last_n=50)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Thread bloqueada (possível deadlock)"


# ---------------------------------------------------------------------------
# 37. Sem regressão de last_n=2000 atravessando backups
# ---------------------------------------------------------------------------
def test_37_sem_regressao_last_n_2000(tmp_path):
    p = tmp_path / "payload_metrics.jsonl"
    b1 = tmp_path / "payload_metrics.jsonl.1"
    b2 = tmp_path / "payload_metrics.jsonl.2"

    b2.write_text("\n".join(json.dumps({"payload_bytes": 10}) for _ in range(500)) + "\n", encoding="utf-8")
    b1.write_text("\n".join(json.dumps({"payload_bytes": 20}) for _ in range(1000)) + "\n", encoding="utf-8")
    p.write_text("\n".join(json.dumps({"payload_bytes": 30}) for _ in range(800)) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(p), last_n=2000)
    assert summary["count"] == 2000
    assert summary["bytes_p50"] == 20.0


# ---------------------------------------------------------------------------
# 38. Sem regressão dos tripwires
# ---------------------------------------------------------------------------
def test_38_sem_regressao_dos_tripwires():
    assert _TRIPWIRE_THRESHOLDS["fallback_rate"] == 0.10
    assert _TRIPWIRE_THRESHOLDS["abort_rate"] == 0.12
    assert _TRIPWIRE_THRESHOLDS["guardrail_block_rate"] == 0.20
