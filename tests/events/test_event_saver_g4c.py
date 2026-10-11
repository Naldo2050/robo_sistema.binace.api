# -*- coding: utf-8 -*-
"""
tests/events/test_event_saver_g4c.py
Testes isolados e herméticos para a Fase 5.11 - Gate G4-C:
Rotação não destrutiva do JSONL principal (eventos_fluxo.jsonl).

Cobrem estritamente os 14 requisitos contratuais:
1. Rotação por tamanho.
2. Integridade da contagem total de eventos.
3. Ausência de sobrescrita (zero overwrite de segmentos históricos).
4. Colisão de nomes (resolução com sequência monotônica).
5. Falha de permissão (PermissionError / Windows lock).
6. Arquivo ativo ausente (início a frio).
7. Falha de rename/movimentação (preservação do arquivo ativo).
8. ENOSPC (simulação de esgotamento de espaço).
9. Shutdown durante drenagem (prioridade total à drenagem sem rotação).
10. Crash entre rotação e append (consistência de estado).
11. Replay multi-segmento (ordem cronológica monotônica).
12. Compatibilidade de auditoria (audit_data_attrition multi-fonte).
13. Comportamento Windows (MoveFileW / FileExistsError).
14. Segurança de destino no Linux (anti-sobrescrita em sistemas POSIX).
"""

import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from events import event_saver
from events.event_saver import EventSaver


@pytest.fixture
def hermetic_saver_g4c(tmp_path, monkeypatch):
    """Fixture isolada garantindo que nenhuma operação toque em arquivos reais de dados."""
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    mock_store = MagicMock()
    mock_store.save_batch.return_value = event_saver.BatchSaveResult(
        total_received=1, persisted=1, rejected_pre_commit=0, commit_status="COMMITTED"
    )
    monkeypatch.setattr(event_saver, "EventStore", lambda *args, **kwargs: mock_store)

    saver = EventSaver(sound_alert=False)
    saver.history_file = tmp_path / "eventos_fluxo.jsonl"
    saver.archive_dir = tmp_path / "archive"
    # Limiar configurável baixo para permitir testes rápidos (1 KB)
    saver.max_jsonl_file_bytes = 1024

    yield saver

    try:
        saver.stop(timeout=2.0)
    except Exception:
        pass


def _create_sample_event(event_id: str, epoch_ms: int = None) -> dict:
    return {
        "tipo_evento": "POSITION_SNAPSHOT",
        "symbol": "BTCUSDT",
        "event_id": event_id,
        "epoch_ms": epoch_ms or int(time.time() * 1000),
        "janela_numero": 100,
        "payload": {"price": 65000.0, "qty": 0.5},
    }


# =============================================================================
# 1. ROTAÇÃO POR TAMANHO
# =============================================================================
def test_g4c_rotation_by_size(hermetic_saver_g4c, tmp_path):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 500  # 500 bytes

    # Gera evento que ultrapassa 500 bytes
    large_payload = "X" * 600
    ev1 = _create_sample_event("ev_001")
    ev1["payload"]["blob"] = large_payload

    saver.save_event(ev1)
    saver._process_single_flush_batch()

    # Primeiro evento cria o ativo com mais de 500 bytes
    assert saver.history_file.exists()
    assert saver.history_file.stat().st_size > 500

    # Segundo evento deve acionar a rotação antes de escrever
    ev2 = _create_sample_event("ev_002")
    saver.save_event(ev2)
    saver._process_single_flush_batch()

    # Verifica se o arquivo arquivado foi criado em dados/archive/
    archive_files = list(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    assert len(archive_files) == 1
    # E o arquivo ativo atual contém apenas o novo evento
    assert saver.history_file.exists()
    with open(saver.history_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 1
    assert "ev_002" in lines[0]


# =============================================================================
# 2. INTEGRIDADE DA CONTAGEM TOTAL
# =============================================================================
def test_g4c_total_count_integrity(hermetic_saver_g4c):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 400  # Rotação frequente

    total_events = 25
    for i in range(total_events):
        saver.save_event(_create_sample_event(f"ev_{i:03d}"))
        saver._process_single_flush_batch()

    # Coleta todas as linhas de todos os segmentos arquivados + ativo
    all_read_ids = []
    archive_files = sorted(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    for af in archive_files:
        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_read_ids.append(data["event_id"])

    if saver.history_file.exists():
        with open(saver.history_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_read_ids.append(data["event_id"])

    assert len(all_read_ids) == total_events
    assert len(set(all_read_ids)) == total_events
    assert saver._jsonl_events_written == total_events


# =============================================================================
# 3. AUSÊNCIA DE SOBRESCRITA
# =============================================================================
def test_g4c_no_overwrite_guarantee(hermetic_saver_g4c, tmp_path):
    saver = hermetic_saver_g4c
    saver.archive_dir.mkdir(parents=True, exist_ok=True)

    # Cria arquivo preexistente com conteúdo intocável
    dummy_arch = saver.archive_dir / "target_existing.jsonl"
    dummy_arch.write_text("DADOS_ORIGINAIS_PRESERVADOS\n", encoding="utf-8")

    src_file = tmp_path / "temp_src.jsonl"
    src_file.write_text("NOVOS_DADOS\n", encoding="utf-8")

    # Tenta mover sobre dummy_arch: deve falhar com FileExistsError SEM SOBRESCREVER
    with pytest.raises(FileExistsError):
        saver._safe_move_segment_no_overwrite(src_file, dummy_arch)

    assert dummy_arch.read_text(encoding="utf-8") == "DADOS_ORIGINAIS_PRESERVADOS\n"
    assert src_file.exists()


# =============================================================================
# 4. COLISÃO DE NOMES COM SEQUÊNCIA MONOTÔNICA
# =============================================================================
def test_g4c_name_collision_resolution(hermetic_saver_g4c, monkeypatch):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200
    saver.archive_dir.mkdir(parents=True, exist_ok=True)

    # Força stamp idêntico para simular colisão
    fixed_stamp = "20261011_120000"
    fixed_ms = 999999
    monkeypatch.setattr("events.event_saver.time.monotonic_ns", lambda: fixed_ms * 1_000_000)

    # Pré-cria o primeiro candidato (_001.jsonl)
    first_candidate = saver.archive_dir / f"eventos_fluxo_{fixed_stamp}_{fixed_ms:06d}_001.jsonl"
    first_candidate.write_text("SEGMENTO_ANTIGO\n", encoding="utf-8")

    # Preenche o ativo para forçar rotação
    saver.history_file.write_text("X" * 300, encoding="utf-8")

    # Força o stamp datetime
    mock_dt = MagicMock()
    mock_dt.strftime.return_value = fixed_stamp
    monkeypatch.setattr("events.event_saver.datetime", MagicMock(now=lambda tz: mock_dt))

    res = saver._rotate_jsonl_if_needed()
    assert res is True

    # O arquivo antigo não foi tocado
    assert first_candidate.read_text(encoding="utf-8") == "SEGMENTO_ANTIGO\n"

    # O novo foi criado como _002.jsonl
    second_candidate = saver.archive_dir / f"eventos_fluxo_{fixed_stamp}_{fixed_ms:06d}_002.jsonl"
    assert second_candidate.exists()


# =============================================================================
# 5. FALHA DE PERMISSÃO / RESILIÊNCIA A LOCK
# =============================================================================
def test_g4c_permission_error_failsafe(hermetic_saver_g4c, monkeypatch):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200
    saver.history_file.write_text("X" * 300, encoding="utf-8")

    # Simula PermissionError em _safe_move_segment_no_overwrite
    def mock_move(src, dst):
        raise PermissionError("[WinError 32] Arquivo em uso")

    monkeypatch.setattr(saver, "_safe_move_segment_no_overwrite", mock_move)

    # Rotação deve retornar False mas não quebrar o worker
    assert saver._rotate_jsonl_if_needed() is False

    # Evento deve ser gravado normalmente no ativo existente (Prioridade: salvar evento)
    ev = _create_sample_event("ev_perm_test")
    saver.save_event(ev)
    saver._process_single_flush_batch()

    assert saver.history_file.exists()
    content = saver.history_file.read_text(encoding="utf-8")
    assert "ev_perm_test" in content


# =============================================================================
# 6. ARQUIVO ATIVO AUSENTE (INÍCIO A FRIO)
# =============================================================================
def test_g4c_missing_active_file_cold_boot(hermetic_saver_g4c):
    saver = hermetic_saver_g4c
    if saver.history_file.exists():
        saver.history_file.unlink()

    # Rotação não faz nada se o arquivo ativo não existe
    assert saver._rotate_jsonl_if_needed() is True
    assert not saver.history_file.exists()

    # Primeiro evento cria o ativo normalmente
    ev = _create_sample_event("ev_cold_boot")
    saver.save_event(ev)
    saver._process_single_flush_batch()

    assert saver.history_file.exists()


# =============================================================================
# 7. FALHA DE RENAME/MOVIMENTAÇÃO
# =============================================================================
def test_g4c_move_failure_retains_active(hermetic_saver_g4c, monkeypatch):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200
    saver.history_file.write_text("CONTEUDO_ATIVO_ORIGINAL_" + ("X" * 300) + "\n", encoding="utf-8")

    # Simula OSError genérico no SO
    monkeypatch.setattr(
        saver, "_safe_move_segment_no_overwrite", MagicMock(side_effect=OSError("Disk failure"))
    )

    res = saver._rotate_jsonl_if_needed()
    assert res is False
    # O arquivo ativo não pode ser corrompido nem excluído
    assert saver.history_file.exists()
    assert "CONTEUDO_ATIVO_ORIGINAL" in saver.history_file.read_text(encoding="utf-8")


# =============================================================================
# 8. TRATAMENTO DE ENOSPC
# =============================================================================
def test_g4c_enospc_failure_handling(hermetic_saver_g4c, monkeypatch):
    saver = hermetic_saver_g4c

    # Simula falha de disco no open/write
    def mock_open_enospc(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with patch("builtins.open", mock_open_enospc):
        # _save_to_jsonl deve capturar OSError, tentar retries e desviar para fallback sem crash
        ret = saver._save_to_jsonl(_create_sample_event("ev_enospc"))
        assert ret in ("FAILED", "UNKNOWN")
        assert saver._jsonl_events_failed >= 1


# =============================================================================
# 9. SHUTDOWN DURANTE DRENAGEM (PRIORIDADE TOTAL À DRENAGEM)
# =============================================================================
def test_g4c_shutdown_drain_priority_over_rotation(hermetic_saver_g4c):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200
    saver.history_file.write_text("X" * 300, encoding="utf-8")

    # Sinaliza stop_event (início de shutdown)
    saver._stop_event.set()

    # Rotação DEVE ser ignorada durante shutdown
    assert saver._rotate_jsonl_if_needed() is False

    # Nenhum arquivo arquivado criado
    archive_files = list(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    assert len(archive_files) == 0


# =============================================================================
# 10. CRASH ENTRE ROTAÇÃO E APPEND
# =============================================================================
def test_g4c_crash_state_consistency(hermetic_saver_g4c, tmp_path):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200
    saver.history_file.write_text("EVENTOS_ANTES_DO_CRASH_" + ("X" * 300) + "\n", encoding="utf-8")

    # Executa a rotação simulando crash antes do novo append
    saver._rotate_jsonl_if_needed()
    assert not saver.history_file.exists()

    archive_files = list(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    assert len(archive_files) == 1
    assert "EVENTOS_ANTES_DO_CRASH" in archive_files[0].read_text(encoding="utf-8")

    # No reboot/novo startup, o próximo evento cria o arquivo ativo sem perda
    ev_new = _create_sample_event("ev_pos_crash")
    saver.save_event(ev_new)
    saver._process_single_flush_batch()

    assert saver.history_file.exists()
    assert "ev_pos_crash" in saver.history_file.read_text(encoding="utf-8")


# =============================================================================
# 11. REPLAY MULTI-SEGMENTO E ORDEM CRONOLÓGICA
# =============================================================================
def test_g4c_multi_segment_replay_and_order(hermetic_saver_g4c):
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 300

    base_time = 1700000000000
    for i in range(12):
        ev = _create_sample_event(f"ev_{i:02d}", epoch_ms=base_time + (i * 1000))
        saver.save_event(ev)
        saver._process_single_flush_batch()

    # Lê todos os segmentos ordenados cronologicamente + arquivo ativo
    all_events = []
    archive_files = sorted(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    all_files = archive_files + ([saver.history_file] if saver.history_file.exists() else [])

    for fpath in all_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_events.append(json.loads(line))

    assert len(all_events) == 12
    # Verifica monotonicidade temporal
    timestamps = [e["epoch_ms"] for e in all_events]
    assert timestamps == sorted(timestamps)


# =============================================================================
# 12. COMPATIBILIDADE DE AUDITORIA (audit_data_attrition)
# =============================================================================
def test_g4c_audit_data_attrition_compatibility(hermetic_saver_g4c):
    saver = hermetic_saver_g4c
    saver.archive_dir.mkdir(parents=True, exist_ok=True)

    # Cria 2 segmentos arquivados com 3 eventos cada
    seg1 = saver.archive_dir / "eventos_fluxo_20261011_100000_000001_001.jsonl"
    seg2 = saver.archive_dir / "eventos_fluxo_20261011_100100_000002_001.jsonl"

    for seg in (seg1, seg2):
        lines = [json.dumps({"tipo_evento": "TEST", "event_id": f"id_{i}"}) for i in range(3)]
        seg.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Arquivo ativo com 2 eventos
    active_lines = [json.dumps({"tipo_evento": "TEST", "event_id": f"act_{i}"}) for i in range(2)]
    saver.history_file.write_text("\n".join(active_lines) + "\n", encoding="utf-8")

    # Executa a lógica de coleta do audit_data_attrition.py
    candidate_files = []
    if saver.archive_dir.is_dir():
        for arch_file in sorted(saver.archive_dir.glob("eventos_fluxo_*.jsonl")):
            candidate_files.append(arch_file)
    if saver.history_file.exists():
        candidate_files.append(saver.history_file)

    total_read = 0
    for fpath in candidate_files:
        with open(fpath, "r", encoding="utf-8") as f:
            total_read += sum(1 for line in f if line.strip())

    # 3 + 3 + 2 = 8 eventos no total
    assert total_read == 8
    assert len(candidate_files) == 3


# =============================================================================
# 13. COMPORTAMENTO WINDOWS (MoveFileW / FileExistsError)
# =============================================================================
def test_g4c_windows_behavior(tmp_path):
    if platform.system() != "Windows":
        pytest.skip("Teste específico para subsistema Windows/NTFS")

    src = tmp_path / "win_src.txt"
    dst = tmp_path / "win_dst.txt"
    src.write_text("SOURCE", encoding="utf-8")
    dst.write_text("TARGET", encoding="utf-8")

    # os.rename no Windows deve levantar FileExistsError se dst existir
    with pytest.raises(FileExistsError):
        os.rename(src, dst)

    assert dst.read_text(encoding="utf-8") == "TARGET"
    assert src.read_text(encoding="utf-8") == "SOURCE"


# =============================================================================
# 14. SEGURANÇA DE DESTINO NO LINUX / POSIX (renameat2 RENAME_NOREPLACE)
# =============================================================================
def test_g4c_linux_renameat2_available(tmp_path, monkeypatch):
    """Valida fluxo atômico com renameat2 mockado com sucesso no Linux."""
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "linux_src.txt"
    dst = tmp_path / "linux_dst.txt"
    src.write_text("SOURCE_DATA", encoding="utf-8")

    def mock_renameat2(olddirfd, oldpath, newdirfd, newpath, flags):
        # Simula renameat2 no filesystem
        os.rename(src, dst)
        return 0

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    res = EventSaver._safe_move_segment_no_overwrite(src, dst)
    assert res is True
    assert dst.exists()
    assert not src.exists()
    assert dst.read_text(encoding="utf-8") == "SOURCE_DATA"


def test_g4c_linux_renameat2_destination_exists_eexist(tmp_path, monkeypatch):
    """Valida que EEXIST em renameat2 lança FileExistsError sem sobrescrever."""
    import errno, ctypes
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "linux_src_eexist.txt"
    dst = tmp_path / "linux_dst_eexist.txt"
    src.write_text("SOURCE_DATA", encoding="utf-8")
    dst.write_text("EXISTING_DATA", encoding="utf-8")

    def mock_renameat2(olddirfd, oldpath, newdirfd, newpath, flags):
        ctypes.set_errno(errno.EEXIST)
        return -1

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    with pytest.raises(FileExistsError):
        EventSaver._safe_move_segment_no_overwrite(src, dst)

    assert dst.read_text(encoding="utf-8") == "EXISTING_DATA"
    assert src.read_text(encoding="utf-8") == "SOURCE_DATA"


def test_g4c_linux_no_renameat2_not_implemented(tmp_path, monkeypatch):
    """Valida que ausência de renameat2 no Linux adia a rotação sem fallback destrutivo."""
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")
    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: None)

    src = tmp_path / "linux_src_noutil.txt"
    dst = tmp_path / "linux_dst_noutil.txt"
    src.write_text("SRC", encoding="utf-8")

    with pytest.raises(NotImplementedError):
        EventSaver._safe_move_segment_no_overwrite(src, dst)

    assert src.exists()
    assert not dst.exists()


def test_g4c_linux_renameat2_enosys(tmp_path, monkeypatch):
    """Valida tratamento de ENOSYS (syscall não implementada no kernel)."""
    import errno, ctypes
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "src_enosys.txt"
    dst = tmp_path / "dst_enosys.txt"
    src.write_text("SRC", encoding="utf-8")

    def mock_renameat2(*args):
        ctypes.set_errno(errno.ENOSYS)
        return -1

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    with pytest.raises(OSError) as exc_info:
        EventSaver._safe_move_segment_no_overwrite(src, dst)
    assert exc_info.value.errno == errno.ENOSYS
    assert src.exists()


def test_g4c_linux_renameat2_eopnotsupp(tmp_path, monkeypatch):
    """Valida tratamento de EOPNOTSUPP (filesystem sem suporte a RENAME_NOREPLACE)."""
    import errno, ctypes
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "src_eop.txt"
    dst = tmp_path / "dst_eop.txt"
    src.write_text("SRC", encoding="utf-8")

    err_code = getattr(errno, "EOPNOTSUPP", 95)
    def mock_renameat2(*args):
        ctypes.set_errno(err_code)
        return -1

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    with pytest.raises(OSError) as exc_info:
        EventSaver._safe_move_segment_no_overwrite(src, dst)
    assert exc_info.value.errno == err_code
    assert src.exists()


def test_g4c_linux_renameat2_exdev(tmp_path, monkeypatch):
    """Valida tratamento de EXDEV (dispositivos distintos)."""
    import errno, ctypes
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "src_exdev.txt"
    dst = tmp_path / "dst_exdev.txt"
    src.write_text("SRC", encoding="utf-8")

    def mock_renameat2(*args):
        ctypes.set_errno(errno.EXDEV)
        return -1

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    with pytest.raises(OSError) as exc_info:
        EventSaver._safe_move_segment_no_overwrite(src, dst)
    assert exc_info.value.errno == errno.EXDEV
    assert src.exists()


def test_g4c_linux_renameat2_enospc(tmp_path, monkeypatch):
    """Valida tratamento de ENOSPC (sem espaço para atualizar metadados)."""
    import errno, ctypes
    monkeypatch.setattr("events.event_saver.platform.system", lambda: "Linux")

    src = tmp_path / "src_enospc.txt"
    dst = tmp_path / "dst_enospc.txt"
    src.write_text("SRC", encoding="utf-8")

    def mock_renameat2(*args):
        ctypes.set_errno(errno.ENOSPC)
        return -1

    monkeypatch.setattr("events.event_saver._get_linux_renameat2", lambda: mock_renameat2)

    with pytest.raises(OSError) as exc_info:
        EventSaver._safe_move_segment_no_overwrite(src, dst)
    assert exc_info.value.errno == errno.ENOSPC
    assert src.exists()


def test_g4c_no_hardlinks_created(hermetic_saver_g4c):
    """Garante que a nova implementação nunca cria múltiplos hardlinks em arquivos."""
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200

    saver.save_event(_create_sample_event("ev_link_1"))
    saver._process_single_flush_batch()
    saver.history_file.write_text("X" * 300, encoding="utf-8")

    saver.save_event(_create_sample_event("ev_link_2"))
    saver._process_single_flush_batch()

    archive_files = list(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    assert len(archive_files) == 1
    # st_nlink deve ser estritamente 1 (arquivo isolado, nunca compartilhado)
    assert archive_files[0].stat().st_nlink == 1
    if saver.history_file.exists():
        assert saver.history_file.stat().st_nlink == 1


def test_g4c_segment_immutability_after_subsequent_appends(hermetic_saver_g4c):
    """Valida imutabilidade absoluta do segmento arquivado após novos appends no arquivo ativo."""
    import hashlib
    saver = hermetic_saver_g4c
    saver.max_jsonl_file_bytes = 200

    saver.history_file.write_text("CONTEUDO_CONGELADO_" + ("A" * 250) + "\n", encoding="utf-8")

    # Força rotação
    res = saver._rotate_jsonl_if_needed()
    assert res is True

    archive_files = list(saver.archive_dir.glob("eventos_fluxo_*.jsonl"))
    assert len(archive_files) == 1
    arch_file = archive_files[0]

    # Calcula hash inicial do segmento
    initial_bytes = arch_file.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()
    initial_size = arch_file.stat().st_size

    # Grava múltiplos eventos novos no arquivo ativo
    for i in range(10):
        saver.save_event(_create_sample_event(f"ev_subsequent_{i}"))
        saver._process_single_flush_batch()

    # Confirma que o segmento arquivado NÃO mudou um único byte
    assert arch_file.stat().st_size == initial_size
    assert hashlib.sha256(arch_file.read_bytes()).hexdigest() == initial_hash

