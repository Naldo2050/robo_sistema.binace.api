# -*- coding: utf-8 -*-
"""
tests/unit/test_event_saver_g4a.py
Testes herméticos para a Fase 5.11 - Gate G4-A (Configuração Operacional e Rotação do Log Visual).

Cobre rigorosamente:
Parte I - Configuração Operacional via Variáveis de Ambiente (ETAPA 4):
1. Ambiente sem variáveis (defaults: True, 10 MiB, 2 backups).
2. WRITE_VISUAL_LOG=0 (desabilitado).
3. WRITE_VISUAL_LOG=false (desabilitado).
4. WRITE_VISUAL_LOG=1 (habilitado).
5. MAX_VISUAL_LOG_MB válido (valor aplicado).
6. MAX_VISUAL_LOG_MB inválido (fallback sem crash).
7. MAX_VISUAL_LOG_MB negativo/zero (fallback).
8. MAX_VISUAL_LOG_MB excessivo (clamping para teto de 1024 MiB).
9. VISUAL_LOG_BACKUPS válido (valor aplicado).
10. VISUAL_LOG_BACKUPS inválido/negativo (fallback).
11. VISUAL_LOG_BACKUPS excessivo (clamping para teto de 10).
12. Flag desabilitada: nenhum arquivo .lock criado nem aberto.

Parte II - Rotação, Falhas e Contratos de Concorrência:
13. Rotação no limite de tamanho.
14. Dois backups em cascata (.1, .2, nenhum .3).
15. Falha intermediária na cascata aborta sem corromper backups.
16. PermissionError (Windows) capturado de forma resiliente.
17. Preservação do bloco quando rotação falha.
18. Preservação do separador visual acionando rotação.
19. Formato textual inalterado (headers, timezones, JSON).
20. Lock interprocessos estável (sem exclusão de .lock).
21. Escrita após rotação.
22. Falha de stat capturada de forma segura.
23. Falha de write registrada no log sem crash.
24. Falha de flush/close com liberação defensiva do lock.
25. Shutdown durante/após rotação (CLEAN_STOP).
26. Nenhum I/O visual no caller (hot path protegido).
27. Contadores G1 e G2 rigorosamente preservados.
"""

import os
from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest

from events import event_saver
from events.event_saver import EventSaver


def sync_env_to_config(monkeypatch):
    """
    Sincroniza as variáveis de ambiente ativas com o módulo config do EventSaver
    usando monkeypatch, garantindo restauração automática sem contaminação global.
    """
    monkeypatch.setattr(
        event_saver.config,
        "EVENT_SAVER_WRITE_VISUAL_LOG",
        os.getenv("EVENT_SAVER_WRITE_VISUAL_LOG", "true")
    )
    monkeypatch.setattr(
        event_saver.config,
        "EVENT_SAVER_MAX_VISUAL_LOG_MB",
        os.getenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", "10")
    )
    monkeypatch.setattr(
        event_saver.config,
        "EVENT_SAVER_VISUAL_LOG_BACKUPS",
        os.getenv("EVENT_SAVER_VISUAL_LOG_BACKUPS", "2")
    )


@pytest.fixture
def hermetic_saver_g4a(tmp_path, monkeypatch):
    """Fixture hermética isolando o ambiente de disco e dependências do EventSaver."""
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)

    mock_store = MagicMock()
    mock_store.save_batch.return_value = event_saver.BatchSaveResult(
        total_received=1, persisted=1, rejected_pre_commit=0, commit_status="COMMITTED"
    )
    monkeypatch.setattr(event_saver, "EventStore", lambda: mock_store)

    sync_env_to_config(monkeypatch)
    saver = EventSaver(sound_alert=False)
    saver.db = mock_store
    saver.history_file = tmp_path / "events.jsonl"
    saver.snapshot_file = tmp_path / "events.json"
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    yield saver

    try:
        saver.stop(timeout=1.0)
    except Exception:
        pass


# ==============================================================================
# PARTE I — CONFIGURAÇÃO OPERACIONAL VIA VARIÁVEIS DE AMBIENTE (ETAPA 4)
# ==============================================================================

def test_env_1_defaults_sem_variaveis(monkeypatch, tmp_path):
    """1. Ambiente sem variáveis: defaults são True, 10 MiB, 2 backups."""
    monkeypatch.delenv("EVENT_SAVER_WRITE_VISUAL_LOG", raising=False)
    monkeypatch.delenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", raising=False)
    monkeypatch.delenv("EVENT_SAVER_VISUAL_LOG_BACKUPS", raising=False)

    sync_env_to_config(monkeypatch)
    saver = EventSaver(sound_alert=False)
    try:
        assert saver.write_visual_log is True
        assert saver.max_visual_log_bytes == 10 * 1024 * 1024
        assert saver.visual_log_backups == 2
    finally:
        saver.stop(timeout=1.0)


def test_env_2_write_visual_log_zero(monkeypatch, tmp_path):
    """2. WRITE_VISUAL_LOG=0: log visual deve ser desabilitado."""
    monkeypatch.setenv("EVENT_SAVER_WRITE_VISUAL_LOG", "0")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.write_visual_log is False
    finally:
        saver.stop(timeout=1.0)


def test_env_3_write_visual_log_false(monkeypatch, tmp_path):
    """3. WRITE_VISUAL_LOG=false: log visual deve ser desabilitado."""
    monkeypatch.setenv("EVENT_SAVER_WRITE_VISUAL_LOG", "false")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.write_visual_log is False
    finally:
        saver.stop(timeout=1.0)


def test_env_4_write_visual_log_um(monkeypatch, tmp_path):
    """4. WRITE_VISUAL_LOG=1: log visual deve ser habilitado."""
    monkeypatch.setenv("EVENT_SAVER_WRITE_VISUAL_LOG", "1")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.write_visual_log is True
    finally:
        saver.stop(timeout=1.0)


def test_env_5_max_mb_valido(monkeypatch, tmp_path):
    """5. MAX_VISUAL_LOG_MB válido: valor aplicado em bytes."""
    monkeypatch.setenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", "25")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.max_visual_log_bytes == 25 * 1024 * 1024
    finally:
        saver.stop(timeout=1.0)


def test_env_6_max_mb_invalido_fallback(monkeypatch, tmp_path):
    """6. MAX_VISUAL_LOG_MB inválido: fallback sem crash para 10 MiB."""
    monkeypatch.setenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", "texto_invalido")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.max_visual_log_bytes == 10 * 1024 * 1024
    finally:
        saver.stop(timeout=1.0)


def test_env_7_max_mb_negativo_zero_fallback(monkeypatch, tmp_path):
    """7. MAX_VISUAL_LOG_MB negativo ou zero: fallback defensivo para 10 MiB."""
    for val in ["-5", "0", "-0.1"]:
        monkeypatch.setenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", val)
        sync_env_to_config(monkeypatch)
        saver = EventSaver(sound_alert=False)
        try:
            assert saver.max_visual_log_bytes == 10 * 1024 * 1024
        finally:
            saver.stop(timeout=1.0)


def test_env_8_max_mb_excessivo_clamping(monkeypatch, tmp_path):
    """8. MAX_VISUAL_LOG_MB excessivo: clamping seguro para teto de 1024 MiB."""
    monkeypatch.setenv("EVENT_SAVER_MAX_VISUAL_LOG_MB", "5000")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.max_visual_log_bytes == 1024 * 1024 * 1024
    finally:
        saver.stop(timeout=1.0)


def test_env_9_backups_valido(monkeypatch, tmp_path):
    """9. VISUAL_LOG_BACKUPS válido: valor aplicado."""
    monkeypatch.setenv("EVENT_SAVER_VISUAL_LOG_BACKUPS", "4")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.visual_log_backups == 4
    finally:
        saver.stop(timeout=1.0)


def test_env_10_backups_invalido_fallback(monkeypatch, tmp_path):
    """10. VISUAL_LOG_BACKUPS inválido/negativo/zero: fallback defensivo para 2."""
    for val in ["invalido", "0", "-1"]:
        monkeypatch.setenv("EVENT_SAVER_VISUAL_LOG_BACKUPS", val)
        sync_env_to_config(monkeypatch)
        saver = EventSaver(sound_alert=False)
        try:
            assert saver.visual_log_backups == 2
        finally:
            saver.stop(timeout=1.0)


def test_env_11_backups_excessivo_clamping(monkeypatch, tmp_path):
    """11. VISUAL_LOG_BACKUPS excessivo: clamping seguro para teto de 10."""
    monkeypatch.setenv("EVENT_SAVER_VISUAL_LOG_BACKUPS", "50")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    try:
        assert saver.visual_log_backups == 10
    finally:
        saver.stop(timeout=1.0)


def test_env_12_flag_desabilitada_sem_arquivo_lock(tmp_path, monkeypatch):
    """12. Flag desabilitada: nenhum arquivo .lock criado nem aberto."""
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)
    monkeypatch.setenv("EVENT_SAVER_WRITE_VISUAL_LOG", "0")
    sync_env_to_config(monkeypatch)

    saver = EventSaver(sound_alert=False)
    saver.visual_log_file = tmp_path / "eventos_visuais.log"

    try:
        assert saver.write_visual_log is False

        # Dispara operações de escrita direta
        saver._add_visual_separator({"janela_numero": 101, "epoch_ms": 1700000000000})
        saver._add_visual_log_entry({"tipo_evento": "TEST_EVENT", "epoch_ms": 1700000000000})

        lock_path = saver.visual_log_file.with_suffix(".lock")
        assert not saver.visual_log_file.exists()
        assert not lock_path.exists(), "Arquivo .lock não deve ser criado quando log visual estiver desligado"
    finally:
        saver.stop(timeout=1.0)


# ==============================================================================
# PARTE II — ROTAÇÃO, FALHAS, LOCKS E CONTRATOS DE PERSISTÊNCIA
# ==============================================================================

def test_13_rotacao_no_limite(hermetic_saver_g4a):
    """13. Rotação no limite: arquivo rotaciona para .1 quando tamanho atinge o teto."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 150
    saver.visual_log_backups = 2

    initial_content = "X" * 200
    saver.visual_log_file.write_text(initial_content, encoding="utf-8")

    event = {"tipo_evento": "NEW_BLOCK", "symbol": "BTCUSDT", "epoch_ms": 1700000000000}
    saver._add_visual_log_entry(event)

    backup1 = Path(f"{saver.visual_log_file}.1")
    assert backup1.exists()
    assert backup1.read_text(encoding="utf-8") == initial_content

    assert saver.visual_log_file.exists()
    new_content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "EVENTO: NEW_BLOCK" in new_content
    assert initial_content not in new_content


def test_14_dois_backups_cascata(hermetic_saver_g4a):
    """14. Dois backups em cascata (.1 e .2 mantidos; .3 nunca criado)."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 50
    saver.visual_log_backups = 2

    backup1 = Path(f"{saver.visual_log_file}.1")
    backup2 = Path(f"{saver.visual_log_file}.2")
    backup3 = Path(f"{saver.visual_log_file}.3")

    # Geração 1
    saver.visual_log_file.write_text("GEN_1" * 15, encoding="utf-8")
    saver._add_visual_log_entry({"tipo_evento": "E1", "epoch_ms": 1700000000000})

    assert backup1.exists()
    assert "GEN_1" in backup1.read_text(encoding="utf-8")
    assert not backup2.exists()

    # Geração 2
    saver.visual_log_file.write_text("GEN_2" * 15, encoding="utf-8")
    saver._add_visual_log_entry({"tipo_evento": "E2", "epoch_ms": 1700000000000})

    assert backup1.exists()
    assert backup2.exists()
    assert "GEN_2" in backup1.read_text(encoding="utf-8")
    assert "GEN_1" in backup2.read_text(encoding="utf-8")

    # Geração 3
    saver.visual_log_file.write_text("GEN_3" * 15, encoding="utf-8")
    saver._add_visual_log_entry({"tipo_evento": "E3", "epoch_ms": 1700000000000})

    assert backup1.exists()
    assert backup2.exists()
    assert "GEN_3" in backup1.read_text(encoding="utf-8")
    assert "GEN_2" in backup2.read_text(encoding="utf-8")
    assert not backup3.exists()


def test_15_falha_intermediaria_na_cascata(hermetic_saver_g4a, monkeypatch):
    """15. Falha intermediária na cascata: aborta sem corromper backups nem ativo."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 50
    saver.visual_log_backups = 2

    backup1 = Path(f"{saver.visual_log_file}.1")
    backup1.write_text("BACKUP_1_ORIGINAL", encoding="utf-8")
    saver.visual_log_file.write_text("ACTIVE_ORIGINAL" * 5, encoding="utf-8")

    real_replace = os.replace

    def mock_replace(src, dst):
        if str(src).endswith(".1") and str(dst).endswith(".2"):
            raise OSError("Simulated cascade rename failure on backup .1 -> .2")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", mock_replace)

    saver._add_visual_log_entry({"tipo_evento": "FALLBACK_TEST", "epoch_ms": 1700000000000})

    assert backup1.read_text(encoding="utf-8") == "BACKUP_1_ORIGINAL"
    active_content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "ACTIVE_ORIGINAL" in active_content
    assert "EVENTO: FALLBACK_TEST" in active_content


def test_16_permission_error_windows(hermetic_saver_g4a, monkeypatch):
    """16. PermissionError comum no Windows: captura de forma resiliente."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 50
    saver.visual_log_backups = 2

    saver.visual_log_file.write_text("INITIAL_DATA" * 10, encoding="utf-8")

    def mock_replace(src, dst):
        raise PermissionError("[WinError 32] O processo nao pode acessar o arquivo")

    monkeypatch.setattr(os, "replace", mock_replace)

    saver._add_visual_log_entry({"tipo_evento": "WIN_LOCK_TEST", "epoch_ms": 1700000000000})

    active_content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "INITIAL_DATA" in active_content
    assert "EVENTO: WIN_LOCK_TEST" in active_content


def test_17_preservacao_do_bloco_sob_falha_de_rotacao(hermetic_saver_g4a, monkeypatch):
    """17. Preservação do bloco: se rotação falhar, o bloco é escrito no arquivo ativo."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 50

    saver.visual_log_file.write_text("EXISTING_ACTIVE" * 10, encoding="utf-8")

    monkeypatch.setattr(saver, "_rotate_visual_log_if_needed", lambda: False)

    saver._add_visual_log_entry({"tipo_evento": "PRESERVED_EVENT", "epoch_ms": 1700000000000})

    content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "EXISTING_ACTIVE" in content
    assert "EVENTO: PRESERVED_EVENT" in content


def test_18_preservacao_do_separador(hermetic_saver_g4a):
    """18. Preservação do separador: rotação funciona acionada por separador."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 100
    saver.visual_log_backups = 2

    saver.visual_log_file.write_text("OLD_JANELA_DATA" * 10, encoding="utf-8")

    saver._add_visual_separator({
        "janela_numero": 42,
        "epoch_ms": 1700000000000,
        "data_context": "unit_test"
    })

    backup1 = Path(f"{saver.visual_log_file}.1")
    assert backup1.exists()
    assert "OLD_JANELA_DATA" in backup1.read_text(encoding="utf-8")

    active_content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "Janela 42" in active_content
    assert "OLD_JANELA_DATA" not in active_content


def test_19_formato_textual_inalterado(hermetic_saver_g4a):
    """19. Formato textual inalterado: headers com UTC, NY, SP e JSON formatado."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    event = {
        "tipo_evento": "ANALYSIS_SIGNAL",
        "symbol": "ETHUSDT",
        "janela_numero": 10,
        "epoch_ms": 1700000000000,
        "signal": "BUY"
    }

    saver._add_visual_log_entry(event)

    content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "-" * 100 in content
    assert "EVENTO: ANALYSIS_SIGNAL | SYMBOL: ETHUSDT | JANELA: 10" in content
    assert "UTC:" in content
    assert "NY:" in content
    assert "SP:" in content
    assert '"signal": "BUY"' in content


def test_20_lock_interprocessos_estavel_sem_unlink(hermetic_saver_g4a):
    """20. Lock interprocessos estável: arquivo .lock NÃO é excluído (sem unlink)."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    saver._add_visual_log_entry({"tipo_evento": "LOCK_TEST", "epoch_ms": 1700000000000})

    lock_file = saver.visual_log_file.with_suffix(".lock")
    assert lock_file.exists(), "O arquivo .lock deve permanecer como âncora fixa de sincronização"


def test_21_escrita_apos_rotacao(hermetic_saver_g4a):
    """21. Escrita após rotação: múltiplos eventos continuam sendo gravados no novo arquivo."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 100

    saver.visual_log_file.write_text("OLD_STREAM" * 15, encoding="utf-8")

    saver._add_visual_log_entry({"tipo_evento": "FIRST_AFTER_ROT", "epoch_ms": 1700000000000})

    saver.max_visual_log_bytes = 10 * 1024 * 1024
    saver._add_visual_log_entry({"tipo_evento": "SECOND_AFTER_ROT", "epoch_ms": 1700000001000})

    active_content = saver.visual_log_file.read_text(encoding="utf-8")
    assert "EVENTO: FIRST_AFTER_ROT" in active_content
    assert "EVENTO: SECOND_AFTER_ROT" in active_content


def test_22_falha_de_stat(hermetic_saver_g4a, monkeypatch):
    """22. Falha de stat: helper captura OSError e escrita prossegue com segurança."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    saver.visual_log_file.write_text("STAT_TEST_DATA", encoding="utf-8")

    def mock_stat(*args, **kwargs):
        raise OSError("Simulated stat hardware failure")

    monkeypatch.setattr(Path, "stat", mock_stat)

    res = saver._rotate_visual_log_if_needed()
    assert res is False

    saver._add_visual_log_entry({"tipo_evento": "STAT_EVENT", "epoch_ms": 1700000000000})


def test_23_falha_de_write(hermetic_saver_g4a, monkeypatch):
    """23. Falha de write: tratada e logada sem crashar aplicação."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    real_open = open

    def failing_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("eventos_visuais.log") and "a" in mode:
            f = real_open(path, mode, *args, **kwargs)
            f.write = MagicMock(side_effect=IOError("Simulated disk write failure"))
            return f
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    saver._add_visual_log_entry({"tipo_evento": "WRITE_FAIL_TEST", "epoch_ms": 1700000000000})


def test_24_falha_de_flush_close(hermetic_saver_g4a, monkeypatch):
    """24. Falha de flush/close: lock é liberado defensivamente no finally."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    real_open = open

    def failing_flush_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("eventos_visuais.log") and "a" in mode:
            f = real_open(path, mode, *args, **kwargs)
            f.flush = MagicMock(side_effect=IOError("Simulated flush failure"))
            return f
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_flush_open)

    saver._add_visual_log_entry({"tipo_evento": "FLUSH_FAIL_TEST", "epoch_ms": 1700000000000})

    lock_file = saver.visual_log_file.with_suffix(".lock")
    with open(lock_file, "a") as lf:
        assert event_saver.acquire_file_lock(lf, blocking=False) is True
        event_saver.release_file_lock(lf)


def test_25_shutdown_durante_rotacao(hermetic_saver_g4a):
    """25. Shutdown durante/após rotação: encerra com CLEAN_STOP sem anomalias."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True
    saver.max_visual_log_bytes = 50

    saver.visual_log_file.write_text("PRE_STOP_DATA" * 10, encoding="utf-8")

    saver.save_event({"tipo_evento": "STOP_ROT_EVENT", "epoch_ms": 1700000000000})
    saver.stop(timeout=5.0)

    stats = saver.get_stats()
    assert stats["shutdown_status"] == "CLEAN_STOP"
    assert stats["worker_exited_cleanly"] is True
    assert stats["shutdown_incomplete"] is False


def test_26_nenhum_io_visual_no_caller(hermetic_saver_g4a, monkeypatch):
    """26. Nenhum I/O visual no caller: save_event apenas enfileira na memória."""
    saver = hermetic_saver_g4a
    saver.write_visual_log = True

    open_called = []
    real_open = open

    def spy_open(path, *args, **kwargs):
        if "eventos_visuais" in str(path):
            open_called.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", spy_open)

    saver.save_event({"tipo_evento": "HOT_PATH_TEST", "epoch_ms": 1700000000000})

    assert len(open_called) == 0, "save_event() não pode realizar I/O visual na thread chamadora"


def test_27_contadores_g1_g2_preservados(hermetic_saver_g4a):
    """27. Contadores G1 e G2 permanecem estritamente preservados com rotação ativa ou inativa."""
    saver = hermetic_saver_g4a
    saver.max_visual_log_bytes = 50

    # 1. Com log visual ativado
    saver.write_visual_log = True
    for i in range(5):
        saver.save_event({"tipo_evento": f"COUNT_EVENT_{i}", "epoch_ms": 1700000000000 + i})

    deadline = time.time() + 2.0
    while time.time() < deadline:
        saver._wake_flush_event.set()
        time.sleep(0.05)
        st = saver.get_stats()
        if st["events_in_flight"] == 0 and st["events_waiting"] == 0:
            break

    stats = saver.get_stats()
    assert stats["events_received"] == 5
    assert stats["events_accepted"] == 5
    assert stats["events_finalized"] == 5
    assert stats["events_rejected_overflow"] == 0
    assert stats["events_rejected_after_stop"] == 0

    # 2. Com log visual desativado
    saver.write_visual_log = False
    for i in range(5, 10):
        saver.save_event({"tipo_evento": f"COUNT_EVENT_{i}", "epoch_ms": 1700000000000 + i})

    deadline = time.time() + 2.0
    while time.time() < deadline:
        saver._wake_flush_event.set()
        time.sleep(0.05)
        st = saver.get_stats()
        if st["events_in_flight"] == 0 and st["events_waiting"] == 0:
            break

    stats2 = saver.get_stats()
    assert stats2["events_received"] == 10
    assert stats2["events_accepted"] == 10
    assert stats2["events_finalized"] == 10
    assert stats2["events_rejected_overflow"] == 0
    assert stats2["events_rejected_after_stop"] == 0
