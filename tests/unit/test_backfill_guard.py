# tests/unit/test_backfill_guard.py
# -*- coding: utf-8 -*-
"""Testes do guarda anti-contaminação backfill x produção (AÇÃO 2)."""
import pytest

from common.backfill_guard import (
    enforce_no_live_bot,
    is_live_bot_running,
)


def test_live_bot_detected_blocks():
    fake = [
        "python scripts/diagnostics/stress_flow_buffer.py --rate 250",
        "python main.py --dump-raw-trades dados/audit/x.jsonl",
    ]
    assert is_live_bot_running(fake) is True
    with pytest.raises(RuntimeError, match="Backfill/replay abortado: main.py"):
        enforce_no_live_bot(cmdlines=fake)


def test_no_live_bot_passes():
    fake = [
        "python scripts/diagnostics/stress_flow_buffer.py --rate 250",
        "C:\\Windows\\System32\\svchost.exe",
    ]
    assert is_live_bot_running(fake) is False
    enforce_no_live_bot(cmdlines=fake)  # não deve levantar


def test_empty_process_list_passes():
    assert is_live_bot_running([]) is False
    enforce_no_live_bot(cmdlines=[])


def test_real_scan_does_not_self_match():
    """Varredura real via psutil não deve detectar o próprio pytest."""
    from common.backfill_guard import _iter_cmdlines

    assert is_live_bot_running(_iter_cmdlines()) is False


def test_shell_mentioning_main_py_does_not_match():
    """Shell cujo comando apenas menciona main.py não é o bot."""
    fake = [
        'powershell -NoProfile -NonInteractive -Command "python -c \'import x\' # main.py check"',
        "code C:\\repo\\main.py",
        "python -c \"from common.backfill_guard import enforce_no_live_bot; enforce_no_live_bot()\"",
        "python -m pytest tests/unit/test_backfill_guard.py",
        "Set-Content -LiteralPath C:\\Temp\\main.py -Value x",
    ]
    assert is_live_bot_running(fake) is False


def test_main_py_token_variants_match():
    assert is_live_bot_running(["python main.py --dump-raw-trades x.jsonl"]) is True
    assert is_live_bot_running(["python C:\\repo\\main.py"]) is True
    assert is_live_bot_running(["python -u main.py --duration-seconds 7200"]) is True


def test_path_portability_posix_isolated_subprocess():
    """Valida identificação de caminhos Windows sob posixpath em subprocesso isolado."""
    import subprocess
    import sys

    code = """
import os, sys, posixpath
import importlib.util

os.path = posixpath
spec = importlib.util.spec_from_file_location('common.backfill_guard', 'common/backfill_guard.py')
bg = importlib.util.module_from_spec(spec)
sys.modules['common.backfill_guard'] = bg
spec.loader.exec_module(bg)

# Casos positivos
positives = [
    'python C:\\\\repo\\\\main.py',
    'python C:\\\\repo\\\\subdir\\\\main.py',
    'python /opt/bot/main.py',
    'python ./main.py',
    'python -u C:\\\\repo\\\\main.py',
]
for cmd in positives:
    assert bg.is_live_bot_running([cmd]) is True, f'Deveria identificar: {cmd}'

# Casos negativos
negatives = [
    'python C:\\\\repo\\\\not_main.py',
    'python /opt/repo/main_orchestrator.py',
    'python -m main.py',
    'code C:\\\\repo\\\\main.py',
]
for neg in negatives:
    assert bg.is_live_bot_running([neg]) is False, f'Não deveria identificar: {neg}'
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocesso falhou:\n{res.stderr}"
