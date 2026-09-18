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
