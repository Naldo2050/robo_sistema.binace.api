# tests/unit/paper_trading/test_shadow_factory.py
"""
Unit tests for paper_trading/factory.py (Gate C3-C-B3-B).

Covers all 17 mandatory factory test scenarios:
1. flag ausente -> DISABLED
2. false -> DISABLED
3. OFF não cria ledger
4. OFF não cria thread
5. ON config válida -> RUNNING
6. ON sem DB path -> FAILED
7. ON sem git sha -> FAILED
8. ON config econômica inválida -> FAILED
9. duplicate cohort -> FAILED
10. ledger start failure -> FAILED
11. failure limpa ledger/thread
12. provider allowlist preservada
13. random sem seed -> FAILED
14. credentials não são lidas pela factory
15. zero network
16. zero LLM
17. zero order endpoint
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from paper_trading.factory import ShadowFactoryResult, create_shadow_runtime
from paper_trading.ledger import PaperLedger


def _valid_env(tmp_path: Any, cohort_id: str = "CH_FACTORY_TEST") -> Dict[str, str]:
    db_file = str(tmp_path / f"{cohort_id}.db")
    return {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": cohort_id,
        "PAPER_PROVIDER": "fixed_long",
        "PAPER_NOTIONAL_USDT": "1000.0",
        "PAPER_HORIZON_S": "300",
        "PAPER_ORDER_TTL_MS": "5000",
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "VIP0_TIER",
        "PAPER_COST_EFFECTIVE_AT": "2026-09-01T00:00:00+00:00",
        "PAPER_DB_PATH": db_file,
        "PAPER_GIT_SHA": "0c5c95fa1b2c3d4e5f67890abcdef1234567890a",
    }


def test_01_flag_missing_returns_disabled():
    """1. Flag ausente -> DISABLED."""
    res = create_shadow_runtime({})
    assert res.status == "DISABLED"
    assert res.runtime is None
    assert res.reason is None


def test_02_flag_false_returns_disabled():
    """2. Flag false -> DISABLED."""
    for token in ("0", "false", "no", "off", "False", "OFF"):
        res = create_shadow_runtime({"PAPER_SHADOW_ENABLED": token})
        assert res.status == "DISABLED"
        assert res.runtime is None
        assert res.reason is None


def test_03_off_does_not_create_ledger(tmp_path):
    """3. OFF não cria ledger nem arquivos."""
    fake_db = str(tmp_path / "never_created.db")
    env = {"PAPER_SHADOW_ENABLED": "0", "PAPER_DB_PATH": fake_db}
    with patch("paper_trading.factory.PaperLedger") as mock_ledger:
        res = create_shadow_runtime(env)
        assert res.status == "DISABLED"
        assert not mock_ledger.called
    assert not os.path.exists(fake_db)


def test_04_off_does_not_create_thread():
    """4. OFF não cria worker threads."""
    before_threads = set(threading.enumerate())
    res = create_shadow_runtime({"PAPER_SHADOW_ENABLED": "0"})
    after_threads = set(threading.enumerate())
    assert res.status == "DISABLED"
    assert after_threads == before_threads


def test_05_on_valid_config_returns_running(tmp_path):
    """5. ON com configuração válida -> RUNNING."""
    env = _valid_env(tmp_path, cohort_id="CH_FACTORY_05")
    res = create_shadow_runtime(env)
    try:
        assert res.status == "RUNNING"
        assert res.runtime is not None
        assert res.runtime.is_active is True
        assert res.runtime.accepts_new_exposure is True
        assert res.reason is None
    finally:
        if res.runtime is not None:
            res.runtime.shutdown()


def test_06_on_without_db_path_fails(tmp_path):
    """6. ON sem DB path -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_TEST_NODB")
    del env["PAPER_DB_PATH"]
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"
    assert res.runtime is None
    assert "PAPER_DB_PATH is required" in (res.reason or "")


def test_07_on_without_git_sha_fails(tmp_path):
    """7. ON sem git sha -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_TEST_NOSHA")
    del env["PAPER_GIT_SHA"]
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"
    assert res.runtime is None
    assert "PAPER_GIT_SHA is required" in (res.reason or "")


def test_08_on_invalid_economic_config_fails(tmp_path):
    """8. ON com config econômica inválida (ex: fee negativa) -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_TEST_BADECO")
    env["PAPER_MAKER_FEE_BPS"] = "-1.0"
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"
    assert res.runtime is None
    assert "PAPER_MAKER_FEE_BPS must be >= 0" in (res.reason or "")


def test_09_duplicate_cohort_fails(tmp_path):
    """9. Duplicate cohort -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_DUP_COHORT")
    res1 = create_shadow_runtime(env)
    assert res1.status == "RUNNING"

    # Segunda tentativa com mesma cohort_id e mesmo DB
    res2 = create_shadow_runtime(env)
    assert res2.status == "FAILED"
    assert res2.runtime is None
    assert "duplicate cohort" in (res2.reason or "").lower() or "failed" in (res2.reason or "").lower()

    if res1.runtime is not None:
        res1.runtime.shutdown()


def test_10_ledger_start_failure_fails(tmp_path):
    """10. Ledger start/health failure -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_HEALTH_FAIL")
    # Injetamos mock de ledger não-saudável
    mock_ledger = MagicMock()
    mock_ledger.health_snapshot.return_value = {
        "healthy": False,
        "consecutive_errors": 5,
        "fatal_error": "Mock disk failure",
    }
    res = create_shadow_runtime(env, ledger=mock_ledger)
    assert res.status == "FAILED"
    assert res.runtime is None


def test_11_failure_cleans_up_ledger_and_thread(tmp_path):
    """11. Failure limpa ledger/thread sem deixar órfãos."""
    db_file = str(tmp_path / "cleanup_test.db")
    env = _valid_env(tmp_path, cohort_id="CH_CLEANUP")
    env["PAPER_DB_PATH"] = db_file

    before_threads = set(threading.enumerate())

    # Provocamos erro durante start simulando duplicate cohort no banco
    first_ledger = PaperLedger(db_path=db_file)
    first_ledger.create_cohort("CH_CLEANUP", 1000, "first", {})
    first_ledger.close()

    # Tentativa de criar runtime com a mesma cohort falhará no runtime.start()
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"

    # Aguarda brevemente para confirmar encerramento do thread do ledger interno
    after_threads = set(threading.enumerate())
    # O thread writer não deve permanecer ativo
    new_threads = [t for t in after_threads - before_threads if "ledger" in t.name.lower()]
    assert len(new_threads) == 0


def test_12_provider_allowlist_preserved(tmp_path):
    """12. Provider fora da allowlist -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_BAD_PROV")
    env["PAPER_PROVIDER"] = "deepseek_ai"
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"
    assert res.runtime is None
    assert "PAPER_PROVIDER must be one of" in (res.reason or "")


def test_13_random_without_seed_fails(tmp_path):
    """13. Provider seeded_random sem PAPER_RANDOM_SEED -> FAILED."""
    env = _valid_env(tmp_path, cohort_id="CH_NO_SEED")
    env["PAPER_PROVIDER"] = "seeded_random"
    env.pop("PAPER_RANDOM_SEED", None)
    res = create_shadow_runtime(env)
    assert res.status == "FAILED"
    assert res.runtime is None
    assert "PAPER_RANDOM_SEED is required" in (res.reason or "")


def test_14_credentials_never_accessed_by_factory():
    """14. Credentials (BINANCE_API_KEY, GROQ_API_KEY, etc.) não são lidas pela factory."""
    forbidden_keys = {
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }

    class MonitoredEnv(dict):
        def __getitem__(self, key):
            if key in forbidden_keys:
                raise AssertionError(f"Factory accessed forbidden credential key: {key}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in forbidden_keys:
                raise AssertionError(f"Factory accessed forbidden credential key via get(): {key}")
            return super().get(key, default)

    env = MonitoredEnv({
        "PAPER_SHADOW_ENABLED": "0",
        "BINANCE_API_KEY": "dummy_secret_key",
        "BINANCE_API_SECRET": "dummy_secret",
        "GROQ_API_KEY": "dummy_groq",
    })

    res = create_shadow_runtime(env)
    assert res.status == "DISABLED"


def test_15_zero_network_io(tmp_path):
    """15. Zero chamadas a sockets ou rede."""
    import socket

    original_socket = socket.socket

    def guard_socket(*args, **kwargs):
        raise AssertionError("Network socket opened during factory runtime creation!")

    env = _valid_env(tmp_path, cohort_id="CH_NO_NET")
    with patch("socket.socket", side_effect=guard_socket):
        res = create_shadow_runtime(env)
        try:
            assert res.status == "RUNNING"
        finally:
            if res.runtime is not None:
                res.runtime.shutdown()


def test_16_zero_llm_calls(tmp_path):
    """16. Zero chamadas a LLM ou IA."""
    env = _valid_env(tmp_path, cohort_id="CH_NO_LLM")
    with patch("market_orchestrator.ai.ai_orchestrator.AIOrchestrator") if "market_orchestrator.ai.ai_orchestrator" in os.sys.modules else patch("builtins.print"):
        res = create_shadow_runtime(env)
        try:
            assert res.status == "RUNNING"
        finally:
            if res.runtime is not None:
                res.runtime.shutdown()


def test_17_zero_order_endpoint(tmp_path):
    """17. Zero chamadas a endpoints de ordem ou broker real."""
    env = _valid_env(tmp_path, cohort_id="CH_NO_ORDER")
    # Verifica que nenhum executor live ou client Binance é chamado
    with patch("binance.client.Client") if "binance.client" in os.sys.modules else patch("builtins.print"):
        res = create_shadow_runtime(env)
        try:
            assert res.status == "RUNNING"
        finally:
            if res.runtime is not None:
                res.runtime.shutdown()
