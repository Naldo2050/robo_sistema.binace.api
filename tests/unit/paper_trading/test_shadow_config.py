# tests/unit/paper_trading/test_shadow_config.py
"""
Unit tests for hermetic ShadowPaperConfig and validation (Gate C3-C-B1).
"""

from __future__ import annotations

import math
from typing import Dict
import pytest

from paper_trading.config import (
    ShadowPaperConfig,
    ShadowConfigResult,
    parse_shadow_config,
)
from paper_trading.cost_model import DEFAULT_PAPER_COST_CONFIG


def valid_enabled_env() -> Dict[str, str]:
    """Provide a minimal valid environment mapping when paper shadow is enabled."""
    return {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": "CH_SHADOW_2026_01",
        "PAPER_PROVIDER": "fixed_long",
        "PAPER_SYMBOL": "BTCUSDT",
        "PAPER_TIMEFRAME": "1m",
        "PAPER_NOTIONAL_USDT": "1000.0",
        "PAPER_HORIZON_S": "300",
        "PAPER_ORDER_TTL_MS": "5000",
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "binance_futures_vip0_2026",
        "PAPER_COST_EFFECTIVE_AT": "2026-01-01T00:00:00Z",
        "PAPER_STRATEGY_VERSION": "c3_shadow_v1.0.0",
    }


def test_1_disabled_without_other_vars_is_valid():
    """1. disabled sem outras vars é válido."""
    res = parse_shadow_config({})
    assert res.is_valid is True
    assert res.error is None
    assert res.config is not None
    assert res.config.enabled is False
    assert res.config.to_cost_config() == DEFAULT_PAPER_COST_CONFIG

    res2 = parse_shadow_config({"PAPER_SHADOW_ENABLED": "0"})
    assert res2.is_valid is True
    assert res2.config is not None
    assert res2.config.enabled is False

    res3 = parse_shadow_config({"PAPER_SHADOW_ENABLED": "false"})
    assert res3.is_valid is True
    assert res3.config is not None
    assert res3.config.enabled is False


def test_2_enabled_without_cohort_is_invalid():
    """2. enabled sem cohort inválido."""
    env = valid_enabled_env()
    del env["PAPER_COHORT_ID"]
    res = parse_shadow_config(env)
    assert res.is_valid is False
    assert res.config is None
    assert "PAPER_COHORT_ID is required" in (res.error or "")

    env["PAPER_COHORT_ID"] = "   "
    res2 = parse_shadow_config(env)
    assert res2.is_valid is False
    assert "PAPER_COHORT_ID is required" in (res2.error or "")


def test_3_enabled_without_notional_is_invalid():
    """3. enabled sem notional inválido."""
    env = valid_enabled_env()
    del env["PAPER_NOTIONAL_USDT"]
    res = parse_shadow_config(env)
    assert res.is_valid is False
    assert res.config is None
    assert "PAPER_NOTIONAL_USDT is required" in (res.error or "")

    env["PAPER_NOTIONAL_USDT"] = "0"
    res2 = parse_shadow_config(env)
    assert res2.is_valid is False
    assert "PAPER_NOTIONAL_USDT must be > 0" in (res2.error or "")

    env["PAPER_NOTIONAL_USDT"] = "-50"
    res3 = parse_shadow_config(env)
    assert res3.is_valid is False
    assert "PAPER_NOTIONAL_USDT must be > 0" in (res3.error or "")


def test_4_enabled_without_costs_is_invalid():
    """4. enabled sem costs inválido."""
    for cost_key in (
        "PAPER_MAKER_FEE_BPS",
        "PAPER_TAKER_FEE_BPS",
        "PAPER_ENTRY_SLIPPAGE_BPS",
        "PAPER_EXIT_SLIPPAGE_BPS",
        "PAPER_COST_SOURCE",
        "PAPER_COST_EFFECTIVE_AT",
    ):
        env = valid_enabled_env()
        del env[cost_key]
        res = parse_shadow_config(env)
        assert res.is_valid is False, f"Failed to reject missing {cost_key}"
        assert cost_key in (res.error or "")


def test_5_random_without_seed_is_invalid():
    """5. random sem seed inválido."""
    env = valid_enabled_env()
    env["PAPER_PROVIDER"] = "seeded_random"
    # Sem seed
    res = parse_shadow_config(env)
    assert res.is_valid is False
    assert "PAPER_RANDOM_SEED is required" in (res.error or "")

    # Com seed válida
    env["PAPER_RANDOM_SEED"] = "42"
    res2 = parse_shadow_config(env)
    assert res2.is_valid is True
    assert res2.config is not None
    assert res2.config.random_seed == 42


def test_6_invalid_provider_rejected():
    """6. provider inválido rejeitado."""
    env = valid_enabled_env()
    for bad_provider in ("xgboost", "llm", "qwen", "eval(foo)", "dynamic", ""):
        env["PAPER_PROVIDER"] = bad_provider
        res = parse_shadow_config(env)
        assert res.is_valid is False
        assert "PAPER_PROVIDER must be one of" in (res.error or "")


def test_7_negative_fees_rejected():
    """7. fees negativos rejeitados."""
    env = valid_enabled_env()
    env["PAPER_MAKER_FEE_BPS"] = "-1.0"
    res = parse_shadow_config(env)
    assert res.is_valid is False
    assert "PAPER_MAKER_FEE_BPS must be >= 0" in (res.error or "")

    env = valid_enabled_env()
    env["PAPER_ENTRY_SLIPPAGE_BPS"] = "-0.5"
    res2 = parse_shadow_config(env)
    assert res2.is_valid is False
    assert "PAPER_ENTRY_SLIPPAGE_BPS must be >= 0" in (res2.error or "")


def test_8_nan_and_inf_rejected():
    """8. NaN/Inf rejeitados."""
    for bad_val in ("nan", "NaN", "inf", "-inf", "Infinity"):
        env = valid_enabled_env()
        env["PAPER_NOTIONAL_USDT"] = bad_val
        res = parse_shadow_config(env)
        assert res.is_valid is False, f"Did not reject {bad_val} for notional"
        assert "must be finite" in (res.error or "")

        env2 = valid_enabled_env()
        env2["PAPER_TAKER_FEE_BPS"] = bad_val
        res2 = parse_shadow_config(env2)
        assert res2.is_valid is False, f"Did not reject {bad_val} for fee"
        assert "must be finite" in (res2.error or "")


def test_9_boolean_numeric_rejected():
    """9. bool numérico rejeitado."""
    from paper_trading.config import _parse_int, _parse_float

    val_int, err_int = _parse_int(True, "test_field")
    assert val_int is None
    assert "got bool" in (err_int or "")

    val_float, err_float = _parse_float(False, "test_field")
    assert val_float is None
    assert "got bool" in (err_float or "")


def test_10_effective_at_without_timezone_rejected():
    """10. effective_at sem timezone rejeitado."""
    env = valid_enabled_env()
    # Data ISO sem timezone (naive)
    env["PAPER_COST_EFFECTIVE_AT"] = "2026-01-01T00:00:00"
    res = parse_shadow_config(env)
    assert res.is_valid is False
    assert "must be timezone-aware ISO-8601 string, got naive" in (res.error or "")

    # Inválida completamente
    env["PAPER_COST_EFFECTIVE_AT"] = "not_a_date"
    res2 = parse_shadow_config(env)
    assert res2.is_valid is False
    assert "invalid ISO-8601" in (res2.error or "")

    # Válida com timezone explícito +00:00
    env["PAPER_COST_EFFECTIVE_AT"] = "2026-01-01T00:00:00+00:00"
    res3 = parse_shadow_config(env)
    assert res3.is_valid is True


def test_11_insecure_cohort_rejected():
    """11. cohort inseguro rejeitado."""
    env = valid_enabled_env()
    # Caracteres ilegais, espaços, path traversal
    for bad_cohort in (
        "ab",  # < 3 chars
        "a" * 65,  # > 64 chars
        "cohort with spaces",
        "cohort/../traversal",
        "cohort;drop table",
        "cohort$var",
        "<script>",
    ):
        env["PAPER_COHORT_ID"] = bad_cohort
        res = parse_shadow_config(env)
        assert res.is_valid is False, f"Failed to reject bad cohort {bad_cohort}"
        assert "must be 3-64 chars [A-Za-z0-9_.-]" in (res.error or "")


def test_12_config_does_not_read_credentials():
    """12. config não lê credentials."""
    env = valid_enabled_env()
    env["BINANCE_API_KEY"] = "TRADING_KEY_SECRET"
    env["BINANCE_API_SECRET"] = "TRADING_SECRET_SECRET"
    env["GROQ_API_KEY"] = "AI_GROQ_SECRET"
    env["OPENAI_API_KEY"] = "AI_OPENAI_SECRET"

    res = parse_shadow_config(env)
    assert res.is_valid is True
    assert res.config is not None

    # Provar que nenhum atributo da config carrega ou toca nessas chaves
    config_dict = res.config.__dict__
    for forbidden in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "GROQ_API_KEY", "OPENAI_API_KEY"):
        assert forbidden not in config_dict
        for val in config_dict.values():
            assert val != "TRADING_KEY_SECRET"
            assert val != "TRADING_SECRET_SECRET"
            assert val != "AI_GROQ_SECRET"
            assert val != "AI_OPENAI_SECRET"
