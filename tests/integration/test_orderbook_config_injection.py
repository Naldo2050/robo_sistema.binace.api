# tests/test_orderbook_config_injection.py
from __future__ import annotations

from typing import Dict, Any, List

import pytest

from orderbook_core.orderbook_config import OrderBookConfig
from orderbook_analyzer import OrderBookAnalyzer
from tests.conftest import make_valid_snapshot


@pytest.mark.asyncio
async def test_analyzer_uses_custom_cfg_basic(tm) -> None:
    """
    Garante que, ao passar um OrderBookConfig customizado no __init__,
    o analyzer usa esses valores em get_stats() e campos internos.
    """
    custom_cfg = OrderBookConfig(
        depth_levels=[1, 3, 5],
        spread_tight_threshold_bps=0.05,
        spread_avg_windows_min=[5, 60],
        critical_imbalance=0.9,
        min_dominant_usd=123456.0,
        min_ratio_dom=15.0,
        request_timeout=7.5,
        retry_delay=1.5,
        max_retries=5,
        max_requests_per_min=42,
        cache_ttl=9.0,
        max_stale=33.0,
        min_depth_usd=777.0,
        allow_partial=True,
        use_fallback=True,
        fallback_max_age=321.0,
        emergency_mode=True,
    )

    oba = OrderBookAnalyzer(
        symbol="BTCUSDT",
        time_manager=tm,
        cfg=custom_cfg,
    )

    stats = oba.get_stats()
    cfg_stats = stats.get("config", {})

    assert oba.cfg is custom_cfg
    assert cfg_stats["min_depth_usd"] == pytest.approx(custom_cfg.min_depth_usd)
    assert cfg_stats["allow_partial"] is custom_cfg.allow_partial
    assert cfg_stats["emergency_mode"] is custom_cfg.emergency_mode

    assert cfg_stats["cache_ttl"] == pytest.approx(custom_cfg.cache_ttl)
    assert cfg_stats["max_stale"] == pytest.approx(custom_cfg.max_stale)

    # rate_limit_threshold deve vir de cfg.max_requests_per_min se não houver override
    assert stats["rate_limit_threshold"] == custom_cfg.max_requests_per_min


@pytest.mark.asyncio
async def test_validate_snapshot_respects_cfg_min_depth(tm) -> None:
    """
    Usa um cfg com min_depth_usd exageradamente alto para forçar falha
    de validação de um snapshot que normalmente seria aceito.
    """
    # min_depth muito alto => snapshot normal deve ser rejeitado por "liquidez muito baixa"
    custom_cfg = OrderBookConfig(
        depth_levels=[1, 5, 10],
        spread_tight_threshold_bps=0.2,
        spread_avg_windows_min=[60, 1440],
        critical_imbalance=0.95,
        min_dominant_usd=2_000_000.0,
        min_ratio_dom=20.0,
        request_timeout=10.0,
        retry_delay=2.0,
        max_retries=3,
        max_requests_per_min=10,
        cache_ttl=15.0,
        max_stale=60.0,
        min_depth_usd=10_000_000.0,  # bem maior que o depth do snapshot normal
        allow_partial=False,
        use_fallback=True,
        fallback_max_age=120.0,
        emergency_mode=False,
    )

    oba = OrderBookAnalyzer(
        symbol="BTCUSDT",
        time_manager=tm,
        cfg=custom_cfg,
    )

    snap = make_valid_snapshot(tm.now_ms())
    ok, issues, _ = oba._validate_snapshot(snap)

    assert ok is False
    assert any("liquidez muito baixa" in s for s in issues)