# tests/unit/test_mtf_nonfinite_none.py
"""
Regressão (NaN/Inf em multi_tf): non-finite -> None -> JSON null.

Bug: ContextCollector._analyze_mtf_trends usava `round(x, N) if x else 0.0`.
Como NaN/±Inf são truthy, vazavam para multi_tf -> prompt da IA (texto "nan"),
histogram=nan nos builders e JSON inválido (RFC 8259 não admite NaN/Infinity).

Contrato (canônico em common/json_safe.py, verificado nos consumidores
ai_payload_builder, payload_compressor_v3, analyzer_qwen, common compressors
e ML feature_calculator — todos aceitam None via `or`-chains/`is not None`):
  - NaN/+Inf/-Inf -> None (vira JSON null; RSI=0 NUNCA significa ausente)
  - 0.0 continua 0.0 (valor legítimo/extremo preservado)
  - finitos continuam arredondados como antes
"""

import json
import math

import pandas as pd
import pytest
from unittest.mock import MagicMock

import fetchers.context_collector as cc_mod
from fetchers.context_collector import ContextCollector
from common.json_safe import json_dumps_rfc8259


def _make_collector(monkeypatch):
    for name in (
        "HistoricalVolumeProfiler",
        "TimeManager",
        "FREDFetcher",
        "ThreadPoolExecutor",
        "OnchainFetcher",
        "FundingAggregator",
        "BinancePositioningFetcher",
    ):
        monkeypatch.setattr(cc_mod, name, MagicMock(), raising=False)
    collector = ContextCollector(symbol="BTCUSDT")
    collector.timeframes = ["15m"]
    return collector


def _df(n=30):
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [100.5 + i * 0.1 for i in range(n)],
            "low": [99.5 + i * 0.1 for i in range(n)],
            "close": [100.0 + i * 0.1 for i in range(n)],
            "volume": [10.0 for _ in range(n)],
        }
    )


async def _fake_klines(self, session, symbol, timeframe, limit=200):
    return _df()


@pytest.mark.asyncio
async def test_rsi_nan_becomes_none(monkeypatch):
    c = _make_collector(monkeypatch)
    monkeypatch.setattr(cc_mod.ContextCollector, "_fetch_klines", _fake_klines)
    monkeypatch.setattr(c, "_calculate_rsi", lambda series, period: float("nan"))
    mtf = await c._analyze_mtf_trends(None)
    assert mtf["15m"]["rsi_short"] is None
    assert mtf["15m"]["rsi_long"] is None


@pytest.mark.asyncio
async def test_macd_inf_becomes_none(monkeypatch):
    c = _make_collector(monkeypatch)
    monkeypatch.setattr(cc_mod.ContextCollector, "_fetch_klines", _fake_klines)
    monkeypatch.setattr(c, "_calculate_macd", lambda s, fast, slow, signal: (float("inf"), float("-inf")))
    mtf = await c._analyze_mtf_trends(None)
    assert mtf["15m"]["macd"] is None
    assert mtf["15m"]["macd_signal"] is None


@pytest.mark.asyncio
async def test_zero_is_preserved_not_none(monkeypatch):
    """RSI=0 é valor legítimo e não deve virar None (ausente)."""
    c = _make_collector(monkeypatch)
    monkeypatch.setattr(cc_mod.ContextCollector, "_fetch_klines", _fake_klines)
    monkeypatch.setattr(c, "_calculate_rsi", lambda series, period: 0.0)
    monkeypatch.setattr(c, "_calculate_macd", lambda s, fast, slow, signal: (0.0, 0.0))
    mtf = await c._analyze_mtf_trends(None)
    assert mtf["15m"]["rsi_short"] == 0.0
    assert mtf["15m"]["macd"] == 0.0


@pytest.mark.asyncio
async def test_valid_values_still_rounded(monkeypatch):
    c = _make_collector(monkeypatch)
    monkeypatch.setattr(cc_mod.ContextCollector, "_fetch_klines", _fake_klines)
    monkeypatch.setattr(c, "_calculate_rsi", lambda series, period: 55.556)
    monkeypatch.setattr(c, "_calculate_adx", lambda df, period: 23.454)
    mtf = await c._analyze_mtf_trends(None)
    assert mtf["15m"]["rsi_short"] == 55.56
    assert mtf["15m"]["adx"] == 23.45


@pytest.mark.asyncio
async def test_mtf_json_has_no_nonfinite_literals(monkeypatch):
    """Prova fim-a-fonte: nenhum JSON final contém NaN/Infinity/-Infinity."""
    c = _make_collector(monkeypatch)
    monkeypatch.setattr(cc_mod.ContextCollector, "_fetch_klines", _fake_klines)
    monkeypatch.setattr(c, "_calculate_rsi", lambda series, period: float("nan"))
    monkeypatch.setattr(c, "_calculate_macd", lambda s, fast, slow, signal: (float("inf"), float("-inf")))
    monkeypatch.setattr(c, "_calculate_adx", lambda df, period: float("nan"))
    monkeypatch.setattr(c, "_calculate_realized_volatility", lambda series: float("nan"))
    mtf = await c._analyze_mtf_trends(None)
    text = json_dumps_rfc8259(mtf)  # lança se houver residual non-finite
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["15m"]["rsi_short"] is None
    assert parsed["15m"]["macd"] is None
    assert parsed["15m"]["adx"] is None
    assert parsed["15m"]["realized_vol"] is None
    for v in parsed["15m"].values():
        assert not (isinstance(v, float) and not math.isfinite(v))
