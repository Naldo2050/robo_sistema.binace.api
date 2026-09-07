# tests/unit/test_alphavantage_fail_closed.py
"""
Regressão de segurança (SEC-1): AlphaVantage fail-closed.

Bug: fetchers/context_collector.py carregava
  os.getenv("ALPHAVANTAGE_API_KEY", "<chave default embutida>")
ou seja, sem env var o código usava uma chave real embutida no fonte
(fail-silent + segredo vazado no repo).

Contrato exigido (fail-closed):
1. Sem ALPHAVANTAGE_API_KEY no ambiente -> collector.alpha_vantage_api_key is None
   (nunca um default embutido).
2. Com ALPHAVANTAGE_API_KEY definida -> a key do ambiente é usada (happy path).
3. Sem key -> _alpha_vantage_history retorna DataFrame vazio SEM tocar na rede
   (feature desabilitada, nenhuma chamada HTTP com apikey ausente/default).
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import fetchers.context_collector as cc_mod
from fetchers.context_collector import ContextCollector


def _make_collector(monkeypatch):
    """Constrói ContextCollector com dependências pesadas mockadas."""
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
    return ContextCollector(symbol="BTCUSDT")


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"Error Message": " apareceu"}

    async def text(self):
        return "erro"


class _FakeSession:
    def __init__(self):
        self.get_calls = []

    def get(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        return _FakeResponse()


def test_no_hardcoded_default_key(monkeypatch):
    """Sem env var, a key deve ser None — nunca um default embutido."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    collector = _make_collector(monkeypatch)
    assert collector.alpha_vantage_api_key is None


def test_env_key_is_used(monkeypatch):
    """Com env var, a key do ambiente é usada (happy path preservado)."""
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "TESTKEY123")
    collector = _make_collector(monkeypatch)
    assert collector.alpha_vantage_api_key == "TESTKEY123"


@pytest.mark.asyncio
async def test_history_fail_closed_without_key(monkeypatch):
    """Sem key, _alpha_vantage_history não faz HTTP e retorna DF vazio."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    collector = _make_collector(monkeypatch)
    session = _FakeSession()
    df = await collector._alpha_vantage_history(session, "BTCUSDT")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert session.get_calls == []
