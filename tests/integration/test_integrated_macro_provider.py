# test_integrated_macro_provider.py

# -*- coding: utf-8 -*-

"""

Teste de integracao do MacroDataProvider com o sistema de correlacoes cross-asset.

Injeta dados sinteticos na camada de transporte (aiohttp e requests) e valida

metricas com asserts de valores concretos.

"""




import logging

import math

from unittest.mock import MagicMock



import numpy as np

import pytest



from common.ml_features import calculate_cross_asset_features

from fetchers.macro_data_provider import MacroDataProvider

from market_analysis.cross_asset_correlations import get_enhanced_cross_asset_correlations



logger = logging.getLogger(__name__)



# Constantes sinteticas

N_KLINES = 724

START_TS = 1700000000000

HOUR_MS = 3600000



BTC_PRICES = [50000.0 + 500.0 * math.sin(i / 12.0) + 200.0 * math.cos(i / 24.0) + 10.0 * i for i in range(N_KLINES)]

ETH_PRICES = [3000.0 + 40.0 * math.sin(i / 12.0) + 15.0 * math.cos(i / 24.0) + 1.0 * i for i in range(N_KLINES)]





def build_binance_klines_payload(prices: list[float], start_ms: int) -> list[list]:

    klines = []

    for i, p in enumerate(prices):

        ot = start_ms + i * HOUR_MS

        ct = ot + HOUR_MS - 1

        p_str = f"{p:.4f}"

        klines.append([

            ot, p_str, f"{p + 5:.4f}", f"{p - 5:.4f}", p_str, "100.0",

            ct, "5000000.0", 1000, "50.0", "2500000.0", "0"

        ])

    return klines





BTC_KLINES_PAYLOAD = build_binance_klines_payload(BTC_PRICES, START_TS)

ETH_KLINES_PAYLOAD = build_binance_klines_payload(ETH_PRICES, START_TS)





def calc_independent_corr(px_a: list[float], px_b: list[float], window: int) -> float:

    tail_a = px_a[-(window + 1):]

    tail_b = px_b[-(window + 1):]

    ra = np.log(np.array(tail_a[1:]) / np.array(tail_a[:-1]))

    rb = np.log(np.array(tail_b[1:]) / np.array(tail_b[:-1]))

    return float(np.corrcoef(ra, rb)[0, 1])





EXP_CORR_7D = calc_independent_corr(BTC_PRICES, ETH_PRICES, 168)

EXP_CORR_30D = calc_independent_corr(BTC_PRICES, ETH_PRICES, 720)





class MockAiohttpResponse:

    def __init__(self, data, status=200):

        self._data = data

        self.status = status



    async def json(self):

        return self._data



    async def text(self):

        import json

        return json.dumps(self._data)



    async def __aenter__(self):

        return self



    async def __aexit__(self, exc_type, exc_val, exc_tb):

        pass





def fake_aiohttp_get(self, url, *args, **kwargs):

    url_str = str(url)

    if "api.coingecko.com/api/v3/global" in url_str:

        return MockAiohttpResponse({"data": {"market_cap_percentage": {"btc": 54.2, "eth": 17.5}}})

    elif "api.binance.com/api/v3/ticker/24hr" in url_str:

        return MockAiohttpResponse([

            {"symbol": "BTCUSDT", "quoteVolume": "5420000000"},

            {"symbol": "ETHUSDT", "quoteVolume": "1750000000"},

        ])

    elif "api.twelvedata.com/time_series" in url_str:

        if "symbol=TNX" in url_str:

            return MockAiohttpResponse({"values": [{"close": "4.25"}]})

        elif "symbol=^GSPC" in url_str:

            return MockAiohttpResponse({"values": [{"close": "5800.0"}]})

        elif "symbol=XAU/USD" in url_str:

            return MockAiohttpResponse({"values": [{"close": "2650.0"}]})

    return MockAiohttpResponse({})





def fake_requests_get(url, *args, **kwargs):

    mock_resp = MagicMock()

    mock_resp.status_code = 200

    mock_resp.raise_for_status = MagicMock()

    params = kwargs.get("params", {})

    symbol = params.get("symbol", "")



    if "fapi.binance.com/fapi/v1/klines" in str(url):

        if symbol == "BTCUSDT":

            mock_resp.json = MagicMock(return_value=BTC_KLINES_PAYLOAD)

        elif symbol == "ETHUSDT":

            mock_resp.json = MagicMock(return_value=ETH_KLINES_PAYLOAD)

        else:

            mock_resp.json = MagicMock(return_value=[])

        return mock_resp



    mock_resp.json = MagicMock(return_value={})

    return mock_resp





@pytest.fixture(autouse=True)

def macro_transport_mock(monkeypatch):

    MacroDataProvider.reset_instance()

    provider = MacroDataProvider()

    with provider._cache_lock:

        provider._cache.clear()

    monkeypatch.setattr("aiohttp.ClientSession.get", fake_aiohttp_get)

    monkeypatch.setattr("requests.get", fake_requests_get)

    yield

    MacroDataProvider.reset_instance()





@pytest.mark.asyncio

async def test_macro_provider_integration():

    """Testa integracao completa do sistema com dados sinteticos e asserts concretos."""

    provider = MacroDataProvider()

    macro_data = await provider.get_all_macro_data()



    assert macro_data["btc_dominance"] == 54.2



    correlations = get_enhanced_cross_asset_correlations()

    assert correlations["status"] == "ok"

    assert correlations["btc_eth_corr_7d"] == round(EXP_CORR_7D, 4)

    assert correlations["btc_eth_corr_30d"] == round(EXP_CORR_30D, 4)

    assert correlations["btc_eth_corr_7d_n"] == 168

    assert correlations["btc_eth_corr_30d_n"] == 720

    assert correlations["macro_regime"] == "UNKNOWN"



    features = calculate_cross_asset_features("BTCUSDT")

    assert len(features) == 41

    assert features["btc_eth_corr_7d"] == round(EXP_CORR_7D, 4)

    assert features["btc_eth_corr_30d"] == round(EXP_CORR_30D, 4)

    assert features["btc_eth_corr_7d_n"] == 168

    assert features["btc_eth_corr_30d_n"] == 720

    assert features["cross_asset_method"] == "shared_session_returns_v2"

