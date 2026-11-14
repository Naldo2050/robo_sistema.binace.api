# test_cross_asset_integration.py
# -*- coding: utf-8 -*-
"""
Teste de Integracao das Features Cross-Asset com hermeticidade de transporte.
Injeta dados sinteticos conhecidos na camada de transporte (requests.get)
e valida metricas e correlacoes contra calculos independentes com numpy.corrcoef.
"""

from datetime import datetime, timezone
import logging
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from common.ml_features import generate_ml_features
from market_orchestrator.ai.ai_payload_builder import build_ai_input

logger = logging.getLogger(__name__)

# Constantes e gerador de dados sinteticos
N_KLINES = 724
START_TS = 1700000000000  # ms
HOUR_MS = 3600000

# Series deterministicas de precos em funcao do indice i
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
def transport_layer_mock(monkeypatch):
    monkeypatch.setattr("requests.get", fake_requests_get)


def create_test_data():
    test_df = pd.DataFrame({
        'p': [45000, 45100, 45050, 45200, 45150, 45300, 45250],
        'q': [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5],
        'm': [False, True, True, False, True, False, True],
    })
    orderbook_data = {
        "order_book_depth": {"L1": {"bids": 100000, "asks": 95000}},
        "spread_metrics": {"spread": 5.0},
        "bid_depth_usd": 100000, "ask_depth_usd": 95000,
        "imbalance": 0.05, "spread_percent": 0.011
    }
    flow_metrics = {
        "order_flow": {"net_flow_1m": -1500.0, "buy_sell_ratio": 0.8},
        "bursts": {"count": 5}, "metadata": {"burst_window_ms": 1000},
        "cvd": -5000, "whale_delta": 1000, "whale_buy_volume": 500,
        "whale_sell_volume": 1500, "tipo_absorcao": "Venda"
    }
    historical_profile = {"daily": {"poc": 45100, "vah": 45300, "val": 44900}}
    macro_context = {
        "trading_session": "NY", "session_phase": "ACTIVE",
        "mtf_trends": {"1h": {"tendencia": "Alta"}}, "atr": 500.0
    }
    market_environment = {
        "volatility_regime": "NORMAL", "trend_direction": "UP",
        "market_structure": "ACCUMULATION", "risk_sentiment": "BULLISH",
        "correlation_spy": 0.3, "correlation_dxy": -0.6
    }
    signal = {
        "tipo_evento": "ANALYSIS_TRIGGER", "symbol": "BTCUSDT",
        "preco_fechamento": 45250.0, "volume_total": 15.5, "delta": -250.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resultado_da_batalha": "VENDEDOR_VENCEDOR", "janela_numero": 1001
    }
    return {
        "df": test_df, "orderbook_data": orderbook_data,
        "flow_metrics": flow_metrics, "historical_profile": historical_profile,
        "macro_context": macro_context, "market_environment": market_environment,
        "signal": signal
    }


def test_ml_features_cross_asset():
    logger.info("TESTANDO: ml_features.py com cross-asset features hermeticas")
    test_data = create_test_data()
    ml_features = generate_ml_features(
        df=test_data["df"],
        orderbook_data=test_data["orderbook_data"],
        flow_metrics=test_data["flow_metrics"],
        symbol="BTCUSDT"
    )

    assert "cross_asset" in ml_features
    cross_asset = ml_features["cross_asset"]
    assert len(cross_asset) == 41
    assert cross_asset["btc_eth_corr_7d"] == round(EXP_CORR_7D, 4)
    assert cross_asset["btc_eth_corr_30d"] == round(EXP_CORR_30D, 4)
    assert cross_asset["btc_eth_corr_7d_n"] == 168
    assert cross_asset["btc_eth_corr_30d_n"] == 720
    assert cross_asset["cross_asset_method"] == "shared_session_returns_v2"
    assert cross_asset["cross_asset_contract_version"] == 2


def test_ai_payload_cross_asset():
    logger.info("TESTANDO: ai_payload_builder.py com cross_asset_context hermetico")
    test_data = create_test_data()
    ml_features = generate_ml_features(
        df=test_data["df"],
        orderbook_data=test_data["orderbook_data"],
        flow_metrics=test_data["flow_metrics"],
        symbol="BTCUSDT"
    )
    ai_payload = build_ai_input(
        symbol="BTCUSDT",
        signal=test_data["signal"],
        enriched={"ohlc": {"close": 45250.0}},
        flow_metrics=test_data["flow_metrics"],
        historical_profile=test_data["historical_profile"],
        macro_context=test_data["macro_context"],
        market_environment=test_data["market_environment"],
        orderbook_data=test_data["orderbook_data"],
        ml_features=ml_features
    )

    assert "cross_asset_context" in ai_payload
    cross_asset_context = ai_payload["cross_asset_context"]
    assert "btc_eth_correlations" in cross_asset_context

    correlations = cross_asset_context["btc_eth_correlations"]
    assert correlations["short_term_7d"] == round(EXP_CORR_7D, 4)
    assert correlations["long_term_30d"] == round(EXP_CORR_30D, 4)


def test_independent_numpy_math():
    calc_7d = calc_independent_corr(BTC_PRICES, ETH_PRICES, 168)
    calc_30d = calc_independent_corr(BTC_PRICES, ETH_PRICES, 720)
    assert round(calc_7d, 4) == 0.9999
    assert round(calc_30d, 4) == 0.9996
