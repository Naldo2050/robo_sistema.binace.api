# tests/unit/test_orderbook_fallback_forensic_phase4b.py
"""
Testes forenses e de regressao para os achados #19A, #19B e #20 (Fase 4B).
Testes hermeticos com mocks de rede e circuit breaker.
"""

import asyncio
import time
from urllib.parse import urlparse
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from orderbook_analyzer.core import OrderBookAnalyzer
from orderbook_core.orderbook_fallback import OrderBookFallback, fetch_with_fallback, FallbackConfig


@pytest.fixture
def mock_depth_payload():
    return {
        "lastUpdateId": 10001,
        "E": int(time.time() * 1000),
        "T": int(time.time() * 1000),
        "bids": [["50000.0", "1.5"], ["49990.0", "2.0"]],
        "asks": [["50010.0", "1.2"], ["50020.0", "2.5"]],
    }


# ==============================================================================
# Caso A: Nenhum caminho de fallback aceita dados de Spot / Binance US
# ==============================================================================
def test_fallback_endpoints_must_only_contain_futures():
    fallback = OrderBookFallback()
    
    # Valida que nenhum endpoint Spot ou Binance US esta na lista
    for ep in fallback.endpoints:
        parsed = urlparse(ep)
        assert parsed.netloc != "api.binance.com", f"Endpoint Spot global proibido encontrado: {ep}"
        assert "binance.us" not in parsed.netloc, f"Endpoint Binance US proibido encontrado: {ep}"
        assert parsed.netloc == "fapi.binance.com", f"Endpoint deve ser Binance Futures USD-M: {ep}"


@pytest.mark.asyncio
async def test_fallback_does_not_call_spot_on_futures_failure():
    # Usar max_retries=1 e base_delay=0 para teste rapido e hermetico
    cfg = FallbackConfig(max_retries=1, base_delay=0.01)
    fallback = OrderBookFallback(config=cfg)
    
    session = MagicMock()
    # Simula erro no endpoint de futures
    session.get = MagicMock(side_effect=Exception("Futures connection error"))
    
    result = await fallback.fetch_orderbook_fallback(symbol="BTCUSDT", limit=50, session=session)
    
    assert result is None
    # Verifica todas as chamadas de URL feitas
    for call in session.get.call_args_list:
        url_called = call[0][0]
        parsed = urlparse(url_called)
        assert parsed.netloc != "api.binance.com", f"Chamou Spot proibido: {url_called}"
        assert "binance.us" not in parsed.netloc, f"Chamou US proibido: {url_called}"
        assert parsed.netloc == "fapi.binance.com", f"Deve chamar somente Futures: {url_called}"


# ==============================================================================
# Caso B: Com circuito OPEN, nao ocorre UnboundLocalError nem bypass do breaker
# ==============================================================================
@pytest.mark.asyncio
async def test_circuit_open_no_unbound_local_lim_and_no_network_bypass():
    oba = OrderBookAnalyzer(symbol="BTCUSDT")
    oba._circuit_breaker = MagicMock()
    oba._circuit_breaker.allow_request.return_value = False  # Circuito OPEN
    
    mock_session = MagicMock()
    mock_session.get = AsyncMock()
    oba._get_session = AsyncMock(return_value=mock_session)
    
    # Sem stale snapshot disponivel
    with oba._cache_lock:
        oba._last_valid_snapshot = None
        oba._cached_snapshot = None
    
    # Deve rodar sem UnboundLocalError e sem tentar chamada de rede
    res = await oba._fetch_orderbook(use_cache=False, allow_stale=True)
    
    assert res is None
    assert mock_session.get.call_count == 0, "Bypass do Circuit Breaker: tentou fazer requisicao durante OPEN!"
    assert oba._last_fetch_source == "circuit_open"


# ==============================================================================
# Caso C: Snapshot stale respeita limite de idade e mantem source 'stale'
# ==============================================================================
@pytest.mark.asyncio
async def test_stale_snapshot_within_and_outside_max_age(mock_depth_payload):
    oba = OrderBookAnalyzer(symbol="BTCUSDT")
    oba._circuit_breaker = MagicMock()
    oba._circuit_breaker.allow_request.return_value = False  # Circuito OPEN
    
    is_valid, _, converted = oba._validate_snapshot(mock_depth_payload)
    assert is_valid
    
    now_m = time.monotonic()
    
    # 1. Stale snapshot DENTRO do limite (age = 10s < fallback_max_age 120s)
    with oba._cache_lock:
        oba._last_valid_snapshot = oba._snapshot_copy(converted)
        oba._last_valid_timestamp_mono = now_m - 10.0
        oba._last_valid_exchange_ts = int(time.time() * 1000)
    
    res = await oba._fetch_orderbook(use_cache=False, allow_stale=True)
    assert res is not None
    assert oba._last_fetch_source == "stale"
    assert oba._last_fetch_age_seconds >= 10.0
    
    # 2. Stale snapshot FORA do limite (age = 200s > fallback_max_age 120s)
    with oba._cache_lock:
        oba._last_valid_snapshot = oba._snapshot_copy(converted)
        oba._last_valid_timestamp_mono = now_m - 200.0
        oba._last_valid_exchange_ts = int((time.time() - 200) * 1000)
    
    res_expired = await oba._fetch_orderbook(use_cache=False, allow_stale=True)
    assert res_expired is None, "Snapshot com idade superior a fallback_max_age deveria ser rejeitado!"


# ==============================================================================
# Caso D: fetch_with_fallback nao substitui excecao por NameError de logger
# ==============================================================================
@pytest.mark.asyncio
async def test_fetch_with_fallback_preserves_exception_without_name_error():
    fb_mock = MagicMock()
    fb_mock.fetch_orderbook_fallback = AsyncMock(side_effect=RuntimeError("Simulated Network Failure"))
    
    with patch("orderbook_core.orderbook_fallback.get_fallback_instance", return_value=fb_mock):
        with pytest.raises(RuntimeError, match="Simulated Network Failure"):
            await fetch_with_fallback("BTCUSDT")


# ==============================================================================
# Caso E: No caminho nominal, snapshot Futures valido continua funcionando
# ==============================================================================
@pytest.mark.asyncio
async def test_nominal_path_returns_live_snapshot(mock_depth_payload):
    oba = OrderBookAnalyzer(symbol="BTCUSDT")
    oba._circuit_breaker = MagicMock()
    oba._circuit_breaker.allow_request.return_value = True  # Circuito CLOSED
    
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=mock_depth_payload)
    
    # Context manager para session.get
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_resp
    mock_ctx.__aexit__.return_value = None
    
    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.get.return_value = mock_ctx
    oba._get_session = AsyncMock(return_value=mock_session)
    
    res = await oba._fetch_orderbook(limit=50, use_cache=False)
    
    assert res is not None
    assert oba._last_fetch_source == "live"
    assert oba._last_fetch_age_seconds == 0.0
    assert len(res["bids"]) == 2
    assert len(res["asks"]) == 2
