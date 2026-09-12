# tests/unit/test_p04_onchain_provenance.py
"""P04: API_ERROR/MISSING nunca vira zero real; REAL_ZERO preservado e distinguível.

Cadeia: fetcher -> merge -> updater snapshot -> data_enricher -> evento -> payload.
Escopo estrito P04 (sem P05/indicadores/CVD/absorção/naming/orderbook/dedup).
"""
import pytest

from fetchers.onchain_fetcher import OnchainFetcher


class _FakeResp:
    def __init__(self, status=200, json_data=None, text_data="", exc=None):
        self.status = status
        self._json = json_data
        self._text = text_data
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    async def text(self):
        return self._text


class _FakeSession:
    """Mapeia prefixo de URL -> _FakeResp ou exceção."""

    def __init__(self, routes):
        self._routes = routes

    def get(self, url, timeout=None):
        for prefix, resp in self._routes.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    return _FakeResp(exc=resp)
                return resp
        return _FakeResp(status=404)


def _fees_routes(fees_json=None, fees_status=200, mempool_json=None, mempool_status=200):
    routes = {
        "https://mempool.space/api/v1/fees/recommended": _FakeResp(
            status=fees_status, json_data=fees_json),
        "https://mempool.space/api/mempool": _FakeResp(
            status=mempool_status, json_data=mempool_json),
        "https://mempool.space/api/v1/difficulty-adjustment": _FakeResp(
            status=500, json_data=None),
    }
    return routes


def _run(coro):
    import asyncio
    return asyncio.run(coro)


# CASO A: API retorna fee=5 -> value=5 VALID
def test_a_valid_fee_preserved():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json={"fastestFee": 5, "halfHourFee": 3, "hourFee": 1,
                   "economyFee": 1, "minimumFee": 1},
        mempool_json={"count": 100, "vsize": 5000000, "total_fee": 100000},
    ))))
    assert data["fees"]["fastest_sat_vb"] == 5
    assert data["_status"]["fees.fastest_sat_vb"] == "VALID"
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] == 5
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "VALID"


# CASO B: API retorna fee=0 explícito -> 0 REAL_ZERO (zero preservado)
def test_b_explicit_zero_is_real_zero_not_missing():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json={"fastestFee": 0, "halfHourFee": 0, "hourFee": 0,
                   "economyFee": 0, "minimumFee": 0},
        mempool_json={"count": 10, "vsize": 1000, "total_fee": 50},
    ))))
    assert data["fees"]["fastest_sat_vb"] == 0
    assert data["_status"]["fees.fastest_sat_vb"] == "REAL_ZERO"
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] == 0
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "REAL_ZERO"


# CASO C: timeout -> None API_ERROR, nunca 0
def test_c_timeout_is_none_api_error():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession({
        "https://mempool.space/api/v1/fees/recommended": TimeoutError("t"),
        "https://mempool.space/api/mempool": _FakeResp(
            status=200, json_data={"count": 10, "vsize": 1000, "total_fee": 50}),
        "https://mempool.space/api/v1/difficulty-adjustment": _FakeResp(
            status=500, json_data=None),
    })))
    assert "fees" not in data
    assert data["_status"]["fees"] == "API_ERROR"
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] is None
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"


# CASO D: HTTP 500 -> None API_ERROR
def test_d_http_500_is_none_api_error():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json=None, fees_status=500,
        mempool_json={"count": 10, "vsize": 1000, "total_fee": 50},
    ))))
    assert "fees" not in data
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] is None
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"


# CASO E: JSON inválido -> None API_ERROR
def test_e_invalid_json_is_none_api_error():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json=ValueError("bad json"),
        mempool_json={"count": 10, "vsize": 1000, "total_fee": 50},
    ))))
    assert "fees" not in data
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] is None
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"


# CASO F: JSON válido sem campo -> None MISSING
def test_f_valid_json_missing_field_is_none_missing():
    f = OnchainFetcher()
    data = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json={"halfHourFee": 3, "hourFee": 1, "economyFee": 1, "minimumFee": 1},
        mempool_json={"count": 10, "vsize": 1000, "total_fee": 50},
    ))))
    assert "fastest_sat_vb" not in data.get("fees", {})
    assert data["_status"]["fees.fastest_sat_vb"] == "MISSING"
    merged = f._merge_metrics({}, data)
    assert merged["fees_fastest_sat_vb"] is None
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "MISSING"


# TESTE DE REAL ZERO: resposta explícita 0 vs exceção são distinguíveis
def test_real_zero_distinguishable_from_api_error():
    f = OnchainFetcher()
    ok = _run(f._fetch_mempool_space(_FakeSession(_fees_routes(
        fees_json={"fastestFee": 0, "halfHourFee": 0, "hourFee": 0,
                   "economyFee": 0, "minimumFee": 0},
        mempool_json={"count": 1, "vsize": 100, "total_fee": 10},
    ))))
    bad = _run(f._fetch_mempool_space(_FakeSession({
        "https://mempool.space/api/v1/fees/recommended": ConnectionError("x"),
        "https://mempool.space/api/mempool": _FakeResp(
            status=200, json_data={"count": 1, "vsize": 100, "total_fee": 10}),
        "https://mempool.space/api/v1/difficulty-adjustment": _FakeResp(
            status=500, json_data=None),
    })))
    m_ok = f._merge_metrics({}, ok)
    m_bad = f._merge_metrics({}, bad)
    assert m_ok["fees_fastest_sat_vb"] == 0
    assert m_ok["_field_status"]["fees_fastest_sat_vb"] == "REAL_ZERO"
    assert m_bad["fees_fastest_sat_vb"] is None
    assert m_bad["_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"
    # numericamente iguais em valor, semanticamente distintos em status
    assert (m_ok["_field_status"]["fees_fastest_sat_vb"]
            != m_bad["_field_status"]["fees_fastest_sat_vb"])


# CASO G: blockchain OK + mempool FAIL -> parcial preservado
def test_g_partial_sources_preserved_per_field():
    f = OnchainFetcher()
    blockchain = {"unconfirmed_txs": 30073.0,
                  "_status": {"unconfirmed_txs": "VALID"}}
    mempool = {"_status": {"fees": "API_ERROR", "mempool": "API_ERROR"}}
    merged = f._merge_metrics(blockchain, mempool)
    # fonte boa preservada (fallback blockchain p/ mempool_size)
    assert merged["mempool_size"] == 30073.0
    assert merged["_field_status"]["mempool_size"] == "VALID"
    # fonte falha => None + API_ERROR (nunca 0)
    assert merged["mempool_vsize_mb"] is None
    assert merged["_field_status"]["mempool_vsize_mb"] == "API_ERROR"
    assert merged["fees_fastest_sat_vb"] is None
    assert merged["_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"


# CASO H/I: cache válido preserva + flag; expirado => stale explícito
def test_h_valid_cache_preserved_with_flag():
    from fetchers.onchain_updater import OnchainUpdater
    now = [1000.0]
    updater = OnchainUpdater(monotonic_fn=lambda: now[0])
    updater._store_snapshot(
        fast={"mempool_size": 30073.0, "fees_fastest_sat_vb": 5},
        slow={"difficulty": 127.45},
        field_status={"mempool_size": "VALID", "fees_fastest_sat_vb": "VALID"},
    )
    view = updater.read_view()
    assert view["fast"]["status"] == "fresh"
    assert view["fast"]["values"]["fees_fastest_sat_vb"] == 5
    assert view["field_status"]["fees_fastest_sat_vb"] == "VALID"


def test_i_expired_cache_is_stale_explicit():
    from fetchers.onchain_updater import OnchainUpdater
    now = [1000.0]
    updater = OnchainUpdater(monotonic_fn=lambda: now[0])
    updater._store_snapshot(
        fast={"mempool_size": 30073.0},
        slow={},
        field_status={"mempool_size": "VALID"},
    )
    now[0] += 1000.0  # dentro do usable fast (1800s) => stale
    view = updater.read_view()
    assert view["fast"]["status"] == "stale"
    assert view["fast"]["age_seconds"] == pytest.approx(1000.0)


# EVENTO: API_ERROR -> null + status; REAL_ZERO -> 0 (sem None->0 no enrichment)
def test_event_api_error_is_null_with_status_and_real_zero_is_zero():
    from data_processing.data_enricher import DataEnricher
    from fetchers.onchain_updater import OnchainUpdater
    now = [5000.0]
    updater = OnchainUpdater(monotonic_fn=lambda: now[0])
    updater._store_snapshot(
        fast={"mempool_size": 30073.0, "fees_fastest_sat_vb": None},
        slow={"difficulty": 127.45},
        field_status={"mempool_size": "VALID",
                      "fees_fastest_sat_vb": "API_ERROR",
                      "mempool_vsize_mb": "API_ERROR"},
    )
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    metrics = enricher._build_onchain_metrics()
    assert metrics["mempool_size"] == 30073.0
    assert "fees_fastest_sat_vb" not in metrics  # None não vira 0
    event = {"raw_event": {"symbol": "BTCUSDT", "preco_fechamento": 80000.0,
                           "volume_total": 1.0}}
    enricher.enrich_event_with_advanced_analysis(event)
    adv = event["raw_event"]["advanced_analysis"]
    assert adv["onchain_metrics"].get("fees_fastest_sat_vb") is None
    assert adv["onchain_field_status"]["fees_fastest_sat_vb"] == "API_ERROR"
    assert adv["onchain_metrics"]["mempool_size"] == 30073.0

    # REAL_ZERO flui como 0 (não é apagado por `if value:`)
    updater2 = OnchainUpdater(monotonic_fn=lambda: now[0])
    updater2._store_snapshot(
        fast={"fees_fastest_sat_vb": 0, "mempool_size": 30073.0},
        slow={},
        field_status={"fees_fastest_sat_vb": "REAL_ZERO"},
    )
    enricher2 = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater2)
    assert enricher2._build_onchain_metrics()["fees_fastest_sat_vb"] == 0


# PAYLOAD: erro omitido (nunca fees_fast=0); zero real preservado
def test_payload_omits_error_but_keeps_real_zero():
    from market_orchestrator.ai.payload_builder_compact import _build_onchain
    err_event = {"raw_event": {"advanced_analysis": {
        "onchain_metrics": {"mempool_size": 30073.0, "fees_fastest_sat_vb": None},
        "onchain_status": "fresh", "onchain_age_seconds": 5.0,
        "onchain_source": "blockchain.info+mempool.space",
        "onchain_field_status": {"fees_fastest_sat_vb": "API_ERROR"},
    }}}
    out = _build_onchain(err_event)
    assert "fees_fast" not in out  # nunca 0 fabricado
    assert out["mempool_sz"] == 30073.0
    assert out["st"] == "fresh"

    zero_event = {"raw_event": {"advanced_analysis": {
        "onchain_metrics": {"mempool_size": 30073.0, "fees_fastest_sat_vb": 0},
        "onchain_status": "fresh", "onchain_age_seconds": 5.0,
        "onchain_source": "blockchain.info+mempool.space",
        "onchain_field_status": {"fees_fastest_sat_vb": "REAL_ZERO"},
    }}}
    out0 = _build_onchain(zero_event)
    assert out0["fees_fast"] == 0  # `if value is not None`, não `if value`
