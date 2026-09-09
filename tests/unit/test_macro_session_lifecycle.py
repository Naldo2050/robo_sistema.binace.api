# tests/unit/test_macro_session_lifecycle.py
"""PF-M4: lifecycle das sessões MacroDataProvider (sem rede real).

Contrato:
  1. 20 refreshes (cache-miss) via caminho real do cross: registry bounded.
  2. Nenhum warning "Unclosed client session".
  3. Com MacroUpdateService ativo (sessão em loop persistente): cross refresh
     não fecha a sessão em uso pelo service.
  4. Shutdown global: registry final == 0, sessões fechadas, idempotente.
  5. Sessão de loop efêmero nunca é reutilizada em loop diferente.
  6. Novo fetch após shutdown/restart: provider recria sessão corretamente.

Folhas de rede stubadas APÓS _get_session: registram sessão real, sem I/O.
"""

import asyncio
import gc
import warnings

import pytest

from fetchers.macro_data_provider import MacroDataProvider


@pytest.fixture
def provider():
    MacroDataProvider.reset_instance()
    p = MacroDataProvider.get_instance()
    yield p
    # Higiene: não vazar sessões para outros testes.
    try:
        asyncio.run(p.close_all_sessions())
    except Exception:
        pass
    MacroDataProvider.reset_instance()


def _stub_leaves(monkeypatch, canned):
    async def _leaf(self):
        await MacroDataProvider.get_instance()._get_session()
        await asyncio.sleep(0)
        return canned

    async def _vix(self):
        return await _leaf(self)

    monkeypatch.setattr(MacroDataProvider, "get_vix", _vix)
    monkeypatch.setattr(MacroDataProvider, "get_treasury_10y", _vix)
    monkeypatch.setattr(MacroDataProvider, "get_dxy", _vix)
    monkeypatch.setattr(MacroDataProvider, "get_sp500", _vix)
    monkeypatch.setattr(MacroDataProvider, "get_gold_price", _vix)
    monkeypatch.setattr(MacroDataProvider, "get_oil_price", _vix)
    monkeypatch.setattr(MacroDataProvider, "calculate_btc_dominance", _vix)
    monkeypatch.setattr(MacroDataProvider, "calculate_eth_dominance", _vix)


def _miss_cycle(provider):
    """Um refresh com cache-miss pelo caminho real do cross (loop efêmero)."""
    import market_analysis.cross_asset_correlations as ca

    provider.clear_cache()
    return asyncio.run(ca._get_macro_data_async())


def _open_sessions(provider):
    return [s for s in provider._sessions.values() if not s.closed]


def test_1_20_refreshes_registry_bounded(provider, monkeypatch):
    _stub_leaves(monkeypatch, 1.0)
    for _ in range(20):
        _miss_cycle(provider)
    assert len(provider._sessions) <= 1, dict(provider._sessions)
    assert _open_sessions(provider) == []


def test_2_no_unclosed_client_session_warnings(provider, monkeypatch, recwarn):
    _stub_leaves(monkeypatch, 1.0)
    for _ in range(5):
        _miss_cycle(provider)
    gc.collect()
    assert not [w for w in recwarn.list if "Unclosed client session" in str(w.message)]


def test_3_cross_refresh_preserves_service_session(provider, monkeypatch):
    """Sessão do service (loop persistente) sobrevive a refreshes do cross."""
    _stub_leaves(monkeypatch, 1.0)
    service_loop = asyncio.new_event_loop()
    try:
        service_session = service_loop.run_until_complete(provider._get_session())
        service_key = id(service_loop)
        assert service_key in provider._sessions
        for _ in range(3):
            _miss_cycle(provider)
        assert service_key in provider._sessions, "sessão do service foi removida!"
        assert not service_session.closed, "sessão em uso pelo service foi fechada!"
        assert _open_sessions(provider) == [service_session]
    finally:
        try:
            service_loop.run_until_complete(
                provider._close_session_for_loop(service_key))
        except Exception:
            pass
        service_loop.close()


def test_4_global_shutdown_idempotent(provider):
    """Shutdown fecha até sessões de loops mortos/estranhos sem estourar."""
    other = asyncio.new_event_loop()
    try:
        s_dead_loop = other.run_until_complete(provider._get_session())
    finally:
        other.close()  # loop morto com sessão ainda aberta

    async def _mk():
        return await provider._get_session()

    s_main = asyncio.run(_mk())  # asyncio.run fecha o loop1 ao sair
    assert not s_dead_loop.closed and not s_main.closed
    # Fecha de loops diferentes (cenário M3): não deve levantar.
    asyncio.run(provider.close_all_sessions())
    assert dict(provider._sessions) == {}
    # Idempotente: segunda chamada não quebra.
    asyncio.run(provider.close_all_sessions())
    assert dict(provider._sessions) == {}


def test_5_no_reuse_across_loops(provider, monkeypatch):
    _stub_leaves(monkeypatch, 1.0)
    _miss_cycle(provider)
    assert _open_sessions(provider) == []
    # Novo loop: fetch funciona e não reaproveita sessão morta.
    data = _miss_cycle(provider)
    assert data.get("vix") == 1.0
    assert _open_sessions(provider) == []


def test_6_fetch_after_shutdown_restart(provider, monkeypatch):
    _stub_leaves(monkeypatch, 2.0)
    _miss_cycle(provider)
    asyncio.run(provider.close_all_sessions())
    assert dict(provider._sessions) == {}
    data = _miss_cycle(provider)
    assert data.get("gold") == 2.0
    assert _open_sessions(provider) == []
    assert len(provider._sessions) <= 1


def test_close_current_loop_only_closes_current(provider):
    # Tudo sequencial, sem loop aninhado: cada run_until_complete usa um
    # loop parado diferente (chamar run dentro de loop rodando é ilegal).
    loop1 = asyncio.new_event_loop()
    try:
        s_main = loop1.run_until_complete(provider._get_session())
        other = asyncio.new_event_loop()
        try:
            s_other = other.run_until_complete(provider._get_session())
            n = loop1.run_until_complete(
                provider.close_sessions_for_current_loop())
            assert n == 1
            assert s_main.closed
            assert not s_other.closed
            assert id(other) in provider._sessions
        finally:
            try:
                other.run_until_complete(
                    provider._close_session_for_loop(id(other)))
            except Exception:
                pass
            other.close()
    finally:
        loop1.close()
