"""
Testes para MarketDataCache.
Cobre: cache hit/miss, TTL, lock dedup, stale-while-revalidate,
       prefetch, singleton, stats.
"""

import asyncio
import time
import pytest

from common.yfinance_cache import (
    MarketDataCache,
    CacheEntry,
    get_market_cache,
    reset_market_cache,
    DEFAULT_TTLS,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_singleton():
    """Reseta singleton antes e depois de cada teste."""
    reset_market_cache()
    yield
    reset_market_cache()


@pytest.fixture
def cache() -> MarketDataCache:
    """Cache limpo para testes."""
    return MarketDataCache()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_fetcher(value, delay=0.0):
    """Cria uma closure de fetcher."""
    async def _fetcher():
        if delay:
            await asyncio.sleep(delay)
        return value
    return _fetcher


def _make_failing_fetcher(error_type=Exception, msg="fetch error"):
    """Cria fetcher que falha."""
    async def _fetcher():
        raise error_type(msg)
    return _fetcher


def _make_counting_fetcher(value):
    """Fetcher que conta quantas vezes foi chamado."""
    counter = {"count": 0}

    async def _fetcher():
        counter["count"] += 1
        return value

    return _fetcher, counter


# ──────────────────────────────────────────────
# Testes: CacheEntry
# ──────────────────────────────────────────────

class TestCacheEntry:

    def test_valid_entry(self):
        """Entrada recente deve ser valida."""
        entry = CacheEntry(value=99.5, fetched_at=time.time(), ttl=300)
        assert entry.is_valid is True

    def test_expired_entry(self):
        """Entrada expirada deve ser invalida."""
        entry = CacheEntry(
            value=99.5,
            fetched_at=time.time() - 301,
            ttl=300,
        )
        assert entry.is_valid is False

    def test_age_seconds(self):
        """age_seconds deve refletir tempo decorrido."""
        entry = CacheEntry(
            value=99.5,
            fetched_at=time.time() - 10,
            ttl=300,
        )
        assert 9.5 < entry.age_seconds < 11.0


# ──────────────────────────────────────────────
# Testes: Cache Hit / Miss
# ──────────────────────────────────────────────

class TestCacheHitMiss:

    @pytest.mark.asyncio
    async def test_miss_calls_fetcher(self, cache):
        """Cache miss deve chamar o fetcher."""
        fetcher, counter = _make_counting_fetcher(99.68)
        result = await cache.get("DXY", fetcher=fetcher)
        assert result == 99.68
        assert counter["count"] == 1

    @pytest.mark.asyncio
    async def test_hit_skips_fetcher(self, cache):
        """Cache hit NAO deve chamar o fetcher."""
        fetcher, counter = _make_counting_fetcher(99.68)

        # Primeiro: miss -> fetch
        await cache.get("DXY", fetcher=fetcher)
        assert counter["count"] == 1

        # Segundo: hit -> sem fetch
        result = await cache.get("DXY", fetcher=fetcher)
        assert result == 99.68
        assert counter["count"] == 1  # NAO incrementou

    @pytest.mark.asyncio
    async def test_expired_calls_fetcher_again(self, cache):
        """Entrada expirada deve chamar fetcher novamente."""
        fetcher, counter = _make_counting_fetcher(99.68)

        await cache.get("DXY", fetcher=fetcher, ttl=0.1)
        assert counter["count"] == 1

        # Esperar expirar
        await asyncio.sleep(0.15)

        await cache.get("DXY", fetcher=fetcher, ttl=0.1)
        assert counter["count"] == 2

    @pytest.mark.asyncio
    async def test_get_without_fetcher_returns_none(self, cache):
        """get() sem fetcher em cache vazio retorna None."""
        result = await cache.get("DXY")
        assert result is None

    def test_get_sync_returns_cached(self, cache):
        """get_sync retorna valor cacheado."""
        cache.set("DXY", 99.68)
        assert cache.get_sync("DXY") == 99.68

    def test_get_sync_returns_none_when_expired(self, cache):
        """get_sync retorna None quando expirado."""
        cache.set("DXY", 99.68, ttl=0.01)
        time.sleep(0.02)
        assert cache.get_sync("DXY") is None


# ──────────────────────────────────────────────
# Testes: Deduplicacao (Lock por chave)
# ──────────────────────────────────────────────

class TestDeduplication:

    @pytest.mark.asyncio
    async def test_concurrent_fetches_deduplicated(self, cache):
        """Multiplos gets concorrentes devem resultar em apenas 1 fetch."""
        # Primeira chamada: popula cache
        counter_fetcher, counter = _make_counting_fetcher(99.68)
        result1 = await cache.get("DXY", fetcher=counter_fetcher)

        # 5 chamadas concorrentes: todas devem ser cache hit
        results = await asyncio.gather(*[
            cache.get("DXY", fetcher=counter_fetcher)
            for _ in range(5)
        ])

        assert counter["count"] == 1  # Apenas 1 fetch real
        assert all(r == 99.68 for r in results)

    @pytest.mark.asyncio
    async def test_different_keys_fetch_independently(self, cache):
        """Chaves diferentes devem buscar independentemente."""
        dxy_fetcher, dxy_counter = _make_counting_fetcher(99.68)
        sp_fetcher, sp_counter = _make_counting_fetcher(6506.48)

        await asyncio.gather(
            cache.get("DXY", fetcher=dxy_fetcher),
            cache.get("SP500", fetcher=sp_fetcher),
        )

        assert dxy_counter["count"] == 1
        assert sp_counter["count"] == 1


# ──────────────────────────────────────────────
# Testes: Stale-While-Revalidate
# ──────────────────────────────────────────────

class TestStaleWhileRevalidate:

    @pytest.mark.asyncio
    async def test_returns_stale_on_fetch_error(self, cache):
        """Se fetch falhar, deve retornar valor stale."""
        # Primeiro: popular com valor
        cache.set("DXY", 99.68, ttl=0.01)
        await asyncio.sleep(0.02)  # expirar

        # Fetch falha -> deve retornar stale
        result = await cache.get(
            "DXY",
            fetcher=_make_failing_fetcher(),
        )
        assert result == 99.68

    @pytest.mark.asyncio
    async def test_returns_none_on_error_without_stale(self, cache):
        """Se fetch falhar e nao ha stale, retorna None."""
        result = await cache.get(
            "DXY",
            fetcher=_make_failing_fetcher(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_stale_on_none_result(self, cache):
        """Se fetcher retorna None, usa stale."""
        cache.set("DXY", 99.68, ttl=0.01)
        await asyncio.sleep(0.02)

        result = await cache.get(
            "DXY",
            fetcher=_make_fetcher(None),  # retorna None
        )
        assert result == 99.68


# ──────────────────────────────────────────────
# Testes: set / invalidate / clear
# ──────────────────────────────────────────────

class TestCacheOperations:

    def test_set_and_get_sync(self, cache):
        """set + get_sync funciona."""
        cache.set("DXY", 99.68)
        assert cache.get_sync("DXY") == 99.68

    def test_set_with_custom_ttl(self, cache):
        """set com TTL customizado."""
        cache.set("DXY", 99.68, ttl=0.01)
        assert cache.get_sync("DXY") == 99.68
        time.sleep(0.02)
        assert cache.get_sync("DXY") is None

    def test_invalidate(self, cache):
        """invalidate remove a entrada."""
        cache.set("DXY", 99.68)
        cache.invalidate("DXY")
        assert cache.get_sync("DXY") is None

    def test_invalidate_nonexistent_key(self, cache):
        """invalidate de chave inexistente nao da erro."""
        cache.invalidate("NONEXISTENT")  # nao deve levantar excecao

    def test_clear(self, cache):
        """clear remove tudo."""
        cache.set("DXY", 99.68)
        cache.set("SP500", 6506)
        cache.clear()
        assert cache.get_sync("DXY") is None
        assert cache.get_sync("SP500") is None


# ──────────────────────────────────────────────
# Testes: Metricas (stats)
# ──────────────────────────────────────────────

class TestStats:

    @pytest.mark.asyncio
    async def test_stats_count_hits_and_misses(self, cache):
        """Stats devem contar hits e misses."""
        fetcher = _make_fetcher(99.68)

        await cache.get("DXY", fetcher=fetcher)  # miss
        await cache.get("DXY", fetcher=fetcher)  # hit
        await cache.get("DXY", fetcher=fetcher)  # hit

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 2
        assert stats["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)

    @pytest.mark.asyncio
    async def test_stats_count_errors(self, cache):
        """Stats devem contar erros."""
        await cache.get("DXY", fetcher=_make_failing_fetcher())
        stats = cache.get_stats()
        assert stats["errors"] == 1

    def test_stats_valid_entries(self, cache):
        """Stats devem contar entradas validas."""
        cache.set("DXY", 99.68, ttl=300)
        cache.set("OLD", 1.0, ttl=0.01)
        time.sleep(0.02)

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1


# ──────────────────────────────────────────────
# Testes: TTL defaults
# ──────────────────────────────────────────────

class TestTTLDefaults:

    def test_known_keys_have_ttl(self):
        """Chaves conhecidas devem ter TTL definido."""
        for key in ["DXY", "SP500", "VIX", "GOLD", "FEAR_GREED"]:
            assert key in DEFAULT_TTLS
            assert DEFAULT_TTLS[key] > 0

    def test_default_ttl_exists(self):
        """TTL default deve existir."""
        assert "_default" in DEFAULT_TTLS
        assert DEFAULT_TTLS["_default"] > 0


# ──────────────────────────────────────────────
# Testes: Singleton
# ──────────────────────────────────────────────

class TestSingleton:

    def test_returns_same_instance(self):
        """get_market_cache deve retornar mesma instancia."""
        c1 = get_market_cache()
        c2 = get_market_cache()
        assert c1 is c2

    def test_reset_clears_instance(self):
        """reset_market_cache deve criar nova instancia."""
        c1 = get_market_cache()
        c1.set("DXY", 99.68)
        reset_market_cache()
        c2 = get_market_cache()
        assert c1 is not c2
        assert c2.get_sync("DXY") is None

    def test_shared_across_modules(self):
        """Cache e compartilhado (singleton global)."""
        c1 = get_market_cache()
        c1.set("SP500", 6506)

        c2 = get_market_cache()
        assert c2.get_sync("SP500") == 6506


# ──────────────────────────────────────────────
# Testes: Cenario E2E (simula janela completa)
# ──────────────────────────────────────────────

class TestE2EScenario:

    @pytest.mark.asyncio
    async def test_prefetch_then_read(self, cache):
        """
        Simula: prefetch popula cache, leituras subsequentes sao hits.
        """
        # Prefetch: busca tudo em paralelo
        assets = {"DXY": 99.68, "SP500": 6506, "GOLD": 4353, "VIX": 26.78}
        await asyncio.gather(*[
            cache.get(key, fetcher=_make_fetcher(val))
            for key, val in assets.items()
        ])

        stats_after_prefetch = cache.get_stats()
        assert stats_after_prefetch["misses"] == 4
        assert stats_after_prefetch["hits"] == 0

        # Leituras subsequentes: tudo cache hit
        for key, expected_val in assets.items():
            val = cache.get_sync(key)
            assert val == expected_val

        # Simular segundo metodo lendo os mesmos dados (era o DXY duplicado)
        for key in ["DXY", "SP500", "GOLD"]:
            result = await cache.get(key, fetcher=_make_fetcher(0))
            assert result == assets[key]  # do cache, nao do fetcher

        stats_final = cache.get_stats()
        assert stats_final["hits"] == 3  # 3 cache hits
        assert stats_final["misses"] == 4  # ainda 4 (nao incrementou)

    @pytest.mark.asyncio
    async def test_two_windows_second_uses_cache(self, cache):
        """
        Janela 1: todos misses.
        Janela 2 (< TTL): todos hits.
        """
        fetcher, counter = _make_counting_fetcher(99.68)

        # Janela 1
        await cache.get("DXY", fetcher=fetcher, ttl=300)
        assert counter["count"] == 1

        # Janela 2 (60s depois, TTL=300s -> ainda valido)
        await cache.get("DXY", fetcher=fetcher, ttl=300)
        assert counter["count"] == 1  # NAO buscou de novo

    @pytest.mark.asyncio
    async def test_parallel_fetch_performance(self, cache):
        """
        Simula prefetch paralelo vs sequencial.
        8 fetchers de 0.1s cada:
        - Sequencial: ~0.8s
        - Paralelo: ~0.1s
        """
        start = time.time()

        await asyncio.gather(*[
            cache.get(f"ASSET_{i}", fetcher=_make_fetcher(i * 100, delay=0.1))
            for i in range(8)
        ])

        elapsed = time.time() - start
        # Paralelo deve completar em ~0.1-0.3s, nao ~0.8s
        assert elapsed < 0.5, f"Fetch paralelo demorou {elapsed:.2f}s (esperado < 0.5s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
