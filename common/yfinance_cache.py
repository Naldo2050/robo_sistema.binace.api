"""
Cache centralizado para dados yFinance/FRED.

Qualquer módulo que precise de DXY, SP500, GOLD etc. usa este cache.
Evita fetches duplicados na mesma janela.

Uso:
    from common.yfinance_cache import MarketDataCache, get_market_cache
    cache = get_market_cache()
    dxy = await cache.get("DXY")  # busca ou retorna cache
"""

import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Uma entrada no cache com TTL."""
    value: Any
    fetched_at: float
    ttl: float
    source: str = ""

    @property
    def is_valid(self) -> bool:
        return time.time() - self.fetched_at < self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


# TTLs por tipo de dado (em segundos)
DEFAULT_TTLS = {
    # Dados que mudam pouco intraday
    "DXY": 300,         # 5 min
    "TNX": 300,         # 5 min
    "SP500": 300,       # 5 min
    "NASDAQ": 300,      # 5 min
    "GOLD": 300,        # 5 min
    "WTI": 300,         # 5 min
    "VIX": 300,         # 5 min
    "FEAR_GREED": 600,  # 10 min
    "BTC_DOMINANCE": 300,
    "ETH_DOMINANCE": 300,
    "SPX_TWELVE": 300,

    # Dados históricos (correlações) — cache mais longo
    "DXY_HISTORY": 900,      # 15 min
    "SP500_HISTORY": 900,
    "GOLD_HISTORY": 900,
    "BTC_HISTORY": 900,
    "NDX_HISTORY": 900,

    # Volume profile
    "VOLUME_PROFILE": 600,   # 10 min

    # Genérico
    "_default": 300,
}


@dataclass
class MarketDataCache:
    """
    Cache thread-safe para dados de mercado.

    Features:
    - TTL configurável por chave
    - Lock por chave (evita fetches duplicados concorrentes)
    - Métricas de hit/miss
    - Stale-while-revalidate (retorna velho se fetch falhar)
    """

    _cache: dict[str, CacheEntry] = field(default_factory=dict, repr=False)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)
    _errors: int = field(default=0, repr=False)

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Retorna lock por chave (cria se não existir)."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _get_ttl(self, key: str) -> float:
        """Retorna TTL para a chave."""
        return DEFAULT_TTLS.get(key, DEFAULT_TTLS["_default"])

    def get_sync(self, key: str) -> Optional[Any]:
        """
        Leitura síncrona do cache (sem fetch).
        Retorna None se não existe ou expirado.
        """
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            return entry.value
        return None

    async def get(
        self,
        key: str,
        fetcher: Optional[Callable[[], Coroutine]] = None,
        ttl: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Retorna valor do cache ou busca via fetcher.

        Args:
            key: Chave do cache (ex: "DXY", "SP500")
            fetcher: Coroutine que busca o dado (só chamada se cache miss)
            ttl: TTL customizado (se None, usa DEFAULT_TTLS)

        Returns:
            Valor ou None se falhou
        """
        # 1) Cache hit?
        entry = self._cache.get(key)
        if entry and entry.is_valid:
            self._hits += 1
            return entry.value

        # 2) Sem fetcher → retorna stale ou None
        if fetcher is None:
            if entry:
                logger.debug(
                    "%s: cache stale (%.0fs), sem fetcher",
                    key, entry.age_seconds,
                )
                return entry.value
            return None

        # 3) Fetch com lock (evita duplicatas concorrentes)
        lock = self._get_lock(key)
        async with lock:
            # Re-check após adquirir lock (outro task pode ter fetched)
            entry = self._cache.get(key)
            if entry and entry.is_valid:
                self._hits += 1
                return entry.value

            # 4) Fetch real
            self._misses += 1
            actual_ttl = ttl if ttl is not None else self._get_ttl(key)

            # Timeout mais longo para primeiro fetch (yfinance cold start)
            _timeout = 15.0
            try:
                value = await asyncio.wait_for(fetcher(), timeout=_timeout)
                if value is not None:
                    self._cache[key] = CacheEntry(
                        value=value,
                        fetched_at=time.time(),
                        ttl=actual_ttl,
                        source="fetcher",
                    )
                    logger.debug("%s: fetched, TTL=%.0fs", key, actual_ttl)
                    return value
                else:
                    self._errors += 1
                    # Retorna stale se disponível
                    if entry:
                        logger.warning(
                            "%s: fetch retornou None, usando stale (%.0fs)",
                            key, entry.age_seconds,
                        )
                        return entry.value
                    return None

            except asyncio.TimeoutError:
                self._errors += 1
                logger.warning("%s: fetch timeout (%.0fs)", key, _timeout)
                return entry.value if entry else None

            except Exception as e:
                self._errors += 1
                logger.warning("%s: fetch error: %s", key, e)
                return entry.value if entry else None

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Inserção direta no cache (para quem já tem o dado)."""
        actual_ttl = ttl if ttl is not None else self._get_ttl(key)
        self._cache[key] = CacheEntry(
            value=value,
            fetched_at=time.time(),
            ttl=actual_ttl,
            source="manual",
        )

    def invalidate(self, key: str):
        """Invalida uma entrada."""
        self._cache.pop(key, None)

    def clear(self):
        """Limpa todo o cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._errors = 0

    def get_stats(self) -> dict:
        """Métricas do cache."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        valid_entries = sum(1 for e in self._cache.values() if e.is_valid)
        return {
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate_pct": round(hit_rate, 1),
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
        }


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_cache_instance: Optional[MarketDataCache] = None


def get_market_cache() -> MarketDataCache:
    """Retorna instância singleton do cache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MarketDataCache()
        logger.info("MarketDataCache inicializado")
    return _cache_instance


def reset_market_cache():
    """Reset singleton (para testes)."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.clear()
    _cache_instance = None
