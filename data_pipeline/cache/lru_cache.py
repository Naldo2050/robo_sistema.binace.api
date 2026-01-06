# data_pipeline/cache/lru_cache.py
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    """
    Entrada de cache com metadados completos.

    Mantém informações sobre:
    - Valor armazenado
    - Timestamp de criação
    - Flag de expiração
    - Contador de acessos

    Permite cache com TTL e análise de uso.
    """

    value: Any
    timestamp: float
    expired: bool = False
    hit_count: int = 0

    def age(self) -> float:
        """Retorna idade em segundos."""
        return time.time() - self.timestamp

    def mark_expired(self) -> None:
        """Marca entrada como expirada."""
        self.expired = True

    def increment_hits(self) -> None:
        """Incrementa contador de acessos."""
        self.hit_count += 1

    def is_fresh(self, ttl_seconds: int) -> bool:
        """Verifica se entrada ainda está fresca."""
        return not self.expired and self.age() <= ttl_seconds


class LRUCache:
    """
    Cache LRU (Least Recently Used) com TTL e flag de expiração.

    Características:
    - Evição automática quando atinge limite
    - TTL configurável por entrada
    - Flag de expiração (retorna valor expirado ao invés de deletar)
    - Estatísticas detalhadas
    - Refresh manual de entradas

    A flag de expiração permite:
    - Evitar recomputação imediata de valores caros
    - Background refresh de dados
    - Melhor performance em picos de carga

    Exemplo de uso:
        cache = LRUCache(max_items=1000, ttl_seconds=3600)

        # Set
        cache.set("key1", {"data": "value"})

        # Get com flag de expiração
        value = cache.get("key1", allow_expired=True)
        if cache.is_expired("key1"):
            # Valor expirado, agendar refresh em background
            schedule_refresh("key1")

        # Refresh manual
        cache.refresh("key1")
    """

    def __init__(self, max_items: int = 1000, ttl_seconds: int = 3600) -> None:
        """
        Inicializa cache LRU.

        Args:
            max_items: Quantidade máxima de itens no cache
            ttl_seconds: Tempo de vida padrão em segundos
        """
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats: Dict[str, int] = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'expired_hits': 0,
            'refreshes': 0
        }

    def get(
        self,
        key: str,
        allow_expired: bool = True
    ) -> Optional[Any]:
        """
        Obtém valor do cache.

        Args:
            key: Chave do cache
            allow_expired: Se True, retorna valor expirado com flag

        Returns:
            Valor armazenado ou None se não existir
        """
        if key not in self._cache:
            self._stats['misses'] += 1
            return None

        entry = self._cache[key]
        age = entry.age()

        # Verificar expiração
        if age > self.ttl_seconds:
            if allow_expired:
                # ⚡ OTIMIZAÇÃO: Retorna valor expirado com flag
                # Permite usar valor antigo enquanto atualiza
                entry.mark_expired()
                self._stats['expired_hits'] += 1
                entry.increment_hits()
                self._cache.move_to_end(key)
                return entry.value
            else:
                # Remove se não permitir expirados
                del self._cache[key]
                self._stats['misses'] += 1
                return None

        # Move para o final (mais recente)
        self._cache.move_to_end(key)
        entry.increment_hits()
        self._stats['hits'] += 1

        return entry.value

    def is_expired(self, key: str) -> bool:
        """
        Verifica se entrada está expirada.

        Args:
            key: Chave do cache

        Returns:
            True se expirada ou não existir
        """
        if key not in self._cache:
            return True

        entry = self._cache[key]
        return entry.expired or entry.age() > self.ttl_seconds

    def set(self, key: str, value: Any, force_fresh: bool = False) -> None:
        """
        Armazena valor no cache.

        Args:
            key: Chave
            value: Valor a armazenar
            force_fresh: Se True, marca como não-expirado mesmo se já existir
        """
        # Remove mais antigo se exceder limite
        if len(self._cache) >= self.max_items:
            removed_key, _ = self._cache.popitem(last=False)
            self._stats['evictions'] += 1

        # Criar ou atualizar entrada
        if key in self._cache and not force_fresh:
            # Atualizar existente
            entry = self._cache[key]
            entry.value = value
            entry.timestamp = time.time()
            entry.expired = False
        else:
            # Nova entrada
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                expired=False
            )

        self._cache[key] = entry
        self._stats['sets'] += 1

    def refresh(self, key: str) -> bool:
        """
        Marca entrada como fresh (não-expirada) sem alterar valor.

        Útil para:
        - Validar que dados externos não mudaram
        - Estender TTL de dados ainda válidos

        Args:
            key: Chave a refreshar

        Returns:
            True se refreshed, False se não existe
        """
        if key not in self._cache:
            return False

        entry = self._cache[key]
        entry.timestamp = time.time()
        entry.expired = False
        self._stats['refreshes'] += 1
        return True

    def clear(self) -> None:
        """Limpa todo o cache."""
        self._cache.clear()

    def remove(self, key: str) -> bool:
        """
        Remove entrada específica do cache.

        Args:
            key: Chave a remover

        Returns:
            True se removida, False se não existia
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retorna informações detalhadas sobre uma entrada.

        Args:
            key: Chave da entrada

        Returns:
            Dicionário com metadados ou None se não existir
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        return {
            'age_seconds': round(entry.age(), 2),
            'expired': entry.expired or entry.age() > self.ttl_seconds,
            'hit_count': entry.hit_count,
            'timestamp': entry.timestamp,
            'created_at_iso': datetime.fromtimestamp(entry.timestamp).isoformat(),
            'is_fresh': entry.is_fresh(self.ttl_seconds)
        }

    def stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas completas do cache.

        Returns:
            Dicionário com todas as métricas
        """
        total = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0

        expired_rate = (
            self._stats['expired_hits'] /
            max(self._stats['hits'] + self._stats['expired_hits'], 1) * 100
        )

        return {
            **self._stats,
            'size': len(self._cache),
            'hit_rate_pct': round(hit_rate, 2),
            'expired_hit_rate_pct': round(expired_rate, 2),
            'memory_items': len(self._cache),
            'utilization_pct': round(len(self._cache) / self.max_items * 100, 2)
        }

    def get_top_accessed(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Retorna as entradas mais acessadas.

        Args:
            limit: Quantidade de entradas a retornar

        Returns:
            Lista de tuplas (key, hit_count) ordenadas por hits
        """
        entries = [
            (key, entry.hit_count)
            for key, entry in self._cache.items()
        ]
        return sorted(entries, key=lambda x: x[1], reverse=True)[:limit]