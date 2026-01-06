# data_pipeline/fallback/registry.py
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional


class FallbackRegistry:
    """
    Registra e rastreia fallbacks do sistema.

    Quando uma operação falha e usa fallback, registra:
    - Componente que falhou
    - Razão da falha
    - Timestamp
    - Exception details

    Permite análise de:
    - Quais componentes falham mais
    - Padrões de falha
    - Impacto de fallbacks

    Exemplo de uso:
        registry = FallbackRegistry()

        try:
            result = expensive_operation()
        except Exception as e:
            fallback_info = registry.register(
                'expensive_operation',
                'timeout',
                e
            )
            result = cheap_fallback()
            result.update(fallback_info)  # Marca que usou fallback

        # Análise
        stats = registry.get_stats()
        print(f"Total de fallbacks: {stats['total_fallbacks']}")
        print(f"Top causas: {stats['by_cause']}")
    """

    def __init__(self, max_entries: int = 100) -> None:
        """
        Inicializa registry de fallbacks.

        Args:
            max_entries: Quantidade máxima de entradas a manter
        """
        self._registry: deque = deque(maxlen=max_entries)
        self._stats: Dict[str, int] = {}

    def register(
        self,
        component: str,
        reason: str,
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """
        Registra um fallback.

        Args:
            component: Nome do componente que falhou
            reason: Razão da falha
            exception: Exception que causou a falha (opcional)

        Returns:
            Dicionário com metadados do fallback para incluir no output
        """
        # Truncar mensagem de erro para evitar logs gigantes
        error_msg = str(exception)[:80] if exception else reason[:80]

        entry: Dict[str, Any] = {
            'timestamp': time.time(),
            'timestamp_iso': datetime.now().isoformat(),
            'component': component,
            'reason': reason,
            'error': error_msg,
            'exception_type': type(exception).__name__ if exception else None
        }

        self._registry.append(entry)

        # Atualizar estatísticas
        key = f"{component}:{reason}"
        self._stats[key] = self._stats.get(key, 0) + 1

        # Retornar info para incluir no resultado
        return {
            'fallback_triggered': True,
            'fallback_component': component,
            'fallback_reason': reason,
            'fallback_error': error_msg
        }

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna fallbacks recentes.

        Args:
            limit: Quantidade de fallbacks a retornar

        Returns:
            Lista com dicionários de fallbacks
        """
        return list(self._registry)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de fallbacks.

        Returns:
            Dicionário com métricas agregadas
        """
        total = sum(self._stats.values())

        # Top 10 causas
        top_causes = dict(sorted(
            self._stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        return {
            'total_fallbacks': total,
            'unique_causes': len(self._stats),
            'by_cause': self._stats,
            'top_causes': top_causes
        }

    def clear(self) -> None:
        """Limpa o registry."""
        self._registry.clear()
        self._stats.clear()