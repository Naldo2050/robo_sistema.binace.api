# fetchers/macro_cache_validator.py
"""
macro_cache_validator.py - FIX #5

Cache validator para dados macro (Oil, SPX, Gold, etc).
Detecta e invalida cache stale quando fetch falha.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MacroCacheValidator:
    """Valida e gerencia cache de dados macroeconomicos com fallback."""
    
    # Máximo tempo que dados podem ficar stale antes de rejeitar
    MAX_STALE_TIME_SECONDS = {
        "oil": 1800,        # 30 min
        "sp500": 1800,      # 30 min
        "gold": 1800,       # 30 min
        "vix": 300,         # 5 min (volatility muda rápido)
        "dxy": 900,         # 15 min
        "treasury_10y": 900, # 15 min
        "_default": 900     # 15 min default
    }
    
    # Máximo de falhas consecutivas antes de rejeitar cache
    MAX_CONSECUTIVE_FAILURES = 3
    
    def __init__(self):
        self._failure_counters: Dict[str, int] = {}
        self._last_success_time: Dict[str, float] = {}
        self._last_fetch_time: Dict[str, float] = {}
        self._cache_invalid: Dict[str, bool] = {}
        
    def record_success(self, key: str) -> None:
        """Registra fetch bem-sucedido."""
        self._failure_counters[key] = 0
        self._last_success_time[key] = time.time()
        self._last_fetch_time[key] = time.time()
        self._cache_invalid[key] = False
        logger.debug(f"✅ Macro cache {key}: fetch success, failures reset to 0")
    
    def record_failure(self, key: str) -> None:
        """Registra falha de fetch."""
        self._failure_counters[key] = self._failure_counters.get(key, 0) + 1
        self._last_fetch_time[key] = time.time()
        
        if self._failure_counters[key] >= self.MAX_CONSECUTIVE_FAILURES:
            self._cache_invalid[key] = True
            logger.warning(
                f"⚠️ Macro cache {key}: {self._failure_counters[key]} consecutive failures, "
                f"cache marked INVALID"
            )
        else:
            logger.debug(
                f"⚠️ Macro cache {key}: fetch failed "
                f"(failures: {self._failure_counters[key]}/{self.MAX_CONSECUTIVE_FAILURES})"
            )
    
    def is_cache_valid(self, key: str, cached_value: any, cached_time: float) -> Tuple[bool, str]:
        """
        Valida se cache pode ser usado.
        
        Returns:
            (is_valid, reason)
        """
        if cached_value is None:
            return (False, "cached_value is None")
        
        # Se cache foi marcado como inválido por falhas
        if self._cache_invalid.get(key, False):
            return (False, f"cache marked invalid after {self.MAX_CONSECUTIVE_FAILURES} failures")
        
        # Se tempo desde último sucesso > max_stale
        max_stale = self.MAX_STALE_TIME_SECONDS.get(key, self.MAX_STALE_TIME_SECONDS["_default"])
        age = time.time() - cached_time
        
        if age > max_stale:
            return (False, f"cache too old ({age:.0f}s > {max_stale}s)")
        
        return (True, f"valid (age: {age:.0f}s, max: {max_stale}s)")
    
    def should_refetch(self, key: str, cached_time: float) -> Tuple[bool, str]:
        """
        Determina se deve tentar refetch mesmo se cache válido.
        
        Returns:
            (should_refetch, reason)
        """
        # Se cache marcado como inválido, sempre refetch
        if self._cache_invalid.get(key, False):
            return (True, "cache marked invalid, forcing refetch")
        
        # Se estamos acumulando falhas, forçar refetch
        failures = self._failure_counters.get(key, 0)
        if failures > 0:
            return (True, f"recovering from failures ({failures} so far)")
        
        return (False, "no refetch needed")
    
    def get_cache_health(self, key: str) -> Dict[str, any]:
        """Retorna saúde completa do cache para um key."""
        return {
            "key": key,
            "is_invalid": self._cache_invalid.get(key, False),
            "consecutive_failures": self._failure_counters.get(key, 0),
            "max_failures_allowed": self.MAX_CONSECUTIVE_FAILURES,
            "last_success_time": self._last_success_time.get(key),
            "last_fetch_time": self._last_fetch_time.get(key),
            "max_stale_seconds": self.MAX_STALE_TIME_SECONDS.get(key, self.MAX_STALE_TIME_SECONDS["_default"]),
        }
    
    def reset_failures(self, key: str) -> None:
        """Reseta contadores de falha (ex: após refetch bem-sucedido)."""
        self._failure_counters[key] = 0
        self._cache_invalid[key] = False
        logger.info(f"🔄 Macro cache {key}: failures reset")
    
    def invalidate_cache(self, key: str) -> None:
        """Força invalidação de cache para um key."""
        self._cache_invalid[key] = True
        self._failure_counters[key] = self.MAX_CONSECUTIVE_FAILURES
        logger.warning(f"🔴 Macro cache {key}: cache INVALIDATED")
    
    def diagnose(self) -> Dict[str, any]:
        """Diagnóstico completo do estado de cache."""
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cache_health": {
                key: self.get_cache_health(key)
                for key in set(list(self._failure_counters.keys()) + list(self._cache_invalid.keys()))
            }
        }
