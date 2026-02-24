"""
Spread Percentile Tracker — Monitora spread do order book ao longo do tempo.

Mantém histórico rolling de spreads e calcula percentil, média, desvio padrão.
Classificação de liquidez baseada no percentil do spread atual.

Uso:
    tracker = SpreadTracker(window_minutes=1440)  # 24h
    tracker.update(0.1)     # Spread atual em USD
    metrics = tracker.get_metrics(0.1)
    print(metrics["spread_percentile"])  # ex: 15.3
    print(metrics["liquidity_signal"])   # ex: "EXCELLENT"
"""

import bisect
import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class SpreadTracker:
    """
    Mantém histórico de spreads e calcula métricas de percentil.
    
    O spread é o indicador #1 de liquidez intraday.
    Spread tight (percentil baixo) = muita liquidez.
    Spread wide (percentil alto) = pouca liquidez ou evento.
    
    Attributes:
        window_minutes: Tamanho da janela rolling em minutos (default 1440 = 24h)
    """

    def __init__(self, window_minutes: int = 1440):
        self._window_minutes = window_minutes
        # Armazena tuplas (timestamp_ms, spread_value)
        self._history: deque = deque()
        self._sorted_cache: list = []
        self._cache_dirty: bool = True
        self._last_cleanup_ms: int = 0
        self._cleanup_interval_ms: int = 60_000  # Limpar expirados a cada 1 min
        self._stats_cache: dict = {}
        self._stats_cache_ms: int = 0

    def update(self, spread: float, timestamp_ms: Optional[int] = None) -> None:
        """
        Registra um novo valor de spread.
        
        Args:
            spread: Spread atual (em USD ou bps, manter unidade consistente)
            timestamp_ms: Timestamp em ms (usa time.time() * 1000 se não fornecido)
        """
        if spread < 0:
            return

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        self._history.append((timestamp_ms, spread))
        self._cache_dirty = True

        # Cleanup periódico
        if timestamp_ms - self._last_cleanup_ms > self._cleanup_interval_ms:
            self._cleanup_expired(timestamp_ms)

    def _cleanup_expired(self, now_ms: Optional[int] = None) -> None:
        """Remove entradas fora da janela."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        cutoff_ms = now_ms - (self._window_minutes * 60 * 1000)

        while self._history and self._history[0][0] < cutoff_ms:
            self._history.popleft()

        self._last_cleanup_ms = now_ms
        self._cache_dirty = True

    def _rebuild_sorted_cache(self) -> None:
        """Reconstrói cache de spreads ordenados para cálculo de percentil."""
        self._sorted_cache = sorted(s for _, s in self._history)
        self._cache_dirty = False

    def get_metrics(self, current_spread: Optional[float] = None) -> dict:
        """
        Calcula métricas de spread com percentil.
        
        Args:
            current_spread: Spread atual. Se None, usa o último registrado.
            
        Returns:
            Dict com percentil, média, std, min, max e classificação.
        """
        if len(self._history) < 5:
            return {
                "status": "insufficient_data",
                "samples": len(self._history),
                "min_required": 5,
                "current_spread": current_spread or 0,
                "spread_percentile": 50.0,
                "liquidity_signal": "UNKNOWN",
            }

        # Cleanup antes de calcular
        self._cleanup_expired()

        if current_spread is None:
            if not self._history:
                return {
                    "status": "no_data",
                    "current_spread": 0.0,
                    "spread_percentile": 50.0,
                    "liquidity_signal": "UNKNOWN",
                }
            last_entry = self._history[-1]
            current_spread = last_entry[1]

        # Ensure current_spread is a valid float
        if current_spread is None:
            return {
                "status": "no_data",
                "current_spread": 0.0,
                "spread_percentile": 50.0,
                "liquidity_signal": "UNKNOWN",
            }

        # Rebuild cache se necessário
        if self._cache_dirty:
            self._rebuild_sorted_cache()

        n = len(self._sorted_cache)
        if n == 0:
            return {
                "status": "no_data",
                "current_spread": current_spread,
                "spread_percentile": 50.0,
                "liquidity_signal": "UNKNOWN",
            }

        # Percentil
        position = bisect.bisect_left(self._sorted_cache, current_spread)
        percentile = round((position / n) * 100, 1)

        # Estatísticas básicas
        spread_sum = sum(self._sorted_cache)
        spread_mean = spread_sum / n

        if n > 1:
            variance = sum((s - spread_mean) ** 2 for s in self._sorted_cache) / (n - 1)
            spread_std = variance ** 0.5
        else:
            spread_std = 0.0

        spread_min = self._sorted_cache[0]
        spread_max = self._sorted_cache[-1]

        # Mediana
        if n % 2 == 0:
            spread_median = (self._sorted_cache[n // 2 - 1] + self._sorted_cache[n // 2]) / 2
        else:
            spread_median = self._sorted_cache[n // 2]

        # Classificação de liquidez
        if percentile < 10:
            liquidity_signal = "EXCELLENT"
        elif percentile < 25:
            liquidity_signal = "GOOD"
        elif percentile < 50:
            liquidity_signal = "NORMAL"
        elif percentile < 75:
            liquidity_signal = "BELOW_NORMAL"
        elif percentile < 90:
            liquidity_signal = "THIN"
        else:
            liquidity_signal = "DANGER"

        # Z-score do spread actual
        spread_std_value = spread_std if spread_std > 0 else 0.0
        z_score = (current_spread - spread_mean) / spread_std_value if spread_std > 0 else 0.0

        return {
            "status": "ok",
            "current_spread": round(float(current_spread), 6),
            "spread_percentile": percentile,
            "spread_mean": round(spread_mean, 6),
            "spread_median": round(spread_median, 6),
            "spread_std": round(spread_std, 6),
            "spread_min": round(spread_min, 6),
            "spread_max": round(spread_max, 6),
            "spread_z_score": round(z_score, 4),
            "is_tight": percentile < 20,
            "is_wide": percentile > 80,
            "is_anomalous": abs(z_score) > 3,
            "liquidity_signal": liquidity_signal,
            "samples": n,
            "window_minutes": self._window_minutes,
        }

    def get_summary(self) -> dict:
        """Retorna resumo rápido sem calcular percentil."""
        n = len(self._history)
        if n == 0:
            return {"status": "empty", "samples": 0}

        spreads = [s for _, s in self._history]
        return {
            "status": "ok",
            "samples": n,
            "latest_spread": round(spreads[-1], 6),
            "mean_spread": round(sum(spreads) / n, 6),
            "min_spread": round(min(spreads), 6),
            "max_spread": round(max(spreads), 6),
            "window_minutes": self._window_minutes,
        }

    def reset(self) -> None:
        """Limpa todo o histórico."""
        self._history.clear()
        self._sorted_cache.clear()
        self._cache_dirty = True
        self._stats_cache.clear()
