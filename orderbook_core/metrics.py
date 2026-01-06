# metrics.py
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any


# Prometheus client (opcional)
try:
    from prometheus_client import Counter, Histogram, Gauge  # type: ignore
    _PROM_AVAILABLE = True
except Exception:
    Counter = Histogram = Gauge = None  # type: ignore
    _PROM_AVAILABLE = False


class _NoOpMetric:
    def labels(self, *args, **kwargs) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1.0) -> None:
        return None

    def observe(self, value: float) -> None:
        return None

    def set(self, value: float) -> None:
        return None


@dataclass
class OrderBookMetrics:
    """
    Métricas exportáveis (Prometheus) com fallback No-Op.

    Observação: se prometheus_client não estiver instalado, todas as operações
    viram no-op (não quebra produção).
    """

    enabled: bool = True
    prom_available: bool = _PROM_AVAILABLE

    # Métricas (Prometheus ou No-Op)
    fetch_total: Any = None
    fetch_latency: Any = None
    cache_hits: Any = None
    validation_failures: Any = None

    imbalance: Any = None
    bid_depth_usd: Any = None
    ask_depth_usd: Any = None
    spread_bps: Any = None
    data_age_seconds: Any = None

    _warned_missing_prom: bool = False

    @classmethod
    def build_default(cls, *, enabled: Optional[bool] = None) -> "OrderBookMetrics":
        """
        Constrói a instância padrão.

        enabled:
          - se None, lê env ORDERBOOK_METRICS_ENABLED (default "1")
          - se prometheus_client não existe, volta No-Op
        """
        if enabled is None:
            enabled = os.getenv("ORDERBOOK_METRICS_ENABLED", "1") == "1"

        m = cls(enabled=bool(enabled), prom_available=_PROM_AVAILABLE)

        if not m.enabled:
            return m._make_noop()

        if not m.prom_available:
            return m._make_noop(warn=True)

        # Prometheus metrics reais
        # Nota: nomes e labels estáveis (compatível com dashboards)
        m.fetch_total = Counter(
            "orderbook_fetch_total",
            "Total de operações de fetch/orderbook (inclui cache e stale)",
            ["symbol", "status", "source"],
        )

        m.fetch_latency = Histogram(
            "orderbook_fetch_latency_seconds",
            "Latência do fetch live em segundos",
            ["symbol"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        m.cache_hits = Counter(
            "orderbook_cache_hits_total",
            "Total de cache hits",
            ["symbol"],
        )

        m.validation_failures = Counter(
            "orderbook_validation_failures_total",
            "Falhas de validação do snapshot",
            ["symbol", "reason"],
        )

        m.imbalance = Gauge(
            "orderbook_imbalance",
            "Imbalance atual do orderbook",
            ["symbol"],
        )

        m.bid_depth_usd = Gauge(
            "orderbook_bid_depth_usd",
            "Bid depth USD (top_n)",
            ["symbol"],
        )

        m.ask_depth_usd = Gauge(
            "orderbook_ask_depth_usd",
            "Ask depth USD (top_n)",
            ["symbol"],
        )

        m.spread_bps = Gauge(
            "orderbook_spread_bps",
            "Spread em basis points",
            ["symbol"],
        )

        m.data_age_seconds = Gauge(
            "orderbook_data_age_seconds",
            "Idade do dado usado no evento (segundos)",
            ["symbol", "source"],
        )

        return m

    def _make_noop(self, warn: bool = False) -> "OrderBookMetrics":
        self.fetch_total = _NoOpMetric()
        self.fetch_latency = _NoOpMetric()
        self.cache_hits = _NoOpMetric()
        self.validation_failures = _NoOpMetric()
        self.imbalance = _NoOpMetric()
        self.bid_depth_usd = _NoOpMetric()
        self.ask_depth_usd = _NoOpMetric()
        self.spread_bps = _NoOpMetric()
        self.data_age_seconds = _NoOpMetric()

        if warn and not self._warned_missing_prom:
            self._warned_missing_prom = True
            logging.warning(
                "Prometheus metrics desabilitadas: prometheus_client não instalado. "
                "Instale 'prometheus-client' para habilitar."
            )
        return self

    # Helpers de registro (centraliza labels e evita repetição)
    def inc_fetch(self, *, symbol: str, status: str, source: str) -> None:
        self.fetch_total.labels(symbol=symbol, status=status, source=source).inc()

    def observe_latency(self, *, symbol: str, seconds: float) -> None:
        self.fetch_latency.labels(symbol=symbol).observe(seconds)

    def inc_cache_hit(self, *, symbol: str) -> None:
        self.cache_hits.labels(symbol=symbol).inc()

    def inc_validation_failure(self, *, symbol: str, reason: str) -> None:
        # limita cardinalidade: usa razão curta
        reason = (reason or "unknown")[:80]
        self.validation_failures.labels(symbol=symbol, reason=reason).inc()

    def set_core_gauges(
        self,
        *,
        symbol: str,
        imbalance: Optional[float],
        bid_depth_usd: Optional[float],
        ask_depth_usd: Optional[float],
        spread_bps: Optional[float],
    ) -> None:
        if imbalance is not None:
            self.imbalance.labels(symbol=symbol).set(float(imbalance))
        if bid_depth_usd is not None:
            self.bid_depth_usd.labels(symbol=symbol).set(float(bid_depth_usd))
        if ask_depth_usd is not None:
            self.ask_depth_usd.labels(symbol=symbol).set(float(ask_depth_usd))
        if spread_bps is not None:
            self.spread_bps.labels(symbol=symbol).set(float(spread_bps))

    def set_data_age(self, *, symbol: str, source: str, age_seconds: float) -> None:
        self.data_age_seconds.labels(symbol=symbol, source=source).set(float(age_seconds))


class MetricsTracker:
    """
    Context manager para medir latência (usa perf_counter).
    Referência: time.perf_counter() é recomendado para medir duração.
    Docs: https://docs.python.org/3/library/time.html#time.perf_counter
    """

    def __init__(self, metrics: OrderBookMetrics, *, symbol: str):
        self.metrics = metrics
        self.symbol = symbol
        self._start: Optional[float] = None

    def __enter__(self) -> "MetricsTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            # latency só faz sentido para live fetch
            self.metrics.observe_latency(symbol=self.symbol, seconds=float(elapsed))