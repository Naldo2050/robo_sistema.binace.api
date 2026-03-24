# institutional/hurst_exponent.py
"""
Hurst Exponent (Expoente de Hurst)

Mede se o mercado está em:
- Tendência (H > 0.5) — momentum
- Aleatório (H ≈ 0.5) — random walk
- Mean-reverting (H < 0.5) — reversão à média

Método #23 do Arsenal Institucional.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    MarketRegime,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class HurstResult:
    """Resultado do cálculo de Hurst."""
    hurst: float
    regime: str  # "trending", "random", "mean_reverting"
    confidence: float
    sample_size: int
    r_squared: float  # qualidade do fit


class HurstCalculator:
    """
    Calculador do Expoente de Hurst via R/S Analysis.

    Método Rescaled Range:
    1. Dividir série em sub-séries
    2. Para cada sub-série: calcular Range/StdDev
    3. H = slope do log-log plot de R/S vs n

    H > 0.5 → Tendência (série persistente)
    H = 0.5 → Random walk
    H < 0.5 → Mean reverting (série anti-persistente)
    """

    def __init__(
        self,
        min_samples: int = 50,
        max_samples: int = 2000,
        min_lag: int = 2,
        max_lag: int = 100,
    ):
        if min_samples < 20:
            raise InvalidParameterError("min_samples must be >= 20")

        self.min_samples = min_samples
        self.max_samples = max_samples
        self.min_lag = min_lag
        self.max_lag = max_lag

        self._prices: deque[float] = deque(maxlen=max_samples)
        self._last_result: Optional[HurstResult] = None

    @property
    def sample_count(self) -> int:
        return len(self._prices)

    @property
    def last_result(self) -> Optional[HurstResult]:
        return self._last_result

    def add_price(self, price: float) -> None:
        """Adiciona preço à série."""
        self._prices.append(price)

    def add_prices(self, prices: list[float]) -> None:
        """Adiciona múltiplos preços."""
        for p in prices:
            self._prices.append(p)

    def calculate(self) -> Optional[HurstResult]:
        """
        Calcula expoente de Hurst via R/S analysis.

        Retorna None se dados insuficientes.
        """
        if len(self._prices) < self.min_samples:
            return None

        prices = list(self._prices)

        # Calcular retornos logarítmicos
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i - 1]))

        if len(returns) < self.min_samples:
            return None

        # R/S Analysis
        max_lag = min(self.max_lag, len(returns) // 4)
        if max_lag < self.min_lag:
            return None

        lags = []
        rs_values = []

        for lag in range(self.min_lag, max_lag + 1):
            rs = self._rescaled_range(returns, lag)
            if rs is not None and rs > 0:
                lags.append(lag)
                rs_values.append(rs)

        if len(lags) < 3:
            return None

        # Regressão log-log
        log_lags = [math.log(l) for l in lags]
        log_rs = [math.log(r) for r in rs_values]

        hurst, r_squared = self._linear_regression(log_lags, log_rs)

        # Classificar regime
        if hurst > 0.55:
            regime = "trending"
        elif hurst < 0.45:
            regime = "mean_reverting"
        else:
            regime = "random"

        # Confiança baseada em R² e tamanho da amostra
        confidence = r_squared * min(len(returns) / 200, 1.0)

        result = HurstResult(
            hurst=hurst,
            regime=regime,
            confidence=confidence,
            sample_size=len(returns),
            r_squared=r_squared,
        )

        self._last_result = result
        return result

    def _rescaled_range(self, returns: list[float], lag: int) -> Optional[float]:
        """Calcula R/S para um dado lag."""
        n = len(returns)
        if lag > n:
            return None

        rs_values = []
        num_segments = n // lag

        for i in range(num_segments):
            segment = returns[i * lag: (i + 1) * lag]
            if len(segment) < lag:
                continue

            mean = sum(segment) / len(segment)

            # Desvios acumulados
            cumdev = []
            running = 0.0
            for val in segment:
                running += val - mean
                cumdev.append(running)

            if not cumdev:
                continue

            R = max(cumdev) - min(cumdev)

            # Desvio padrão
            variance = sum((x - mean) ** 2 for x in segment) / len(segment)
            S = variance ** 0.5

            if S > 0:
                rs_values.append(R / S)

        if rs_values:
            return sum(rs_values) / len(rs_values)
        return None

    def _linear_regression(
        self,
        x: list[float],
        y: list[float],
    ) -> tuple[float, float]:
        """
        Regressão linear simples.
        Retorna (slope, r_squared).
        """
        n = len(x)
        if n < 2:
            return 0.5, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-12:
            return 0.5, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # R²
        ss_tot = sum_y2 - (sum_y ** 2) / n
        y_mean = sum_y / n
        y_pred = [slope * xi + (y_mean - slope * (sum_x / n)) for xi in x]
        ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        return slope, r_squared

    def analyze(self) -> AnalysisResult:
        """Análise completa com recomendações de estratégia."""
        result = AnalysisResult(
            source="hurst_calculator",
            timestamp=time.time(),
        )

        hurst_result = self.calculate()
        if hurst_result is None:
            result.confidence = 0.0
            result.metrics = {"sample_count": len(self._prices)}
            return result

        result.metrics = {
            "hurst": hurst_result.hurst,
            "regime": hurst_result.regime,
            "r_squared": hurst_result.r_squared,
            "sample_size": hurst_result.sample_size,
        }

        # Mapear regime
        if hurst_result.regime == "trending":
            result.regime = MarketRegime.TRENDING_UP  # Direção desconhecida aqui
            strategy = "Use momentum/trend following strategies"
        elif hurst_result.regime == "mean_reverting":
            result.regime = MarketRegime.RANGING
            strategy = "Use mean reversion strategies (buy dips, sell rallies)"
        else:
            result.regime = MarketRegime.UNKNOWN
            strategy = "Market is random — reduce position sizes"

        result.signals.append(
            Signal(
                timestamp=time.time(),
                signal_type="hurst_regime",
                direction=Side.UNKNOWN,
                strength=(
                    SignalStrength.STRONG if hurst_result.confidence > 0.7
                    else SignalStrength.MODERATE if hurst_result.confidence > 0.4
                    else SignalStrength.WEAK
                ),
                price=self._prices[-1] if self._prices else 0.0,
                confidence=hurst_result.confidence,
                source="hurst_calculator",
                description=(
                    f"Hurst={hurst_result.hurst:.3f} "
                    f"({hurst_result.regime}). {strategy}"
                ),
                metadata={
                    "hurst": hurst_result.hurst,
                    "r_squared": hurst_result.r_squared,
                    "recommended_strategy": strategy,
                },
            )
        )

        result.confidence = hurst_result.confidence

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self._prices.clear()
        self._last_result = None
