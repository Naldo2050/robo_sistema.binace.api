"""
Entropia de Shannon aplicada ao mercado.

Mede o nível de desordem/incerteza no fluxo de preço.
- Baixa entropia = preço previsível = oportunidade
- Alta entropia = ruído = ficar fora

Método #25 do Arsenal Institucional.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
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
class EntropyState:
    """Estado da entropia em um ponto."""
    timestamp: float
    entropy: float
    max_entropy: float
    normalized_entropy: float  # 0-1
    regime: str  # "ordered", "normal", "chaotic"
    n_bins: int


class EntropyAnalyzer:
    """
    Analisador de Entropia de Shannon para séries de preço.

    H(X) = -Σ p(x) · log₂(p(x))

    Discretiza retornos em bins e calcula entropia da distribuição.

    Entropia alta (próximo de log₂(n_bins)):
    - Retornos uniformemente distribuídos = imprevisível
    - Mercado em ruído = difícil operar

    Entropia baixa:
    - Retornos concentrados em poucos bins = previsível
    - Mercado em tendência ou com padrão = oportunidade
    """

    def __init__(
        self,
        n_bins: int = 20,
        window_size: int = 100,
        max_history: int = 2000,
    ):
        if n_bins < 3:
            raise InvalidParameterError("n_bins must be >= 3")
        if window_size < 10:
            raise InvalidParameterError("window_size must be >= 10")

        self.n_bins = n_bins
        self.window_size = window_size

        self._prices: deque[float] = deque(maxlen=max_history)
        self._returns: deque[float] = deque(maxlen=max_history)
        self._entropy_history: deque[EntropyState] = deque(maxlen=max_history)

        # Máximo teórico = log₂(n_bins)
        self.max_entropy = math.log2(n_bins)

    @property
    def current_entropy(self) -> Optional[float]:
        if self._entropy_history:
            return self._entropy_history[-1].entropy
        return None

    @property
    def normalized_entropy(self) -> Optional[float]:
        if self._entropy_history:
            return self._entropy_history[-1].normalized_entropy
        return None

    def _discretize_returns(self, returns: list[float]) -> list[int]:
        """Discretiza retornos em bins."""
        if not returns:
            return []

        min_ret = min(returns)
        max_ret = max(returns)
        range_ret = max_ret - min_ret

        if range_ret == 0:
            return [self.n_bins // 2] * len(returns)

        bins = []
        for r in returns:
            bin_idx = int((r - min_ret) / range_ret * (self.n_bins - 1))
            bin_idx = max(0, min(bin_idx, self.n_bins - 1))
            bins.append(bin_idx)

        return bins

    def _calculate_entropy(self, bins: list[int]) -> float:
        """Calcula entropia de Shannon."""
        if not bins:
            return 0.0

        n = len(bins)
        counter = Counter(bins)
        entropy = 0.0

        for count in counter.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def add_price(self, timestamp: float, price: float) -> Optional[EntropyState]:
        """
        Adiciona preço e calcula entropia se janela completa.
        """
        self._prices.append(price)

        if len(self._prices) < 2:
            return None

        prev = self._prices[-2]
        if prev > 0 and price > 0:
            ret = math.log(price / prev)
            self._returns.append(ret)

        if len(self._returns) < self.window_size:
            return None

        # Calcular entropia na janela
        window = list(self._returns)[-self.window_size:]
        bins = self._discretize_returns(window)
        entropy = self._calculate_entropy(bins)
        normalized = entropy / self.max_entropy if self.max_entropy > 0 else 0

        if normalized > 0.8:
            regime = "chaotic"
        elif normalized < 0.4:
            regime = "ordered"
        else:
            regime = "normal"

        state = EntropyState(
            timestamp=timestamp,
            entropy=entropy,
            max_entropy=self.max_entropy,
            normalized_entropy=normalized,
            regime=regime,
            n_bins=self.n_bins,
        )

        self._entropy_history.append(state)
        return state

    def get_entropy_trend(self, lookback: int = 20) -> dict:
        """Calcula tendência da entropia."""
        if len(self._entropy_history) < lookback:
            return {"trend": "unknown", "change": 0.0}

        recent = list(self._entropy_history)[-lookback:]
        start_e = recent[0].normalized_entropy
        end_e = recent[-1].normalized_entropy

        change = end_e - start_e

        if change > 0.1:
            trend = "increasing"  # Mercado ficando mais caótico
        elif change < -0.1:
            trend = "decreasing"  # Mercado ficando mais ordenado
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": change,
            "start": start_e,
            "end": end_e,
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa de entropia."""
        result = AnalysisResult(
            source="entropy_analyzer",
            timestamp=time.time(),
        )

        if not self._entropy_history:
            result.confidence = 0.0
            return result

        latest = self._entropy_history[-1]
        trend = self.get_entropy_trend()

        result.metrics = {
            "entropy": latest.entropy,
            "normalized_entropy": latest.normalized_entropy,
            "max_entropy": latest.max_entropy,
            "regime": latest.regime,
            "entropy_trend": trend["trend"],
            "entropy_change": trend["change"],
            "data_points": len(self._returns),
        }

        # Regime
        if latest.regime == "ordered":
            result.regime = MarketRegime.TRENDING_UP  # Direção desconhecida
        elif latest.regime == "chaotic":
            result.regime = MarketRegime.HIGH_VOLATILITY

        # Sinais
        if latest.regime == "ordered":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="entropy_low",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.MODERATE,
                    price=self._prices[-1] if self._prices else 0,
                    confidence=1.0 - latest.normalized_entropy,
                    source="entropy_analyzer",
                    description=(
                        f"LOW entropy ({latest.normalized_entropy:.2f}): "
                        f"market is ordered/predictable. "
                        f"Good conditions for trend-following."
                    ),
                )
            )
        elif latest.regime == "chaotic":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="entropy_high",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.MODERATE,
                    price=self._prices[-1] if self._prices else 0,
                    confidence=latest.normalized_entropy,
                    source="entropy_analyzer",
                    description=(
                        f"HIGH entropy ({latest.normalized_entropy:.2f}): "
                        f"market is chaotic/noisy. "
                        f"Reduce position sizes or stay flat."
                    ),
                )
            )

        # Sinal de mudança de regime de entropia
        if trend["trend"] == "decreasing" and abs(trend["change"]) > 0.15:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="entropy_decreasing",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.MODERATE,
                    price=self._prices[-1] if self._prices else 0,
                    confidence=abs(trend["change"]),
                    source="entropy_analyzer",
                    description=(
                        f"Entropy decreasing rapidly ({trend['change']:.2f}): "
                        f"market transitioning from chaos to order. "
                        f"Trend may be forming."
                    ),
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        self._prices.clear()
        self._returns.clear()
        self._entropy_history.clear()
