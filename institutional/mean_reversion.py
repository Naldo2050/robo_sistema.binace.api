"""
Mean Reversion Models.

Baseado no princípio de que preços extremos tendem a voltar à média.
Usa Z-Score, Bandas de Bollinger avançadas e distância da média.

Método #15 do Arsenal Institucional.
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
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class MeanReversionState:
    """Estado da análise de reversão à média."""
    timestamp: float
    price: float
    mean: float
    std_dev: float
    z_score: float
    bollinger_upper: float
    bollinger_lower: float
    percent_b: float  # Posição dentro das bandas (0-1)
    distance_from_mean_pct: float
    regime: str  # "overbought", "oversold", "fair_value"


class MeanReversionAnalyzer:
    """
    Analisador de reversão à média.

    Combina:
    1. Z-Score: (preço - média) / desvio padrão
    2. Bollinger Bands: média ± K * desvio
    3. %B: posição relativa dentro das bandas
    4. Distância percentual da média

    Sinais:
    - Z > 2.0: sobrecomprado (possível venda)
    - Z < -2.0: sobrevendido (possível compra)
    - %B > 1.0: acima da banda superior
    - %B < 0.0: abaixo da banda inferior
    """

    def __init__(
        self,
        lookback: int = 20,
        bb_multiplier: float = 2.0,
        z_threshold: float = 2.0,
        max_history: int = 2000,
    ):
        if lookback < 5:
            raise InvalidParameterError("lookback must be >= 5")

        self.lookback = lookback
        self.bb_multiplier = bb_multiplier
        self.z_threshold = z_threshold

        self._prices: deque[float] = deque(maxlen=max_history)
        self._states: deque[MeanReversionState] = deque(maxlen=max_history)

    @property
    def data_points(self) -> int:
        return len(self._prices)

    @property
    def last_state(self) -> Optional[MeanReversionState]:
        return self._states[-1] if self._states else None

    def add_price(self, timestamp: float, price: float) -> Optional[MeanReversionState]:
        """Adiciona preço e calcula métricas de reversão."""
        self._prices.append(price)

        if len(self._prices) < self.lookback:
            return None

        window = list(self._prices)[-self.lookback:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std_dev = math.sqrt(max(variance, 0))

        # Z-Score
        z_score = (price - mean) / std_dev if std_dev > 0 else 0.0

        # Bollinger Bands
        bb_upper = mean + self.bb_multiplier * std_dev
        bb_lower = mean - self.bb_multiplier * std_dev

        # %B
        bb_range = bb_upper - bb_lower
        percent_b = (price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # Distância da média
        dist_pct = ((price - mean) / mean * 100) if mean > 0 else 0

        # Classificar regime
        if z_score > self.z_threshold:
            regime = "overbought"
        elif z_score < -self.z_threshold:
            regime = "oversold"
        else:
            regime = "fair_value"

        state = MeanReversionState(
            timestamp=timestamp,
            price=price,
            mean=mean,
            std_dev=std_dev,
            z_score=z_score,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            percent_b=percent_b,
            distance_from_mean_pct=dist_pct,
            regime=regime,
        )

        self._states.append(state)
        return state

    def get_reversion_probability(self) -> float:
        """
        Estima probabilidade de reversão à média.

        Baseado na distância atual e no Hurst do Z-Score.
        """
        if not self._states:
            return 0.0

        z = abs(self._states[-1].z_score)

        # Probabilidade baseada em distribuição normal
        # P(|Z| > z) usando aproximação
        if z <= 0:
            return 0.0

        # Aproximação da CDF normal
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804 * math.exp(-z * z / 2.0)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))

        # P(reverter) = 1 - P(continuar) = 1 - 2*p (two-tailed)
        return min(1.0 - 2 * p, 1.0)

    def analyze(self) -> AnalysisResult:
        """Análise completa de reversão à média."""
        result = AnalysisResult(
            source="mean_reversion",
            timestamp=time.time(),
        )

        if not self._states:
            result.confidence = 0.0
            return result

        state = self._states[-1]
        rev_prob = self.get_reversion_probability()

        result.metrics = {
            "price": state.price,
            "mean": state.mean,
            "std_dev": state.std_dev,
            "z_score": state.z_score,
            "bollinger_upper": state.bollinger_upper,
            "bollinger_lower": state.bollinger_lower,
            "percent_b": state.percent_b,
            "distance_from_mean_pct": state.distance_from_mean_pct,
            "regime": state.regime,
            "reversion_probability": rev_prob,
        }

        # Sinais de sobrecompra/sobrevenda
        if state.regime == "overbought":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="mean_rev_overbought",
                    direction=Side.SELL,
                    strength=(
                        SignalStrength.STRONG if abs(state.z_score) > 3
                        else SignalStrength.MODERATE
                    ),
                    price=state.price,
                    confidence=rev_prob,
                    source="mean_reversion",
                    description=(
                        f"OVERBOUGHT: Z={state.z_score:.2f}, "
                        f"{state.distance_from_mean_pct:.1f}% above mean. "
                        f"Reversion probability: {rev_prob:.1%}"
                    ),
                )
            )
        elif state.regime == "oversold":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="mean_rev_oversold",
                    direction=Side.BUY,
                    strength=(
                        SignalStrength.STRONG if abs(state.z_score) > 3
                        else SignalStrength.MODERATE
                    ),
                    price=state.price,
                    confidence=rev_prob,
                    source="mean_reversion",
                    description=(
                        f"OVERSOLD: Z={state.z_score:.2f}, "
                        f"{abs(state.distance_from_mean_pct):.1f}% below mean. "
                        f"Reversion probability: {rev_prob:.1%}"
                    ),
                )
            )

        # Sinal de Bollinger squeeze (bandas estreitas)
        if len(self._states) >= 20:
            recent_bw = [
                (s.bollinger_upper - s.bollinger_lower) / s.mean * 100
                for s in list(self._states)[-20:]
                if s.mean > 0
            ]
            if recent_bw:
                avg_bw = sum(recent_bw) / len(recent_bw)
                current_bw = recent_bw[-1]

                if current_bw < avg_bw * 0.5:
                    result.signals.append(
                        Signal(
                            timestamp=time.time(),
                            signal_type="bollinger_squeeze",
                            direction=Side.UNKNOWN,
                            strength=SignalStrength.MODERATE,
                            price=state.price,
                            confidence=0.6,
                            source="mean_reversion",
                            description=(
                                f"Bollinger Squeeze: bandwidth={current_bw:.2f}% "
                                f"({current_bw/avg_bw:.0%} of average). "
                                f"Breakout may be imminent."
                            ),
                        )
                    )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        self._prices.clear()
        self._states.clear()
