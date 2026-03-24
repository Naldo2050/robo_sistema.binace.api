"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

Prevê volatilidade futura baseado em volatilidade passada.
Usado para dimensionar risco e detectar regimes de volatilidade.

Método #20 do Arsenal Institucional.

Implementação simplificada sem dependência da biblioteca `arch`.
Usa GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
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
class GARCHState:
    """Estado do modelo GARCH."""
    timestamp: float
    price: float
    log_return: float
    conditional_variance: float
    conditional_volatility: float
    standardized_residual: float


@dataclass
class VolatilityForecast:
    """Previsão de volatilidade."""
    steps_ahead: int
    forecasted_variance: list[float]
    forecasted_volatility: list[float]
    annualized_volatility: list[float]
    current_volatility: float
    long_run_variance: float


class GARCHModel:
    """
    Modelo GARCH(1,1) para previsão de volatilidade.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Onde:
    - ω (omega): constante (peso do longo prazo)
    - α (alpha): peso do choque passado (reação)
    - β (beta): peso da variância passada (persistência)
    - α + β < 1 para estacionariedade

    Parâmetros típicos para crypto:
    - α ≈ 0.05-0.15 (reação a choques)
    - β ≈ 0.80-0.92 (persistência)
    """

    def __init__(
        self,
        omega: float = 0.00001,
        alpha: float = 0.10,
        beta: float = 0.85,
        max_history: int = 2000,
        annualization_factor: float = 365.0,
    ):
        if alpha < 0 or beta < 0:
            raise InvalidParameterError("alpha and beta must be >= 0")
        if alpha + beta >= 1.0:
            raise InvalidParameterError(
                f"alpha + beta must be < 1.0 for stationarity, "
                f"got {alpha + beta}"
            )
        if omega <= 0:
            raise InvalidParameterError("omega must be > 0")

        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.annualization_factor = annualization_factor

        self._history: deque[GARCHState] = deque(maxlen=max_history)
        self._prices: deque[float] = deque(maxlen=max_history)
        self._conditional_variance: float = omega / (1 - alpha - beta)
        self._last_return: float = 0.0
        self._initialized: bool = False

    @property
    def long_run_variance(self) -> float:
        """Variância incondicional de longo prazo."""
        denom = 1.0 - self.alpha - self.beta
        if denom > 0:
            return self.omega / denom
        return self.omega

    @property
    def long_run_volatility(self) -> float:
        """Volatilidade incondicional de longo prazo."""
        return math.sqrt(self.long_run_variance)

    @property
    def current_variance(self) -> float:
        return self._conditional_variance

    @property
    def current_volatility(self) -> float:
        return math.sqrt(max(self._conditional_variance, 0))

    @property
    def persistence(self) -> float:
        """Persistência da volatilidade (α + β)."""
        return self.alpha + self.beta

    @property
    def half_life(self) -> float:
        """Meia-vida do choque de volatilidade (em períodos)."""
        p = self.persistence
        if 0 < p < 1:
            return math.log(0.5) / math.log(p)
        return float("inf")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def update(self, timestamp: float, price: float) -> Optional[GARCHState]:
        """
        Atualiza modelo com novo preço.
        Retorna GARCHState ou None (primeiro preço).
        """
        self._prices.append(price)

        if len(self._prices) < 2:
            return None

        prev_price = self._prices[-2]
        if prev_price <= 0 or price <= 0:
            return None

        log_return = math.log(price / prev_price)

        if not self._initialized:
            self._conditional_variance = log_return ** 2 if log_return != 0 else self.long_run_variance
            self._initialized = True

        # GARCH(1,1) update
        new_variance = (
            self.omega
            + self.alpha * (self._last_return ** 2)
            + self.beta * self._conditional_variance
        )

        new_variance = max(new_variance, 1e-20)

        std_residual = log_return / math.sqrt(new_variance) if new_variance > 0 else 0

        state = GARCHState(
            timestamp=timestamp,
            price=price,
            log_return=log_return,
            conditional_variance=new_variance,
            conditional_volatility=math.sqrt(new_variance),
            standardized_residual=std_residual,
        )

        self._conditional_variance = new_variance
        self._last_return = log_return
        self._history.append(state)

        return state

    def forecast(self, steps: int = 10) -> VolatilityForecast:
        """
        Previsão de volatilidade para N passos à frente.

        σ²_{t+h} = σ̄² + (α+β)^(h-1) · (σ²_t - σ̄²)

        A volatilidade converge para σ̄² (longo prazo).
        """
        lr_var = self.long_run_variance
        current_var = self._conditional_variance
        p = self.persistence

        variances = []
        volatilities = []
        annual_vols = []
        sqrt_annual = math.sqrt(self.annualization_factor)

        for h in range(1, steps + 1):
            fcast_var = lr_var + (p ** h) * (current_var - lr_var)
            fcast_var = max(fcast_var, 1e-20)
            fcast_vol = math.sqrt(fcast_var)

            variances.append(fcast_var)
            volatilities.append(fcast_vol)
            annual_vols.append(fcast_vol * sqrt_annual * 100)

        return VolatilityForecast(
            steps_ahead=steps,
            forecasted_variance=variances,
            forecasted_volatility=volatilities,
            annualized_volatility=annual_vols,
            current_volatility=self.current_volatility,
            long_run_variance=lr_var,
        )

    def get_volatility_regime(self) -> str:
        """
        Classifica regime de volatilidade atual.

        Compara variância atual com variância de longo prazo.
        """
        if not self._initialized:
            return "unknown"

        lr_var = self.long_run_variance
        if lr_var <= 0:
            return "unknown"

        ratio = self._conditional_variance / lr_var

        if ratio > 2.0:
            return "extreme_high"
        elif ratio > 1.5:
            return "high"
        elif ratio > 0.8:
            return "normal"
        elif ratio > 0.5:
            return "low"
        else:
            return "extreme_low"

    def get_volatility_percentile(self) -> float:
        """Percentil da volatilidade atual vs histórico."""
        if len(self._history) < 10:
            return 50.0

        vols = sorted(s.conditional_volatility for s in self._history)
        current = self.current_volatility

        count_below = sum(1 for v in vols if v <= current)
        return (count_below / len(vols)) * 100.0

    def analyze(self) -> AnalysisResult:
        """Análise completa de volatilidade."""
        result = AnalysisResult(
            source="garch_model",
            timestamp=time.time(),
        )

        if not self._initialized or len(self._history) < 5:
            result.confidence = 0.0
            return result

        forecast = self.forecast(5)
        regime = self.get_volatility_regime()
        percentile = self.get_volatility_percentile()

        result.metrics = {
            "current_volatility": self.current_volatility,
            "current_variance": self.current_variance,
            "long_run_volatility": self.long_run_volatility,
            "volatility_regime": regime,
            "volatility_percentile": percentile,
            "persistence": self.persistence,
            "half_life": self.half_life,
            "forecast_1step_vol": forecast.forecasted_volatility[0],
            "forecast_5step_vol": forecast.forecasted_volatility[-1],
            "forecast_1step_annual": forecast.annualized_volatility[0],
        }

        # Regime de mercado
        if regime in ("extreme_high", "high"):
            result.regime = MarketRegime.HIGH_VOLATILITY
        elif regime in ("extreme_low", "low"):
            result.regime = MarketRegime.LOW_VOLATILITY
        else:
            result.regime = MarketRegime.UNKNOWN

        # Sinais
        if regime == "extreme_high":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="garch_extreme_volatility",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.STRONG,
                    price=self._prices[-1] if self._prices else 0,
                    confidence=0.8,
                    source="garch_model",
                    description=(
                        f"EXTREME volatility: {self.current_volatility:.6f} "
                        f"({percentile:.0f}th percentile). "
                        f"Reduce position sizes."
                    ),
                    metadata={
                        "regime": regime,
                        "percentile": percentile,
                        "forecast_annualized": forecast.annualized_volatility[0],
                    },
                )
            )
        elif regime == "extreme_low":
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="garch_low_volatility",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.MODERATE,
                    price=self._prices[-1] if self._prices else 0,
                    confidence=0.6,
                    source="garch_model",
                    description=(
                        f"LOW volatility: {self.current_volatility:.6f} "
                        f"({percentile:.0f}th percentile). "
                        f"Breakout may be imminent."
                    ),
                    metadata={
                        "regime": regime,
                        "percentile": percentile,
                    },
                )
            )

        # Sinal de volatilidade crescente
        if len(self._history) >= 10:
            recent_vols = [s.conditional_volatility for s in list(self._history)[-10:]]
            vol_trend = recent_vols[-1] / max(recent_vols[0], 1e-20)
            if vol_trend > 2.0:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="garch_vol_expanding",
                        direction=Side.UNKNOWN,
                        strength=SignalStrength.MODERATE,
                        price=self._prices[-1] if self._prices else 0,
                        confidence=min(vol_trend / 4.0, 1.0),
                        source="garch_model",
                        description=(
                            f"Volatility expanding {vol_trend:.1f}x "
                            f"over last 10 periods"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.3
        )

        return result

    def reset(self) -> None:
        """Reseta modelo."""
        self._history.clear()
        self._prices.clear()
        self._conditional_variance = self.long_run_variance
        self._last_return = 0.0
        self._initialized = False
