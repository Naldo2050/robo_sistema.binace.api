"""
Monte Carlo Simulations.

Simula milhares de cenários probabilísticos futuros para
calcular risco e probabilidade de cada cenário.

Método #18 do Arsenal Institucional.
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
    Side,
    Signal,
    SignalStrength,
)


# LCG simples para não depender de random
class _SimpleRNG:
    """Linear Congruential Generator para independência de imports."""

    def __init__(self, seed: int = 42):
        self._state = seed

    def random(self) -> float:
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._state / 0x7FFFFFFF

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box-Muller transform para normal distribution."""
        u1 = max(self.random(), 1e-10)
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z


@dataclass
class SimulationResult:
    """Resultado de uma simulação Monte Carlo."""
    n_simulations: int
    n_steps: int
    current_price: float

    # Distribuição de preços finais
    mean_final_price: float
    median_final_price: float
    std_final_price: float

    # Percentis
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float

    # Probabilidades
    prob_above_current: float
    prob_gain_5pct: float
    prob_loss_5pct: float
    prob_gain_10pct: float
    prob_loss_10pct: float

    # Drawdown
    avg_max_drawdown: float
    worst_max_drawdown: float

    # Value at Risk
    var_95: float  # 95% VaR (perda máxima com 95% confiança)
    var_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR (Expected Shortfall)


class MonteCarloSimulator:
    """
    Simulador Monte Carlo para previsão de preço e gestão de risco.

    Gera milhares de caminhos aleatórios baseados em:
    1. Retorno médio histórico
    2. Volatilidade histórica
    3. Geometric Brownian Motion (GBM)

    S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        max_price_history: int = 500,
        seed: int = 42,
    ):
        if n_simulations < 100:
            raise InvalidParameterError("n_simulations must be >= 100")

        self.n_simulations = n_simulations
        self._prices: deque[float] = deque(maxlen=max_price_history)
        self._rng = _SimpleRNG(seed)

    @property
    def data_points(self) -> int:
        return len(self._prices)

    def add_price(self, price: float) -> None:
        """Adiciona preço histórico."""
        self._prices.append(price)

    def add_prices(self, prices: list[float]) -> None:
        """Adiciona múltiplos preços."""
        for p in prices:
            self._prices.append(p)

    def _calculate_params(self) -> tuple[float, float]:
        """Calcula retorno médio e volatilidade dos dados."""
        prices = list(self._prices)

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i - 1]))

        if not returns:
            return 0.0, 0.01

        mu = sum(returns) / len(returns)
        variance = sum((r - mu) ** 2 for r in returns) / max(len(returns) - 1, 1)
        sigma = math.sqrt(max(variance, 1e-10))

        return mu, sigma

    def simulate(
        self,
        steps: int = 30,
        custom_mu: Optional[float] = None,
        custom_sigma: Optional[float] = None,
    ) -> Optional[SimulationResult]:
        """
        Executa simulação Monte Carlo.

        Args:
            steps: número de passos à frente
            custom_mu: retorno médio customizado
            custom_sigma: volatilidade customizada
        """
        if len(self._prices) < 10:
            return None

        current_price = self._prices[-1]
        mu, sigma = self._calculate_params()

        if custom_mu is not None:
            mu = custom_mu
        if custom_sigma is not None:
            sigma = custom_sigma

        dt = 1.0  # 1 período
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)

        final_prices = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            price = current_price
            peak = price
            max_dd = 0.0

            for _ in range(steps):
                z = self._rng.gauss()
                price = price * math.exp(drift + diffusion * z)
                price = max(price, current_price * 0.01)

                peak = max(peak, price)
                dd = (peak - price) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_prices.append(price)
            max_drawdowns.append(max_dd)

        # Estatísticas
        final_prices.sort()
        n = len(final_prices)

        mean_price = sum(final_prices) / n
        median_price = final_prices[n // 2]

        variance = sum((p - mean_price) ** 2 for p in final_prices) / n
        std_price = math.sqrt(variance)

        # Percentis
        p5 = final_prices[int(n * 0.05)]
        p25 = final_prices[int(n * 0.25)]
        p75 = final_prices[int(n * 0.75)]
        p95 = final_prices[int(n * 0.95)]

        # Probabilidades
        prob_above = sum(1 for p in final_prices if p > current_price) / n
        prob_gain_5 = sum(1 for p in final_prices if p > current_price * 1.05) / n
        prob_loss_5 = sum(1 for p in final_prices if p < current_price * 0.95) / n
        prob_gain_10 = sum(1 for p in final_prices if p > current_price * 1.10) / n
        prob_loss_10 = sum(1 for p in final_prices if p < current_price * 0.90) / n

        # VaR
        returns_sorted = sorted(
            [(p - current_price) / current_price for p in final_prices]
        )
        var_95 = -returns_sorted[int(n * 0.05)] * 100
        var_99 = -returns_sorted[int(n * 0.01)] * 100

        # CVaR (Expected Shortfall)
        tail_5pct = returns_sorted[:int(n * 0.05)]
        cvar_95 = -sum(tail_5pct) / max(len(tail_5pct), 1) * 100

        # Drawdowns
        avg_dd = sum(max_drawdowns) / n * 100
        worst_dd = max(max_drawdowns) * 100

        return SimulationResult(
            n_simulations=self.n_simulations,
            n_steps=steps,
            current_price=current_price,
            mean_final_price=mean_price,
            median_final_price=median_price,
            std_final_price=std_price,
            percentile_5=p5,
            percentile_25=p25,
            percentile_75=p75,
            percentile_95=p95,
            prob_above_current=prob_above,
            prob_gain_5pct=prob_gain_5,
            prob_loss_5pct=prob_loss_5,
            prob_gain_10pct=prob_gain_10,
            prob_loss_10pct=prob_loss_10,
            avg_max_drawdown=avg_dd,
            worst_max_drawdown=worst_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
        )

    def analyze(self, steps: int = 30) -> AnalysisResult:
        """Análise completa via Monte Carlo."""
        result = AnalysisResult(
            source="monte_carlo",
            timestamp=time.time(),
        )

        sim = self.simulate(steps)
        if sim is None:
            result.confidence = 0.0
            return result

        result.metrics = {
            "current_price": sim.current_price,
            "mean_final": sim.mean_final_price,
            "median_final": sim.median_final_price,
            "std_final": sim.std_final_price,
            "percentile_5": sim.percentile_5,
            "percentile_95": sim.percentile_95,
            "prob_above_current": sim.prob_above_current,
            "prob_gain_5pct": sim.prob_gain_5pct,
            "prob_loss_5pct": sim.prob_loss_5pct,
            "var_95": sim.var_95,
            "var_99": sim.var_99,
            "cvar_95": sim.cvar_95,
            "avg_max_drawdown": sim.avg_max_drawdown,
            "simulations": sim.n_simulations,
            "steps": sim.n_steps,
        }

        # Sinal direcional
        if sim.prob_above_current > 0.65:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="mc_bullish_bias",
                    direction=Side.BUY,
                    strength=(
                        SignalStrength.STRONG if sim.prob_above_current > 0.75
                        else SignalStrength.MODERATE
                    ),
                    price=sim.current_price,
                    confidence=sim.prob_above_current,
                    source="monte_carlo",
                    description=(
                        f"MC simulation: {sim.prob_above_current:.0%} prob of higher price. "
                        f"Expected: ${sim.mean_final_price:,.0f} "
                        f"(range: ${sim.percentile_5:,.0f} - ${sim.percentile_95:,.0f})"
                    ),
                )
            )
        elif sim.prob_above_current < 0.35:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="mc_bearish_bias",
                    direction=Side.SELL,
                    strength=(
                        SignalStrength.STRONG if sim.prob_above_current < 0.25
                        else SignalStrength.MODERATE
                    ),
                    price=sim.current_price,
                    confidence=1.0 - sim.prob_above_current,
                    source="monte_carlo",
                    description=(
                        f"MC simulation: {1-sim.prob_above_current:.0%} prob of lower price. "
                        f"Expected: ${sim.mean_final_price:,.0f}"
                    ),
                )
            )

        # Sinal de risco
        if sim.var_95 > 10:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="mc_high_risk",
                    direction=Side.UNKNOWN,
                    strength=SignalStrength.STRONG,
                    price=sim.current_price,
                    confidence=min(sim.var_95 / 20, 1.0),
                    source="monte_carlo",
                    description=(
                        f"HIGH RISK: 95% VaR={sim.var_95:.1f}%, "
                        f"99% VaR={sim.var_99:.1f}%, "
                        f"Avg MaxDD={sim.avg_max_drawdown:.1f}%"
                    ),
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.3
        )

        return result

    def reset(self) -> None:
        self._prices.clear()
