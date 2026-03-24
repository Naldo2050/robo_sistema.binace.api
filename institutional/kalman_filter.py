# institutional/kalman_filter.py
"""
Kalman Filter para tendência de preço.

Filtra ruído e extrai tendência real em tempo real.
Superior a médias móveis: sem atraso fixo, adaptativo,
fornece estimativa de incerteza.

Método #27 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from institutional.base import (
    AnalysisResult,
    MarketRegime,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class KalmanState:
    """Estado do filtro de Kalman."""
    timestamp: float
    price: float
    filtered_price: float
    velocity: float  # Taxa de mudança estimada
    uncertainty: float  # Covariância do erro
    prediction: float  # Previsão para próximo passo
    innovation: float  # Diferença entre previsão e observação


class KalmanTrendFilter:
    """
    Filtro de Kalman para extração de tendência.

    Modelo de estado:
    - x[0] = preço (posição)
    - x[1] = velocidade (taxa de mudança)

    Equações:
    - Predição: x_pred = F @ x + noise
    - Atualização: x_new = x_pred + K @ (z - H @ x_pred)

    Onde K é o ganho de Kalman (adaptativo).
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 5.0,
        initial_uncertainty: float = 1.0,
        max_history: int = 1000,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Estado: [preço, velocidade]
        self._state = [0.0, 0.0]

        # Covariância do erro (2x2)
        self._P = [
            [initial_uncertainty, 0.0],
            [0.0, initial_uncertainty],
        ]

        # Matrizes do modelo
        # F = [[1, dt], [0, 1]] — atualizado dinamicamente
        # H = [[1, 0]] — observamos só o preço
        # Q = process_noise * I — ruído do processo
        # R = measurement_noise — ruído da medição

        self._initialized = False
        self._last_timestamp: float = 0.0
        self._history: deque[KalmanState] = deque(maxlen=max_history)
        self._total_updates: int = 0

    @property
    def filtered_price(self) -> float:
        return self._state[0]

    @property
    def velocity(self) -> float:
        return self._state[1]

    @property
    def uncertainty(self) -> float:
        return self._P[0][0]

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def history(self) -> list[KalmanState]:
        return list(self._history)

    def update(self, timestamp: float, price: float) -> KalmanState:
        """
        Atualiza filtro com nova observação.

        Retorna estado atualizado.
        """
        self._total_updates += 1

        if not self._initialized:
            self._state = [price, 0.0]
            self._last_timestamp = timestamp
            self._initialized = True

            state = KalmanState(
                timestamp=timestamp,
                price=price,
                filtered_price=price,
                velocity=0.0,
                uncertainty=self._P[0][0],
                prediction=price,
                innovation=0.0,
            )
            self._history.append(state)
            return state

        # Calcular dt
        dt = max(timestamp - self._last_timestamp, 0.001)
        self._last_timestamp = timestamp

        # === PREDIÇÃO ===
        # x_pred = F @ x
        x_pred = [
            self._state[0] + self._state[1] * dt,
            self._state[1],
        ]

        # P_pred = F @ P @ F^T + Q
        q = self.process_noise
        P_pred = [
            [
                self._P[0][0] + dt * (self._P[1][0] + self._P[0][1])
                + dt * dt * self._P[1][1] + q,
                self._P[0][1] + dt * self._P[1][1],
            ],
            [
                self._P[1][0] + dt * self._P[1][1],
                self._P[1][1] + q,
            ],
        ]

        # === ATUALIZAÇÃO ===
        # Innovation (residual)
        innovation = price - x_pred[0]

        # S = H @ P_pred @ H^T + R
        S = P_pred[0][0] + self.measurement_noise

        # Ganho de Kalman: K = P_pred @ H^T / S
        K = [P_pred[0][0] / S, P_pred[1][0] / S]

        # Estado atualizado: x = x_pred + K * innovation
        self._state = [
            x_pred[0] + K[0] * innovation,
            x_pred[1] + K[1] * innovation,
        ]

        # Covariância atualizada: P = (I - K @ H) @ P_pred
        self._P = [
            [
                (1 - K[0]) * P_pred[0][0],
                (1 - K[0]) * P_pred[0][1],
            ],
            [
                -K[1] * P_pred[0][0] + P_pred[1][0],
                -K[1] * P_pred[0][1] + P_pred[1][1],
            ],
        ]

        # Previsão para próximo passo
        prediction = self._state[0] + self._state[1] * dt

        state = KalmanState(
            timestamp=timestamp,
            price=price,
            filtered_price=self._state[0],
            velocity=self._state[1],
            uncertainty=self._P[0][0],
            prediction=prediction,
            innovation=innovation,
        )
        self._history.append(state)

        return state

    def get_trend_direction(self) -> str:
        """Direção da tendência baseada na velocidade."""
        if not self._initialized:
            return "unknown"

        vel = self._state[1]
        uncertainty = abs(self._P[1][1]) ** 0.5

        if abs(vel) < uncertainty:
            return "flat"
        elif vel > 0:
            return "up"
        else:
            return "down"

    def get_trend_strength(self) -> float:
        """
        Força da tendência (0.0 a 1.0).

        Baseada na razão velocidade/incerteza (SNR).
        """
        if not self._initialized:
            return 0.0

        vel = abs(self._state[1])
        uncertainty = max(abs(self._P[1][1]) ** 0.5, 1e-10)

        snr = vel / uncertainty
        return min(snr / 5.0, 1.0)  # Normaliza: SNR 5 = força 1.0

    def predict_price(self, steps_ahead: int = 1, dt: float = 1.0) -> list[float]:
        """Prevê preços futuros."""
        if not self._initialized:
            return [0.0] * steps_ahead

        predictions = []
        price = self._state[0]
        vel = self._state[1]

        for i in range(1, steps_ahead + 1):
            price = price + vel * dt
            predictions.append(price)

        return predictions

    def analyze(self, current_price: float) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="kalman_filter",
            timestamp=time.time(),
        )

        if not self._initialized or len(self._history) < 5:
            result.confidence = 0.0
            return result

        direction = self.get_trend_direction()
        strength = self.get_trend_strength()
        predictions = self.predict_price(5)

        filtered = self.filtered_price
        deviation_pct = ((current_price - filtered) / filtered * 100
                         if filtered > 0 else 0)

        result.metrics = {
            "filtered_price": filtered,
            "raw_price": current_price,
            "velocity": self.velocity,
            "uncertainty": self.uncertainty,
            "trend_direction": direction,
            "trend_strength": strength,
            "deviation_pct": deviation_pct,
            "prediction_1step": predictions[0] if predictions else 0,
            "prediction_5step": predictions[-1] if predictions else 0,
        }

        # Regime
        if direction == "up" and strength > 0.5:
            result.regime = MarketRegime.TRENDING_UP
        elif direction == "down" and strength > 0.5:
            result.regime = MarketRegime.TRENDING_DOWN
        elif strength < 0.2:
            result.regime = MarketRegime.RANGING
        else:
            result.regime = MarketRegime.UNKNOWN

        # Sinal se preço desvia muito do filtrado
        if abs(deviation_pct) > 0.3:
            if deviation_pct > 0:
                sig_direction = Side.SELL  # Preço acima do "justo"
                desc = (
                    f"Price {deviation_pct:.2f}% above Kalman filtered — "
                    f"possible overextension"
                )
            else:
                sig_direction = Side.BUY  # Preço abaixo do "justo"
                desc = (
                    f"Price {abs(deviation_pct):.2f}% below Kalman filtered — "
                    f"possible undervaluation"
                )

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="kalman_deviation",
                    direction=sig_direction,
                    strength=(
                        SignalStrength.STRONG if abs(deviation_pct) > 1.0
                        else SignalStrength.MODERATE
                    ),
                    price=current_price,
                    confidence=min(abs(deviation_pct) / 2.0, 1.0),
                    source="kalman_filter",
                    description=desc,
                )
            )

        # Sinal de tendência forte
        if strength > 0.6:
            sig_dir = Side.BUY if direction == "up" else Side.SELL
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="kalman_trend",
                    direction=sig_dir,
                    strength=(
                        SignalStrength.STRONG if strength > 0.8
                        else SignalStrength.MODERATE
                    ),
                    price=current_price,
                    confidence=strength,
                    source="kalman_filter",
                    description=(
                        f"Strong {direction} trend: "
                        f"strength={strength:.2f}, "
                        f"velocity={self.velocity:.4f}"
                    ),
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=strength
        )

        return result

    def reset(self) -> None:
        """Reseta filtro."""
        self._state = [0.0, 0.0]
        self._P = [[1.0, 0.0], [0.0, 1.0]]
        self._initialized = False
        self._last_timestamp = 0.0
        self._history.clear()
        self._total_updates = 0
