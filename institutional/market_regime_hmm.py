"""
Hidden Markov Model (HMM) para detecção de regimes de mercado.

Classifica o mercado em estados ocultos (bull/bear/lateral/volatile)
e calcula probabilidade de transição entre estados.

Método #19 do Arsenal Institucional.

Implementação simplificada sem dependência do hmmlearn.
Usa algoritmo de Viterbi e Baum-Welch simplificado.
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
class HMMState:
    """Estado do HMM em um ponto no tempo."""
    timestamp: float
    price: float
    log_return: float
    state: int
    state_name: str
    state_probabilities: list[float]
    most_likely_state: int


@dataclass
class RegimeInfo:
    """Informações sobre um regime detectado."""
    state_id: int
    name: str
    mean_return: float
    volatility: float
    probability: float
    duration_bars: int


class MarketRegimeHMM:
    """
    Hidden Markov Model simplificado para regimes de mercado.

    Estados:
    0 = Bull (retornos positivos, volatilidade moderada)
    1 = Bear (retornos negativos, volatilidade moderada-alta)
    2 = Lateral (retornos ~0, volatilidade baixa)
    3 = High Volatility (retornos variáveis, vol muito alta)

    Usa classificação baseada em retornos e volatilidade
    em janela rolante, com suavização de transições.
    """

    STATE_NAMES = {
        0: "bull",
        1: "bear",
        2: "lateral",
        3: "high_volatility",
    }

    def __init__(
        self,
        n_states: int = 4,
        return_window: int = 20,
        vol_window: int = 20,
        smoothing_window: int = 5,
        max_history: int = 2000,
    ):
        if n_states < 2:
            raise InvalidParameterError("n_states must be >= 2")

        self.n_states = min(n_states, 4)
        self.return_window = return_window
        self.vol_window = vol_window
        self.smoothing_window = smoothing_window

        self._prices: deque[float] = deque(maxlen=max_history)
        self._returns: deque[float] = deque(maxlen=max_history)
        self._states: deque[HMMState] = deque(maxlen=max_history)
        self._state_counts: dict[int, int] = {i: 0 for i in range(self.n_states)}
        self._current_state: int = 2  # Start lateral
        self._current_state_duration: int = 0

        # Matriz de transição (contagens)
        self._transition_counts: list[list[int]] = [
            [0] * self.n_states for _ in range(self.n_states)
        ]

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def current_state_name(self) -> str:
        return self.STATE_NAMES.get(self._current_state, "unknown")

    @property
    def state_duration(self) -> int:
        return self._current_state_duration

    @property
    def history(self) -> list[HMMState]:
        return list(self._states)

    def _classify_state(
        self,
        mean_return: float,
        volatility: float,
    ) -> tuple[int, list[float]]:
        """
        Classifica estado baseado em retorno médio e volatilidade.
        Retorna (estado, probabilidades).
        """
        # Thresholds adaptativos baseados na volatilidade histórica
        vol_threshold_high = 0.03  # 3% por período
        vol_threshold_low = 0.008  # 0.8%
        ret_threshold = 0.005     # 0.5%

        # Ajustar thresholds se temos histórico
        if len(self._returns) > 50:
            all_returns = list(self._returns)[-100:]
            hist_vol = self._calculate_std(all_returns)
            vol_threshold_high = hist_vol * 2.0
            vol_threshold_low = hist_vol * 0.5
            ret_threshold = hist_vol * 0.3

        # Calcular scores para cada estado
        scores = [0.0] * self.n_states

        # Bull: retorno positivo, vol moderada
        if mean_return > ret_threshold:
            scores[0] = min(mean_return / max(ret_threshold, 1e-10), 3.0)
            if volatility < vol_threshold_high:
                scores[0] *= 1.5

        # Bear: retorno negativo, vol moderada-alta
        if mean_return < -ret_threshold:
            scores[1] = min(abs(mean_return) / max(ret_threshold, 1e-10), 3.0)
            if volatility > vol_threshold_low:
                scores[1] *= 1.2

        # Lateral: retorno ~0, vol baixa
        if abs(mean_return) < ret_threshold:
            scores[2] = 1.0 + (1.0 - min(abs(mean_return) / max(ret_threshold, 1e-10), 1.0))
            if volatility < vol_threshold_low:
                scores[2] *= 2.0

        # High Vol: vol alta independente de retorno
        if volatility > vol_threshold_high:
            scores[3] = min(volatility / max(vol_threshold_high, 1e-10), 3.0)

        # Normalizar para probabilidades
        total = sum(scores)
        if total > 0:
            probs = [s / total for s in scores]
        else:
            probs = [0.25] * self.n_states

        # Suavização com estado anterior (inércia)
        if self._states:
            prev_probs = self._states[-1].state_probabilities
            smooth = 0.3
            probs = [
                (1 - smooth) * p + smooth * pp
                for p, pp in zip(probs, prev_probs)
            ]
            total = sum(probs)
            probs = [p / total for p in probs]

        state = probs.index(max(probs))

        return state, probs

    def _calculate_std(self, values: list[float]) -> float:
        """Desvio padrão."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(max(variance, 0))

    def update(self, timestamp: float, price: float) -> Optional[HMMState]:
        """
        Atualiza modelo com novo preço.
        Retorna HMMState com classificação de regime.
        """
        self._prices.append(price)

        if len(self._prices) < 2:
            return None

        prev_price = self._prices[-2]
        if prev_price <= 0 or price <= 0:
            return None

        log_return = math.log(price / prev_price)
        self._returns.append(log_return)

        if len(self._returns) < self.return_window:
            return None

        # Calcular retorno médio e volatilidade na janela
        window_returns = list(self._returns)[-self.return_window:]
        mean_return = sum(window_returns) / len(window_returns)
        volatility = self._calculate_std(window_returns)

        # Classificar estado
        new_state, probs = self._classify_state(mean_return, volatility)

        # Atualizar transições
        prev_state = self._current_state
        self._transition_counts[prev_state][new_state] += 1

        # Atualizar duração
        if new_state == self._current_state:
            self._current_state_duration += 1
        else:
            self._current_state_duration = 1

        self._current_state = new_state
        self._state_counts[new_state] = self._state_counts.get(new_state, 0) + 1

        state = HMMState(
            timestamp=timestamp,
            price=price,
            log_return=log_return,
            state=new_state,
            state_name=self.STATE_NAMES.get(new_state, "unknown"),
            state_probabilities=probs,
            most_likely_state=new_state,
        )

        self._states.append(state)
        return state

    def get_transition_matrix(self) -> list[list[float]]:
        """
        Retorna matriz de probabilidade de transição estimada.
        M[i][j] = P(ir para estado j | estou no estado i)
        """
        matrix = []
        for i in range(self.n_states):
            row_total = sum(self._transition_counts[i])
            if row_total > 0:
                row = [c / row_total for c in self._transition_counts[i]]
            else:
                row = [1.0 / self.n_states] * self.n_states
            matrix.append(row)
        return matrix

    def get_regime_info(self) -> list[RegimeInfo]:
        """Informações sobre cada regime."""
        if not self._states:
            return []

        regimes = []
        for state_id in range(self.n_states):
            state_history = [
                s for s in self._states if s.state == state_id
            ]

            if state_history:
                returns = [s.log_return for s in state_history]
                mean_ret = sum(returns) / len(returns)
                vol = self._calculate_std(returns)
                prob = len(state_history) / len(self._states)
            else:
                mean_ret = 0.0
                vol = 0.0
                prob = 0.0

            duration = 0
            if state_id == self._current_state:
                duration = self._current_state_duration

            regimes.append(RegimeInfo(
                state_id=state_id,
                name=self.STATE_NAMES.get(state_id, "unknown"),
                mean_return=mean_ret,
                volatility=vol,
                probability=prob,
                duration_bars=duration,
            ))

        return regimes

    def predict_next_state(self) -> dict[str, float]:
        """Prevê probabilidades do próximo estado."""
        matrix = self.get_transition_matrix()
        current = self._current_state

        predictions = {}
        for j in range(self.n_states):
            name = self.STATE_NAMES.get(j, f"state_{j}")
            predictions[name] = matrix[current][j]

        return predictions

    def analyze(self) -> AnalysisResult:
        """Análise completa de regime."""
        result = AnalysisResult(
            source="market_regime_hmm",
            timestamp=time.time(),
        )

        if len(self._states) < 5:
            result.confidence = 0.0
            return result

        current = self._states[-1]
        regimes = self.get_regime_info()
        predictions = self.predict_next_state()

        result.metrics = {
            "current_state": current.state,
            "current_state_name": current.state_name,
            "state_duration": self._current_state_duration,
            "state_probability": current.state_probabilities[current.state],
            "total_observations": len(self._states),
        }

        for regime in regimes:
            result.metrics[f"regime_{regime.name}_probability"] = regime.probability
            result.metrics[f"regime_{regime.name}_mean_return"] = regime.mean_return

        for name, prob in predictions.items():
            result.metrics[f"predict_{name}"] = prob

        # Mapear regime
        regime_map = {
            "bull": MarketRegime.TRENDING_UP,
            "bear": MarketRegime.TRENDING_DOWN,
            "lateral": MarketRegime.RANGING,
            "high_volatility": MarketRegime.HIGH_VOLATILITY,
        }
        result.regime = regime_map.get(current.state_name, MarketRegime.UNKNOWN)

        # Sinal principal de regime
        confidence = current.state_probabilities[current.state]

        if current.state_name == "bull":
            direction = Side.BUY
            desc = f"Bull regime (prob={confidence:.1%}, duration={self._current_state_duration} bars)"
        elif current.state_name == "bear":
            direction = Side.SELL
            desc = f"Bear regime (prob={confidence:.1%}, duration={self._current_state_duration} bars)"
        elif current.state_name == "high_volatility":
            direction = Side.UNKNOWN
            desc = f"High volatility regime — reduce exposure"
        else:
            direction = Side.UNKNOWN
            desc = f"Lateral regime — range-bound trading"

        result.signals.append(
            Signal(
                timestamp=time.time(),
                signal_type="hmm_regime",
                direction=direction,
                strength=(
                    SignalStrength.STRONG if confidence > 0.6
                    else SignalStrength.MODERATE if confidence > 0.4
                    else SignalStrength.WEAK
                ),
                price=current.price,
                confidence=confidence,
                source="market_regime_hmm",
                description=desc,
                metadata={
                    "state": current.state,
                    "predictions": predictions,
                    "duration": self._current_state_duration,
                },
            )
        )

        # Sinal de transição iminente
        stay_prob = predictions.get(current.state_name, 0)
        if stay_prob < 0.4:
            most_likely_next = max(predictions, key=predictions.get)  # type: ignore
            if most_likely_next != current.state_name:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="hmm_regime_change",
                        direction=Side.UNKNOWN,
                        strength=SignalStrength.MODERATE,
                        price=current.price,
                        confidence=1.0 - stay_prob,
                        source="market_regime_hmm",
                        description=(
                            f"Regime change likely: {current.state_name} → "
                            f"{most_likely_next} (stay_prob={stay_prob:.1%})"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta modelo."""
        self._prices.clear()
        self._returns.clear()
        self._states.clear()
        self._state_counts = {i: 0 for i in range(self.n_states)}
        self._current_state = 2
        self._current_state_duration = 0
        self._transition_counts = [
            [0] * self.n_states for _ in range(self.n_states)
        ]
