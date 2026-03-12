# ml/feature_calculator.py

import numpy as np
from collections import deque


class LiveFeatureCalculator:
    """Calcula as 9 features do modelo a partir do stream de preços."""

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

        self.price_history: deque = deque(maxlen=max(bb_period, rsi_period) + 10)
        self.volume_history: deque = deque(maxlen=20)

    def update(self, price: float, volume: float = 0.0):
        """Adiciona novo preço/volume ao histórico."""
        if price is None or price <= 0:
            return
            
        self.price_history.append(price)
        if volume > 0:
            self.volume_history.append(volume)

    def compute(self) -> dict:
        """Calcula as 9 features para o modelo, garantindo que as chaves existam."""
        prices = list(self.price_history)
        n = len(prices)

        current = prices[-1] if n > 0 else 0.0

        # Returns (Safe handling)
        return_1 = (current / prices[-2] - 1) if n >= 2 else 0.0
        return_5 = (current / prices[-5] - 1) if n >= 5 else 0.0
        return_10 = (current / prices[-10] - 1) if n >= 10 else 0.0

        # Bollinger Bands
        if n >= self.bb_period:
            window = prices[-self.bb_period:]
            sma = np.mean(window)
            std = np.std(window)
            bb_upper = sma + self.bb_std * std
            bb_lower = sma - self.bb_std * std
            bb_width = (bb_upper - bb_lower) / sma if sma > 0 else 0.0
        else:
            # Fallback dinâmico baseado no preço atual
            bb_upper = current * 1.01
            bb_lower = current * 0.99
            bb_width = 0.02

        # RSI
        if n >= self.rsi_period + 1:
            deltas = np.diff(prices[-(self.rsi_period + 1):])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0

        # Volume ratio
        if len(self.volume_history) >= 2:
            vol_sma = np.mean(list(self.volume_history))
            volume_ratio = (self.volume_history[-1] / vol_sma) if vol_sma > 0 else 1.0
        else:
            volume_ratio = 1.0

        return {
            'price_close': current,
            'return_1': return_1,
            'return_5': return_5,
            'return_10': return_10,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
        }
