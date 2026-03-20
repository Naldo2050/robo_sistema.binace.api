# ml/feature_calculator.py
# -*- coding: utf-8 -*-

"""
LiveFeatureCalculator v2 — com detecção de warmup.

Changelog v2 (2026-03-17):
  - _warmup_ready: False enquanto features são fallback
  - _features_real_count: quantas das 9 features são reais
  - _features_default_list: lista das features em fallback
  - _ml_usable: True quando >=7 de 9 features são reais
  - RSI com avg_loss==0 retorna 95.0 (clamped, não 100.0)
  - _rsi_from_multi_tf retorna Optional[float] (None se não encontrar)
  - Volume ratio usa >= 2 corretamente
"""

import logging
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class LiveFeatureCalculator:
    """Calcula as 9 features do modelo a partir do stream de preços."""

    # Mínimo de preços para cada feature ser "real" (atualizado em __init__)
    _WARMUP_MAP = {
        'price_close': 1,
        'return_1': 2,
        'return_5': 5,
        'return_10': 10,
        'bb_upper': 20,
        'bb_lower': 20,
        'bb_width': 20,
        'rsi': 15,
        'volume_ratio': 2,
    }
    TOTAL_FEATURES = 9
    # 5/9: price_close + RSI + bb_width + volume_ratio + 1 return é suficiente
    # para que ML participe da decisão híbrida (antes: 7, bloqueava por muito tempo)
    MIN_REAL_FEATURES_FOR_ML = 5

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

        self.price_history: deque = deque(maxlen=max(bb_period, rsi_period) + 10)
        self.volume_history: deque = deque(maxlen=20)

        # Atualiza warmup map com parâmetros reais do construtor
        self._WARMUP_MAP = dict(LiveFeatureCalculator._WARMUP_MAP)
        self._WARMUP_MAP['bb_upper'] = bb_period
        self._WARMUP_MAP['bb_lower'] = bb_period
        self._WARMUP_MAP['bb_width'] = bb_period
        self._WARMUP_MAP['rsi'] = rsi_period + 1

        self._min_history_needed = max(self._WARMUP_MAP.values())

    def update(self, price: float, volume: float = 0.0):
        """Adiciona novo preço/volume ao histórico."""
        if price is None or price <= 0:
            return
        self.price_history.append(price)
        if volume > 0:
            self.volume_history.append(volume)

    @property
    def history_count(self) -> int:
        return len(self.price_history)

    @property
    def warmup_ready(self) -> bool:
        return len(self.price_history) >= self._min_history_needed

    @property
    def warmup_progress(self) -> float:
        return min(1.0, len(self.price_history) / self._min_history_needed)

    def _rsi_from_multi_tf(self, multi_tf: dict) -> Optional[float]:
        """Extrai RSI de multi_tf como fallback. Retorna None se não encontrar."""
        for tf in ("15m", "1h", "4h", "1d"):
            tf_data = multi_tf.get(tf, {})
            if not isinstance(tf_data, dict):
                continue
            for rsi_key in ("rsi_short", "rsi", "rsi_14"):
                rsi_val = tf_data.get(rsi_key)
                if rsi_val is not None:
                    try:
                        v = float(rsi_val)
                        if 5.0 < v < 95.0:
                            return v
                    except (ValueError, TypeError):
                        continue
        return None

    def compute(self, multi_tf: dict | None = None) -> dict:
        """
        Calcula as 9 features para o modelo.

        Retorna as 9 features do XGBoost + metadados de warmup:
          _warmup_ready (bool): True se todas as features são reais
          _ml_usable (bool): True se >=7 de 9 são reais
          _features_real_count (int): quantas são baseadas em dados reais
          _features_default_list (list): nomes das features em fallback
          _history_count (int): número de preços no histórico
          _min_history_needed (int): mínimo para warmup completo

        IMPORTANTE: Filtrar chaves com prefixo '_' antes de passar ao XGBoost.
        """
        prices = list(self.price_history)
        n = len(prices)
        current = prices[-1] if n > 0 else 0.0

        real_features: set = set()
        default_features: set = set()

        def _mark(name: str, is_real: bool) -> None:
            if is_real:
                real_features.add(name)
            else:
                default_features.add(name)

        # price_close
        _mark('price_close', n >= 1)

        # Returns
        if n >= 2:
            return_1 = current / prices[-2] - 1
            _mark('return_1', True)
        else:
            return_1 = 0.0
            _mark('return_1', False)

        if n >= 5:
            return_5 = current / prices[-5] - 1
            _mark('return_5', True)
        else:
            return_5 = 0.0
            _mark('return_5', False)

        if n >= 10:
            return_10 = current / prices[-10] - 1
            _mark('return_10', True)
        else:
            return_10 = 0.0
            _mark('return_10', False)

        # Bollinger Bands
        if n >= self.bb_period:
            window = prices[-self.bb_period:]
            sma = np.mean(window)
            std = np.std(window)
            bb_upper = sma + self.bb_std * std
            bb_lower = sma - self.bb_std * std
            bb_width = (bb_upper - bb_lower) / sma if sma > 0 else 0.0
            _mark('bb_upper', True)
            _mark('bb_lower', True)
            _mark('bb_width', True)
        else:
            bb_upper = current * 1.01
            bb_lower = current * 0.99
            bb_width = 0.02
            _mark('bb_upper', False)
            _mark('bb_lower', False)
            _mark('bb_width', False)

        # RSI — SEMPRE preferir multi_tf (klines reais da exchange)
        # Local RSI (de window closes) é fallback apenas quando multi_tf ausente
        rsi_mtf = self._rsi_from_multi_tf(multi_tf) if multi_tf else None
        if rsi_mtf is not None:
            rsi = rsi_mtf
            _mark('rsi', True)
        elif n >= self.rsi_period + 1:
            # Fallback: calcular localmente a partir dos closes de janela
            deltas = np.diff(prices[-(self.rsi_period + 1):])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 80.0 if avg_gain > 1e-10 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            _mark('rsi', True)
        else:
            rsi = 50.0
            _mark('rsi', False)

        # Volume ratio
        if len(self.volume_history) >= 2:
            vol_sma = np.mean(list(self.volume_history))
            volume_ratio = (self.volume_history[-1] / vol_sma) if vol_sma > 0 else 1.0
            _mark('volume_ratio', True)
        else:
            volume_ratio = 1.0
            _mark('volume_ratio', False)

        # Metadados de warmup
        real_count = len(real_features)
        is_ready = real_count >= self.TOTAL_FEATURES
        ml_usable = real_count >= self.MIN_REAL_FEATURES_FOR_ML

        if not is_ready and n > 0:
            logger.info(
                "ML_WARMUP features_real=%d/%d history=%d/%d ready=%s usable=%s defaults=%s",
                real_count, self.TOTAL_FEATURES,
                n, self._min_history_needed,
                is_ready, ml_usable,
                sorted(default_features),
            )

        return {
            # Features do modelo (9) — únicas enviadas ao XGBoost
            'price_close': current,
            'return_1': return_1,
            'return_5': return_5,
            'return_10': return_10,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            # Metadados de warmup (prefixo _ — NÃO enviar ao XGBoost)
            '_warmup_ready': is_ready,
            '_ml_usable': ml_usable,
            '_features_real_count': real_count,
            '_features_default_list': sorted(default_features),
            '_history_count': n,
            '_min_history_needed': self._min_history_needed,
        }
