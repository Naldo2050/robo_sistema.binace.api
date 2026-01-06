# data_pipeline/validation/adaptive.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Dict, Any, Tuple
from collections import deque
from datetime import datetime
import time

import numpy as np


@dataclass
class AdaptiveThresholds:
    """
    Sistema de thresholds adaptativos baseado em observações históricas.

    Aprende com o padrão real de dados recebidos e ajusta automaticamente
    os thresholds mínimos de trades para processamento.

    Benefícios:
    - Adapta-se a períodos de baixa/alta liquidez
    - Evita rejeição desnecessária de dados
    - Melhora utilização de recursos
    - Previne oscilações com learning rate

    Exemplo de uso:
        adaptive = AdaptiveThresholds(
            initial_min_trades=100,
            absolute_min_trades=10,
            learning_rate=0.2
        )

        # A cada batch de dados
        adaptive.record_observation(len(trades))

        # Periodicamente verificar se deve ajustar
        new_threshold, reason = adaptive.adjust()
        if reason.startswith('adjusted'):
            print(f"Threshold adaptado para {new_threshold}")
    """

    initial_min_trades: int = 10
    absolute_min_trades: int = 3
    max_min_trades: int = 50
    history_size: int = 20
    learning_rate: float = 0.1
    confidence_threshold: float = 0.7

    # Estado interno (não passado no __init__)
    _trade_counts: Deque[int] = field(init=False)
    _adjustment_history: List[Dict[str, Any]] = field(default_factory=list)
    _current_min_trades: int = field(init=False)
    _adjustments_made: int = 0

    def __post_init__(self) -> None:
        """Inicializa estado interno."""
        self._current_min_trades = self.initial_min_trades
        self._trade_counts = deque(maxlen=self.history_size)

    def record_observation(self, trade_count: int) -> None:
        """
        Registra nova observação de quantidade de trades.

        Args:
            trade_count: Quantidade de trades recebidos
        """
        self._trade_counts.append(trade_count)

    def should_adjust(self) -> bool:
        """
        Determina se deve fazer ajuste baseado no histórico.

        Returns:
            True se deve ajustar
        """
        # Precisa ter pelo menos 50% do histórico preenchido
        if len(self._trade_counts) < self.history_size * 0.5:
            return False

        # Calcular quantos batches ficaram abaixo do threshold
        trades_array = np.array(self._trade_counts)
        below_threshold = np.sum(trades_array < self._current_min_trades)
        below_ratio = below_threshold / len(trades_array)

        # Ajustar se >70% dos batches estão abaixo do threshold
        return below_ratio > self.confidence_threshold

    def adjust(self, allow_limited_data: bool = True) -> Tuple[int, str]:
        """
        Ajusta threshold adaptativamente.

        Args:
            allow_limited_data: Se False, não faz ajustes

        Returns:
            Tupla (novo_threshold, motivo)
        """
        if not allow_limited_data:
            return self._current_min_trades, "adjustment_disabled"

        if not self.should_adjust():
            return self._current_min_trades, "no_adjustment_needed"

        trades_array = np.array(self._trade_counts)
        median_trades = int(np.median(trades_array))

        # Novo threshold = 90% da mediana observada
        new_threshold = max(
            self.absolute_min_trades,
            min(int(median_trades * 0.9), self.max_min_trades)
        )

        # Aplicar learning rate para mudanças graduais
        if new_threshold != self._current_min_trades:
            old_threshold = self._current_min_trades
            delta = int((new_threshold - old_threshold) * self.learning_rate)

            # Só ajusta se delta significativo
            if abs(delta) > 0:
                self._current_min_trades = old_threshold + delta
                self._adjustments_made += 1

                # Registrar ajuste
                self._adjustment_history.append({
                    'timestamp': time.time(),
                    'timestamp_iso': datetime.now().isoformat(),
                    'old': old_threshold,
                    'new': self._current_min_trades,
                    'median_observed': median_trades,
                    'reason': f'adaptive_learning_{self._adjustments_made}'
                })

                return self._current_min_trades, f"adjusted_to_{self._current_min_trades}"

        return self._current_min_trades, "no_change"

    def get_current_threshold(self) -> int:
        """Retorna threshold atual."""
        return self._current_min_trades

    def reset(self) -> None:
        """Reseta thresholds para valores iniciais."""
        self._current_min_trades = self.initial_min_trades
        self._trade_counts.clear()
        self._adjustment_history.clear()
        self._adjustments_made = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do sistema adaptativo.

        Returns:
            Dicionário com métricas e histórico
        """
        if not self._trade_counts:
            return {
                'current_threshold': self._current_min_trades,
                'adjustments_made': self._adjustments_made,
                'observations': 0
            }

        trades_array = np.array(self._trade_counts)

        return {
            'current_threshold': self._current_min_trades,
            'initial_threshold': self.initial_min_trades,
            'adjustments_made': self._adjustments_made,
            'observations': len(self._trade_counts),
            'trade_stats': {
                'min': int(trades_array.min()),
                'max': int(trades_array.max()),
                'mean': float(trades_array.mean()),
                'median': float(np.median(trades_array)),
                'std': float(trades_array.std()),
                'p25': float(np.percentile(trades_array, 25)),
                'p75': float(np.percentile(trades_array, 75)),
            },
            'last_adjustment': self._adjustment_history[-1] if self._adjustment_history else None,
            'adjustment_history': self._adjustment_history[-5:]  # Últimos 5
        }