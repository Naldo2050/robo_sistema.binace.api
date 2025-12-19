# constants.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class OrderBookThresholds:
    """
    Conjunto de thresholds e parâmetros "mágicos" do OrderBook.
    Centraliza valores para facilitar tuning e documentação.
    """

    # Validação de ordenação (quantos níveis checar)
    ORDER_VALIDATION_DEPTH: int = 20

    # Spread máximo aceitável em %
    MAX_SPREAD_PERCENT: float = 10.0

    # Liquidez parcial: percentual mínimo do mínimo configurado
    MIN_LIQUIDITY_PARTIAL_RATIO: float = 0.50  # 50%

    # Iceberg: delta mínimo de recarga (em qty) e score mínimo
    ICEBERG_RELOAD_MIN_DELTA: float = 3.0
    ICEBERG_SCORE_THRESHOLD: float = 0.5

    # Weighted imbalance: pesos padrão nos primeiros níveis
    WEIGHTED_IMBALANCE_WEIGHTS: Tuple[float, ...] = (
        1.0,
        0.7,
        0.5,
        0.3,
        0.2,
        0.1,
    )

    # Histórico de snapshots (para iceberg/microestrutura)
    MIN_HISTORY_SIZE: int = 1
    MAX_HISTORY_SIZE: int = 500
    DEFAULT_HISTORY_SIZE: int = 50


THRESHOLDS = OrderBookThresholds()