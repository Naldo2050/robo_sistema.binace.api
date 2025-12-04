# data_pipeline/metrics/processor.py
from __future__ import annotations

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Union

from ..config import PipelineConfig
from ..logging_utils import PipelineLogger
from ..cache.lru_cache import LRUCache


class MetricsProcessor:
    """
    Processador de métricas com arredondamento inteligente e cache.

    Responsável por:
    - Calcular OHLC
    - Calcular volumes
    - Arredondar valores com precisão correta
    - Cachear resultados
    """

    def __init__(
        self,
        config: PipelineConfig,
        symbol: str,
        logger: Optional[PipelineLogger] = None
    ) -> None:
        """
        Inicializa processador.

        Args:
            config: Configurações do pipeline
            symbol: Símbolo do ativo
            logger: Logger especializado (opcional)
        """
        self.config = config
        self.symbol = symbol
        self.logger = logger
        self.precision = config.get_price_precision(symbol)
        self._cache = LRUCache(max_items=100, ttl_seconds=300)

    def round_value(self, value: float, decimals: Optional[int] = None) -> float:
        """
        Arredonda valor com precisão configurada.

        Args:
            value: Valor a arredondar
            decimals: Casas decimais (None = usar padrão do símbolo)

        Returns:
            Valor arredondado
        """
        if value is None or not isinstance(value, (int, float)):
            return 0.0

        if np.isnan(value) or np.isinf(value):
            return 0.0

        decimals = decimals if decimals is not None else self.precision

        if decimals == 0:
            return float(int(round(value)))

        try:
            decimal_value = Decimal(str(value))
            rounded = decimal_value.quantize(
                Decimal(10) ** -decimals,
                rounding=ROUND_HALF_UP
            )
            return float(rounded)
        except:
            return round(value, decimals)

    def calculate_ohlc(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula OHLC do DataFrame com cache.

        Args:
            df: DataFrame com trades

        Returns:
            Dicionário com OHLC
        """
        cache_key = f"ohlc_{len(df)}_{int(df['T'].iloc[-1])}"

        # Verificar cache
        cached = self._cache.get(cache_key, allow_expired=True)
        if cached and not self._cache.is_expired(cache_key):
            if self.logger:
                self.logger.performance_info(
                    "✨ OHLC cache hit",
                    expired=self._cache.is_expired(cache_key)
                )
            return cached

        # Calcular OHLC (vetorizado)
        prices = df["p"].values
        quantities = df["q"].values

        open_price = self.round_value(float(prices[0]))
        close_price = self.round_value(float(prices[-1]))
        high_price = self.round_value(float(prices.max()))
        low_price = self.round_value(float(prices.min()))

        # VWAP
        quote_volume = (prices * quantities).sum()
        base_volume = quantities.sum()
        vwap = self.round_value(
            quote_volume / base_volume if base_volume > 0 else close_price
        )

        result: Dict[str, Any] = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "open_time": int(df["T"].iloc[0]),
            "close_time": int(df["T"].iloc[-1]),
            "vwap": vwap,
        }

        # Armazenar no cache
        self._cache.set(cache_key, result, force_fresh=True)

        return result

    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, Union[float, int]]:
        """
        Calcula métricas de volume.

        Args:
            df: DataFrame com trades

        Returns:
            Dicionário com volumes
        """
        base_volume = self.round_value(float(df["q"].sum()), 2)
        quote_volume = int(round(float((df["p"] * df["q"]).sum())))

        return {
            "volume_total": base_volume,
            "volume_total_usdt": quote_volume,
            "num_trades": len(df),
        }