# institutional/vwap_twap.py
"""
VWAP (Volume Weighted Average Price) e TWAP (Time Weighted Average Price)

VWAP: Benchmark institucional de execução.
TWAP: Preço médio ao longo do tempo.

Métodos #11 e #12 do Arsenal Institucional.
"""
from __future__ import annotations

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


@dataclass
class VWAPBand:
    """Bandas de desvio padrão ao redor do VWAP."""
    vwap: float
    upper_1sd: float
    lower_1sd: float
    upper_2sd: float
    lower_2sd: float
    upper_3sd: float
    lower_3sd: float


@dataclass
class PriceVolume:
    """Par preço-volume."""
    timestamp: float
    price: float
    volume: float


class VWAPCalculator:
    """
    Calculador de VWAP com bandas de desvio padrão.

    VWAP = Σ(Preço × Volume) / Σ(Volume)

    Institucional usa como:
    - Benchmark de execução (comprou abaixo = boa execução)
    - Suporte/resistência dinâmico
    - Desvios padrão = zonas de sobrecompra/sobrevenda
    """

    def __init__(
        self,
        anchor_period: str = "session",
        max_data_points: int = 5000,
    ):
        self.anchor_period = anchor_period
        self.max_points = max_data_points

        self._data: deque[PriceVolume] = deque(maxlen=max_data_points)
        self._cumulative_pv: float = 0.0  # Σ(price * volume)
        self._cumulative_vol: float = 0.0  # Σ(volume)
        self._cumulative_pv2: float = 0.0  # Σ(price² * volume) para desvio

    @property
    def vwap(self) -> float:
        """VWAP atual."""
        if self._cumulative_vol > 0:
            return self._cumulative_pv / self._cumulative_vol
        return 0.0

    @property
    def data_points(self) -> int:
        return len(self._data)

    def add_candle(
        self,
        timestamp: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> float:
        """
        Adiciona candle e retorna VWAP atualizado.
        Usa preço típico = (High + Low + Close) / 3
        """
        typical_price = (high + low + close) / 3.0
        return self.add_price_volume(timestamp, typical_price, volume)

    def add_price_volume(
        self,
        timestamp: float,
        price: float,
        volume: float,
    ) -> float:
        """Adiciona par preço-volume e retorna VWAP."""
        self._data.append(PriceVolume(timestamp, price, volume))
        self._cumulative_pv += price * volume
        self._cumulative_vol += volume
        self._cumulative_pv2 += (price ** 2) * volume

        return self.vwap

    def get_bands(self) -> VWAPBand:
        """
        Calcula VWAP com bandas de desvio padrão.

        Desvio padrão ponderado pelo volume.
        """
        vwap_val = self.vwap
        if self._cumulative_vol <= 0:
            return VWAPBand(0, 0, 0, 0, 0, 0, 0)

        # Variância ponderada
        variance = (
            self._cumulative_pv2 / self._cumulative_vol
        ) - (vwap_val ** 2)

        # Proteção contra variância negativa (arredondamento)
        variance = max(variance, 0.0)
        std_dev = variance ** 0.5

        return VWAPBand(
            vwap=vwap_val,
            upper_1sd=vwap_val + std_dev,
            lower_1sd=vwap_val - std_dev,
            upper_2sd=vwap_val + 2 * std_dev,
            lower_2sd=vwap_val - 2 * std_dev,
            upper_3sd=vwap_val + 3 * std_dev,
            lower_3sd=vwap_val - 3 * std_dev,
        )

    def get_deviation(self, current_price: float) -> dict:
        """
        Calcula desvio do preço atual em relação ao VWAP.

        Retorna distância em %, desvios padrão e zona.
        """
        bands = self.get_bands()
        vwap_val = bands.vwap

        if vwap_val <= 0:
            return {
                "deviation_pct": 0.0,
                "std_devs": 0.0,
                "zone": "no_data",
            }

        deviation_pct = ((current_price - vwap_val) / vwap_val) * 100
        std_dev = bands.upper_1sd - vwap_val

        if std_dev > 0:
            std_devs = (current_price - vwap_val) / std_dev
        else:
            std_devs = 0.0

        # Zona
        if abs(std_devs) < 1:
            zone = "fair_value"
        elif abs(std_devs) < 2:
            zone = "extended" if std_devs > 0 else "oversold"
        elif abs(std_devs) < 3:
            zone = "overbought" if std_devs > 0 else "deeply_oversold"
        else:
            zone = "extreme_overbought" if std_devs > 0 else "extreme_oversold"

        return {
            "deviation_pct": deviation_pct,
            "std_devs": std_devs,
            "zone": zone,
            "vwap": vwap_val,
            "bands": {
                "upper_1sd": bands.upper_1sd,
                "lower_1sd": bands.lower_1sd,
                "upper_2sd": bands.upper_2sd,
                "lower_2sd": bands.lower_2sd,
            },
        }

    def reset(self) -> None:
        """Reseta para nova sessão."""
        self._data.clear()
        self._cumulative_pv = 0.0
        self._cumulative_vol = 0.0
        self._cumulative_pv2 = 0.0


class TWAPCalculator:
    """
    Calculador de TWAP.

    TWAP = Média simples do preço ao longo do tempo.
    Usado para benchmark de execução e como
    suporte/resistência simples.
    """

    def __init__(self, max_data_points: int = 5000):
        self.max_points = max_data_points
        self._prices: deque[PriceVolume] = deque(maxlen=max_data_points)
        self._cumulative_price: float = 0.0

    @property
    def twap(self) -> float:
        """TWAP atual."""
        if self._prices:
            return self._cumulative_price / len(self._prices)
        return 0.0

    @property
    def data_points(self) -> int:
        return len(self._prices)

    def add_price(self, timestamp: float, price: float) -> float:
        """Adiciona preço e retorna TWAP."""
        self._prices.append(PriceVolume(timestamp, price, 0))
        self._cumulative_price += price
        return self.twap

    def get_deviation(self, current_price: float) -> dict:
        """Calcula desvio do preço em relação ao TWAP."""
        twap_val = self.twap
        if twap_val <= 0:
            return {"deviation_pct": 0.0, "twap": 0.0}

        deviation_pct = ((current_price - twap_val) / twap_val) * 100

        return {
            "deviation_pct": deviation_pct,
            "twap": twap_val,
            "above_twap": current_price > twap_val,
        }

    def reset(self) -> None:
        self._prices.clear()
        self._cumulative_price = 0.0


class VWAPTWAPAnalyzer:
    """
    Analisador combinado VWAP + TWAP.
    Gera sinais quando preço desvia significativamente.
    """

    def __init__(self, max_data_points: int = 5000):
        self.vwap = VWAPCalculator(max_data_points=max_data_points)
        self.twap = TWAPCalculator(max_data_points=max_data_points)

    def add_candle(
        self,
        timestamp: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> dict:
        """Adiciona candle e retorna métricas atuais."""
        vwap_val = self.vwap.add_candle(timestamp, high, low, close, volume)
        twap_val = self.twap.add_price(timestamp, close)

        return {
            "vwap": vwap_val,
            "twap": twap_val,
            "price": close,
            "vwap_deviation": self.vwap.get_deviation(close),
            "twap_deviation": self.twap.get_deviation(close),
        }

    def analyze(self, current_price: float) -> AnalysisResult:
        """Análise completa VWAP + TWAP."""
        result = AnalysisResult(
            source="vwap_twap_analyzer",
            timestamp=time.time(),
        )

        vwap_dev = self.vwap.get_deviation(current_price)
        twap_dev = self.twap.get_deviation(current_price)
        bands = self.vwap.get_bands()

        result.metrics = {
            "vwap": bands.vwap,
            "twap": self.twap.twap,
            "price": current_price,
            "vwap_deviation_pct": vwap_dev["deviation_pct"],
            "vwap_std_devs": vwap_dev["std_devs"],
            "vwap_zone": vwap_dev.get("zone", "unknown"),
            "twap_deviation_pct": twap_dev["deviation_pct"],
        }

        # Sinais baseados em VWAP
        std_devs = vwap_dev.get("std_devs", 0)
        if abs(std_devs) >= 2:
            if std_devs >= 2:
                direction = Side.SELL  # Overbought — probabilidade de reverter
                desc = f"Price {std_devs:.1f} std devs ABOVE VWAP — overbought"
            else:
                direction = Side.BUY  # Oversold
                desc = f"Price {abs(std_devs):.1f} std devs BELOW VWAP — oversold"

            strength = (
                SignalStrength.STRONG if abs(std_devs) >= 3
                else SignalStrength.MODERATE
            )

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="vwap_deviation",
                    direction=direction,
                    strength=strength,
                    price=current_price,
                    confidence=min(abs(std_devs) / 4.0, 1.0),
                    source="vwap_twap_analyzer",
                    description=desc,
                )
            )

        # Sinal de convergência VWAP/TWAP
        if bands.vwap > 0 and self.twap.twap > 0:
            vt_diff = abs(bands.vwap - self.twap.twap) / bands.vwap * 100
            if vt_diff > 0.5:
                # VWAP e TWAP divergem = mercado desequilibrado
                if bands.vwap > self.twap.twap:
                    direction = Side.BUY  # Volume-weighted price is higher
                    desc = "VWAP > TWAP — volume concentrated at higher prices"
                else:
                    direction = Side.SELL
                    desc = "VWAP < TWAP — volume concentrated at lower prices"

                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="vwap_twap_divergence",
                        direction=direction,
                        strength=SignalStrength.WEAK,
                        price=current_price,
                        confidence=min(vt_diff / 2.0, 1.0),
                        source="vwap_twap_analyzer",
                        description=desc,
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        self.vwap.reset()
        self.twap.reset()
