# institutional/footprint.py
"""
Footprint Charts (Gráficos de Pegada)

Mostra volume negociado em cada nível de preço,
separando compradores de vendedores agressivos.

Método #1 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InsufficientDataError,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class FootprintLevel:
    """Nível individual no footprint."""
    price: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance_ratio(self) -> float:
        """Ratio de desequilíbrio. >1 = compra domina, <1 = venda domina."""
        if self.sell_volume > 0:
            return self.buy_volume / self.sell_volume
        return float("inf") if self.buy_volume > 0 else 1.0


@dataclass
class FootprintBar:
    """Barra completa do footprint."""
    timestamp: float
    duration_seconds: float
    levels: dict[float, FootprintLevel] = field(default_factory=dict)
    price_open: float = 0.0
    price_close: float = 0.0
    price_high: float = 0.0
    price_low: float = float("inf")

    @property
    def total_buy_volume(self) -> float:
        return sum(level.buy_volume for level in self.levels.values())

    @property
    def total_sell_volume(self) -> float:
        return sum(level.sell_volume for level in self.levels.values())

    @property
    def total_delta(self) -> float:
        return self.total_buy_volume - self.total_sell_volume

    @property
    def poc_price(self) -> float:
        """Point of Control - preço com maior volume."""
        if not self.levels:
            return 0.0
        return max(self.levels.values(), key=lambda x: x.total_volume).price

    @property
    def high_volume_nodes(self) -> list[FootprintLevel]:
        """Níveis com volume significativo (top 20%)."""
        if not self.levels:
            return []
        sorted_levels = sorted(
            self.levels.values(),
            key=lambda x: x.total_volume,
            reverse=True,
        )
        cutoff = max(1, len(sorted_levels) // 5)
        return sorted_levels[:cutoff]

    def get_imbalances(self, threshold: float = 3.0) -> list[FootprintLevel]:
        """
        Retorna níveis com desequilíbrio >= threshold.
        Threshold 3.0 = um lado tem 3x mais volume que o outro.
        """
        result = []
        for level in self.levels.values():
            if level.total_volume == 0:
                continue
            ratio = level.imbalance_ratio
            if ratio >= threshold or (ratio > 0 and 1.0 / ratio >= threshold):
                result.append(level)
        return result


class FootprintAnalyzer:
    """
    Analisador de Footprint Charts.

    Agrupa trades por nível de preço em barras temporais,
    separando volume comprador e vendedor.
    """

    def __init__(
        self,
        tick_size: float = 1.0,
        bar_interval_seconds: float = 60.0,
        max_bars: int = 500,
        imbalance_threshold: float = 3.0,
    ):
        if tick_size <= 0:
            raise InvalidParameterError(
                f"tick_size must be > 0, got {tick_size}"
            )
        if bar_interval_seconds <= 0:
            raise InvalidParameterError(
                f"bar_interval_seconds must be > 0, got {bar_interval_seconds}"
            )

        self.tick_size = tick_size
        self.bar_interval_seconds = bar_interval_seconds
        self.max_bars = max_bars
        self.imbalance_threshold = imbalance_threshold

        self.bars: list[FootprintBar] = []
        self._current_bar: Optional[FootprintBar] = None
        self._bar_start_time: float = 0.0
        self._total_trades: int = 0

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def _price_to_level(self, price: float) -> float:
        """Arredonda preço para o tick_size mais próximo."""
        return round(price / self.tick_size) * self.tick_size

    def process_trade(self, trade: Trade) -> Optional[FootprintBar]:
        """
        Processa trade e retorna FootprintBar se barra fechou.
        """
        self._total_trades += 1

        # Criar barra se necessário
        if self._current_bar is None:
            self._current_bar = FootprintBar(
                timestamp=trade.timestamp,
                duration_seconds=self.bar_interval_seconds,
                price_open=trade.price,
                price_high=trade.price,
                price_low=trade.price,
            )
            self._bar_start_time = trade.timestamp

        # Verificar se precisa fechar barra
        elapsed = trade.timestamp - self._bar_start_time
        if elapsed >= self.bar_interval_seconds:
            completed = self._close_bar()
            # Iniciar nova barra com este trade
            self._current_bar = FootprintBar(
                timestamp=trade.timestamp,
                duration_seconds=self.bar_interval_seconds,
                price_open=trade.price,
                price_high=trade.price,
                price_low=trade.price,
            )
            self._bar_start_time = trade.timestamp
            self._add_trade_to_bar(trade)
            return completed

        self._add_trade_to_bar(trade)
        return None

    def _add_trade_to_bar(self, trade: Trade) -> None:
        """Adiciona trade à barra atual."""
        bar = self._current_bar
        if bar is None:
            return

        level_price = self._price_to_level(trade.price)

        if level_price not in bar.levels:
            bar.levels[level_price] = FootprintLevel(price=level_price)

        level = bar.levels[level_price]

        if trade.side == Side.BUY:
            level.buy_volume += trade.quantity
            level.buy_count += 1
        elif trade.side == Side.SELL:
            level.sell_volume += trade.quantity
            level.sell_count += 1

        bar.price_close = trade.price
        bar.price_high = max(bar.price_high, trade.price)
        bar.price_low = min(bar.price_low, trade.price)

    def _close_bar(self) -> FootprintBar:
        """Fecha barra atual e retorna."""
        bar = self._current_bar
        assert bar is not None

        self.bars.append(bar)

        # Manter limite de barras
        if len(self.bars) > self.max_bars:
            self.bars = self.bars[-self.max_bars:]

        return bar

    def process_trades(self, trades: list[Trade]) -> list[FootprintBar]:
        """Processa lista de trades, retorna barras fechadas."""
        completed: list[FootprintBar] = []
        for trade in trades:
            bar = self.process_trade(trade)
            if bar is not None:
                completed.append(bar)
        return completed

    def detect_stacked_imbalances(
        self,
        bar: Optional[FootprintBar] = None,
        min_consecutive: int = 3,
    ) -> list[dict]:
        """
        Detecta imbalances empilhados (3+ níveis consecutivos
        com desequilíbrio no mesmo lado).

        Stacked imbalances = forte presença institucional.
        """
        target_bar = bar or (self.bars[-1] if self.bars else self._current_bar)
        if target_bar is None:
            return []

        imbalances = target_bar.get_imbalances(self.imbalance_threshold)
        if len(imbalances) < min_consecutive:
            return []

        # Ordenar por preço
        sorted_imb = sorted(imbalances, key=lambda x: x.price)

        stacked: list[dict] = []
        current_stack: list[FootprintLevel] = [sorted_imb[0]]
        current_side = "buy" if sorted_imb[0].buy_volume > sorted_imb[0].sell_volume else "sell"

        for i in range(1, len(sorted_imb)):
            level = sorted_imb[i]
            level_side = "buy" if level.buy_volume > level.sell_volume else "sell"
            price_diff = abs(level.price - sorted_imb[i - 1].price)

            # Consecutivo = mesmo lado e diferença de 1 tick
            if level_side == current_side and price_diff <= self.tick_size * 1.5:
                current_stack.append(level)
            else:
                if len(current_stack) >= min_consecutive:
                    stacked.append({
                        "side": current_side,
                        "levels": current_stack.copy(),
                        "price_start": current_stack[0].price,
                        "price_end": current_stack[-1].price,
                        "total_volume": sum(
                            l.total_volume for l in current_stack
                        ),
                        "count": len(current_stack),
                    })
                current_stack = [level]
                current_side = level_side

        # Checar último stack
        if len(current_stack) >= min_consecutive:
            stacked.append({
                "side": current_side,
                "levels": current_stack.copy(),
                "price_start": current_stack[0].price,
                "price_end": current_stack[-1].price,
                "total_volume": sum(l.total_volume for l in current_stack),
                "count": len(current_stack),
            })

        return stacked

    def analyze(self) -> AnalysisResult:
        """Análise completa do footprint atual."""
        result = AnalysisResult(
            source="footprint_analyzer",
            timestamp=time.time(),
        )

        if not self.bars:
            result.confidence = 0.0
            return result

        last_bar = self.bars[-1]

        # Métricas básicas
        result.metrics = {
            "poc_price": last_bar.poc_price,
            "total_delta": last_bar.total_delta,
            "total_buy_volume": last_bar.total_buy_volume,
            "total_sell_volume": last_bar.total_sell_volume,
            "levels_count": len(last_bar.levels),
            "imbalance_count": len(
                last_bar.get_imbalances(self.imbalance_threshold)
            ),
            "bar_count": len(self.bars),
        }

        # Detectar stacked imbalances
        stacked = self.detect_stacked_imbalances(last_bar)
        for stack in stacked:
            direction = Side.BUY if stack["side"] == "buy" else Side.SELL
            strength = SignalStrength.STRONG if stack["count"] >= 5 else SignalStrength.MODERATE

            result.signals.append(
                Signal(
                    timestamp=last_bar.timestamp,
                    signal_type="footprint_stacked_imbalance",
                    direction=direction,
                    strength=strength,
                    price=stack["price_start"],
                    confidence=min(stack["count"] / 7.0, 1.0),
                    source="footprint_analyzer",
                    metadata={"stack": {
                        "side": stack["side"],
                        "count": stack["count"],
                        "price_range": [stack["price_start"], stack["price_end"]],
                    }},
                    description=(
                        f"Stacked {stack['side']} imbalance: "
                        f"{stack['count']} levels "
                        f"from {stack['price_start']} to {stack['price_end']}"
                    ),
                )
            )

        # Sinal de delta extremo
        if last_bar.total_buy_volume + last_bar.total_sell_volume > 0:
            delta_ratio = abs(last_bar.total_delta) / (
                last_bar.total_buy_volume + last_bar.total_sell_volume
            )
            if delta_ratio > 0.4:
                direction = Side.BUY if last_bar.total_delta > 0 else Side.SELL
                result.signals.append(
                    Signal(
                        timestamp=last_bar.timestamp,
                        signal_type="footprint_extreme_delta",
                        direction=direction,
                        strength=SignalStrength.MODERATE,
                        price=last_bar.price_close,
                        confidence=min(delta_ratio, 1.0),
                        source="footprint_analyzer",
                        description=(
                            f"Extreme delta: {last_bar.total_delta:.4f} "
                            f"({delta_ratio:.1%} of total volume)"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self.bars.clear()
        self._current_bar = None
        self._bar_start_time = 0.0
        self._total_trades = 0
