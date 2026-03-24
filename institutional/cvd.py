# institutional/cvd.py
"""
Cumulative Volume Delta (CVD)

Acumula a diferença entre volume de compra agressiva e venda agressiva.
Divergências entre CVD e preço antecipam reversões.

Método #3 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InsufficientDataError,
    InvalidParameterError,
    MarketRegime,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class CVDBar:
    """Barra de CVD."""
    timestamp: float
    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_delta: float
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    trade_count: int


@dataclass
class CVDDivergence:
    """Divergência detectada entre CVD e preço."""
    divergence_type: str  # "bullish" or "bearish"
    price_direction: str  # "up" or "down"
    cvd_direction: str    # "up" or "down"
    strength: float       # 0.0 a 1.0
    bars_count: int       # quantas barras compõem a divergência
    start_timestamp: float
    end_timestamp: float


class CVDAnalyzer:
    """
    Analisador de Cumulative Volume Delta.

    Calcula delta de volume (compra - venda agressiva),
    acumula ao longo do tempo e detecta divergências com preço.
    """

    def __init__(
        self,
        bar_interval_seconds: float = 60.0,
        max_bars: int = 1000,
        divergence_lookback: int = 10,
        divergence_threshold: float = 0.6,
    ):
        if bar_interval_seconds <= 0:
            raise InvalidParameterError(
                f"bar_interval_seconds must be > 0, got {bar_interval_seconds}"
            )
        if max_bars < 10:
            raise InvalidParameterError(
                f"max_bars must be >= 10, got {max_bars}"
            )

        self.bar_interval_seconds = bar_interval_seconds
        self.max_bars = max_bars
        self.divergence_lookback = divergence_lookback
        self.divergence_threshold = divergence_threshold

        self.bars: deque[CVDBar] = deque(maxlen=max_bars)
        self.cumulative_delta: float = 0.0

        # Buffer para barra atual
        self._current_bar_start: float = 0.0
        self._bar_initialized: bool = False
        self._current_buy_vol: float = 0.0
        self._current_sell_vol: float = 0.0
        self._current_price_open: float = 0.0
        self._current_price_close: float = 0.0
        self._current_price_high: float = 0.0
        self._current_price_low: float = float("inf")
        self._current_trade_count: int = 0
        self._total_trades_processed: int = 0

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades_processed

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def process_trade(self, trade: Trade) -> Optional[CVDBar]:
        """
        Processa um trade e retorna CVDBar se uma barra foi completada.
        """
        self._total_trades_processed += 1

        completed_bar = None

        # Inicializar barra se necessário
        if not self._bar_initialized:
            self._current_bar_start = trade.timestamp
            self._current_price_open = trade.price
            self._current_price_high = trade.price
            self._current_price_low = trade.price
            self._bar_initialized = True
        else:
            # Verificar se barra completou ANTES de adicionar o trade
            elapsed = trade.timestamp - self._current_bar_start
            if elapsed >= self.bar_interval_seconds:
                completed_bar = self._close_bar(trade.timestamp)
                # Iniciar nova barra com este trade
                self._current_price_open = trade.price
                self._current_price_high = trade.price
                self._current_price_low = trade.price
                self._bar_initialized = True

        # Atualizar preço
        self._current_price_close = trade.price
        self._current_price_high = max(self._current_price_high, trade.price)
        self._current_price_low = min(self._current_price_low, trade.price)
        self._current_trade_count += 1

        # Acumular volume por lado
        if trade.side == Side.BUY:
            self._current_buy_vol += trade.quantity
        elif trade.side == Side.SELL:
            self._current_sell_vol += trade.quantity

        return completed_bar

    def process_trades(self, trades: list[Trade]) -> list[CVDBar]:
        """Processa lista de trades, retorna barras completadas."""
        completed: list[CVDBar] = []
        for trade in trades:
            bar = self.process_trade(trade)
            if bar is not None:
                completed.append(bar)
        return completed

    def _close_bar(self, timestamp: float) -> CVDBar:
        """Fecha a barra atual e inicia nova."""
        delta = self._current_buy_vol - self._current_sell_vol
        self.cumulative_delta += delta

        bar = CVDBar(
            timestamp=self._current_bar_start,
            buy_volume=self._current_buy_vol,
            sell_volume=self._current_sell_vol,
            delta=delta,
            cumulative_delta=self.cumulative_delta,
            price_open=self._current_price_open,
            price_close=self._current_price_close,
            price_high=self._current_price_high,
            price_low=self._current_price_low,
            trade_count=self._current_trade_count,
        )

        self.bars.append(bar)
        self._reset_current_bar(timestamp)
        return bar

    def _reset_current_bar(self, timestamp: float) -> None:
        """Reseta buffer da barra atual."""
        self._current_bar_start = timestamp
        self._bar_initialized = False
        self._current_buy_vol = 0.0
        self._current_sell_vol = 0.0
        self._current_price_open = 0.0
        self._current_price_close = 0.0
        self._current_price_high = 0.0
        self._current_price_low = float("inf")
        self._current_trade_count = 0

    def get_current_delta(self) -> float:
        """Delta da barra atual (ainda não fechada)."""
        return self._current_buy_vol - self._current_sell_vol

    def get_current_cvd(self) -> float:
        """CVD total incluindo barra atual não fechada."""
        return self.cumulative_delta + self.get_current_delta()

    def detect_divergences(
        self,
        lookback: Optional[int] = None,
    ) -> list[CVDDivergence]:
        """
        Detecta divergências entre CVD e preço.

        Divergência bullish: preço faz lower low, CVD faz higher low
        Divergência bearish: preço faz higher high, CVD faz lower high
        """
        lookback = lookback or self.divergence_lookback

        if len(self.bars) < lookback + 1:
            return []

        recent_bars = list(self.bars)[-lookback:]
        divergences: list[CVDDivergence] = []

        # Calcular direções
        price_start = recent_bars[0].price_close
        price_end = recent_bars[-1].price_close
        cvd_start = recent_bars[0].cumulative_delta
        cvd_end = recent_bars[-1].cumulative_delta

        price_change = (price_end - price_start) / price_start if price_start else 0
        cvd_change = cvd_end - cvd_start

        price_direction = "up" if price_change > 0 else "down"
        cvd_direction = "up" if cvd_change > 0 else "down"

        # Divergência: preço e CVD vão em direções opostas
        if price_direction != cvd_direction:
            # Calcular força da divergência
            strength = min(abs(price_change) * 100, 1.0)

            if price_direction == "down" and cvd_direction == "up":
                div_type = "bullish"
            else:
                div_type = "bearish"

            divergences.append(
                CVDDivergence(
                    divergence_type=div_type,
                    price_direction=price_direction,
                    cvd_direction=cvd_direction,
                    strength=strength,
                    bars_count=lookback,
                    start_timestamp=recent_bars[0].timestamp,
                    end_timestamp=recent_bars[-1].timestamp,
                )
            )

        return divergences

    def analyze(self) -> AnalysisResult:
        """Análise completa do CVD atual."""
        result = AnalysisResult(
            source="cvd_analyzer",
            timestamp=time.time(),
        )

        if len(self.bars) < 3:
            result.confidence = 0.0
            return result

        recent = list(self.bars)[-5:] if len(self.bars) >= 5 else list(self.bars)

        # Métricas
        current_delta = self.get_current_delta()
        total_buy_vol = sum(b.buy_volume for b in recent)
        total_sell_vol = sum(b.sell_volume for b in recent)
        avg_delta = sum(b.delta for b in recent) / len(recent)

        result.metrics = {
            "cumulative_delta": self.cumulative_delta,
            "current_bar_delta": current_delta,
            "avg_delta_5bars": avg_delta,
            "total_buy_volume": total_buy_vol,
            "total_sell_volume": total_sell_vol,
            "buy_sell_ratio": (
                total_buy_vol / total_sell_vol if total_sell_vol > 0 else 0
            ),
            "bar_count": len(self.bars),
        }

        # Divergências
        divergences = self.detect_divergences()
        for div in divergences:
            signal_dir = Side.BUY if div.divergence_type == "bullish" else Side.SELL

            strength = SignalStrength.STRONG
            if div.strength < 0.3:
                strength = SignalStrength.WEAK
            elif div.strength < 0.6:
                strength = SignalStrength.MODERATE

            result.signals.append(
                Signal(
                    timestamp=div.end_timestamp,
                    signal_type=f"cvd_divergence_{div.divergence_type}",
                    direction=signal_dir,
                    strength=strength,
                    price=recent[-1].price_close,
                    confidence=div.strength,
                    source="cvd_analyzer",
                    description=(
                        f"CVD {div.divergence_type} divergence: "
                        f"price {div.price_direction}, "
                        f"CVD {div.cvd_direction} "
                        f"over {div.bars_count} bars"
                    ),
                )
            )

        # Sinal de pressão dominante
        if total_buy_vol > 0 or total_sell_vol > 0:
            ratio = total_buy_vol / max(total_sell_vol, 1e-10)
            if ratio > 3.0:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="cvd_buy_pressure",
                        direction=Side.BUY,
                        strength=SignalStrength.STRONG,
                        price=recent[-1].price_close,
                        confidence=min(ratio / 5.0, 1.0),
                        source="cvd_analyzer",
                        description=f"Strong buy pressure: ratio {ratio:.1f}x",
                    )
                )
            elif ratio < 0.33:
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="cvd_sell_pressure",
                        direction=Side.SELL,
                        strength=SignalStrength.STRONG,
                        price=recent[-1].price_close,
                        confidence=min(1.0 / max(ratio, 1e-10) / 5.0, 1.0),
                        source="cvd_analyzer",
                        description=(
                            f"Strong sell pressure: ratio {1/max(ratio,1e-10):.1f}x"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta todo o estado."""
        self.bars.clear()
        self.cumulative_delta = 0.0
        self._reset_current_bar(0.0)
        self._total_trades_processed = 0
